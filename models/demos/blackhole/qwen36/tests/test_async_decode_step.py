# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Correctness gate for QWEN36_ASYNC_DECODE_STEP (tt/async_decode.py).

Runs the SAME post-prefill decode twice from an identical GDN/KV starting
state — once through the stock traced loop (host input staging + host
winner-pick) and once through the device-resident loop (in-trace winner-pick,
token feedback, on-device pos/rope advance) — and asserts the greedy token
sequences are exactly equal for every user. The two paths share the per-shard
ttnn.argmax and an identical tie-break rule (lowest shard == lowest global
index), and the async rope values are gathered from a table built with the
same host math, so any mismatch is a real bug, not numerics-by-construction.

Run (P150x8):
    MESH_DEVICE=P150x8 HF_MODEL=/path/to/qwen38-27b-weights \\
        pytest models/demos/blackhole/qwen36/tests/test_async_decode_step.py -v -s

Knobs: QWEN36_ASYNC_GATE_ISL (default 128), QWEN36_ASYNC_GATE_STEPS (default 64).
"""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen36.campaign.bench_common import (
    bench_prompt,
    restore_gdn_tp,
    restore_gdn_tp_staged,
    snapshot_gdn_tp,
    stage_gdn_tp,
)
from models.demos.blackhole.qwen36.demo.text_demo import BLOCK_SIZE, DEVICE_PARAMS, _MESH_SHAPE, _MULTI
from models.demos.blackhole.qwen36.tt.async_decode import AsyncDecodeStep
from models.demos.blackhole.qwen36.tt.model import Qwen36Model
from models.tt_transformers.tt.common import copy_host_to_device

_ISL = int(os.environ.get("QWEN36_ASYNC_GATE_ISL", "128"))
_STEPS = int(os.environ.get("QWEN36_ASYNC_GATE_STEPS", "64"))
_B = 8


@run_for_blackhole()
@pytest.mark.timeout(3600)
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_async_decode_step_matches_stock(mesh_device):
    if not _MULTI:
        pytest.skip("async decode step is the TP serving path; set MESH_DEVICE=P150x4 or P150x8")
    device = mesh_device
    device.enable_program_cache()
    B, ISL = _B, _ISL

    bpu = max(8, -(-(ISL + 2 * _STEPS + 16) // BLOCK_SIZE))
    bpu = ((bpu + 7) // 8) * 8
    max_seq_len = bpu * BLOCK_SIZE
    model = Qwen36Model.from_pretrained(device, max_batch_size=B, max_seq_len=max_seq_len)
    assert model.sampling is not None, "gate needs the on-device sampler topology (1x4/1x8)"
    mesh = model.mesh_device
    vocab = model.args.vocab_size
    nd = model.num_devices
    per_shard = vocab // nd

    kv_shape = [B * bpu, model.args.n_local_kv_heads, BLOCK_SIZE, model.args.head_dim]
    model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=B)
    page_table = torch.stack([torch.arange(u * bpu, (u + 1) * bpu, dtype=torch.int32) for u in range(B)])

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model.args.CKPT_DIR, trust_remote_code=True)
    token_ids = bench_prompt(ISL, tokenizer)
    pf_logits = model.prefill_chunked_peruser([token_ids for _ in range(B)], page_table, valid_lens=[ISL] * B)
    ttnn.synchronize_device(mesh)
    comp0 = ttnn.ConcatMeshToTensor(mesh, dim=0)
    nxt = [
        int(ttnn.to_torch(pf_logits[u], mesh_composer=comp0).reshape(-1, vocab)[0].float().argmax()) for u in range(B)
    ]
    snap = snapshot_gdn_tp(model)

    dev = model.prepare_inputs_decode(
        torch.tensor(nxt, dtype=torch.int32).reshape(B, 1),
        torch.tensor([ISL] * B, dtype=torch.int32),
        page_table=page_table,
    )

    read_comp = ttnn.ConcatMeshToTensor(mesh, dim=0)
    maxval_c = 32
    maxval_r = (((per_shard + maxval_c - 1) // maxval_c) + 31) // 32 * 32

    def _argmax_dev(sharded_logits):
        # Same per-shard argmax + max the bench readback uses.
        Bn = sharded_logits.shape[2]
        logits_rm = ttnn.to_layout(sharded_logits, ttnn.ROW_MAJOR_LAYOUT)
        idx = ttnn.argmax(logits_rm, dim=-1, keepdim=False)
        ttnn.deallocate(logits_rm)
        padded = ttnn.pad(sharded_logits, [(0, 0), (0, 0), (0, 0), (0, maxval_r * maxval_c - per_shard)], value=-1e30)
        grid = ttnn.reshape(padded, (1, Bn, maxval_r, maxval_c))
        part = ttnn.max(grid, dim=-1)
        part_row = ttnn.reshape(part, (1, 1, Bn, maxval_r))
        val = ttnn.max(part_row, dim=-1)
        for t in (padded, grid, part, part_row):
            ttnn.deallocate(t)
        return idx, val

    def _host_pick(idx_t, val_t):
        Bn = idx_t.shape[-1]
        idxs = ttnn.to_torch(idx_t, mesh_composer=read_comp).reshape(nd, Bn)[:, :B].to(torch.int64)
        vals = ttnn.to_torch(val_t, mesh_composer=read_comp).reshape(nd, Bn)[:, :B]
        d = torch.argmax(vals, dim=0)
        return (d * per_shard + idxs[d, torch.arange(B)]).tolist()

    def _fwd():
        return model.ttnn_decode_forward(dev[0], dev[1], rot_mat_idxs=dev[2], page_table=dev[3], on_device_logits=True)

    def _update(tokens_row, positions):
        host = model.prepare_decode_inputs_host(
            torch.tensor(tokens_row, dtype=torch.int32).reshape(B, 1),
            torch.tensor(positions, dtype=torch.int32),
            page_table=None,
        )
        copy_host_to_device(host[:3], device_tensors=dev[:3])

    def _capture(stepper):
        gdn_staging = stage_gdn_tp(model, snap)
        warm = _fwd()
        wi, wv = _argmax_dev(warm)
        if stepper is not None:
            stepper.emit_step_tail(wi, wv)
            ttnn.synchronize_device(mesh)
        ttnn.deallocate(wi)
        ttnn.deallocate(wv)
        trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
        logits = _fwd()
        tt_idx, tt_val = _argmax_dev(logits)
        if stepper is not None:
            stepper.emit_step_tail(tt_idx, tt_val)
        ttnn.end_trace_capture(mesh, trace_id, cq_id=0)
        restore_gdn_tp_staged(model, gdn_staging)
        return trace_id, tt_idx, tt_val, gdn_staging

    # Pass 1: stock traced loop (host staging + host winner-pick).
    _update(nxt, [ISL] * B)
    trace_id, tt_idx, tt_val, staging = _capture(stepper=None)
    seq_stock = [[t] for t in nxt]
    pos = [ISL] * B
    for _ in range(_STEPS):
        _update([s[-1] for s in seq_stock], pos)
        ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        toks = _host_pick(tt_idx, tt_val)
        for u in range(B):
            seq_stock[u].append(toks[u])
        pos = [p + 1 for p in pos]
    ttnn.release_trace(mesh, trace_id)
    del staging

    # Reset to the identical post-prefill state (KV slots get rewritten in stride).
    restore_gdn_tp(model, snap)

    # Pass 2: device-resident loop.
    stepper = AsyncDecodeStep(model, dev[0], dev[1], dev[2], batch=B, table_len=max_seq_len)
    trace_id, tt_idx, tt_val, staging = _capture(stepper)
    stepper.resync(nxt, [ISL] * B)
    seq_async = [[t] for t in nxt]
    for _ in range(_STEPS):
        ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        toks = stepper.read_tokens()
        for u in range(B):
            seq_async[u].append(toks[u])
    ttnn.release_trace(mesh, trace_id)
    del staging
    stepper.release()

    for u in range(B):
        if seq_async[u] != seq_stock[u]:
            first = next(i for i, (a, b) in enumerate(zip(seq_async[u], seq_stock[u])) if a != b)
            logger.error(
                f"user {u} diverges at step {first}: stock={seq_stock[u][first - 2 : first + 3]} "
                f"async={seq_async[u][first - 2 : first + 3]}"
            )
        assert seq_async[u] == seq_stock[u], f"user {u}: async decode step diverged from stock greedy sequence"
    sample = tokenizer.decode(seq_async[0][:32])
    logger.info(f"PASSED: {_STEPS}-step token equality across {B} users; user0 text: {sample!r}")
