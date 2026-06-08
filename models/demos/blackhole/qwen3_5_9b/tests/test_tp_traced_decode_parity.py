# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""TEMP validation: the TP traced decode loop (begin/end_trace_capture + execute_trace, as wired
into demo/text_demo._run_tp_generation) must produce the SAME per-step logits as the eager
contract decode (prepare_inputs_decode -> ttnn_decode_forward), teacher-forced with identical
tokens. Trace replay should be numerically faithful to eager re-issue; a high per-step PCC proves
Tier-1 tracing is correct and that any greedy-argmax divergence is benign thinking-mode chaos.

Run:
  source python_env/bin/activate
  MESH_DEVICE=P150x4 HF_MODEL=/home/ttuser/models/Qwen3.5-27B-FP8 \
    pytest -svq models/demos/blackhole/qwen3_5_9b/tests/test_tp_traced_decode_parity.py
"""
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.blackhole.qwen3_5_9b.tt.model import Qwen35Model
from models.tt_transformers.tt.common import copy_host_to_device


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [{"N150": (1, 1), "P150x4": (1, 4)}.get(os.environ.get("MESH_DEVICE"), (1, min(len(ttnn.get_device_ids()), 4)))],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 256 * 1024 * 1024}],
    indirect=True,
)
def test_tp_traced_decode_parity(mesh_device, reset_seeds, ensure_gc):
    nd = mesh_device.get_num_devices()
    assert nd > 1, "this test exercises the TP (num_devices>1) traced decode path"
    mesh_device.enable_program_cache()
    model = Qwen35Model.from_pretrained(mesh_device, max_batch_size=1, max_seq_len=512, n_layers=8)
    args = model.args
    vocab = args.vocab_size
    T, N_DEC = 64, 6
    torch.manual_seed(0)
    prompt = torch.randint(0, vocab, (T,)).tolist()
    teacher = torch.randint(0, vocab, (N_DEC,)).tolist()  # fixed tokens fed to BOTH passes
    comp0 = ttnn.ConcatMeshToTensor(mesh_device, dim=0)

    block_size = 64
    num_blocks = T // block_size + N_DEC + 8
    page_table = torch.arange(num_blocks, dtype=torch.int32).reshape(1, num_blocks)
    kv_shape = (num_blocks, args.n_local_kv_heads, block_size, args.head_dim)

    def _read(out):
        return model.process_output_decode(out, 1).reshape(-1)[:vocab].float()

    def _dev_inputs(token, pos):
        return model.prepare_inputs_decode(
            torch.tensor([[token]], dtype=torch.int32), torch.tensor([pos], dtype=torch.int32), page_table
        )

    # Per-device GDN state snapshot/restore (the exact mechanism demo/text_demo uses to undo the
    # throwaway capture's state advance). Snapshot ALL ranks, restore by sharding back in place.
    gdn = [layer.attention for layer in model.layers if not layer.is_full_attention]

    def _snapshot_gdn():
        comp = ttnn.ConcatMeshToTensor(mesh_device, dim=0)
        return [
            (
                ttnn.to_torch(dn.rec_state, mesh_composer=comp),
                [ttnn.to_torch(c, mesh_composer=comp) for c in dn.conv_states],
            )
            for dn in gdn
        ]

    def _restore_gdn(snap):
        mapper = ttnn.ShardTensorToMesh(mesh_device, dim=0)
        back = lambda t: ttnn.from_torch(
            t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=mapper
        )
        for dn, (rec, convs) in zip(gdn, snap):
            r = back(rec)
            ttnn.copy(r, dn.rec_state)
            ttnn.deallocate(r)
            for j, c in enumerate(convs):
                cc = back(c)
                ttnn.copy(cc, dn.conv_states[j])
                ttnn.deallocate(cc)

    # ---- prefill once; snapshot the exact post-prefill GDN state ----
    model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=1)
    model.prefill_paged(torch.tensor([prompt], dtype=torch.long), page_table, valid_len=T)
    post_prefill = _snapshot_gdn()

    # ---- EAGER contract decode (oracle) ----
    eager_logits = []
    pos = T
    for i in range(N_DEC):
        dev = _dev_inputs(teacher[i], pos)
        out, _ = model.ttnn_decode_forward(dev[0], dev[1], rot_mat_idxs=dev[2], page_table=dev[3])
        eager_logits.append(_read(out))
        pos += 1
    logger.info(f"eager argmax:  {[int(l.argmax()) for l in eager_logits]}")

    # ---- TRACED contract decode ----
    _restore_gdn(post_prefill)  # undo the eager-pass decode (back to post-prefill)
    dev = _dev_inputs(teacher[0], T)
    # Compile the decode programs (eager) then capture a THROWAWAY trace; both advance GDN state,
    # so restore the post-prefill snapshot afterward. Every REAL decode step is a pure execute_trace
    # replay; the capture run's own output is not used (unreliable in this tt-metal version).
    model.ttnn_decode_forward(dev[0], dev[1], rot_mat_idxs=dev[2], page_table=dev[3])  # compile
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    out, _ = model.ttnn_decode_forward(dev[0], dev[1], rot_mat_idxs=dev[2], page_table=dev[3])
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    _restore_gdn(post_prefill)

    traced_logits = []
    pos = T
    for i in range(N_DEC):
        host = model.prepare_decode_inputs_host(
            torch.tensor([[teacher[i]]], dtype=torch.int32), torch.tensor([pos], dtype=torch.int32), page_table
        )
        copy_host_to_device(host, device_tensors=dev)
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        traced_logits.append(_read(out))
        pos += 1
    ttnn.release_trace(mesh_device, trace_id)
    logger.info(f"traced argmax: {[int(l.argmax()) for l in traced_logits]}")

    pccs = []
    for i, (e, t) in enumerate(zip(eager_logits, traced_logits)):
        _, pcc = comp_pcc(e, t, 0.99)
        pccs.append(float(pcc))
        logger.info(f"step {i} (replay) eager-vs-traced logits PCC = {pcc}")
    # Every step is a faithful trace replay from clean state, so all should match eager >= 0.99.
    assert all(p >= 0.99 for p in pccs), f"traced decode diverges from eager (per-step PCC): {pccs}"
    logger.info("PASSED: TP traced decode is numerically faithful to eager contract decode")
