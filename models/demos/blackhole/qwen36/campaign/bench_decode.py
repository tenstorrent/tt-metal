# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Campaign goal metric: decode t/s/u at concurrency 8 with ISL 10,240 (target 100 t/s/u).

Wraps the qwen36 TP serving path (Qwen3.8-27B selected via HF_MODEL): each user
is prefilled to ISL depth via per-user chunked prefill (KV + GDN state at real
depth), then one B-wide traced decode step is timed end-to-end (input update +
trace replay + token readback) for QWEN38_BENCH_STEPS steps. Emits one
BENCH_JSON line with per-step min/median/p90, t/s/u, aggregate tok/s, TTFT, and
the pipelined replay-only device ceiling (bursts of enqueued replays, one sync).

Run (P150x8):
    MESH_DEVICE=P150x8 HF_MODEL=/path/to/qwen38-27b-weights \\
        pytest models/demos/blackhole/qwen36/campaign/bench_decode.py -v -s

Knobs (env):
    QWEN38_BENCH_ISL          prompt depth per user (default 10240)
    QWEN38_BENCH_BATCH        concurrency (default 8)
    QWEN38_BENCH_STEPS        measured decode steps (default 256; use 1024 for the
                              headline OSL-matched number)
    QWEN38_BENCH_WARMUP       leading steps excluded from stats (default 16)
    QWEN38_BENCH_MODE         traced (default) | eager. Eager runs the decode step
                              op-by-op so a Tracy device profile gets per-op rows;
                              it is for attribution, not throughput numbers.
    QWEN38_BENCH_SYNTH_STATE  1 = skip prefill and decode at ISL-depth positions over
                              an unfilled KV cache. Device timing is content-
                              independent (same reads/writes at the same positions),
                              so this is the fast iteration mode; TTFT is not
                              reported and generated tokens are meaningless.
    QWEN38_BENCH_REAL_PROMPT  1 = corpus prompts instead of tiled local text
"""

import os
import time

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen36.campaign.bench_common import (
    bench_prompt,
    emit_bench_json,
    restore_gdn_tp,
    snapshot_gdn_tp,
    stats_ms,
)
from models.demos.blackhole.qwen36.demo.text_demo import BLOCK_SIZE, DEVICE_PARAMS, _MESH_SHAPE, _MULTI
from models.demos.blackhole.qwen36.tt.model import Qwen36Model
from models.tt_transformers.tt.common import copy_host_to_device

_ISL = int(os.environ.get("QWEN38_BENCH_ISL", "10240"))
_BATCH = int(os.environ.get("QWEN38_BENCH_BATCH", "8"))
_STEPS = int(os.environ.get("QWEN38_BENCH_STEPS", "256"))
_WARMUP = int(os.environ.get("QWEN38_BENCH_WARMUP", "16"))
_MODE = os.environ.get("QWEN38_BENCH_MODE", "traced")
_SYNTH = os.environ.get("QWEN38_BENCH_SYNTH_STATE") == "1"
_REPLAY_TRIALS = int(os.environ.get("QWEN38_BENCH_REPLAY_TRIALS", "3"))
_REPLAY_ITERS = int(os.environ.get("QWEN38_BENCH_REPLAY_ITERS", "32"))


@run_for_blackhole()
@pytest.mark.timeout(7200)
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_bench_decode(mesh_device):
    assert _MODE in ("traced", "eager"), f"QWEN38_BENCH_MODE must be traced|eager, got {_MODE}"
    assert _STEPS > 0 and _WARMUP >= 0 and _BATCH >= 1 and _ISL >= 1
    if not _MULTI:
        pytest.skip("goal metric is the TP serving path; set MESH_DEVICE=P150x4 or P150x8")
    device = mesh_device
    device.enable_program_cache()
    B, ISL = _BATCH, _ISL
    total_steps = _WARMUP + _STEPS

    # Per-user contiguous block range covering prompt + all decode steps. Round to a
    # multiple of 8 blocks: chunked SDPA reads each page-table row as an int32 stick
    # that must be 32-byte aligned (same constraint text_demo enforces).
    bpu = max(8, -(-(ISL + total_steps + 8) // BLOCK_SIZE))
    bpu = ((bpu + 7) // 8) * 8
    max_seq_len = bpu * BLOCK_SIZE

    t0 = time.time()
    model = Qwen36Model.from_pretrained(device, max_batch_size=B, max_seq_len=max_seq_len)
    load_s = time.time() - t0
    logger.info(f"model load {load_s:.1f}s ({len(model.layers)} layers, {model.num_devices} devices)")

    mesh = model.mesh_device
    vocab = model.args.vocab_size
    kv_shape = [B * bpu, model.args.n_local_kv_heads, BLOCK_SIZE, model.args.head_dim]
    model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=B)
    page_table = torch.stack([torch.arange(u * bpu, (u + 1) * bpu, dtype=torch.int32) for u in range(B)])

    ttft = None
    if _SYNTH:
        nxt = [100 + u for u in range(B)]
    else:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model.args.CKPT_DIR, trust_remote_code=True)
        token_ids = bench_prompt(ISL, tokenizer)
        assert token_ids.shape[1] == ISL, f"prompt is {token_ids.shape[1]} tokens, wanted {ISL}"
        token_list = [token_ids for _ in range(B)]
        signpost("inference_prefill")
        t0 = time.time()
        # The serving route for T > 256: sequential per-user chunk-outer prefill into
        # each user's block range. TTFT here is the last user's (worst-case) TTFT.
        pf_logits = model.prefill_chunked_peruser(token_list, page_table, valid_lens=[ISL] * B)
        ttnn.synchronize_device(mesh)
        ttft = time.time() - t0
        comp0 = ttnn.ConcatMeshToTensor(mesh, dim=0)
        nxt = [
            int(ttnn.to_torch(pf_logits[u], mesh_composer=comp0).reshape(-1, vocab)[0].float().argmax())
            for u in range(B)
        ]
        logger.info(f"prefill {B}x{ISL}: {ttft:.2f}s total ({ttft / B / ISL * 1000:.3f} ms/token/user)")

    pos = [ISL] * B
    dev = model.prepare_inputs_decode(
        torch.tensor(nxt, dtype=torch.int32).reshape(B, 1),
        torch.tensor(pos, dtype=torch.int32),
        page_table=page_table,
    )

    # Readback: per-shard on-device argmax folded into the trace (the fast served
    # greedy path from text_demo) when the sampler exists; full-logits host argmax
    # otherwise. on_device_logits=True requires model.sampling.
    use_shard = model.sampling is not None
    if use_shard:
        _per_shard = vocab // model.num_devices
        _MAXVAL_C = 32
        _MAXVAL_R = (((_per_shard + _MAXVAL_C - 1) // _MAXVAL_C) + 31) // 32 * 32
        _read_comp = ttnn.ConcatMeshToTensor(mesh, dim=0)
        _nd = model.num_devices

        def _maxval_dev_b(sharded_logits, Bn):
            padded = ttnn.pad(
                sharded_logits, [(0, 0), (0, 0), (0, 0), (0, _MAXVAL_R * _MAXVAL_C - _per_shard)], value=-1e30
            )
            grid = ttnn.reshape(padded, (1, Bn, _MAXVAL_R, _MAXVAL_C))
            part = ttnn.max(grid, dim=-1)
            part_row = ttnn.reshape(part, (1, 1, Bn, _MAXVAL_R))
            val = ttnn.max(part_row, dim=-1)
            ttnn.deallocate(padded)
            ttnn.deallocate(grid)
            ttnn.deallocate(part)
            ttnn.deallocate(part_row)
            return val

        def _argmax_dev_b(sharded_logits, Bn):
            logits_rm = ttnn.to_layout(sharded_logits, ttnn.ROW_MAJOR_LAYOUT)
            idx = ttnn.argmax(logits_rm, dim=-1, keepdim=False)
            ttnn.deallocate(logits_rm)
            return idx, _maxval_dev_b(sharded_logits, Bn)

        def _read_tok_b(idx_t, val_t, Bn):
            # Per-user winning device; only the first B columns are real users
            # (the rest are sampler-width padding).
            idxs = ttnn.to_torch(idx_t, mesh_composer=_read_comp).reshape(_nd, Bn)[:, :B].to(torch.int64)
            vals = ttnn.to_torch(val_t, mesh_composer=_read_comp).reshape(_nd, Bn)[:, :B]
            d = torch.argmax(vals, dim=0)
            return (d * _per_shard + idxs[d, torch.arange(B)]).tolist()

    def _fwd():
        out = model.ttnn_decode_forward(
            dev[0], dev[1], rot_mat_idxs=dev[2], page_table=dev[3], on_device_logits=use_shard
        )
        return out if use_shard else out[0]

    def _update(tokens_row, positions):
        # page_table lives in dev[3] with its address baked into the trace; only the
        # per-step inputs (tokens, cur_pos, rope idxs) change.
        host = model.prepare_decode_inputs_host(
            torch.tensor(tokens_row, dtype=torch.int32).reshape(B, 1),
            torch.tensor(positions, dtype=torch.int32),
            page_table=None,
        )
        copy_host_to_device(host[:3], device_tensors=dev[:3])

    eager = _MODE == "eager"
    trace_id = tt_logits = tt_idx = tt_val = None
    signpost("compile_decode")
    if eager:
        # Compile pass so the measured eager steps hit the program cache. It advances
        # GDN state one extra step, which is acceptable in an attribution-only mode.
        out = _fwd()
        if use_shard:
            wi, wv = _argmax_dev_b(out, out.shape[2])
            ttnn.deallocate(wi)
            ttnn.deallocate(wv)
        ttnn.synchronize_device(mesh)
    else:
        snap = snapshot_gdn_tp(model)
        warm = _fwd()
        if use_shard:
            wi, wv = _argmax_dev_b(warm, warm.shape[2])
            ttnn.deallocate(wi)
            ttnn.deallocate(wv)
        trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
        tt_logits = _fwd()
        if use_shard:
            # Fold per-shard argmax+max into the trace: tiny [num_devices, padded_B]
            # readback per step instead of full [B,1,vocab] logits.
            tt_idx, tt_val = _argmax_dev_b(tt_logits, tt_logits.shape[2])
        ttnn.end_trace_capture(mesh, trace_id, cq_id=0)
        restore_gdn_tp(model, snap)

    generated = [[t] for t in nxt]
    step_times = []
    signpost("inference_decode")
    for _ in range(total_steps):
        t_step = time.time()
        _update([g[-1] for g in generated], pos)
        if eager:
            out = _fwd()
            if use_shard:
                tt_idx, tt_val = _argmax_dev_b(out, out.shape[2])
            else:
                tt_logits = out
        else:
            ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        if use_shard:
            toks = _read_tok_b(tt_idx, tt_val, tt_idx.shape[-1])
        else:
            logits_step = model.process_output_decode(tt_logits, B)
            toks = torch.argmax(logits_step[:, 0, :vocab].float(), dim=-1).tolist()
        step_times.append(time.time() - t_step)
        for u in range(B):
            generated[u].append(toks[u])
        pos = [p + 1 for p in pos]
    signpost("inference_done")

    # Pipelined ceiling: bursts of enqueued replays with one sync. Positions and GDN
    # state stop advancing meaningfully, which does not affect timing.
    replay_ms = None
    if not eager:
        trials = []
        for _ in range(_REPLAY_TRIALS):
            t0 = time.time()
            for _ in range(_REPLAY_ITERS):
                ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            trials.append((time.time() - t0) / _REPLAY_ITERS)
        replay_ms = round(min(trials) * 1000.0, 3)
        ttnn.release_trace(mesh, trace_id)

    st = stats_ms(step_times[_WARMUP:])
    median_s = st["median_ms"] / 1000.0
    rows_identical = all(g == generated[0] for g in generated)
    degenerate = len(set(generated[0])) <= 1
    if not _SYNTH and not rows_identical:
        logger.warning("identical prompts decoded differently across users -- investigate before trusting numbers")
    if not _SYNTH and degenerate:
        logger.warning(f"degenerate generation (single repeated token {generated[0][:4]})")

    config = {
        "batch": B,
        "isl": ISL,
        "steps": _STEPS,
        "warmup": _WARMUP,
        "mode": _MODE,
        "synth_state": _SYNTH,
        "readback": "shard" if use_shard else "host",
        "n_layers": len(model.layers),
        "num_devices": model.num_devices,
    }
    metrics = {
        "step": st,
        "tsu_median": round(1.0 / median_s, 3),
        "tsu_mean": round(1000.0 / st["mean_ms"], 3),
        "agg_tok_s": round(B / median_s, 3),
        "replay_only_ms": replay_ms,
        "ttft_s": round(ttft, 3) if ttft is not None else None,
        "prefill_ms_per_tok_user": round(ttft / B / ISL * 1000, 4) if ttft is not None else None,
        "model_load_s": round(load_s, 1),
        "rows_identical": rows_identical,
        "degenerate": degenerate,
    }
    emit_bench_json("decode", config, metrics)
    logger.info(
        f"[bench_decode] B={B} ISL={ISL}: {metrics['tsu_median']:.2f} t/s/u "
        f"(median {st['median_ms']:.2f} ms, p90 {st['p90_ms']:.2f} ms, "
        f"replay-only {replay_ms} ms)"
        + (f", TTFT {ttft:.2f}s" if ttft is not None else "")
    )
    assert st["median_ms"] > 0
