# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""ISL perf sweep: steady-state prefill (TTFT, prefill tok/s) + traced decode (tok/s/user).

Warms up (compile + trace capture) then measures steady-state, so the reported numbers exclude
one-time kernel-compile overhead. Sweeps input-sequence-length; decode uses the captured trace
(TracedDecode) for the realistic serving step. Reports device-bound (execute_trace blocking) and
E2E (incl. host input staging + argmax read-back) decode latency.

Layer count via M4_PERF_LAYERS (default 2 to validate the harness cheaply; set 36 for the headline
full-depth numbers). ISLs via M4_PERF_ISLS (comma-sep). bfp8 sharded experts = production config.
"""
import os
import time

import pytest
import torch
from loguru import logger
from transformers import AutoConfig

import ttnn
from models.demos.mistral4.tests.m4_text_reference import load_m4_weights
from models.demos.mistral4.tt.mistral4_generator import TracedDecode, TracedDecodeMLA, _repl
from models.demos.mistral4.tt.mistral4_text import TtMistral4TextModel

N_LAYERS = int(os.environ.get("M4_PERF_LAYERS", "2"))
ISLS = [int(s) for s in os.environ.get("M4_PERF_ISLS", "128,512,1024,2048,4096").split(",")]
DECODE_BATCHES = [int(b) for b in os.environ.get("M4_PERF_BATCHES", "1,8,32").split(",")]
DECODE_STEPS = 32


@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 200000000, "num_command_queues": 1}],
    indirect=True,
)
def test_m4_perf(mesh_device, reset_seeds):
    ckpt = os.environ["HF_MODEL"]
    cfg = AutoConfig.from_pretrained(ckpt).text_config
    tsd = load_m4_weights(ckpt, N_LAYERS)
    tt = TtMistral4TextModel(
        mesh_device, tsd, cfg, N_LAYERS, cfg.rms_norm_eps, shard_experts=True, expert_dtype=ttnn.bfloat8_b
    )
    rope, hidden = cfg.qk_rope_head_dim, cfg.hidden_size
    table = []

    # M4_PERF_SPARSE=1: also measure the sparse-MoE decode (top-k dispatch) per batch alongside dense.
    sparse_on = os.environ.get("M4_PERF_SPARSE") == "1"
    sdid = None
    if sparse_on:
        grid = mesh_device.compute_with_storage_grid_size()
        crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
        sdm = mesh_device.create_sub_device_manager([ttnn.SubDevice([crs])], 0)
        mesh_device.load_sub_device_manager(sdm)
        mesh_device.set_sub_device_stall_group([ttnn.SubDeviceId(0)])
        sdid = ttnn.SubDeviceId(0)

    def _time_decode(td, B, base=0):
        # base = the cur_pos the steps decode at; SDPA-decode attends the expanded-kv cache up to
        # cur_pos, so a larger base measures the realistic decode-step latency at that context length.
        x1, c1, s1 = torch.randn(B, 1, hidden) * 0.1, torch.randn(B, 1, 1, rope), torch.randn(B, 1, 1, rope)
        td.step(x1, c1, s1, [base] * B)  # warmup replay at this position
        ttnn.synchronize_device(mesh_device)
        t0 = time.time()
        for i in range(DECODE_STEPS):
            td.step(x1, c1, s1, [base + i] * B)
        ttnn.synchronize_device(mesh_device)
        return (time.time() - t0) / DECODE_STEPS

    # ---- decode throughput vs batch: per-step latency is ~flat across batch (the MoE streams each
    # expert's weights ONCE and applies them to all B tokens), so aggregate tok/s scales with batch.
    decode_rows = []
    for B in DECODE_BATCHES:
        kv = tt.init_kv_caches(B, max_seq=max(ISLS) + DECODE_STEPS + 64)
        td = TracedDecode(tt, mesh_device, B, hidden, rope, kv)
        step = _time_decode(td, B)
        decode_rows.append((B, step, 1 / step, B / step))
        logger.info(
            f"DECODE B={B}: {step*1e3:.2f}ms/step, {1/step:.1f} tok/s/user, {B/step:.0f} tok/s aggregate ({N_LAYERS}L)"
        )
        if sparse_on and B >= mesh_device.get_num_devices():
            kvs = tt.init_kv_caches(B, max_seq=max(ISLS) + DECODE_STEPS + 64)
            tds = TracedDecode(tt, mesh_device, B, hidden, rope, kvs, use_sparse=True, sub_device_id=sdid)
            ss = _time_decode(tds, B)
            spd = (step / ss - 1) * 100
            logger.info(
                f"DECODE-SPARSE B={B}: {ss*1e3:.2f}ms/step, {B/ss:.0f} tok/s aggregate ({N_LAYERS}L) — {spd:+.1f}% vs dense"
            )
    dev_dec = decode_rows[0][1]

    # ---- decode tok/s/user vs CONTEXT (batch-1): proves decode stays ~flat as cur_pos grows.
    # M4_PERF_DECODE_CTX=128,4096,8192,16384,32768 — one trace, timed at each starting position.
    ctx_rows = []
    decode_ctx = os.environ.get("M4_PERF_DECODE_CTX")
    if decode_ctx:
        ctxs = [int(c) for c in decode_ctx.split(",")]
        kvc = tt.init_kv_caches(1, max_seq=max(ctxs) + DECODE_STEPS + 64)
        tdc = TracedDecode(tt, mesh_device, 1, hidden, rope, kvc)
        for C in ctxs:
            step = _time_decode(tdc, 1, base=C)
            ctx_rows.append((C, step, 1 / step))
            logger.info(f"DECODE-CTX ctx={C}: {step*1e3:.2f}ms/step, {1/step:.1f} tok/s/user ({N_LAYERS}L)")

    # ---- COMPRESSED-latent (A6) decode tok/s/user vs CONTEXT (batch-1): the compressed KV (latent
    # kvl+rope=320/pos/layer vs expanded n_heads*qk*2=8192/pos/layer, ~25x smaller per device) reaches
    # far past the ~8K expanded-KV ceiling at full depth. M4_PERF_COMPRESSED_CTX=128,8192,32768,65536.
    cctx_rows = []
    comp_ctx = os.environ.get("M4_PERF_COMPRESSED_CTX")
    if comp_ctx:
        cctxs = [int(c) for c in comp_ctx.split(",")]
        ccache = tt.init_compressed_caches(1, max_seq=max(cctxs) + DECODE_STEPS + 64)
        tdcc = TracedDecodeMLA(tt, mesh_device, 1, hidden, rope, ccache)
        for C in cctxs:
            step = _time_decode(tdcc, 1, base=C)
            cctx_rows.append((C, step, 1 / step))
            logger.info(f"DECODE-COMPRESSED-CTX ctx={C}: {step*1e3:.2f}ms/step, {1/step:.1f} tok/s/user ({N_LAYERS}L)")

    # ---- prefill TTFT sweep (steady-state; warm up each shape once) ----
    for S in ISLS:
        x = _repl(torch.randn(1, S, hidden) * 0.1, mesh_device)
        cos = _repl(torch.randn(1, 1, S, rope), mesh_device)
        sin = _repl(torch.randn(1, 1, S, rope), mesh_device)
        kvp = tt.init_kv_caches(1, max_seq=S + 64)
        tt.forward_prefill(x, cos, sin, kvp)  # warmup (compile)
        ttnn.synchronize_device(mesh_device)
        t0 = time.time()
        tt.forward_prefill(x, cos, sin, kvp)
        ttnn.synchronize_device(mesh_device)
        ttft = time.time() - t0
        table.append((S, ttft, S / ttft))
        logger.info(f"PREFILL S={S}: TTFT {ttft*1e3:.1f}ms, {S/ttft:.0f} tok/s (steady-state, {N_LAYERS}L)")

    logger.info(f"=== Mistral-Small-4 perf ({N_LAYERS}L, sharded bfp8) ===")
    for B, step, tsu, agg in decode_rows:
        logger.info(f"  decode B={B:>3}: {step*1e3:7.2f}ms/step  {tsu:6.1f} tok/s/user  {agg:6.0f} tok/s aggregate")
    for C, step, tsu in ctx_rows:
        logger.info(f"  decode ctx {C:>6} (expanded-kv): {step*1e3:7.2f}ms/step  {tsu:6.1f} tok/s/user (B=1)")
    for C, step, tsu in cctx_rows:
        logger.info(f"  decode ctx {C:>6} (compressed) : {step*1e3:7.2f}ms/step  {tsu:6.1f} tok/s/user (B=1)")
    for S, ttft, tps in table:
        logger.info(f"  ISL {S:>5}: TTFT {ttft*1e3:8.1f}ms  prefill {tps:7.0f} tok/s")
    assert table and dev_dec > 0
