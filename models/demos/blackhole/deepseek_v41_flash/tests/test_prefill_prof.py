# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Whole-layer device-time profile of ONE traced prefill chunk (unified MoE, async host, optional DSV41_PREFILL_OPT): a subset of layers is built, the chunk forward
runs eagerly in the dynamic (traced-chunk) mode with the op recorder + device profiler drained after every layer (rows.json for tools/opsum.py / tools/prof2_sum.py),
then the same subset is captured as a trace and replayed for the wall time per chunk (no profiler, or profiler with PROF_TRACE_PROFILE=1).

Env: PROF_LAYERS ("0,1,2,3,14,20,21,24"), PROF_U (users per row), PROF_C (chunk tokens per user), PROF_S (padded prompt length), PROF_S0 (chunk start of the profiled chunk,
default S-C), PROF_MODE (eager|trace|both), PROF_REPS (timed full-prompt replays). Needs TT_METAL_DEVICE_PROFILER=1 + TT_METAL_PROFILER_DIR for the profile part."""

import gc
import json
import os
import time

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.reference import ref_layer as R
from models.demos.blackhole.deepseek_v41_flash.tt.decoder import DSV41Decoder  # noqa: F401
from models.demos.blackhole.deepseek_v41_flash.tt.device_head import DSV41DeviceEmbedding, DSV41DeviceHead
from models.demos.blackhole.deepseek_v41_flash.tt.engram import DSV41DeviceEngram
from models.demos.blackhole.deepseek_v41_flash.tt.loader import load_layer
from models.demos.blackhole.deepseek_v41_flash.tt.model import DSV41DecodeChain
from models.demos.blackhole.deepseek_v41_flash.tt.moe_weights import _Shards
from models.demos.blackhole.deepseek_v41_flash.tt.prefill_attention import DSV41PrefillAttention
from models.demos.blackhole.deepseek_v41_flash.tt.prefill_layer import DSV41PrefillLayer, DSV41PrefillMoE
from models.demos.blackhole.deepseek_v41_flash.tt.prefill_model import DSV41PrefillModel, T

U = int(os.environ.get("PROF_U", "4"))
C = int(os.environ.get("PROF_C", "1024"))
S = int(os.environ.get("PROF_S", "4096"))
S0 = int(os.environ.get("PROF_S0", str(S - C)))
LAYERS = [int(x) for x in os.environ.get("PROF_LAYERS", "0,1,2,3,14,20,21,24").split(",")]
MODE = os.environ.get("PROF_MODE", "both")
REPS = int(os.environ.get("PROF_REPS", "2"))
BASE = "/mnt/tt-data/ssinghal/dsv4-prefill-s128"


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "l1_small_size": 16384,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "trace_region_size": int(os.environ.get("DSV41_TRACE_REGION", "700000000")),
            },
            id="ring",
        )
    ],
    indirect=True,
)
@pytest.mark.timeout(7200)
@torch.no_grad()
def test_prefill_prof(mesh_device):
    md = mesh_device
    rows, cols = tuple(md.shape)
    log = lambda m: print(m, flush=True)
    B = 4 * U
    os.environ["DSV41_PF_SPARSE"] = "1"
    chain = DSV41DecodeChain(md, users_per_row=U, max_comp=256, log=log)
    sh = _Shards()
    pls, idx_w, sinks, first_moe = [], {}, {}, None
    for L in LAYERS:
        ref = torch.load(os.path.join(BASE, f"layer_{L}.pt"), mmap=True)
        meta = {
            "state": {k: torch.cat([v] * -(-B // v.shape[0]))[:B] for k, v in ref["state"].items()},
            "S": 1,
            "gate_cutoff": ref["gate_cutoff"],
        }
        w = load_layer(L, True, max(256, S + 64), True)
        if "indexer" in w:
            idx_w[L] = w["indexer"]
        sinks[L] = w["attn"]["attn_sink"]
        layer, attn = chain.build_layer(L, meta, w)
        del w, ref
        attn.prefill = DSV41PrefillAttention(attn, sh.get(f"layers.{L}.attn.attn_sink").float())
        pmoe = DSV41PrefillMoE(layer.moe, T=T, buffers=None if first_moe is None else first_moe.decode.buffers)
        first_moe = first_moe or pmoe
        pls.append((L, DSV41PrefillLayer(layer, attn.prefill, pmoe, T=T)))
        from models.demos.blackhole.deepseek_v41_flash.tt import uni_policy

        if uni_policy.decide(md, U, log):
            from models.demos.blackhole.deepseek_v41_flash.tt.dsv41_model import UNI_LAYERS
            from models.demos.blackhole.deepseek_v41_flash.tt.prefill_unified_moe import DSV41UnifiedMoE

            if L in UNI_LAYERS(LAYERS):
                es = layer.moe.decode.expert_state
                pls[-1][1].umoe = (
                    DSV41UnifiedMoE(md, L, log=log, ring=(es.tt_w0_w1, es.tt_w2))
                    if uni_policy.ring_requested()
                    else DSV41UnifiedMoE(md, L, log=log)
                )
        gc.collect()
        log(f"PROF layer {L} built")
    from models.demos.blackhole.deepseek_v41_flash.tt.prefill_sparse import attach_prefill_sparse

    attach_prefill_sparse({L: pl.pa for L, pl in pls}, idx_w, U, S, C, sinks, force=False, enable=True)
    engram_ids = [l for l in (1, 14) if l in LAYERS]
    dev_engram = {l: DSV41DeviceEngram(md, l, sh, mesh_config=chain.mesh_config, ccl=chain.ccl) for l in engram_ids}
    embedding = DSV41DeviceEmbedding(md, sh.get("embed.weight"), users_per_row=U)
    head = DSV41DeviceHead(md, sh.get("norm.weight").float(), sh.get("head.weight"), norm_eps=R.model_args().norm_eps)
    model = DSV41PrefillModel(md, pls, embedding, head, dev_engram, None, users_per_row=U)
    for _, pl in pls:
        pl.pa.U = U
    model.fake_rows = True
    g = torch.Generator().manual_seed(0)
    tokens = torch.randint(1000, 100000, (B, S), generator=g)
    if (
        os.environ.get("PROF_TOKENS", "real") == "real"
    ):  # real text (routing skew is realistic): a dumped prompt, rolled per user, tiled to S
        base = torch.load("/mnt/tt-data/ssinghal/dsv4-prefill-s4096b1/tokens.pt")["prefill_tokens"].reshape(-1)
        reps = -(-S // base.numel())
        base = base.repeat(reps)
        tokens = torch.stack([base.roll(u * 997)[:S] for u in range(B)])
    log(f"PROF built layers {LAYERS} U={U} C={C} S={S}")

    if MODE in ("eager", "both"):
        from models.demos.blackhole.deepseek_v41_flash.tests.op_table_recorder import OpRecorder

        prof = os.environ.get("TT_METAL_PROFILER_DIR")
        model.setup_dyn(C, S)
        bufs = model.alloc_inputs(C)
        model.upload_inputs(model.prep_inputs(tokens[:, S0 : S0 + C]), bufs)
        model.begin_chunk(S0, C)
        model.forward_device(bufs, S, S0, C, dyn=True)  # compile pass
        ttnn.synchronize_device(md)
        for _, pl in pls:
            pl.pa.reset_dyn()
        model.begin_chunk(S0, C)
        saved = {}
        rec = OpRecorder()
        rec.install()
        rec.rows, rec.zero, rec.marks = [], {}, []
        rec.enabled = True
        t_last = [time.perf_counter()]

        def hook(lid, xs, pres):
            ttnn.synchronize_device(md)
            rec.enabled = False
            if prof:
                ttnn.ReadDeviceProfiler(md)
                saved[lid] = list(rec.rows)
                json.dump(saved, open(os.path.join(prof, "rows.json"), "w"))
            rec.rows = []
            rec.enabled = True

        from models.demos.blackhole.deepseek_v41_flash.tt import prefill_layer as _pl

        _pl._PROF_EVERY = (
            1  # drain the profiler inside the layers only in the profiled pass (the compile pass is not drained)
        )
        os.environ["DSV41_PROF_EVERY"] = "1"
        model.forward_device(bufs, S, S0, C, hook=hook, dyn=True)
        _pl._PROF_EVERY = 0
        os.environ["DSV41_PROF_EVERY"] = "0"
        ttnn.synchronize_device(md)
        rec.enabled = False
        log("PROF eager profiled chunk done")
        if MODE == "both":
            ttnn.synchronize_device(md)

    if MODE in ("trace", "both"):
        for rep in range(REPS):
            t1 = time.perf_counter()
            lg = model.run_traced_chunks(tokens, C, hashes=torch.zeros(B, S, 1, dtype=torch.long))
            ttft = time.perf_counter() - t1
            n = S // C
            tm = model.timing
            log(
                f"PROF TRACED run {rep}: U={U} C={C} S={S} layers={len(LAYERS)}: total {ttft:.3f} s, replay loop {tm.get('total_replay_loop', 0):.3f} s = "
                f"{tm.get('total_replay_loop', 0) / n * 1e3:.1f} ms/chunk ({tm.get('total_replay_loop', 0) / n * 1e3 / len(LAYERS):.2f} ms/layer/chunk); "
                f"replay_per_chunk {tm.get('replay_per_chunk', 0) / n * 1e3:.1f} ms, host_per_chunk {tm.get('host_per_chunk', 0) / n * 1e3:.1f} ms, finite={bool(torch.isfinite(lg).all())}"
            )
        if os.environ.get("PROF_TRACE_PROFILE") == "1":
            ttnn.ReadDeviceProfiler(md)  # drop what the earlier replays logged
            model.run_traced_chunks(
                tokens[:, : 2 * C], C, hashes=torch.zeros(B, 2 * C, 1, dtype=torch.long), S_pad_max=S
            )
            ttnn.ReadDeviceProfiler(md)
    log("PROF DONE")
