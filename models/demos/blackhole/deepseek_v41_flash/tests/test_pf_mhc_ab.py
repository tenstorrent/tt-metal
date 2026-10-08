# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Whole-layer device-time profile of ONE traced prefill chunk (unified MoE, async host, optional DSV41_PREFILL_OPT): a subset of layers is built, the chunk forward
runs eagerly in the dynamic (traced-chunk) mode with the op recorder + device profiler drained after every layer (rows.json for tools/opsum.py / tools/prof2_sum.py),
then the same subset is captured as a trace and replayed for the wall time per chunk (no profiler, or profiler with PROF_TRACE_PROFILE=1).

Env: PROF_LAYERS ("0,1,2,3,14,20,21,24"), PROF_U (users per row), PROF_C (chunk tokens per user), PROF_S (padded prompt length), PROF_S0 (chunk start of the profiled chunk,
default S-C), PROF_MODE (eager|trace|both), PROF_REPS (timed full-prompt replays). Needs TT_METAL_DEVICE_PROFILER=1 + TT_METAL_PROFILER_DIR for the profile part."""

import gc
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

    tokens = torch.stack([tokens[0]] * B)  # identical prompts: every user must give identical numbers
    model.setup_dyn(C, S)
    bufs = model.alloc_inputs(C)
    caps = {}

    def run(mode):
        os.environ["DSV41_PF_MHC_UMOE"] = mode
        model.upload_inputs(model.prep_inputs(tokens[:, S0 : S0 + C]), bufs)
        for _, pl in pls:
            pl.pa.reset_dyn()
        model.begin_chunk(S0, C)
        out = {}

        def hook(lid, xs, pres):
            ttnn.synchronize_device(md)
            out[lid] = [
                [ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(x)] for x in xs
            ]  # [group][device] -> [32,1,4,D]

        model.forward_device(bufs, S, S0, C, hook=hook, dyn=True)
        ttnn.synchronize_device(md)
        return out

    first = os.environ.get("PROF_FIRST", "1")
    t0 = time.perf_counter()
    run(first)  # cold compile of the first pass
    log(f"COLD first pass (flag={first}): {time.perf_counter() - t0:.1f} s")
    t0 = time.perf_counter()
    run("1" if first == "0" else "0")
    log(f"COLD second pass (flag={'1' if first == '0' else '0'}): {time.perf_counter() - t0:.1f} s")
    a = run("0")
    b = run("1")
    n8 = len(a[LAYERS[0]])
    upr = C * U // 32  # chunks per row = U*C/32
    for lid in LAYERS:
        worst = {}
        for g in range(n8):
            for d in range(rows * cols):
                x0, x1 = a[lid][g][d], b[lid][g][d]
                col = d % cols
                chunk = g * 8 + col
                user = (chunk * 32) // C
                err = float((x0 - x1).norm() / x0.norm())
                worst[user] = max(worst.get(user, 0.0), err)
        log(f"AB layer {lid}: max rel L2 err (off vs on) per user-in-row {dict(sorted(worst.items()))}")
        # user-invariance inside one mode: same offset in different users must be identical
        for tag, o in (("off", a), ("on", b)):
            w = 0.0
            for d in range(rows * cols):
                col = d % cols
                ref = {}
                for g in range(n8):
                    chunk = g * 8 + col
                    key = (chunk * 32) % C
                    u = (chunk * 32) // C
                    if u == 0:
                        ref[key] = o[lid][g][d]
                for g in range(n8):
                    chunk = g * 8 + col
                    key = (chunk * 32) % C
                    if key in ref:
                        w = max(w, float((o[lid][g][d] - ref[key]).norm() / ref[key].norm()))
            log(f"AB layer {lid} {tag}: max rel L2 |user u - user 0| (same offset) = {w}")
    log("PROF DONE")
