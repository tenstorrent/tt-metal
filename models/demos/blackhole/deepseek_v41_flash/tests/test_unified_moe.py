"""Standalone single-layer check of the deepseek_prefill routed-expert pipeline (tt/prefill_unified_moe.py) against the moe_compute prefill path
(DSV41PrefillMoE, T=256 per call, the baseline) and a torch reference (fp4-dequantised weights, swiglu clamp 10) on real layer weights.
Env: DSV41_UM_LAYER (3), DSV41_UM_N tokens per mesh row (4096), DSV41_UM_REF (16 tokens checked against torch), DSV41_UM_TIME (1), DSV41_UM_BASE (1: run the baseline)."""

import os
import time

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.reference.ref_layer import pcc
from models.demos.blackhole.deepseek_v41_flash.tests.test_moe_stages import chain_ms
from models.demos.blackhole.deepseek_v41_flash.tt import moe_weights as mw
from models.demos.blackhole.deepseek_v41_flash.tt.moe_block import DSV41MoEBlock
from models.demos.blackhole.deepseek_v41_flash.tt.prefill_layer import DSV41PrefillMoE
from models.demos.blackhole.deepseek_v41_flash.tt.prefill_unified_moe import (
    COLS,
    ROWS,
    TOPK,
    DSV41UnifiedMoE,
    get_shared,
    moe_cols,
    reduce_scatter_tokens,
    route_cols,
)
from models.demos.gpt_oss.config import mesh_4x8
from models.demos.gpt_oss.tt.ccl import CCLManager

D = 5120


def route(gate, h, N):
    parts = [gate.forward(ttnn.slice(h, [0, 0, 32 * i, 0], [1, 1, 32 * (i + 1), D])) for i in range(N // 32)]
    return ttnn.concat([p[0] for p in parts], dim=0), ttnn.concat([p[1] for p in parts], dim=0)


def torch_ref(x, idx, wgt, layer, limit=10.0):
    """x [t, D] bf16/fp32, idx/wgt [t, k] -> [t, D] fp32 (experts of the real checkpoint, fp4 dequantised, swiglu clamp)."""
    sh = mw._Shards()
    cache = {}

    def W(e):
        if e not in cache:
            p = f"layers.{layer}.ffn.experts.{e}."
            cache[e] = tuple(mw._fp4_expert(sh, p + k, torch.float32) for k in ("w1", "w3", "w2"))
        return cache[e]

    out = torch.zeros(x.shape[0], D)
    for t in range(x.shape[0]):
        for j in range(idx.shape[1]):
            g, u, dn = W(int(idx[t, j]))
            gg = (x[t].float() @ g).clamp(max=limit)
            uu = (x[t].float() @ u).clamp(-limit, limit)
            out[t] += float(wgt[t, j]) * ((torch.nn.functional.silu(gg) * uu) @ dn)
    return out


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "l1_small_size": 16384,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "trace_region_size": 400_000_000,
            },
            id="ring",
        )
    ],
    indirect=True,
)
@pytest.mark.timeout(7200)
@torch.no_grad()
def test_unified_moe(mesh_device):
    md = mesh_device
    layer = int(os.environ.get("DSV41_UM_LAYER", "3"))
    N = int(os.environ.get("DSV41_UM_N", "4096"))
    nref = int(os.environ.get("DSV41_UM_REF", "16"))
    do_base = os.environ.get("DSV41_UM_BASE", "1") == "1"
    w = mw.load_moe_layer(layer)
    blk = DSV41MoEBlock(md, w, batch_per_device=32, gate_bias_shift=0.0)
    blk.warmup()
    dec_chk = (
        os.environ.get("DSV41_UM_DECODE", "0") == "1"
    )  # decode block (own moe_compute weights, T=32) before / after the ring-weights prefill ops
    if dec_chk:
        torch.manual_seed(1)
        hj = ttnn.from_torch(
            torch.randn(1, 1, 32, D).to(torch.bfloat16),
            device=md,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(md),
        )
        hj_tok = ttnn.reshape(ttnn.to_layout(hj, ttnn.ROW_MAJOR_LAYOUT), [32, 1, 1, D])
        dec_out0 = ttnn.to_torch(ttnn.get_device_tensors(blk.forward(hj, hj_tok))[0]).float()
        dec_ms0 = chain_ms(md, lambda: ttnn.deallocate(blk.forward(hj, hj_tok)))
        print(f"UM decode block T=32 before: {dec_ms0:.3f} ms", flush=True)
    pm = None
    if do_base:
        pm = DSV41PrefillMoE(blk, T=32, g=8)
        pm.warmup()
    t0 = time.time()
    shared = get_shared(md, N)
    ring_mode = os.environ.get(
        "DSV41_UM_RING", "0"
    )  # 1: unified op reads the decode ring weights in place; 2: also build the copy and compare
    um_copy = None
    if ring_mode == "0":
        um = DSV41UnifiedMoE(md, layer)
    else:
        um = DSV41UnifiedMoE(md, layer, ring=(blk.decode.expert_state.tt_w0_w1, blk.decode.expert_state.tt_w2))
        if ring_mode == "2":
            um_copy = DSV41UnifiedMoE(md, layer)
    print(
        f"UM weights ready {time.time() - t0:.1f}s buffers: max_buf={shared.max_buf} max_per_expert={shared.max_per_expert}",
        flush=True,
    )

    torch.manual_seed(0)
    x = torch.randn(ROWS, N, D).to(torch.bfloat16)
    real = os.environ.get(
        "DSV41_UM_REAL"
    )  # real post-norm FFN inputs of the S=128 CPU dump (16 users x 128 tokens -> N = 512 tokens per mesh row)
    if real:
        assert N == 512
        d_ = torch.load(f"/mnt/tt-data/ssinghal/dsv4-prefill-s128/layer_{layer}.pt", mmap=True)["prefill"]["ffn_in"]
        x = d_.float().reshape(ROWS, N, D).to(torch.bfloat16)
    xd = ttnn.from_torch(
        x.reshape(ROWS, 1, N, D),
        device=md,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=(ROWS, COLS)),
    )
    dev = lambda t, r=0, c=0: ttnn.to_torch(ttnn.get_device_tensors(t)[r * COLS + c])

    mc, ccl = mesh_4x8(), CCLManager(md, num_links=2, topology=ttnn.Topology.Ring)
    n_own = N // COLS
    xo = (
        x.reshape(ROWS, N // 256, COLS, 32, D).permute(0, 2, 1, 3, 4).reshape(ROWS, COLS, n_own, D)
    )  # [row, col, own rows (g-major), D]
    xod = ttnn.from_torch(
        xo,
        device=md,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(0, 1), mesh_shape=(ROWS, COLS)),
    )  # per device [1,1,n_own,D]

    def unified():
        return moe_cols(um, blk.gate, xod, mc, ccl)

    if os.environ.get("DSV41_UM_DEBUG") == "1":
        h = mc.allgather(xod, ccl, axis=1, dim=2)
        ttnn.synchronize_device(md)
        want = xo.reshape(ROWS, COLS * n_own, D)
        badd = [
            (r, c) for r in range(ROWS) for c in range(COLS) if not torch.equal(dev(h, r, c).reshape(-1, D), want[r])
        ]
        print(f"UM debug: all-gather hidden mismatching devices: {len(badd)} {badd[:8]}", flush=True)
        sc_, ix_ = route(blk.gate, h, COLS * n_own)
        dbg = []
        sc2, ix2, _ = route_cols(blk.gate, xod, h, mc, ccl, mode="own", dbg=dbg)
        ttnn.synchronize_device(md)
        pk_, pg_ = dbg
        exp = torch.zeros(ROWS, COLS * n_own, 12)
        for r in range(ROWS):
            exp[r, :, :6] = dev(ix_, r, 0).float().reshape(-1, TOPK)
            exp[r, :, 6:] = dev(sc_, r, 0).float().reshape(-1, TOPK)
        for r in range(ROWS):
            for c in (0, 5):
                got = dev(pg_, r, c).float().reshape(-1, 12)
                bad_pg = (got != exp[r]).any(1).nonzero().flatten()
                # the own block of this device must also match its own pack
                own_pk = dev(pk_, r, c).float().reshape(-1, 12)
                print(
                    f"UM debug: packed AG dev ({r},{c}): rows differing from full-route expectation {len(bad_pg)} {bad_pg[:6].tolist()}; own pack vs full-route rows {int((own_pk != exp[r, c * n_own : (c + 1) * n_own]).any(1).sum())}",
                    flush=True,
                )
        for r in range(ROWS):
            for c in (0, 5):
                a_, b_ = dev(ix_, r, c).long().reshape(-1, TOPK), dev(ix2, r, c).long().reshape(-1, TOPK)
                wa, wb = dev(sc_, r, c).float().reshape(-1, TOPK), dev(sc2, r, c).float().reshape(-1, TOPK)
                bad_i = (a_ != b_).any(1).nonzero().flatten()
                print(
                    f"UM debug: routing own vs full dev ({r},{c}): idx mismatching rows {len(bad_i)} {bad_i[:6].tolist()} max |dw| {(wa - wb).abs().max():.4f}",
                    flush=True,
                )
        x_rm = ttnn.reshape(ttnn.to_layout(h, ttnn.ROW_MAJOR_LAYOUT), [1, COLS * n_own, D])
        part = um.forward(x_rm, sc_, ix_)
        ttnn.synchronize_device(md)
        pt = torch.zeros(ROWS, COLS, COLS * n_own, D)
        for r in range(ROWS):
            for c in range(COLS):
                pt[r, c] = dev(part, r, c).float().reshape(-1, D)
        rsd = reduce_scatter_tokens(part, ccl)
        ttnn.synchronize_device(md)
        for r in range(ROWS):
            for c in range(COLS):
                got = dev(rsd, r, c).float().reshape(n_own, D)
                ref = pt[r, :, c * n_own : (c + 1) * n_own].sum(0)
                bb = [i for i in range(0, n_own, 32) if pcc(got[i : i + 32], ref[i : i + 32]) < 0.999]
                print(f"UM debug: RS dev ({r},{c}) bad 32-row blocks {len(bb)}/{n_own // 32} {bb[:10]}", flush=True)
        # partial sums vs baseline (host sum over columns), natural order
        fu = torch.zeros(ROWS, N, D)
        ps = pt.sum(1)  # [row, c*n+g*32.. , D]
        for r in range(ROWS):
            for c in range(COLS):
                o = ps[r, c * n_own : (c + 1) * n_own].reshape(N // 256, 32, D)
                for g in range(N // 256):
                    fu[r, 256 * g + 32 * c : 256 * g + 32 * c + 32] = o[g]
        full_dbg = fu
    own = unified()
    ttnn.synchronize_device(md)
    full_u = torch.zeros(ROWS, N, D)
    for r in range(ROWS):
        for c in range(COLS):
            o = dev(own, r, c).float().reshape(N // 256, 32, D)
            for g in range(N // 256):
                full_u[r, 256 * g + 32 * c : 256 * g + 32 * c + 32] = o[g]
    if um_copy is not None:
        own_c = moe_cols(um_copy, blk.gate, xod, mc, ccl)
        ttnn.synchronize_device(md)
        full_c = torch.zeros(ROWS, N, D)
        for r in range(ROWS):
            for c in range(COLS):
                o = dev(own_c, r, c).float().reshape(N // 256, 32, D)
                for g in range(N // 256):
                    full_c[r, 256 * g + 32 * c : 256 * g + 32 * c + 32] = o[g]
        for r in range(ROWS):
            print(
                f"UM row {r}: PCC ring-weights vs unified-copy {pcc(full_u[r], full_c[r]):.6f} max|d| {(full_u[r] - full_c[r]).abs().max():.4f}",
                flush=True,
            )
    sc, ix = route(blk.gate, xd, N)
    sc_h = torch.stack([dev(sc, r).float().reshape(N, TOPK) for r in range(ROWS)])
    ix_h = torch.stack([dev(ix, r).long().reshape(N, TOPK) for r in range(ROWS)])
    cnt = torch.bincount(ix_h.flatten(), minlength=384)
    print(
        f"UM expert counts (all rows): max {cnt.max().item()} mean {cnt.float().mean():.1f} nonzero {int((cnt > 0).sum())}",
        flush=True,
    )
    for r in range(ROWS):
        print(f"UM row {r} tokens finite: {bool(torch.isfinite(full_u[r]).all())}", flush=True)

    if do_base:
        full_b = torch.zeros(ROWS, N, D)
        for g in range(N // 256):
            hg = ttnn.slice(xd, [0, 0, 256 * g, 0], [1, 1, 256 * (g + 1), D])
            tok = ttnn.reshape(ttnn.to_layout(hg, ttnn.ROW_MAJOR_LAYOUT), [256, 1, 1, D])
            mg = pm.forward(hg, tok)
            ttnn.synchronize_device(md)
            for r in range(ROWS):
                full_b[r, 256 * g : 256 * (g + 1)] = torch.cat(
                    [dev(mg, r, c).float().reshape(256, -1) for c in range(COLS)], dim=1
                )
        if os.environ.get("DSV41_UM_DEBUG") == "1":
            for r in range(ROWS):
                print(
                    f"UM debug row {r}: PCC host-summed partials vs baseline {pcc(full_dbg[r], full_b[r]):.6f}",
                    flush=True,
                )
        for r in range(ROWS):
            print(f"UM row {r}: PCC unified vs moe_compute baseline {pcc(full_u[r], full_b[r]):.6f}", flush=True)
            bad = [i for i in range(0, N, 32) if pcc(full_u[r, i : i + 32], full_b[r, i : i + 32]) < 0.99]
            print(f"UM row {r}: bad 32-row blocks {len(bad)}/{N // 32}: {bad[:40]}", flush=True)
    if nref:
        for r in (0, 3):
            ref = torch_ref(x[r, :nref], ix_h[r, :nref], sc_h[r, :nref], layer)
            print(f"UM row {r}: PCC unified vs torch (clamp) {pcc(full_u[r, :nref], ref):.6f}", flush=True)
            if do_base:
                print(f"UM row {r}: PCC baseline vs torch (clamp) {pcc(full_b[r, :nref], ref):.6f}", flush=True)
        # all-token self consistency: reduce-scatter path == host sum
    if os.environ.get("DSV41_UM_TIME", "1") == "1":

        def moe_only():
            x_rm = ttnn.reshape(ttnn.to_layout(xd, ttnn.ROW_MAJOR_LAYOUT), [1, N, D])
            s_ = um.forward(x_rm, sc, ix)
            ttnn.deallocate(x_rm)
            ttnn.deallocate(s_)

        def full():
            ttnn.deallocate(unified())

        def router_only():
            a, b = route(blk.gate, xod, n_own)
            ttnn.deallocate(a)
            ttnn.deallocate(b)

        def stage(k):
            def f():
                x_rm = ttnn.reshape(ttnn.to_layout(xd, ttnn.ROW_MAJOR_LAYOUT), [1, N, D])
                r_ = um.forward(x_rm, sc, ix, upto=k)
                ttnn.deallocate(x_rm)
                for t_ in r_ if isinstance(r_, tuple) else (r_,):
                    ttnn.deallocate(t_)

            return f

        if os.environ.get("DSV41_UM_STAGES") == "1":
            prev = 0.0
            for k, name in (
                (1, "bincount+cumsum"),
                (2, "+dispatch"),
                (3, "+experts"),
                (4, "+combine"),
                (99, "+post_combine_reduce"),
            ):
                t_ = chain_ms(md, stage(k))
                print(f"UM stage N={N} {name:24s} cumulative {t_:.3f} ms (+{t_ - prev:.3f})", flush=True)
                prev = t_
        print(f"UM time N={N}: router {chain_ms(md, router_only):.3f} ms", flush=True)
        print(f"UM time N={N}: unified moe (no router, no RS) {chain_ms(md, moe_only):.3f} ms", flush=True)
        print(f"UM time N={N}: unified full (router+moe+RS) {chain_ms(md, full):.3f} ms", flush=True)
        if do_base:

            def base():
                for g in range(N // 256):
                    hg = ttnn.slice(xd, [0, 0, 256 * g, 0], [1, 1, 256 * (g + 1), D])
                    tok = ttnn.reshape(ttnn.to_layout(hg, ttnn.ROW_MAJOR_LAYOUT), [256, 1, 1, D])
                    ttnn.deallocate(pm.forward(hg, tok))

            print(
                f"UM time N={N}: baseline moe_compute path ({N // 256} calls of 256) {chain_ms(md, base):.3f} ms",
                flush=True,
            )

    if dec_chk:
        dec_out1 = ttnn.to_torch(ttnn.get_device_tensors(blk.forward(hj, hj_tok))[0]).float()
        dec_ms1 = chain_ms(md, lambda: ttnn.deallocate(blk.forward(hj, hj_tok)))
        print(
            f"UM decode block T=32 after: {dec_ms1:.3f} ms (before {dec_ms0:.3f}); output bit-identical: {bool(torch.equal(dec_out0, dec_out1))}",
            flush=True,
        )
