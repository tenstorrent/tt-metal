# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""SCRATCH standalone microbenchmarks for the TP=1 prefill (Qwen3.8-27B on one P150 die), S=2048.

test_prefill_matmul_sweep_tp1: per prefill matmul shape, times candidate program configs (2D mcast, 1D mcast_in0,
N-split into k matmuls) with the model's ACTUAL compute configs (HiFi2 fp32-acc packer_l1_acc for attn/GDN, LoFi
fp32-acc packer_l1_acc for the MLP -- mlp.py picks compute_kernel_config_decode for the 4D prefill input). Prints
SWEEP_RESULT lines (min wall us over iterations, PCC / max-abs vs the current config's output).

test_gdn_out_chain_tp1: the GDN prefill output relayout (fused scan output -> gated bf16 [1,T,Nv*Dv]) as chains:
current (token-major op output + padded tilize + rms_norm + 5.7 ms ReshapeView) vs head-major alternatives.
Prints CHAIN_RESULT lines with wall us and bit-exactness vs the current chain.

  SWEEP_SHAPES=attn_qkv,mlp_gate SWEEP_ITERS=5 python -m tracy -r -v --op-support-count 20000 -m pytest \
      models/demos/blackhole/qwen36/tests/test_prefill_matmul_sweep_tp1_scratch.py -k sweep
"""

import math
import os
import time

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.demos.blackhole.qwen36.tt import tp_common as tpc

DEVICE_PARAMS = [{"l1_small_size": 24576, "trace_region_size": 64 * 1024 * 1024}]
S = int(os.environ.get("SWEEP_S", "2048"))

HIFI2 = tpc.COMPUTE_HIFI2  # HiFi2, fp32 acc, packer_l1_acc=True (attn / GDN projections)
# mlp.py: `ckc = compute_kernel_config_decode if T <= 1` with T = x.shape[1] == 1 for the 4D prefill input.
LOFI_MLP = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, packer_l1_acc=True
)

# name -> (K, N, weight dtype, compute cfg, out memory config, fused activation)
SHAPES = {
    "attn_qkv": (5120, 14336, ttnn.bfloat8_b, HIFI2, ttnn.DRAM_MEMORY_CONFIG, None),
    "attn_wo": (6144, 5120, ttnn.bfloat8_b, HIFI2, ttnn.L1_MEMORY_CONFIG, None),
    "gdn_qkvzab": (5120, 16480, ttnn.bfloat8_b, HIFI2, ttnn.DRAM_MEMORY_CONFIG, None),
    "gdn_out": (6144, 5120, ttnn.bfloat8_b, HIFI2, ttnn.DRAM_MEMORY_CONFIG, None),
    "mlp_gate": (5120, 17408, ttnn.bfloat4_b, LOFI_MLP, ttnn.DRAM_MEMORY_CONFIG, ttnn.UnaryOpType.SILU),
    "mlp_up": (5120, 17408, ttnn.bfloat4_b, LOFI_MLP, ttnn.DRAM_MEMORY_CONFIG, None),
    "mlp_down": (17408, 5120, ttnn.bfloat8_b, LOFI_MLP, ttnn.L1_MEMORY_CONFIG, None),
}
WTB = {ttnn.bfloat8_b: tpc.TILE_BYTES_BFP8, ttnn.bfloat4_b: tpc.TILE_BYTES_BFP4}


def _pc2d(cols, rows, m, k, n, in0_bw, sub_h, sub_w, obh=None, obw=None, act=None):
    per_core_M = max(1, math.ceil(m / 32 / rows))
    per_core_N = max(1, math.ceil(n / 32 / cols))
    kw = {}
    if obh is not None or obw is not None:
        kw = dict(out_block_h=obh or per_core_M, out_block_w=obw or per_core_N)
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(cols, rows),
        in0_block_w=in0_bw,
        out_subblock_h=sub_h,
        out_subblock_w=sub_w,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        transpose_mcast=False,
        fused_activation=act,
        fuse_batch=False,
        **kw,
    )


def _pc1d(cols, rows, m, k, n, per_core_N, in0_bw, sub_h, sub_w, obh, act=None):
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(cols, rows),
        in0_block_w=in0_bw,
        out_subblock_h=sub_h,
        out_subblock_w=sub_w,
        out_block_h=obh,
        out_block_w=per_core_N,
        per_core_M=math.ceil(m / 32),
        per_core_N=per_core_N,
        fuse_batch=True,
        fused_activation=act,
        mcast_in0=True,
    )


def _ob(pc):
    obh = getattr(pc, "out_block_h", None) or pc.per_core_M
    obw = getattr(pc, "out_block_w", None) or pc.per_core_N
    return obh, obw


def _cb_kb(pc, wtb):
    obh, obw = _ob(pc)
    return (2 * obh * pc.in0_block_w * 2048 + 2 * pc.in0_block_w * obw * wtb + obh * obw * (2048 + 4096)) / 1024


def _desc(pc):
    g = pc.compute_with_storage_grid_size
    obh, obw = _ob(pc)
    kind = "1D" if isinstance(pc, ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig) else "2D"
    return f"{kind}_g{g.x}x{g.y}_M{pc.per_core_M}_N{pc.per_core_N}_bw{pc.in0_block_w}_sb{pc.out_subblock_h}x{pc.out_subblock_w}_ob{obh}x{obw}"


def _legal2d(pc, k, n):
    per_core_M, per_core_N = pc.per_core_M, pc.per_core_N
    obh, obw = _ob(pc)
    if math.ceil(k / 32) % pc.in0_block_w:
        return False
    if obh % pc.out_subblock_h or obw % pc.out_subblock_w:
        return False
    if per_core_M % obh or per_core_N % obw:
        return False
    if not (obw == per_core_N or obh == 1):
        return False
    if pc.out_subblock_h * pc.out_subblock_w > 4:
        return False
    if per_core_N * (pc.compute_with_storage_grid_size.x - 1) >= math.ceil(n / 32):
        return False
    return True


def _cands2d(k, n, wtb, act, budget_kb):
    out = {}
    for rows in (8, 10):
        for cols in range(8, 12):
            for bw in (2, 4, 8):
                per_core_M = math.ceil(S / 32 / rows)
                per_core_N = math.ceil(n / 32 / cols)
                for obh in sorted({d for d in range(1, per_core_M + 1) if per_core_M % d == 0}):
                    for sh, sw in ((1, 4), (2, 2), (4, 1), (1, 2), (1, 3)):
                        pc = _pc2d(cols, rows, S, k, n, bw, sh, sw, obh, per_core_N, act)
                        if not _legal2d(pc, k, n) or _cb_kb(pc, wtb) > budget_kb:
                            continue
                        d = _desc(pc)
                        if d not in out:
                            out[d] = pc
    # keep area-4 subblocks, plus smaller ones only when no area-4 sibling exists
    keep = {d: pc for d, pc in out.items() if pc.out_subblock_h * pc.out_subblock_w == 4}
    have = {d.split("_sb")[0] for d in keep}
    for d, pc in out.items():
        if d.split("_sb")[0] not in have:
            keep[d] = pc
            have.add(d.split("_sb")[0])
    return list(keep.values())


def _cands1d(k, n, wtb, act, budget_kb, grid):
    out = {}
    nt = math.ceil(n / 32)
    kt = math.ceil(k / 32)
    mt = math.ceil(S / 32)
    for pcn in (4, 5, 6, 7, 8, 10, 12, 16):
        cores = math.ceil(nt / pcn)
        if cores > grid[0] * grid[1] or cores < 8:
            continue
        cols = min(grid[0], cores)
        rows = math.ceil(cores / cols)
        for bw in (4, 8):
            if kt % bw:
                continue
            for obh in (2, 4, 8, 16, 32, 64):
                if mt % obh:
                    continue
                subs = [(1, w) for w in (4, 3, 2, 1) if pcn % w == 0][:1]
                if pcn % 2 == 0 and obh % 2 == 0:
                    subs.append((2, 2))
                for sh, sw in subs:
                    if obh % sh:
                        continue
                    pc = _pc1d(cols, rows, S, k, n, pcn, bw, sh, sw, obh, act)
                    if _cb_kb(pc, wtb) > budget_kb:
                        continue
                    out.setdefault(_desc(pc), pc)
    return list(out.values())


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_prefill_matmul_sweep_tp1(mesh_device):
    dev = mesh_device
    dev.enable_program_cache()
    grid = dev.compute_with_storage_grid_size()
    grid = (grid.x, grid.y)
    logger.info(f"[SWEEP] grid {grid}  S={S}")
    shapes = os.environ.get("SWEEP_SHAPES", "attn_qkv,gdn_qkvzab,mlp_gate,mlp_down").split(",")
    iters = int(os.environ.get("SWEEP_ITERS", "5"))
    budget_kb = float(os.environ.get("SWEEP_CB_KB", "1300"))
    nsplits = [int(x) for x in os.environ.get("SWEEP_NSPLIT", "2,3,4").split(",") if x]
    only = os.environ.get("SWEEP_ONLY")
    kinds = os.environ.get("SWEEP_KINDS", "2d,1d,nsplit").split(",")
    tuning = tpc._PREFILL_TUNING[1]
    cfg_idx = 0
    torch.manual_seed(0)
    for name in shapes:
        k, n, wdt, ckc, omc, act = SHAPES[name]
        wtb = WTB[wdt]
        x_t = torch.randn(1, 1, S, k, dtype=torch.bfloat16) * 0.5
        w_t = torch.randn(k, n, dtype=torch.bfloat16) * 0.02
        x = ttnn.from_torch(
            x_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        w = ttnn.from_torch(
            w_t.reshape(1, 1, k, n),
            dtype=wdt,
            layout=ttnn.TILE_LAYOUT,
            device=dev,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        cur = tpc.create_prefill_mlp_matmul_program_config(
            S, k, n, fused_activation=act, max_cols=11, tuning=tuning, weight_tile_bytes=wtb
        )
        cands = [("CUR", cur)]
        if "2d" in kinds:
            cands += [("", pc) for pc in _cands2d(k, n, wtb, act, budget_kb) if _desc(pc) != _desc(cur)]
        if "1d" in kinds:
            cands += [("", pc) for pc in _cands1d(k, n, wtb, act, budget_kb, grid)]
        if only:
            cands = [c for c in cands if c[0] == "CUR" or only in _desc(c[1])]
        splits = []
        if "nsplit" in kinds:
            for ns in nsplits:
                nt = math.ceil(n / 32)
                per = math.ceil(nt / ns) * 32
                bounds = []
                s0 = 0
                while s0 < n:
                    s1 = min(n, s0 + per)
                    bounds.append((s0, s1))
                    s0 = s1
                ws = [
                    ttnn.from_torch(
                        w_t[:, a:b].reshape(1, 1, k, b - a).contiguous(),
                        dtype=wdt,
                        layout=ttnn.TILE_LAYOUT,
                        device=dev,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
                    for a, b in bounds
                ]
                pcs = [
                    tpc.create_prefill_mlp_matmul_program_config(
                        S, k, b - a, fused_activation=act, max_cols=11, tuning=tuning, weight_tile_bytes=wtb
                    )
                    for a, b in bounds
                ]
                splits.append((ns, bounds, ws, pcs))
        ref_out = None

        def _run(fn):
            o = fn()
            ttnn.synchronize_device(dev)
            best = 1e9
            signpost(f"cfg{cfg_idx}_start")
            for _ in range(iters):
                t0 = time.perf_counter()
                o2 = fn()
                ttnn.synchronize_device(dev)
                best = min(best, time.perf_counter() - t0)
                ttnn.deallocate(o2)
            signpost(f"cfg{cfg_idx}_stop")
            return o, best

        for tag, pc in cands:
            try:
                o, best = _run(
                    lambda: ttnn.linear(
                        x, w, compute_kernel_config=ckc, program_config=pc, memory_config=omc, dtype=ttnn.bfloat16
                    )
                )
            except Exception as e:  # noqa: BLE001
                msg = str(e).split("\n")[0][:160]
                print(
                    f"SWEEP_RESULT shape={name} cfg={cfg_idx} {tag or '-'} {_desc(pc)} cb={_cb_kb(pc, wtb):.0f}KB  FAILED {msg}",
                    flush=True,
                )
                cfg_idx += 1
                continue
            ot = ttnn.to_torch(o).float()
            if ref_out is None:
                ref_out, pcc, mad = ot, 1.0, 0.0
            else:
                pcc = float(torch.corrcoef(torch.stack([ref_out.flatten(), ot.flatten()]))[0, 1])
                mad = float((ref_out - ot).abs().max())
            ttnn.deallocate(o)
            print(
                f"SWEEP_RESULT shape={name} cfg={cfg_idx} {tag or '-'} {_desc(pc)} cb={_cb_kb(pc, wtb):.0f}KB  wall_us={best*1e6:.0f}  pcc_vs_cur={pcc:.6f} maxabs={mad:.4f}",
                flush=True,
            )
            cfg_idx += 1
        for ns, bounds, ws, pcs in splits:

            def _fn():
                outs = [
                    ttnn.linear(
                        x,
                        ws[i],
                        compute_kernel_config=ckc,
                        program_config=pcs[i],
                        memory_config=omc,
                        dtype=ttnn.bfloat16,
                    )
                    for i in range(len(ws))
                ]
                for oo in outs[1:]:
                    ttnn.deallocate(oo)
                return outs[0]

            try:
                o, best = _run(_fn)
            except Exception as e:  # noqa: BLE001
                print(
                    f"SWEEP_RESULT shape={name} cfg={cfg_idx} NSPLIT{ns} FAILED {str(e).split(chr(10))[0][:160]}",
                    flush=True,
                )
                cfg_idx += 1
                continue
            ot = ttnn.to_torch(o).float()
            a, b = bounds[0]
            pcc = (
                float(torch.corrcoef(torch.stack([ref_out[..., a:b].flatten(), ot.flatten()]))[0, 1])
                if ref_out is not None
                else 1.0
            )
            mad = float((ref_out[..., a:b] - ot).abs().max()) if ref_out is not None else 0.0
            ttnn.deallocate(o)
            print(
                f"SWEEP_RESULT shape={name} cfg={cfg_idx} NSPLIT{ns} slices={[b - a for a, b in bounds]} pcs={[_desc(p) for p in pcs]} wall_us={best*1e6:.0f}  pcc_vs_cur(slice0)={pcc:.6f} maxabs={mad:.4f}",
                flush=True,
            )
            cfg_idx += 1
            for wt in ws:
                ttnn.deallocate(wt)
        ttnn.deallocate(x)
        ttnn.deallocate(w)


# --------------------------------------------------------------------------------------------------------------- #
# GDN prefill output relayout chains (TP=1: Nv=48, Dv=128, T=S).
# --------------------------------------------------------------------------------------------------------------- #
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_gdn_out_chain_tp1(mesh_device):
    dev = mesh_device
    dev.enable_program_cache()
    Nv, Dv, T = 48, 128, S
    iters = int(os.environ.get("SWEEP_ITERS", "5"))
    torch.manual_seed(0)
    o_t = torch.randn(Nv, T, Dv, dtype=torch.float32)  # head-major [B*Nv, T, Dv] as the fused op returns (return_o_bh)
    z_t = torch.randn(1, T, Nv * Dv, dtype=torch.bfloat16) * 2
    nw_t = (torch.rand(1, 1, Dv) + 0.5).to(torch.bfloat16)
    _dram = ttnn.DRAM_MEMORY_CONFIG
    o_bh = ttnn.from_torch(o_t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=_dram)
    z = ttnn.from_torch(z_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=_dram)
    nw = ttnn.from_torch(nw_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=_dram)
    # Token-major ROW_MAJOR [1, T, Nv, Dv] fp32 = what the fused op hands back with return_o_bh=False
    # (it untilizes + permutes internally; those two ops are ~0.6 ms and are counted in the "current" chain below).
    o_tm_rm = ttnn.from_torch(
        o_t.permute(1, 0, 2).reshape(1, T, Nv, Dv).contiguous(),
        dtype=ttnn.float32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=dev,
        memory_config=_dram,
    )

    def _time(fn, tag):
        out = fn()
        ttnn.synchronize_device(dev)
        best = 1e9
        for _ in range(iters):
            t0 = time.perf_counter()
            o2 = fn()
            ttnn.synchronize_device(dev)
            best = min(best, time.perf_counter() - t0)
            ttnn.deallocate(o2)
        return out, best

    # CURRENT chain (tp.py TP=1 branch, _gdn_fuse_out=False): op-internal untilize+permute (emulated), tilize (pads Nv 48->64),
    # rms_norm on [1,T,64,Dv], reshape -> [1,T,Nv*Dv] (the 5.7 ms ReshapeView), gated = out_f * silu(z) -> bf16.
    def chain_cur():
        u = ttnn.to_layout(o_bh, ttnn.ROW_MAJOR_LAYOUT, memory_config=_dram)  # op-internal
        p = ttnn.permute(
            ttnn.reshape(u, (1, Nv, T, Dv)), (0, 2, 1, 3), memory_config=_dram
        )  # op-internal -> [1,T,Nv,Dv] RM
        ttnn.deallocate(u)
        o = ttnn.to_layout(p, ttnn.TILE_LAYOUT)
        ttnn.deallocate(p)
        n = ttnn.rms_norm(o, weight=nw, epsilon=1e-6, memory_config=_dram)
        ttnn.deallocate(o)
        f = ttnn.reshape(n, (1, T, Nv * Dv), memory_config=_dram)
        ttnn.deallocate(n)
        g = ttnn.multiply(f, ttnn.silu(z, memory_config=_dram), memory_config=_dram)
        ttnn.deallocate(f)
        return g

    # A: head-major norm, then untilize -> permute -> free RM reshape -> tilize.
    def chain_a():
        n = ttnn.rms_norm(o_bh, weight=nw, epsilon=1e-6, memory_config=_dram)  # [Nv, T, Dv] fp32 TILE
        u = ttnn.to_layout(n, ttnn.ROW_MAJOR_LAYOUT, memory_config=_dram)
        ttnn.deallocate(n)
        p = ttnn.permute(ttnn.reshape(u, (1, Nv, T, Dv)), (0, 2, 1, 3), memory_config=_dram)  # [1,T,Nv,Dv] RM
        ttnn.deallocate(u)
        f = ttnn.reshape(p, (1, 1, T, Nv * Dv))
        f = ttnn.to_layout(f, ttnn.TILE_LAYOUT, memory_config=_dram)
        ttnn.deallocate(p)
        g = ttnn.multiply(ttnn.reshape(f, (1, T, Nv * Dv)), ttnn.silu(z, memory_config=_dram), memory_config=_dram)
        ttnn.deallocate(f)
        return g

    # B: head-major norm, two half nlp_concat_heads (fp32 CBs fit at 24 heads), fp32 concat, silu_mul.
    def chain_b():
        n = ttnn.rms_norm(o_bh, weight=nw, epsilon=1e-6, memory_config=_dram)
        n4 = ttnn.reshape(n, (1, Nv, T, Dv))
        h = Nv // 2
        parts = []
        for i in range(2):
            sl = ttnn.slice(n4, (0, i * h, 0, 0), (1, (i + 1) * h, T, Dv), memory_config=_dram)
            parts.append(ttnn.experimental.nlp_concat_heads(sl, memory_config=_dram))
            ttnn.deallocate(sl)
        ttnn.deallocate(n)
        f = ttnn.concat(parts, dim=-1, memory_config=_dram)
        for p in parts:
            ttnn.deallocate(p)
        g = ttnn.multiply(ttnn.reshape(f, (1, T, Nv * Dv)), ttnn.silu(z, memory_config=_dram), memory_config=_dram)
        ttnn.deallocate(f)
        return g

    # B2: as B but silu_mul per half on the fp32 halves, then concat the bf16 halves (half the concat bytes).
    def chain_b2():
        n = ttnn.rms_norm(o_bh, weight=nw, epsilon=1e-6, memory_config=_dram)
        n4 = ttnn.reshape(n, (1, Nv, T, Dv))
        h = Nv // 2
        gs = []
        for i in range(2):
            sl = ttnn.slice(n4, (0, i * h, 0, 0), (1, (i + 1) * h, T, Dv), memory_config=_dram)
            f = ttnn.experimental.nlp_concat_heads(sl, memory_config=_dram)  # [1,1,T,h*Dv]
            ttnn.deallocate(sl)
            zs = ttnn.slice(z, (0, 0, i * h * Dv), (1, T, (i + 1) * h * Dv), memory_config=_dram)
            g = ttnn.multiply(ttnn.reshape(f, (1, T, h * Dv)), ttnn.silu(zs, memory_config=_dram), memory_config=_dram)
            ttnn.deallocate(f)
            ttnn.deallocate(zs)
            gs.append(g)
        ttnn.deallocate(n)
        out = ttnn.concat(gs, dim=-1, memory_config=_dram)
        for g in gs:
            ttnn.deallocate(g)
        return out

    # F: bring z to head-major instead (bf16, half the bytes), gate in head-major, ONE bf16 nlp_concat_heads.
    def chain_f():
        n = ttnn.rms_norm(o_bh, weight=nw, epsilon=1e-6, memory_config=_dram)  # [Nv,T,Dv] fp32
        zu = ttnn.to_layout(z, ttnn.ROW_MAJOR_LAYOUT, memory_config=_dram)
        zr = ttnn.reshape(zu, (1, T, Nv, Dv))
        zp = ttnn.permute(zr, (0, 2, 1, 3), memory_config=_dram)  # [1,Nv,T,Dv] RM
        ttnn.deallocate(zu)
        zt = ttnn.to_layout(zp, ttnn.TILE_LAYOUT, memory_config=_dram)
        ttnn.deallocate(zp)
        sz = ttnn.silu(zt, memory_config=_dram)
        ttnn.deallocate(zt)
        g = ttnn.multiply(ttnn.reshape(n, (1, Nv, T, Dv)), sz, memory_config=_dram)  # bf16 [1,Nv,T,Dv]
        ttnn.deallocate(sz)
        ttnn.deallocate(n)
        out = ttnn.experimental.nlp_concat_heads(g, memory_config=_dram)  # [1,1,T,Nv*Dv] bf16
        ttnn.deallocate(g)
        return ttnn.reshape(out, (1, T, Nv * Dv))

    chains = [
        ("CUR", chain_cur),
        ("A_untilize_permute_tilize", chain_a),
        ("B_2xconcat_heads_fp32", chain_b),
        ("B2_2xconcat_heads_gate_halves", chain_b2),
        ("F_z_headmajor_bf16_concat_heads", chain_f),
    ]
    only = os.environ.get("CHAIN_ONLY")
    ref = None
    for tag, fn in chains:
        if only and tag != "CUR" and only not in tag:
            continue
        try:
            out, best = _time(fn, tag)
        except Exception as e:  # noqa: BLE001
            print(f"CHAIN_RESULT {tag} FAILED {str(e).split(chr(10))[0][:200]}", flush=True)
            continue
        ot = ttnn.to_torch(out).float().reshape(T, Nv * Dv)
        ttnn.deallocate(out)
        if ref is None:
            ref = ot
            eq = "ref"
        else:
            d = (ref - ot).abs().max().item()
            eq = f"maxabs_vs_cur={d:.6f} bitexact={d == 0.0}"
        print(f"CHAIN_RESULT {tag} wall_us={best*1e6:.0f} {eq}", flush=True)
