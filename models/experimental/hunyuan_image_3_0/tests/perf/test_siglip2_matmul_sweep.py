# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Matmul program-config sweep for the SigLIP2 vision tower's linear shapes.

The Tracy report's aggregate "Stacked report" merges program-cache-hit repeats
across layers/devices in a way that makes its per-op-code Matmul totals hard to
trust in isolation (see persiglip1.txt discussion). This sweeps each DISTINCT
shape as a clean, isolated microbenchmark instead — same harness pattern as
tests/perf/test_matmul_shard_sweep.py — against the production baseline
(``wide_mm_program_config``, what ``l1_sharded_linear`` actually installs for
these M=1024 / Mt=32 shapes, since Mt>=8 routes there).

Shapes (VIT_CONFIG: hidden_size=1152, num_heads=16, intermediate_size=4304
padded to 4320; S=1024 max_num_patches; patch_dim=768):

  * patch_embed   1024 x  768 x 1152  (embeddings, once per forward)
  * qkv_proj      1024 x 1152 x 1536  (q/k/v proj, 3x per layer; padded_qkv_dim=1536)
  * out_proj      1024 x 1536 x 1152  (attn output proj, 1x per layer)
  * mlp_fc1       1024 x 1152 x 4320  (MLP up-proj, 1x per layer)
  * mlp_fc2       1024 x 4320 x 1152  (MLP down-proj, 1x per layer)

Run:
    pytest models/experimental/hunyuan_image_3_0/tests/perf/test_siglip2_matmul_sweep.py -s
"""

from __future__ import annotations

import time

import pytest
import torch
import ttnn
from models.common.utility_functions import nearest_32
from models.experimental.hunyuan_image_3_0.tt.matmul_utils import matmul_1d_program_config
from models.experimental.hunyuan_image_3_0.tt.parallel_utils import _largest_divisor_leq, wide_mm_program_config

_TILE = 32

# (name, M, K, N)
SHAPES = [
    ("patch_embed", 1024, 768, 1152),
    ("qkv_proj", 1024, 1152, 1536),
    ("out_proj", 1024, 1536, 1152),
    ("mlp_fc1", 1024, 1152, 4320),
    ("mlp_fc2", 1024, 4320, 1152),
]


def _bench(dev, fn, it=40):
    for _ in range(6):
        ttnn.deallocate(fn())
    ttnn.synchronize_device(dev)
    t = time.perf_counter()
    for _ in range(it):
        ttnn.deallocate(fn())
    ttnn.synchronize_device(dev)
    return (time.perf_counter() - t) / it * 1e6


def _pcc(a, b):
    a = a.float().flatten() - a.float().mean()
    b = b.float().flatten() - b.float().mean()
    return (a @ b / (a.norm() * b.norm()).clamp(min=1e-12)).item()


def _div_leq(x, cap):
    return next((d for d in range(min(x, cap), 0, -1) if x % d == 0), 1)


def _fit_ws_grids(device, k: int, n: int):
    """Candidate (gx, gy) grids where both K and N tiles divide evenly by nc."""
    dg = device.compute_with_storage_grid_size()
    kt, nt = k // _TILE, n // _TILE
    out = []
    for gy in range(1, dg.y + 1):
        for gx in range(1, dg.x + 1):
            nc = gx * gy
            if kt % nc == 0 and nt % nc == 0:
                out.append((gx, gy, nc))
    out.sort(key=lambda t: -t[2])
    keep, seen = [], set()
    for g in out:
        if g[2] in seen:
            continue
        seen.add(g[2])
        keep.append(g)
        if len(keep) >= 4:
            break
    return keep


def _ws_act_mc(m: int, k: int, grid_size: tuple[int, int]) -> ttnn.MemoryConfig:
    nc = grid_size[0] * grid_size[1]
    return ttnn.create_sharded_memory_config_(
        [nearest_32(m), k // nc],
        ttnn.CoreGrid(x=grid_size[0], y=grid_size[1]),
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.ShardOrientation.ROW_MAJOR,
        tile_layout=True,
        use_height_and_width_as_shard_shape=True,
    )


def _ws_out_mc(m: int, n: int, grid_size: tuple[int, int]) -> ttnn.MemoryConfig:
    nc = grid_size[0] * grid_size[1]
    return ttnn.create_sharded_memory_config_(
        [nearest_32(m), n // nc],
        ttnn.CoreGrid(x=grid_size[0], y=grid_size[1]),
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.ShardOrientation.ROW_MAJOR,
        tile_layout=True,
        use_height_and_width_as_shard_shape=True,
    )


def _cfg_2d(gx, gy, Mt, Kt, Nt, ibw):
    if Mt % gy or Nt % gx:
        return None
    pcm, pcn = Mt // gy, Nt // gx
    osw = next((w for w in (4, 3, 2, 1) if pcn % w == 0), 1)
    osh = next((h for h in range(max(1, 4 // osw), 0, -1) if pcm % h == 0), 1)
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(gx, gy),
        in0_block_w=ibw,
        out_subblock_h=osh,
        out_subblock_w=osw,
        per_core_M=pcm,
        per_core_N=pcn,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=False,
    )


@pytest.mark.parametrize("shape", SHAPES, ids=[s[0] for s in SHAPES])
def test_siglip2_matmul_sweep(device, shape):
    name, M, K, N = shape
    device.enable_program_cache()
    ckc = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    torch.manual_seed(0)
    x = torch.randn(1, M, K, dtype=torch.bfloat16) * 0.05
    w = torch.randn(K, N, dtype=torch.bfloat16) * 0.02
    ref = x.float().reshape(-1, K) @ w.float()

    xt_dram = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    wt_il = ttnn.from_torch(
        w, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    Mt, Kt, Nt = M // _TILE, K // _TILE, N // _TILE
    prod_pc = wide_mm_program_config(device, M, K, N)  # what l1_sharded_linear actually installs (Mt=32>=8)

    results = []

    # --- 1) Production baseline: DRAM-interleaved act + wide_mm 2D-mcast --------
    def mm_prod(pc=prod_pc):
        return ttnn.linear(
            xt_dram,
            wt_il,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=ckc,
            program_config=pc,
        )

    for label, pc in [("prod DRAM-IL+wide_mm", prod_pc), ("DRAM-IL+auto", None)]:
        try:
            out = ttnn.to_torch(mm_prod(pc)).reshape(-1, N)
            us = _bench(device, lambda pc=pc: mm_prod(pc))
            results.append((label, us, _pcc(out, ref)))
        except Exception as e:
            results.append((label, None, str(e)[:90]))

    # --- 2) WIDTH_SHARDED act + MultiCast1D (interleaved weight) ---------------
    for gx, gy, nc in _fit_ws_grids(device, K, N):
        gsz = (gx, gy)
        act_mc = _ws_act_mc(M, K, gsz)
        out_mc = _ws_out_mc(M, N, gsz)
        pc = matmul_1d_program_config(M, K, N, gsz)
        pcn = Nt // nc
        osw = next((w for w in (4, 3, 2, 1) if pcn % w == 0), 1)
        pc_wide = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=gsz,
            in0_block_w=_div_leq(Kt // nc, 8) if Kt % nc == 0 else 1,
            out_subblock_h=1,
            out_subblock_w=osw,
            per_core_M=Mt,
            per_core_N=pcn,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )
        try:
            x_sh = ttnn.interleaved_to_sharded(xt_dram, act_mc)

            def mm_ws(pc=pc_wide, x_sh=x_sh, out_mc=out_mc):
                o = ttnn.linear(
                    x_sh,
                    wt_il,
                    dtype=ttnn.bfloat16,
                    memory_config=out_mc,
                    compute_kernel_config=ckc,
                    program_config=pc,
                )
                return ttnn.sharded_to_interleaved(o, memory_config=ttnn.DRAM_MEMORY_CONFIG)

            out = ttnn.to_torch(mm_ws()).reshape(-1, N)
            us = _bench(device, mm_ws)
            results.append((f"WS-act nc={nc} ({gx}x{gy}) osw={osw}", us, _pcc(out, ref)))
            ttnn.deallocate(x_sh)
        except Exception as e:
            results.append((f"WS-act nc={nc} ({gx}x{gy})", None, str(e)[:90]))

    # --- 3) 2D-mcast grid variants (rectangular, full grid coverage) -----------
    dg = device.compute_with_storage_grid_size()
    for gy in sorted({dg.y, _largest_divisor_leq(Mt, dg.y), max(1, dg.y // 2)}):
        if Mt % gy:
            continue
        for gx in sorted({dg.x, _largest_divisor_leq(Nt, dg.x), max(1, dg.x // 2)}):
            if Nt % gx:
                continue
            pc = _cfg_2d(gx, gy, Mt, Kt, Nt, _div_leq(Kt, 4))
            if pc is None:
                continue
            try:

                def mm_2d(pc=pc):
                    return ttnn.linear(
                        xt_dram,
                        wt_il,
                        dtype=ttnn.bfloat16,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        compute_kernel_config=ckc,
                        program_config=pc,
                    )

                out = ttnn.to_torch(mm_2d()).reshape(-1, N)
                us = _bench(device, mm_2d)
                results.append((f"2D-mcast {gx}x{gy}", us, _pcc(out, ref)))
            except Exception as e:
                results.append((f"2D-mcast {gx}x{gy}", None, str(e)[:90]))

    # --- report ------------------------------------------------------------
    base = next((us for _, us, _ in results if us is not None), None)
    print(f"\n=== {name} {M} x {K} x {N} (prod_pc={'yes' if prod_pc else 'None'}) ===")
    ranked = sorted(((us, lab, p) for lab, us, p in results if us is not None), key=lambda t: t[0])
    for lab, us, pcc in results:
        if us is None:
            print(f"  {lab:42}    FAIL  {pcc}")
        else:
            pcc_s = f"PCC={pcc:.6f}" if isinstance(pcc, float) else "PCC=n/a"
            print(f"  {lab:42} {us:8.1f} us  ({base/us:4.2f}x)  {pcc_s}")
    if ranked:
        best_us, best_lab, best_pcc = ranked[0]
        print(f"  >> BEST: {best_lab} @ {best_us:.1f} us ({base/best_us:.2f}x vs prod)")
        if best_lab.startswith("prod"):
            print("  >> no candidate beat the production wide_mm config for this shape")
        elif best_us < base * 0.98:
            print(f"  >> candidate beat prod ({base/best_us:.2f}x) — worth trying in-model")
        else:
            print(f"  >> best non-prod within noise of prod ({base/best_us:.2f}x)")
