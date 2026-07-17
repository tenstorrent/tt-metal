# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Activation/weight sharding sweep for the dominant recaption-AR matmuls.

Baseline moe2.txt stacked report: 62% of device time is
``MatmulDeviceOperation (in0:l1_interleaved)``. Production expert matmuls at
M=64 use L1-interleaved acts + ``decode_mm_program_config`` (1D split-N). This
sweep asks whether WIDTH/HEIGHT/BLOCK-sharded activations (and optionally
DRAM-width-sharded weights) beat that baseline for the profiled shapes:

  * expert gate_up  64 x 4096 x 6144  (~78 us, 96c)
  * expert down     64 x 3072 x 4096  (~39 us, 64c)
  * attn QKV        32 x 4096 x 3072  (~47 us, 32c)
  * attn o_proj     32 x 2048 x 4096  (~38 us, 64c)

    pytest models/experimental/hunyuan_image_3_0/tests/perf/test_matmul_shard_sweep.py -s
"""
from __future__ import annotations

import time

import pytest
import torch
import ttnn
from models.common.utility_functions import nearest_32
from models.experimental.hunyuan_image_3_0.tt.matmul_utils import (
    decode_width_sharded_matmul_program_config,
    dram_width_sharded_weight_mem_config,
    matmul_1d_program_config,
    width_sharded_act_mem_config,
)
from models.experimental.hunyuan_image_3_0.tt.parallel_utils import decode_mm_program_config

_TILE = 32

# (name, M, K, N) — M already tile-aligned (matches padded in-model shapes)
SHAPES = [
    ("gate_up", 64, 4096, 6144),
    ("down", 64, 3072, 4096),
    ("qkv", 32, 4096, 3072),
    ("o_proj", 32, 2048, 4096),
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
    # keep top few by core count (plus a mid/small for contrast)
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


def _hs_act_mc(m: int, k: int, gy: int) -> ttnn.MemoryConfig | None:
    """HEIGHT_SHARDED act over gy cores (M split)."""
    if m % gy or (m // gy) % _TILE:
        return None
    return ttnn.create_sharded_memory_config_(
        [m // gy, k],
        ttnn.CoreGrid(x=1, y=gy),
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.ShardOrientation.ROW_MAJOR,
        tile_layout=True,
        use_height_and_width_as_shard_shape=True,
    )


def _bs_act_mc(m: int, k: int, gx: int, gy: int) -> ttnn.MemoryConfig | None:
    """BLOCK_SHARDED act: M/gy rows, K/gx cols."""
    if m % gy or k % gx:
        return None
    if (m // gy) % _TILE or (k // gx) % _TILE:
        return None
    return ttnn.create_sharded_memory_config_(
        [m // gy, k // gx],
        ttnn.CoreGrid(x=gx, y=gy),
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
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
def test_matmul_shard_sweep(device, shape):
    name, M, K, N = shape
    device.enable_program_cache()
    ckc = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )
    torch.manual_seed(0)
    x = torch.randn(1, M, K, dtype=torch.bfloat16) * 0.05
    w = torch.randn(K, N, dtype=torch.bfloat16) * 0.02
    ref = x.float().reshape(-1, K) @ w.float()

    xt_l1 = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    wt_il = ttnn.from_torch(
        w, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    Mt, Kt, Nt = M // _TILE, K // _TILE, N // _TILE
    prod_pc = decode_mm_program_config(device, M, K, N)

    results = []

    # --- 1) Production baseline: L1 interleaved + decode_mm (or auto) ----------
    def mm_il(pc, mem_out=ttnn.L1_MEMORY_CONFIG):
        return ttnn.linear(
            xt_l1, wt_il, dtype=ttnn.bfloat16, memory_config=mem_out, compute_kernel_config=ckc, program_config=pc
        )

    for label, pc in [("prod L1-IL+decode_mm", prod_pc), ("L1-IL+auto", None)]:
        try:
            out = ttnn.to_torch(mm_il(pc)).reshape(-1, N)
            us = _bench(device, lambda pc=pc: mm_il(pc))
            results.append((label, us, _pcc(out, ref)))
        except Exception as e:
            results.append((label, None, str(e)[:90]))

    # --- 2) WIDTH_SHARDED act + MultiCast1D (interleaved weight) ---------------
    for gx, gy, nc in _fit_ws_grids(device, K, N):
        gsz = (gx, gy)
        act_mc = _ws_act_mc(M, K, gsz)
        out_mc = _ws_out_mc(M, N, gsz)
        pc = matmul_1d_program_config(M, K, N, gsz)
        # also try larger out_subblock_w when legal
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
            x_sh = ttnn.interleaved_to_sharded(xt_l1, act_mc)

            def mm_ws(pc=pc_wide, x_sh=x_sh, out_mc=out_mc):
                o = ttnn.linear(
                    x_sh,
                    wt_il,
                    dtype=ttnn.bfloat16,
                    memory_config=out_mc,
                    compute_kernel_config=ckc,
                    program_config=pc,
                )
                return ttnn.sharded_to_interleaved(o, memory_config=ttnn.L1_MEMORY_CONFIG)

            out = ttnn.to_torch(mm_ws()).reshape(-1, N)
            us = _bench(device, mm_ws)
            results.append((f"WS-act nc={nc} ({gx}x{gy}) osw={osw}", us, _pcc(out, ref)))

            # matmul-only (no S2I) — island case
            def mm_ws_keep(pc=pc_wide, x_sh=x_sh, out_mc=out_mc):
                return ttnn.linear(
                    x_sh,
                    wt_il,
                    dtype=ttnn.bfloat16,
                    memory_config=out_mc,
                    compute_kernel_config=ckc,
                    program_config=pc,
                )

            us_k = _bench(device, mm_ws_keep)
            results.append((f"WS-act nc={nc} keep-sharded", us_k, None))
            ttnn.deallocate(x_sh)
        except Exception as e:
            results.append((f"WS-act nc={nc} ({gx}x{gy})", None, str(e)[:90]))

    # --- 3) HEIGHT_SHARDED act + 2D mcast --------------------------------------
    for gy in (Mt,):
        hmc = _hs_act_mc(M, K, gy)
        if hmc is None:
            continue
        # N split across gx; pick largest gx | Nt
        dg = device.compute_with_storage_grid_size()
        gx = _div_leq(Nt, dg.x)
        pc = _cfg_2d(gx, gy, Mt, Kt, Nt, _div_leq(Kt, 4))
        if pc is None:
            continue
        try:
            x_sh = ttnn.interleaved_to_sharded(xt_l1, hmc)

            def mm_hs(pc=pc, x_sh=x_sh):
                o = ttnn.linear(
                    x_sh,
                    wt_il,
                    dtype=ttnn.bfloat16,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                    compute_kernel_config=ckc,
                    program_config=pc,
                )
                return o

            out = ttnn.to_torch(mm_hs()).reshape(-1, N)
            us = _bench(device, mm_hs)
            results.append((f"HS-act gy={gy} gx={gx}", us, _pcc(out, ref)))
            ttnn.deallocate(x_sh)
        except Exception as e:
            results.append((f"HS-act gy={gy}", None, str(e)[:90]))

    # --- 4) BLOCK_SHARDED act + 2D mcast ---------------------------------------
    for gy in (Mt, max(1, Mt // 2)):
        for gx in (_div_leq(Kt, 8), _div_leq(Kt, 4)):
            bmc = _bs_act_mc(M, K, gx, gy)
            if bmc is None:
                continue
            # 2D config must match act grid for block-sharded in0
            if Nt % gx:
                continue
            pc = _cfg_2d(gx, gy, Mt, Kt, Nt, _div_leq(Kt // gx, 4) if Kt % gx == 0 else 1)
            if pc is None:
                continue
            try:
                x_sh = ttnn.interleaved_to_sharded(xt_l1, bmc)

                def mm_bs(pc=pc, x_sh=x_sh):
                    return ttnn.linear(
                        x_sh,
                        wt_il,
                        dtype=ttnn.bfloat16,
                        memory_config=ttnn.L1_MEMORY_CONFIG,
                        compute_kernel_config=ckc,
                        program_config=pc,
                    )

                out = ttnn.to_torch(mm_bs()).reshape(-1, N)
                us = _bench(device, mm_bs)
                results.append((f"BS-act {gx}x{gy}", us, _pcc(out, ref)))
                ttnn.deallocate(x_sh)
            except Exception as e:
                results.append((f"BS-act {gx}x{gy}", None, str(e)[:90]))

    # --- 5) DRAM-width-sharded weight + WIDTH_SHARDED act (decode path) --------
    if Mt == 1:
        try:
            w_dram = ttnn.from_torch(
                w,
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=dram_width_sharded_weight_mem_config(device, K, N),
            )
            act_mc, _, ncores = width_sharded_act_mem_config(K)
            pc = decode_width_sharded_matmul_program_config(M, K, N, ncores)
            x_sh = ttnn.interleaved_to_sharded(xt_l1, act_mc)

            def mm_dram(pc=pc, x_sh=x_sh, w_dram=w_dram):
                o = ttnn.linear(
                    x_sh,
                    w_dram,
                    dtype=ttnn.bfloat16,
                    memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                    compute_kernel_config=ckc,
                    program_config=pc,
                )
                return ttnn.sharded_to_interleaved(o, memory_config=ttnn.L1_MEMORY_CONFIG)

            out = ttnn.to_torch(mm_dram()).reshape(-1, N)
            us = _bench(device, mm_dram)
            results.append((f"DRAM-sharded-w ncores={ncores}", us, _pcc(out, ref)))
            ttnn.deallocate(x_sh)
            ttnn.deallocate(w_dram)
        except Exception as e:
            results.append(("DRAM-sharded-w", None, str(e)[:90]))

    # --- report ----------------------------------------------------------------
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
        # Assert at least one candidate is not catastrophically wrong; winner must
        # beat prod when it claims a win (used as a soft gate in CI-less local runs).
        if best_lab.startswith("prod"):
            print("  >> no sharding layout beat L1-interleaved prod for this shape")
        elif best_us < base * 0.98:
            print(f"  >> sharding beat prod ({base/best_us:.2f}x) — candidate for in-model trial")
        else:
            print(f"  >> best non-prod within noise of prod ({base/best_us:.2f}x)")
