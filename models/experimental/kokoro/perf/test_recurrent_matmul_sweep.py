# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Device-time sweep for the TextEncoder fused recurrent matmul ``[B,2H]@[2H,8H]``.

Isolates the exact production shape (B=2, H=256 -> [32,512]@[512,2048], LoFi bf16) and runs one
program-config candidate per process (selected by ``KOKORO_MM_CFG``), repeated N times so the
tracy device readback gives a stable median. Drive with ``perf/run_recurrent_matmul_sweep.sh``.

The matmul is bit-exact across program configs / output layouts (only the tiling + mcast schedule
change), so this is a pure device-time search with no PCC risk.
"""

from __future__ import annotations

import os

import torch

import ttnn

_ITERS = 60
_B, _H = 2, 256
_TILE = 32


def _configs(device):
    """Candidate (label, program_config, out_memcfg_fn, fp32_acc) tuples for [32,512]@[512,2048].

    v2: the first pass established 1D-mcast 8x8/ibw8/pcn1/ws = 3.73µs. Here we open the knobs it
    fixed — ``fp32_dest_acc`` on/off, 2D-mcast configs on fewer cores (per_core_N>1) with a wide
    ``out_subblock_w`` (the gatex win), and interleaved-vs-sharded output — for the skinny M=1-tile
    recurrent shape."""
    Nt = (8 * _H) // 32  # 64
    Kt = (2 * _H) // 32  # 16
    pad_m = ((_B + _TILE - 1) // _TILE) * _TILE

    def ws(gy, gx):
        return lambda: ttnn.create_sharded_memory_config(
            (pad_m, 8 * _H),
            core_grid=ttnn.CoreGrid(y=gy, x=gx),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        )

    def mm1d(gy, gx, ibw, pcn, mcast_in0=True):
        return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
            in0_block_w=ibw,
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=1,
            per_core_N=pcn,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=mcast_in0,
        )

    def mm2d(gy, gx, ibw, sw, pcn):
        return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
            in0_block_w=ibw,
            out_subblock_h=1,
            out_subblock_w=sw,
            per_core_M=1,
            per_core_N=pcn,
            transpose_mcast=False,
            fused_activation=None,
        )

    L1 = lambda: ttnn.L1_MEMORY_CONFIG
    DRAM = lambda: ttnn.DRAM_MEMORY_CONFIG
    T, F = True, False
    cfgs = [
        # 1D-mcast winner + fp32_dest_acc off + ibw variants at bf16-acc.
        ("1d_8x8_ibw8_pcn1_ws_fp32", mm1d(8, 8, 8, 1), ws(8, 8), T),  # current production
        ("1d_8x8_ibw8_pcn1_ws_bf16", mm1d(8, 8, 8, 1), ws(8, 8), F),
        ("1d_8x8_ibw16_pcn1_ws_bf16", mm1d(8, 8, 16, 1), ws(8, 8), F),
        ("1d_8x8_ibw8_pcn1_L1_bf16", mm1d(8, 8, 8, 1), L1, F),
        # 2D-mcast on fewer cores so per_core_N>1 lets out_subblock_w grow (the gatex lever).
        ("2d_1x8_ibw8_pcn8_sub1_ws_fp32", mm2d(1, 8, 8, 1, 8), ws(1, 8), T),
        ("2d_1x8_ibw8_pcn8_sub2_ws_fp32", mm2d(1, 8, 8, 2, 8), ws(1, 8), T),
        ("2d_1x8_ibw8_pcn8_sub4_ws_fp32", mm2d(1, 8, 8, 4, 8), ws(1, 8), T),
        ("2d_1x8_ibw8_pcn8_sub4_ws_bf16", mm2d(1, 8, 8, 4, 8), ws(1, 8), F),
        ("2d_1x8_ibw8_pcn8_sub8_ws_bf16", mm2d(1, 8, 8, 8, 8), ws(1, 8), F),
        ("2d_1x8_ibw8_pcn8_sub4_L1_fp32", mm2d(1, 8, 8, 4, 8), L1, T),
        ("2d_2x8_ibw8_pcn8_sub4_ws_fp32", mm2d(2, 8, 8, 4, 8), ws(2, 8), T),  # gy=2 (M pads to 2 tiles?)
        ("2d_1x16_ibw8_pcn4_sub4_ws_fp32", mm2d(1, 16, 8, 4, 4), ws(1, 16), T),  # 16 cores
        ("default_L1_fp32", None, L1, T),
    ]
    return cfgs


def test_recurrent_matmul_sweep(device):
    idx = int(os.environ.get("KOKORO_MM_CFG", "0"))
    cfgs = _configs(device)
    assert 0 <= idx < len(cfgs), f"KOKORO_MM_CFG must be 0..{len(cfgs)-1}"
    label, pc, out_mc_fn, fp32_acc = cfgs[idx]

    ckc = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_acc,
    )

    torch.manual_seed(0)
    h = ttnn.from_torch(
        torch.randn(_B, 2 * _H),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    w = ttnn.from_torch(
        torch.randn(2 * _H, 8 * _H),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    print(f"\n[SWEEP idx={idx}] cfg={label}")
    for _ in range(_ITERS):
        out = ttnn.linear(
            h,
            w,
            bias=None,
            memory_config=out_mc_fn(),
            dtype=h.dtype,
            compute_kernel_config=ckc,
            program_config=pc,
        )
        ttnn.deallocate(out)
    ttnn.synchronize_device(device)
