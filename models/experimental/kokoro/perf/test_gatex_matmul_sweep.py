# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Device-time sweep for the TextEncoder BiLSTM's non-recurrent matmuls.

Three shape families (selected by ``KOKORO_MM_SHAPE``), each mirroring the production op exactly
(B=2, L=48, H=256, LoFi bf16, L1 in/out):

* ``gatex``   : gate precompute ``[2,48,512] @ [512,1024]`` (in1 broadcast, +bias) — runs 2x/forward
                (fwd + reversed input), 17.4µs each = the biggest non-recurrent matmul.
* ``reorder_in``  : anti-identity input reorder ``[2,48,48] @ [2,48,512]`` (batched, 5.4µs).
* ``reorder_out`` : anti-identity output reorder ``[2,48,48] @ [2,48,256]`` (batched, 2.7µs).

One program-config candidate per process (``KOKORO_MM_CFG``), repeated N times for a stable tracy
device-readback median. Drive with ``perf/run_gatex_matmul_sweep.sh``. Matmul numerics are
config-invariant (only tiling/mcast changes), so this is a pure device-time search.
"""

from __future__ import annotations

import os

import torch

import ttnn

_ITERS = 60
_B, _L, _H = 2, 48, 256
_TILE = 32


def _ckc(device, fp32_acc=True):
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_acc,
    )


def _mm2d(gy, gx, ibw, sh, sw, pm, pn):
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
        in0_block_w=ibw,
        out_subblock_h=sh,
        out_subblock_w=sw,
        per_core_M=pm,
        per_core_N=pn,
        transpose_mcast=False,
        fused_activation=None,
    )


def _gatex_configs():
    """[2,48,512]@[512,1024]: fuse-batched M=B*ceil(L/32)=4 tiles, K=512 (16 tiles), N=1024 (32 tiles).
    Sweep grid x out_subblock x fp32_dest_acc. Prior best: 2x8/ibw8/pm2/pn4/sub1x1/fp32acc = 11.6µs.
    out_subblock h*w cap: 4 tiles with fp32_dest_acc (DST=4), 8 without."""
    L1 = ttnn.L1_MEMORY_CONFIG
    cfgs = [("default_L1_fp32", None, L1, True), ("default_L1_bf16acc", None, L1, False)]
    # (gy, gx, pm, pn) tilings that satisfy gy*pm=4 (M-tiles), gx*pn=32 (N-tiles). Prior sweep
    # established gx=8/ibw8 best; here we open up out_subblock and fp32_dest_acc on the two best grids.
    tilings = [(2, 8, 2, 4), (4, 8, 1, 4)]
    subblocks = [(1, 1), (2, 1), (1, 2), (2, 2), (1, 4), (2, 4), (1, 8)]
    for gy, gx, pm, pn in tilings:
        for fp32 in (True, False):
            cap = 4 if fp32 else 8
            for sh, sw in subblocks:
                if sh > pm or sw > pn or sh * sw > cap:
                    continue
                ibws = (8, 16) if (sh, sw) == (1, 1) else (8,)  # only re-check ibw16 at the base subblock
                for ibw in ibws:
                    tag = f"2d_{gy}x{gx}_pm{pm}pn{pn}_sub{sh}x{sw}_ibw{ibw}_{'fp32' if fp32 else 'bf16'}"
                    cfgs.append((tag, _mm2d(gy, gx, ibw, sh, sw, pm, pn), L1, fp32))
    return cfgs


def _reorder_configs():
    """anti[2,48,48] @ X[2,48,N]: genuinely batched (both 3D). Try default L1/DRAM + fp32-off, and a
    few 2D program configs (num_blocks_y = B*ceil(L/32)=4 couples to grid.y — many will FATAL)."""
    L1 = ttnn.L1_MEMORY_CONFIG
    DRAM = ttnn.DRAM_MEMORY_CONFIG
    cfgs = [
        ("default_L1_fp32", None, L1, True),
        ("default_L1_bf16acc", None, L1, False),
        ("default_DRAM_fp32", None, DRAM, True),
    ]
    # N in tiles: reorder_in N=512->16 tiles, reorder_out N=256->8 tiles. K=48->2 tiles. M=4 fuse-batched.
    # Try grids sized so gy>=total M-tiles (4) and gx divides the N-tiles.
    for gy, gx, pm, pn, sh, sw in [
        (4, 8, 1, 2, 1, 1),
        (4, 4, 1, 4, 1, 1),
        (4, 8, 1, 1, 1, 1),
        (4, 2, 1, 8, 1, 1),
    ]:
        cfgs.append((f"2d_{gy}x{gx}_pm{pm}pn{pn}", _mm2d(gy, gx, 2, sh, sw, pm, pn), L1, True))
    return cfgs


def _build(device, shape):
    L1 = ttnn.L1_MEMORY_CONFIG
    torch.manual_seed(0)
    if shape == "gatex":
        a = ttnn.from_torch(
            torch.randn(_B, _L, 512), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=L1
        )
        b = ttnn.from_torch(
            torch.randn(512, 4 * _H), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=L1
        )
        bias = ttnn.from_torch(
            torch.randn(1, 1, 4 * _H), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=L1
        )
        return a, b, bias, _gatex_configs()
    if shape == "reorder_in":
        anti = torch.eye(_L).flip(0).reshape(1, _L, _L).expand(_B, _L, _L).contiguous()
        a = ttnn.from_torch(anti, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=L1)
        b = ttnn.from_torch(
            torch.randn(_B, _L, 512), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=L1
        )
        return a, b, None, _reorder_configs()
    if shape == "reorder_out":
        anti = torch.eye(_L).flip(0).reshape(1, _L, _L).expand(_B, _L, _L).contiguous()
        a = ttnn.from_torch(anti, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=L1)
        b = ttnn.from_torch(
            torch.randn(_B, _L, _H), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=L1
        )
        return a, b, None, _reorder_configs()
    raise AssertionError(shape)


def test_gatex_matmul_sweep(device):
    shape = os.environ.get("KOKORO_MM_SHAPE", "gatex")
    idx = int(os.environ.get("KOKORO_MM_CFG", "0"))
    a, b, bias, cfgs = _build(device, shape)
    assert 0 <= idx < len(cfgs), f"KOKORO_MM_CFG must be 0..{len(cfgs)-1} for shape={shape}"
    label, pc, out_mc, fp32_acc = cfgs[idx]
    ckc = _ckc(device, fp32_acc=fp32_acc)

    print(f"\n[SWEEP shape={shape} idx={idx}] cfg={label} (n_cfgs={len(cfgs)})")
    for _ in range(_ITERS):
        out = ttnn.linear(a, b, bias=bias, memory_config=out_mc, compute_kernel_config=ckc, program_config=pc)
        ttnn.deallocate(out)
    ttnn.synchronize_device(device)
