# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Standalone repro for the Quasar 1D-mcast matmul DFB tile-counter underflow (llama32_1b decode QKV matmul).

The llama decode QKV matmul (forced interleaved on Quasar) is a 1D mcast_in0 matmul:
    in0 [1,1,32,2048] bf16 DRAM-interleaved  (M=32, K=2048 -> 64 K-tiles)
    w   [1,1,2048,3072] bf16 DRAM-interleaved (K=2048, N=3072 = Q2048+K512+V512)
    program_config: MatmulMultiCoreReuseMultiCast1DProgramConfig(grid 8x4, in0_block_w=2,
        per_core_M=1, per_core_N=3, out_subblock_h=1, out_subblock_w=3, mcast_in0=1) -> WIDTH_SHARDED L1

It runs to completion (after the mcast-rectangle fix) but then aborts:
    ERROR: UndefinedBehavior: qsr_tile_counter_check_error:
        tile counter occupancy=65535 exceeds capacity=4 (posted=64 acked=65)
occupancy = posted - acked = 64 - 65 = -1: the in0 DFB (cap 4 = 2 blocks x in0_block_num_tiles=2) is popped
ONE more than pushed (64 K-tiles pushed by the mcast receiver, 65 acked). This isolates just that matmul so
it can be run under TTSIM_QSR_DFB_TRACE=1 to name the counter and the extra-ack event.

Inputs are built bf16 via row-major upload + quasar.tilize (not from_torch(TILE), which hangs on the sim).

Run (Quasar sim, DFB trace to a file):
    TTSIM_QSR_DFB_TRACE=1 MESH_DEVICE=<qsr> TT_METAL_SIMULATOR=~/sim/libttsim.so \
        pytest tests/ttnn/unit_tests/operations/test_quasar_qkv_matmul_dfb.py 2> qkv_matmul_dfb_trace.txt
"""

import pytest
import torch
from loguru import logger

import ttnn

M, K, N = 32, 2048, 3072
GRID_X, GRID_Y = 8, 4


def _tile_bf16_dram(t_bf16, mesh_device):
    rm = ttnn.from_torch(
        t_bf16,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(mesh_device),
    )
    try:
        return ttnn.experimental.quasar.tilize(rm, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16)
    except (AttributeError, RuntimeError) as e:
        logger.info(f"[qkv-mm-repro] quasar.tilize unavailable ({e}); using mainline ttnn.tilize")
        return ttnn.tilize(rm, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def _pick_subblock_w(per_core_N):
    """Largest out_subblock_w <= 4 that divides per_core_N (out_subblock_h=1 keeps DEST usage small)."""
    for d in (4, 3, 2, 1):
        if per_core_N % d == 0:
            return d
    return 1


def _run_mcast_in0(mesh_device, gx, gy):
    """1D mcast_in0 matmul at the QKV decode K (=2048) on a gx*gy grid. in0 posts K/32=64 tiles (in0_block_w=2).
    N sized to the grid (per_core_N*num_cores*32) so it fits. GRID-DEPENDENT on craq-sim: 8x4 hits the in0 DFB
    tile-counter underflow (posted=64 acked=65); 3x2 / 2x3 / 2x1 pass -- so the sim remapper bug only trips at
    the larger grid. On the emulator, TT_METAL_LLK_ASSERTS=1 triggers a separate assert (LLK bug to fix)."""

    class _G:  # tiny grid holder so the shared body reads like the device-grid path
        x = gx
        y = gy

    grid = _G()
    num_cores = grid.x * grid.y
    # The in0 DFB underflow is driven by K (in0 = 64 tiles -> posted=64), NOT by N, so fix per_core_N small
    # (3 tiles, like the model's 8x4 case) and size N to the grid: N = per_core_N * num_cores * 32. This keeps
    # each core's output/in1 shard tiny so it fits L1 on the 2x1 emulator while still exercising the in0 DFB.
    per_core_N = 3
    n_used = per_core_N * num_cores * 32  # 3072 on 8x4 (matches the model), 192 on 2x1
    out_subblock_w = _pick_subblock_w(per_core_N)

    torch.manual_seed(0)
    a = torch.randn(1, 1, M, K, dtype=torch.bfloat16)
    w = torch.randn(1, 1, K, n_used, dtype=torch.bfloat16)
    at = _tile_bf16_dram(a, mesh_device)
    wt = _tile_bf16_dram(w, mesh_device)

    # The 1D-mcast program config type must match the linear op we call. On a build where ttnn.linear is the
    # experimental quasar op, use ttnn.experimental.quasar.{linear, MatmulMultiCoreReuseMultiCast1DProgramConfig};
    # else the mainline pair. (Both configs take the same fields.) This lets the repro run either place and
    # directly compares whether the experimental quasar matmul avoids the mainline in0 DFB underflow.
    cfg_kwargs = dict(
        compute_with_storage_grid_size=(grid.x, grid.y),
        in0_block_w=2,  # K=2048 -> 64 tiles -> 32 blocks -> in0 posted=64 (the DFB under test)
        out_subblock_h=1,
        out_subblock_w=out_subblock_w,
        per_core_M=1,
        per_core_N=per_core_N,
        fuse_batch=False,
        fused_activation=None,
        mcast_in0=True,
    )
    _q = getattr(ttnn.experimental, "quasar", None)
    _qcfg = getattr(_q, "MatmulMultiCoreReuseMultiCast1DProgramConfig", None)
    _qlin = getattr(_q, "linear", None)
    if _qcfg is not None and _qlin is not None:
        linear_fn = _qlin
        prog_cfg = _qcfg(**cfg_kwargs)
        which = "experimental.quasar.linear"
    else:
        linear_fn = ttnn.linear
        prog_cfg = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(**cfg_kwargs)
        which = "ttnn.linear (mainline)"
    out_memcfg = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1)
    compute_cfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    logger.info(
        f"[qkv-mm-repro] {which}: a[1,1,{M},{K}] x w[1,1,{K},{n_used}] mcast_in0 grid {grid.x}x{grid.y} "
        f"(num_cores={num_cores} per_core_N={per_core_N} out_subblock_w={out_subblock_w})"
    )
    out = linear_fn(
        at,
        wt,
        program_config=prog_cfg,
        memory_config=out_memcfg,
        dtype=ttnn.bfloat16,
        compute_kernel_config=compute_cfg,
    )
    ttnn.synchronize_device(mesh_device)
    o = ttnn.to_torch(ttnn.sharded_to_interleaved(out, ttnn.DRAM_MEMORY_CONFIG))
    logger.info(f"[qkv-mm-repro] out shape {tuple(o.shape)} finite={torch.isfinite(o).all().item()}")
    assert torch.isfinite(o).all(), "QKV matmul produced non-finite output"


def test_qkv_matmul_mcast_in0(mesh_device):
    """Device-grid run (adapts to whatever the device exposes)."""
    grid = mesh_device.compute_with_storage_grid_size()
    _run_mcast_in0(mesh_device, grid.x, grid.y)


@pytest.mark.parametrize("grid_xy", [(8, 4), (3, 2), (2, 3), (2, 1)], ids=["8x4", "3x2", "2x3", "2x1"])
def test_qkv_matmul_mcast_in0_grids(mesh_device, grid_xy):
    """Grid-dependence of the in0 DFB tile-counter underflow (per the craq-sim investigation):
    8x4 FAILS with the underflow; 3x2 / 2x3 / 2x1 PASS. Documents that the sim remapper bug only trips at the
    larger grid. Skips a grid the device can't provide."""
    gx, gy = grid_xy
    dev = mesh_device.compute_with_storage_grid_size()
    if dev.x < gx or dev.y < gy:
        pytest.skip(f"grid {gx}x{gy} needs a device >= that; device is {dev.x}x{dev.y}")
    _run_mcast_in0(mesh_device, gx, gy)
