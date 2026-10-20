# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Standalone repro for the Quasar 2D-mcast matmul (unported Gen1 in0_sender), seen in llama32_1b prefill.

The llama32_1b prefill QKV/WO/MLP matmuls, when run interleaved on Quasar and left to the auto-picker,
select the 2D-mcast factory (matmul_multi_core_reuse_mcast_2d_optimized,
MatmulMultiCoreReuseMultiCastProgramConfig). That factory's `in0_sender` data-movement kernel still carries a
DataMovementGen1Config, so on Quasar (Gen2) program-spec validation FATALs:

    program_spec.cpp:1325: std::holds_alternative<DataMovementGen2Config>(data_movement_config)
    KernelSpec 'in0_sender' targets Gen2 (Quasar) but its DataMovementHardwareConfig holds a
    DataMovementGen1Config. Supply a Gen2 config (DataMovementGen2Config{}).

The e2e bring-up sidesteps this by pinning the 1D mcast_in0 config instead (that factory IS Gen2-ported --
see test_quasar_qkv_matmul_dfb.py). This file isolates the 2D-mcast path so it can be PORTED to Gen2: it is
expected to FAIL on Quasar today (the port target) and PASS once in0_sender (and any sibling 2D-mcast DM
kernels: in1_sender, in0/in1 receivers) are given DataMovementGen2Config. On WH/BH it passes now.

Shapes = the e2e prefill QKV matmul:
    a (in0): [1, 1, M=1024, K=2048]  bf16 TILE DRAM-interleaved
    w (in1): [1, 1, K=2048, N=3072]  bf16 TILE DRAM-interleaved
    2D mcast splits M across grid.y and N across grid.x.

Inputs are built bf16 via row-major upload + quasar.tilize (not from_torch(TILE), which hangs on the sim).

Run (Quasar sim):
    TTSIM_QSR_TC_LEGACY_TRUNCATION_ALIAS=0 TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE="3,2" \
        TT_METAL_SIMULATOR=~/sim/libttsim.so MESH_DEVICE=N150 \
        pytest tests/ttnn/unit_tests/operations/test_quasar_matmul_2d_mcast.py
"""

import pytest
import torch
from loguru import logger

import ttnn

# llama-3.2-1B prefill QKV matmul dims (the e2e case that auto-picked the 2D-mcast factory)
M, K, N = 1024, 2048, 3072


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
        logger.info(f"[mm-2d-repro] quasar.tilize unavailable ({e}); using mainline ttnn.tilize")
        return ttnn.tilize(rm, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def _compute_cfg():
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,  # Quasar bf16->Tf32 unpack gap
        packer_l1_acc=False,
    )


def _pcc(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    if torch.allclose(a, b):
        return 1.0
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def _divisor(v, cands=(4, 2, 1)):
    return next((d for d in cands if v % d == 0), 1)


def _run_2d(mesh_device, gx, gy):
    """2D-mcast matmul on a gx*gy grid. 2D mcast splits M across grid.y (per_core_M) and N across grid.x
    (per_core_N). EXPECTED TO FATAL on Quasar today: the 2D factory's in0_sender holds a Gen1 config
    (program_spec.cpp:1325). Port target -- passes once the 2D-mcast DM kernels get DataMovementGen2Config."""
    torch.manual_seed(0)
    a = torch.randn(1, 1, M, K, dtype=torch.bfloat16)
    w = torch.randn(1, 1, K, N, dtype=torch.bfloat16)
    at = _tile_bf16_dram(a, mesh_device)
    wt = _tile_bf16_dram(w, mesh_device)

    mt, kt, nt = M // 32, K // 32, N // 32
    per_core_M = max((mt + gy - 1) // gy, 1)  # M across grid.y
    per_core_N = max((nt + gx - 1) // gx, 1)  # N across grid.x
    osw = _divisor(per_core_N)

    def _blk(v, sub, cap):
        # largest divisor of v that is a multiple of `sub` and <= cap (bounds the L1 output block so the
        # 2D matmul STREAMS out_block-sized chunks to DRAM instead of holding the full per-core output --
        # otherwise out_block defaults to per_core and OOMs L1 at 2 nodes, exactly like 1D).
        best, d = sub, sub
        while d <= min(v, cap):
            if v % d == 0:
                best = d
            d += sub
        return best

    out_block_h = _blk(per_core_M, 1, 8)
    out_block_w = _blk(per_core_N, osw, 16)
    prog_cfg = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(gx, gy),
        in0_block_w=_divisor(kt),
        out_subblock_h=1,
        out_subblock_w=osw,
        out_block_h=out_block_h,
        out_block_w=out_block_w,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=False,
    )
    logger.info(
        f"[mm-2d-repro] 2D mcast a[1,1,{M},{K}] x w[1,1,{K},{N}] grid {gx}x{gy} "
        f"per_core_M={per_core_M} per_core_N={per_core_N} out_block={out_block_h}x{out_block_w}"
    )
    out = ttnn.linear(
        at,
        wt,
        program_config=prog_cfg,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        compute_kernel_config=_compute_cfg(),
    )
    ttnn.synchronize_device(mesh_device)
    o = ttnn.to_torch(out)
    ref = (a.float().reshape(M, K) @ w.float().reshape(K, N)).reshape(1, 1, M, N)
    pcc = _pcc(o, ref)
    logger.info(f"[mm-2d-repro] out shape {tuple(o.shape)} finite={torch.isfinite(o).all().item()} PCC={pcc:.5f}")
    assert torch.isfinite(o).all(), "2D-mcast matmul produced non-finite output"
    assert pcc > 0.99, f"2D-mcast matmul PCC too low: {pcc}"


@pytest.mark.timeout(3600)
def test_matmul_2d_mcast(mesh_device):
    """Device-grid 2D-mcast matmul (adapts to whatever the device exposes; needs a 2D grid to mcast both ways).
    NOTE the 3600s timeout: a 1024x2048x3072 matmul on the ~50 KHz functional sim takes many minutes -- the
    repo-default 300s pytest-timeout kills it mid-run (looks like a hang but is just slow)."""
    grid = mesh_device.compute_with_storage_grid_size()
    gx, gy = min(int(grid.x), 2), min(int(grid.y), 2)
    if gx < 2 or gy < 2:
        pytest.skip(f"2D mcast needs a >=2x2 grid; device is {grid.x}x{grid.y}")
    _run_2d(mesh_device, gx, gy)


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("grid_xy", [(2, 2), (2, 1)], ids=["2x2", "2x1"])
def test_matmul_2d_mcast_grids(mesh_device, grid_xy):
    """2D-mcast at 2x2 (true 2D) and 2x1 (the grid the e2e auto-picker used). After the Quasar port (Gen2 DM
    configs + mcast ascending-normalization in the factory, pack_init-on-pack-output-switch in the compute
    kernel) these PASS. out_block_h/w are bounded (see _run_2d) so the output streams and fits L1."""
    gx, gy = grid_xy
    dev = mesh_device.compute_with_storage_grid_size()
    if dev.x < gx or dev.y < gy:
        pytest.skip(f"grid {gx}x{gy} needs a device >= that; device is {dev.x}x{dev.y}")
    _run_2d(mesh_device, gx, gy)
