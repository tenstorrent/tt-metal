# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Standalone repro for the Quasar sharded-RMSNorm mcast tile-counter fault, seen in llama32_1b decode.

The first decode-layer sharded RMSNorm runs a WIDTH-sharded LayerNorm: the reduction dim (dim=2048) is
split into 32 shards across an 8x4 core grid (shard [32, 64]), so RMSNorm's sum-of-squares must be summed
ACROSS cores via an all-to-all MCAST reduction. On the Quasar simulator that mcast write does not post the
receiver cores' DFB counters, so:

    [ttsim-qsr] ERROR: UndefinedBehavior: qsr_tile_counter_check_error:
        tile counter occupancy=65534 exceeds capacity=2 (posted=0 acked=2)

occupancy=65534 = 0xFFFE = -2 (16-bit underflow): a consumer popped 2 tiles the mcast producer never
posted. It is independent of the DFB implicit-sync config (fails with it on or off) -- a deeper Quasar
mcast-DFB-credit bug (possibly the known Quasar mcast-coordinate class, WH/BH NOC-swap on single-NOC
Quasar). The same mechanism underlies the sharded matmuls' in0/in1 mcast, so this is the first hit of
"sharded mcast ops broken on Quasar".

This reproduces just the sharded RMSNorm with the model's exact program config (rmsnorm_1d.py
_create_sharded_norm_program_config: block_w=2, subblock_w=2, block_h=1, grid 8x4). fp32_dest_acc_en is
False here to isolate the mcast fault from the separate bf16->Tf32 unpack gap (that is
test_quasar_fp32_acc_tf32_unpack.py).

NOT marked xfail -- the craq-sim tooling drives off a real FAIL. On Quasar this FAILS at the mcast
tile-counter check; on WH/BH it passes.

Run (Quasar sim):
    MESH_DEVICE=<qsr> TT_METAL_SIMULATOR=~/sim/libttsim.so \
        pytest tests/ttnn/unit_tests/operations/test_quasar_sharded_rmsnorm_mcast_fault.py
"""

import pytest
import torch
from loguru import logger

import ttnn

GRID_X, GRID_Y = 8, 4  # 32 cores -> 32 width shards of dim=2048 (matches the model's decode norm)
DIM = 2048
TILE = 32


def _readback(tt, mesh_device):
    try:
        num = mesh_device.get_num_devices()
    except Exception:
        num = 1
    if num > 1:
        return ttnn.to_torch(tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    return ttnn.to_torch(tt)


def _tile_bf16_interleaved(t_bf16, mesh_device):
    """bf16 TILE, DRAM-interleaved without the mainline from_torch(TILE) tilize (hangs on the Quasar sim):
    upload row-major, then tilize via the Gen2-native quasar op where available."""
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
        logger.info(f"[ln-mcast-repro] quasar.tilize unavailable ({e}); using mainline ttnn.tilize")
        return ttnn.tilize(rm, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def _i2s(x, memcfg):
    """interleaved_to_sharded via the Gen2-native quasar op where available (full-tile shard, so it does
    not hit the degenerate-shape i2s fault)."""
    qi2s = getattr(getattr(ttnn.experimental, "quasar", None), "interleaved_to_sharded", None)
    return (qi2s or ttnn.interleaved_to_sharded)(x, memcfg)


def test_sharded_rms_norm_mcast(mesh_device):
    """WIDTH-sharded ttnn.rms_norm (8x4 grid) -- reproduces the all-to-all mcast reduction that trips the
    Quasar tile-counter check (posted=0 acked=2). FAILS on Quasar, passes on WH/BH."""
    grid = mesh_device.compute_with_storage_grid_size()
    if grid.x < GRID_X or grid.y < GRID_Y:
        pytest.skip(f"needs an {GRID_X}x{GRID_Y} grid; device has {grid.x}x{grid.y}")

    ncores = GRID_X * GRID_Y
    shard_w = DIM // ncores  # 64
    block_w = DIM // ncores // TILE  # 2
    subblock_w = 2  # block_w % 2 == 0 (matches _create_sharded_norm_program_config)

    torch.manual_seed(0)
    x = torch.randn(1, 1, TILE, DIM, dtype=torch.bfloat16)
    xt = _tile_bf16_interleaved(x, mesh_device)

    core_rs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(GRID_X - 1, GRID_Y - 1))})
    shard_spec = ttnn.ShardSpec(core_rs, (TILE, shard_w), ttnn.ShardOrientation.ROW_MAJOR)
    sharded_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, shard_spec)

    logger.info(
        f"[ln-mcast-repro] interleaved_to_sharded (1,1,{TILE},{DIM}) -> WIDTH_SHARDED [32,{shard_w}] grid {GRID_X}x{GRID_Y}"
    )
    xs = _i2s(xt, sharded_mem)

    prog_cfg = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=[GRID_X, GRID_Y],
        subblock_w=subblock_w,
        block_h=1,
        block_w=block_w,
        inplace=False,
    )
    # fp32_dest_acc_en=False to isolate the mcast tile-counter fault from the bf16->Tf32 unpack gap.
    compute_cfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    logger.info("[ln-mcast-repro] sharded rms_norm (all-to-all mcast reduction) begin")
    out = ttnn.rms_norm(
        xs,
        epsilon=1e-5,
        program_config=prog_cfg,
        memory_config=sharded_mem,
        compute_kernel_config=compute_cfg,
    )
    ot = _readback(out, mesh_device).float()
    logger.info("[ln-mcast-repro] sharded rms_norm readback complete")

    assert torch.isfinite(ot).all(), "non-finite after sharded rms_norm"
    xf = x.float()
    ref = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-5)
    assert torch.allclose(ot.reshape(ref.shape), ref, atol=0.1, rtol=0.1), "value mismatch after sharded rms_norm"
