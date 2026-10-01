# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Regression test for the Quasar sharded-RMSNorm tile-counter fault (#57771), seen in llama32_1b decode.

The first decode-layer sharded RMSNorm runs a WIDTH-sharded LayerNorm: the reduction dim (dim=2048) is
split into 32 shards across an 8x4 core grid (shard [32, 64]), with the all-to-all mcast reduction on.
On Quasar it faulted:

    [ttsim-qsr] ERROR: UndefinedBehavior: qsr_tile_counter_check_error:
        tile counter occupancy=65534 exceeds capacity=2 (posted=0 acked=2)

Root cause (craq-sim, 2026-09-24/25): the compute kernel (kernels/compute/layernorm_sharded.cpp, RMSNORM
build) pops the input DFB -- dfb_in0 borrows the already-resident input shard -- during the x - E[x] pass
without ever pushing it. WH/BH tolerate an ack with no post; Quasar's hardware tile counters do not
(occupancy 0xFFFE = -2: two pops, zero posts). The mcast reduction is not at fault: its ex_global counter
was balanced. Fix: the kernel reads the resident input by absolute tile index (index_h_offset) and never
pushes or pops it -- the idiom layernorm_sharded_pre_allgather.cpp already used -- so the DFB's tile
counters are never touched; only the fused pre-add scratch buffer is still popped, once, after its last
read. Plus a guarded pack_init after every packer output switch (Quasar bakes the pack destination at
pack_init; pack_reconfig_data_format only reprograms the format gasket). A second, independent fault --
the packer firmware clearing the intra-tensix remapper pairs while the unpacker still pops -- is fixed in
firmware by #57984.

This reproduces just the sharded RMSNorm with the model's exact program config (rmsnorm_1d.py
_create_sharded_norm_program_config: block_w=2, subblock_w=2, block_h=1, grid 8x4). fp32_dest_acc_en is
False here to keep it separate from the bf16->Tf32 unpack gap in craq-sim (that is
test_quasar_fp32_acc_tf32_unpack.py, tenstorrent/craq-sim#403).

NOT marked xfail -- the craq-sim tooling drives off a real FAIL. Needs the full 8x4 worker grid and skips
on smaller grids (TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE is ignored on Quasar fast dispatch, so it cannot
shrink this test).

Run (Quasar sim):
    MESH_DEVICE=<qsr> TT_METAL_SIMULATOR=~/sim/libttsim.so \
        pytest models/experimental/llama32_1b_quasar/tests/debug_ops/test_quasar_sharded_rmsnorm_mcast_fault.py
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
    """WIDTH-sharded ttnn.rms_norm on an 8x4 grid: the RMSNORM layernorm_sharded.cpp path whose in0
    pop-without-push tripped Quasar's tile counters (posted=0 acked=2). Passes on WH/BH and, with the
    kernel fix, on Quasar."""
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
    # fp32_dest_acc_en=False to keep this test separate from the craq-sim bf16->Tf32 unpack gap.
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
