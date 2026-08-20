# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Refinement 1 — dim=2 scatter (hardware verification suite).

The dim=2 slice is a contiguous tile-ROW block per (batch, channel): the reduce
reader walks B*C dense runs of slice_Ht*Wt tiles (SliceRowWalker degenerating to
walk_slice_Wt = Wt), base from sched::slice_tile_offset(dim=2, ...) and a
bump_base(Ht*Wt) hop between channel blocks tracked PER TILE — the run boundary
need not align with the g-granule boundary.

Deliberate traps exercised here:
  * (2, 1, 256, 256)  — the multibatch golden INPUT: a walker cursor hoisted out
    of the (batch, channel) loop silently walks the wrong slice from batch 2 on.
  * (2, 1, 256, 32)   — g-granule STRADDLES the channel-run boundary (N=4:
    run = 2 tiles, g = 4 → one granule covers two runs; N=8: run = 1 tile,
    g = 2), so the per-tile boundary tracking is load-bearing.
  * (1, 1, 256, 96)   — W is NOT N-divisible: dim=2 must not require a
    splittable width.
  * dim=-2 alias, program-cache hit (R1 re-arm under the dim=2 program variant),
    and hand-calculable row-index constants (a wrong slice base shifts values by
    >= 32 — far above any datapath noise).

Mesh-shape-adaptive via CCL_HW_MESH_SHAPE (default (1, 4)). Drive via:

    scripts/run_multidevice_sim_pytest.py --runtime hardware \
        --op reduce_scatter_average -- <this file>
"""

import os
from math import prod

import pytest
import torch
from loguru import logger

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

from ttnn.operations.reduce_scatter_average import reduce_scatter_average

MESH_SHAPE = tuple(int(x) for x in os.environ.get("CCL_HW_MESH_SHAPE", "1,4").split(","))

PCC = {ttnn.float32: 0.999, ttnn.bfloat16: 0.99}
DTYPES = [ttnn.bfloat16, ttnn.float32]
LINEAR = ({"fabric_config": ttnn.FabricConfig.FABRIC_1D}, ttnn.Topology.Linear)

# H is a multiple of 256 = lcm(tile, 8 devices), so the per-device dim=2 slice
# stays tile-aligned on both a (1, 8) grade mesh and a (1, 4) hardware box.
SHARD_SHAPES_DIM2 = [
    (1, 1, 256, 32),  # single tile column (Wt = 1)
    (1, 1, 256, 256),  # the golden square
    (1, 1, 256, 96),  # W not N-divisible — legal for dim=2
    (2, 1, 256, 256),  # multibatch golden INPUT: per-(batch, channel) restart trap
    (2, 1, 256, 32),  # g-granule straddles the channel-run boundary
]


def _make_sharded_input(mesh_device, shard_shape, dtype, dim, torch_full=None):
    """Shard a full tensor along dim 0; return (ttnn input, per-device oracles).

    Oracle: shards quantized to dtype first, mean accumulated in fp32, sliced on
    the SCATTER dim — same construction as the acceptance/debug suites."""
    num_devices = prod(tuple(mesh_device.shape))
    full_shape = (shard_shape[0] * num_devices, *shard_shape[1:])

    if torch_full is None:
        torch.manual_seed(1234)
        torch_full = torch.randn(full_shape, dtype=torch.float32)
    assert tuple(torch_full.shape) == full_shape
    if dtype == ttnn.bfloat16:
        torch_full = torch_full.to(torch.bfloat16)

    shards = torch_full.reshape(num_devices, *shard_shape).to(torch.float32)
    mean = shards.mean(dim=0)
    oracle_slices = list(mean.chunk(num_devices, dim=dim))
    if dtype == ttnn.bfloat16:
        oracle_slices = [s.to(torch.bfloat16) for s in oracle_slices]

    input_tensor = ttnn.from_torch(
        torch_full,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )
    ttnn.synchronize_device(mesh_device)
    return input_tensor, oracle_slices


@pytest.mark.parametrize("device_params, topology", [LINEAR], indirect=["device_params"])
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shard_shape", SHARD_SHAPES_DIM2)
def test_dim2_hw(mesh_device, topology, dtype, shard_shape):
    """Device i's output equals H-slice i of the fp32-accumulated mean."""
    num_devices = prod(tuple(mesh_device.shape))
    if num_devices < 2:
        pytest.skip("requires at least 2 mesh devices")

    input_tensor, oracle_slices = _make_sharded_input(mesh_device, shard_shape, dtype, dim=2)

    output_tensor = reduce_scatter_average(input_tensor, dim=2, topology=topology)
    ttnn.synchronize_device(mesh_device)

    output_shards = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(output_tensor)]
    assert len(output_shards) == num_devices

    expected_shape = list(shard_shape)
    expected_shape[2] //= num_devices
    for dev_idx, dev_out in enumerate(output_shards):
        assert tuple(dev_out.shape) == tuple(expected_shape)
        assert_with_pcc(oracle_slices[dev_idx], dev_out, PCC[dtype])
    logger.info(f"dim=2 {dtype} shard={shard_shape}: all {num_devices} devices OK")


@pytest.mark.parametrize("device_params, topology", [LINEAR], indirect=["device_params"])
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_dim2_negative_alias(mesh_device, topology):
    """dim=-2 canonicalizes to 2 before the SUPPORTED membership test."""
    num_devices = prod(tuple(mesh_device.shape))
    if num_devices < 2:
        pytest.skip("requires at least 2 mesh devices")

    input_tensor, oracle_slices = _make_sharded_input(mesh_device, (1, 1, 256, 64), ttnn.bfloat16, dim=2)
    output_tensor = reduce_scatter_average(input_tensor, dim=-2, topology=topology)
    ttnn.synchronize_device(mesh_device)
    for dev_idx, t in enumerate(ttnn.get_device_tensors(output_tensor)):
        assert_with_pcc(oracle_slices[dev_idx], ttnn.to_torch(t), PCC[ttnn.bfloat16])


@pytest.mark.parametrize("device_params, topology", [LINEAR], indirect=["device_params"])
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_dim2_program_cache(mesh_device, topology):
    """Second dim=2 call (program-cache hit) still averages correctly — catches a
    missing semaphore re-arm (R1) under the dim=2 program variant."""
    num_devices = prod(tuple(mesh_device.shape))
    if num_devices < 2:
        pytest.skip("requires at least 2 mesh devices")

    shard_shape = (1, 1, 256, 64)
    for call in range(2):
        input_tensor, oracle_slices = _make_sharded_input(mesh_device, shard_shape, ttnn.bfloat16, dim=2)
        output_tensor = reduce_scatter_average(input_tensor, dim=2, topology=topology)
        ttnn.synchronize_device(mesh_device)
        for dev_idx, t in enumerate(ttnn.get_device_tensors(output_tensor)):
            assert_with_pcc(oracle_slices[dev_idx], ttnn.to_torch(t), PCC[ttnn.bfloat16])
        logger.info(f"dim=2 program-cache call {call}: OK")


@pytest.mark.parametrize("device_params, topology", [LINEAR], indirect=["device_params"])
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_dim2_row_index_deterministic(mesh_device, topology):
    """Every shard's H-row r carries the constant r, IDENTICAL across shards; the
    true mean is exactly the row index. Device i's output must be rows
    [i*sliceH, (i+1)*sliceH) — a wrong slice base shows as a constant offset >= 32.
    Tolerance note: the op's RUNNING partial sums round through the bf16
    accumulator CB between passes (e.g. 3*93 = 279 -> 280 at N=4 — measured max
    0.5 on hardware; bound ~ N * ulp(N*255) / N <= ~6 at N=8), so atol=10 cleanly
    separates rounding (<10) from a slice-walk bug (>=32)."""
    num_devices = prod(tuple(mesh_device.shape))
    if num_devices < 2:
        pytest.skip("requires at least 2 mesh devices")
    if num_devices & (num_devices - 1):
        pytest.skip("clean-tolerance argument needs power-of-2 N (1/N exact in bf16)")

    H = 32 * num_devices  # row values 0..H-1 <= 255: exact in bf16
    shard_shape = (1, 1, H, 32)
    row_vals = torch.arange(H, dtype=torch.float32).view(1, 1, H, 1).expand(1, 1, H, 32)
    full = row_vals.expand(num_devices, 1, H, 32).contiguous()
    input_tensor, _ = _make_sharded_input(mesh_device, shard_shape, ttnn.bfloat16, dim=2, torch_full=full)

    output_tensor = reduce_scatter_average(input_tensor, dim=2, topology=topology)
    ttnn.synchronize_device(mesh_device)

    for dev_idx, dev_out in enumerate(ttnn.to_torch(t) for t in ttnn.get_device_tensors(output_tensor)):
        expected = row_vals[:, :, dev_idx * 32 : (dev_idx + 1) * 32, :].to(torch.bfloat16).float()
        assert torch.allclose(
            dev_out.float(), expected, rtol=0, atol=10.0
        ), f"device {dev_idx}: max diff {(dev_out.float() - expected).abs().max()} (wrong slice base?)"


@pytest.mark.parametrize("device_params, topology", [LINEAR], indirect=["device_params"])
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_dim2_multibatch_deterministic(mesh_device, topology):
    """Multibatch determinism: value = batch*512 + row, identical across shards
    (fp32 — exact integers). Distinguishes THE trap classes: a per-batch walker
    restart bug mixes batches (error ~512); a slice-base bug shifts rows (error
    >= 32). fp32 datapath noise (FPU srcA/srcB truncation, ~1e-3 relative on
    partial sums up to ~6k) stays well under atol=16 < 32."""
    num_devices = prod(tuple(mesh_device.shape))
    if num_devices < 2:
        pytest.skip("requires at least 2 mesh devices")

    H = 32 * num_devices
    B = 2
    shard_shape = (B, 1, H, 32)
    rows = torch.arange(H, dtype=torch.float32).view(1, 1, H, 1)
    batches = 512.0 * torch.arange(B, dtype=torch.float32).view(B, 1, 1, 1)
    shard = (rows + batches).expand(B, 1, H, 32)
    full = shard.repeat(num_devices, 1, 1, 1).contiguous()
    input_tensor, _ = _make_sharded_input(mesh_device, shard_shape, ttnn.float32, dim=2, torch_full=full)

    output_tensor = reduce_scatter_average(input_tensor, dim=2, topology=topology)
    ttnn.synchronize_device(mesh_device)

    for dev_idx, dev_out in enumerate(ttnn.to_torch(t) for t in ttnn.get_device_tensors(output_tensor)):
        expected = (rows[:, :, dev_idx * 32 : (dev_idx + 1) * 32, :] + batches).expand(B, 1, 32, 32)
        assert torch.allclose(
            dev_out.float(), expected, rtol=0, atol=16.0
        ), f"device {dev_idx}: max diff {(dev_out.float() - expected).abs().max()} (batch mix or wrong slice?)"


@pytest.mark.parametrize("device_params, topology", [LINEAR], indirect=["device_params"])
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_dim3_unchanged_after_refinement(mesh_device, topology):
    """Non-regression spot check: the dim-aware reader keeps the Phase-0 dim=3
    behavior bit-for-bit (stride-0 boundary fire is a no-op)."""
    num_devices = prod(tuple(mesh_device.shape))
    if num_devices < 2:
        pytest.skip("requires at least 2 mesh devices")

    shard_shape = (2, 1, 64, 256)
    num = prod(tuple(mesh_device.shape))
    full_shape = (shard_shape[0] * num, *shard_shape[1:])
    torch.manual_seed(99)
    full = torch.randn(full_shape, dtype=torch.float32)
    input_tensor, oracle_slices = _make_sharded_input(mesh_device, shard_shape, ttnn.bfloat16, dim=3, torch_full=full)

    output_tensor = reduce_scatter_average(input_tensor, dim=3, topology=topology)
    ttnn.synchronize_device(mesh_device)
    for dev_idx, t in enumerate(ttnn.get_device_tensors(output_tensor)):
        assert_with_pcc(oracle_slices[dev_idx], ttnn.to_torch(t), PCC[ttnn.bfloat16])
