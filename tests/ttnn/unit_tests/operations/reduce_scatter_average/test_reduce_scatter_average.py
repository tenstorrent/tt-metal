# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for the self-contained Python reduce_scatter_average CCL op.

reduce_scatter_average computes the element-wise MEAN of each device's shard
across all N devices on a MeshDevice line and scatters it: device i's output is
the i-th of N equal slices (along dim, Phase-0 dim=3) of the averaged tensor.
Equivalently: reduce_scatter with a SUM reduction scaled by 1/N before landing.
The scaling is part of the op — the caller passes nothing but the tensor.

This file is the immutable spec — the implementer must not modify it.

Verification topology (MUST match the sim's fixed mesh-graph descriptor):
a Wormhole T3K **line mesh of shape (1, 8)** with
``fabric_config = ttnn.FabricConfig.FABRIC_1D``, driven by
``scripts/run_multidevice_sim_pytest.py --op reduce_scatter_average``. A
different mesh shape hangs fabric init ("Fabric Router Sync: Timeout"). The
proven first case is bfloat16, TILE_LAYOUT, dim=3, Linear topology.
"""

from math import prod

import pytest
import torch
from loguru import logger

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

from ttnn.operations.reduce_scatter_average import reduce_scatter_average


# PCC tolerances keyed by dtype. A bf16 sum of N terms plus the 1/N scale
# accumulates rounding, so the bf16 threshold matches this op's golden suite
# (0.99) rather than the generic pure-movement 0.995.
PCC = {
    ttnn.float32: 0.999,
    ttnn.bfloat16: 0.99,
}

# The reduction and scaling are tile computes: TILE_LAYOUT only. bf16 is the
# proven primary dtype; float32 is the secondary supported dtype.
DTYPES = [ttnn.bfloat16, ttnn.float32]

# Per-device shard shapes: single-output-tile, multi-tile, non-square,
# multi-batch. Every device holds a shard of the SAME shape (distinct values).
# Widths are multiples of 256 = 8 devices x tile 32 so the per-device output
# slice (width / N) stays tile-aligned on the (1, 8) mesh.
SHARD_SHAPES = [
    (1, 1, 32, 256),  # single output tile per device (slice 32x32)
    (1, 1, 256, 256),  # multi-tile square (slice 256x32)
    (1, 1, 64, 512),  # non-square (slice 64x64)
    (2, 1, 32, 256),  # multi-batch (slice (2, 1, 32, 32))
]

# Topology <-> fabric_config pairing. The sim is a FABRIC_1D line.
LINEAR = ({"fabric_config": ttnn.FabricConfig.FABRIC_1D}, ttnn.Topology.Linear)

SCATTER_DIM = 3  # Phase-0 scatter dimension (the default)


def _make_sharded_input(mesh_device, shard_shape, dtype):
    """Shard a freshly-seeded full tensor along dim 0 across the whole line.

    The full tensor has shape ``(N * shard_shape[0], *shard_shape[1:])``; each
    device receives exactly ``shard_shape`` (distinct values). Returns the ttnn
    input tensor and the list of N per-device torch oracles: the shards are
    quantized to ``dtype`` first (what the device actually sees), then the mean
    is accumulated in fp32 and sliced along dim 3, so the reference is
    fp32-accumulated and not itself limited by bf16 rounding.
    """
    num_devices = prod(tuple(mesh_device.shape))
    full_shape = (shard_shape[0] * num_devices, *shard_shape[1:])

    torch.manual_seed(42)
    torch_full = torch.randn(full_shape, dtype=torch.float32)
    if dtype == ttnn.bfloat16:
        # Quantize BEFORE the oracle: the mean is over the values the device
        # holds, accumulated in fp32.
        torch_full = torch_full.to(torch.bfloat16)

    shards = torch_full.reshape(num_devices, *shard_shape).to(torch.float32)
    mean = shards.mean(dim=0)  # fp32-accumulated mean of the N shards
    oracle_slices = list(mean.chunk(num_devices, dim=SCATTER_DIM))
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


def _expected_output_shape(shard_shape, num_devices):
    out_shape = list(shard_shape)
    out_shape[SCATTER_DIM] //= num_devices
    return tuple(out_shape)


@pytest.mark.parametrize("device_params, topology", [LINEAR], indirect=["device_params"])
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shard_shape", SHARD_SHAPES)
def test_reduce_scatter_average(mesh_device, topology, dtype, shard_shape):
    """Device i's output equals slice i of the fp32-accumulated mean of all N shards."""
    num_devices = prod(tuple(mesh_device.shape))
    if num_devices < 2:
        pytest.skip("reduce_scatter_average requires at least 2 mesh devices")

    input_tensor, oracle_slices = _make_sharded_input(mesh_device, shard_shape, dtype)

    output_tensor = reduce_scatter_average(input_tensor, topology=topology)
    ttnn.synchronize_device(mesh_device)

    output_shards = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(output_tensor)]
    assert len(output_shards) == num_devices

    expected_shape = _expected_output_shape(shard_shape, num_devices)
    pcc = PCC[dtype]
    for dev_idx, dev_out in enumerate(output_shards):
        assert (
            tuple(dev_out.shape) == expected_shape
        ), f"device {dev_idx} output shape {tuple(dev_out.shape)} != expected slice {expected_shape}"
        assert_with_pcc(oracle_slices[dev_idx], dev_out, pcc)
    logger.info(
        f"reduce_scatter_average {dtype} shard={shard_shape} {topology}: "
        f"all {num_devices} devices hold their slice of the mean"
    )


@pytest.mark.parametrize("device_params, topology", [LINEAR], indirect=["device_params"])
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
def test_reduce_scatter_average_program_cache(mesh_device, topology):
    """Second call (program-cache hit) still averages correctly.

    The op-internal GlobalSemaphores must survive the cache hit (created once,
    not re-created per call), and every consumer must have re-armed its local
    semaphore counter to 0 on the first run — a missing re-arm shows up here as
    a hang or corruption on the second call.
    """
    num_devices = prod(tuple(mesh_device.shape))
    if num_devices < 2:
        pytest.skip("reduce_scatter_average requires at least 2 mesh devices")

    shard_shape = (1, 1, 32, 256)
    for call in range(2):
        input_tensor, oracle_slices = _make_sharded_input(mesh_device, shard_shape, ttnn.bfloat16)
        output_tensor = reduce_scatter_average(input_tensor, topology=topology)
        ttnn.synchronize_device(mesh_device)
        output_shards = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(output_tensor)]
        for dev_idx, dev_out in enumerate(output_shards):
            assert_with_pcc(oracle_slices[dev_idx], dev_out, PCC[ttnn.bfloat16])
        logger.info(f"program-cache call {call}: all {num_devices} devices hold their slice of the mean")


@pytest.mark.parametrize("device_params, topology", [LINEAR], indirect=["device_params"])
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
def test_reduce_scatter_average_output_tensor(mesh_device, topology):
    """The output_tensor path writes into the supplied tensor and returns it."""
    num_devices = prod(tuple(mesh_device.shape))
    if num_devices < 2:
        pytest.skip("reduce_scatter_average requires at least 2 mesh devices")

    shard_shape = (1, 1, 64, 512)
    input_tensor, oracle_slices = _make_sharded_input(mesh_device, shard_shape, ttnn.bfloat16)

    # Pre-allocate a slice-shape output buffer on every device (replicated
    # zeros; the op overwrites every output page).
    out_shape = _expected_output_shape(shard_shape, num_devices)
    preallocated = ttnn.from_torch(
        torch.zeros(out_shape, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    ttnn.synchronize_device(mesh_device)

    returned = reduce_scatter_average(input_tensor, topology=topology, output_tensor=preallocated)
    ttnn.synchronize_device(mesh_device)

    # Same handle is returned.
    assert returned.buffer_address() == preallocated.buffer_address()

    output_shards = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(returned)]
    for dev_idx, dev_out in enumerate(output_shards):
        assert_with_pcc(oracle_slices[dev_idx], dev_out, PCC[ttnn.bfloat16])


@pytest.mark.parametrize("device_params, topology", [LINEAR], indirect=["device_params"])
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
def test_reduce_scatter_average_rejects_unsplittable_width(mesh_device, topology):
    """Shapes whose slice is not tile-aligned are rejected loudly (ValueError).

    shard width 96 on an 8-device line: 96 % 8 == 0 but 96 / 8 = 12 is not a
    multiple of 32 — the op must raise ValueError, not silently pad.
    """
    num_devices = prod(tuple(mesh_device.shape))
    if num_devices < 2:
        pytest.skip("reduce_scatter_average requires at least 2 mesh devices")
    if 96 % num_devices == 0 and (96 // num_devices) % 32 == 0:
        pytest.skip("width 96 is splittable on this mesh; rejection case does not apply")

    shard_shape = (1, 1, 32, 96)  # tile-aligned tensor, unsplittable slice
    input_tensor, _ = _make_sharded_input(mesh_device, shard_shape, ttnn.bfloat16)

    with pytest.raises(ValueError):
        reduce_scatter_average(input_tensor, topology=topology)
