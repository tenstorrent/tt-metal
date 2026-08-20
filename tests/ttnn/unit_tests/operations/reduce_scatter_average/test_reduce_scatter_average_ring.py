# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Refinement 2 — Ring topology (hardware verification suite).

Ring keeps the fused gather + arrival-ordered-reduce program and swaps ONLY the
host-side block-flow table + neighbour wiring: each direction carries its
SHORT-WAY half of the ring (fwd depth N//2, bwd (N-1)//2), the (N-1) <-> 0 hops
cross the wrap link (1-hop routes from ccl_dm_route(.., Ring), probed green
under FABRIC_1D by reduce_scatter/test_ring_fabric_probe.py), and every block
lands EXACTLY once per device. The kernels are topology-agnostic (ring-modular
block indices).

Ring-specific traps exercised here:
  * Wrap-link data path — every Ring test moves real blocks across the
    (N-1) <-> 0 seam in both directions (devices 0 and N-1 are interior on a
    ring: they send AND relay in both directions, unlike the Linear line ends).
  * Exactly-once cover — per-shard power-of-two constants (shard c = 2^c): the
    true mean is exact in bf16 and any duplicated / missing block shifts it by
    a unique, loud amount identifying WHICH block leaked.
  * dim=2 x Ring — the composed cells Refinement 1 left refusing on topology
    alone, incl. the multibatch cursor trap (2,1,256,256) and the
    granule-straddles-channel-run shape (2,1,256,32).
  * Program-cache hit — the R1 semaphore re-arm is doubly load-bearing on a
    ring (every device consumes arrivals in BOTH directions); a missing re-arm
    passes the first call and hangs/corrupts the second.
  * Topology is selected by the kwarg alone — every test runs under the SAME
    device_params (fabric_config = FABRIC_1D) as the Linear suites.

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
# Ring runs under the SAME fabric config as Linear — the op must select ring
# behavior from the topology kwarg alone (op_requirements.md caution b).
RING = ({"fabric_config": ttnn.FabricConfig.FABRIC_1D}, ttnn.Topology.Ring)

# W a multiple of 256 = lcm(tile, 8 devices) keeps the per-device dim=3 slice
# tile-aligned on both a (1, 8) grade mesh and a (1, 4) hardware box.
SHARD_SHAPES_DIM3 = [
    (1, 1, 32, 256),  # single tile row (the smallest slice: slice_Wt = Wt/N)
    (1, 1, 64, 256),  # acceptance-class shape
    (1, 1, 256, 512),  # the worst-case L1 golden class (S = 16 at N = 8)
    (2, 1, 64, 256),  # multibatch
]

# H a multiple of 256 for the dim=2 x Ring composition (Refinement 1's traps
# replayed under the ring flow table).
SHARD_SHAPES_DIM2 = [
    (1, 1, 256, 256),  # the golden square
    (2, 1, 256, 256),  # multibatch cursor trap
    (2, 1, 256, 32),  # g-granule straddles the channel-run boundary
]


def _skip_if_too_few(mesh_device):
    num_devices = prod(tuple(mesh_device.shape))
    if num_devices < 2:
        pytest.skip("requires at least 2 mesh devices")
    return num_devices


def _make_input(mesh_device, shard_shape, dtype, dim=3, torch_shards=None):
    """One SAME-shape shard per device (distinct values) + the fp32-mean oracle."""
    num_devices = prod(tuple(mesh_device.shape))
    full_shape = (shard_shape[0] * num_devices, *shard_shape[1:])
    if torch_shards is None:
        torch.manual_seed(23)
        torch_full = torch.randn(full_shape, dtype=torch.float32)
        if dtype == ttnn.bfloat16:
            torch_full = torch_full.to(torch.bfloat16)
    else:
        torch_full = torch.cat(torch_shards, dim=0)
    shards = torch_full.reshape(num_devices, *shard_shape).to(torch.float32)
    mean = shards.mean(dim=0)
    oracle = list(mean.chunk(num_devices, dim=dim))
    input_tensor = ttnn.from_torch(
        torch_full,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )
    ttnn.synchronize_device(mesh_device)
    return input_tensor, oracle


def _check(output_tensor, oracle, dtype):
    for dev_idx, t in enumerate(ttnn.get_device_tensors(output_tensor)):
        assert_with_pcc(oracle[dev_idx], ttnn.to_torch(t), PCC[dtype])


@pytest.mark.parametrize("device_params, topology", [RING], indirect=["device_params"])
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shard_shape", SHARD_SHAPES_DIM3)
def test_ring_dim3(mesh_device, topology, dtype, shard_shape):
    """Ring correctness at dim=3 across shapes and both dtypes."""
    _skip_if_too_few(mesh_device)
    input_tensor, oracle = _make_input(mesh_device, shard_shape, dtype, dim=3)
    output_tensor = reduce_scatter_average(input_tensor, dim=3, topology=topology)
    ttnn.synchronize_device(mesh_device)
    _check(output_tensor, oracle, dtype)


@pytest.mark.parametrize("device_params, topology", [RING], indirect=["device_params"])
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shard_shape", SHARD_SHAPES_DIM2)
def test_ring_dim2(mesh_device, topology, dtype, shard_shape):
    """dim=2 x Ring — the composed cells that refused on topology alone after
    Refinement 1, incl. the multibatch cursor and granule-straddle traps."""
    _skip_if_too_few(mesh_device)
    input_tensor, oracle = _make_input(mesh_device, shard_shape, dtype, dim=2)
    output_tensor = reduce_scatter_average(input_tensor, dim=2, topology=topology)
    ttnn.synchronize_device(mesh_device)
    _check(output_tensor, oracle, dtype)


@pytest.mark.parametrize("device_params, topology", [RING], indirect=["device_params"])
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("dim, alias", [(3, -1), (2, -2)])
def test_ring_negative_dim_alias(mesh_device, topology, dim, alias):
    """Negative dim aliases canonicalize before the SUPPORTED test under Ring too."""
    _skip_if_too_few(mesh_device)
    shard_shape = (1, 1, 256, 256)
    input_tensor, oracle = _make_input(mesh_device, shard_shape, ttnn.bfloat16, dim=dim)
    output_tensor = reduce_scatter_average(input_tensor, dim=alias, topology=topology)
    ttnn.synchronize_device(mesh_device)
    _check(output_tensor, oracle, ttnn.bfloat16)


@pytest.mark.parametrize("device_params, topology", [RING], indirect=["device_params"])
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_ring_exactly_once_deterministic(mesh_device, topology):
    """Exactly-once block cover, hand-calculable: shard c is the constant 2^c, so
    the true mean is (2^N - 1) / N — every running partial sum is a small integer,
    EXACT in bf16 (values <= 255 at N = 8), and the 1/N scale is exact for
    power-of-2 N. Any block delivered twice or missed shifts the mean by
    2^c / N — a unique, loud signature per block c (>= 0.25 at N = 4), while a
    correct ring split reproduces the oracle EXACTLY (atol 0 modulo fp noise)."""
    num_devices = _skip_if_too_few(mesh_device)
    if num_devices & (num_devices - 1):
        pytest.skip("exactness argument needs power-of-2 N (1/N exact in bf16)")
    shard_shape = (1, 1, 32, 32 * num_devices)
    torch_shards = [torch.full(shard_shape, float(2**c), dtype=torch.float32) for c in range(num_devices)]
    input_tensor, oracle = _make_input(mesh_device, shard_shape, ttnn.bfloat16, dim=3, torch_shards=torch_shards)
    output_tensor = reduce_scatter_average(input_tensor, dim=3, topology=topology)
    ttnn.synchronize_device(mesh_device)
    expected = (2**num_devices - 1) / num_devices
    for dev_idx, t in enumerate(ttnn.get_device_tensors(output_tensor)):
        got = ttnn.to_torch(t).to(torch.float32)
        max_err = (got - expected).abs().max().item()
        assert max_err < 1e-6, (
            f"device {dev_idx}: expected exact mean {expected}, max err {max_err} — a block "
            f"was duplicated or missed (2^c/N signature identifies which)"
        )
        assert torch.allclose(got, oracle[dev_idx], atol=1e-6)
    logger.info(f"ring exactly-once cover OK: all devices at exact mean {expected}")


@pytest.mark.parametrize("device_params, topology", [RING], indirect=["device_params"])
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_ring_program_cache(mesh_device, topology):
    """Second identical Ring call (program-cache hit) still averages correctly —
    the R1 semaphore re-arm is doubly load-bearing on a ring (every device
    consumes arrivals in both directions; a missing re-arm hangs call 2)."""
    _skip_if_too_few(mesh_device)
    shard_shape = (1, 1, 64, 256)
    for call in range(2):
        input_tensor, oracle = _make_input(mesh_device, shard_shape, ttnn.bfloat16, dim=3)
        output_tensor = reduce_scatter_average(input_tensor, topology=topology)
        ttnn.synchronize_device(mesh_device)
        _check(output_tensor, oracle, ttnn.bfloat16)
        logger.info(f"ring program-cache call {call}: OK")


@pytest.mark.parametrize("device_params, topology", [RING], indirect=["device_params"])
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_ring_output_tensor(mesh_device, topology):
    """The output_tensor path writes into the supplied tensor under Ring."""
    num_devices = _skip_if_too_few(mesh_device)
    shard_shape = (1, 1, 64, 256)
    input_tensor, oracle = _make_input(mesh_device, shard_shape, ttnn.float32, dim=3)
    out_shape = list(shard_shape)
    out_shape[3] //= num_devices
    preallocated = ttnn.from_torch(
        torch.zeros(out_shape, dtype=torch.float32),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    ttnn.synchronize_device(mesh_device)
    returned = reduce_scatter_average(input_tensor, topology=topology, output_tensor=preallocated)
    ttnn.synchronize_device(mesh_device)
    assert returned.buffer_address() == preallocated.buffer_address()
    _check(returned, oracle, ttnn.float32)


@pytest.mark.parametrize("device_params, topology", [RING], indirect=["device_params"])
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_ring_matches_linear(mesh_device, topology):
    """Ring and Linear compute the SAME mean on the same input — they differ only
    in arrival order, so per-element differences are bounded by the documented
    bf16 running-partial-sum rounding (Phase-0 class), far below signal."""
    _skip_if_too_few(mesh_device)
    shard_shape = (1, 1, 64, 256)
    input_tensor, oracle = _make_input(mesh_device, shard_shape, ttnn.bfloat16, dim=3)
    out_ring = reduce_scatter_average(input_tensor, topology=ttnn.Topology.Ring)
    ttnn.synchronize_device(mesh_device)
    out_linear = reduce_scatter_average(input_tensor, topology=ttnn.Topology.Linear)
    ttnn.synchronize_device(mesh_device)
    ring_shards = [ttnn.to_torch(t).to(torch.float32) for t in ttnn.get_device_tensors(out_ring)]
    linear_shards = [ttnn.to_torch(t).to(torch.float32) for t in ttnn.get_device_tensors(out_linear)]
    for dev_idx, (r, l) in enumerate(zip(ring_shards, linear_shards)):
        assert_with_pcc(l, r, 0.999)
    _check(out_ring, oracle, ttnn.bfloat16)
