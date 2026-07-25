# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Extended (verifier-authored) coverage for the all_reduce CCL op.

Two gaps the acceptance suite cannot reach, both found during verification:

1. **Back-to-back calls with NO intervening host sync** — `op_design.md` Risk 5
   documents a cross-call window: the receive counter is reset by the receiver
   *after* its `wait_min(sem, N-1)`, so a peer's NEXT call's increment could in
   principle be wiped by this device's reset. The acceptance/program-cache tests
   both `synchronize_device` between calls, which closes the window by
   construction — this test deliberately does not.

2. **Odd N (the `seeded` fold branch)** — the compute kernel's
   `if constexpr (num_devices % 2 == 1)` seed-with-`copy_tile` path is DEAD CODE
   on the `(1, 8)` verification mesh (N is always even). A `(1, 3)` submesh view
   of the same 8-chip line exercises it, plus a shorter multicast range on both
   fabric directions.

MULTI-DEVICE: drive through the deterministic craq-sim runner; the parent mesh
MUST be `(1, 8)` + `FABRIC_1D` or fabric init hangs.

    scripts/run_multidevice_sim_pytest.py --op all_reduce -- \
        tests/ttnn/unit_tests/operations/all_reduce/test_all_reduce_extended.py -v
"""

from math import prod

import pytest
import torch
from loguru import logger

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

from ttnn.operations.all_reduce import all_reduce

PCC_BF16 = 0.99

LINEAR = ({"fabric_config": ttnn.FabricConfig.FABRIC_1D}, ttnn.Topology.Linear)

pytestmark = [
    pytest.mark.parametrize("device_params, topology", [LINEAR], indirect=["device_params"]),
    pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True),
]


def _shard_and_oracle(mesh_device, shard_shape, seed, dtype=ttnn.bfloat16, memory_config=None):
    """Build a mesh-sharded input + the fp32-accumulated element-wise SUM oracle.

    Deliberately does NOT `synchronize_device` — the caller controls where syncs
    land (test 1 needs both inputs staged before either op is dispatched).
    """
    num_devices = prod(tuple(mesh_device.shape))
    full_shape = (shard_shape[0] * num_devices, *shard_shape[1:])
    torch.manual_seed(seed)
    torch_full = torch.randn(full_shape, dtype=torch.float32)
    oracle = torch_full.reshape(num_devices, *shard_shape).sum(dim=0)
    if dtype == ttnn.bfloat16:
        torch_full = torch_full.to(torch.bfloat16)
        oracle = oracle.to(torch.bfloat16)
    tensor = ttnn.from_torch(
        torch_full,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=memory_config or ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )
    return tensor, oracle


def test_all_reduce_back_to_back_no_sync(mesh_device, topology):
    """Two all_reduce dispatches with NO host sync between them (design Risk 5).

    Both inputs are staged first, so nothing between the two `all_reduce` calls
    forces device completion. A lost semaphore increment shows up as a HANG; a
    torn gathered buffer shows up as a PCC miss on the second result.
    """
    num_devices = prod(tuple(mesh_device.shape))
    if num_devices < 2:
        pytest.skip("all_reduce requires at least 2 mesh devices")

    shard_shape = (1, 1, 32, 64)
    in_a, oracle_a = _shard_and_oracle(mesh_device, shard_shape, seed=0)
    in_b, oracle_b = _shard_and_oracle(mesh_device, shard_shape, seed=1)
    ttnn.synchronize_device(mesh_device)  # inputs staged; the window under test starts here

    out_a = all_reduce(in_a, topology=topology)
    out_b = all_reduce(in_b, topology=topology)  # <-- no sync in between, on purpose
    ttnn.synchronize_device(mesh_device)

    for label, out, oracle in (("call0", out_a, oracle_a), ("call1", out_b, oracle_b)):
        for dev_idx, dev_out in enumerate(ttnn.get_device_tensors(out)):
            assert tuple(dev_out.shape) == tuple(shard_shape)
            assert_with_pcc(oracle, ttnn.to_torch(dev_out), PCC_BF16)
        logger.info(f"back-to-back {label}: all {num_devices} devices hold the element-wise sum")


def test_all_reduce_l1_interleaved(mesh_device, topology):
    """L1-interleaved input/output (`validate()` allows any interleaved buffer type).

    The op-internal landing buffer inherits `input_tensor.memory_config()`, so an
    L1 input also puts an `N x shard` gathered buffer in L1 — keep the shape small.
    Only DRAM is exercised by the acceptance/golden suites, so this is the cell
    that keeps the "interleaved DRAM **or L1**" claim honest.
    """
    if prod(tuple(mesh_device.shape)) < 2:
        pytest.skip("all_reduce requires at least 2 mesh devices")

    shard_shape = (1, 1, 32, 32)  # 1 tile/shard -> 8 tiles (16 kB) of L1 for the landing buffer
    in_t, oracle = _shard_and_oracle(mesh_device, shard_shape, seed=3, memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn.synchronize_device(mesh_device)

    out = all_reduce(in_t, topology=topology)
    ttnn.synchronize_device(mesh_device)

    assert out.memory_config().buffer_type == in_t.memory_config().buffer_type
    for dev_out in ttnn.get_device_tensors(out):
        assert tuple(dev_out.shape) == tuple(shard_shape)
        assert_with_pcc(oracle, ttnn.to_torch(dev_out), PCC_BF16)
    logger.info("L1-interleaved all_reduce: every device holds the element-wise sum")


@pytest.mark.parametrize("line_len", [3], ids=["N3_odd"])
def test_all_reduce_odd_line_submesh(mesh_device, topology, line_len):
    """Odd N on a `(1, line_len)` submesh — exercises the compute kernel's
    `seeded` (copy_tile + all-pairs-accumulate) fold branch, which the even-N
    verification mesh never reaches, and a shorter multicast range per direction.
    """
    if prod(tuple(mesh_device.shape)) < line_len:
        pytest.skip(f"parent mesh smaller than the requested {line_len}-device line")

    submesh = mesh_device.create_submesh(ttnn.MeshShape(1, line_len), ttnn.MeshCoordinate(0, 0))
    assert prod(tuple(submesh.shape)) == line_len

    shard_shape = (1, 1, 32, 64)
    in_t, oracle = _shard_and_oracle(submesh, shard_shape, seed=7)
    ttnn.synchronize_device(submesh)

    out = all_reduce(in_t, topology=topology)
    ttnn.synchronize_device(submesh)

    for dev_idx, dev_out in enumerate(ttnn.get_device_tensors(out)):
        assert tuple(dev_out.shape) == tuple(shard_shape)
        assert_with_pcc(oracle, ttnn.to_torch(dev_out), PCC_BF16)
    logger.info(f"odd-N submesh N={line_len}: all devices hold the element-wise sum (seeded fold path)")
