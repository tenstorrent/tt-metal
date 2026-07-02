# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Extended device coverage for the refined all_gather axes (verifier-added).

The immutable acceptance spec (``test_all_gather.py``) only exercises the Phase-0
primary case: ``gather_dim=0``, ``TILE``, ``Linear``. Refinements 2 (strided
concat addressing for ``gather_dim != 0``) and 3 (``Topology.Ring``) added those
axes to ``SUPPORTED`` and were validated on the golden suite — but the golden
suite lives behind the eval harness. This file is a small, harness-independent
device regression guard for the refined axes, so a future kernel change that
breaks strided addressing or the Ring path is caught by the op's own unit tests.

Keep it small: a handful of cells across the newly-refined axes (gather_dim ∈
{-1, -2}, Ring, bf8b), NOT an exhaustive matrix — the golden suite owns the
cartesian. Oracle is identity: every device holds ``torch.cat`` of all N shards
along ``gather_dim``.

Run on the deterministic WH sim (mesh (1, 8) + FABRIC_1D MUST match the sim's
mesh-graph descriptor, else fabric init hangs):

    scripts/run_multidevice_sim_pytest.py --op all_gather -- \
        tests/ttnn/unit_tests/operations/all_gather/test_all_gather_extended.py -v
"""

from math import prod

import pytest
import torch
from loguru import logger

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

from ttnn.operations.all_gather import all_gather


PCC = {
    ttnn.float32: 0.9999,
    ttnn.bfloat16: 0.999,
    ttnn.bfloat8_b: 0.99,
}

LINEAR = ({"fabric_config": ttnn.FabricConfig.FABRIC_1D}, ttnn.Topology.Linear)
RING = ({"fabric_config": ttnn.FabricConfig.FABRIC_1D}, ttnn.Topology.Ring)


def _torch_dtype(dtype):
    return torch.float32 if dtype == ttnn.float32 else torch.bfloat16


def _run(mesh_device, shard_shape, dtype, layout, gather_dim, topology):
    """Shard a full tensor along `gather_dim`, gather, assert every device holds
    the full concat (== torch.cat of the shards along the same axis)."""
    num_devices = prod(tuple(mesh_device.shape))
    if num_devices < 2:
        pytest.skip("all_gather requires at least 2 mesh devices")

    rank = len(shard_shape)
    # Canonicalize to the op's negative convention, then to a positive torch axis.
    gd = gather_dim if gather_dim < 0 else gather_dim - rank
    shard_axis = gd % rank

    full_shape = list(shard_shape)
    full_shape[shard_axis] *= num_devices

    torch.manual_seed(42)
    torch_full = torch.randn(tuple(full_shape), dtype=torch.float32).to(_torch_dtype(dtype))

    input_tensor = ttnn.from_torch(
        torch_full,
        dtype=dtype,
        layout=layout,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=shard_axis),
    )
    ttnn.synchronize_device(mesh_device)

    output_tensor = all_gather(input_tensor, gather_dim, topology=topology)
    ttnn.synchronize_device(mesh_device)

    output_shards = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(output_tensor)]
    for dev_idx, dev_out in enumerate(output_shards):
        assert tuple(dev_out.shape) == tuple(
            torch_full.shape
        ), f"device {dev_idx} output shape {tuple(dev_out.shape)} != full {tuple(torch_full.shape)}"
        assert_with_pcc(torch_full, dev_out, PCC[dtype])
    logger.info(
        f"all_gather {dtype} {layout} shard={shard_shape} gd={gather_dim} {topology}: "
        f"all {num_devices} devices hold the full concat"
    )


# Refined axes (R2 strided gather_dim, R3 Ring, R1 bf8b). Each cell holds every
# other axis at a proven value so it isolates ONE refined axis. All are inside
# SUPPORTED (the two EXCLUSIONS — TILE gd=-2 non-aligned, RM gd=-1 — are covered
# by the golden xfail, not duplicated here).
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params, topology, shard_shape, dtype, layout, gather_dim",
    [
        # R2: gather_dim=-1 (inner, whole-tile strided concat), TILE, Linear.
        (*LINEAR, (1, 1, 32, 64), ttnn.bfloat16, ttnn.TILE_LAYOUT, -1),
        # R2: gather_dim=-2 (H page-grid axis), ROW_MAJOR, Linear.
        (*LINEAR, (1, 1, 32, 64), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, -2),
        # R2 + R3: gather_dim=-1, TILE, Ring (topology-agnostic adjacent-hop path).
        (*RING, (1, 1, 32, 64), ttnn.bfloat16, ttnn.TILE_LAYOUT, -1),
        # R1: bfloat8_b, gather_dim=0 (page-contiguous), TILE, Linear.
        (*LINEAR, (1, 1, 64, 128), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, 0),
    ],
    indirect=["device_params"],
    ids=["gd-1_tile_linear", "gd-2_rm_linear", "gd-1_tile_ring", "bf8b_gd0_tile_linear"],
)
def test_all_gather_refined_axes(mesh_device, topology, shard_shape, dtype, layout, gather_dim):
    _run(mesh_device, shard_shape, dtype, layout, gather_dim, topology)
