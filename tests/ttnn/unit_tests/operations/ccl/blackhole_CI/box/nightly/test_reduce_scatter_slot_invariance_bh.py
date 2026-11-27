# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Regression guard: Ring ttnn.reduce_scatter must give identical batched-prefill users identical
results, whatever their slot.

The ring kernels choose each chunk's ring direction, and so the order in which the per-device
partials are added, from the chunk's tile index inside a (batch, channel) unit. If 32 users are
flattened into one [1, 1, 32 * 128, W] unit, users at different row offsets start at different
chunk positions and can round differently. The llama3_70b_galaxy TT_CCL prefill reduce-scatter
therefore gives each user its own unit, [1, 32, 128, W]. This test asserts that this layout is
slot-invariant and matches the torch sum, and logs how many users differ in the flat layout.

Uses bh_1d_mesh_device (auto-sizes to the BH box as a 1D ring). Summation order matters only with
three or more partials, so the test needs at least three chips:
  pytest .../blackhole_CI/box/nightly/test_reduce_scatter_slot_invariance_bh.py -s
"""

import pytest
import torch
from loguru import logger

import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc

NUM_USERS = 32
USER_ROWS = 128
DIM = 3


def _reduce_scatter(mesh_device, global_input, dtype, num_links, sub_device_id, sub_stall_group):
    """Reduce-scatter global_input [1, B, S, W * num_devices] on DIM the way TT_CCL prefill calls it.

    Shards DIM across the chips, so each chip holds a [1, B, S, W] partial; the reduce-scatter sums
    the partials and scatters DIM. Returns the composed [1, B, S, W] output.
    """
    mesh_shape = tuple(mesh_device.shape)
    shard_axis = 0 if mesh_shape[0] > 1 else 1
    placements = [ttnn.PlacementReplicate(), ttnn.PlacementReplicate()]
    placements[shard_axis] = ttnn.PlacementShard(DIM)

    input_mesh = ttnn.from_torch(
        global_input,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.create_mesh_mapper(
            mesh_device,
            ttnn.MeshMapperConfig(placements, ttnn.MeshShape(*mesh_shape)),
        ),
    )
    out = ttnn.reduce_scatter(
        input_mesh,
        DIM,
        cluster_axis=shard_axis,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        topology=ttnn.Topology.Ring,
        num_links=num_links,
        num_workers_per_link=1,
        subdevice_id=sub_device_id,
    )
    ttnn.synchronize_device(mesh_device, sub_device_ids=sub_stall_group)
    out_torch = ttnn.to_torch(ttnn.from_device(out), mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=DIM))
    out.deallocate(True)
    input_mesh.deallocate(True)
    return out_torch.float()


def _users_differing_from_user_0(out_per_user):
    """out_per_user is [1, NUM_USERS, USER_ROWS, W]. Returns the users whose rows differ from user 0."""
    return [u for u in range(1, NUM_USERS) if not torch.equal(out_per_user[:, u], out_per_user[:, 0])]


# slice_wt is the output width per chip in tiles. The flat layout keeps users aligned to chunk
# boundaries only for some widths, so cover several. 5 and 25 are the Qwen3-32B FF2 and FF1/FF3
# widths on Galaxy.
@pytest.mark.parametrize("slice_wt", [1, 3, 5, 20, 25], ids=lambda wt: f"slice_wt{wt}")
@pytest.mark.parametrize("num_links", [1, 2], ids=["1link", "2link"])
@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat16], ids=["bfloat8_b", "bfloat16"])
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
            "trace_region_size": 500_000,
        }
    ],
    ids=["fabric_1D_ring"],
    indirect=True,
)
def test_reduce_scatter_slot_invariance(bh_1d_mesh_device, dtype, num_links, slice_wt):
    mesh_device = bh_1d_mesh_device
    num_devices = mesh_device.get_num_devices()
    if num_devices < 3:
        pytest.skip(f"summation order needs >=3 chips to matter, got {num_devices}")

    grid = mesh_device.compute_with_storage_grid_size()
    ccl_crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    worker_sub_device = ttnn.SubDevice([ccl_crs])
    sub_device_id = ttnn.SubDeviceId(0)
    sub_stall_group = [sub_device_id]
    sub_device_manager = mesh_device.create_sub_device_manager([worker_sub_device], 0)
    mesh_device.load_sub_device_manager(sub_device_manager)
    mesh_device.set_sub_device_stall_group(sub_stall_group)

    try:
        per_device_n = slice_wt * ttnn.TILE_SIZE * num_devices
        torch.manual_seed(0)
        user = torch.rand((1, 1, USER_ROWS, per_device_n * num_devices)).bfloat16().float()
        per_user_input = user.repeat(1, NUM_USERS, 1, 1)
        flat_input = per_user_input.reshape(1, 1, NUM_USERS * USER_ROWS, -1)

        out_flat = _reduce_scatter(mesh_device, flat_input, dtype, num_links, sub_device_id, sub_stall_group)
        flat_differing = _users_differing_from_user_0(out_flat.reshape(1, NUM_USERS, USER_ROWS, -1))
        logger.info(
            f"[{dtype} {num_links}link slice_wt={slice_wt}] flat [1, 1, {NUM_USERS * USER_ROWS}, W]: "
            f"{len(flat_differing)}/{NUM_USERS - 1} users differ from user 0: {flat_differing}"
        )

        out_per_user = _reduce_scatter(mesh_device, per_user_input, dtype, num_links, sub_device_id, sub_stall_group)
        per_user_differing = _users_differing_from_user_0(out_per_user)
        logger.info(
            f"[{dtype} {num_links}link slice_wt={slice_wt}] per-user [1, {NUM_USERS}, {USER_ROWS}, W]: "
            f"{len(per_user_differing)}/{NUM_USERS - 1} users differ from user 0: {per_user_differing}"
        )

        assert not per_user_differing, (
            f"Ring reduce_scatter is slot-dependent with one unit per user (dtype={dtype}, num_links={num_links}, "
            f"slice_wt={slice_wt}): users {per_user_differing} differ from user 0 on identical input"
        )

        # Every user equals user 0 (asserted above), so checking user 0 covers all of them.
        ref = torch.stack(torch.chunk(user, num_devices, DIM)).sum(0)
        passed, pcc = comp_pcc(ref, out_per_user[:, :1], 0.999)
        logger.info(f"[{dtype} {num_links}link slice_wt={slice_wt}] per-user PCC vs torch: {pcc}")
        assert passed, f"Ring reduce_scatter per-user output does not match the torch sum: {pcc}"
    finally:
        mesh_device.reset_sub_device_stall_group()
        mesh_device.clear_loaded_sub_device_manager()
