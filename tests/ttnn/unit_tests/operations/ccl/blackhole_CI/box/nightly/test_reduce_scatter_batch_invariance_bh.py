# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Regression guard: reduce_scatter_minimal_async (Ring) must be BATCH-INVARIANT (tt-metal#47238,
tt-inference-server#4004).

Feeds bit-identical input rows at M=1024 and M=128 (the small one is a slice of the big one) and
asserts output rows [0:128] are exactly equal. They weren't: the bidirectional ring sent even
tile-chunks forward and odd backward, accumulating partials in opposite (non-associative) order,
and that even/odd assignment was M-dependent -> a ~1-ULP delta that compounded over Qwen3-32B's
layers and broke seeded sampling under batched prefill. Fixed by deriving chunk parity from the
global tile index (chunk_ring_parity<>() in the 3 ring kernels: reader/reduction/writer).

All cases (bf8/bf16 x 1/2 link) should PASS with 0 differing elements; pre-fix only num_links=2
failed. PCC vs torch stays ~0.99994 throughout -- this was reduction order, never correctness.

Uses bh_1d_mesh_device (auto-sizes to the BH box as a 1D line/ring; RS spans all chips):
  pytest .../blackhole_CI/box/nightly/test_reduce_scatter_batch_invariance_bh.py -s
"""

import pytest
import torch
from loguru import logger

import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc

# Mirrors the Qwen3-32B / P150x4 wo reduce-scatter (per-device hidden):
PER_DEVICE_N = 5120  # self.dim for Qwen3-32B; global hidden = PER_DEVICE_N * num_devices
DIM = 3  # scatter dim (hidden)
M_BIG = 1024  # batched prefill: 8 users x 128 tokens packed
M_SMALL = 128  # single-user prefill
# tt_all_reduce() defaults used by the model for this path:
CHUNKS_PER_SYNC = 10
NUM_WORKERS_PER_LINK = 2
NUM_BUFFERS_PER_CHANNEL = 2


def _run_reduce_scatter(mesh_device, global_input, dtype, topology, sub_device_id, sub_stall_group, num_links):
    """Reduce-scatter one global [1,1,M,PER_DEVICE_N*num_devices] tensor across the mesh on DIM.

    Shards DIM across all chips (one chip per shard) and replicates the trivial mesh axis, so each
    chip holds a [1,1,M,PER_DEVICE_N] partial; reduce_scatter sums the num_devices partials and
    scatters DIM. Returns the composed [1,1,M,PER_DEVICE_N] output (torch) and the torch reference.
    """
    num_devices = mesh_device.get_num_devices()
    grid = mesh_device.compute_with_storage_grid_size()
    ccl_crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    rs_semaphores = [ttnn.create_global_semaphore(mesh_device, ccl_crs, 0) for _ in range(3)]
    barrier_semaphore = ttnn.create_global_semaphore(mesh_device, ccl_crs, 0)
    mesh_shape = tuple(mesh_device.shape)

    # bh_1d_mesh_device lays the chips out on a single axis (the other is size 1). Shard DIM across
    # whichever axis actually holds the devices and replicate the trivial one.
    shard_axis = 0 if mesh_shape[0] > 1 else 1
    placements = [ttnn.PlacementReplicate(), ttnn.PlacementReplicate()]
    placements[shard_axis] = ttnn.PlacementShard(DIM)

    # Per-device input is [1,1,M,PER_DEVICE_N].
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

    out = ttnn.experimental.reduce_scatter_minimal_async(
        input_mesh,
        persistent_output_buffers=None,
        dim=DIM,
        multi_device_global_semaphore=rs_semaphores,
        barrier_semaphore=barrier_semaphore,
        num_links=num_links,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
        topology=topology,
        chunks_per_sync=CHUNKS_PER_SYNC,
        num_workers_per_link=NUM_WORKERS_PER_LINK,
        num_buffers_per_channel=NUM_BUFFERS_PER_CHANNEL,
        subdevice_id=sub_device_id,
    )
    ttnn.synchronize_device(mesh_device, sub_device_ids=sub_stall_group)

    out_torch = ttnn.to_torch(ttnn.from_device(out), mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=DIM))
    out.deallocate(True)
    input_mesh.deallocate(True)

    chunks = torch.chunk(global_input.float(), num_devices, DIM)
    ref = torch.stack(chunks).sum(0)  # [1,1,M,PER_DEVICE_N]
    return out_torch.float(), ref


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
def test_reduce_scatter_batch_invariance(bh_1d_mesh_device, dtype, num_links):
    # bh_1d_mesh_device auto-sizes to the box (1/2/4/8/32 chips); a 1-chip mesh can't reduce-scatter.
    mesh_device = bh_1d_mesh_device
    num_devices = mesh_device.get_num_devices()
    if num_devices < 2:
        pytest.skip(f"reduce-scatter needs >=2 chips, got {num_devices}")

    topology = ttnn.Topology.Ring

    grid = mesh_device.compute_with_storage_grid_size()
    ccl_crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    worker_sub_device = ttnn.SubDevice([ccl_crs])
    sub_device_id = ttnn.SubDeviceId(0)
    sub_stall_group = [sub_device_id]
    sub_device_manager = mesh_device.create_sub_device_manager([worker_sub_device], 0)
    mesh_device.load_sub_device_manager(sub_device_manager)
    mesh_device.set_sub_device_stall_group(sub_stall_group)

    try:
        torch.manual_seed(0)
        # ONE global tensor at M=1024; the M=128 input is its first 128 rows (bit-identical).
        global_big = torch.rand((1, 1, M_BIG, PER_DEVICE_N * num_devices)).bfloat16().float()
        global_small = global_big[:, :, :M_SMALL, :].clone()

        out_big, ref_big = _run_reduce_scatter(
            mesh_device, global_big, dtype, topology, sub_device_id, sub_stall_group, num_links
        )
        print(f"{out_big.shape=} {ref_big.shape=}")

        out_small, ref_small = _run_reduce_scatter(
            mesh_device, global_small, dtype, topology, sub_device_id, sub_stall_group, num_links
        )

        print(f"{out_small.shape=} {ref_small.shape=}")

        # Each individually matches the torch reference (NOT a correctness bug):
        _, pcc_big = comp_pcc(out_big, ref_big)
        _, pcc_small = comp_pcc(out_small, ref_small)
        logger.info(f"[{dtype} {num_links}link] PCC vs torch  M=1024: {pcc_big}   M=128: {pcc_small}")

        big_slice = out_big[:, :, :M_SMALL, :]
        diff = torch.abs(big_slice - out_small)
        max_abs_diff = diff.max().item()
        max_idx = diff.argmax()
        val = big_slice.view(-1)[max_idx].item()

        num_diff = (big_slice != out_small).sum().item()
        total = out_small.numel()
        logger.warning(
            f"[{dtype} {num_links}link] BATCH-VARIANCE: max|out(M=1024)[:128] - out(M=128)| = "
            f"{max_abs_diff:.3e} {val=} ({num_diff}/{total} elements differ)"
        )

        assert max_abs_diff == 0.0, (
            f"reduce_scatter_minimal_async regressed to BATCH-VARIANT (dtype={dtype}, num_links={num_links}): "
            f"identical rows [0:128] differ by max {max_abs_diff:.3e} between M=1024 and M=128 "
            f"({num_diff}/{total} elements) -- ring tile-chunk parity is M-dependent again; check "
            f"chunk_ring_parity() in the 3 ring kernels."
        )
    finally:
        mesh_device.reset_sub_device_stall_group()
        mesh_device.clear_loaded_sub_device_manager()
