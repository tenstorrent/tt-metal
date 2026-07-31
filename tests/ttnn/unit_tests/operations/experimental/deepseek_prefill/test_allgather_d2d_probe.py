# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Two-galaxy probe: high_bw_all_gather -> D2D socket hop -> high_bw_all_gather.

Exercises the large-message all_gather collective while a device-to-device (inter-galaxy) transfer is
in flight, to later characterise all_gather bandwidth under concurrent D2D traffic. Launched as two MPI
ranks under tt-run, one 8x4 galaxy each (mesh 0 / mesh 1) over the connected 2-galaxy MGD:

    rank 0 (mesh 0): build a sharded input -> all_gather (intra-mesh) -> ship the gathered result to
                     rank 1 over the D2D MeshSocket.
    rank 1 (mesh 1): receive the gathered tensor -> all_gather it again (intra-mesh).

The D2D transport mirrors the pipeline prefill runner's ttnn.D2DStreamService usage (create_sender /
create_receiver, outbound_/inbound_socket_service_sync, share_fabric_links lease). Semantics of the
second gather are not meaningful (its input is replicated, not sharded) — the point is the traffic.
"""

import os

import pytest
import torch
from loguru import logger

import ttnn

# num_links=2 keeps the collective portable across galaxy link counts (matches the op's own tests).
_NUM_LINKS = 2
# One tile of rows per device, narrow width: enough to move real data over both fabrics, small enough
# to stay well clear of L1/fifo limits. Override to scale the message up for bandwidth runs.
_ROWS_PER_DEVICE = int(os.environ.get("PROBE_ROWS_PER_DEVICE", 32))
_WIDTH = int(os.environ.get("PROBE_WIDTH", 512))
# all_gather along the 8-long mesh dim (dim_types LINE); a line schedule under FABRIC_2D.
_CLUSTER_AXIS = 0

# D2D socket knobs, same shape as the prefill runner's endpoints.
_SYNC_WORKER_CORES = ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))
_METADATA_SIZE_BYTES = 12
_FIFO_SIZE_BYTES = int(os.environ.get("PROBE_D2D_FIFO_BYTES", 64 * 1024))
_REPLICATE_2D = ttnn.MeshMapperConfig(placements=[ttnn.PlacementReplicate(), ttnn.PlacementReplicate()])


def _fabric_router_config():
    config = ttnn.FabricRouterConfig()
    config.max_packet_payload_size_bytes = 14 * 1024
    return config


_DEVICE_PARAMS = {
    "fabric_config": ttnn.FabricConfig.FABRIC_2D,
    "fabric_router_config": _fabric_router_config(),
    "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
    "l1_small_size": 2048,
}


def _data_valid_semaphore(mesh_device):
    grid = mesh_device.compute_with_storage_grid_size()
    cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    return ttnn.create_global_semaphore(mesh_device, cores, 0, buffer_type=ttnn.BufferType.L1_SMALL)


def _gathered_spec(gathered_rows):
    return ttnn.TensorSpec(
        shape=ttnn.Shape([1, 1, gathered_rows, _WIDTH]),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        buffer_type=ttnn.BufferType.DRAM,
    )


def _persistent_replica(mesh_device, rows):
    return ttnn.from_torch(
        torch.zeros((1, 1, rows, _WIDTH), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        device=mesh_device,
    )


def _all_gather(mesh_device, device_input, semaphore):
    """high_bw_all_gather along _CLUSTER_AXIS into a fresh persistent replicated output; returns it."""
    local_padded = ttnn.get_device_tensors(device_input)[0].padded_shape
    gathered_rows = local_padded[2] * mesh_device.shape[_CLUSTER_AXIS]
    output = _persistent_replica(mesh_device, gathered_rows)
    ttnn.experimental.deepseek_prefill.high_bw_all_gather(
        device_input,
        dim=2,
        output_tensor=output,
        cluster_axis=_CLUSTER_AXIS,
        data_valid_semaphore=semaphore,
        num_links=_NUM_LINKS,
    )
    ttnn.synchronize_device(mesh_device)
    return output, gathered_rows


def _socket_common(mesh_device, gathered_rows):
    # Fresh mapper per call: create_sender/create_receiver MOVE the mapper (std::unique_ptr).
    return dict(
        global_spec=_gathered_spec(gathered_rows),
        mapper=ttnn.create_mesh_mapper(mesh_device, _REPLICATE_2D),
        fifo_size_bytes=_FIFO_SIZE_BYTES,
        sender_worker_cores=_SYNC_WORKER_CORES,
        receiver_worker_cores=_SYNC_WORKER_CORES,
        metadata_size_bytes=_METADATA_SIZE_BYTES,
        share_fabric_links=True,
        socket_buffer_type=ttnn.BufferType.L1,
    )


def _send(outbound, tensor):
    backing = outbound.get_backing_tensor()
    md = ttnn.from_torch(
        torch.zeros((1, 1, 1, 3), dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=backing.device(),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.create_mesh_mapper(backing.device(), _REPLICATE_2D),
    )
    ttnn.experimental.deepseek_prefill.outbound_socket_service_sync(outbound, tensor, metadata=md)


@pytest.mark.parametrize("device_params", [_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
def test_allgather_d2d_probe(mesh_device):
    if not ttnn.distributed_context_is_initialized():
        ttnn.init_distributed_context()
    rank = int(ttnn.distributed_context_get_rank())
    num_ranks = int(ttnn.distributed_context_get_size())
    assert num_ranks == 2, f"probe needs exactly 2 ranks (2 galaxies), got {num_ranks}"

    axis = mesh_device.shape[_CLUSTER_AXIS]
    gathered_rows = _ROWS_PER_DEVICE * axis
    logger.info(f"[probe rank {rank}] mesh={tuple(mesh_device.shape)} axis={axis} gathered_rows={gathered_rows}")

    # Both ranks create their D2D endpoint first: create_sender/create_receiver rendezvous point-to-point
    # and each MeshSocket ctor blocks until its peer's, so this is the sync point before any compute.
    common = _socket_common(mesh_device, gathered_rows)
    if rank == 0:
        outbound = ttnn.D2DStreamService.create_sender(
            sender_mesh=mesh_device, sender_rank=0, receiver_rank=1, **common
        )
        logger.info("[probe rank 0] D2D sender up")
    else:
        inbound = ttnn.D2DStreamService.create_receiver(
            receiver_mesh=mesh_device, sender_rank=0, receiver_rank=1, **common
        )
        logger.info("[probe rank 1] D2D receiver up")

    semaphore = _data_valid_semaphore(mesh_device)

    if rank == 0:
        host_input = torch.rand((1, 1, gathered_rows, _WIDTH), dtype=torch.bfloat16)
        device_input = ttnn.from_torch(
            host_input,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(2, None), mesh_shape=tuple(mesh_device.shape)),
            device=mesh_device,
        )

        # Lease mode: the sender holds no fabric link until granted, so all_gather has the links to
        # itself. wait_for_fabric_links() here just confirms the link is free (no grant outstanding).
        outbound.wait_for_fabric_links()
        gathered, _ = _all_gather(mesh_device, device_input, semaphore)
        logger.info(f"[probe rank 0] all_gather done -> {tuple(gathered.shape)}; shipping over D2D")

        _send(outbound, gathered)
        outbound.release_fabric_links()  # grant the one transfer; the service ships now
        outbound.wait_for_fabric_links()  # block until the transfer is off the link
        logger.info("[probe rank 0] D2D send complete")
    else:
        inbound.release_fabric_links()  # grant the receive turn
        received, _ = ttnn.experimental.deepseek_prefill.inbound_socket_service_sync(
            inbound, metadata_size_bytes=_METADATA_SIZE_BYTES
        )
        inbound.wait_for_fabric_links()  # block until off the link before the next fabric op
        logger.info(f"[probe rank 1] D2D recv done -> {tuple(received.shape)}; second all_gather")

        gathered2, rows2 = _all_gather(mesh_device, received, semaphore)
        logger.info(f"[probe rank 1] second all_gather done -> {tuple(gathered2.shape)}")
        assert gathered2.shape[2] == rows2

    ttnn.distributed_context_barrier()
    logger.info(f"[probe rank {rank}] done")
