# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Row-block D2D services on whole stage meshes: two ranks, each on a 4x4 mesh, joined by TWO D2D services, one
per half of the mesh rows (mesh_shape_override + mesh_offset_override on the segment mapper). Both services live
on the same whole mesh, so the sender passes one whole-mesh activation to both sends and the receiver drains both
into one whole-mesh destination; each op runs only on its service's rows. This is the building block of a
[RING,RING] boundary between a one-host torus and a two-host seam stage, exercised here between the two tray
halves of one Galaxy (each rank owns both of its peer's row blocks, so the test needs one host).

Launch with the intragalaxy binding of this host:
    tt-run --rank-binding models/demos/common/prefill/runners/topology_configuration/pipeline_prefill_request_intragalaxy_2rank_4x4_sp1.$(hostname).yaml \\
        --mpi-args "--bind-to none" pytest models/demos/deepseek_v3_d_p/tests/test_d2d_row_block_services.py
"""

import pytest
import torch
from loguru import logger

import ttnn

SIDE = 4
SP_AXIS = 1  # tokens (dim 2) over mesh axis 1, hidden (dim 3) over mesh axis 0, as a 4x4 prefill stage's D2D
TOKENS = SIDE * 32 * 2
HIDDEN = SIDE * 64
METADATA_WORDS = [11, 22, 33, 44]
WORKERS = ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))
BLOCKS = [(0, 2), (2, 4)]


@pytest.fixture(scope="module")
def mesh():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D_TORUS_X)
    try:
        mesh_device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(SIDE, SIDE))
        try:
            yield mesh_device
        finally:
            ttnn.distributed_context_barrier()
            ttnn.close_mesh_device(mesh_device)
    finally:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def _placements():
    return [ttnn.PlacementShard(3), ttnn.PlacementShard(2)]  # axis 0: hidden, axis 1 (SP): tokens


def _full_mapper(mesh):
    return ttnn.create_mesh_mapper(mesh, ttnn.MeshMapperConfig(placements=_placements()))


def _block(mesh, rows):
    lo, hi = rows
    spec = ttnn.TensorSpec(
        shape=ttnn.Shape([1, 1, TOKENS, HIDDEN // SIDE * (hi - lo)]),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        buffer_type=ttnn.BufferType.DRAM,
    )
    mapper = ttnn.create_mesh_mapper(
        mesh, ttnn.MeshMapperConfig(_placements(), ttnn.MeshShape(hi - lo, SIDE), ttnn.MeshCoordinate([lo, 0]))
    )
    return spec, mapper


def _kwargs(mesh, rows):
    spec, mapper = _block(mesh, rows)
    return dict(
        global_spec=spec,
        mapper=mapper,
        fifo_size_bytes=16384,
        sender_worker_cores=WORKERS,
        receiver_worker_cores=WORKERS,
        metadata_size_bytes=len(METADATA_WORDS) * 4,
        share_fabric_links=False,
        socket_buffer_type=ttnn.BufferType.L1,
    )


def _chunk():
    return (torch.arange(TOKENS * HIDDEN, dtype=torch.float32).reshape(1, 1, TOKENS, HIDDEN) % 251).to(torch.bfloat16)


def _replicated_u32(mesh, words):
    return ttnn.from_torch(
        torch.tensor(words, dtype=torch.int32).reshape(1, 1, 1, -1),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.create_mesh_mapper(
            mesh, ttnn.MeshMapperConfig(placements=[ttnn.PlacementReplicate(), ttnn.PlacementReplicate()])
        ),
    )


def test_two_row_block_services_on_whole_meshes(mesh):
    rank = int(ttnn.distributed_context_get_rank())
    assert int(ttnn.distributed_context_get_size()) == 2, "needs the 2-rank intragalaxy 4x4 binding"
    chunk = _chunk()
    if rank == 0:
        senders = [
            ttnn.D2DStreamService.create_sender(sender_mesh=mesh, sender_rank=0, receiver_rank=1, **_kwargs(mesh, rows))
            for rows in BLOCKS
        ]
        ttnn.distributed_context_barrier()
        payload = ttnn.from_torch(
            chunk,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=_full_mapper(mesh),
        )
        md = _replicated_u32(mesh, METADATA_WORDS)
        for service in senders:
            ttnn.experimental.deepseek_prefill.outbound_socket_service_sync(service, payload, metadata=md)
        ttnn.synchronize_device(mesh)
        ttnn.distributed_context_barrier()
        logger.info("rank 0: ROW_BLOCK_SENT")
        return

    receivers = [
        ttnn.D2DStreamService.create_receiver(receiver_mesh=mesh, sender_rank=0, receiver_rank=1, **_kwargs(mesh, rows))
        for rows in BLOCKS
    ]
    ttnn.distributed_context_barrier()
    tokens_out = ttnn.from_torch(
        torch.zeros(1, 1, TOKENS, HIDDEN, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=_full_mapper(mesh),
    )
    metadata_out = _replicated_u32(mesh, [0] * len(METADATA_WORDS))
    for service in receivers:
        ttnn.experimental.deepseek_prefill.inbound_socket_service_sync(
            service,
            metadata_size_bytes=len(METADATA_WORDS) * 4,
            metadata_out=metadata_out,
            tokens_out=tokens_out,
            extras_out=[],
        )
    ttnn.synchronize_device(mesh)
    got = ttnn.to_torch(tokens_out, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh, mesh_shape=(SIDE, SIDE), dims=(3, 2)))
    for dev_md in ttnn.get_device_tensors(metadata_out):
        assert ttnn.to_torch(dev_md).flatten().to(torch.int64).tolist() == METADATA_WORDS
    ttnn.distributed_context_barrier()
    assert torch.equal(got, chunk), "whole-mesh destination differs from the whole-mesh payload"
    logger.info(f"rank 1: ROW_BLOCK_OK {tuple(got.shape)} bit-exact over {len(receivers)} row-block services")
