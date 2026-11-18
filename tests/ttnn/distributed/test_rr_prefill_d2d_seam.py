# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""D2D streaming across the seams of a torus-seam-torus [RING,RING] pipeline, one service per host row block.

Four ranks (the 2-Galaxy rr binding): rank 0 owns the first in-Galaxy 4x4 torus, ranks 1 and 2 the two hosts of
the seam 4x4 torus (mesh_host_rank 0 owns mesh rows 0-1, 1 owns rows 2-3), rank 3 the last torus. A stage
boundary joining a one-host and a two-host stage cannot be one service over the whole mesh, so each rank carves a
2x4 submesh of the rows it shares with its peer and builds one D2D service per peer on that submesh
(StageLayout.boundary_pairs). A chunk goes torus -> seam -> torus: the first rank sends the hidden-dim half of each
service's row block, the seam hosts forward what they drain, the last rank drains both halves and checks the
reassembled tensor against what the first rank sent.

Launch with a binding from models/demos/deepseek_v3_d_p/utils/rr_pipeline_binding.py (sc2 template, 4 ranks):

    tt-run --rank-binding <binding.yaml> --mpi-args "--map-by rankfile:file=<rankfile>" \\
        pytest tests/ttnn/distributed/test_rr_prefill_d2d_seam.py
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.common.prefill.runners.stage_layout import StageLayout

SIDE = 4
SP_AXIS = 1
TOKENS_PER_COL = 64
HIDDEN = 1024
FIFO_BYTES = 16384
METADATA_WORDS = [7, 128, 256, 1]
WORKER_CORES = ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))
LAYOUT = StageLayout((1, 2, 1))


def _barrier():
    ttnn.distributed_context_barrier()


@pytest.fixture(scope="module")
def mesh():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D_TORUS_XY)
    try:
        mesh_device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(SIDE, SIDE))
        try:
            yield mesh_device
        finally:
            _barrier()
            ttnn.close_mesh_device(mesh_device)
    finally:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def _chunk() -> torch.Tensor:
    tokens = SIDE * TOKENS_PER_COL
    return (torch.arange(tokens * HIDDEN, dtype=torch.int32) * 7 + 3).reshape(1, 1, tokens, HIDDEN)


def _submesh(mesh, rows):
    lo, hi = rows
    return mesh.create_submesh(ttnn.MeshShape(hi - lo, SIDE), offset=ttnn.MeshCoordinate(lo, 0))


def _shard_mapper(sub):
    """Hidden (dim 3) over the submesh rows, tokens (dim 2) over the columns: per chip the shard the full 4x4
    mesh would hold for the same rows."""
    placements = [ttnn.PlacementShard(3), ttnn.PlacementShard(2)]
    return ttnn.create_mesh_mapper(sub, ttnn.MeshMapperConfig(placements=placements))


def _replicate_mapper(sub):
    placements = [ttnn.PlacementReplicate(), ttnn.PlacementReplicate()]
    return ttnn.create_mesh_mapper(sub, ttnn.MeshMapperConfig(placements=placements))


def _block_spec(rows):
    lo, hi = rows
    width = HIDDEN * (hi - lo) // SIDE
    return ttnn.TensorSpec(
        shape=ttnn.Shape([1, 1, SIDE * TOKENS_PER_COL, width]),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        buffer_type=ttnn.BufferType.DRAM,
    )


def _to_device(torch_u32, sub, mapper):
    return ttnn.from_torch(
        torch_u32,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=sub,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mapper,
    )


def _service_kwargs(sub, rows):
    return dict(
        global_spec=_block_spec(rows),
        mapper=_shard_mapper(sub),
        fifo_size_bytes=FIFO_BYTES,
        sender_worker_cores=WORKER_CORES,
        receiver_worker_cores=WORKER_CORES,
        metadata_size_bytes=len(METADATA_WORDS) * 4,
        share_fabric_links=False,
        socket_buffer_type=ttnn.BufferType.L1,
    )


def _row_block(chunk, rows):
    """The hidden-dim slice of the chunk that rows [lo, hi) of the mesh hold."""
    lo, hi = rows
    width = HIDDEN // SIDE
    return chunk[..., lo * width : hi * width]


def _to_host(tensor, sub, rows):
    composer = ttnn.ConcatMesh2dToTensor(sub, mesh_shape=(rows[1] - rows[0], SIDE), dims=(3, 2))
    return ttnn.to_torch(tensor, mesh_composer=composer).to(torch.int32)


def test_d2d_across_one_host_to_two_host_seams(mesh):
    rank = int(ttnn.distributed_context_get_rank())
    assert int(ttnn.distributed_context_get_size()) == LAYOUT.num_ranks, "needs the 4-rank torus-seam-torus binding"
    chunk = _chunk()
    inbound = LAYOUT.inbound_pairs(rank, SIDE)
    outbound = LAYOUT.outbound_pairs(rank, SIDE)

    receivers = []
    for pair in inbound:
        sub = _submesh(mesh, pair.rows)
        service = ttnn.D2DStreamService.create_receiver(
            receiver_mesh=sub,
            sender_rank=pair.sender_rank,
            receiver_rank=rank,
            reserve_links_for_later_service=bool(outbound),
            **_service_kwargs(sub, pair.rows),
        )
        receivers.append((pair, sub, service))
    senders = []
    for pair in outbound:
        sub = _submesh(mesh, pair.rows)
        service = ttnn.D2DStreamService.create_sender(
            sender_mesh=sub,
            sender_rank=rank,
            receiver_rank=pair.receiver_rank,
            **_service_kwargs(sub, pair.rows),
        )
        senders.append((pair, sub, service))
    logger.info(f"rank {rank}: {len(receivers)} inbound and {len(senders)} outbound services up")
    _barrier()

    md_words = torch.tensor(METADATA_WORDS, dtype=torch.int32).reshape(1, 1, 1, -1)
    md_bytes = len(METADATA_WORDS) * 4

    if not receivers:
        for pair, sub, service in senders:
            payload = _to_device(_row_block(chunk, pair.rows).contiguous(), sub, _shard_mapper(sub))
            md = _to_device(md_words, sub, _replicate_mapper(sub))
            ttnn.experimental.deepseek_prefill.outbound_socket_service_sync(service, payload, metadata=md)
        _barrier()
        logger.info(f"rank {rank}: FIRST_STAGE_SENT")
        return

    parts = []
    for index, (pair, sub, service) in enumerate(receivers):
        drained = ttnn.experimental.deepseek_prefill.inbound_socket_service_sync(service, metadata_size_bytes=md_bytes)
        for dev_md in ttnn.get_device_tensors(drained[1]):
            assert ttnn.to_torch(dev_md).flatten().to(torch.int64).tolist() == METADATA_WORDS
        got = _to_host(drained[0], sub, pair.rows)
        assert torch.equal(got, _row_block(chunk, pair.rows)), f"rank {rank}: drained block {pair.rows} differs"
        parts.append(got)
        if senders:
            sender = senders[index][2]
            assert senders[index][0].rows == pair.rows
            ttnn.experimental.deepseek_prefill.outbound_socket_service_sync(sender, drained[0], metadata=drained[1])
    _barrier()
    if senders:
        logger.info(f"rank {rank}: SEAM_FORWARD_OK")
        return
    got = torch.cat(parts, dim=3)
    assert torch.equal(got, chunk), "reassembled chunk differs from what the first stage sent"
    logger.info(f"rank {rank}: D2D_SEAM_OK {tuple(got.shape)} bit-exact over {len(parts)} services")
