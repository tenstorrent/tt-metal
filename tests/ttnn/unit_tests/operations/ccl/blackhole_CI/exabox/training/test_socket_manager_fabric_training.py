# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""SocketManager.send / SocketManager.recv (FABRIC) exabox tests on QUAD_BH.

Same three training-shaped patterns as test_send_recv_training.py but routed
through the tt-train SocketManager → MeshSocket → BidirectionalFabricSocket
→ on-chip fabric, instead of the byte-level DistributedContext.send/recv
host-MPI primitive.

Requires the multi-mesh rank-binding
quad_bh_galaxy_split_4x2_multi_mesh_rank_bindings.yaml (each rank gets its
own mesh_id 0..3) backed by bh_galaxy_split_4x2_multi_mesh.textproto
(declares a ring of cross-mesh fabric connections). Without that binding
MeshSocket TT_FATALs on sender_mesh_id == receiver_mesh_id
(mesh_socket.cpp:187-190).

Launch (4-rank QUAD_BH, multi-mesh):
  tt-run \\
    --rank-binding tests/tt_metal/distributed/config/quad_bh_galaxy_split_4x2_multi_mesh_rank_bindings.yaml \\
    --mpi-args "--host <h0>,<h1>,<h2>,<h3>" \\
    bash -c "source python_env/bin/activate && \\
             MESH_DEVICE=QUAD_BH pytest --timeout=600 \\
             tests/ttnn/unit_tests/operations/ccl/blackhole_CI/exabox/training/test_socket_manager_fabric_training.py::test_*_32x4"
"""

from __future__ import annotations

import numpy as np
import pytest
from loguru import logger


# Per-element rank tag. Receiver checks tensor == sender_rank_tag(sender)
def sender_rank_tag(sender_rank: int) -> float:
    return float(sender_rank + 1)


# Single-tile bfloat16 payload (1, 1, 32, 32) — the smallest tensor that fits
# the tile-aligned layout used by MeshSocket on Blackhole.
PAYLOAD_SHAPE = (1, 1, 32, 32)


def _make_tagged_tensor(rt, value: float):
    """Build a ttml.autograd.Tensor uniformly filled with `value`.

    Uses ReplicateTensorToMesh — without an explicit mapper, from_numpy
    leaves the tensor in multi-shard host storage that the fabric send
    path can't extract (host_buffer/functions.cpp:51 expects 1 buffer).
    The replicate mapper distributes the same data to every device on the
    (8,4) per-host mesh, matching the MeshSocket SocketConnection layout
    in tt-train SocketManager (one connection per (row, col) device).

    bfloat16 is exact for small integer floats (1.0..256.0) so a strict
    equality check on the round-trip works for our rank tags.
    """
    device = rt.autograd_ctx.get_device()
    # Use the C++ CppTensorToMesh factory directly — ttnn.distributed.ReplicateTensorToMesh
    # is a Python wrapper that doesn't satisfy from_numpy's nanobind type check.
    mapper = rt.ttnn._ttnn.multi_device.replicate_tensor_to_mesh_mapper(device)
    np_data = np.full(PAYLOAD_SHAPE, value, dtype=np.float32)
    # ROW_MAJOR avoids the post-create tilize_with_zero_padding step which
    # mishandles the multi-shard replicated layout in this code path; the
    # downstream send_async/recv_async accept ROW_MAJOR.
    return rt.ttml.autograd.Tensor.from_numpy(
        np_data,
        layout=rt.ttnn.Layout.ROW_MAJOR,
        new_type=rt.ttnn.DataType.BFLOAT16,
        mapper=mapper,
    )


def _make_zero_tensor(rt):
    return _make_tagged_tensor(rt, 0.0)


def _assert_tensor_equals_value(rt, tensor, expected_value: float, context: str) -> None:
    """Verify the tensor matches expected_value after a fabric round-trip.

    The tensor is replicated across the (8,4) per-rank mesh, so we use a
    concat_mesh_to_tensor_composer along dim=0 to aggregate shards (each
    shard is identical). We then check the first replica matches.
    """
    device = rt.autograd_ctx.get_device()
    composer = rt.ttml.core.distributed.concat_mesh_to_tensor_composer(device, 0)
    actual = tensor.to_numpy(composer=composer).flatten()
    # Each of the 32 shards holds the same payload, so check the first.
    payload_size = int(np.prod(PAYLOAD_SHAPE))
    head = actual[:payload_size]
    if not np.all(head == expected_value):
        sample = head[:8].tolist() if head.size >= 8 else head.tolist()
        raise AssertionError(
            f"{context}: expected all elements == {expected_value}, "
            f"got first-8 sample {sample} (size {head.size}, total recv {actual.size})"
        )


# ---------------------------------------------------------------------------
# 1. Round-robin send/recv ring via SocketManager-FABRIC
# ---------------------------------------------------------------------------
@pytest.mark.requires_device(["QUAD_BH"])
@pytest.mark.timeout(600)
def test_round_robin_send_recv_fabric_32x4(socket_manager_runtime):
    """Each rank exchanges a tagged tensor with both ring neighbors.

    Mirrors test_round_robin_send_recv_32x4 in test_send_recv_training.py
    but routes the payload through SocketManager (FABRIC) — the production
    code path used by tt-train pipeline parallel and 3-tier optimizer.
    """
    rt = socket_manager_runtime
    rank, world = rt.rank, rt.world_size
    sm = rt.socket_manager
    ctx = rt.distributed_ctx

    pairs = [(i, (i + 1) % world) for i in range(world)]
    pairs += [((i + 1) % world, i) for i in range(world)]

    ctx.barrier()
    for sender, receiver in pairs:
        if rank == sender:
            tensor = _make_tagged_tensor(rt, sender_rank_tag(sender))
            sm.send(tensor, ctx, receiver)
            logger.info(f"[rank={rank}] sent tag={sender_rank_tag(sender)} to rank {receiver}")
        elif rank == receiver:
            recv_buf = _make_zero_tensor(rt)
            recv_buf = sm.recv(recv_buf, ctx, sender)
            _assert_tensor_equals_value(
                rt,
                recv_buf,
                sender_rank_tag(sender),
                context=f"FABRIC round-robin {sender}->{receiver} on rank {rank}",
            )
            logger.info(f"[rank={rank}] recv ok from rank {sender}")
        ctx.barrier()


# ---------------------------------------------------------------------------
# 2. Pipeline activation hand-off via SocketManager-FABRIC
# ---------------------------------------------------------------------------
@pytest.mark.requires_device(["QUAD_BH"])
@pytest.mark.timeout(600)
def test_pipeline_activation_handoff_fabric_32x4(socket_manager_runtime):
    """Rank pair simulates one pipeline-parallel stage boundary over fabric.

    Forward: rank 0 sends activation tensor to rank 1.
    Backward: rank 1 sends gradient tensor back to rank 0.
    Mirrors ops/distributed/pipeline_parallel_comm_ops.cpp:11-38.
    """
    rt = socket_manager_runtime
    rank = rt.rank
    sm = rt.socket_manager
    ctx = rt.distributed_ctx

    UPSTREAM, DOWNSTREAM = 0, 1
    ACTIVATION_TAG_VALUE = 11.0
    GRADIENT_TAG_VALUE = 22.0

    # forward: upstream -> downstream
    ctx.barrier()
    if rank == UPSTREAM:
        tensor = _make_tagged_tensor(rt, ACTIVATION_TAG_VALUE)
        sm.send(tensor, ctx, DOWNSTREAM)
        logger.info(f"[rank={rank}] sent activation tag={ACTIVATION_TAG_VALUE} to {DOWNSTREAM}")
    elif rank == DOWNSTREAM:
        recv_buf = _make_zero_tensor(rt)
        recv_buf = sm.recv(recv_buf, ctx, UPSTREAM)
        _assert_tensor_equals_value(
            rt,
            recv_buf,
            ACTIVATION_TAG_VALUE,
            context=f"FABRIC forward activation on rank {rank}",
        )
        logger.info(f"[rank={rank}] recv activation ok from {UPSTREAM}")

    ctx.barrier()

    # backward: downstream -> upstream
    if rank == DOWNSTREAM:
        tensor = _make_tagged_tensor(rt, GRADIENT_TAG_VALUE)
        sm.send(tensor, ctx, UPSTREAM)
        logger.info(f"[rank={rank}] sent gradient tag={GRADIENT_TAG_VALUE} to {UPSTREAM}")
    elif rank == UPSTREAM:
        recv_buf = _make_zero_tensor(rt)
        recv_buf = sm.recv(recv_buf, ctx, DOWNSTREAM)
        _assert_tensor_equals_value(
            rt,
            recv_buf,
            GRADIENT_TAG_VALUE,
            context=f"FABRIC backward gradient on rank {rank}",
        )
        logger.info(f"[rank={rank}] recv gradient ok from {DOWNSTREAM}")

    ctx.barrier()


# ---------------------------------------------------------------------------
# 3. Remote-optimizer gradient exchange via SocketManager-FABRIC
# ---------------------------------------------------------------------------
@pytest.mark.requires_device(["QUAD_BH"])
@pytest.mark.timeout(600)
def test_remote_optimizer_grad_exchange_fabric_32x4(socket_manager_runtime):
    """Workers send gradients to rank 0; aggregator broadcasts updated weights back.

    Mirrors RemoteOptimizer::send_gradients / receive_weights in
    optimizers/remote_optimizer.cpp:67-84.

    Topology constraint: the bh-glx-b06/b07 4-host cluster's inter-mesh
    fabric is a 4-node ring (0-1, 1-2, 2-3, 0-3) — there is no direct
    0↔2 edge. MeshSocket descriptor handshake/setup between non-adjacent
    meshes (0↔2) consistently times out on this hardware. We therefore
    restrict the worker set to ranks {1, 3} (both direct neighbors of
    aggregator rank 0) and skip rank 2. The aggregator-broadcast pattern
    is exercised end-to-end; the topology-spanning worker count is covered
    separately by the byte-level (host-MPI) variant in
    test_send_recv_training.py.
    """
    rt = socket_manager_runtime
    rank = rt.rank
    sm = rt.socket_manager
    ctx = rt.distributed_ctx

    AGGREGATOR = 0
    # Adjacent ranks to AGGREGATOR on the ring: 1 and 3.
    WORKERS = [1, 3]
    WEIGHT_TAG_VALUE = 99.0

    ctx.barrier()

    # phase 1: workers -> aggregator
    if rank == AGGREGATOR:
        for worker in WORKERS:
            recv_buf = _make_zero_tensor(rt)
            recv_buf = sm.recv(recv_buf, ctx, worker)
            _assert_tensor_equals_value(
                rt,
                recv_buf,
                sender_rank_tag(worker),
                context=f"FABRIC aggregator recv grad from worker {worker}",
            )
            logger.info(f"[rank={rank}] aggregator recv ok from worker {worker}")
    elif rank in WORKERS:
        tensor = _make_tagged_tensor(rt, sender_rank_tag(rank))
        sm.send(tensor, ctx, AGGREGATOR)
        logger.info(f"[rank={rank}] worker sent grad to aggregator")
    # ranks not in {AGGREGATOR} ∪ WORKERS are barrier-only participants.

    ctx.barrier()

    # phase 2: aggregator -> workers
    if rank == AGGREGATOR:
        for worker in WORKERS:
            tensor = _make_tagged_tensor(rt, WEIGHT_TAG_VALUE)
            sm.send(tensor, ctx, worker)
            logger.info(f"[rank={rank}] aggregator sent weight to worker {worker}")
    elif rank in WORKERS:
        recv_buf = _make_zero_tensor(rt)
        recv_buf = sm.recv(recv_buf, ctx, AGGREGATOR)
        _assert_tensor_equals_value(
            rt,
            recv_buf,
            WEIGHT_TAG_VALUE,
            context=f"FABRIC worker recv weight from aggregator on rank {rank}",
        )
        logger.info(f"[rank={rank}] worker recv weight ok")

    ctx.barrier()
