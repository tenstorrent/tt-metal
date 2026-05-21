# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""DistributedContext.send / DistributedContext.recv exabox tests on QUAD_BH.

Three training-shaped uses of the byte-level point-to-point primitive — the
op underlying every multi-host send/recv in the C++ training stack:

1. test_round_robin_send_recv_32x4
   Adjacent-rank ring: rank i sends to (i+1) % world, then (i+1) sends back
   to i. Mirrors the warm-up loop in
   tt-train/sources/examples/python/multihost/pipeline_parallel_training/training.py:96-120.

2. test_pipeline_activation_handoff_32x4
   Sender → receiver pair, simulating one pipeline parallel stage boundary:
   forward sends an "activation" payload; backward sends a "gradient" payload
   in the reverse direction. Mirrors
   tt-train/sources/ttml/ops/distributed/pipeline_parallel_comm_ops.cpp:11-38.

3. test_remote_optimizer_grad_exchange_32x4
   Rank 0 = aggregator/optimizer, ranks 1..N-1 = workers. Workers send
   gradients to rank 0; rank 0 sends updated weights back. Mirrors
   tt-train/sources/ttml/optimizers/remote_optimizer.cpp:67-84
   and tt-train/sources/examples/nano_gpt/3tier/.

These exercise the *underlying* MPI primitive (MPI_Send/MPI_Recv via
DistributedContext::send/recv in mpi_distributed_context.cpp:227-240).
The MeshSocket-based wrappers (FABRIC and SocketType::MPI) cannot be used
on Galaxy single-mesh layouts; see conftest.py for the rationale.

Launch (4-rank QUAD_BH):
  tt-run \\
    --rank-binding tests/tt_metal/distributed/config/32x4_quad_bh_galaxy_rank_bindings.yaml \\
    --mpi-args "--host <h0>,<h1>,<h2>,<h3>" \\
    bash -c "source python_env/bin/activate && \\
             MESH_DEVICE=QUAD_BH pytest --timeout=300 \\
             tests/ttnn/unit_tests/operations/ccl/blackhole_CI/exabox/training/test_send_recv_training.py::test_*_32x4"
"""

from __future__ import annotations

import pytest
from loguru import logger

from tests.ttnn.unit_tests.operations.ccl.blackhole_CI.exabox.training._training_helpers import (
    DEFAULT_PAYLOAD_DTYPE,
    DEFAULT_PAYLOAD_SHAPE,
    assert_array_equals_value,
    make_constant_payload,
    make_tagged_payload,
    recv_array,
    send_array,
    sender_rank_tag,
)


# ---------------------------------------------------------------------------
# 1. Round-robin send/recv ring
# ---------------------------------------------------------------------------
@pytest.mark.requires_device(["QUAD_BH"])
@pytest.mark.timeout(300)
def test_round_robin_send_recv_32x4(distributed_runtime):
    """Each rank exchanges a tagged payload with both neighbors in the ring.

    Pattern: for every (sender, receiver) in [(i, (i+1)%W) for i in 0..W-1]
    plus the reverse direction, sender publishes a rank-tagged payload and
    receiver verifies the bytes. Every rank acts as both sender and receiver
    across the loop, so a hang or data-corruption on any link is flagged.
    """
    rt = distributed_runtime
    rank, world = rt.rank, rt.world_size
    ctx = rt.distributed_ctx

    pairs = [(i, (i + 1) % world) for i in range(world)]
    pairs += [((i + 1) % world, i) for i in range(world)]

    ctx.barrier()
    for sender, receiver in pairs:
        if rank == sender:
            payload = make_tagged_payload(sender)
            send_array(ctx, payload, dest=receiver)
            logger.info(f"[rank={rank}] sent tag={sender_rank_tag(sender)} to rank {receiver}")
        elif rank == receiver:
            recv = recv_array(ctx, source=sender, shape=DEFAULT_PAYLOAD_SHAPE, dtype=DEFAULT_PAYLOAD_DTYPE)
            assert_array_equals_value(
                recv,
                sender_rank_tag(sender),
                context=f"round-robin {sender}->{receiver} on rank {rank}",
            )
            logger.info(f"[rank={rank}] recv ok from rank {sender}")
        ctx.barrier()


# ---------------------------------------------------------------------------
# 2. Pipeline activation hand-off (forward + backward over one stage boundary)
# ---------------------------------------------------------------------------
@pytest.mark.requires_device(["QUAD_BH"])
@pytest.mark.timeout(300)
def test_pipeline_activation_handoff_32x4(distributed_runtime):
    """Rank pair simulates one pipeline-parallel stage boundary.

    Forward pass: stage_i (sender) sends an "activation" payload to
    stage_{i+1} (receiver). Backward pass: stage_{i+1} sends a "gradient"
    payload back to stage_i. Tags use distinct values so a forward/backward
    swap or a leftover buffer would be caught.

    Uses ranks (0, 1) as the stage pair on QUAD_BH; the remaining ranks
    barrier-only to keep the collective ordered.
    """
    rt = distributed_runtime
    rank = rt.rank
    ctx = rt.distributed_ctx

    UPSTREAM, DOWNSTREAM = 0, 1
    ACTIVATION_TAG_VALUE = 11.0
    GRADIENT_TAG_VALUE = 22.0

    # --- forward: upstream -> downstream
    ctx.barrier()
    if rank == UPSTREAM:
        send_array(ctx, make_constant_payload(ACTIVATION_TAG_VALUE), dest=DOWNSTREAM)
        logger.info(f"[rank={rank}] sent activation tag={ACTIVATION_TAG_VALUE} to {DOWNSTREAM}")
    elif rank == DOWNSTREAM:
        recv = recv_array(ctx, source=UPSTREAM, shape=DEFAULT_PAYLOAD_SHAPE, dtype=DEFAULT_PAYLOAD_DTYPE)
        assert_array_equals_value(
            recv,
            ACTIVATION_TAG_VALUE,
            context=f"forward activation on rank {rank}",
        )
        logger.info(f"[rank={rank}] recv activation ok from {UPSTREAM}")

    ctx.barrier()

    # --- backward: downstream -> upstream
    if rank == DOWNSTREAM:
        send_array(ctx, make_constant_payload(GRADIENT_TAG_VALUE), dest=UPSTREAM)
        logger.info(f"[rank={rank}] sent gradient tag={GRADIENT_TAG_VALUE} to {UPSTREAM}")
    elif rank == UPSTREAM:
        recv = recv_array(ctx, source=DOWNSTREAM, shape=DEFAULT_PAYLOAD_SHAPE, dtype=DEFAULT_PAYLOAD_DTYPE)
        assert_array_equals_value(
            recv,
            GRADIENT_TAG_VALUE,
            context=f"backward gradient on rank {rank}",
        )
        logger.info(f"[rank={rank}] recv gradient ok from {DOWNSTREAM}")

    ctx.barrier()


# ---------------------------------------------------------------------------
# 3. Remote-optimizer gradient exchange
# ---------------------------------------------------------------------------
@pytest.mark.requires_device(["QUAD_BH"])
@pytest.mark.timeout(300)
def test_remote_optimizer_grad_exchange_32x4(distributed_runtime):
    """Workers send gradients to rank 0 (aggregator/optimizer); aggregator
    sends updated weights back to each worker.

    Mirrors RemoteOptimizer::send_gradients / receive_weights from
    tt-train/sources/ttml/optimizers/remote_optimizer.cpp:67-84. The
    aggregator verifies it received the right per-worker tag; each worker
    verifies it got the broadcast weight tag back. A leftover or misrouted
    buffer would fail the per-worker tag check at the aggregator.
    """
    rt = distributed_runtime
    rank, world = rt.rank, rt.world_size
    ctx = rt.distributed_ctx

    AGGREGATOR = 0
    WEIGHT_TAG_VALUE = 99.0

    ctx.barrier()

    # Phase 1: workers -> aggregator (gradient send)
    if rank == AGGREGATOR:
        for worker in range(1, world):
            recv = recv_array(ctx, source=worker, shape=DEFAULT_PAYLOAD_SHAPE, dtype=DEFAULT_PAYLOAD_DTYPE)
            assert_array_equals_value(
                recv,
                sender_rank_tag(worker),
                context=f"aggregator recv grad from worker {worker}",
            )
            logger.info(f"[rank={rank}] aggregator recv ok from worker {worker}")
    else:
        send_array(ctx, make_tagged_payload(rank), dest=AGGREGATOR)
        logger.info(f"[rank={rank}] worker sent grad to aggregator")

    ctx.barrier()

    # Phase 2: aggregator -> workers (weight broadcast back)
    if rank == AGGREGATOR:
        for worker in range(1, world):
            send_array(ctx, make_constant_payload(WEIGHT_TAG_VALUE), dest=worker)
            logger.info(f"[rank={rank}] aggregator sent weight to worker {worker}")
    else:
        recv = recv_array(ctx, source=AGGREGATOR, shape=DEFAULT_PAYLOAD_SHAPE, dtype=DEFAULT_PAYLOAD_DTYPE)
        assert_array_equals_value(
            recv,
            WEIGHT_TAG_VALUE,
            context=f"worker recv weight from aggregator on rank {rank}",
        )
        logger.info(f"[rank={rank}] worker recv weight ok")

    ctx.barrier()
