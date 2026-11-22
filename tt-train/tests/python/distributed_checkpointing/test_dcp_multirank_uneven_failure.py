# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""A load that stops on one rank before the other, on a mesh the two ranks share, in a tt-run world of its own.

Ranks that share a mesh must make the same device allocations in the same order. Here rank 1 fails placing its first
tensor while rank 0 places all of its own, which leaves the ranks' allocations out of step for anything run after it
in the same world, so the test has a module, and a tt-run launch, of its own. Launch it as ``test_dcp_multirank`` is
launched:

    tt-run --mesh-graph-descriptor tests/ttnn/distributed/config/bh_1x4_dual_rank_mesh_graph_descriptor.textproto \\
        --hosts "$(hostname)" --bare pytest tt-train/tests/python/distributed_checkpointing/test_dcp_multirank_uneven_failure.py

Without ``tt-run`` the module skips; ``test_dcp_cross_world`` runs it under tt-run, which is how CI runs it.
"""

import os

import pytest
import ttnn
from dcp_harness import two_rank_world
from tp_block import fresh_tp, train_step

import ttml.distributed_checkpointing as dcp
from ttml.distributed_checkpointing import _load

pytestmark = [pytest.mark.requires_device, pytest.mark.timeout(600)]

MESH = (1, 4)
DIM = 64


@pytest.fixture(scope="module")
def two_ranks():
    with two_rank_world(MESH, "multirank-uneven", "two ranks co-owning a [1, 4] mesh") as (rank, root):
        yield rank, root


def test_a_load_one_rank_stops_before_the_other_says_to_restart(two_ranks, monkeypatch, expect_error):
    """Rank 1 fails placing its first tensor while rank 0 places all of its own, as a read error on one rank does.
    The ranks then hold different device allocations on the mesh they share, so rank 0's error says that it placed its
    tensors and that the job has to restart rather than load again; rank 1's, which placed none, passes as raised."""
    rank, root = two_ranks
    model, params, opt = fresh_tp(DIM)
    train_step(model, opt, DIM)
    path = os.path.join(root, "one_rank_stops_first")
    dcp.save(path, model=model)
    fresh, fresh_params, _ = fresh_tp(DIM)
    n = len(fresh_params)
    if rank == 1:

        def fail(live, *args):
            raise RuntimeError("simulated device out of memory")

        monkeypatch.setattr(_load, "_place", fail)
    pattern = (
        r"^reading and placing the shards failed on rank\(s\) \[1\] of 2: rank 1: RuntimeError: simulated device out of "
        r"memory"
    )
    try:
        with expect_error(dcp.CheckpointError, pattern) as raised:
            dcp.load(path, model=fresh)
        message = str(raised.value)
        if rank == 0:
            assert (
                f"; rank 0 had placed {n} of {n} tensors, so the model and optimizer are partly restored, and the ranks "
                "sharing this mesh may now hold different device allocations: end the job and restart it instead of "
                "loading again" in message
            ), message
        else:
            assert "partly restored" not in message, "rank 1 placed nothing"
    finally:
        ttnn.distributed_context_barrier()
