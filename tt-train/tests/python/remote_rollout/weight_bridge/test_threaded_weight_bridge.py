# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""2-rank test for :class:`ThreadedWeightBridge` (1->1) over real MPI.

Rank 0 opens a 2-CQ device through ttml and sends weight versions 0, 1 and 2, then closes the bridge. Rank 1 opens a
2-CQ mesh, receives each version and checks its values, then checks that the peer closed. Runs under tt-run with
world_size == 2; self-skips otherwise (see ``runner_weight_bridge.sh``).
"""

from __future__ import annotations

import math
import os
import sys

import pytest

_WORLD_SIZE = int(os.environ.get("OMPI_COMM_WORLD_SIZE", "0"))
if _WORLD_SIZE != 2:
    pytest.skip(
        "test_threaded_weight_bridge must run under tt-run with world_size == 2 (use runner_weight_bridge.sh).",
        allow_module_level=True,
    )

_MPI_RANK = int(os.environ["OMPI_COMM_WORLD_RANK"])

import torch  # noqa: E402
import ttml  # noqa: E402
import ttnn  # noqa: E402

from ttml.trainers.grpo_trainer.remote_rollout.threaded_weight_bridge import ThreadedWeightBridge  # noqa: E402
from ttml.trainers.grpo_trainer.remote_rollout.weight_bridge import TTML_RANK, TTT_RANK  # noqa: E402

_SPECS = {"weight.alpha": (1, 1, 32, 64), "weight.beta": (1, 1, 64, 32)}
_VERSIONS = (0, 1, 2)


def _weights(version: int) -> dict:
    """Values below 256, so bfloat16 holds them exactly; each version has different values."""
    return {
        key: ((torch.arange(math.prod(shape)) % 64 + 64 * version).reshape(shape)).to(torch.bfloat16)
        for key, shape in _SPECS.items()
    }


def _sender_side() -> None:
    ctx = ttml.autograd.AutoContext.get_instance()
    ctx.open_device([1, 1], [], 2)
    mesh = ctx.get_device()
    bridge = ThreadedWeightBridge.sender(peer_rank=TTT_RANK, mesh_device=mesh)
    bridge.connect()
    try:
        for version in _VERSIONS:
            weights = {
                key: ttnn.from_torch(
                    value,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=mesh,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(mesh),
                )
                for key, value in _weights(version).items()
            }
            bridge.send_weights(weights, version)
    finally:
        bridge.close()
        ctx.close_device()


def _receiver_side() -> None:
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), num_command_queues=2)
    bridge = ThreadedWeightBridge.receiver(peer_rank=TTML_RANK, submeshes=[mesh])
    bridge.connect()
    received = []
    try:
        while True:
            with bridge.receive_weights(block=True) as got:
                if got is None:
                    break
                version, (pads,) = got
                for key, expected in _weights(version).items():
                    assert torch.equal(ttnn.to_torch(pads[key], cq_id=0), expected), (version, key)
                received.append(version)
        assert bridge.peer_closed
    finally:
        bridge.close()
        ttnn.close_mesh_device(mesh)
    assert received == list(_VERSIONS)


def test_threaded_weight_bridge_sends_versions_then_closes() -> None:
    if not ttnn.distributed_context_is_initialized():
        ttnn.init_distributed_context()
    ttml.autograd.AutoContext.get_instance().initialize_distributed_context(*sys.argv)
    if _MPI_RANK == TTML_RANK:
        _sender_side()
    elif _MPI_RANK == TTT_RANK:
        _receiver_side()
    else:
        raise RuntimeError(f"Unexpected MPI rank {_MPI_RANK} (world_size={_WORLD_SIZE}).")
