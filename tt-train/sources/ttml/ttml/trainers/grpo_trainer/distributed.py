# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import os
import sys
import traceback
from typing import TYPE_CHECKING, Iterator

import ttnn

from .remote_rollout.weight_bridge import TTT_RANK

if TYPE_CHECKING:
    from .grpo_trainer import GRPOConfig


def init_distributed_training(config: "GRPOConfig") -> None:
    """Check the process layout the rollout mode needs, and set it up for remote modes."""
    if config.backend.in_process:
        world = int(ttnn.distributed_context_get_size()) if ttnn.distributed_context_is_initialized() else 1
        if world != 1:
            raise ValueError(
                f"rollout_backend={config.rollout_backend!r} runs in a single process, got world size {world}"
            )
        return
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)
    if not ttnn.distributed_context_is_initialized():
        ttnn.init_distributed_context()
    world = int(ttnn.distributed_context_get_size())
    if world != 2:
        raise ValueError(
            f"rollout_backend={config.rollout_backend!r} needs 2 ranks (tt-run with rank bindings), got world size {world}"
        )


def is_rollout_rank(config: "GRPOConfig") -> bool:
    return not config.backend.in_process and int(ttnn.distributed_context_get_rank()) == TTT_RANK


@contextlib.contextmanager
def exit_on_error() -> Iterator[None]:
    """Print the traceback and ``os._exit(1)`` on any exception.

    A Python exception exits through ``MPI_Finalize``, which waits for the peer rank; ``os._exit`` ends the job.
    """
    try:
        yield
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
