# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import ttnn

from .remote_rollout.weight_bridge import TTT_RANK

if TYPE_CHECKING:
    from .grpo_trainer import GRPOConfig


def init_distributed_training(config: "GRPOConfig", device_config: Any) -> None:
    """Check the process layout the rollout mode needs, and set it up for remote modes.

    Cross-rank traffic goes over MPI, so fabric is only enabled on a rank whose own mesh has more than one device.
    """
    if config.backend.in_process:
        world = int(ttnn.distributed_context_get_size()) if ttnn.distributed_context_is_initialized() else 1
        if world != 1:
            raise ValueError(
                f"rollout_backend={config.rollout_backend!r} runs in a single process, got world size {world}"
            )
        return
    if not ttnn.distributed_context_is_initialized():
        ttnn.init_distributed_context()
    world = int(ttnn.distributed_context_get_size())
    if world != 2:
        raise ValueError(
            f"rollout_backend={config.rollout_backend!r} needs 2 ranks (tt-run with rank bindings), got world size {world}"
        )
    if is_rollout_rank(config):
        num_devices = math.prod(config.remote_rollout.mesh_shape)
    else:
        num_devices = device_config.total_devices()
    if num_devices > 1:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)


def is_rollout_rank(config: "GRPOConfig") -> bool:
    return not config.backend.in_process and int(ttnn.distributed_context_get_rank()) == TTT_RANK
