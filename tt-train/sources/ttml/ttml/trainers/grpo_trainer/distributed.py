# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import ttnn

from .remote_rollout.weight_bridge import TTT_RANK

if TYPE_CHECKING:
    from .grpo_trainer import GRPOConfig


def init_distributed_training(config: "GRPOConfig") -> None:
    """Check the process layout the rollout mode needs, and set it up for remote modes."""
    if config.rollout_mode == "in_process":
        world = int(ttnn.distributed_context_get_size()) if ttnn.distributed_context_is_initialized() else 1
        if world != 1:
            raise ValueError(f"rollout_mode='in_process' runs in a single process, got world size {world}")
        return
    if not ttnn.distributed_context_is_initialized():
        ttnn.init_distributed_context()
    world = int(ttnn.distributed_context_get_size())
    if world != 2:
        raise ValueError(
            f"rollout_mode={config.rollout_mode!r} needs 2 ranks (tt-run with rank bindings), got world size {world}"
        )
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)


def is_rollout_rank(config: "GRPOConfig") -> bool:
    return config.rollout_mode != "in_process" and int(ttnn.distributed_context_get_rank()) == TTT_RANK
