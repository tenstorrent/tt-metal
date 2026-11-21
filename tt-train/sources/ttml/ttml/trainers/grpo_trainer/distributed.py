# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import math
import os
import sys
import traceback
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterator, List, Optional, Tuple

import ttml
import ttnn

from .remote_rollout.weight_bridge import TTT_RANK

if TYPE_CHECKING:
    from .grpo_trainer import GRPOConfig


@dataclass(frozen=True)
class BatchLayout:
    """How batch rows are split along axis 0 of the open device."""

    num_shards: int
    mapper: Any
    composer: Any
    seed_axes: Optional[List[int]]
    ddp_context_enabled: bool
    ddp_enabled: bool
    fsdp_enabled: bool
    fsdp_sync_axes: Tuple[str, ...]
    grad_sync_world_size: int


def resolve_batch_layout() -> BatchLayout:
    """Read the batch layout from the open device, its named mesh and the parallelism context.

    DDP comes from the parallelism context (Llama) or a "dp" mesh axis (Qwen3); FSDP from an "fsdp" axis.
    """
    autograd_ctx = ttml.autograd.AutoContext.get_instance()
    device = autograd_ctx.get_device()
    mesh = ttml.maybe_mesh()

    def axis_size(name: str) -> int:
        return mesh.axis_size(name) if mesh is not None and mesh.has_axis(name) else 1

    ddp_context_enabled = (
        autograd_ctx.is_parallelism_context_initialized() and autograd_ctx.get_parallelism_context().is_ddp_enabled()
    )
    fsdp_sync_axes = tuple(name for name in ("dp", "fsdp") if axis_size(name) > 1)
    ddp_enabled = ddp_context_enabled or "dp" in fsdp_sync_axes
    fsdp_enabled = "fsdp" in fsdp_sync_axes
    batch_sharded = ddp_enabled or fsdp_enabled

    # Must equal the divisor the matching gradient sync averages by (see _compute_grpo_loss).
    if fsdp_enabled:
        grad_sync_world_size = math.prod(axis_size(name) for name in fsdp_sync_axes)
    elif ddp_context_enabled:
        # synchronize_gradients also divides by the CP size; CP cannot be enabled from Python.
        grad_sync_world_size = autograd_ctx.get_parallelism_context().get_ddp_size()
    else:
        grad_sync_world_size = axis_size("dp")

    if ddp_context_enabled:
        ddp_axis = autograd_ctx.get_parallelism_context().get_ddp_axis()
        seed_axes = [int(ddp_axis)] if ddp_axis is not None else None
    else:
        seed_axes = [mesh.axis_index(name) for name in fsdp_sync_axes] or None

    return BatchLayout(
        num_shards=device.get_num_devices() if batch_sharded else 1,
        mapper=ttml.core.distributed.shard_tensor_to_mesh_mapper(device, 0) if batch_sharded else None,
        composer=ttml.core.distributed.concat_mesh_to_tensor_composer(device, 0) if batch_sharded else None,
        seed_axes=seed_axes,
        ddp_context_enabled=ddp_context_enabled,
        ddp_enabled=ddp_enabled,
        fsdp_enabled=fsdp_enabled,
        fsdp_sync_axes=fsdp_sync_axes,
        grad_sync_world_size=grad_sync_world_size,
    )


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
