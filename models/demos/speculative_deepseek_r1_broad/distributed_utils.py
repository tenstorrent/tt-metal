# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistContext:
    enabled: bool
    rank: int
    world_size: int
    local_rank: int


def init_distributed(tp_size: int, local_rank_arg: int = -1) -> DistContext:
    if tp_size <= 1:
        return DistContext(enabled=False, rank=0, world_size=1, local_rank=0)

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    local_rank = int(local_rank_arg if local_rank_arg >= 0 else os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != tp_size:
        raise RuntimeError(f"tp_size={tp_size} but world_size={world_size}. Launch with matching torchrun nproc.")
    return DistContext(enabled=True, rank=rank, world_size=world_size, local_rank=local_rank)


def is_main_process(ctx: DistContext) -> bool:
    return ctx.rank == 0


def barrier(ctx: DistContext) -> None:
    if ctx.enabled:
        dist.barrier()
