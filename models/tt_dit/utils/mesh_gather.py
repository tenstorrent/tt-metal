# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Host-side SP/TP gather helpers for LTX distributed tensors."""

from __future__ import annotations

import torch

import ttnn
from models.tt_dit.parallel.config import DiTParallelConfig


def gather_host_1bnd(
    tensor: ttnn.Tensor,
    *,
    ccl_manager,
    parallel_config: DiTParallelConfig,
    tp_already_gathered: bool = False,
    sp_already_gathered: bool = False,
) -> torch.Tensor:
    """All-gather SP/TP shards → float torch (1, 1, N, D)."""
    cap = tensor
    if not tp_already_gathered and parallel_config.tensor_parallel.factor > 1:
        cap = ccl_manager.all_gather_persistent_buffer(cap, dim=3, mesh_axis=parallel_config.tensor_parallel.mesh_axis)
    if not sp_already_gathered and parallel_config.sequence_parallel.factor > 1:
        cap = ccl_manager.all_gather_persistent_buffer(
            cap, dim=2, mesh_axis=parallel_config.sequence_parallel.mesh_axis
        )
    return ttnn.to_torch(ttnn.get_device_tensors(cap)[0]).float().clone()
