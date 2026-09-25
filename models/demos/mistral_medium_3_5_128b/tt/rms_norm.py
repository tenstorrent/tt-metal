# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""RMSNorm on the SP x TP sharded residual stream.

Residual layout (minimax_m3 ``tt/residual.py``, sharded scheme): each chip holds
``[1, 1, s_local, hidden / tp]``. The norm feeds column-parallel projections that need the full hidden
width, so it returns ``[1, 1, s_local, hidden]`` replicated across the TP cols.

M3's default ("gather_first": all-gather, then one ``ttnn.rms_norm`` over the full width) was measured at
hidden 6144; at 12288 the single-pass kernel's circular buffers need 3.36 MB of a 1.57 MB L1. So this is
the distributed form on the ``hidden/tp`` = 3072 shard: ``rms_norm_pre_all_gather`` (sum x^2, kept in
fp32) -> TP all-gather of the per-row stats -> ``rms_norm_post_all_gather`` with the TP-sharded gain ->
one TP all-gather of the normed shard to full width.
"""

import torch

import ttnn

from .common import cache_name, compute_config


class RMSNorm:
    def __init__(self, mesh_device, mesh_config, ccl_manager, weight, eps: float, *, tensor_cache_path=None):
        """``weight``: HF ``[hidden]`` gain (None when loading from ``tensor_cache_path``)."""
        self.mesh_device = mesh_device
        self.mesh_config = mesh_config
        self.ccl_manager = ccl_manager
        self.eps = eps
        torch_weight = None if weight is None else weight.to(torch.bfloat16).reshape(1, 1, -1, ttnn.TILE_SIZE)
        # Gain rows (32 channels each) split over TP, matching the residual's hidden split.
        self.weight = ttnn.as_tensor(
            torch_weight,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_config.mapper(mesh_device, tp_dim=2),
            cache_file_name=cache_name(tensor_cache_path, f"weight_tp{mesh_config.tp}"),
        )
        self.compute_kernel_config = compute_config(mesh_device, packer_l1_acc=False)

    def __call__(self, x):
        """``x`` sharded residual ``[1, 1, s_local, hidden/tp]`` -> normed ``[1, 1, s_local, hidden]``."""
        mc, ccl = self.mesh_config, self.ccl_manager
        stats = ttnn.rms_norm_pre_all_gather(x, compute_kernel_config=self.compute_kernel_config, dtype=ttnn.float32)
        gathered = mc.allgather(stats, ccl, axis=mc.tp_axis, dim=3)
        stats.deallocate(True)
        normed = ttnn.rms_norm_post_all_gather(
            x,
            gathered,
            epsilon=self.eps,
            weight=self.weight,
            compute_kernel_config=self.compute_kernel_config,
            dtype=ttnn.bfloat16,
        )
        gathered.deallocate(True)
        out = mc.allgather(normed, ccl, axis=mc.tp_axis, dim=3)
        normed.deallocate(True)
        return out
