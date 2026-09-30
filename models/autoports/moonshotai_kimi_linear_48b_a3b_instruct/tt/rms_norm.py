# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Replicated RMSNorm (pre-norm blocks and the final norm)."""

from __future__ import annotations

from pathlib import Path

import torch

import ttnn
from models.autoports.moonshotai_kimi_linear_48b_a3b_instruct.tt.weights import as_device_tensor


class RMSNorm:
    def __init__(self, mesh_device, weight: torch.Tensor | None, *, eps: float, name: str, cache_path: Path | None):
        dim = None if weight is None else weight.numel()
        self.weight = as_device_tensor(
            mesh_device,
            None if weight is None else weight.reshape(1, 1, 1, -1),
            name=name,
            dtype=ttnn.bfloat16,
            shard_dim=None,
            cache_path=cache_path,
        )
        self.eps = eps
        self.compute = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.rms_norm(x, epsilon=self.eps, weight=self.weight, compute_kernel_config=self.compute)
