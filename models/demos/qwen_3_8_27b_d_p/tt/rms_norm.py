# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""RMSNorm — the decoder norms, the final norm, and the per-head QK-norm.

One class covers all three because Qwen3.5 uses the same form everywhere: the Gemma
``out = x_normed * (1 + weight)`` fold, which is folded into the gain at load so the device runs a
plain ``ttnn.rms_norm``. The per-head QK-norm is the same op at ``width = head_dim`` on a
head-split ``[1, n_heads, S, head_dim]`` tensor: head_dim is not TP-sharded, so the reduction is
local to each chip and no collective is involved.

The Gated DeltaNet's output norm is a DIFFERENT function (plain gain, silu gate) and lives in
``gdn/operations.py``; it is not this class with a flag.
"""

from __future__ import annotations

from typing import Optional

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule

from ..config import MeshConfig
from ..utils.general_utils import get_cache_file_name
from .compute import matmul_compute_config


class RMSNorm(LightweightModule):
    def __init__(
        self,
        mesh_device,
        hidden_size: int,
        eps: float,
        state_dict: dict[str, torch.Tensor],
        *,
        mesh_config: MeshConfig,
        tensor_cache_path: Optional[str] = None,
        gemma_fold: bool = True,
    ) -> None:
        """``gemma_fold``: add 1 to the gain at load (Qwen3.5's ``(1 + weight)`` form). Always True
        for this model; the parameter exists so the fold is visible at every call site rather than
        buried, and so a plain-RMSNorm reference can be tested against the same class."""
        self.mesh_device = mesh_device
        self.mesh_config = mesh_config
        self.hidden_size = hidden_size
        self.eps = eps
        self.gemma_fold = gemma_fold
        # HiFi4 + fp32 accumulation here too, not just on the matmuls: this norm reads the
        # residual stream, so a low-fidelity sum of 5120 squares is an error injected into
        # every block below it and compounded by all 64 layers.
        self.compute_config = matmul_compute_config(mesh_device)

        torch_weight = None
        if state_dict:
            weight = state_dict["weight"]
            if gemma_fold:
                # In fp32, BEFORE the bf16 cast: folding after the cast loses the bits the +1 shifts.
                weight = weight.float() + 1.0
            torch_weight = weight.reshape((1, 1, -1, ttnn.TILE_SIZE))

        self.tt_weight = ttnn.as_tensor(
            torch_weight,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            cache_file_name=get_cache_file_name(tensor_cache_path, "weight"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_config.replicate(mesh_device),
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.rms_norm(x, weight=self.tt_weight, epsilon=self.eps, compute_kernel_config=self.compute_config)
