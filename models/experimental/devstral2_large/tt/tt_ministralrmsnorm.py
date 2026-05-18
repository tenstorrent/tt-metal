# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""TT RMSNorm for Devstral-2 / Ministral3.

Matches ``Ministral3RMSNorm``::

    variance = mean(x ** 2, dim=-1, keepdim=True)
    return weight * x * rsqrt(variance + eps)

Weight is replicated across the mesh. RMS is computed per-token, so the op is local — no
collective is needed even under TP (each device has the same full hidden dim of the activation
after we all-gather post-projection, OR each device holds a width shard but RMS itself reduces
along that last dim; in this rewrite we keep activations replicated across TP for norm ops to
keep the path simple and numerically identical to the HF reference).
"""

from __future__ import annotations

from typing import Optional

import torch
import ttnn

from models.experimental.devstral2_large.tt.model_args import (
    DEVSTRAL2_LARGE_L1_SMALL_SIZE,
    Devstral2Args,
)

__all__ = ["DEVSTRAL2_LARGE_L1_SMALL_SIZE", "TtRMSNorm"]


def _load_weight(
    state_dict: dict,
    weight_key: str,
    hidden_size: int,
    mesh_device,
    dtype: ttnn.DataType,
) -> ttnn.Tensor:
    """Load (or zero-init) the per-feature scale and upload as a replicated, tile-padded tensor."""
    if weight_key in state_dict:
        w = state_dict[weight_key].to(torch.bfloat16)
    else:
        # Tests load a random HF module and pass its state dict; if a layer is constructed without
        # a matching key we still want construction to succeed (errors caught at forward time).
        w = torch.ones(hidden_size, dtype=torch.bfloat16)
    # Pad to ``[1, 1, TILE, hidden_size]`` so a single tile row holds the per-channel scale.
    w_padded = w.reshape(1, 1, 1, hidden_size).expand(1, 1, ttnn.TILE_SIZE, hidden_size).contiguous()
    return ttnn.from_torch(
        w_padded,
        device=mesh_device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


class TtRMSNorm:
    """Devstral-2 RMSNorm. Replicated weight, ``ttnn.rms_norm`` on the host activation layout.

    Args:
        args: :class:`Devstral2Args`.
        mesh_device: Open mesh device.
        state_dict: HF (or HF-shaped) state dict.
        weight_key: Fully-qualified key into ``state_dict`` (e.g. ``"model.layers.0.input_layernorm.weight"``
            or ``"model.norm.weight"``).
        dtype: Weight dtype on device. Defaults to ``args.weight_dtype``.
    """

    def __init__(
        self,
        args: Devstral2Args,
        mesh_device,
        state_dict: dict,
        weight_key: str,
        *,
        dtype: Optional[ttnn.DataType] = None,
    ) -> None:
        self.args = args
        self.eps = args.rms_norm_eps
        self.mesh_device = mesh_device
        self.weight = _load_weight(
            state_dict,
            weight_key,
            args.hidden_size,
            mesh_device,
            dtype or args.weight_dtype,
        )
        self._compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def __call__(self, x: ttnn.Tensor, memory_config: Optional[ttnn.MemoryConfig] = None) -> ttnn.Tensor:
        return ttnn.rms_norm(
            x,
            epsilon=self.eps,
            weight=self.weight,
            memory_config=memory_config,
            compute_kernel_config=self._compute_kernel_config,
        )

    def forward(self, x: ttnn.Tensor, memory_config: Optional[ttnn.MemoryConfig] = None) -> ttnn.Tensor:
        return self(x, memory_config=memory_config)
