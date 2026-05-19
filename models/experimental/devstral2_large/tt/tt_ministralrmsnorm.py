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

from models.experimental.devstral2_large.tt.mem_config import get_compute_kernel_config
from models.experimental.devstral2_large.tt.model_args import (
    DEVSTRAL2_LARGE_L1_SMALL_SIZE,
    Devstral2Args,
)
from models.experimental.devstral2_large.tt.weight_loading import (
    NORM_WEIGHT_MEM_CONFIG,
    resolve_weight_cache_path,
    upload_replicated_tile,
)

__all__ = ["DEVSTRAL2_LARGE_L1_SMALL_SIZE", "TtRMSNorm"]


def _load_weight(
    state_dict: dict,
    weight_key: str,
    hidden_size: int,
    mesh_device,
    dtype: ttnn.DataType,
    *,
    weight_cache_path: Optional[str] = None,
) -> ttnn.Tensor:
    """Load (or zero-init) the per-feature scale and upload as a replicated, tile-padded tensor."""
    if weight_key in state_dict:
        w = state_dict[weight_key]
    else:
        w = torch.ones(hidden_size, dtype=torch.bfloat16)
    w_padded = w.reshape(1, 1, 1, hidden_size).expand(1, 1, ttnn.TILE_SIZE, hidden_size).contiguous()
    return upload_replicated_tile(
        w_padded,
        mesh_device,
        dtype=dtype,
        memory_config=NORM_WEIGHT_MEM_CONFIG,
        weight_cache_path=weight_cache_path,
        cache_key=weight_key,
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
        weight_cache_path: Optional[str] = None,
    ) -> None:
        self.args = args
        self.eps = args.rms_norm_eps
        self.mesh_device = mesh_device
        cache_path = resolve_weight_cache_path(weight_cache_path, args)
        self.weight = _load_weight(
            state_dict,
            weight_key,
            args.hidden_size,
            mesh_device,
            dtype or args.weight_dtype,
            weight_cache_path=cache_path,
        )
        self._compute_kernel_config = get_compute_kernel_config(mesh_device)

    def __call__(self, x: ttnn.Tensor, memory_config: Optional[ttnn.MemoryConfig] = None) -> ttnn.Tensor:
        out_mem = memory_config if memory_config is not None else ttnn.L1_MEMORY_CONFIG
        return ttnn.rms_norm(
            x,
            epsilon=self.eps,
            weight=self.weight,
            memory_config=out_mem,
            compute_kernel_config=self._compute_kernel_config,
        )

    def forward(self, x: ttnn.Tensor, memory_config: Optional[ttnn.MemoryConfig] = None) -> ttnn.Tensor:
        return self(x, memory_config=memory_config)
