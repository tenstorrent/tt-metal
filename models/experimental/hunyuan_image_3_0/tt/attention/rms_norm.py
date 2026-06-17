# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# TTNN implementation of HunyuanRMSNorm.
# Thin wrapper around models/common/rmsnorm.py (shared with tt_transformers).

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.rmsnorm import RMSNorm
from models.tt_transformers.tt.common import Mode


class HunyuanTtRMSNorm(LightweightModule):
    """
    Single-device TTNN RMSNorm for HunyuanImage-3.0.

    Args:
        device:           TTNN device.
        dim:              Hidden size (e.g. 4096 or head_dim for QK norm).
        state_dict:       Model state_dict (plain torch tensors).
        weight_key:       Key of the weight tensor in state_dict, with or without
                          the ``.weight`` suffix, e.g.
                          ``model.layers.0.input_layernorm`` or
                          ``model.layers.0.input_layernorm.weight``.
        eps:              Variance epsilon (default 1e-5).
        weight_dtype:     TTNN dtype for the weight tensor (default bfloat16).
        weight_cache_path: Optional pathlib.Path for weight caching.
    """

    def __init__(
        self,
        device,
        dim: int,
        state_dict: dict,
        weight_key: str,
        eps: float = 1e-5,
        weight_dtype=ttnn.bfloat16,
        weight_cache_path=None,
    ):
        super().__init__()
        key = weight_key.removesuffix(".weight")
        # Norm weights are 1-D (loaded ROW_MAJOR) and precision-sensitive; bf8/bf4
        # are TILE-only and inappropriate here. Keep norms in bf16 even when the
        # backbone runs bf8 matmul weights (mixed precision — see MEMORY_FIT_PLAN.md).
        if weight_dtype in (ttnn.bfloat8_b, ttnn.bfloat4_b):
            weight_dtype = ttnn.bfloat16
        self._norm = RMSNorm(
            device=device,
            dim=dim,
            state_dict=state_dict,
            weight_key=key,
            layer_num=None,
            eps=eps,
            weight_dtype=weight_dtype,
            weight_cache_path=weight_cache_path,
            fp32_dest_acc_en=True,
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """
        Args:
            x: TTNN tensor, shape [B, S, H] or [B, heads, S, head_dim] in TILE_LAYOUT.
        Returns:
            Normalised tensor, same shape and memory config as input.
        """
        return self._norm(x, mode=Mode.PREFILL)
