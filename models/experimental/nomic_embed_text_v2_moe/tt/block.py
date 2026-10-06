# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""One encoder block, the TTNN form of reference.NomicBertBlock.

Post-norm: norm1(attn(x) + x), then norm2(mlp(h) + h). The residual is added before the norm, so
every sub-block output is re-centred, which is why bfloat16 error does not compound across the
12 layers the way it does in a pre-norm decoder.

Both residual adds are fused into their layer norm via residual_input_tensor, so the two aten
add + native_layer_norm pairs become two device ops rather than four.
"""

from __future__ import annotations

from typing import Optional

import ttnn

from models.common.lightweightmodule import LightweightModule
from models.experimental.nomic_embed_text_v2_moe.tt.attention import TtNomicBertAttention
from models.experimental.nomic_embed_text_v2_moe.tt.common import LayerNormParameters
from models.experimental.nomic_embed_text_v2_moe.tt.mlp import TtNomicBertMLP
from models.experimental.nomic_embed_text_v2_moe.tt.model_config import OpGroup
from models.experimental.nomic_embed_text_v2_moe.tt.moe import TtNomicMoELayer


class TtNomicBertBlock(LightweightModule):
    """Attention then an FFN, each followed by a fused residual-add layer norm.

    The FFN is dense or MoE depending on the layer index; the caller decides via moe, since the
    placement predicate belongs to the config rather than the block.
    """

    def __init__(self, device, config, tt_config, state_dict, state_dict_prefix, moe: bool):
        super().__init__()
        self.tt_config = tt_config
        self.epsilon = config.layer_norm_epsilon

        self.attn = TtNomicBertAttention(device, config, tt_config, state_dict, f"{state_dict_prefix}attn.")
        ffn = TtNomicMoELayer if moe else TtNomicBertMLP
        self.mlp = ffn(device, config, tt_config, state_dict, f"{state_dict_prefix}mlp.")

        def norm(name):
            return LayerNormParameters(
                state_dict[f"{state_dict_prefix}{name}.weight"],
                state_dict[f"{state_dict_prefix}{name}.bias"],
                device,
                tt_config.weight_dtype,
            )

        self.norm1, self.norm2 = norm("norm1"), norm("norm2")

    def _norm(self, x: ttnn.Tensor, residual: ttnn.Tensor, parameters: LayerNormParameters) -> ttnn.Tensor:
        # The attention and fc2 outputs may sit in L1 (dense_linear); the block's activations stay
        # in DRAM.
        weight, bias = parameters.for_input(x)
        return ttnn.layer_norm(
            x,
            residual_input_tensor=residual,
            weight=weight,
            bias=bias,
            epsilon=self.epsilon,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.tt_config.compute_kernel_config(OpGroup.NORM),
        )

    def forward(
        self,
        x: ttnn.Tensor,
        rot_mats: tuple[ttnn.Tensor, ttnn.Tensor],
        attn_mask: Optional[ttnn.Tensor] = None,
    ) -> ttnn.Tensor:
        """Run the block.

        Args:
            x: (B, 1, S, H) block input.
            rot_mats: (cos, sin), each (1, 1, S, D) or longer, from tt.common.RotaryTables.
            attn_mask: (B, 1, S, S) additive mask, or None.

        Returns:
            ttnn.Tensor: (B, 1, S, H), shape unchanged.
        """
        attn_out = self.attn(x, rot_mats, attn_mask)
        hidden = self._norm(attn_out, x, self.norm1)
        ttnn.deallocate(attn_out)

        mlp_out = self.mlp(hidden)
        out = self._norm(mlp_out, hidden, self.norm2)
        ttnn.deallocate(mlp_out)
        ttnn.deallocate(hidden)
        return out
