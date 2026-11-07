# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""One encoder block, the TTNN form of reference.NomicBertBlock.

Post-norm: norm1(attn(x) + x), then norm2(mlp(h) + h). The residual is added before the norm, so
every sub-block output is re-centred, which is why bfloat16 error does not compound across the
12 layers the way it does in a pre-norm decoder.

Both residual adds are fused into their layer norm via residual_input_tensor, so the two aten
add + native_layer_norm pairs become two norms rather than four ops. Up to SHARDED_NORM_DIRECT_MAX_TILES tile
rows of M the norms split the hidden axis over sharded copies (tt.common.layer_norm), and each norm's
sharded output is the next norm's residual as it stands: one conversion a norm fewer. Above
SMALL_M_TILES the output stays in that layout alone, which QKV, fc1 and the MoE take as it is.
"""

from __future__ import annotations

from typing import Optional

import ttnn

from models.common.lightweightmodule import LightweightModule
from models.experimental.nomic_embed_text_v2_moe.tt.attention import TtNomicBertAttention
from models.experimental.nomic_embed_text_v2_moe.tt.common import (
    LayerNormParameters,
    activation_memory_config,
    layer_norm,
)
from models.experimental.nomic_embed_text_v2_moe.tt.mlp import TtNomicBertMLP
from models.experimental.nomic_embed_text_v2_moe.tt.model_config import OpGroup
from models.experimental.nomic_embed_text_v2_moe.tt.moe import TtNomicMoELayer


class TtNomicBertBlock(LightweightModule):
    """Attention then an FFN, each followed by a fused residual-add layer norm.

    The FFN is dense or MoE depending on the layer index; the caller decides via moe, since the
    placement predicate belongs to the config rather than the block.

    Both norms write where activation_memory_config puts an activation of their size, the output
    where output_memory_config says when it is given.
    """

    def __init__(
        self,
        device,
        config,
        tt_config,
        state_dict,
        state_dict_prefix,
        moe: bool,
        output_memory_config: ttnn.MemoryConfig | None = None,
        rotation_matrix: ttnn.Tensor | None = None,
    ):
        super().__init__()
        self.tt_config = tt_config
        self.epsilon = config.layer_norm_epsilon
        self.output_memory_config = output_memory_config

        self.attn = TtNomicBertAttention(
            device, config, tt_config, state_dict, f"{state_dict_prefix}attn.", rotation_matrix
        )
        prefix = f"{state_dict_prefix}mlp."
        if moe:
            self.mlp = TtNomicMoELayer(device, config, tt_config, state_dict, prefix)
        else:
            self.mlp = TtNomicBertMLP(device, config, tt_config, state_dict, prefix)

        def norm(name):
            return LayerNormParameters(
                state_dict[f"{state_dict_prefix}{name}.weight"],
                state_dict[f"{state_dict_prefix}{name}.bias"],
                device,
                tt_config.weight_dtype,
            )

        self.norm1, self.norm2 = norm("norm1"), norm("norm2")

    def _norm(
        self,
        x: ttnn.Tensor,
        residual: ttnn.Tensor,
        parameters: LayerNormParameters,
        memory_config: ttnn.MemoryConfig | None = None,
        keep_sharded: bool = False,
        in_place: bool = True,
    ) -> ttnn.Tensor | tuple[ttnn.Tensor, ttnn.Tensor | None]:
        return layer_norm(
            x,
            parameters,
            residual=residual,
            epsilon=self.epsilon,
            memory_config=activation_memory_config(x.shape) if memory_config is None else memory_config,
            compute_kernel_config=self.tt_config.compute_kernel_config(OpGroup.NORM),
            keep_sharded=keep_sharded,
            # The sub-block output has no other reader.
            consume_input=True,
            in_place=in_place,
        )

    def forward(
        self,
        x: ttnn.Tensor,
        rot_mats: tuple[ttnn.Tensor, ttnn.Tensor],
        attn_mask: Optional[ttnn.Tensor] = None,
        free_input: bool = False,
        x_sharded: ttnn.Tensor | None = None,
        keep_sharded: bool = False,
    ) -> ttnn.Tensor | tuple[ttnn.Tensor, ttnn.Tensor | None]:
        """Run the block.

        Args:
            x: (B, 1, S, H) block input.
            rot_mats: (cos, sin), each (1, 1, S, D) or longer, from tt.common.RotaryTables.
            attn_mask: (B, 1, S, S) additive mask, or None.
            free_input: free x once norm1 has read it, the last use, rather than leave it to the caller.
                In L1 that keeps the residual stream near the top: x freed after the whole block leaves
                a hole above the block's output, which then lands deeper, and every later op plans its
                buffers below it. With x_sharded, attention is x's last reader.
            x_sharded: x in the split norms' layout, the previous norm's keep_sharded output, which norm1
                reads as its residual; freed here.
            keep_sharded: also return the output in that layout, for the next block's norm1.

        Returns:
            ttnn.Tensor: (B, 1, S, H), shape unchanged; with keep_sharded=True, (output, its split-norm
            layout copy or None above the split limit).
        """
        attn_out = self.attn(x, rot_mats, attn_mask)
        if x_sharded is not None and free_input:
            ttnn.deallocate(x)
        residual = x if x_sharded is None else x_sharded
        hidden, hidden_sharded = self._norm(attn_out, residual, self.norm1, keep_sharded=True)
        if x_sharded is not None:
            ttnn.deallocate(x_sharded)
        elif free_input:
            ttnn.deallocate(x)

        mlp_out = self.mlp(hidden)
        residual = hidden if hidden_sharded is None else hidden_sharded
        # Where a split norm's output lands decides the next plans (8x512): norm1 writes over attn_out, since a
        # new output below attn_out and the residual sent the routed w1 output to DRAM. After the MoE, mlp_out
        # sits below the freed routed intermediates, and a new output lands in their hole: written over
        # mlp_out, it left the next QKV 18 KB short of its one-M-block blocking and slowed the next SDPA.
        out = self._norm(
            mlp_out,
            residual,
            self.norm2,
            self.output_memory_config,
            keep_sharded=keep_sharded,
            in_place=not isinstance(self.mlp, TtNomicMoELayer),
        )
        ttnn.deallocate(hidden)
        if hidden_sharded is not None:
            ttnn.deallocate(hidden_sharded)
        return out
