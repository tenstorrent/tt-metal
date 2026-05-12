# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
PI0.5 PaliGemma backbone (TTNN).

VLM path is reused from PI0. The action expert uses AdaRMSGemmaBlockTTNN, and
the final expert norm is also adaRMS (`model.norm.dense.{weight,bias}`).

Expert checkpoint must contain, per layer:
  - input_layernorm.dense.{weight,bias}        (3*width, width)
  - post_attention_layernorm.dense.{weight,bias}
  - self_attn.{q,k,v,o}_proj.weight
  - mlp.{gate,up,down}_proj.weight
and at the top:
  - model.norm.dense.{weight,bias}
"""

from typing import Dict, List, Optional, Tuple

import torch
import ttnn

from models.experimental.pi0.common.configs import PaliGemmaConfig
from models.experimental.pi0.tt.ttnn_common import tensor_1d_to_2d_ttnn
from models.experimental.pi0.tt.ttnn_paligemma import PaliGemmaBackboneTTNN

from models.experimental.pi0_5.tt.ttnn_gemma import (
    AdaRMSGemmaBlockTTNN,
    ada_rms_norm_no_gate_ttnn,
)


def _convert_linear_to_ttnn(w: torch.Tensor, device, dtype=ttnn.bfloat16) -> "ttnn.Tensor":
    return ttnn.from_torch(w.T.contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


class Pi0_5PaliGemmaBackboneTTNN(PaliGemmaBackboneTTNN):
    """PaliGemma backbone with adaRMS action expert (TTNN)."""

    def __init__(
        self,
        config: PaliGemmaConfig,
        weights: Dict[str, Dict[str, torch.Tensor]],
        device: "ttnn.Device",
    ):
        # Parent reads plain `model.norm.weight` + plain `input_layernorm.weight`
        # tensors for the expert side. PI0.5 expert has neither — inject zero
        # placeholders so super().__init__ runs, then overwrite expert artifacts.
        ae = weights["action_expert"]
        expert_w = config.expert_config.width
        placeholders = {}
        for i in range(config.expert_config.depth):
            for name in ("input_layernorm.weight", "post_attention_layernorm.weight"):
                key = f"model.layers.{i}.{name}"
                if key not in ae:
                    placeholders[key] = torch.zeros(expert_w)
        if "model.norm.weight" not in ae:
            placeholders["model.norm.weight"] = torch.zeros(expert_w)
        ae.update(placeholders)

        super().__init__(config, weights, device)

        # Rebuild expert blocks with adaRMS, injecting modulation tensors.
        # adaRMS now fuses the (1+scale)/shift modulation into ttnn.rms_norm via
        # its weight/bias args — no separate "ones" identity weight needed.
        self.expert_blocks = []
        for i in range(config.expert_config.depth):
            block_weights = self._get_expert_block_weights_ttnn(weights["action_expert"], i)
            self._inject_adarms_weights(block_weights, weights["action_expert"], i)
            self.expert_blocks.append(
                AdaRMSGemmaBlockTTNN(
                    config.expert_config,
                    block_weights,
                    i,
                    device,
                    self.expert_cos_meta,
                    self.expert_sin_meta,
                )
            )

        # Final expert norm: adaRMS.
        self.expert_final_norm_mod_weight = _convert_linear_to_ttnn(
            weights["action_expert"]["model.norm.dense.weight"], device
        )
        if "model.norm.dense.bias" in weights["action_expert"]:
            self.expert_final_norm_mod_bias = tensor_1d_to_2d_ttnn(
                weights["action_expert"]["model.norm.dense.bias"], device, dtype=ttnn.bfloat16
            )
        else:
            self.expert_final_norm_mod_bias = None

        device_grid = device.compute_with_storage_grid_size()
        self.core_grid = ttnn.CoreGrid(y=device_grid.y, x=device_grid.x)

    def _inject_adarms_weights(
        self,
        block_weights: Dict[str, "ttnn.Tensor"],
        all_weights: Dict[str, torch.Tensor],
        layer_idx: int,
    ) -> None:
        """
        Fuse the pre-attention and pre-FFW modulation Dense projections into
        a single (in=W, out=6*W) linear per block (mirrors tt-dit's `norm1_linear`
        pattern). At forward time one matmul produces all 6 modulation tensors
        (scale_a, shift_a, gate_a, scale_f, shift_f, gate_f) for the block.

        Concatenation order on the output dim:
          [pre_attn.weight (3*W rows), pre_ffw.weight (3*W rows)]
        so slicing `[:W]` gives scale_a, `[W:2W]` shift_a, `[2W:3W]` gate_a,
        `[3W:4W]` scale_f, `[4W:5W]` shift_f, `[5W:6W]` gate_f.
        """
        prefix = f"model.layers.{layer_idx}."

        names = ("input_layernorm.dense", "post_attention_layernorm.dense")
        w_keys = [f"{prefix}{n}.weight" for n in names]
        b_keys = [f"{prefix}{n}.bias" for n in names]

        for wk in w_keys:
            if wk not in all_weights:
                raise KeyError(f"PI0.5 expects adaRMS weight '{wk}' in the action_expert checkpoint.")

        # Concat along output dim. Each input is (3*W, W); result is (6*W, W).
        fused_w_torch = torch.cat([all_weights[wk] for wk in w_keys], dim=0).contiguous()
        block_weights["adarms_mod.weight"] = _convert_linear_to_ttnn(fused_w_torch, self.device)

        biases = [all_weights[bk] for bk in b_keys if bk in all_weights]
        if biases:
            assert len(biases) == 2, "expected biases for both modulation Denses or neither"
            fused_b_torch = torch.cat(biases, dim=0).contiguous()
            block_weights["adarms_mod.bias"] = tensor_1d_to_2d_ttnn(fused_b_torch, self.device, dtype=ttnn.bfloat16)

    def forward_expert(
        self,
        hidden_states: "ttnn.Tensor",
        adarms_cond: "ttnn.Tensor",
        attention_mask: Optional["ttnn.Tensor"] = None,
        position_ids: Optional["ttnn.Tensor"] = None,
        past_key_values: Optional[List[Tuple["ttnn.Tensor", "ttnn.Tensor"]]] = None,
        use_cache: bool = False,
    ) -> Tuple["ttnn.Tensor", Optional[List[Tuple["ttnn.Tensor", "ttnn.Tensor"]]]]:
        new_cache = [] if use_cache else None

        for i, block in enumerate(self.expert_blocks):
            past_kv = past_key_values[i] if past_key_values else None
            hidden_states, new_kv = block.forward(
                hidden_states,
                None,
                None,
                adarms_cond,
                attention_mask,
                position_ids,
                past_kv,
                use_cache,
            )
            if use_cache:
                new_cache.append(new_kv)

        hidden_states = ada_rms_norm_no_gate_ttnn(
            hidden_states,
            adarms_cond,
            self.expert_final_norm_mod_weight,
            self.expert_final_norm_mod_bias,
            self.config.expert_config.rms_norm_eps,
            self.core_grid,
        )
        return hidden_states, new_cache
