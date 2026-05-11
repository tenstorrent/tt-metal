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
    _plain_rms_norm_weight,
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

        # Shared "ones" tile used as the plain-RMS weight for every adaRMS norm.
        self.ones_weight = _plain_rms_norm_weight(device, config.expert_config.width)

        # Rebuild expert blocks with adaRMS, injecting modulation tensors.
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
                    self.ones_weight,
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
        """Pull adaRMS modulation weights for layer_idx and add to block_weights."""
        prefix = f"model.layers.{layer_idx}."
        for name in ("input_layernorm.dense", "post_attention_layernorm.dense"):
            w_key = f"{prefix}{name}.weight"
            b_key = f"{prefix}{name}.bias"
            if w_key not in all_weights:
                raise KeyError(f"PI0.5 expects adaRMS weight '{w_key}' in the action_expert checkpoint.")
            block_weights[f"{name}.weight"] = _convert_linear_to_ttnn(all_weights[w_key], self.device)
            if b_key in all_weights:
                block_weights[f"{name}.bias"] = tensor_1d_to_2d_ttnn(
                    all_weights[b_key], self.device, dtype=ttnn.bfloat16
                )

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
            self.ones_weight,
            adarms_cond,
            self.expert_final_norm_mod_weight,
            self.expert_final_norm_mod_bias,
            self.config.expert_config.rms_norm_eps,
            self.core_grid,
        )
        return hidden_states, new_cache
