# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
PI0.5 PaliGemma backbone (PyTorch reference).

VLM path is identical to PI0. The action expert uses AdaRMSGemmaBlock and the
final expert norm is also adaRMS (`model.norm.dense.{weight,bias}`).
"""

from typing import Dict, List, Optional, Tuple

import torch

from models.experimental.pi0.common.configs import PaliGemmaConfig
from models.experimental.pi0.reference.torch_paligemma import PaliGemmaBackbone
from models.experimental.pi0.reference.torch_gemma import precompute_freqs_cis

from models.experimental.pi0_5.reference.torch_gemma import AdaRMSGemmaBlock, ada_rms_norm_no_gate


class Pi0_5PaliGemmaBackbone(PaliGemmaBackbone):
    def __init__(self, config: PaliGemmaConfig, weights: Dict[str, torch.Tensor]):
        # The parent builds plain GemmaBlocks for the expert and reads
        # `input_layernorm.weight` / `post_attention_layernorm.weight` /
        # `model.norm.weight` from `weights["action_expert"]`. PI0.5 uses
        # adaRMS everywhere in the expert and those tensors don't exist in
        # the checkpoint. Inject zero placeholders so super().__init__ runs,
        # then discard the parent's expert artifacts.
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

        super().__init__(config, weights)

        # Replace expert blocks with adaRMS variants.
        self.expert_blocks = []
        for i in range(config.expert_config.depth):
            block_weights = self._get_block_weights(weights["action_expert"], i)
            self.expert_blocks.append(AdaRMSGemmaBlock(config.expert_config, block_weights, i))

        # Final expert norm: adaRMS, not plain.
        self.expert_norm_mod_weight = ae["model.norm.dense.weight"]
        self.expert_norm_mod_bias = ae.get("model.norm.dense.bias")

    def forward_expert(
        self,
        hidden_states: torch.Tensor,
        adarms_cond: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        new_cache = [] if use_cache else None

        cos, sin = precompute_freqs_cis(
            self.config.expert_config.head_dim,
            self.config.max_seq_len,
            self.config.expert_config.rope_base,
        )

        for i, block in enumerate(self.expert_blocks):
            past_kv = past_key_values[i] if past_key_values else None
            hidden_states, new_kv = block.forward(
                hidden_states,
                cos,
                sin,
                adarms_cond,
                attention_mask,
                position_ids,
                past_kv,
                use_cache,
            )
            if use_cache:
                new_cache.append(new_kv)

        hidden_states = ada_rms_norm_no_gate(
            hidden_states,
            adarms_cond,
            self.expert_norm_mod_weight,
            self.expert_norm_mod_bias,
            self.config.expert_config.rms_norm_eps,
        )
        return hidden_states, new_cache
