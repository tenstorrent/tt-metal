# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Full Qwen3-Coder-Next transformer model.

48-layer hybrid transformer with:
- 36 DeltaNet (linear attention) layers
- 12 GQA (softmax attention) layers
- MoE FFN in every layer (512 experts, 10 active)
- Token embedding + final RMSNorm + LM head

Reference: DeepSeek V3 model at deepseek_v3/tt/model/row_batched_model.py
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from models.demos.qwen3_coder_next.tt.decoder_block import HybridDecoderBlock, RMSNorm
from models.demos.qwen3_coder_next.tt.embedding import Embedding
from models.demos.qwen3_coder_next.tt.gqa_attention import GQAAttention
from models.demos.qwen3_coder_next.tt.lm_head import LMHead
from models.demos.qwen3_coder_next.tt.model_config import Qwen3CoderNextConfig
from models.demos.qwen3_coder_next.tt.rope import PartialRoPE


class Qwen3CoderNextModel(nn.Module):
    """Full Qwen3-Coder-Next transformer.

    PyTorch reference implementation for correctness validation.
    Stacks 48 HybridDecoderBlock layers with proper attention type assignment.
    """

    def __init__(self, config: Qwen3CoderNextConfig):
        super().__init__()
        self.config = config

        # Token embedding
        self.embedding = Embedding(config)

        # Decoder layers
        self.layers = nn.ModuleList([HybridDecoderBlock(config, layer_idx=i) for i in range(config.num_hidden_layers)])

        # Final norm
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # LM head
        self.lm_head = LMHead(config)

        # RoPE (for GQA layers)
        self.rope = PartialRoPE(config, max_seq_len=min(config.max_position_embeddings, 8192))

        # Verify layer type distribution
        num_gqa = sum(1 for i in range(config.num_hidden_layers) if config.is_gqa_layer(i))
        num_deltanet = config.num_hidden_layers - num_gqa
        assert num_gqa == config.num_gqa_layers, f"Expected {config.num_gqa_layers} GQA layers, got {num_gqa}"
        assert num_deltanet == config.num_deltanet_layers

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        layer_states: Optional[List[dict]] = None,
    ) -> Tuple[torch.Tensor, List[dict]]:
        """Forward pass through the full model.

        Args:
            input_ids: (batch, seq_len) token IDs.
            attention_mask: Optional causal mask for GQA layers.
            position_ids: Optional position IDs for RoPE.
            layer_states: Optional list of per-layer states from previous forward pass
                (for incremental decode). Each dict has 'recurrent_state' or 'kv_cache'.

        Returns:
            Tuple of:
                - logits: (batch, seq_len, vocab_size)
                - new_layer_states: Updated per-layer states
        """
        batch_size, seq_len = input_ids.shape

        # Embed tokens
        hidden_states = self.embedding(input_ids)

        # Prepare RoPE frequencies
        cos, sin = self.rope.get_cos_sin(seq_len, position_ids)

        # Prepare causal mask for GQA layers
        if attention_mask is None and seq_len > 1:
            attention_mask = GQAAttention.make_causal_mask(seq_len, dtype=hidden_states.dtype)
            attention_mask = attention_mask.to(hidden_states.device)

        # Initialize layer states if not provided
        if layer_states is None:
            layer_states = [{}] * self.config.num_hidden_layers

        # Forward through all layers
        new_layer_states = []
        for i, layer in enumerate(self.layers):
            state = layer_states[i] if i < len(layer_states) else {}

            hidden_states, new_state = layer(
                hidden_states,
                cos=cos,
                sin=sin,
                attention_mask=attention_mask,
                recurrent_state=state.get("recurrent_state"),
                kv_cache=state.get("kv_cache"),
            )
            new_layer_states.append(new_state)

        # Final norm
        hidden_states = self.norm(hidden_states)

        # LM head
        logits = self.lm_head(hidden_states)

        return logits, new_layer_states

    def get_layer_type_summary(self) -> str:
        """Return a human-readable summary of layer types."""
        types = []
        for i in range(self.config.num_hidden_layers):
            if self.config.is_gqa_layer(i):
                types.append(f"Layer {i:2d}: GQA")
            else:
                types.append(f"Layer {i:2d}: DeltaNet")
        return "\n".join(types)
