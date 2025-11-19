# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
PyTorch reference implementation of ConditionalChatTTS Decoder for PCC validation.

Simplified implementation focusing on the core transformer decoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


class PyTorchChatTTSDecoder(nn.Module):
    """
    Simplified PyTorch reference implementation of ChatTTS decoder.

    Focuses on core transformer decoder functionality for PCC validation.
    """

    def __init__(
        self,
        llm_dim: int = 3584,
        hidden_size: int = 768,
        num_attention_heads: int = 12,
        num_hidden_layers: int = 20,
        intermediate_size: int = 3072,
        num_text_tokens: int = 21178,
        num_audio_tokens: int = 626,
        num_vq: int = 4,
        num_spk_embs: int = 1,
    ):
        super().__init__()
        self.llm_dim = llm_dim
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.num_text_tokens = num_text_tokens
        self.num_audio_tokens = num_audio_tokens
        self.num_vq = num_vq
        self.num_spk_embs = num_spk_embs

        # LLM projector (MLP)
        self.projector = nn.Sequential(
            nn.Linear(llm_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        # Embeddings
        self.emb_text = nn.Embedding(num_text_tokens, hidden_size)

        # Audio code embeddings (4 codebooks)
        self.emb_code = nn.ModuleList([nn.Embedding(num_audio_tokens, hidden_size) for _ in range(num_vq)])

        # Transformer decoder (simplified Llama-style)
        self.layers = nn.ModuleList(
            [
                PyTorchTransformerLayer(hidden_size, num_attention_heads, intermediate_size)
                for _ in range(num_hidden_layers)
            ]
        )

        # Final layer norm
        self.norm = nn.LayerNorm(hidden_size)

        # Output heads (weight normalized)
        self.head_code = nn.ModuleList(
            [nn.utils.weight_norm(nn.Linear(hidden_size, num_audio_tokens, bias=False)) for _ in range(num_vq)]
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        lm_spk_emb_last_hidden_states: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """
        Forward pass.

        Args:
            input_ids: [batch_size, seq_len, num_vq]
            attention_mask: [batch_size, seq_len]
            position_ids: [batch_size, seq_len]
            lm_spk_emb_last_hidden_states: [batch_size, num_spk_embs, llm_dim]

        Returns:
            List[torch.Tensor]: Logits for each codebook [batch_size, seq_len, num_audio_tokens]
        """
        # Create embeddings
        inputs_embeds = self._create_embeddings(input_ids, lm_spk_emb_last_hidden_states)

        # Apply transformer layers
        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)

        # Final layer norm
        hidden_states = self.norm(hidden_states)

        # Output heads
        logits = []
        for head in self.head_code:
            logit = head(hidden_states)
            logits.append(logit)

        return logits

    def _create_embeddings(
        self,
        input_ids: torch.Tensor,
        lm_spk_emb_last_hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Create input embeddings.
        """
        batch_size, seq_len, _ = input_ids.shape

        # Text embeddings (simplified - use first codebook)
        text_ids = input_ids[:, :, 0]
        inputs_embeds = self.emb_text(text_ids)

        # Add speaker embeddings if provided
        if lm_spk_emb_last_hidden_states is not None:
            projected_spk_emb = self.projector(lm_spk_emb_last_hidden_states)
            projected_spk_emb = F.normalize(projected_spk_emb, p=2, dim=-1)

            # Simplified: add mean speaker embedding
            inputs_embeds = inputs_embeds + projected_spk_emb.mean(dim=1, keepdim=True)

        return inputs_embeds


class PyTorchTransformerLayer(nn.Module):
    """
    Simplified transformer layer (Llama-style).
    """

    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        # Self-attention
        self.self_attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)

        # Layer norms
        self.input_layernorm = nn.LayerNorm(hidden_size)
        self.post_attention_layernorm = nn.LayerNorm(hidden_size)

        # MLP (Llama-style: gate_proj, up_proj, down_proj)
        self.mlp = nn.ModuleDict(
            {
                "gate_proj": nn.Linear(hidden_size, intermediate_size),
                "up_proj": nn.Linear(hidden_size, intermediate_size),
                "down_proj": nn.Linear(intermediate_size, hidden_size),
            }
        )

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of transformer layer.
        """
        # Input layer norm
        normed_hidden = self.input_layernorm(hidden_states)

        # Self-attention
        attn_output, _ = self.self_attn(
            normed_hidden, normed_hidden, normed_hidden, attn_mask=attention_mask, need_weights=False
        )

        # Residual connection
        hidden_states = hidden_states + attn_output

        # Post-attention layer norm
        normed_hidden = self.post_attention_layernorm(hidden_states)

        # MLP (Llama-style: gate_proj -> SiLU -> up_proj -> mul -> down_proj)
        gate = self.mlp["gate_proj"](normed_hidden)
        gate = F.silu(gate)
        up = self.mlp["up_proj"](normed_hidden)
        mlp_hidden = gate * up  # Element-wise multiplication
        mlp_output = self.mlp["down_proj"](mlp_hidden)

        # Residual connection
        hidden_states = hidden_states + mlp_output

        return hidden_states
