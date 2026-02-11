# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Bark Small - Semantic-to-Coarse stage (Stage 2).

Takes semantic tokens and generates coarse EnCodec tokens (2 codebooks).
Uses causal attention.

HF prefix: "coarse_acoustics"
Input/Output vocab: 12,096
"""

import torch
import ttnn

from models.demos.wormhole.bark.tt.bark_gpt import TtBarkGPT
from models.demos.wormhole.bark.tt.common import BarkConfig

# Special tokens for coarse generation
COARSE_SEMANTIC_PAD_TOKEN = 12_048
COARSE_INFER_TOKEN = 12_050


class TtBarkCoarseModel:
    """Stage 2: Semantic Tokens → Coarse EnCodec Tokens (2 codebooks)."""

    def __init__(self, state_dict: dict, device: ttnn.Device, config: BarkConfig):
        self.device = device
        self.config = config

        self.gpt = TtBarkGPT(
            state_dict=state_dict,
            prefix="coarse_acoustics",
            device=device,
            config=config,
            input_vocab_size=config.coarse_input_vocab_size,
            output_vocab_size=config.coarse_output_vocab_size,
            is_causal=True,
        )

    def generate(
        self,
        semantic_tokens: torch.Tensor,
        max_new_tokens: int = 768,
        temperature: float = 0.7,
        top_k: int = 50,
    ) -> torch.Tensor:
        """
        Autoregressively generate coarse acoustic tokens from semantic tokens.

        The coarse model alternates between predicting codebook 0 and codebook 1
        tokens. Generation produces interleaved tokens for the 2 codebooks.

        Args:
            semantic_tokens: Semantic token IDs from Stage 1 [1, seq_len]
            max_new_tokens: Maximum number of coarse tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter

        Returns:
            coarse_tokens: Generated coarse tokens (interleaved 2 codebooks) [1, total_len]
        """
        generated = semantic_tokens.clone()
        if len(generated.shape) == 1:
            generated = generated.unsqueeze(0)

        for step in range(max_new_tokens):
            context = generated[:, -self.config.block_size :]

            logits = self.gpt(context)
            logits = ttnn.to_torch(logits).to(torch.float32)

            # Get last token logits
            next_token_logits = logits[0, 0, -1, :].unsqueeze(0)
            next_token_logits = next_token_logits / temperature

            # Top-k filtering
            if top_k > 0:
                top_k_clamped = min(top_k, next_token_logits.shape[-1])
                top_k_values, _ = torch.topk(next_token_logits, top_k_clamped)
                min_top_k = top_k_values[:, -1].unsqueeze(-1)
                next_token_logits = torch.where(
                    next_token_logits < min_top_k,
                    torch.full_like(next_token_logits, float("-inf")),
                    next_token_logits,
                )

            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=-1)

            # Check for special end token
            if next_token.item() == COARSE_SEMANTIC_PAD_TOKEN:
                break

        return generated
