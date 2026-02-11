# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Bark Small - Text-to-Semantic stage (Stage 1).

Takes text tokens (BERT tokenizer) and generates semantic tokens.
Uses causal attention with 10k semantic vocab output.

HF prefix: "semantic"
Input vocab: 129,600 (BERT tokenizer space + special tokens)
Output vocab: 10,048 (semantic token space)
"""

import torch
import ttnn

from models.demos.wormhole.bark.tt.bark_gpt import TtBarkGPT
from models.demos.wormhole.bark.tt.common import BarkConfig


# Semantic EOS token from Bark
SEMANTIC_PAD_TOKEN = 10_000
SEMANTIC_INFER_TOKEN = 129_595


class TtBarkSemanticModel:
    """Stage 1: Text → Semantic Tokens using a causal GPT."""

    def __init__(self, state_dict: dict, device: ttnn.Device, config: BarkConfig):
        self.device = device
        self.config = config

        self.gpt = TtBarkGPT(
            state_dict=state_dict,
            prefix="semantic",
            device=device,
            config=config,
            input_vocab_size=config.semantic_input_vocab_size,
            output_vocab_size=config.semantic_output_vocab_size,
            is_causal=True,
        )

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 768,
        temperature: float = 0.7,
        top_k: int = 50,
        min_eos_p: float = 0.2,
    ) -> torch.Tensor:
        """
        Autoregressively generate semantic tokens from text tokens.

        Args:
            input_ids: BERT-tokenized text [1, seq_len]
            max_new_tokens: Maximum number of semantic tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            min_eos_p: Minimum EOS probability to stop early

        Returns:
            semantic_tokens: Generated semantic token IDs [1, total_len]
        """
        generated = input_ids.clone()
        if len(generated.shape) == 1:
            generated = generated.unsqueeze(0)

        for _ in range(max_new_tokens):
            # Truncate to max context
            context = generated[:, -self.config.block_size :]

            # Forward pass
            logits = self.gpt(context)
            logits = ttnn.to_torch(logits).to(torch.float32)

            # Get last token logits: [1, 1, seq_len, vocab] -> [1, vocab]
            next_token_logits = logits[0, 0, -1, :].unsqueeze(0)

            # Check EOS probability before temperature scaling
            probs_raw = torch.nn.functional.softmax(next_token_logits, dim=-1)
            if SEMANTIC_PAD_TOKEN < probs_raw.shape[-1]:
                eos_prob = probs_raw[0, SEMANTIC_PAD_TOKEN].item()
                if eos_prob > min_eos_p:
                    break

            # Temperature scaling
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

            # Sample
            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=-1)

            # Check for EOS token
            if next_token.item() == SEMANTIC_PAD_TOKEN:
                break

        return generated
