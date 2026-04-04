# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Text generation for Qwen3-Coder-Next.

Implements prefill + decode loop for autoregressive generation.
Handles both DeltaNet recurrent states and GQA KV caches.

Reference: DeepSeek V3 generator at deepseek_v3/tt/generator.py
"""


import torch

from models.demos.qwen3_coder_next.tt.model import Qwen3CoderNextModel
from models.demos.qwen3_coder_next.tt.model_config import Qwen3CoderNextConfig


class Generator:
    """Text generation wrapper for Qwen3-Coder-Next.

    Handles the prefill/decode state management for both
    DeltaNet (recurrent state) and GQA (KV cache) layers.
    """

    def __init__(self, model: Qwen3CoderNextModel, config: Qwen3CoderNextConfig):
        self.model = model
        self.config = config

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        greedy: bool = False,
    ) -> torch.Tensor:
        """Generate tokens autoregressively.

        Args:
            input_ids: (batch, prompt_len) input token IDs.
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature (ignored if greedy=True).
            top_k: Top-K sampling parameter.
            top_p: Top-P (nucleus) sampling parameter.
            greedy: If True, use greedy decoding (argmax).

        Returns:
            Generated token IDs (batch, prompt_len + max_new_tokens).
        """
        batch_size, prompt_len = input_ids.shape
        generated = input_ids.clone()

        # Prefill: process the entire prompt
        logits, layer_states = self.model(input_ids)

        # Get next token from last position
        next_token_logits = logits[:, -1, :]
        next_token = self._sample(next_token_logits, temperature, top_k, top_p, greedy)
        generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)

        # Decode: generate one token at a time
        for _ in range(max_new_tokens - 1):
            # Forward with just the new token, passing previous states
            position_ids = torch.tensor([[generated.shape[1] - 1]] * batch_size)
            logits, layer_states = self.model(
                next_token.unsqueeze(1),
                position_ids=position_ids,
                layer_states=layer_states,
            )

            next_token_logits = logits[:, -1, :]
            next_token = self._sample(next_token_logits, temperature, top_k, top_p, greedy)
            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)

        return generated

    def _sample(
        self,
        logits: torch.Tensor,
        temperature: float,
        top_k: int,
        top_p: float,
        greedy: bool,
    ) -> torch.Tensor:
        """Sample next token from logits.

        Args:
            logits: (batch, vocab_size) unnormalized log-probabilities.
            temperature: Sampling temperature.
            top_k: Top-K filtering.
            top_p: Top-P nucleus filtering.
            greedy: Use argmax instead of sampling.

        Returns:
            Selected token IDs (batch,).
        """
        if greedy:
            return logits.argmax(dim=-1)

        # Temperature scaling
        if temperature != 1.0:
            logits = logits / temperature

        # Top-K filtering
        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float("-inf")

        # Top-P filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float("-inf")

        # Sample from distribution
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)
