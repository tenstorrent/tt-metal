# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Bark Small - Coarse-to-Fine stage (Stage 3).

Takes coarse tokens (2 codebooks) and generates all 8 fine EnCodec codebooks.
Uses NON-causal attention (bidirectional, can see full context).

HF prefix: "fine_acoustics"
Input/Output vocab: 1,056 (per codebook)

Unlike stages 1-2, this model:
- Has bias=True for LayerNorm
- Uses non-causal (full) attention
- Has multiple embedding layers (one per codebook)
- Predicts remaining 6 codebooks iteratively
"""

import torch
import ttnn

from models.demos.wormhole.bark.tt.bark_gpt import TtBarkGPT
from models.demos.wormhole.bark.tt.common import BarkConfig, load_tt_tensor


class TtBarkFineModel:
    """Stage 3: Coarse Tokens (2 codebooks) → Fine Tokens (8 codebooks)."""

    def __init__(self, state_dict: dict, device: ttnn.Device, config: BarkConfig):
        self.device = device
        self.config = config
        self.n_fine_codebooks = config.n_fine_codebooks
        self.n_coarse_codebooks = config.n_coarse_codebooks

        # The fine model has a separate GPT but with non-causal attention
        self.gpt = TtBarkGPT(
            state_dict=state_dict,
            prefix="fine_acoustics",
            device=device,
            config=config,
            input_vocab_size=config.fine_input_vocab_size,
            output_vocab_size=config.fine_output_vocab_size,
            is_causal=False,  # Stage 3 is non-causal!
        )

    def generate(
        self,
        coarse_tokens: torch.Tensor,
        temperature: float = 0.5,
    ) -> torch.Tensor:
        """
        Generate fine acoustic tokens from coarse tokens.

        Unlike stages 1-2, this iteratively predicts codebooks 3-8
        given the existing codebooks. Each pass adds one codebook.

        Args:
            coarse_tokens: Coarse codebook tokens [n_coarse_codebooks, seq_len]
            temperature: Sampling temperature

        Returns:
            fine_tokens: All 8 codebook tokens [n_fine_codebooks, seq_len]
        """
        seq_len = coarse_tokens.shape[-1]

        # Start with coarse codebooks
        all_codebooks = coarse_tokens.clone()
        if len(all_codebooks.shape) == 1:
            all_codebooks = all_codebooks.unsqueeze(0)

        # Generate remaining codebooks (3 through 8)
        for codebook_idx in range(self.n_coarse_codebooks, self.n_fine_codebooks):
            # Flatten codebooks into a single sequence for the model
            input_ids = all_codebooks.reshape(1, -1)

            # Truncate if too long
            if input_ids.shape[-1] > self.config.block_size:
                input_ids = input_ids[:, -self.config.block_size :]

            logits = self.gpt(input_ids)
            logits = ttnn.to_torch(logits).to(torch.float32)

            # Get logits for the last seq_len positions (corresponding to new codebook)
            cb_logits = logits[0, 0, -seq_len:, :]
            cb_logits = cb_logits / temperature

            # Sample from each position
            probs = torch.nn.functional.softmax(cb_logits, dim=-1)
            new_codebook = torch.multinomial(probs.view(-1, probs.shape[-1]), num_samples=1)
            new_codebook = new_codebook.view(1, seq_len)

            all_codebooks = torch.cat([all_codebooks, new_codebook], dim=0)

        return all_codebooks
