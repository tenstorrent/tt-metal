# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Gemma text encoder for LTX-2.

Wraps HuggingFace Gemma3 model to produce text conditioning embeddings.
Currently runs on CPU/GPU (torch-only). Follows the T5TokenizerEncoderPair
pattern from tt_dit encoders.

The LTX-2 pipeline uses Gemma3's hidden states (not logits) as text conditioning.
The hidden states from a specific layer are extracted and used as the context
tensor for the DiT cross-attention.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from loguru import logger
from transformers import AutoTokenizer

if TYPE_CHECKING:
    from collections.abc import Iterable


class GemmaTokenizerEncoderPair:
    """
    Tokenizer + encoder pair for Gemma text encoder.

    For now, runs entirely on CPU/GPU (no ttnn). The output is a torch tensor
    that gets pushed to the TT device by the pipeline.

    Args:
        checkpoint: HuggingFace model ID or local path (e.g., "google/gemma-3-4b-it")
        sequence_length: Maximum sequence length for tokenization
        embedding_dim: Expected embedding dimension (for empty prompt fallback)
        hidden_layer_index: Which hidden layer to extract (default -1 = last)
        enabled: If False, returns zero embeddings without loading model
    """

    def __init__(
        self,
        checkpoint: str,
        *,
        sequence_length: int = 256,
        embedding_dim: int = 4096,
        hidden_layer_index: int = -1,
        enabled: bool = True,
    ) -> None:
        self._sequence_length = sequence_length
        self._embedding_dim = embedding_dim
        self._hidden_layer_index = hidden_layer_index

        if enabled:
            self._tokenizer = AutoTokenizer.from_pretrained(checkpoint)
            self._model = self._load_model(checkpoint)
        else:
            self._tokenizer = None
            self._model = None

    def _load_model(self, checkpoint: str):
        """Load the Gemma model. Uses the inner model (no lm_head) for efficiency."""
        try:
            from transformers import Gemma3ForConditionalGeneration

            model = Gemma3ForConditionalGeneration.from_pretrained(checkpoint, torch_dtype=torch.bfloat16)
            model.eval()
            logger.info(f"Loaded Gemma model from {checkpoint}")
            return model
        except ImportError:
            logger.warning("Gemma3ForConditionalGeneration not available, trying AutoModel")
            from transformers import AutoModel

            model = AutoModel.from_pretrained(checkpoint, torch_dtype=torch.bfloat16)
            model.eval()
            return model

    def encode(self, prompts: Iterable[str], *, num_images_per_prompt: int = 1) -> torch.Tensor:
        """
        Encode text prompts into embeddings.

        Args:
            prompts: List of text prompts
            num_images_per_prompt: Number of images per prompt (for repeating)

        Returns:
            torch.Tensor of shape (B * num_images_per_prompt, seq_len, embedding_dim)
        """
        prompts = list(prompts)
        B = len(prompts)

        if self._model is None:
            return torch.zeros(B * num_images_per_prompt, self._sequence_length, self._embedding_dim)

        # Tokenize
        tokenizer_out = self._tokenizer(
            prompts,
            return_tensors="pt",
            padding="max_length",
            max_length=self._sequence_length,
            truncation=True,
        )

        input_ids = tokenizer_out.input_ids.to(self._model.device)
        attention_mask = tokenizer_out.attention_mask.to(self._model.device)

        # Run model and extract hidden states
        with torch.no_grad():
            if hasattr(self._model, "model"):
                # Gemma3ForConditionalGeneration: use inner model to skip lm_head
                outputs = self._model.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
            else:
                outputs = self._model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )

            hidden_states = outputs.hidden_states
            # Extract the specified layer's hidden states
            prompt_embeds = hidden_states[self._hidden_layer_index].float()

        # Repeat for num_images_per_prompt
        if num_images_per_prompt > 1:
            prompt_embeds = prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)

        logger.info(f"Gemma encoded {B} prompts -> shape {prompt_embeds.shape}")
        return prompt_embeds
