# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Standard generator interface for text generation"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional, Union

import torch

import ttnn


@dataclass
class GenerationConfig:
    """Configuration for text generation"""

    max_new_tokens: int = 100
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    repetition_penalty: float = 1.0
    do_sample: bool = True
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[Union[int, List[int]]] = None
    return_dict_in_generate: bool = False
    output_attentions: bool = False
    output_hidden_states: bool = False


@dataclass
class GenerationOutput:
    """Output from generation"""

    sequences: torch.Tensor
    scores: Optional[List[torch.Tensor]] = None
    attentions: Optional[List[torch.Tensor]] = None
    hidden_states: Optional[List[torch.Tensor]] = None


class Generator(ABC):
    """
    Abstract base class for text generation.

    Provides a standard interface for generating text with transformer models.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        device: ttnn.Device,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    @abstractmethod
    def generate(
        self,
        input_ids: Union[torch.Tensor, ttnn.Tensor],
        generation_config: Optional[GenerationConfig] = None,
        attention_mask: Optional[Union[torch.Tensor, ttnn.Tensor]] = None,
        **kwargs,
    ) -> Union[torch.Tensor, GenerationOutput]:
        """
        Generate text from input IDs.

        Args:
            input_ids: Input token IDs
            generation_config: Generation configuration
            attention_mask: Optional attention mask
            **kwargs: Additional generation arguments

        Returns:
            Generated token IDs or GenerationOutput if return_dict_in_generate=True
        """

    @abstractmethod
    def generate_from_prompt(
        self,
        prompt: Union[str, List[str]],
        generation_config: Optional[GenerationConfig] = None,
        **kwargs,
    ) -> Union[str, List[str], GenerationOutput]:
        """
        Generate text from prompt string(s).

        Args:
            prompt: Input prompt(s)
            generation_config: Generation configuration
            **kwargs: Additional generation arguments

        Returns:
            Generated text or GenerationOutput
        """


class StandardGenerator(Generator):
    """
    Standard implementation of text generator.

    Implements common generation strategies like greedy, sampling, beam search.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        device: ttnn.Device,
        default_generation_config: Optional[GenerationConfig] = None,
    ):
        super().__init__(model, tokenizer, device)
        self.default_generation_config = default_generation_config or GenerationConfig()

    def generate(
        self,
        input_ids: Union[torch.Tensor, ttnn.Tensor],
        generation_config: Optional[GenerationConfig] = None,
        attention_mask: Optional[Union[torch.Tensor, ttnn.Tensor]] = None,
        **kwargs,
    ) -> Union[torch.Tensor, GenerationOutput]:
        """Generate text using the specified strategy"""
        config = generation_config or self.default_generation_config

        # Convert to ttnn tensors if needed
        if isinstance(input_ids, torch.Tensor):
            input_ids = ttnn.from_torch(input_ids, device=self.device)
        if attention_mask is not None and isinstance(attention_mask, torch.Tensor):
            attention_mask = ttnn.from_torch(attention_mask, device=self.device)

        # Initialize generation
        batch_size, seq_len = input_ids.shape
        generated_tokens = []
        past_key_values = None

        # Generation loop
        for _ in range(config.max_new_tokens):
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )

            logits = outputs.logits
            past_key_values = outputs.past_key_values

            # Get next token logits
            next_token_logits = logits[:, -1, :]

            # Apply temperature
            if config.temperature != 1.0:
                next_token_logits = next_token_logits / config.temperature

            # Apply repetition penalty
            if config.repetition_penalty != 1.0:
                self._apply_repetition_penalty(next_token_logits, input_ids, config.repetition_penalty)

            # Select next tokens
            if config.do_sample:
                next_tokens = self._sample(
                    next_token_logits,
                    top_k=config.top_k,
                    top_p=config.top_p,
                )
            else:
                next_tokens = ttnn.argmax(next_token_logits, dim=-1)

            # Update input_ids
            input_ids = ttnn.concat([input_ids, next_tokens.unsqueeze(1)], dim=1)
            generated_tokens.append(next_tokens)

            # Update attention mask
            if attention_mask is not None:
                attention_mask = ttnn.concat([attention_mask, ttnn.ones((batch_size, 1), device=self.device)], dim=1)

            # Check for EOS
            if self._check_eos(next_tokens, config.eos_token_id):
                break

        # Prepare output
        generated_sequence = ttnn.stack(generated_tokens, dim=1)

        if config.return_dict_in_generate:
            return GenerationOutput(sequences=generated_sequence)
        else:
            return generated_sequence

    def generate_from_prompt(
        self,
        prompt: Union[str, List[str]],
        generation_config: Optional[GenerationConfig] = None,
        **kwargs,
    ) -> Union[str, List[str], GenerationOutput]:
        """Generate text from prompt(s)"""
        # Handle single prompt
        if isinstance(prompt, str):
            prompts = [prompt]
            single_prompt = True
        else:
            prompts = prompt
            single_prompt = False

        # Tokenize prompts
        encoded = self.tokenizer(
            prompts,
            padding=True,
            return_tensors="pt",
        )

        # Generate
        outputs = self.generate(
            input_ids=encoded["input_ids"],
            attention_mask=encoded.get("attention_mask"),
            generation_config=generation_config,
            **kwargs,
        )

        # Decode outputs
        if isinstance(outputs, GenerationOutput):
            # Return GenerationOutput with decoded text
            decoded = self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
            # Update sequences with decoded text
            outputs.sequences = decoded[0] if single_prompt else decoded
            return outputs
        else:
            # Decode token IDs
            decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            return decoded[0] if single_prompt else decoded

    def _sample(
        self,
        logits: ttnn.Tensor,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> ttnn.Tensor:
        """Sample from logits with top-k/top-p filtering"""
        # Apply top-k filtering
        if top_k is not None:
            top_k_logits, top_k_indices = ttnn.topk(logits, k=top_k, dim=-1)
            # Create mask for non-top-k values
            min_top_k = ttnn.min(top_k_logits, dim=-1, keepdim=True)
            logits = ttnn.where(logits >= min_top_k, logits, -float("inf"))

        # Apply top-p (nucleus) filtering
        if top_p is not None:
            sorted_logits, sorted_indices = ttnn.sort(logits, dim=-1, descending=True)
            cumulative_probs = ttnn.cumsum(ttnn.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Keep at least one token
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = False

            # Scatter back to original indices
            indices_to_remove = ttnn.scatter(
                sorted_indices_to_remove, dim=-1, index=sorted_indices, src=sorted_indices_to_remove
            )
            logits = ttnn.where(indices_to_remove, -float("inf"), logits)

        # Sample from distribution
        probs = ttnn.softmax(logits, dim=-1)
        next_tokens = ttnn.multinomial(probs, num_samples=1).squeeze(1)

        return next_tokens

    def _apply_repetition_penalty(
        self,
        logits: ttnn.Tensor,
        input_ids: ttnn.Tensor,
        penalty: float,
    ):
        """Apply repetition penalty to logits"""
        # Get unique tokens in input
        for i in range(input_ids.shape[0]):
            unique_ids = ttnn.unique(input_ids[i])
            for token_id in unique_ids:
                # Reduce logit by penalty
                logits[i, token_id] = logits[i, token_id] / penalty

    def _check_eos(
        self,
        token_ids: ttnn.Tensor,
        eos_token_id: Optional[Union[int, List[int]]],
    ) -> bool:
        """Check if any sequence has reached EOS"""
        if eos_token_id is None:
            return False

        if isinstance(eos_token_id, int):
            eos_token_ids = [eos_token_id]
        else:
            eos_token_ids = eos_token_id

        # Check if any token matches EOS
        for eos_id in eos_token_ids:
            if ttnn.any(token_ids == eos_id):
                return True

        return False
