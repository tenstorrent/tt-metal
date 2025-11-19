# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional
import torch
from loguru import logger

import ttnn

from .sampling import SamplingParams
from .generation import generate_text_tokens, generate_text_with_callback
from .tokenizer_utils import decode_tokens, get_special_tokens


class MiniCPMGenerator:
    """
    High-level generator for MiniCPM-o-2_6 multimodal model.

    Adapted from Generator class in models/tt_transformers/tt/generator.py
    """

    def __init__(self, pipeline, tokenizer, max_seq_len: int = 2048):
        """
        Initialize MiniCPM Generator.

        Args:
            pipeline: TtnnMiniCPMPipeline instance
            tokenizer: HuggingFace tokenizer
            max_seq_len: Maximum sequence length
        """
        self.pipeline = pipeline
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.special_tokens = get_special_tokens(tokenizer)

        # Extract mesh device if using mesh mode
        self.mesh_device = pipeline.device if hasattr(pipeline.device, "get_num_devices") else None

        logger.info(
            f"Initialized MiniCPMGenerator with max_seq_len={max_seq_len}, mesh_device={self.mesh_device is not None}"
        )

    def generate_text(
        self,
        prompt: str,
        max_gen_len: int = 512,
        temperature: float = 0.6,
        top_p: float = 0.9,
        top_k: Optional[int] = None,
        audio_input: Optional[torch.Tensor] = None,
        image_input: Optional[torch.Tensor] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Generate text from multimodal inputs.

        Args:
            prompt: Text prompt
            max_gen_len: Maximum generation length
            temperature: Sampling temperature
            top_p: Top-p probability
            top_k: Top-k sampling (optional)
            audio_input: Audio input tensor
            image_input: Image input tensor
            system_prompt: System prompt (optional)

        Returns:
            Generated text string
        """
        # Create sampling parameters
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k if top_k is not None else -1,
        )

        # Determine generation mode
        if audio_input is not None or image_input is not None:
            return self.multimodal_to_text(
                prompt_text=prompt,
                audio_input=audio_input,
                image_input=image_input,
                max_gen_len=max_gen_len,
                sampling_params=sampling_params,
                system_prompt=system_prompt,
            )
        else:
            return self.text_to_text(
                prompt_text=prompt,
                max_gen_len=max_gen_len,
                sampling_params=sampling_params,
                system_prompt=system_prompt,
            )

    def text_to_text(
        self,
        prompt_text: str,
        max_gen_len: int = 512,
        sampling_params: Optional[SamplingParams] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Generate text from text input only.

        Args:
            prompt_text: Input text prompt
            max_gen_len: Maximum generation length
            sampling_params: Sampling parameters
            system_prompt: System prompt (optional)

        Returns:
            Generated text string
        """
        if sampling_params is None:
            sampling_params = SamplingParams()

        logger.info(f"Generating text-to-text with prompt: {prompt_text[:100]}...")

        # Encode prompt
        input_tokens = self._encode_prompt(prompt_text, system_prompt)
        input_tokens = torch.tensor(input_tokens, dtype=torch.long).unsqueeze(0)

        # Generate tokens
        generated_tokens = generate_text_tokens(
            model=self.pipeline.qwen_llm,
            input_tokens=input_tokens,
            mesh_device=self.mesh_device,
            max_gen_len=max_gen_len,
            sampling_params=sampling_params,
            encoder_hidden_states=None,  # No multimodal inputs
            stop_token_ids=[self.special_tokens["eos_token_id"]],
        )

        # Decode to text
        generated_text = decode_tokens(self.tokenizer, generated_tokens, skip_special_tokens=True)

        logger.info(f"Generated {len(generated_tokens)} tokens")
        return generated_text

    def multimodal_to_text(
        self,
        prompt_text: str,
        audio_input: Optional[torch.Tensor] = None,
        image_input: Optional[torch.Tensor] = None,
        max_gen_len: int = 512,
        sampling_params: Optional[SamplingParams] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Generate text from multimodal inputs (audio + text + image).

        Args:
            prompt_text: Text prompt
            audio_input: Audio input tensor
            image_input: Image input tensor
            max_gen_len: Maximum generation length
            sampling_params: Sampling parameters
            system_prompt: System prompt (optional)

        Returns:
            Generated text string
        """
        if sampling_params is None:
            sampling_params = SamplingParams()

        logger.info(f"Generating multimodal-to-text with prompt: {prompt_text[:100]}...")

        # Process multimodal inputs
        encoder_features = self._process_multimodal_inputs(audio_input, image_input)

        # Encode text prompt
        input_tokens = self._encode_prompt(prompt_text, system_prompt)
        input_tokens = torch.tensor(input_tokens, dtype=torch.long).unsqueeze(0)

        # Generate tokens with multimodal features
        generated_tokens = generate_text_tokens(
            model=self.pipeline.qwen_llm,
            input_tokens=input_tokens,
            mesh_device=self.mesh_device,
            max_gen_len=max_gen_len,
            sampling_params=sampling_params,
            encoder_hidden_states=encoder_features,
            stop_token_ids=[self.special_tokens["eos_token_id"]],
        )

        # Decode to text
        generated_text = decode_tokens(self.tokenizer, generated_tokens, skip_special_tokens=True)

        logger.info(f"Generated {len(generated_tokens)} tokens with multimodal inputs")
        return generated_text

    def generate_tokens(
        self,
        input_tokens: torch.Tensor,
        encoder_features: Optional[ttnn.Tensor] = None,
        max_gen_len: int = 512,
        sampling_params: Optional[SamplingParams] = None,
    ) -> List[int]:
        """
        Low-level token generation.

        Args:
            input_tokens: Input token IDs [batch, seq_len]
            encoder_features: Encoder features for cross-attention
            max_gen_len: Maximum generation length
            sampling_params: Sampling parameters

        Returns:
            List of generated token IDs
        """
        if sampling_params is None:
            sampling_params = SamplingParams()

        return generate_text_tokens(
            model=self.pipeline.qwen_llm,
            input_tokens=input_tokens,
            mesh_device=self.mesh_device,
            max_gen_len=max_gen_len,
            sampling_params=sampling_params,
            encoder_hidden_states=encoder_features,
            stop_token_ids=[self.special_tokens["eos_token_id"]],
        )

    def generate_with_callback(
        self,
        prompt: str,
        callback: callable,
        max_gen_len: int = 512,
        temperature: float = 0.6,
        top_p: float = 0.9,
        top_k: Optional[int] = None,
        audio_input: Optional[torch.Tensor] = None,
        image_input: Optional[torch.Tensor] = None,
    ) -> str:
        """
        Generate text with real-time callback.

        Args:
            prompt: Text prompt
            callback: Callback function called for each token
            max_gen_len: Maximum generation length
            temperature: Sampling temperature
            top_p: Top-p probability
            top_k: Top-k sampling
            audio_input: Audio input
            image_input: Image input

        Returns:
            Complete generated text
        """
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k if top_k is not None else -1,
        )

        # Process multimodal inputs
        encoder_features = self._process_multimodal_inputs(audio_input, image_input)

        # Encode prompt
        input_tokens = self._encode_prompt(prompt)
        input_tokens = torch.tensor(input_tokens, dtype=torch.long).unsqueeze(0)

        # Generate with callback
        generated_tokens, generated_texts = generate_text_with_callback(
            model=self.pipeline.qwen_llm,
            input_tokens=input_tokens,
            max_gen_len=max_gen_len,
            sampling_params=sampling_params,
            encoder_hidden_states=encoder_features,
            stop_token_ids=[self.special_tokens["eos_token_id"]],
            callback=callback,
        )

        # Return complete text
        return "".join(generated_texts)

    def _encode_prompt(self, prompt_text: str, system_prompt: Optional[str] = None) -> List[int]:
        """Encode prompt text to token IDs."""
        from .tokenizer_utils import encode_prompt

        return encode_prompt(self.tokenizer, prompt_text, system_prompt)

    def _process_multimodal_inputs(
        self,
        audio_input: Optional[torch.Tensor] = None,
        image_input: Optional[torch.Tensor] = None,
    ) -> Optional[ttnn.Tensor]:
        """
        Process multimodal inputs through encoders and projectors.

        Args:
            audio_input: Audio input tensor
            image_input: Image input tensor

        Returns:
            Combined encoder features for cross-attention
        """
        encoder_features = []

        # Process audio input
        if audio_input is not None:
            logger.debug("Processing audio input")
            # Use pipeline's audio processing
            audio_features = self.pipeline._preprocess_audio(audio_input)
            audio_features = self.pipeline.audio_projector(audio_features)
            encoder_features.append(audio_features)

        # Process image input
        if image_input is not None:
            logger.debug("Processing image input")
            # Use pipeline's vision processing
            image_features = self.pipeline._preprocess_image(image_input)
            image_features = self.pipeline.vision_resampler(image_features)
            encoder_features.append(image_features)

        if not encoder_features:
            return None

        # Combine multimodal features (concatenate along sequence dimension)
        if len(encoder_features) == 1:
            return encoder_features[0]
        else:
            # Concatenate features from different modalities
            return ttnn.concat(encoder_features, dim=-2)  # Concat along sequence dim

    def chat_completion(
        self,
        messages: List[dict],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        audio_input: Optional[torch.Tensor] = None,
        image_input: Optional[torch.Tensor] = None,
    ) -> str:
        """
        Chat completion with conversation history.

        Args:
            messages: List of chat messages
            temperature: Sampling temperature
            top_p: Top-p probability
            max_gen_len: Maximum generation length
            audio_input: Audio input
            image_input: Image input

        Returns:
            Generated response text
        """
        if max_gen_len is None:
            max_gen_len = min(512, self.max_seq_len - 100)  # Leave room for input

        # Process multimodal inputs
        encoder_features = self._process_multimodal_inputs(audio_input, image_input)

        # Apply chat template
        chat_tokens = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
        input_tokens = torch.tensor(chat_tokens, dtype=torch.long).unsqueeze(0)

        # Generate
        sampling_params = SamplingParams(temperature=temperature, top_p=top_p, top_k=-1)

        generated_tokens = generate_text_tokens(
            model=self.pipeline.qwen_llm,
            input_tokens=input_tokens,
            mesh_device=self.mesh_device,
            max_gen_len=max_gen_len,
            sampling_params=sampling_params,
            encoder_hidden_states=encoder_features,
            stop_token_ids=[self.special_tokens["eos_token_id"]],
        )

        return decode_tokens(self.tokenizer, generated_tokens, skip_special_tokens=True)
