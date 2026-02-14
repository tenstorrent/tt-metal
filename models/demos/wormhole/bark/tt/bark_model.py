# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Full Bark Small Pipeline: Text → Audio.

Orchestrates the three stages + EnCodec decoder to produce audio from text.
Uses HuggingFace Bark model for weight loading and EnCodec for audio decoding.
"""

import time
from typing import Optional

import numpy as np
import torch
from loguru import logger

import ttnn

from models.demos.wormhole.bark.tt.bark_semantic import TtBarkSemanticModel
from models.demos.wormhole.bark.tt.bark_coarse import TtBarkCoarseModel
from models.demos.wormhole.bark.tt.bark_fine import TtBarkFineModel
from models.demos.wormhole.bark.tt.common import BarkConfig, get_bark_small_config


class TtBarkModel:
    """
    Full Bark Small pipeline: Text → Semantic → Coarse → Fine → Audio.

    Usage:
        model = TtBarkModel.from_pretrained(device)
        audio = model.generate("Hello, my dog is cooler than you!")
    """

    def __init__(
        self,
        semantic_model: TtBarkSemanticModel,
        coarse_model: TtBarkCoarseModel,
        fine_model: TtBarkFineModel,
        config: BarkConfig,
        processor,
        hf_model,
    ):
        self.semantic = semantic_model
        self.coarse = coarse_model
        self.fine = fine_model
        self.config = config
        self.processor = processor
        self.hf_model = hf_model  # Keep for codec_decode and reference

    @classmethod
    def from_pretrained(
        cls,
        device: ttnn.Device,
        model_name: str = "suno/bark-small",
    ):
        """
        Load all 3 stages from a HuggingFace checkpoint and place on TT device.
        """
        from transformers import AutoProcessor, BarkModel as HFBarkModel

        logger.info(f"Loading HuggingFace Bark model: {model_name}")
        hf_model = HFBarkModel.from_pretrained(model_name)
        processor = AutoProcessor.from_pretrained(model_name)
        state_dict = hf_model.state_dict()

        config = get_bark_small_config()

        # Verify dimensions from state dict
        if "semantic.input_embeds_layer.weight" in state_dict:
            actual_hidden = state_dict["semantic.input_embeds_layer.weight"].shape[1]
            actual_input_vocab = state_dict["semantic.input_embeds_layer.weight"].shape[0]
            logger.info(f"Verified from state_dict: hidden_size={actual_hidden}, semantic_input_vocab={actual_input_vocab}")
            config.hidden_size = actual_hidden
            config.semantic_input_vocab_size = actual_input_vocab

        # Count layers
        layer_count = sum(
            1 for key in state_dict
            if key.startswith("semantic.layers.") and key.endswith(".layernorm_1.weight")
        )
        if layer_count > 0:
            config.num_hidden_layers = layer_count
            logger.info(f"Verified from state_dict: num_layers={layer_count}")

        # Infer intermediate_size from MLP weights
        if "semantic.layers.0.mlp.in_proj.weight" in state_dict:
            config.intermediate_size = state_dict["semantic.layers.0.mlp.in_proj.weight"].shape[0]
            logger.info(f"Verified from state_dict: intermediate_size={config.intermediate_size}")

        # Check attention heads from att_proj
        if "semantic.layers.0.attn.att_proj.weight" in state_dict:
            qkv_size = state_dict["semantic.layers.0.attn.att_proj.weight"].shape[0]
            config.hidden_size = qkv_size // 3
            logger.info(f"Verified: hidden_size={config.hidden_size} (from QKV dim {qkv_size})")

        # Check output vocab
        if "semantic.lm_head.weight" in state_dict:
            config.semantic_output_vocab_size = state_dict["semantic.lm_head.weight"].shape[0]
        if "coarse_acoustics.lm_head.weight" in state_dict:
            config.coarse_output_vocab_size = state_dict["coarse_acoustics.lm_head.weight"].shape[0]

        logger.info(
            f"Bark config: hidden={config.hidden_size}, layers={config.num_hidden_layers}, "
            f"heads={config.num_attention_heads}, intermediate={config.intermediate_size}"
        )

        logger.info("Initializing Stage 1: Text → Semantic")
        semantic_model = TtBarkSemanticModel(state_dict, device, config)

        logger.info("Initializing Stage 2: Semantic → Coarse")
        coarse_model = TtBarkCoarseModel(state_dict, device, config)

        logger.info("Initializing Stage 3: Coarse → Fine")
        fine_model = TtBarkFineModel(state_dict, device, config)

        return cls(
            semantic_model=semantic_model,
            coarse_model=coarse_model,
            fine_model=fine_model,
            config=config,
            processor=processor,
            hf_model=hf_model,
        )

    def generate(
        self,
        text: str,
        voice_preset: Optional[str] = "v2/en_speaker_6",
        max_semantic_tokens: int = 256,
        max_coarse_tokens: int = 256,
        semantic_temperature: float = 0.7,
        coarse_temperature: float = 0.7,
        fine_temperature: float = 0.5,
    ) -> np.ndarray:
        """
        Generate audio from text.

        Args:
            text: Input text string
            voice_preset: Optional HuggingFace voice preset
            max_semantic_tokens: Max tokens for Stage 1
            max_coarse_tokens: Max tokens for Stage 2
            semantic_temperature: Temperature for Stage 1
            coarse_temperature: Temperature for Stage 2
            fine_temperature: Temperature for Stage 3

        Returns:
            audio: numpy array of audio samples at 24 kHz
        """
        logger.info(f"Generating audio for: '{text[:60]}...'")

        # Tokenize text using HF processor
        inputs = self.processor(text=[text], return_tensors="pt", voice_preset=voice_preset)
        input_ids = inputs.get("input_ids", inputs.get("input_values"))

        # Stage 1: Text → Semantic
        t0 = time.time()
        semantic_tokens = self.semantic.generate(
            input_ids,
            max_new_tokens=max_semantic_tokens,
            temperature=semantic_temperature,
        )
        t1 = time.time()
        semantic_len = semantic_tokens.shape[-1] - input_ids.shape[-1]
        if t1 - t0 > 0:
            logger.info(
                f"Stage 1 (Semantic): {semantic_len} tokens in {t1 - t0:.2f}s "
                f"({semantic_len / (t1 - t0):.1f} tok/s)"
            )

        # Stage 2: Semantic → Coarse
        t0 = time.time()
        coarse_tokens = self.coarse.generate(
            semantic_tokens,
            max_new_tokens=max_coarse_tokens,
            temperature=coarse_temperature,
        )
        t1 = time.time()
        coarse_len = coarse_tokens.shape[-1] - semantic_tokens.shape[-1]
        if t1 - t0 > 0:
            logger.info(
                f"Stage 2 (Coarse): {coarse_len} tokens in {t1 - t0:.2f}s "
                f"({coarse_len / (t1 - t0):.1f} tok/s)"
            )

        # Extract coarse codebook tokens and de-interleave into [n_coarse_codebooks, seq_len]
        coarse_output = coarse_tokens[:, semantic_tokens.shape[-1] :]
        if coarse_output.numel() > 0:
            # Interleaved format: [cb0_0, cb1_0, cb0_1, cb1_1, ...]
            n_coarse = self.config.n_coarse_codebooks
            coarse_seq_len = coarse_output.shape[-1] // n_coarse
            flat = coarse_output[:, : coarse_seq_len * n_coarse]
            # First reshape to [batch, seq_len, n_coarse], then permute to [n_coarse, batch, seq_len]
            interleaved = flat.reshape(1, coarse_seq_len, n_coarse)
            coarse_codebooks = interleaved.permute(2, 0, 1).reshape(n_coarse, coarse_seq_len)
        else:
            coarse_codebooks = torch.zeros(self.config.n_coarse_codebooks, 1, dtype=torch.long)

        # Stage 3: Coarse → Fine
        t0 = time.time()
        fine_tokens = self.fine.generate(
            coarse_codebooks,
            temperature=fine_temperature,
        )
        t1 = time.time()
        logger.info(f"Stage 3 (Fine): Generated {fine_tokens.shape[0]} codebooks in {t1 - t0:.2f}s")

        # Decode to audio using HF's codec_decode
        t0 = time.time()
        audio = self._decode_audio(fine_tokens)
        t1 = time.time()
        logger.info(f"EnCodec decode: {len(audio) / self.config.sample_rate:.2f}s audio in {t1 - t0:.2f}s")

        return audio

    def _decode_audio(self, fine_tokens: torch.Tensor) -> np.ndarray:
        """
        Decode EnCodec tokens to audio waveform using HuggingFace's codec model.

        Args:
            fine_tokens: [n_codebooks, seq_len] tensor of codebook indices

        Returns:
            audio: numpy array of audio samples
        """
        # HF Bark uses codec_decode which expects [n_codebooks, 1, seq_len]
        codes = fine_tokens.unsqueeze(1).long()

        with torch.no_grad():
            audio = self.hf_model.codec_decode(codes)

        if isinstance(audio, torch.Tensor):
            audio = audio.squeeze().cpu().numpy()
        else:
            audio = np.array(audio).squeeze()

        return audio
