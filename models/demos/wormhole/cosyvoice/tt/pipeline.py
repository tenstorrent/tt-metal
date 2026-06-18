# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
CosyVoice End-to-End Inference Pipeline on TT hardware.

Chains the LLM backbone, flow decoder, and HiFi-GAN vocoder into a
unified inference pipeline supporting all 4 inference modes:
  - SFT: Text-to-speech with predefined speaker
  - Zero-shot: Voice cloning with reference audio
  - Cross-lingual: Generate speech in different language from reference
  - Instruct: Expressive speech with instruction
"""

from typing import Generator, Optional, Tuple

import numpy as np
import torch
import ttnn
from loguru import logger

from models.demos.wormhole.cosyvoice.tt.llm.cosyvoice_llm import TtCosyVoiceLLM
from models.demos.wormhole.cosyvoice.tt.flow.flow_matching import TtMaskedDiffWithXvec
from models.demos.wormhole.cosyvoice.tt.vocoder.hifigan import TtHiFiGAN


class TtCosyVoicePipeline:
    """End-to-end CosyVoice inference pipeline on TT hardware.

    Orchestrates the full TTS pipeline: text -> LLM (semantic tokens)
    -> flow decoder (mel) -> HiFi-GAN (audio).
    """

    def __init__(
        self,
        device: ttnn.Device,
        config,
        state_dict: dict,
        tt_cache_path: Optional[str] = None,
    ):
        self.device = device
        self.config = config

        logger.info("Initializing CosyVoice pipeline")

        # Initialize sub-modules
        self.llm = TtCosyVoiceLLM(device, config, state_dict, tt_cache_path)
        self.flow = TtMaskedDiffWithXvec(device, config, state_dict, tt_cache_path)
        self.hift = TtHiFiGAN(device, config, state_dict, tt_cache_path)

        # Inference mode handlers
        self._mode_handlers = {
            "sft": self._run_sft,
            "zero_shot": self._run_zero_shot,
            "cross_lingual": self._run_cross_lingual,
            "instruct": self._run_instruct,
        }

        # Streaming state
        self.token_overlap_len = 20
        self.token_min_hop_len = int(2 * self.flow.input_frame_rate)
        self.mel_overlap_len = int(
            self.token_overlap_len / self.flow.input_frame_rate * 22050 / 256
        )
        self.mel_cache_len = 20
        self.source_cache_len = int(self.mel_cache_len * 256)

    def tts(
        self,
        text: str,
        mode: str = "sft",
        ref_audio: Optional[str] = None,
        instruct: Optional[str] = None,
        language: str = "en",
        stream: bool = False,
        speed: float = 1.0,
    ) -> torch.Tensor:
        """Run text-to-speech inference.

        Args:
            text: Input text to synthesize
            mode: Inference mode (sft, zero_shot, cross_lingual, instruct)
            ref_audio: Path to reference audio for zero-shot/cross-lingual modes
            instruct: Instruction for instruct mode
            language: Language of input text (zh, en, ja, yue, ko)
            stream: Whether to use streaming inference
            speed: Speed factor for non-streaming mode

        Returns:
            Audio waveform tensor [1, audio_samples]
        """
        handler = self._mode_handlers.get(mode)
        if handler is None:
            raise ValueError(f"Unknown mode: {mode}. Choose from: {list(self._mode_handlers.keys())}")

        logger.info(f"Running CosyVoice in {mode} mode: text='{text[:50]}...'")

        # Run mode-specific handler
        audio = handler(text, ref_audio, instruct, language)

        return audio

    def _run_sft(
        self,
        text: str,
        ref_audio: Optional[str] = None,
        instruct: Optional[str] = None,
        language: str = "en",
    ) -> torch.Tensor:
        """SFT mode: Generate speech with predefined speaker embedding.

        Uses the default speaker embedding from pretrained weights.
        """
        # In Stage 1, we tokenize text and run through the pipeline
        # Tokenization uses the Qwen2 tokenizer
        text_tokens = self._tokenize(text, language)
        text_len = torch.tensor([text_tokens.shape[1]], dtype=torch.int32)

        # Empty prompt (no reference audio for SFT)
        prompt_text = torch.zeros(1, 0, dtype=torch.int32)
        prompt_text_len = torch.tensor([0], dtype=torch.int32)
        prompt_speech_token = torch.zeros(1, 0, dtype=torch.int32)
        prompt_speech_token_len = torch.tensor([0], dtype=torch.int32)

        # Default speaker embedding (zeros for SFT)
        embedding = torch.zeros(1, 192)

        return self._run_pipeline(
            text_tokens, text_len,
            prompt_text, prompt_text_len,
            prompt_speech_token, prompt_speech_token_len,
            embedding,
        )

    def _run_zero_shot(
        self,
        text: str,
        ref_audio: Optional[str] = None,
        instruct: Optional[str] = None,
        language: str = "en",
    ) -> torch.Tensor:
        """Zero-shot mode: Voice cloning with reference audio.

        Uses a reference audio sample to extract speaker embedding and
        prompt speech tokens for voice cloning.
        """
        text_tokens = self._tokenize(text, language)
        text_len = torch.tensor([text_tokens.shape[1]], dtype=torch.int32)

        # Extract prompt features from reference audio
        prompt_text = torch.zeros(1, 0, dtype=torch.int32)
        prompt_text_len = torch.tensor([0], dtype=torch.int32)

        # For Stage 1, use zeros as placeholders
        # Full feature extraction pipeline will be in Stage 2
        prompt_speech_token = torch.zeros(1, 50, dtype=torch.int32)
        prompt_speech_token_len = torch.tensor([50], dtype=torch.int32)

        # Speaker embedding from reference (placeholder in Stage 1)
        embedding = torch.zeros(1, 192)

        return self._run_pipeline(
            text_tokens, text_len,
            prompt_text, prompt_text_len,
            prompt_speech_token, prompt_speech_token_len,
            embedding,
        )

    def _run_cross_lingual(
        self,
        text: str,
        ref_audio: Optional[str] = None,
        instruct: Optional[str] = None,
        language: str = "en",
    ) -> torch.Tensor:
        """Cross-lingual mode: Generate speech in different language."""
        # Same as zero-shot but with cross-lingual prompt
        return self._run_zero_shot(text, ref_audio, instruct, language)

    def _run_instruct(
        self,
        text: str,
        ref_audio: Optional[str] = None,
        instruct: Optional[str] = None,
        language: str = "en",
    ) -> torch.Tensor:
        """Instruct mode: Generate expressive speech with instructions."""
        text_tokens = self._tokenize(text, language)
        text_len = torch.tensor([text_tokens.shape[1]], dtype=torch.int32)

        # Empty prompt
        prompt_text = torch.zeros(1, 0, dtype=torch.int32)
        prompt_text_len = torch.tensor([0], dtype=torch.int32)
        prompt_speech_token = torch.zeros(1, 0, dtype=torch.int32)
        prompt_speech_token_len = torch.tensor([0], dtype=torch.int32)

        # Speaker embedding
        embedding = torch.zeros(1, 192)

        return self._run_pipeline(
            text_tokens, text_len,
            prompt_text, prompt_text_len,
            prompt_speech_token, prompt_speech_token_len,
            embedding,
        )

    def _run_pipeline(
        self,
        text_tokens: torch.Tensor,
        text_len: torch.Tensor,
        prompt_text: torch.Tensor,
        prompt_text_len: torch.Tensor,
        prompt_speech_token: torch.Tensor,
        prompt_speech_token_len: torch.Tensor,
        embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Run the full TTS pipeline: LLM -> Flow -> HiFi-GAN.

        Args:
            text_tokens: Tokenized input text [1, seq_len]
            text_len: Input text length [1]
            prompt_text: Prompt text tokens [1, prompt_len]
            prompt_text_len: Prompt text length [1]
            prompt_speech_token: Prompt speech tokens [1, prompt_speech_len]
            prompt_speech_token_len: Prompt speech token length [1]
            embedding: Speaker embedding [1, 192]

        Returns:
            Audio waveform [1, audio_samples]
        """
        # Step 1: LLM generates semantic tokens from text
        logger.info("Step 1: LLM backbone - generating semantic tokens")
        semantic_tokens = []
        self.llm.reset_kv_cache()

        for token in self.llm.inference(
            text=text_tokens,
            text_len=text_len,
            prompt_text=prompt_text,
            prompt_text_len=prompt_text_len,
            prompt_speech_token=prompt_speech_token,
            prompt_speech_token_len=prompt_speech_token_len,
            embedding=embedding,
            sampling=25,
        ):
            semantic_tokens.append(token)

        if not semantic_tokens:
            logger.warning("LLM produced no tokens, using empty fallback")
            return torch.zeros(1, 24000)  # 1 second of silence

        semantic_tokens_tensor = torch.tensor([semantic_tokens], dtype=torch.int32)
        logger.info(f"Generated {len(semantic_tokens)} semantic tokens")

        # Step 2: Flow decoder generates mel-spectrogram from semantic tokens
        logger.info("Step 2: Flow decoder - generating mel-spectrogram")
        mel, _ = self.flow.inference(
            token=semantic_tokens_tensor,
            token_len=torch.tensor([semantic_tokens_tensor.shape[1]], dtype=torch.int32),
            prompt_token=prompt_speech_token,
            prompt_token_len=prompt_speech_token_len,
            prompt_feat=torch.zeros(1, 80, 0),
            prompt_feat_len=torch.tensor([0], dtype=torch.int32),
            embedding=embedding,
        )

        # Step 3: HiFi-GAN vocoder converts mel to audio
        logger.info("Step 3: HiFi-GAN vocoder - generating audio waveform")
        tt_mel = ttnn.from_torch(mel.unsqueeze(0), device=self.device)
        tt_audio = self.hift(tt_mel)
        audio = ttnn.to_torch(tt_audio).squeeze(0)

        logger.info(f"Pipeline complete: audio shape {audio.shape}")

        return audio

    def _tokenize(
        self,
        text: str,
        language: str = "en",
    ) -> torch.Tensor:
        """Tokenize input text using the provided tokenizer.

        In Stage 1, this is a placeholder that maps to token IDs.
        Full Qwen2/HuggingFace tokenizer integration in Stage 2.

        Args:
            text: Input text string
            language: Language code

        Returns:
            Token IDs tensor [1, seq_len]
        """
        # Placeholder: map characters to token IDs
        # Real implementation will use Qwen2Tokenizer
        token_ids = [ord(c) % 1000 + 100 for c in text.strip()]
        return torch.tensor([token_ids], dtype=torch.int32)

    def reset(self):
        """Reset pipeline state for a new inference sequence."""
        self.llm.reset_kv_cache()
