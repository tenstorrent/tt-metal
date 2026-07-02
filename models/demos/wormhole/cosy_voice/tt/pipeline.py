# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
CosyVoice3 End-to-End TTS Pipeline on Tenstorrent Wormhole.

Orchestrates: Text → LLM (speech tokens) → Flow Decoder (mel) → Vocoder (audio)

Device allocation:
- LLM: mesh device (N300, 2 chips) — for tensor-parallel transformer
- Flow Decoder DiT: single device (device 0) — 22 DiT blocks
- Vocoder: single device (device 0) — causal HiFT-GAN (host-side for now)
"""

import os

import torch
from loguru import logger

import ttnn
from models.demos.wormhole.cosy_voice.tt.cosyvoice_llm import CosyVoice3LM
from models.demos.wormhole.cosy_voice.tt.flow.flow import TtCausalMaskedDiffWithDiT
from models.demos.wormhole.cosy_voice.tt.model_config import CosyVoiceModelConfig


class CosyVoicePipeline:
    """
    End-to-end CosyVoice3 TTS pipeline on Tenstorrent Wormhole.

    Usage:
        pipeline = CosyVoicePipeline.from_pretrained(weights_dir, mesh_device)
        audio = pipeline.tts(text_tokens, prompt_text_tokens,
                             prompt_speech_tokens, prompt_feat, embedding)
    """

    def __init__(self, llm, flow_decoder, weights_dir, mesh_device):
        self.llm = llm
        self.flow = flow_decoder
        self.mesh_device = mesh_device
        self.weights_dir = weights_dir

        # Flow decoder config
        self.token_mel_ratio = 2

    @classmethod
    def from_pretrained(cls, weights_dir, mesh_device, dtype=ttnn.bfloat16):
        """
        Load all pipeline components from pretrained weights.

        Args:
            weights_dir: Path to Fun-CosyVoice3-0.5B directory
            mesh_device: TTNN mesh device (N300)
            dtype: TTNN dtype for weights
        """
        # 1. LLM (uses mesh for tensor-parallel)
        logger.info("Loading LLM...")
        config = CosyVoiceModelConfig(
            mesh_device=mesh_device,
            max_batch_size=1,
            weights_dir=weights_dir,
        )
        llm = CosyVoice3LM(config, mesh_device, dtype=ttnn.bfloat8_b)

        # 2. Flow Decoder (uses single device for DiT blocks)
        logger.info("Loading Flow Decoder...")
        flow_sd = torch.load(os.path.join(weights_dir, "flow.pt"), map_location="cpu")
        # Flow decoder runs on a single device — open device 0 directly
        single_device = ttnn.open_device(device_id=0)
        flow_decoder = TtCausalMaskedDiffWithDiT(single_device, flow_sd, dtype=dtype)

        # 3. Vocoder — loaded on-demand in token2wav (host-side for now)

        logger.info("Pipeline loaded successfully")
        return cls(llm, flow_decoder, weights_dir, mesh_device)

    def token2wav(self, speech_tokens, prompt_speech_tokens, prompt_feat, embedding):
        """
        Convert speech tokens to audio waveform.

        Args:
            speech_tokens: (1, N) - generated speech token IDs
            prompt_speech_tokens: (1, M) - prompt speech token IDs
            prompt_feat: (1, mel_frames, 80) - prompt mel features
            embedding: (1, 192) - speaker embedding

        Returns:
            mel: (1, 80, generated_frames) - generated mel spectrogram
        """
        token = speech_tokens.to(torch.int32)
        prompt_token = prompt_speech_tokens.to(torch.int32)

        mel, _ = self.flow.inference(
            token=token,
            token_len=torch.tensor([token.shape[1]], dtype=torch.int32),
            prompt_token=prompt_token,
            prompt_token_len=torch.tensor([prompt_token.shape[1]], dtype=torch.int32),
            prompt_feat=prompt_feat,
            prompt_feat_len=torch.tensor([prompt_feat.shape[1]], dtype=torch.int32),
            embedding=embedding,
            streaming=False,
            finalize=True,
        )
        return mel

    @torch.inference_mode()
    def tts(
        self,
        text_tokens,
        prompt_text_tokens,
        prompt_speech_tokens,
        prompt_feat,
        embedding,
        max_token_text_ratio=20,
        min_token_text_ratio=2,
    ):
        """
        Run full TTS pipeline: text → speech tokens → mel spectrogram.

        Args:
            text_tokens: (1, N) - target text token IDs
            prompt_text_tokens: (1, M) - prompt text token IDs
            prompt_speech_tokens: (1, K) - prompt speech token IDs
            prompt_feat: (1, mel_frames, 80) - prompt mel features
            embedding: (1, 192) - speaker embedding

        Returns:
            dict with:
                'speech_tokens': list of generated speech token IDs
                'mel': (1, 80, frames) - generated mel spectrogram
        """
        logger.info("Starting TTS pipeline...")

        # Stage 1: LLM — generate speech tokens
        logger.info("Stage 1: LLM inference (speech token generation)")
        speech_token_list = []
        token_generator = self.llm.inference(
            text=text_tokens,
            text_len=torch.tensor([text_tokens.shape[1]], dtype=torch.int32),
            prompt_text=prompt_text_tokens,
            prompt_text_len=torch.tensor([prompt_text_tokens.shape[1]], dtype=torch.int32),
            prompt_speech_token=prompt_speech_tokens,
            prompt_speech_token_len=torch.tensor([prompt_speech_tokens.shape[1]], dtype=torch.int32),
            embedding=embedding,
        )
        for token_id in token_generator:
            speech_token_list.append(token_id)

        logger.info(f"Stage 1 complete: generated {len(speech_token_list)} speech tokens")

        if len(speech_token_list) == 0:
            logger.warning("LLM generated 0 tokens!")
            return {"speech_tokens": [], "mel": None}

        # Stage 2: Flow Decoder — speech tokens → mel spectrogram
        logger.info("Stage 2: Flow decoder (speech tokens → mel)")
        speech_tokens = torch.tensor(speech_token_list).unsqueeze(0)  # (1, N)
        mel = self.token2wav(speech_tokens, prompt_speech_tokens, prompt_feat, embedding)
        logger.info(f"Stage 2 complete: mel shape = {mel.shape}")

        return {
            "speech_tokens": speech_token_list,
            "mel": mel,
        }
