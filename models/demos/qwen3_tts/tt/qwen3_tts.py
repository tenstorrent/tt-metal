# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
Qwen3-TTS full model implementation.

This module combines the Talker, CodePredictor, and SpeakerEncoder models
to form the complete Qwen3-TTS text-to-speech system.

Components (all in model weights):
    - Talker: 28-layer transformer (talker.model.*)
    - Talker codec_head: LM head for code 0 (talker.codec_head.*)
    - Text projection: MLP for text embeddings (talker.text_projection.*)
    - CodePredictor: 5-layer transformer + 15 LM heads (talker.code_predictor.*)
    - Speaker encoder: ECAPA-TDNN for speaker embeddings (speaker_encoder.*)

Supports both prefill mode (full sequence) and decode mode (single token with KV cache).
"""

from typing import List, Optional, Tuple

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.qwen3_tts.tt.code_predictor import CodePredictor
from models.demos.qwen3_tts.tt.model_config import Qwen3TTSCodePredictorConfig, Qwen3TTSTalkerConfig
from models.demos.qwen3_tts.tt.speaker_encoder import SpeakerEncoder
from models.demos.qwen3_tts.tt.talker import Talker


class Qwen3TTS(LightweightModule):
    """
    Qwen3-TTS full model.

    Architecture:
        - SpeakerEncoder: ECAPA-TDNN for extracting speaker embeddings from audio
        - Talker: 28-layer transformer that processes input codec embeddings
          - text_projection: MLP to project text embeddings
          - codec_head: LM head for predicting code 0
        - CodePredictor: 5-layer transformer that predicts audio codec tokens 1-15

    The model takes codec token IDs as input and outputs logits for predicting
    the next codec tokens across multiple code groups.

    Args:
        device: TTNN device
        state_dict: Model weights
        talker_config: Optional Talker configuration
        code_predictor_config: Optional CodePredictor configuration
        weight_cache_path: Optional path for weight caching
    """

    def __init__(
        self,
        device,
        state_dict: dict,
        talker_config: Qwen3TTSTalkerConfig = None,
        code_predictor_config: Qwen3TTSCodePredictorConfig = None,
        weight_cache_path=None,
    ):
        super().__init__()
        self.device = device

        # Use default configs if not provided
        if talker_config is None:
            talker_config = Qwen3TTSTalkerConfig()
        if code_predictor_config is None:
            code_predictor_config = Qwen3TTSCodePredictorConfig()

        self.talker_config = talker_config
        self.code_predictor_config = code_predictor_config

        # Initialize Talker (includes text_projection and codec_head)
        self.talker = Talker(
            device=device,
            config=talker_config,
            state_dict=state_dict,
            weight_cache_path=weight_cache_path,
        )

        # Initialize CodePredictor
        self.code_predictor = CodePredictor(
            device=device,
            config=code_predictor_config,
            talker_hidden_size=talker_config.hidden_size,
            state_dict=state_dict,
            weight_cache_path=weight_cache_path,
        )

        # Initialize Speaker Encoder
        self.speaker_encoder = SpeakerEncoder(
            device=device,
            state_dict=state_dict,
            weight_cache_path=weight_cache_path,
        )

    # =========================================================================
    # Embedding and projection methods (delegated to Talker)
    # =========================================================================

    def get_text_embedding(self, text_ids: ttnn.Tensor) -> ttnn.Tensor:
        """Get text embeddings for token IDs."""
        return self.talker.get_text_embedding(text_ids)

    def get_codec_embedding(self, codec_ids: ttnn.Tensor) -> ttnn.Tensor:
        """Get codec embeddings for token IDs."""
        return self.talker.get_codec_embedding(codec_ids)

    def project_text(self, text_embeds: ttnn.Tensor) -> ttnn.Tensor:
        """Apply text projection MLP to text embeddings."""
        return self.talker.project_text(text_embeds)

    # =========================================================================
    # Speaker encoder methods
    # =========================================================================

    def extract_speaker_embedding(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Extract speaker embedding from audio waveform.

        Args:
            audio: Audio waveform [num_samples] or [batch, num_samples]

        Returns:
            Speaker embedding [batch, 2048] as PyTorch tensor
        """
        return self.speaker_encoder.forward_from_audio(audio)

    def speaker_embedding_to_ttnn(self, speaker_embedding: torch.Tensor) -> ttnn.Tensor:
        """
        Convert speaker embedding to TTNN tensor.

        Args:
            speaker_embedding: PyTorch tensor [batch, 2048]

        Returns:
            TTNN tensor [batch, 1, 1, 2048]
        """
        return self.speaker_encoder.to_ttnn(speaker_embedding)

    def forward(
        self,
        input_ids: ttnn.Tensor,
        talker_cos: ttnn.Tensor,
        talker_sin: ttnn.Tensor,
        talker_transformation_mat: ttnn.Tensor,
        code_predictor_cos: ttnn.Tensor,
        code_predictor_sin: ttnn.Tensor,
        code_predictor_transformation_mat: ttnn.Tensor,
        attention_mask: ttnn.Tensor = None,
        use_text_embedding: bool = False,
        talker_kv_caches: Optional[List[Tuple[ttnn.Tensor, ttnn.Tensor]]] = None,
        cp_kv_caches: Optional[List[Tuple[ttnn.Tensor, ttnn.Tensor]]] = None,
        start_pos: int = 0,
        mode: str = "prefill",
    ) -> Tuple[
        List[ttnn.Tensor],
        Optional[List[Tuple[ttnn.Tensor, ttnn.Tensor]]],
        Optional[List[Tuple[ttnn.Tensor, ttnn.Tensor]]],
    ]:
        """
        Forward pass of the full Qwen3-TTS model.

        Supports both prefill (full sequence) and decode (single token) modes.

        Args:
            input_ids: Input token IDs [batch, seq_len]
            talker_cos: Cosine frequencies for Talker RoPE (MROPE format)
            talker_sin: Sine frequencies for Talker RoPE (MROPE format)
            talker_transformation_mat: Transformation matrix for Talker RoPE
            code_predictor_cos: Cosine frequencies for CodePredictor RoPE
            code_predictor_sin: Sine frequencies for CodePredictor RoPE
            code_predictor_transformation_mat: Transformation matrix for CodePredictor RoPE
            attention_mask: Optional attention mask
            use_text_embedding: If True, use text embedding for text tokens (real TTS mode)
                               If False, use codec embedding for audio codec tokens (benchmark mode)
            talker_kv_caches: Optional KV caches for Talker (list per layer)
            cp_kv_caches: Optional KV caches for CodePredictor (list per layer)
            start_pos: Starting position in sequence (for KV cache)
            mode: "prefill" for full sequence or "decode" for single token

        Returns:
            Tuple of (codec_logits, cp_logits_list, updated_talker_kv_caches, updated_cp_kv_caches) where:
            - codec_logits: Logits for code 0 from Talker's codec_head [batch, seq_len, vocab_size]
            - cp_logits_list: List of logits for codes 1-15 from CodePredictor
            - updated_talker_kv_caches: Updated Talker KV caches or None
            - updated_cp_kv_caches: Updated CodePredictor KV caches or None
        """
        # Run Talker to get hidden states
        hidden_states, updated_talker_kv_caches = self.talker(
            input_ids,
            talker_cos,
            talker_sin,
            talker_transformation_mat,
            attention_mask,
            use_text_embedding=use_text_embedding,
            kv_caches=talker_kv_caches,
            start_pos=start_pos,
            mode=mode,
        )

        # Get codec logits (code 0) from Talker's codec_head
        codec_logits = self.talker.get_codec_logits(hidden_states)

        # Run CodePredictor to get logits for codes 1-15
        cp_logits_list, updated_cp_kv_caches = self.code_predictor(
            hidden_states,
            code_predictor_cos,
            code_predictor_sin,
            code_predictor_transformation_mat,
            attention_mask,
            kv_caches=cp_kv_caches,
            start_pos=start_pos,
            mode=mode,
        )

        return codec_logits, cp_logits_list, updated_talker_kv_caches, updated_cp_kv_caches

    def forward_from_hidden(
        self,
        hidden_states: ttnn.Tensor,
        talker_cos: ttnn.Tensor,
        talker_sin: ttnn.Tensor,
        talker_transformation_mat: ttnn.Tensor,
        code_predictor_cos: ttnn.Tensor,
        code_predictor_sin: ttnn.Tensor,
        code_predictor_transformation_mat: ttnn.Tensor,
        attention_mask: ttnn.Tensor = None,
        talker_kv_caches: Optional[List[Tuple[ttnn.Tensor, ttnn.Tensor]]] = None,
        cp_kv_caches: Optional[List[Tuple[ttnn.Tensor, ttnn.Tensor]]] = None,
        start_pos: int = 0,
        mode: str = "prefill",
    ) -> Tuple[
        ttnn.Tensor,
        List[ttnn.Tensor],
        Optional[List[Tuple[ttnn.Tensor, ttnn.Tensor]]],
        Optional[List[Tuple[ttnn.Tensor, ttnn.Tensor]]],
    ]:
        """
        Forward pass starting from pre-computed hidden states (for ICL mode).

        This is used when input consists of mixed embeddings (text + codec + speaker)
        that have been pre-computed outside the model.

        Args:
            hidden_states: Pre-computed embeddings [batch, 1, seq_len, hidden_size]
            talker_cos: Cosine frequencies for Talker RoPE
            talker_sin: Sine frequencies for Talker RoPE
            talker_transformation_mat: Transformation matrix for Talker RoPE
            code_predictor_cos: Cosine frequencies for CodePredictor RoPE
            code_predictor_sin: Sine frequencies for CodePredictor RoPE
            code_predictor_transformation_mat: Transformation matrix for CodePredictor RoPE
            attention_mask: Optional attention mask
            talker_kv_caches: Optional KV caches for Talker
            cp_kv_caches: Optional KV caches for CodePredictor
            start_pos: Starting position in sequence
            mode: "prefill" or "decode"

        Returns:
            Tuple of (codec_logits, cp_logits_list, updated_talker_kv_caches, updated_cp_kv_caches)
        """
        # Run Talker from hidden states
        talker_hidden, updated_talker_kv_caches = self.talker.forward_from_hidden(
            hidden_states,
            talker_cos,
            talker_sin,
            talker_transformation_mat,
            attention_mask,
            kv_caches=talker_kv_caches,
            start_pos=start_pos,
            mode=mode,
        )

        # Get codec logits (code 0) from Talker's codec_head
        codec_logits = self.talker.get_codec_logits(talker_hidden)

        # Run CodePredictor to get logits for codes 1-15
        cp_logits_list, updated_cp_kv_caches = self.code_predictor(
            talker_hidden,
            code_predictor_cos,
            code_predictor_sin,
            code_predictor_transformation_mat,
            attention_mask,
            kv_caches=cp_kv_caches,
            start_pos=start_pos,
            mode=mode,
        )

        return codec_logits, cp_logits_list, updated_talker_kv_caches, updated_cp_kv_caches

    def prefill(
        self,
        input_ids: ttnn.Tensor,
        talker_cos: ttnn.Tensor,
        talker_sin: ttnn.Tensor,
        talker_transformation_mat: ttnn.Tensor,
        code_predictor_cos: ttnn.Tensor,
        code_predictor_sin: ttnn.Tensor,
        code_predictor_transformation_mat: ttnn.Tensor,
        attention_mask: ttnn.Tensor = None,
        use_text_embedding: bool = False,
        talker_kv_caches: Optional[List[Tuple[ttnn.Tensor, ttnn.Tensor]]] = None,
        cp_kv_caches: Optional[List[Tuple[ttnn.Tensor, ttnn.Tensor]]] = None,
    ) -> Tuple[
        ttnn.Tensor,
        List[ttnn.Tensor],
        Optional[List[Tuple[ttnn.Tensor, ttnn.Tensor]]],
        Optional[List[Tuple[ttnn.Tensor, ttnn.Tensor]]],
    ]:
        """
        Prefill pass - processes full sequence and populates KV caches.

        Args:
            Same as forward(), plus KV cache parameters

        Returns:
            Tuple of (logits_list, updated_talker_kv_caches, updated_cp_kv_caches)
        """
        return self.forward(
            input_ids,
            talker_cos,
            talker_sin,
            talker_transformation_mat,
            code_predictor_cos,
            code_predictor_sin,
            code_predictor_transformation_mat,
            attention_mask,
            use_text_embedding=use_text_embedding,
            talker_kv_caches=talker_kv_caches,
            cp_kv_caches=cp_kv_caches,
            start_pos=0,
            mode="prefill",
        )

    def decode(
        self,
        input_ids: ttnn.Tensor,
        talker_cos: ttnn.Tensor,
        talker_sin: ttnn.Tensor,
        talker_transformation_mat: ttnn.Tensor,
        code_predictor_cos: ttnn.Tensor,
        code_predictor_sin: ttnn.Tensor,
        code_predictor_transformation_mat: ttnn.Tensor,
        talker_kv_caches: List[Tuple[ttnn.Tensor, ttnn.Tensor]],
        cp_kv_caches: List[Tuple[ttnn.Tensor, ttnn.Tensor]],
        start_pos: int,
        attention_mask: ttnn.Tensor = None,
        use_text_embedding: bool = False,
    ) -> Tuple[List[ttnn.Tensor], List[Tuple[ttnn.Tensor, ttnn.Tensor]], List[Tuple[ttnn.Tensor, ttnn.Tensor]]]:
        """
        Decode pass - processes single token using KV caches.

        Args:
            input_ids: Single token input [batch, 1]
            *_cos, *_sin, *_transformation_mat: RoPE tensors for single position
            talker_kv_caches, cp_kv_caches: KV caches from prefill or previous decode
            start_pos: Current position in sequence
            attention_mask: Optional attention mask
            use_text_embedding: Whether to use text embedding

        Returns:
            Tuple of (logits_list, updated_talker_kv_caches, updated_cp_kv_caches)
        """
        return self.forward(
            input_ids,
            talker_cos,
            talker_sin,
            talker_transformation_mat,
            code_predictor_cos,
            code_predictor_sin,
            code_predictor_transformation_mat,
            attention_mask,
            use_text_embedding=use_text_embedding,
            talker_kv_caches=talker_kv_caches,
            cp_kv_caches=cp_kv_caches,
            start_pos=start_pos,
            mode="decode",
        )


def create_qwen3_tts_model(
    device,
    state_dict: dict,
    weight_cache_path=None,
) -> Qwen3TTS:
    """
    Factory function to create a Qwen3-TTS model with default configurations.

    Args:
        device: TTNN device
        state_dict: Model weights from HuggingFace
        weight_cache_path: Optional path for weight caching

    Returns:
        Initialized Qwen3TTS model
    """
    return Qwen3TTS(
        device=device,
        state_dict=state_dict,
        talker_config=Qwen3TTSTalkerConfig(),
        code_predictor_config=Qwen3TTSCodePredictorConfig(),
        weight_cache_path=weight_cache_path,
    )
