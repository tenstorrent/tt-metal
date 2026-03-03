# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
Qwen3-TTS full model implementation.

This module combines the Talker and CodePredictor models to form the complete
Qwen3-TTS text-to-speech system.
"""

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.qwen3_tts.tt.code_predictor import CodePredictor
from models.demos.qwen3_tts.tt.model_config import Qwen3TTSCodePredictorConfig, Qwen3TTSTalkerConfig
from models.demos.qwen3_tts.tt.talker import Talker


class Qwen3TTS(LightweightModule):
    """
    Qwen3-TTS full model.

    Architecture:
        - Talker: 28-layer transformer that processes input codec embeddings
        - CodePredictor: 5-layer transformer that predicts audio codec tokens

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

        # Initialize Talker
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
    ) -> list:
        """
        Forward pass of the full Qwen3-TTS model.

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

        Returns:
            List of logits tensors, one per code group [batch, 1, seq_len, vocab_size]
        """
        # Run Talker to get hidden states
        hidden_states = self.talker(
            input_ids,
            talker_cos,
            talker_sin,
            talker_transformation_mat,
            attention_mask,
            use_text_embedding=use_text_embedding,
        )

        # Run CodePredictor to get logits
        logits_list = self.code_predictor(
            hidden_states,
            code_predictor_cos,
            code_predictor_sin,
            code_predictor_transformation_mat,
            attention_mask,
        )

        return logits_list

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
    ) -> list:
        """
        Prefill pass (same as forward for now, KV-cache to be added).

        Args:
            Same as forward()

        Returns:
            List of logits tensors, one per code group
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
