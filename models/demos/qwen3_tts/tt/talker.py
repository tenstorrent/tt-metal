# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
Talker model implementation for Qwen3-TTS.

The Talker is a 28-layer transformer decoder that processes codec embeddings
and generates hidden states for the CodePredictor.
"""

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.qwen3_tts.tt.decoder_layer import DecoderLayer
from models.demos.qwen3_tts.tt.rmsnorm import RMSNorm


class Talker(LightweightModule):
    """
    Qwen3-TTS Talker model.

    Architecture:
        - Codec embedding layer
        - 28 decoder layers with MROPE
        - Final RMSNorm

    Args:
        device: TTNN device
        config: Talker configuration (Qwen3TTSTalkerConfig)
        state_dict: Model weights
        weight_cache_path: Optional path for weight caching
    """

    def __init__(
        self,
        device,
        config,
        state_dict: dict,
        weight_cache_path=None,
    ):
        super().__init__()
        self.device = device
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_hidden_layers
        self.vocab_size = config.audio_vocab_size

        is_mesh_device = device.__class__.__name__ == "MeshDevice"

        def get_cache_name(name):
            if weight_cache_path is None:
                return None
            return weight_cache_path / f"talker_{name}".replace(".", "_")

        # Codec embedding (for audio codec tokens)
        codec_embedding_weight = state_dict["talker.model.codec_embedding.weight"]
        self.codec_embedding = ttnn.as_tensor(
            codec_embedding_weight.unsqueeze(0).unsqueeze(0),
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=get_cache_name("codec_embedding"),
            mesh_mapper=ttnn.ReplicateTensorToMesh(device) if is_mesh_device else None,
        )

        # Text embedding (for text tokens - used in real TTS)
        if "talker.model.text_embedding.weight" in state_dict:
            text_embedding_weight = state_dict["talker.model.text_embedding.weight"]
            self.text_embedding = ttnn.as_tensor(
                text_embedding_weight.unsqueeze(0).unsqueeze(0),
                device=device,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cache_file_name=get_cache_name("text_embedding"),
                mesh_mapper=ttnn.ReplicateTensorToMesh(device) if is_mesh_device else None,
            )
            self.text_vocab_size = text_embedding_weight.shape[0]
        else:
            self.text_embedding = None
            self.text_vocab_size = 0

        # Decoder layers
        self.layers = []
        for i in range(self.num_layers):
            layer = DecoderLayer(
                device=device,
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                num_kv_heads=config.num_key_value_heads,
                head_dim=config.head_dim,
                intermediate_size=config.intermediate_size,
                state_dict=state_dict,
                layer_idx=i,
                layer_prefix="talker.model",
                rms_norm_eps=config.rms_norm_eps,
                weight_dtype=ttnn.bfloat16,
                weight_cache_path=weight_cache_path,
            )
            self.layers.append(layer)

        # Final layer norm
        self.norm = RMSNorm(
            device=device,
            dim=config.hidden_size,
            state_dict=state_dict,
            weight_key="talker.model.norm.weight",
            eps=config.rms_norm_eps,
            weight_dtype=ttnn.bfloat16,
            weight_cache_path=weight_cache_path,
        )

        # Codec head for predicting first RVQ codebook (vocab 3072)
        # This is used during autoregressive generation
        codec_head_key = "talker.codec_head.weight"
        if codec_head_key in state_dict:
            codec_head_weight = state_dict[codec_head_key]
            # Shape: [3072, 2048] -> transpose to [2048, 3072] for matmul
            codec_head_weight = torch.transpose(codec_head_weight, -2, -1).unsqueeze(0).unsqueeze(0)
            self.codec_head = ttnn.as_tensor(
                codec_head_weight,
                device=device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cache_file_name=get_cache_name("codec_head"),
                mesh_mapper=ttnn.ReplicateTensorToMesh(device) if is_mesh_device else None,
            )
            self.codec_head_vocab_size = state_dict[codec_head_key].shape[0]  # 3072
        else:
            self.codec_head = None
            self.codec_head_vocab_size = 0

        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def forward(
        self,
        input_ids: ttnn.Tensor,
        cos: ttnn.Tensor,
        sin: ttnn.Tensor,
        transformation_mat: ttnn.Tensor,
        attention_mask: ttnn.Tensor = None,
        use_text_embedding: bool = False,
    ) -> ttnn.Tensor:
        """
        Forward pass of the Talker model.

        Args:
            input_ids: Input token IDs [batch, seq_len]
            cos: Cosine frequencies for RoPE
            sin: Sine frequencies for RoPE
            transformation_mat: Transformation matrix for RoPE
            attention_mask: Optional attention mask
            use_text_embedding: If True, use text embedding (for text tokens in TTS)
                               If False, use codec embedding (for audio codec tokens)

        Returns:
            Hidden states [batch, 1, seq_len, hidden_size]
        """
        # Choose embedding based on input type
        if use_text_embedding and self.text_embedding is not None:
            embedding_weight = self.text_embedding
        else:
            embedding_weight = self.codec_embedding

        # Embedding lookup
        hidden_states = ttnn.embedding(
            input_ids,
            embedding_weight,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        return self._forward_layers(hidden_states, cos, sin, transformation_mat, attention_mask)

    def forward_from_hidden(
        self,
        hidden_states: ttnn.Tensor,
        cos: ttnn.Tensor,
        sin: ttnn.Tensor,
        transformation_mat: ttnn.Tensor,
        attention_mask: ttnn.Tensor = None,
    ) -> ttnn.Tensor:
        """
        Forward pass starting from hidden states (for mixed embeddings).

        Args:
            hidden_states: Pre-computed hidden states [batch, 1, seq_len, hidden_size]
            cos: Cosine frequencies for RoPE
            sin: Sine frequencies for RoPE
            transformation_mat: Transformation matrix for RoPE
            attention_mask: Optional attention mask

        Returns:
            Hidden states [batch, 1, seq_len, hidden_size]
        """
        return self._forward_layers(hidden_states, cos, sin, transformation_mat, attention_mask)

    def _forward_layers(
        self,
        hidden_states: ttnn.Tensor,
        cos: ttnn.Tensor,
        sin: ttnn.Tensor,
        transformation_mat: ttnn.Tensor,
        attention_mask: ttnn.Tensor = None,
    ) -> ttnn.Tensor:
        """Internal: Apply decoder layers and final norm."""
        # Add batch dimension if needed: [batch, seq_len, hidden] -> [batch, 1, seq_len, hidden]
        if len(hidden_states.shape) == 3 or hidden_states.shape[1] != 1:
            if len(hidden_states.shape) == 3:
                hidden_states = ttnn.reshape(
                    hidden_states,
                    (hidden_states.shape[0], 1, hidden_states.shape[1], hidden_states.shape[2]),
                )

        # Apply decoder layers
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                cos,
                sin,
                transformation_mat,
                attention_mask,
            )

        # Final norm
        hidden_states = self.norm(hidden_states)

        return hidden_states

    def get_codec_logits(self, hidden_states: ttnn.Tensor) -> ttnn.Tensor:
        """
        Apply codec_head to get logits for first RVQ codebook prediction.

        Args:
            hidden_states: Hidden states from Talker [batch, 1, seq_len, hidden_size]

        Returns:
            Logits [batch, 1, seq_len, 3072]
        """
        if self.codec_head is None:
            raise ValueError("codec_head not loaded. Cannot compute codec logits.")

        logits = ttnn.linear(
            hidden_states,
            self.codec_head,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return logits
