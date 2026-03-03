# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
CodePredictor model implementation for Qwen3-TTS.

The CodePredictor is a 5-layer transformer decoder that takes hidden states
from the Talker and predicts audio codec tokens for multiple code groups.
"""

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.qwen3_tts.tt.decoder_layer import DecoderLayer


class CodePredictor(LightweightModule):
    """
    Qwen3-TTS CodePredictor model.

    Architecture:
        - Input projection (from Talker hidden_size to CodePredictor hidden_size)
        - 5 decoder layers with standard RoPE
        - Multiple LM heads (one per code group)

    Args:
        device: TTNN device
        config: CodePredictor configuration (Qwen3TTSCodePredictorConfig)
        talker_hidden_size: Hidden size of the Talker model (for input projection)
        state_dict: Model weights
        weight_cache_path: Optional path for weight caching
    """

    def __init__(
        self,
        device,
        config,
        talker_hidden_size: int,
        state_dict: dict,
        weight_cache_path=None,
    ):
        super().__init__()
        self.device = device
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_hidden_layers
        self.num_code_groups = config.num_code_groups
        self.vocab_size = config.vocab_size

        is_mesh_device = device.__class__.__name__ == "MeshDevice"

        def get_cache_name(name):
            if weight_cache_path is None:
                return None
            return weight_cache_path / f"code_predictor_{name}".replace(".", "_")

        # Input projection (if Talker and CodePredictor have different hidden sizes)
        # The projection is called "small_to_mtp_projection" in HuggingFace model
        self.needs_projection = talker_hidden_size != config.hidden_size
        if self.needs_projection:
            # Project from talker hidden size to code predictor hidden size
            proj_key = "talker.code_predictor.small_to_mtp_projection.weight"
            if proj_key in state_dict:
                proj_weight = state_dict[proj_key]
                # Shape: [1024, 2048] -> transpose to [2048, 1024] for matmul
                proj_weight = torch.transpose(proj_weight, -2, -1).unsqueeze(0).unsqueeze(0)
                self.input_proj = ttnn.as_tensor(
                    proj_weight,
                    device=device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    cache_file_name=get_cache_name("input_proj"),
                    mesh_mapper=ttnn.ReplicateTensorToMesh(device) if is_mesh_device else None,
                )
                # Also load bias if present
                bias_key = "talker.code_predictor.small_to_mtp_projection.bias"
                if bias_key in state_dict:
                    bias = state_dict[bias_key].unsqueeze(0).unsqueeze(0).unsqueeze(0)
                    self.input_proj_bias = ttnn.as_tensor(
                        bias,
                        device=device,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        cache_file_name=get_cache_name("input_proj_bias"),
                        mesh_mapper=ttnn.ReplicateTensorToMesh(device) if is_mesh_device else None,
                    )
                else:
                    self.input_proj_bias = None
            else:
                # If no projection weight, assume sizes match or use identity
                self.needs_projection = False

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
                layer_prefix="talker.code_predictor.model",
                rms_norm_eps=config.rms_norm_eps,
                weight_dtype=ttnn.bfloat16,
                weight_cache_path=weight_cache_path,
            )
            self.layers.append(layer)

        # LM heads (one per code group)
        self.lm_heads = []
        for g in range(self.num_code_groups):
            lm_head_key = f"talker.code_predictor.lm_head.{g}.weight"
            if lm_head_key in state_dict:
                lm_head_weight = state_dict[lm_head_key]
                lm_head_weight = torch.transpose(lm_head_weight, -2, -1).unsqueeze(0).unsqueeze(0)
                lm_head = ttnn.as_tensor(
                    lm_head_weight,
                    device=device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    cache_file_name=get_cache_name(f"lm_head_{g}"),
                    mesh_mapper=ttnn.ReplicateTensorToMesh(device) if is_mesh_device else None,
                )
                self.lm_heads.append(lm_head)

        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        cos: ttnn.Tensor,
        sin: ttnn.Tensor,
        transformation_mat: ttnn.Tensor,
        attention_mask: ttnn.Tensor = None,
    ) -> list:
        """
        Forward pass of the CodePredictor model.

        Args:
            hidden_states: Input hidden states from Talker [batch, 1, seq_len, talker_hidden_size]
            cos: Cosine frequencies for RoPE
            sin: Sine frequencies for RoPE
            transformation_mat: Transformation matrix for RoPE
            attention_mask: Optional attention mask

        Returns:
            List of logits tensors, one per code group [batch, 1, seq_len, vocab_size]
        """
        # Input projection if needed (from Talker's 2048 dim to CodePredictor's 1024 dim)
        if self.needs_projection:
            hidden_states = ttnn.linear(
                hidden_states,
                self.input_proj,
                bias=self.input_proj_bias if hasattr(self, "input_proj_bias") else None,
                compute_kernel_config=self.compute_kernel_config,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
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

        # Compute logits for each code group
        logits_list = []
        for lm_head in self.lm_heads:
            logits = ttnn.linear(
                hidden_states,
                lm_head,
                compute_kernel_config=self.compute_kernel_config,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            logits_list.append(logits)

        return logits_list
