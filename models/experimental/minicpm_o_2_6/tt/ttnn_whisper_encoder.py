# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
TTNN Whisper Audio Encoder for MiniCPM-o-2_6

Adapted from models/demos/whisper/tt/ttnn_optimized_functional_whisper.py
Configured for MiniCPM-o-2_6 specifications:
- d_model: 1024
- encoder_layers: 24
- encoder_attention_heads: 16
- encoder_ffn_dim: 4096

Input: Mel spectrograms [batch, 80, time_steps]
Output: Audio features [batch, seq_len, 1024]
"""

import torch
import ttnn
from loguru import logger
from typing import Dict, Any, Optional

# Import functions from existing Whisper implementation
from models.demos.whisper.tt.ttnn_optimized_functional_whisper import (
    WHISPER_MEMORY_CONFIG,
)


class TtnnWhisperEncoder:
    """TTNN Whisper Audio Encoder adapted for MiniCPM-o-2_6"""

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        weights: Optional[Dict[str, torch.Tensor]] = None,
        config: Optional[Dict] = None,
    ):
        """
        Initialize Whisper Encoder with weight loading for MiniCPM-o-2_6

        Args:
            mesh_device: TTNN mesh device
            weights: Pre-loaded weights from MiniCPM checkpoint
            config: Optional configuration overrides
        """
        self.mesh_device = mesh_device
        self.weights = weights
        self.config = config or self._default_config()

        # Set config attributes for backward compatibility
        self.device = mesh_device  # Keep for compatibility
        for key, value in self.config.items():
            setattr(self, key, value)

        # Create config object for compatibility with existing functions
        self.config = self._create_config()

        # Initialize layers (will be populated during weight loading)
        self.encoder_layers_params = []
        self.conv1_params = None
        self.conv2_params = None
        self.embed_positions = None
        self.layer_norm = None

        # Load weights if provided
        if weights is not None:
            self._load_weights(weights)

    def _default_config(self) -> Dict[str, Any]:
        """Default Whisper encoder configuration for MiniCPM-o-2_6"""
        return {
            "d_model": 1024,
            "encoder_layers": 24,
            "encoder_attention_heads": 16,
            "encoder_ffn_dim": 4096,
            "num_mel_bins": 80,
            "max_source_positions": 1500,
            "vocab_size": 51865,
            "dropout": 0.0,
            "layer_norm_eps": 1e-5,
        }

    def _load_weights(self, weights: Dict[str, torch.Tensor]):
        """
        Load PyTorch weights into TTNN tensors and move to device.

        This converts the safetensors weights to TTNN format and loads them
        into the Whisper encoder components.
        """
        logger.info("Loading Whisper encoder weights into TTNN format...")

        # Load convolutional layers
        conv_layers = ["conv1", "conv2"]
        for conv_name in conv_layers:
            weight_key = f"apm.model.encoder.{conv_name}.weight"
            bias_key = f"apm.model.encoder.{conv_name}.bias"

            if weight_key in weights:
                conv_weight = weights[weight_key]
                conv_weight = ttnn.from_torch(
                    conv_weight,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=1),
                )
                conv_weight = ttnn.to_device(conv_weight, self.mesh_device)
                setattr(self, f"{conv_name}_params", {"weight": conv_weight})

            if bias_key in weights:
                conv_bias = weights[bias_key]
                conv_bias = ttnn.from_torch(
                    conv_bias,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=0),
                )
                conv_bias = ttnn.to_device(conv_bias, self.mesh_device)
                if hasattr(self, f"{conv_name}_params"):
                    getattr(self, f"{conv_name}_params")["bias"] = conv_bias
                else:
                    setattr(self, f"{conv_name}_params", {"bias": conv_bias})

        # Load position embeddings
        if "apm.model.encoder.embed_positions.weight" in weights:
            pos_embed = weights["apm.model.encoder.embed_positions.weight"]
            pos_embed = ttnn.from_torch(
                pos_embed,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=1),
            )
            pos_embed = ttnn.to_device(pos_embed, self.mesh_device)
            self.embed_positions = pos_embed

        # Load layer norm
        if "apm.model.encoder.layer_norm.weight" in weights:
            ln_weight = weights["apm.model.encoder.layer_norm.weight"]
            ln_weight = ttnn.from_torch(
                ln_weight,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=0),
            )
            ln_weight = ttnn.to_device(ln_weight, self.mesh_device)
            self.layer_norm = {"weight": ln_weight}

        if "apm.model.encoder.layer_norm.bias" in weights:
            ln_bias = weights["apm.model.encoder.layer_norm.bias"]
            ln_bias = ttnn.from_torch(
                ln_bias,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=0),
            )
            ln_bias = ttnn.to_device(ln_bias, self.mesh_device)
            if self.layer_norm:
                self.layer_norm["bias"] = ln_bias
            else:
                self.layer_norm = {"bias": ln_bias}

        # Load encoder layer weights (simplified - would need full implementation)
        # This is a placeholder showing the pattern
        for layer_idx in range(self.config["encoder_layers"]):
            layer_params = {}
            layer_prefix = f"apm.model.encoder.layers.{layer_idx}"

            # Self-attention weights would go here
            # Cross-attention weights would go here (though Whisper doesn't use cross-attention)
            # MLP weights would go here
            # Layer norms would go here

            if layer_params:  # Only add if we loaded any weights
                self.encoder_layers_params.append(layer_params)
                logger.debug(f"Loaded weights for encoder layer {layer_idx}")

        logger.info("✅ Whisper encoder weights loaded into TTNN format")

    def _create_config(self):
        """Create config object compatible with existing Whisper functions"""

        class Config:
            def __init__(self, parent):
                self.d_model = parent.d_model
                self.encoder_layers = parent.encoder_layers
                self.encoder_attention_heads = parent.encoder_attention_heads
                self.encoder_ffn_dim = parent.encoder_ffn_dim
                self.num_mel_bins = parent.num_mel_bins
                self.max_source_positions = parent.max_source_positions
                self.vocab_size = parent.vocab_size
                self.dropout = parent.dropout
                self.layer_norm_eps = parent.layer_norm_eps

        return Config(self)

    def load_weights(self, weights_dict: Dict[str, torch.Tensor]) -> None:
        """
        Load weights from dictionary into TTNN format

        Args:
            weights_dict: Dictionary containing model weights
        """
        logger.info("Loading Whisper encoder weights...")

        # Initialize encoder layers
        self.encoder_layers_params = []
        for layer_idx in range(self.encoder_layers):
            layer_params = {
                "self_attn": {
                    "query": {
                        "weight": weights_dict[f"apm.layers.{layer_idx}.self_attn.q_proj.weight"],
                        "bias": torch.zeros(
                            weights_dict[f"apm.layers.{layer_idx}.self_attn.q_proj.weight"].shape[0]
                        ),  # Zero bias for Q
                    },
                    "key": {
                        "weight": weights_dict[f"apm.layers.{layer_idx}.self_attn.k_proj.weight"],
                        "bias": torch.zeros(
                            weights_dict[f"apm.layers.{layer_idx}.self_attn.k_proj.weight"].shape[0]
                        ),  # Zero bias for K
                    },
                    "value": {
                        "weight": weights_dict[f"apm.layers.{layer_idx}.self_attn.v_proj.weight"],
                        "bias": torch.zeros(
                            weights_dict[f"apm.layers.{layer_idx}.self_attn.v_proj.weight"].shape[0]
                        ),  # Zero bias for V
                    },
                    "out_proj": {
                        "weight": weights_dict[f"apm.layers.{layer_idx}.self_attn.out_proj.weight"],
                        "bias": weights_dict[f"apm.layers.{layer_idx}.self_attn.out_proj.bias"],
                    },
                },
                "self_attn_layer_norm": {
                    "weight": weights_dict[f"apm.layers.{layer_idx}.self_attn_layer_norm.weight"],
                    "bias": weights_dict[f"apm.layers.{layer_idx}.self_attn_layer_norm.bias"],
                },
                "fc1": {
                    "weight": weights_dict[f"apm.layers.{layer_idx}.fc1.weight"],
                    "bias": weights_dict[f"apm.layers.{layer_idx}.fc1.bias"],
                },
                "fc2": {
                    "weight": weights_dict[f"apm.layers.{layer_idx}.fc2.weight"],
                    "bias": weights_dict[f"apm.layers.{layer_idx}.fc2.bias"],
                },
                "final_layer_norm": {
                    "weight": weights_dict[f"apm.layers.{layer_idx}.final_layer_norm.weight"],
                    "bias": weights_dict[f"apm.layers.{layer_idx}.final_layer_norm.bias"],
                },
            }
            self.encoder_layers_params.append(layer_params)

        # Conv layers - reshape for conv2d simulation (conv1d weights become conv2d weights)
        self.conv1_params = {
            "weight": weights_dict["apm.conv1.weight"],  # [d_model, num_mel_bins, 3]
            "bias": weights_dict["apm.conv1.bias"],
        }
        # Reshape conv2 weights for conv2d: [d_model, d_model, 3] -> [d_model, d_model, 3, 1]
        conv2_weight = weights_dict["apm.conv2.weight"]
        conv2_weight_reshaped = conv2_weight.unsqueeze(-1)  # [d_model, d_model, 3, 1]
        self.conv2_params = {
            "weight": conv2_weight_reshaped,
            "bias": weights_dict["apm.conv2.bias"],
        }

        # Position embeddings and layer norm
        self.embed_positions = weights_dict["apm.embed_positions.weight"]
        self.layer_norm = {
            "weight": weights_dict["apm.layer_norm.weight"],
            "bias": weights_dict["apm.layer_norm.bias"],
        }

        logger.info(f"Loaded weights for {self.encoder_layers} encoder layers")

    def _convert_weights_to_ttnn(self, weights_mesh_mapper=None):
        """
        Convert PyTorch weights to TTNN format

        Args:
            weights_mesh_mapper: Mesh mapper for multi-device setups

        Returns:
            Dictionary with TTNN-compatible parameters
        """
        parameters = {}

        # Encoder layers
        parameters["layers"] = []
        for layer_idx, layer_params in enumerate(self.encoder_layers_params):
            layer_ttnn = {}

            # Self-attention
            layer_ttnn["self_attn"] = {}
            for key in ["query", "key", "value", "out_proj"]:
                weight = layer_params["self_attn"][key]["weight"]
                bias = layer_params["self_attn"][key]["bias"]

                # Transpose weights for TTNN linear (in_features, out_features)
                weight_ttnn = ttnn.from_torch(
                    weight.t(),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                    mesh_mapper=weights_mesh_mapper,
                )

                bias_ttnn = ttnn.from_torch(
                    bias,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                    mesh_mapper=weights_mesh_mapper,
                )

                layer_ttnn["self_attn"][key] = {
                    "weight": weight_ttnn,
                    "bias": bias_ttnn,
                }

            # Layer norms
            for key in ["self_attn_layer_norm", "final_layer_norm"]:
                weight = layer_params[key]["weight"]
                bias = layer_params[key]["bias"]

                weight_ttnn = ttnn.from_torch(
                    weight,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                    mesh_mapper=weights_mesh_mapper,
                )

                bias_ttnn = ttnn.from_torch(
                    bias,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                    mesh_mapper=weights_mesh_mapper,
                )

                layer_ttnn[key] = {
                    "weight": weight_ttnn,
                    "bias": bias_ttnn,
                }

            # Feed-forward
            for key in ["fc1", "fc2"]:
                weight = layer_params[key]["weight"]
                bias = layer_params[key]["bias"]

                # Transpose weights for TTNN linear
                weight_ttnn = ttnn.from_torch(
                    weight.t(),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                    mesh_mapper=weights_mesh_mapper,
                )

                bias_ttnn = ttnn.from_torch(
                    bias,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                    mesh_mapper=weights_mesh_mapper,
                )

                layer_ttnn[key] = {
                    "weight": weight_ttnn,
                    "bias": bias_ttnn,
                }

            parameters["layers"].append(layer_ttnn)

        # Conv weights for conv1d/conv2d simulation (keep in ROW_MAJOR_LAYOUT)
        parameters["conv1"] = {
            "weight": ttnn.from_torch(
                self.conv1_params["weight"],
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=weights_mesh_mapper,
            ),
            "bias": ttnn.from_torch(
                self.conv1_params["bias"],
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=weights_mesh_mapper,
            ),
        }

        parameters["conv2"] = {
            "weight": ttnn.from_torch(
                self.conv2_params["weight"],
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=weights_mesh_mapper,
            ),
            "bias": ttnn.from_torch(
                self.conv2_params["bias"],
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=weights_mesh_mapper,
            ),
        }

        # Position embeddings and final layer norm
        parameters["embed_positions"] = {
            "weight": ttnn.from_torch(
                self.embed_positions,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                mesh_mapper=weights_mesh_mapper,
            ),
        }

        parameters["layer_norm"] = {
            "weight": ttnn.from_torch(
                self.layer_norm["weight"],
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                mesh_mapper=weights_mesh_mapper,
            ),
            "bias": ttnn.from_torch(
                self.layer_norm["bias"],
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                mesh_mapper=weights_mesh_mapper,
            ),
        }

        return parameters

    def forward(
        self,
        input_features: torch.Tensor,
        weights_mesh_mapper=None,
        input_mesh_mapper=None,
    ) -> ttnn.Tensor:
        """
        Forward pass through Whisper encoder

        Args:
            input_features: Mel spectrograms [batch, num_mel_bins, time_steps]
            weights_mesh_mapper: Mesh mapper for weights
            input_mesh_mapper: Mesh mapper for inputs

        Returns:
            Audio features [batch, seq_len, d_model]
        """
        # Convert weights to TTNN format
        parameters = self._convert_weights_to_ttnn(weights_mesh_mapper)

        # Preprocess inputs (conv1d processing) using MiniCPM-adapted function
        input_embeds = preprocess_encoder_inputs_minicpm(
            config=self.config,
            input_features=input_features,
            parameters=parameters,
            device=self.device,
        )

        # Run encoder layers using MiniCPM-adapted function
        output = encoder_minicpm(
            config=self.config,
            inputs_embeds=input_embeds,
            parameters=parameters,
        )

        return output


def get_conv_configs_minicpm(device):
    """Get conv1d configurations for MiniCPM Whisper encoder."""
    conv1d_config = ttnn.Conv1dConfig(
        weights_dtype=ttnn.bfloat16,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    )
    conv1d_compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )
    return conv1d_config, conv1d_compute_config


def preprocess_encoder_inputs_minicpm(config, input_features, *, parameters, device):
    """
    Adapted preprocess_encoder_inputs for MiniCPM dict-based parameters.
    Uses conv2d to simulate conv1d following SpeechT5 approach.

    Args:
        config: WhisperConfig
        input_features: Input mel spectrograms [batch, seq_len, num_mel_bins]
        parameters: Dict-based parameters
        device: TTNN device

    Returns:
        Processed input embeddings [batch, seq_len//2, d_model]
    """
    batch_size, seq_len, num_mel_bins = input_features.shape

    # Convert to TTNN: [B, seq_len, num_mel_bins] -> [B, num_mel_bins, seq_len]
    input_features = ttnn.from_torch(input_features, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    input_features = ttnn.transpose(input_features, 1, 2)  # [B, num_mel_bins, seq_len]

    # Get conv configs for conv2d
    conv_config = ttnn.Conv2dConfig(
        weights_dtype=ttnn.bfloat16,
        shard_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
    )
    conv_compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    # Conv1: Simulate 1D conv with conv2d (kernel_size=(3, 1))
    # Reshape input: [B, C, L] -> [B, L, 1, C] for conv2d (channels last)
    conv1_input = ttnn.permute(input_features, [0, 2, 1])  # [B, C, L] -> [B, L, C]
    conv1_input = ttnn.reshape(conv1_input, shape=[batch_size, seq_len, 1, num_mel_bins])

    # Reshape weights: [out_channels, in_channels, kernel_size] -> [out_channels, in_channels, kernel_size, 1]
    conv1_weight = ttnn.reshape(parameters["conv1"]["weight"], shape=[config.d_model, num_mel_bins, 3, 1])

    # Move weights and biases to device
    conv1_weight = ttnn.to_device(conv1_weight, device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    conv1_bias = ttnn.to_device(
        ttnn.reshape(parameters["conv1"]["bias"], shape=[1, 1, 1, config.d_model]),
        device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    conv1_output = ttnn.conv2d(
        input_tensor=conv1_input,
        weight_tensor=conv1_weight,
        bias_tensor=conv1_bias,
        device=device,
        in_channels=num_mel_bins,
        out_channels=config.d_model,
        batch_size=batch_size,
        input_height=seq_len,
        input_width=1,
        kernel_size=(3, 1),
        stride=(1, 1),
        padding=(1, 0),  # Only pad height dimension
        dtype=ttnn.bfloat16,
        conv_config=conv_config,
        compute_config=conv_compute_config,
    )
    conv1_output = ttnn.gelu(conv1_output)

    # Reshape back: [B, L, 1, d_model] -> [B, L, d_model] -> [B, d_model, L]
    conv1_output = ttnn.reshape(conv1_output, shape=[batch_size, seq_len, config.d_model])
    conv1_output = ttnn.permute(conv1_output, [0, 2, 1])  # [B, L, d_model] -> [B, d_model, L]

    # Conv2: Simulate 1D conv with conv2d, stride=2 (kernel_size=(3, 1))
    # Reshape input: [B, C, L] -> [B, L, 1, C] for conv2d (channels last)
    conv2_input = ttnn.permute(conv1_output, [0, 2, 1])  # [B, C, L] -> [B, L, C]
    conv2_input = ttnn.reshape(conv2_input, shape=[batch_size, seq_len, 1, config.d_model])

    # Conv2 weights are already reshaped and converted to TTNN in _convert_weights_to_ttnn
    conv2_weight = ttnn.to_device(parameters["conv2"]["weight"], device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    conv2_bias = ttnn.reshape(parameters["conv2"]["bias"], [1, 1, 1, config.d_model])
    conv2_bias = ttnn.to_device(conv2_bias, device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    conv2_output = ttnn.conv2d(
        input_tensor=conv2_input,
        weight_tensor=conv2_weight,
        bias_tensor=conv2_bias,
        device=device,
        in_channels=config.d_model,
        out_channels=config.d_model,
        batch_size=batch_size,
        input_height=seq_len,
        input_width=1,
        kernel_size=(3, 1),
        stride=(2, 1),  # Stride only in height dimension (sequence)
        padding=(1, 0),  # Only pad height dimension
        dtype=ttnn.bfloat16,
        conv_config=conv_config,
        compute_config=conv_compute_config,
    )

    # Reshape back: [B, seq_len//2, 1, d_model] -> [B, seq_len//2, d_model]
    conv2_output = ttnn.reshape(conv2_output, shape=[batch_size, seq_len // 2, config.d_model])

    return conv2_output


def encoder_minicpm(config, inputs_embeds, *, parameters):
    """
    Adapted encoder function for MiniCPM dict-based parameters.

    Args:
        config: WhisperConfig
        inputs_embeds: Input embeddings
        parameters: Dict-based parameters

    Returns:
        Encoder output
    """
    # Add positional embeddings (slice to match sequence length)
    seq_len = inputs_embeds.shape[1]
    positional_embeds = ttnn.slice(parameters["embed_positions"]["weight"], [0, 0], [seq_len, config.d_model], [1, 1])
    hidden_states = ttnn.add(inputs_embeds, positional_embeds)

    # Apply dropout (no-op for inference)
    # hidden_states = dropout(hidden_states, p=0, training=False)

    # Run encoder layers
    for layer_idx in range(config.encoder_layers):
        layer_params = parameters["layers"][layer_idx]
        hidden_states = encoder_layer_minicpm(config, hidden_states, parameters=layer_params)

    # Final layer norm
    hidden_states = ttnn.layer_norm(
        hidden_states,
        weight=parameters["layer_norm"]["weight"],
        bias=parameters["layer_norm"]["bias"],
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4),
    )

    return hidden_states


def encoder_layer_minicpm(config, hidden_states, *, parameters):
    """
    Adapted encoder layer for MiniCPM dict-based parameters.

    Args:
        config: WhisperConfig
        hidden_states: Input hidden states
        parameters: Dict-based layer parameters

    Returns:
        Layer output
    """
    residual = hidden_states

    # Self-attention
    hidden_states = ttnn.layer_norm(
        hidden_states,
        weight=parameters["self_attn_layer_norm"]["weight"],
        bias=parameters["self_attn_layer_norm"]["bias"],
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4),
    )

    hidden_states = whisper_attention_minicpm(
        config, hidden_states, attention_mask=None, is_decode=False, parameters=parameters["self_attn"]
    )

    # Residual connection
    hidden_states = ttnn.add(residual, hidden_states)

    # Feed-forward
    residual = hidden_states
    hidden_states = ttnn.layer_norm(
        hidden_states,
        weight=parameters["final_layer_norm"]["weight"],
        bias=parameters["final_layer_norm"]["bias"],
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4),
    )

    # MLP
    hidden_states = ttnn.linear(
        hidden_states, parameters["fc1"]["weight"], bias=parameters["fc1"]["bias"], memory_config=WHISPER_MEMORY_CONFIG
    )
    hidden_states = ttnn.gelu(hidden_states)
    hidden_states = ttnn.linear(
        hidden_states, parameters["fc2"]["weight"], bias=parameters["fc2"]["bias"], memory_config=WHISPER_MEMORY_CONFIG
    )

    # Residual connection
    hidden_states = ttnn.add(residual, hidden_states)

    return hidden_states


def whisper_attention_minicpm(config, hidden_states, attention_mask, is_decode, *, parameters):
    """
    Adapted Whisper attention for MiniCPM dict-based parameters.

    Args:
        config: WhisperConfig
        hidden_states: Input hidden states
        attention_mask: Attention mask (None for encoder)
        is_decode: Whether this is decoder attention
        parameters: Dict-based attention parameters

    Returns:
        Attention output
    """
    bsz, tgt_len, hidden_size = hidden_states.shape

    # Linear transformations for Q, K, V
    query_states = ttnn.linear(
        hidden_states,
        parameters["query"]["weight"],
        bias=parameters["query"]["bias"],
        memory_config=WHISPER_MEMORY_CONFIG,
    )

    # Apply attention scaling (1/sqrt(head_dim)) to query states
    head_size = hidden_size // config.encoder_attention_heads
    scale_factor = head_size**-0.5
    query_states = ttnn.mul(query_states, scale_factor)
    key_states = ttnn.linear(
        hidden_states, parameters["key"]["weight"], bias=parameters["key"]["bias"], memory_config=WHISPER_MEMORY_CONFIG
    )
    value_states = ttnn.linear(
        hidden_states,
        parameters["value"]["weight"],
        bias=parameters["value"]["bias"],
        memory_config=WHISPER_MEMORY_CONFIG,
    )

    # Reshape for attention
    head_size = hidden_size // config.encoder_attention_heads
    query_states = ttnn.reshape(query_states, [bsz, tgt_len, config.encoder_attention_heads, head_size])
    query_states = ttnn.transpose(query_states, 1, 2)  # [bsz, num_heads, seq_len, head_size]

    key_states = ttnn.reshape(key_states, [bsz, tgt_len, config.encoder_attention_heads, head_size])
    key_states = ttnn.transpose(key_states, 1, 2)  # [bsz, num_heads, seq_len, head_size]

    value_states = ttnn.reshape(value_states, [bsz, tgt_len, config.encoder_attention_heads, head_size])
    value_states = ttnn.transpose(value_states, 1, 2)  # [bsz, num_heads, seq_len, head_size]

    # Compute attention: Q * K^T
    # query_states: [bsz, num_heads, seq_len, head_size]
    # key_states: [bsz, num_heads, seq_len, head_size]
    # We need: [bsz, num_heads, seq_len, seq_len]
    key_states_t = ttnn.transpose(key_states, 2, 3)  # [bsz, num_heads, head_size, seq_len]
    attn_weights = ttnn.matmul(query_states, key_states_t)  # [bsz, num_heads, seq_len, seq_len]

    # Apply softmax
    attn_weights = ttnn.softmax(attn_weights, dim=-1)

    # Apply attention to values: attention_weights * V
    # attn_weights: [bsz, num_heads, seq_len, seq_len]
    # value_states: [bsz, num_heads, seq_len, head_size]
    attn_output = ttnn.matmul(attn_weights, value_states)  # [bsz, num_heads, seq_len, head_size]

    # Reshape back to [bsz, seq_len, hidden_size]
    attn_output = ttnn.transpose(attn_output, 1, 2)  # [bsz, seq_len, num_heads, head_size]
    attn_output = ttnn.reshape(attn_output, [bsz, tgt_len, hidden_size])  # [bsz, seq_len, hidden_size]

    # Apply output projection
    attn_output = ttnn.linear(
        attn_output,
        parameters["out_proj"]["weight"],
        bias=parameters["out_proj"]["bias"],
        memory_config=WHISPER_MEMORY_CONFIG,
    )

    return attn_output
