"""
SpeechT5 Speech Decoder Post-Net - TTNN Implementation

Converts decoder hidden states to mel-spectrograms with refinement.
Translated from reference/speecht5_postnet.py

Target: PCC > 0.94 vs PyTorch reference
"""

import ttnn
import torch
from typing import Tuple
from dataclasses import dataclass


@dataclass
class TTNNPostNetConfig:
    """Configuration for TTNN Speech Decoder Post-Net."""

    hidden_size: int = 768
    num_mel_bins: int = 80
    reduction_factor: int = 2
    postnet_layers: int = 5
    postnet_units: int = 256
    postnet_kernel: int = 5
    postnet_dropout: float = 0.5


class TtConv1d:
    """
    Conv1D wrapper class similar to YOLOv8's TtConv.

    Encapsulates conv2d call with weights, bias, and config stored in __init__.
    """

    def __init__(self, device, parameters):
        """
        Initialize Conv1D.

        Args:
            device: TTNN device
            parameters: Dictionary with 'conv' weights and bias
        """
        self.device = device
        self.weight = parameters["conv"]["weight"]
        self.bias = parameters["conv"].get("bias", None)

        # Extract dimensions from weight shape [out_channels, in_channels, kernel_size, 1]
        self.out_channels = self.weight.shape[0]
        self.in_channels = self.weight.shape[1]
        self.kernel_size = self.weight.shape[2]
        self.padding = (self.kernel_size - 1) // 2

        # Create conv config once
        self.conv_config = ttnn.Conv2dConfig(
            weights_dtype=self.weight.dtype,
        )

    def __call__(self, x, batch_size, input_length):
        """
        Apply Conv1D using conv2d.

        Args:
            x: Input tensor [B, C, L]
            batch_size: Batch size
            input_length: Sequence length

        Returns:
            Output tensor [B, out_channels, L]
        """
        # Reshape: [B, C, L] -> [B, L, C] -> [B, L, 1, C]
        x = ttnn.permute(x, [0, 2, 1])
        x = ttnn.reshape(x, [batch_size, input_length, 1, self.in_channels])

        # Apply conv2d with return_weights_and_bias=True to get prepared weights
        # This prevents re-preparation during trace
        result, _, [self.weight, self.bias] = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=self.weight,
            device=self.device,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            batch_size=batch_size,
            input_height=input_length,
            input_width=1,
            kernel_size=(self.kernel_size, 1),
            stride=(1, 1),
            padding=(self.padding, 0),
            bias_tensor=self.bias,
            conv_config=self.conv_config,
            return_weights_and_bias=True,
            return_output_dim=True,
        )

        # Reshape back: [B, L, 1, C] -> [B, L, C] -> [B, C, L]
        result = ttnn.reshape(result, [batch_size, input_length, self.out_channels])
        result = ttnn.permute(result, [0, 2, 1])

        return result


class TTNNSpeechT5BatchNormConvLayer:
    """
    Single Conv1D layer with BatchNorm, optional Tanh, and Dropout.

    TTNN implementation matching PyTorch reference.
    """

    def __init__(
        self,
        device,
        parameters,
        config: TTNNPostNetConfig,
        has_activation: bool,
    ):
        """
        Initialize Conv1D + BatchNorm layer.

        Args:
            device: TTNN device
            parameters: Dictionary with 'conv', 'batch_norm' sub-dicts
            config: Post-net configuration
            has_activation: Whether to apply Tanh activation
        """
        self.device = device
        self.config = config
        self.has_activation = has_activation

        # Create conv instance (similar to YOLOv8 pattern)
        self.conv = TtConv1d(device, parameters)

        # Store BatchNorm weights as class members
        self.bn_weight = parameters["batch_norm"]["weight"]
        self.bn_bias = parameters["batch_norm"]["bias"]
        self.bn_running_mean = parameters["batch_norm"]["running_mean"]
        self.bn_running_var = parameters["batch_norm"]["running_var"]

        # Get dimensions from conv
        self.out_channels = self.conv.out_channels
        self.in_channels = self.conv.in_channels

    def __call__(self, hidden_states: ttnn.Tensor) -> ttnn.Tensor:
        """
        Forward pass.

        Args:
            hidden_states: [batch, channels, time_steps] in TTNN format

        Returns:
            output: [batch, channels, time_steps]
        """
        # Op 1: Conv1d
        # Get batch and sequence length from input
        # hidden_states shape: [batch, in_channels, seq_len]
        batch_size = hidden_states.shape[0]
        input_length = hidden_states.shape[2]

        # Apply Conv1d using self.conv instance (similar to YOLOv8 pattern)
        conv_result = self.conv(hidden_states, batch_size, input_length)

        # Op 2: BatchNorm
        # Reshape for batch_norm: [B, C, L] -> [B, C, L, 1] for TTNN
        conv_result = ttnn.reshape(conv_result, [batch_size, self.out_channels, input_length, 1])

        bn_result = ttnn.batch_norm(
            conv_result,
            running_mean=self.bn_running_mean,
            running_var=self.bn_running_var,
            weight=self.bn_weight,
            bias=self.bn_bias,
            training=False,  # Inference mode
            eps=1e-05,
        )

        # Reshape back: [B, C, L, 1] -> [B, C, L]
        bn_result = ttnn.reshape(bn_result, [batch_size, self.out_channels, input_length])

        # Op 3: Tanh activation (if present)
        if self.has_activation:
            bn_result = ttnn.tanh(bn_result)

        # Op 4: Dropout (only in training mode, skip in inference)
        # In inference, dropout is a no-op
        hidden_states = bn_result

        return hidden_states


class TTNNSpeechT5SpeechDecoderPostnet:
    """
    TTNN Speech Decoder Post-Net: Converts decoder hidden states to mel-spectrograms.

    Components:
    1. feat_out: Projects hidden states to mel features
    2. prob_out: Predicts stop tokens
    3. postnet: 5-layer convolutional network for mel refinement
    """

    def __init__(
        self,
        device,
        parameters,
        config: TTNNPostNetConfig,
    ):
        """
        Initialize TTNN Post-Net.

        Args:
            device: TTNN device
            parameters: Dictionary with model parameters
            config: Post-net configuration
        """
        self.device = device
        self.parameters = parameters
        self.config = config

        # Create convolutional layers
        self.layers = []
        for layer_id in range(config.postnet_layers):
            has_activation = layer_id < config.postnet_layers - 1
            layer = TTNNSpeechT5BatchNormConvLayer(
                device=device,
                parameters=parameters["layers"][layer_id],
                config=config,
                has_activation=has_activation,
            )
            self.layers.append(layer)

    def postnet(self, hidden_states: ttnn.Tensor) -> ttnn.Tensor:
        """
        Apply convolutional post-net with residual connection.

        Args:
            hidden_states: [batch, time_steps, mel_bins]

        Returns:
            refined: [batch, time_steps, mel_bins]
        """
        # Save input for residual connection
        residual = hidden_states

        # Op 1: Transpose for Conv1d ([B, L, C] → [B, C, L])
        layer_output = ttnn.permute(hidden_states, [0, 2, 1])

        # Op 2-6: Apply 5 conv layers
        for layer in self.layers:
            layer_output = layer(layer_output)

        # Op 7: Transpose back ([B, C, L] → [B, L, C])
        layer_output = ttnn.permute(layer_output, [0, 2, 1])

        # Op 8: Residual connection
        output = ttnn.add(residual, layer_output)

        return output

    def __call__(self, hidden_states: ttnn.Tensor) -> Tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        """
        Forward pass with explicit operations.

        Args:
            hidden_states: [batch, decoder_seq_len, hidden_size]

        Returns:
            outputs_before_postnet: [batch, mel_seq_len, num_mel_bins]
            outputs_after_postnet: [batch, mel_seq_len, num_mel_bins]
            stop_logits: [batch, mel_seq_len]
        """
        batch_size = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]
        hidden_size = hidden_states.shape[2]

        # Op 1: Project to mel features
        # [batch, seq_len, hidden_size] → [batch, seq_len, mel_bins * reduction_factor]
        feat_out = ttnn.linear(
            hidden_states,
            self.parameters["feat_out"]["weight"],
            bias=self.parameters["feat_out"]["bias"],
        )

        # Op 2: Reshape to unfold reduction factor
        # [batch, seq_len, mel_bins * reduction_factor] → [batch, seq_len * reduction_factor, mel_bins]
        mel_seq_len = seq_len * self.config.reduction_factor
        outputs_before_postnet = ttnn.reshape(feat_out, [batch_size, mel_seq_len, self.config.num_mel_bins])

        # Op 3: Apply convolutional post-net (with residual)
        outputs_after_postnet = self.postnet(outputs_before_postnet)

        # Op 4: Predict stop tokens
        # [batch, seq_len, hidden_size] → [batch, seq_len, reduction_factor]
        prob_out = ttnn.linear(
            hidden_states,
            self.parameters["prob_out"]["weight"],
            bias=self.parameters["prob_out"]["bias"],
        )

        # Op 5: Reshape stop tokens
        # [batch, seq_len, reduction_factor] → [batch, seq_len * reduction_factor]
        stop_logits = ttnn.reshape(prob_out, [batch_size, mel_seq_len])

        return outputs_before_postnet, outputs_after_postnet, stop_logits


def preprocess_postnet_parameters(torch_model, config: TTNNPostNetConfig, device):
    """
    Preprocess PyTorch post-net parameters for TTNN.

    Args:
        torch_model: PyTorch SpeechT5SpeechDecoderPostnet
        config: Post-net configuration
        device: TTNN device

    Returns:
        parameters: Dictionary of TTNN parameters

    Note:
        Weights are stored in DRAM for better memory management.
        Intermediate tensors will use L1 memory config during forward pass.
    """
    parameters = {}

    # Memory config: Weights go to DRAM, activations will use L1
    DRAM_MEMCFG = ttnn.DRAM_MEMORY_CONFIG
    L1_MEMCFG = ttnn.L1_MEMORY_CONFIG

    # Process feat_out (Linear layer)
    feat_out_weight = torch_model.feat_out.weight.data  # [out, in]
    feat_out_bias = torch_model.feat_out.bias.data  # [out]

    # TTNN linear expects weights to be transposed: [in, out]
    # Weights stored in DRAM
    parameters["feat_out"] = {
        "weight": ttnn.from_torch(
            feat_out_weight.T,  # Transpose: [out, in] -> [in, out]
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=DRAM_MEMCFG,  # Weights in DRAM
        ),
        "bias": ttnn.from_torch(
            feat_out_bias,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=DRAM_MEMCFG,  # Weights in DRAM
        ),
    }

    # Process prob_out (Linear layer)
    prob_out_weight = torch_model.prob_out.weight.data  # [out, in]
    prob_out_bias = torch_model.prob_out.bias.data  # [out]

    # TTNN linear expects weights to be transposed: [in, out]
    # Weights stored in DRAM
    parameters["prob_out"] = {
        "weight": ttnn.from_torch(
            prob_out_weight.T,  # Transpose: [out, in] -> [in, out]
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=DRAM_MEMCFG,  # Weights in DRAM
        ),
        "bias": ttnn.from_torch(
            prob_out_bias,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=DRAM_MEMCFG,  # Weights in DRAM
        ),
    }

    # Process convolutional layers
    parameters["layers"] = []
    for i, torch_layer in enumerate(torch_model.layers):
        layer_params = {}

        # Conv1d weights: [out_channels, in_channels, kernel_size]
        # Reshape to [out_channels, in_channels, kernel_size, 1] for conv2d
        # Conv weights must be in ROW_MAJOR layout and on device
        conv_weight = torch_layer.conv.weight.data.unsqueeze(-1)  # Add 4th dimension
        layer_params["conv"] = {
            "weight": ttnn.from_torch(
                conv_weight,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
                memory_config=DRAM_MEMCFG,  # Weights in DRAM
            ),
        }

        # BatchNorm parameters
        # Reshape to [1, C, 1, 1] for TTNN batch_norm
        bn_weight = torch_layer.batch_norm.weight.data.reshape(1, -1, 1, 1)
        bn_bias = torch_layer.batch_norm.bias.data.reshape(1, -1, 1, 1)
        bn_running_mean = torch_layer.batch_norm.running_mean.data.reshape(1, -1, 1, 1)
        bn_running_var = torch_layer.batch_norm.running_var.data.reshape(1, -1, 1, 1)

        layer_params["batch_norm"] = {
            "weight": ttnn.from_torch(
                bn_weight,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=DRAM_MEMCFG,  # Weights in DRAM
            ),
            "bias": ttnn.from_torch(
                bn_bias,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=DRAM_MEMCFG,  # Weights in DRAM
            ),
            "running_mean": ttnn.from_torch(
                bn_running_mean,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=DRAM_MEMCFG,  # Weights in DRAM
            ),
            "running_var": ttnn.from_torch(
                bn_running_var,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=DRAM_MEMCFG,  # Weights in DRAM
            ),
        }

        parameters["layers"].append(layer_params)

    return parameters


if __name__ == "__main__":
    import sys

    sys.path.append("/home/ttuser/ssinghal/PR-fix/speecht5_tts/tt-metal")

    from models.experimental.speecht5_tts.reference.speecht5_postnet import (
        load_from_huggingface,
    )

    print("=" * 80)
    print("TTNN SpeechT5 Post-Net Test")
    print("=" * 80)

    # Load PyTorch model
    print("\n1. Loading PyTorch post-net from HuggingFace...")
    torch_postnet = load_from_huggingface()
    torch_postnet.eval()

    # Create TTNN config
    torch_config = torch_postnet.config
    ttnn_config = TTNNPostNetConfig(
        hidden_size=torch_config.hidden_size,
        num_mel_bins=torch_config.num_mel_bins,
        reduction_factor=torch_config.reduction_factor,
        postnet_layers=torch_config.postnet_layers,
        postnet_units=torch_config.postnet_units,
        postnet_kernel=torch_config.postnet_kernel,
        postnet_dropout=torch_config.postnet_dropout,
    )

    print(f"   Config: {ttnn_config}")

    # Initialize TTNN device
    print("\n2. Initializing TTNN device...")
    device = ttnn.open_device(device_id=0, l1_small_size=24576)  # Larger L1 for conv operations

    # Preprocess parameters
    print("\n3. Converting parameters to TTNN...")
    parameters = preprocess_postnet_parameters(torch_postnet, ttnn_config, device)

    # Create TTNN model
    print("\n4. Creating TTNN post-net...")
    ttnn_postnet = TTNNSpeechT5SpeechDecoderPostnet(
        device=device,
        parameters=parameters,
        config=ttnn_config,
    )

    # Test forward pass
    print("\n5. Testing forward pass...")
    batch_size = 1
    seq_len = 10
    hidden_size = 768

    # Create test input
    torch_input = torch.randn(batch_size, seq_len, hidden_size)

    # PyTorch forward
    with torch.no_grad():
        torch_pre, torch_post, torch_stop = torch_postnet(torch_input)

    # TTNN forward
    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    ttnn_pre, ttnn_post, ttnn_stop = ttnn_postnet(ttnn_input)

    # Convert back to PyTorch for comparison
    ttnn_pre_torch = ttnn.to_torch(ttnn_pre)
    ttnn_post_torch = ttnn.to_torch(ttnn_post)
    ttnn_stop_torch = ttnn.to_torch(ttnn_stop)

    # Compute PCC
    def compute_pcc(torch_tensor, ttnn_tensor):
        torch_flat = torch_tensor.flatten()
        ttnn_flat = ttnn_tensor.flatten()
        return torch.corrcoef(torch.stack([torch_flat, ttnn_flat]))[0, 1].item()

    pcc_pre = compute_pcc(torch_pre, ttnn_pre_torch)
    pcc_post = compute_pcc(torch_post, ttnn_post_torch)
    pcc_stop = compute_pcc(torch_stop, ttnn_stop_torch)

    print(f"\n6. Results:")
    print(f"   Pre-postnet mel PCC:  {pcc_pre:.6f}")
    print(f"   Post-postnet mel PCC: {pcc_post:.6f}")
    print(f"   Stop logits PCC:      {pcc_stop:.6f}")

    # Check success
    success = all([pcc_pre > 0.94, pcc_post > 0.94, pcc_stop > 0.94])
    print(f"\n{'✓' if success else '✗'} Target PCC > 0.94: {'PASS' if success else 'FAIL'}")

    # Cleanup
    ttnn.close_device(device)

    print("\n" + "=" * 80)
