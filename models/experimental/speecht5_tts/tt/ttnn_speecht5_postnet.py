# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

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
from ttnn.model_preprocessing import fold_batch_norm2d_into_conv2d

# ============================================================================
# High-Performance Compute Kernel Configs - Maximum Core Utilization
# ============================================================================


def get_high_perf_compute_config(device=None):
    """
    Get compute kernel config optimized for accuracy and numerical stability.
    Uses HiFi4 with float32 weights (like YOLOv5x) for best conv2d accuracy.
    Based on analysis of other models like Llama Vision.
    """
    if device is not None:
        return ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,  # Disable for maximum accuracy
            fp32_dest_acc_en=True,  # Keep FP32 dest acc for stability
            packer_l1_acc=False,  # Disable L1 accumulation when using FP32 dest acc
        )
    else:
        # Fallback for when device is not available
        return ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )


def get_ultra_high_precision_compute_config():
    """
    Ultra high precision config for debugging conv2d noise issues.
    Uses HiFi4 with additional numerical stability settings and experimental features.
    """
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,  # Keep HiFi4 for ultra-high precision debugging
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
        # Additional precision settings
        dst_full_sync_en=True,  # Full sync for destination
    )


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

        # Create conv config - Use float32 for maximum accuracy (not bfloat16)
        self.conv_config = ttnn.Conv2dConfig(
            weights_dtype=ttnn.float32,  # HIGHEST PRECISION: Use float32 instead of bfloat16
            output_layout=ttnn.TILE_LAYOUT,
            deallocate_activation=False,  # Keep activation for numerical stability
            reallocate_halo_output=False,  # Keep halo output for stability
            enable_act_double_buffer=False,  # Keep disabled for precision
            enable_weights_double_buffer=False,  # Keep disabled for precision
            config_tensors_in_dram=True,  # Put config tensors in DRAM for consistency
            reshard_if_not_optimal=False,  # Keep disabled for consistency
            enable_kernel_stride_folding=False,  # Disable stride folding for better precision
            force_split_reader=False,  # Don't force split reader
            # Add missing parameters from conv2d tests
            transpose_shards=False,
            enable_activation_reuse=False,
            full_inner_dim=False,
        )

        # DEBUG: Create ultra-high precision config for debugging conv2d noise
        self.debug_conv_config = ttnn.Conv2dConfig(
            weights_dtype=self.weight.dtype,
            output_layout=ttnn.TILE_LAYOUT,
            deallocate_activation=True,
            reallocate_halo_output=True,
            enable_act_double_buffer=True,
            enable_weights_double_buffer=True,
            config_tensors_in_dram=False,
            reshard_if_not_optimal=False,
        )

        # ULTRA-HIGH PRECISION: Conservative config with selective experimental features
        self.ultra_precision_conv_config = ttnn.Conv2dConfig(
            weights_dtype=ttnn.float32,  # Back to float32 weights
            output_layout=ttnn.TILE_LAYOUT,
            deallocate_activation=False,  # Keep activation for numerical stability
            reallocate_halo_output=False,  # Keep halo output for stability
            enable_act_double_buffer=False,  # Keep disabled for precision
            enable_weights_double_buffer=False,  # Keep disabled for precision
            config_tensors_in_dram=True,  # Put config tensors in DRAM for consistency
            reshard_if_not_optimal=False,  # Keep disabled for consistency
            enable_kernel_stride_folding=True,  # EXPERIMENTAL: Enable tensor folding optimization
            enable_activation_reuse=False,  # DISABLED: Causes assertion failure
            force_split_reader=False,  # DISABLED: May cause issues
            full_inner_dim=False,  # DISABLED: Only for block sharding
        )

    def __call__(self, x, batch_size, input_length, debug_pytorch_ref=None):
        """
        Apply Conv1D using conv2d.

        Args:
            x: Input tensor [B, C, L]
            batch_size: Batch size
            input_length: Sequence length

        Returns:
            Output tensor [B, out_channels, L]
        """
        # PHASE 1: Ensure input is in L1 and reshape (L1 outputs)
        x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
        x = ttnn.permute(x, [0, 2, 1], memory_config=ttnn.L1_MEMORY_CONFIG)
        x = ttnn.reshape(x, [batch_size, 1, input_length, self.in_channels], memory_config=ttnn.L1_MEMORY_CONFIG)

        # DEBUG: Compare input PCC if PyTorch reference provided
        if debug_pytorch_ref is not None and len(debug_pytorch_ref) > 0:
            x_torch = ttnn.to_torch(x).squeeze(1).permute(0, 2, 1)  # Convert back to [B, C, L]
            pytorch_ref = debug_pytorch_ref.pop(0)  # Get and remove first reference

            if x_torch.shape == pytorch_ref.shape:
                from scipy.stats import pearsonr

                try:
                    pcc = pearsonr(x_torch.flatten(), pytorch_ref.flatten())[0]
                    print(f"DEBUG: Conv input PCC: {pcc:.4f}, shape: {x_torch.shape}")
                except:
                    print(f"DEBUG: Conv input shapes - TTNN: {x_torch.shape}, PyTorch: {pytorch_ref.shape}")
            else:
                print(f"DEBUG: Conv input shape mismatch - TTNN: {x_torch.shape}, PyTorch: {pytorch_ref.shape}")

        # PHASE 2: Apply conv2d with return_weights_and_bias=True to get prepared weights
        # This prevents re-preparation during trace
        # Reshape bias to [1, 1, 1, out_channels] for conv2d
        bias_reshaped = ttnn.reshape(self.bias, [1, 1, 1, self.out_channels])

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
            bias_tensor=bias_reshaped,
            conv_config=self.conv_config,  # Use ultra-high precision config with experimental features
            return_weights_and_bias=True,
            return_output_dim=True,
        )

        # PHASE 3: Reshape back (L1 outputs)
        result = ttnn.reshape(
            result, [batch_size, input_length, self.out_channels], memory_config=ttnn.L1_MEMORY_CONFIG
        )
        result = ttnn.permute(result, [0, 2, 1], memory_config=ttnn.L1_MEMORY_CONFIG)

        # PHASE 4: Final output must be in L1
        return ttnn.to_memory_config(result, ttnn.L1_MEMORY_CONFIG)


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
        Initialize Conv1D layer with BatchNorm folded in.

        Args:
            device: TTNN device
            parameters: Dictionary with 'conv' sub-dict (BatchNorm already folded)
            config: Post-net configuration
            has_activation: Whether to apply Tanh activation
        """
        self.device = device
        self.config = config
        self.has_activation = has_activation

        # Create conv instance (BatchNorm effects already folded into conv weights)
        self.conv = TtConv1d(device, parameters)

        # Get dimensions from conv
        self.out_channels = self.conv.out_channels
        self.in_channels = self.conv.in_channels

    def __call__(self, hidden_states: ttnn.Tensor) -> ttnn.Tensor:
        """
        Forward pass with comprehensive L1 memory management.

        Args:
            hidden_states: [batch, channels, time_steps] in TTNN format

        Returns:
            output: [batch, channels, time_steps]
        """
        # PHASE 1: Ensure input is in L1
        hidden_states = ttnn.to_memory_config(hidden_states, ttnn.L1_MEMORY_CONFIG)

        # Get batch and sequence length from input
        # hidden_states shape: [batch, in_channels, seq_len]
        batch_size = hidden_states.shape[0]
        input_length = hidden_states.shape[2]

        # PHASE 2: Op 1: Conv1d (BatchNorm effects already folded in) (L1 output)
        conv_result = self.conv(hidden_states, batch_size, input_length)
        conv_result = ttnn.to_memory_config(conv_result, ttnn.L1_MEMORY_CONFIG)

        # PHASE 3: Op 2: Tanh activation (if present) (L1 output)
        # NOTE: TTNN internally performs WH transposes around tanh despite it being element-wise
        # This is an inefficiency in TTNN - tanh doesn't need layout changes
        if self.has_activation:
            conv_result = ttnn.tanh(conv_result)
            conv_result = ttnn.to_memory_config(conv_result, ttnn.L1_MEMORY_CONFIG)

        # PHASE 4: Op 3: Dropout (only in training mode, skip in inference)
        # In inference, dropout is a no-op
        hidden_states = conv_result
        hidden_states = ttnn.to_memory_config(hidden_states, ttnn.L1_MEMORY_CONFIG)

        # PHASE 6: Final output must be in L1
        return ttnn.to_memory_config(hidden_states, ttnn.L1_MEMORY_CONFIG)


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

    def postnet(self, hidden_states: ttnn.Tensor, debug_pytorch_ref=None) -> ttnn.Tensor:
        """
        Apply convolutional post-net with residual connection and comprehensive L1 memory management.

        Args:
            hidden_states: [batch, time_steps, mel_bins]

        Returns:
            refined: [batch, time_steps, mel_bins]
        """
        # PHASE 1: Ensure input is in L1
        hidden_states = ttnn.to_memory_config(hidden_states, ttnn.L1_MEMORY_CONFIG)

        # Save input for residual connection
        residual = hidden_states
        residual = ttnn.to_memory_config(residual, ttnn.L1_MEMORY_CONFIG)

        # PHASE 2: Op 1: Transpose for Conv1d ([B, L, C] → [B, C, L]) (L1 output)
        layer_output = ttnn.permute(hidden_states, [0, 2, 1], memory_config=ttnn.L1_MEMORY_CONFIG)

        # PHASE 3: Op 2-6: Apply 5 conv layers (L1 outputs)
        for layer in self.layers:
            layer_output = layer(layer_output)
            layer_output = ttnn.to_memory_config(layer_output, ttnn.L1_MEMORY_CONFIG)

        # PHASE 4: Op 7: Transpose back ([B, C, L] → [B, L, C]) (L1 output)
        layer_output = ttnn.permute(layer_output, [0, 2, 1], memory_config=ttnn.L1_MEMORY_CONFIG)

        # PHASE 5: Op 8: Residual connection (L1 output)
        output = ttnn.add(residual, layer_output, memory_config=ttnn.L1_MEMORY_CONFIG)
        output = ttnn.to_memory_config(output, ttnn.L1_MEMORY_CONFIG)

        # PHASE 6: Final output must be in L1
        return ttnn.to_memory_config(output, ttnn.L1_MEMORY_CONFIG)

    def __call__(
        self, hidden_states: ttnn.Tensor, timing_details: bool = False, debug_pytorch_ref=None
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        """
        Forward pass with comprehensive L1 memory management.

        Args:
            hidden_states: [batch, decoder_seq_len, hidden_size]
            timing_details: If True, return (outputs..., timing_dict)

        Returns:
            outputs_before_postnet: [batch, mel_seq_len, num_mel_bins]
            outputs_after_postnet: [batch, mel_seq_len, num_mel_bins]
            stop_logits: [batch, mel_seq_len] or (outputs..., timing_dict)
        """
        import time

        timing = {}

        batch_size = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]
        hidden_size = hidden_states.shape[2]

        # PHASE 1: Ensure input is in L1
        start_time = time.time()
        hidden_states = ttnn.to_memory_config(hidden_states, ttnn.L1_MEMORY_CONFIG)
        timing["memory_input"] = time.time() - start_time

        # PHASE 2: Op 1: Project to mel features (high-performance compute kernel)
        # [batch, seq_len, hidden_size] → [batch, seq_len, mel_bins * reduction_factor]
        start_time = time.time()
        feat_out = ttnn.linear(
            hidden_states,
            self.parameters["feat_out"]["weight"],
            bias=self.parameters["feat_out"]["bias"],
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=get_high_perf_compute_config(self.device),
        )
        timing["mel_projection"] = time.time() - start_time

        # PHASE 3: Op 2: Reshape to unfold reduction factor (L1 output)
        # [batch, seq_len, mel_bins * reduction_factor] → [batch, seq_len * reduction_factor, mel_bins]
        start_time = time.time()
        mel_seq_len = seq_len * self.config.reduction_factor
        outputs_before_postnet = ttnn.reshape(
            feat_out, [batch_size, mel_seq_len, self.config.num_mel_bins], memory_config=ttnn.L1_MEMORY_CONFIG
        )
        timing["mel_reshape"] = time.time() - start_time

        # PHASE 4: Op 3: Apply convolutional post-net (with residual) (L1 output)
        start_time = time.time()
        outputs_after_postnet = self.postnet(outputs_before_postnet, debug_pytorch_ref=debug_pytorch_ref)
        outputs_after_postnet = ttnn.to_memory_config(outputs_after_postnet, ttnn.L1_MEMORY_CONFIG)
        timing["conv_postnet"] = time.time() - start_time

        # PHASE 5: Op 4: Predict stop tokens (high-performance compute kernel)
        # [batch, seq_len, hidden_size] → [batch, seq_len, reduction_factor]
        start_time = time.time()
        prob_out = ttnn.linear(
            hidden_states,
            self.parameters["prob_out"]["weight"],
            bias=self.parameters["prob_out"]["bias"],
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=get_high_perf_compute_config(self.device),
        )
        timing["stop_projection"] = time.time() - start_time

        # PHASE 6: Op 5: Reshape stop tokens (L1 output)
        # [batch, seq_len, reduction_factor] → [batch, seq_len * reduction_factor]
        start_time = time.time()
        stop_logits = ttnn.reshape(prob_out, [batch_size, mel_seq_len], memory_config=ttnn.L1_MEMORY_CONFIG)
        timing["stop_reshape"] = time.time() - start_time

        # PHASE 7: All outputs must be in L1
        start_time = time.time()
        outputs_before_postnet = ttnn.to_memory_config(outputs_before_postnet, ttnn.L1_MEMORY_CONFIG)
        outputs_after_postnet = ttnn.to_memory_config(outputs_after_postnet, ttnn.L1_MEMORY_CONFIG)
        stop_logits = ttnn.to_memory_config(stop_logits, ttnn.L1_MEMORY_CONFIG)
        timing["memory_output"] = time.time() - start_time

        if timing_details:
            return (outputs_before_postnet, outputs_after_postnet, stop_logits), timing
        return outputs_before_postnet, outputs_after_postnet, stop_logits

    def prepare_postnet_inputs(self, hidden_states: ttnn.Tensor):
        """
        Prepare inputs for trace execution by ensuring proper memory config.

        This method separates input preparation from the forward pass to support trace capture.

        Args:
            hidden_states: [batch, seq_len, hidden_size] - decoder output

        Returns:
            List of prepared input tensors
        """
        # Ensure input is in L1 memory config for trace compatibility
        prepared_hidden_states = ttnn.to_memory_config(hidden_states, ttnn.L1_MEMORY_CONFIG)
        return [prepared_hidden_states]


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
            dtype=ttnn.float32,  # HIGHEST PRECISION: Use float32
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=DRAM_MEMCFG,  # Weights in DRAM
        ),
        "bias": ttnn.from_torch(
            feat_out_bias,
            dtype=ttnn.float32,  # HIGHEST PRECISION: Use float32
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
            dtype=ttnn.float32,  # HIGHEST PRECISION: Use float32
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=DRAM_MEMCFG,  # Weights in DRAM
        ),
        "bias": ttnn.from_torch(
            prob_out_bias,
            dtype=ttnn.float32,  # HIGHEST PRECISION: Use float32
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
        conv_weight = torch_layer.conv.weight.data.unsqueeze(-1)  # Add 4th dimension

        # Extract BatchNorm parameters and fold into Conv
        # Create mock conv/bn objects compatible with fold_batch_norm2d_into_conv2d
        class MockConv:
            def __init__(self, weight, bias):
                self.weight = weight
                self.bias = bias

        class MockBN:
            def __init__(self, weight, bias, running_mean, running_var, eps):
                self.weight = weight
                self.bias = bias
                self.running_mean = running_mean
                self.running_var = running_var
                self.eps = eps
                self.track_running_stats = True

        mock_conv = MockConv(conv_weight, None)  # Conv has no bias
        mock_bn = MockBN(
            torch_layer.batch_norm.weight.data,
            torch_layer.batch_norm.bias.data,
            torch_layer.batch_norm.running_mean.data,
            torch_layer.batch_norm.running_var.data,
            torch_layer.batch_norm.eps,
        )

        # Fold BatchNorm into Conv weights and bias
        folded_weight, folded_bias = fold_batch_norm2d_into_conv2d(mock_conv, mock_bn)

        # Store only folded conv parameters
        layer_params["conv"] = {
            "weight": ttnn.from_torch(
                folded_weight,
                dtype=ttnn.float32,  # HIGHEST PRECISION: Use float32 for maximum accuracy
                layout=ttnn.ROW_MAJOR_LAYOUT,  # TTNN conv2d requires ROW_MAJOR for weights
                device=device,
                memory_config=DRAM_MEMCFG,  # Weights in DRAM
            ),
            "bias": ttnn.from_torch(
                folded_bias,  # Use folded bias (not reshaped)
                dtype=ttnn.float32,  # HIGHEST PRECISION: Use float32 for maximum accuracy
                layout=ttnn.ROW_MAJOR_LAYOUT,  # BIAS MUST BE ROW_MAJOR for conv2d
                device=device,
                memory_config=DRAM_MEMCFG,  # Weights in DRAM
            ),
        }

        parameters["layers"].append(layer_params)

    return parameters


if __name__ == "__main__":
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
