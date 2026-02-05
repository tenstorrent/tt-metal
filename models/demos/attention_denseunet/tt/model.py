# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
TTNN implementation of Attention DenseUNet model.

This module provides the TTNN implementation of all model components:
- DenseLayer, DenseBlock
- TransitionDown, TransitionUp
- AttentionGate
- DecoderBlock
- Full Attention DenseUNet model
"""

import torch
import ttnn
from models.demos.attention_denseunet.tt.config import (
    TtAttentionDenseUNetConfigs,
    DenseLayerConfiguration,
    AttentionGateConfiguration,
    UpconvConfiguration,
)
from models.tt_cnn.tt.builder import TtConv2d, TtMaxPool2d


def concatenate_features(
    x1: ttnn.Tensor, x2: ttnn.Tensor, use_row_major_layout=True
) -> ttnn.Tensor:
    """
    Concatenate two tensors along the channel dimension.
    
    Args:
        x1: First tensor
        x2: Second tensor
        use_row_major_layout: Whether to use row major layout for concatenation
        
    Returns:
        Concatenated tensor
    """
    assert x1.shape[:-1] == x2.shape[:-1], \
        f"Spatial dimensions must match for concatenation (got {x1.shape} and {x2.shape})"
    
    if not x2.is_sharded() and x1.is_sharded():
        input_core_grid = x1.memory_config().shard_spec.grid
        input_shard_shape = x1.memory_config().shard_spec.shape
        input_shard_spec = ttnn.ShardSpec(input_core_grid, input_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
        input_memory_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, input_shard_spec
        )
        x2 = ttnn.to_memory_config(x2, input_memory_config)
    
    if x1.is_sharded():
        output_core_grid = x1.memory_config().shard_spec.grid
        output_shard_shape = (
            x1.memory_config().shard_spec.shape[0],
            x1.memory_config().shard_spec.shape[1] + x2.memory_config().shard_spec.shape[1],
        )
        output_shard_spec = ttnn.ShardSpec(output_core_grid, output_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
        output_memory_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, output_shard_spec
        )
    else:
        output_memory_config = ttnn.DRAM_MEMORY_CONFIG
    
    if use_row_major_layout:
        x1_rm = ttnn.to_layout(x1, ttnn.ROW_MAJOR_LAYOUT)
        x2_rm = ttnn.to_layout(x2, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(x1)
        ttnn.deallocate(x2)
        
        concatenated = ttnn.concat([x1_rm, x2_rm], dim=3, memory_config=output_memory_config)
        ttnn.deallocate(x1_rm)
        ttnn.deallocate(x2_rm)
        
        concat_tiled = ttnn.to_layout(concatenated, ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)
        ttnn.deallocate(concatenated)
        return concat_tiled
    else:
        concatenated = ttnn.concat([x1, x2], dim=3, memory_config=output_memory_config)
        ttnn.deallocate(x1)
        ttnn.deallocate(x2)
        return concatenated


def transpose_conv2d(
    input_tensor: ttnn.Tensor,
    upconv_config: UpconvConfiguration,
    act_block_h_override=32,
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
) -> ttnn.Tensor:
    """
    Transposed convolution for upsampling.
    
    Args:
        input_tensor: Input tensor
        upconv_config: Upconv configuration
        act_block_h_override: Activation block height override
        fp32_dest_acc_en: Enable FP32 destination accumulation
        packer_l1_acc: Enable packer L1 accumulation
        
    Returns:
        Upsampled tensor
    """
    conv_config = ttnn.Conv2dConfig(
        weights_dtype=ttnn.bfloat8_b,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED if input_tensor.is_sharded() else None,
        deallocate_activation=True,
        enable_act_double_buffer=False,
        output_layout=ttnn.TILE_LAYOUT,
        act_block_h_override=act_block_h_override,
    )
    compute_config = ttnn.init_device_compute_kernel_config(
        input_tensor.device().arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        fp32_dest_acc_en=fp32_dest_acc_en,
        packer_l1_acc=packer_l1_acc,
    )
    
    output, [upconv_config.weight, upconv_config.bias] = ttnn.conv_transpose2d(
        input_tensor=input_tensor,
        weight_tensor=upconv_config.weight,
        bias_tensor=upconv_config.bias,
        in_channels=upconv_config.in_channels,
        out_channels=upconv_config.out_channels,
        kernel_size=upconv_config.kernel_size,
        stride=upconv_config.stride,
        padding=upconv_config.padding,
        batch_size=upconv_config.batch_size,
        input_height=upconv_config.input_height,
        input_width=upconv_config.input_width,
        conv_config=conv_config,
        compute_config=compute_config,
        device=input_tensor.device(),
        return_output_dim=False,
        return_weights_and_bias=True,
    )
    
    return output


class TtDenseLayer:
    """
    TTNN implementation of a DenseLayer.
    
    Follows pattern: BN-ReLU-Conv1x1-BN-ReLU-Conv3x3-Concat
    (BatchNorm is folded into Conv during preprocessing)
    """
    
    def __init__(self, config: DenseLayerConfiguration, device: ttnn.Device):
        self.device = device
        self.bottleneck = TtConv2d(config.bottleneck_conv, device)
        self.expansion = TtConv2d(config.expansion_conv, device)
    
    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """
        Forward pass: bottleneck conv -> expansion conv -> concatenate with input.
        
        Args:
            x: Input tensor with shape [B, H, W, C_in]
            
        Returns:
            Concatenated tensor with shape [B, H, W, C_in + growth_rate]
        """
        identity = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        out = self.bottleneck(x)
        out = self.expansion(out)
        result = concatenate_features(identity, out, use_row_major_layout=True)
        return result


class TtDenseBlock:
    """
    TTNN implementation of a DenseBlock.
    
    Contains multiple DenseLayers where each layer concatenates its output
    to the input, creating dense connections.
    """
    
    def __init__(self, layers: list, device: ttnn.Device):
        self.device = device
        self.layers = [TtDenseLayer(layer_config, device) for layer_config in layers]
    
    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """
        Forward pass through all dense layers.
        
        Args:
            x: Input tensor
            
        Returns:
            Output with concatenated features from all layers
        """
        for layer in self.layers:
            x = layer(x)
        return x


class TtTransitionDown:
    """
    TTNN implementation of TransitionDown layer.
    
    Compresses channels and reduces spatial dimensions via pooling.
    """
    
    def __init__(self, conv_config, pool_config, device: ttnn.Device):
        self.device = device
        self.conv = TtConv2d(conv_config, device)
        self.pool = TtMaxPool2d(pool_config, device)
    
    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """
        Forward pass: conv (BN+ReLU fused) -> max pool.
        
        Args:
            x: Input tensor
            
        Returns:
            Downsampled tensor with compressed channels
        """
        x = self.conv(x)
        x = self.pool(x)
        return x


class TtTransitionUp:
    """
    TTNN implementation of TransitionUp layer.
    
    Upsamples spatial dimensions via transposed convolution.
    """
    
    def __init__(self, upconv_config: UpconvConfiguration):
        self.upconv_config = upconv_config
    
    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """
        Forward pass: transposed convolution for upsampling.
        
        Args:
            x: Input tensor
            
        Returns:
            Upsampled tensor
        """
        return transpose_conv2d(x, self.upconv_config)


class TtAttentionGate:
    """
    TTNN implementation of AttentionGate.
    
    Computes spatial attention to emphasize relevant features from
    skip connections based on the gating signal from the decoder.
    """
    
    def __init__(self, config: AttentionGateConfiguration, device: ttnn.Device):
        self.device = device
        self.config = config
        
        # Initialize convolution layers
        self.theta = TtConv2d(config.theta_conv, device)
        self.phi = TtConv2d(config.phi_conv, device)
        self.psi = TtConv2d(config.psi_conv, device)
        self.W = TtConv2d(config.W_conv, device)
    
    def __call__(self, x: ttnn.Tensor, g: ttnn.Tensor) -> ttnn.Tensor:
        """
        Forward pass for attention gate.
        
        Args:
            x: Skip connection features (higher resolution)
            g: Gating signal from decoder (may be lower resolution)
            
        Returns:
            Attention-weighted skip connection features
        """
        theta_x = self.theta(x)
        phi_g = self.phi(g)
        if phi_g.shape[1] != theta_x.shape[1] or phi_g.shape[2] != theta_x.shape[2]:
            phi_g_torch = ttnn.to_torch(phi_g)
            phi_g_torch = torch.nn.functional.interpolate(
                phi_g_torch.permute(0, 3, 1, 2),  # NHWC -> NCHW
                size=(theta_x.shape[1], theta_x.shape[2]),
                mode='bilinear',
                align_corners=False
            ).permute(0, 2, 3, 1)  # NCHW -> NHWC
            phi_g = ttnn.from_torch(
                phi_g_torch,
                dtype=ttnn.bfloat16,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
        f = ttnn.add(theta_x, phi_g)
        f = ttnn.relu(f)
        attention = self.psi(f)
        attention = ttnn.sigmoid(attention)
        y = ttnn.mul(attention, x)
        output = self.W(y)
        return output


class TtDecoderBlock:
    """
    TTNN implementation of DecoderBlock.
    
    Two sequential convolution blocks.
    """
    
    def __init__(self, conv1_config, conv2_config, device: ttnn.Device):
        self.device = device
        self.conv1 = TtConv2d(conv1_config, device)
        self.conv2 = TtConv2d(conv2_config, device)
    
    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """
        Forward pass: conv1 (BN+ReLU) -> conv2 (BN+ReLU).
        
        Args:
            x: Input tensor (concatenated upsampled + attended skip)
            
        Returns:
            Processed tensor
        """
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class TtAttentionDenseUNet:
    """
    TTNN implementation of complete Attention DenseUNet model.
    
    Combines DenseNet encoder with attention-gated skip connections
    in a U-Net style architecture.
    """
    
    def __init__(self, configs: TtAttentionDenseUNetConfigs, device: ttnn.Device):
        self.device = device
        self.configs = configs
        self.conv0 = TtConv2d(configs.conv0, device)
        self.encoder_blocks = []
        for block_configs in configs.encoder_blocks:
            self.encoder_blocks.append(TtDenseBlock(block_configs, device))
        
        self.transitions_down = []
        for i, (trans_conv, pool_config) in enumerate(zip(configs.transitions_down, configs.pools)):
            self.transitions_down.append(TtTransitionDown(trans_conv, pool_config, device))
        self.bottleneck_conv1 = TtConv2d(configs.bottleneck_conv1, device)
        self.bottleneck_conv2 = TtConv2d(configs.bottleneck_conv2, device)
        self.transitions_up = []
        for upconv_config in configs.upconvs:
            self.transitions_up.append(TtTransitionUp(upconv_config))
        
        self.attention_gates = []
        for att_config in configs.attention_gates:
            self.attention_gates.append(TtAttentionGate(att_config, device))
        
        self.decoder_blocks = []
        for dec_config in configs.decoder_blocks:
            self.decoder_blocks.append(
                TtDecoderBlock(dec_config.conv1, dec_config.conv2, device)
            )
        self.conv_out = TtConv2d(configs.conv_out, device)
    
    def preprocess_input_tensor(self, x: ttnn.Tensor, deallocate_input_activation: bool = True):
        """
        Preprocess input tensor to HWC layout.
        
        Args:
            x: Input tensor
            deallocate_input_activation: Whether to deallocate input
            
        Returns:
            Preprocessed tensor in HWC format
        """
        output = ttnn.experimental.convert_to_hwc(x)
        if deallocate_input_activation:
            ttnn.deallocate(x)
        return output
    
    def __call__(self, input_tensor: ttnn.Tensor, deallocate_input_activation: bool = True) -> ttnn.Tensor:
        """
        Forward pass through the complete Attention DenseUNet model.
        
        Args:
            input_tensor: Input image tensor
            deallocate_input_activation: Whether to deallocate intermediate activations
            
        Returns:
            Segmentation output tensor
        """
        x = self.preprocess_input_tensor(input_tensor, deallocate_input_activation)
        x = self.conv0(x)
        skips = []
        for encoder_block, transition_down in zip(self.encoder_blocks, self.transitions_down):
            x = encoder_block(x)
            skips.append(ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG))
            x = transition_down(x)
        x = self.bottleneck_conv1(x)
        x = self.bottleneck_conv2(x)
        reversed_skips = list(reversed(skips))
        for i, (trans_up, att_gate, decoder) in enumerate(
            zip(self.transitions_up, self.attention_gates, self.decoder_blocks)
        ):
            x = trans_up(x)
            skip = reversed_skips[i]
            attended_skip = att_gate(skip, x)
            x = concatenate_features(x, attended_skip, use_row_major_layout=True)
            x = decoder(x)
        output = self.conv_out(x)
        output = ttnn.experimental.convert_to_chw(output, dtype=ttnn.bfloat16)
        
        return output


def create_model_from_configs(
    configs: TtAttentionDenseUNetConfigs, device: ttnn.Device
) -> TtAttentionDenseUNet:
    """
    Construct Attention DenseUNet instance from configuration.
    
    Args:
        configs: Model configuration object
        device: TTNN device
        
    Returns:
        TtAttentionDenseUNet instance
    """
    return TtAttentionDenseUNet(configs, device)
