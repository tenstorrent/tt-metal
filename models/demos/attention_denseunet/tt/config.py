# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Configuration module for Attention DenseUNet TTNN implementation.

This module defines configuration dataclasses for all model components
and provides a builder to create configurations from preprocessed parameters.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import ttnn
from models.tt_cnn.tt.builder import (
    AutoShardedStrategyConfiguration,
    Conv2dConfiguration,
    HeightShardedStrategyConfiguration,
    L1FullSliceStrategyConfiguration,
    MaxPool2dConfiguration,
    ShardingStrategy,
)


@dataclass
class UpconvConfiguration:
    """Configuration for upsampling via transposed convolution."""
    
    input_height: int
    input_width: int
    in_channels: int
    out_channels: int
    batch_size: int
    kernel_size: Tuple[int, int] = (2, 2)
    stride: Tuple[int, int] = (2, 2)
    padding: Tuple[int, int] = (0, 0)
    weight: ttnn.Tensor = None
    bias: ttnn.Tensor = None


@dataclass
class DenseLayerConfiguration:
    """Configuration for a single dense layer (bottleneck + expansion convs)."""
    
    bottleneck_conv: Conv2dConfiguration  
    expansion_conv: Conv2dConfiguration   


@dataclass
class AttentionGateConfiguration:
    """Configuration for attention gate module."""
    
    theta_conv: Conv2dConfiguration  
    phi_conv: Conv2dConfiguration    
    psi_conv: Conv2dConfiguration    
    W_conv: Conv2dConfiguration      
    in_channels: int
    gating_channels: int
    inter_channels: int


@dataclass
class DecoderBlockConfiguration:
    """Configuration for decoder block (2 conv-bn-relu sequences)."""
    
    conv1: Conv2dConfiguration
    conv2: Conv2dConfiguration


@dataclass
class TtAttentionDenseUNetConfigs:
    """
    Complete configuration for Attention DenseUNet model.
    
    Contains configurations for all layers in the network.
    """
    l1_input_memory_config: ttnn.MemoryConfig
    conv0: Conv2dConfiguration
    encoder_blocks: List[List[DenseLayerConfiguration]]  # List of blocks, each containing dense layers
    transitions_down: List[Conv2dConfiguration]
    pools: List[MaxPool2dConfiguration]
    bottleneck_conv1: Conv2dConfiguration
    bottleneck_conv2: Conv2dConfiguration
    upconvs: List[UpconvConfiguration]
    attention_gates: List[AttentionGateConfiguration]
    decoder_blocks: List[DecoderBlockConfiguration]
    conv_out: Conv2dConfiguration
    batch_size: int
    in_channels: int
    out_channels: int
    input_height: int
    input_width: int


class TtAttentionDenseUNetConfigBuilder:
    """
    Builder class for creating TtAttentionDenseUNetConfigs from preprocessed parameters.
    
    This builder:
    1. Takes preprocessed PyTorch weights (TTNN tensors)
    2. Tracks spatial dimensions through the network
    3. Creates Conv2dConfiguration objects for all layers
    4. Handles dense block channel growth
    5. Configures attention gates
    """
    
    def __init__(
        self,
        parameters: Dict,
        in_channels: int = 3,
        out_channels: int = 1,
        input_height: int = 256,
        input_width: int = 256,
        batch_size: int = 1,
        init_features: int = 32,
        growth_rate: int = 16,
        num_layers_per_block: Tuple[int, ...] = (4, 4, 4, 4),
        compression: float = 0.5,
    ):
        self.parameters = parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_height = input_height
        self.input_width = input_width
        self.batch_size = batch_size
        self.init_features = init_features
        self.growth_rate = growth_rate
        self.num_layers_per_block = num_layers_per_block
        self.compression = compression
        self.num_encoder_blocks = len(num_layers_per_block)
        
    def build_configs(self) -> TtAttentionDenseUNetConfigs:
        """Build complete model configuration."""
        l1_input_memory_config = ttnn.L1_MEMORY_CONFIG
        current_height = self.input_height
        current_width = self.input_width
        conv0 = self._create_conv_config_from_params(
            input_height=current_height,
            input_width=current_width,
            in_channels=self.in_channels,
            out_channels=self.init_features,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            params_key="conv0",
        )
        encoder_blocks = []
        transitions_down = []
        pools = []
        current_channels = self.init_features
        skip_channels = []  # Track channels for skip connections
        
        for block_idx in range(self.num_encoder_blocks):
            num_layers = self.num_layers_per_block[block_idx]
            block_configs = []
            for layer_idx in range(num_layers):
                layer_in_channels = current_channels + layer_idx * self.growth_rate
                bottleneck_out = self.growth_rate * 4  # bn_size = 4
                bottleneck = self._create_conv_config_from_params(
                    input_height=current_height,
                    input_width=current_width,
                    in_channels=layer_in_channels,
                    out_channels=bottleneck_out,
                    kernel_size=(1, 1),
                    stride=(1, 1),
                    padding=(0, 0),
                    params_key=f"encoder{block_idx}.layer{layer_idx}.bottleneck",
                )
                expansion = self._create_conv_config_from_params(
                    input_height=current_height,
                    input_width=current_width,
                    in_channels=bottleneck_out,
                    out_channels=self.growth_rate,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=(1, 1),
                    params_key=f"encoder{block_idx}.layer{layer_idx}.expansion",
                )
                
                block_configs.append(DenseLayerConfiguration(
                    bottleneck_conv=bottleneck,
                    expansion_conv=expansion,
                ))
            
            encoder_blocks.append(block_configs)
            
            current_channels = current_channels + num_layers * self.growth_rate
            skip_channels.append(current_channels)
            trans_out_channels = int(current_channels * self.compression)
            transition = self._create_conv_config_from_params(
                input_height=current_height,
                input_width=current_width,
                in_channels=current_channels,
                out_channels=trans_out_channels,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                params_key=f"transition_down{block_idx}",
            )
            transitions_down.append(transition)
            pool = self._create_pool_config(current_height, current_width, trans_out_channels)
            pools.append(pool)
            current_height = current_height // 2
            current_width = current_width // 2
            current_channels = trans_out_channels
        
        bottleneck_conv1 = self._create_conv_config_from_params(
            input_height=current_height,
            input_width=current_width,
            in_channels=current_channels,
            out_channels=current_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            params_key="bottleneck.conv1",
        )
        
        bottleneck_conv2 = self._create_conv_config_from_params(
            input_height=current_height,
            input_width=current_width,
            in_channels=current_channels,
            out_channels=current_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            params_key="bottleneck.conv2",
        )
        upconvs = []
        attention_gates = []
        decoder_blocks = []
        reversed_skip_channels = list(reversed(skip_channels))
        
        for stage_idx in range(self.num_encoder_blocks):
            skip_ch = reversed_skip_channels[stage_idx]
            upconv = self._create_upconv_config(
                input_height=current_height,
                input_width=current_width,
                in_channels=current_channels,
                out_channels=skip_ch,
                params_key=f"upconv{stage_idx}",
            )
            upconvs.append(upconv)
            current_height = current_height * 2
            current_width = current_width * 2
            inter_channels = skip_ch // 2 if skip_ch // 2 > 0 else 1
            attention = self._create_attention_gate_config(
                input_height=current_height,
                input_width=current_width,
                in_channels=skip_ch,
                gating_channels=skip_ch,
                inter_channels=inter_channels,
                params_key=f"attention{stage_idx}",
            )
            attention_gates.append(attention)
            decoder_in_channels = skip_ch + skip_ch  # Concatenated
            decoder_out_channels = skip_ch
            
            decoder_conv1 = self._create_conv_config_from_params(
                input_height=current_height,
                input_width=current_width,
                in_channels=decoder_in_channels,
                out_channels=decoder_out_channels,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                params_key=f"decoder{stage_idx}.conv1",
            )
            
            decoder_conv2 = self._create_conv_config_from_params(
                input_height=current_height,
                input_width=current_width,
                in_channels=decoder_out_channels,
                out_channels=decoder_out_channels,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                params_key=f"decoder{stage_idx}.conv2",
            )
            
            decoder_blocks.append(DecoderBlockConfiguration(
                conv1=decoder_conv1,
                conv2=decoder_conv2,
            ))
            
            current_channels = decoder_out_channels
        conv_out = self._create_conv_config_from_params(
            input_height=current_height,
            input_width=current_width,
            in_channels=current_channels,
            out_channels=self.out_channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            params_key="conv_out",
        )
        
        return TtAttentionDenseUNetConfigs(
            l1_input_memory_config=l1_input_memory_config,
            conv0=conv0,
            encoder_blocks=encoder_blocks,
            transitions_down=transitions_down,
            pools=pools,
            bottleneck_conv1=bottleneck_conv1,
            bottleneck_conv2=bottleneck_conv2,
            upconvs=upconvs,
            attention_gates=attention_gates,
            decoder_blocks=decoder_blocks,
            conv_out=conv_out,
            batch_size=self.batch_size,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            input_height=self.input_height,
            input_width=self.input_width,
        )
    
    def _create_conv_config_from_params(
        self,
        input_height: int,
        input_width: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int],
        stride: Tuple[int, int],
        padding: Tuple[int, int],
        params_key: str,
        output_dtype=ttnn.bfloat16,
        math_fidelity=ttnn.MathFidelity.LoFi,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
        enable_weights_double_buffer=False,
        enable_act_double_buffer=False,
    ) -> Conv2dConfiguration:
        """
        Create Conv2dConfiguration from preprocessed parameters.
        
        For Stage 1, we use simple DRAM memory configuration.
        Stage 2 will add sharding optimizations.
        """
        params = self.parameters
        for key in params_key.split("."):
            params = params[key]
        
        weight = params["weight"]
        bias = params["bias"]
        strategy = AutoShardedStrategyConfiguration()
        
        return Conv2dConfiguration(
            input_height=input_height,
            input_width=input_width,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            batch_size=self.batch_size,
            weight=weight,
            bias=bias,
            output_dtype=output_dtype,
            math_fidelity=math_fidelity,
            fp32_dest_acc_en=fp32_dest_acc_en,
            packer_l1_acc=packer_l1_acc,
            enable_weights_double_buffer=enable_weights_double_buffer,
            enable_act_double_buffer=enable_act_double_buffer,
            sharding_strategy=strategy,
        )
    
    def _create_pool_config(
        self,
        input_height: int,
        input_width: int,
        channels: int,
    ) -> MaxPool2dConfiguration:
        """Create MaxPool2dConfiguration."""
        
        strategy = AutoShardedStrategyConfiguration()
        
        return MaxPool2dConfiguration(
            input_height=input_height,
            input_width=input_width,
            channels=channels,
            batch_size=self.batch_size,
            kernel_size=(2, 2),
            stride=(2, 2),
            padding=(0, 0),
        )
    
    def _create_upconv_config(
        self,
        input_height: int,
        input_width: int,
        in_channels: int,
        out_channels: int,
        params_key: str,
    ) -> UpconvConfiguration:
        """Create UpconvConfiguration from parameters."""
        params = self.parameters
        for key in params_key.split("."):
            params = params[key]
        
        weight = params["weight"]
        bias = params.get("bias", None)
        
        return UpconvConfiguration(
            input_height=input_height,
            input_width=input_width,
            in_channels=in_channels,
            out_channels=out_channels,
            batch_size=self.batch_size,
            kernel_size=(2, 2),
            stride=(2, 2),
            padding=(0, 0),
            weight=weight,
            bias=bias,
        )
    
    def _create_attention_gate_config(
        self,
        input_height: int,
        input_width: int,
        in_channels: int,
        gating_channels: int,
        inter_channels: int,
        params_key: str,
    ) -> AttentionGateConfiguration:
        """Create AttentionGateConfiguration from parameters."""
        params = self.parameters
        for key in params_key.split("."):
            params = params[key]
        theta_conv = Conv2dConfiguration(
            input_height=input_height,
            input_width=input_width,
            in_channels=in_channels,
            out_channels=inter_channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            batch_size=self.batch_size,
            weight=params["theta"]["weight"],
            bias=params["theta"]["bias"],
            output_dtype=ttnn.bfloat16,
            math_fidelity=ttnn.MathFidelity.LoFi,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
            enable_weights_double_buffer=False,
            enable_act_double_buffer=False,
            sharding_strategy=AutoShardedStrategyConfiguration(),
        )
        phi_conv = Conv2dConfiguration(
            input_height=input_height,
            input_width=input_width,
            in_channels=gating_channels,
            out_channels=inter_channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            batch_size=self.batch_size,
            weight=params["phi"]["weight"],
            bias=params["phi"]["bias"],
            output_dtype=ttnn.bfloat16,
            math_fidelity=ttnn.MathFidelity.LoFi,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
            enable_weights_double_buffer=False,
            enable_act_double_buffer=False,
            sharding_strategy=AutoShardedStrategyConfiguration(),
        )
        psi_conv = Conv2dConfiguration(
            input_height=input_height,
            input_width=input_width,
            in_channels=inter_channels,
            out_channels=1,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            batch_size=self.batch_size,
            weight=params["psi"]["weight"],
            bias=params["psi"]["bias"],
            output_dtype=ttnn.bfloat16,
            math_fidelity=ttnn.MathFidelity.LoFi,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
            enable_weights_double_buffer=False,
            enable_act_double_buffer=False,
            sharding_strategy=AutoShardedStrategyConfiguration(),
        )
        W_conv = Conv2dConfiguration(
            input_height=input_height,
            input_width=input_width,
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            batch_size=self.batch_size,
            weight=params["W"]["weight"],
            bias=params["W"]["bias"],
            output_dtype=ttnn.bfloat16,
            math_fidelity=ttnn.MathFidelity.LoFi,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
            enable_weights_double_buffer=False,
            enable_act_double_buffer=False,
            sharding_strategy=AutoShardedStrategyConfiguration(),
        )
        
        return AttentionGateConfiguration(
            theta_conv=theta_conv,
            phi_conv=phi_conv,
            psi_conv=psi_conv,
            W_conv=W_conv,
            in_channels=in_channels,
            gating_channels=gating_channels,
            inter_channels=inter_channels,
        )


def create_configs_from_parameters(
    parameters: Dict,
    in_channels: int = 3,
    out_channels: int = 1,
    input_height: int = 256,
    input_width: int = 256,
    batch_size: int = 1,
    init_features: int = 32,
    growth_rate: int = 16,
) -> TtAttentionDenseUNetConfigs:
    """
    Create Attention DenseUNet configuration from preprocessed parameters.
    
    Args:
        parameters: Preprocessed weights dictionary
        in_channels: Number of input channels (default: 3 for RGB)
        out_channels: Number of output classes (default: 1 for binary segmentation)
        input_height: Input image height
        input_width: Input image width
        batch_size: Batch size
        init_features: Initial number of features
        growth_rate: DenseNet growth rate
        
    Returns:
        TtAttentionDenseUNetConfigs object
    """
    
    builder = TtAttentionDenseUNetConfigBuilder(
        parameters=parameters,
        in_channels=in_channels,
        out_channels=out_channels,
        input_height=input_height,
        input_width=input_width,
        batch_size=batch_size,
        init_features=init_features,
        growth_rate=growth_rate,
    )
    
    return builder.build_configs()
