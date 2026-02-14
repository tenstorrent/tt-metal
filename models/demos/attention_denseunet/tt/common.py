# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Common utilities for Attention DenseUNet TTNN implementation.

This module provides:
- Memory configuration constants
- Model preprocessing functions for converting PyTorch weights to TTNN format
"""

import torch
import ttnn


# Memory Configuration Constants
ATTENTION_DENSEUNET_L1_SMALL_SIZE = 24 * 8192  # Larger than vanilla unet due to dense connections
ATTENTION_DENSEUNET_TRACE_SIZE = 512 * 1024
ATTENTION_DENSEUNET_PCC = 0.97  # Target PCC for validation


def create_preprocessor(device, mesh_mapper=None):
    """
    Create custom preprocessor for Attention DenseUNet model.
    
    This preprocessor converts PyTorch model weights to TTNN format and folds
    BatchNorm layers into Conv layers for efficiency.
    
    Args:
        device: TTNN device
        mesh_mapper: Optional mesh mapper for multi-device setups
        
    Returns:
        Preprocessing function that takes model, name, and ttnn_module_args
    """
    
    assert (
        device.get_num_devices() == 1 or mesh_mapper is not None
    ), "Expected a mesh mapper for weight tensors if we are using multiple devices"
    
    def custom_preprocessor(model, name, ttnn_module_args):
        from models.demos.attention_denseunet.reference.model import AttentionDenseUNet
        
        parameters = {}
        
        if not isinstance(model, AttentionDenseUNet):
            return parameters
        
        def fuse_bn_before_conv(bn_layer, conv_layer):
            """
            Fuse BN->Conv pattern (BN applied before Conv).
            
            Since BN in eval mode is: y = gamma * (x - mean) / sqrt(var + eps) + beta
            And we apply this BEFORE the conv, we can fuse by:
            1. Scaling each input channel of conv by gamma/sqrt(var + eps)
            2. Adding a bias term to conv to account for BN's shift
            """
            running_mean = bn_layer.running_mean
            running_var = bn_layer.running_var
            eps = bn_layer.eps
            gamma = bn_layer.weight  # scale
            beta = bn_layer.bias     # shift
            
            bn_scale = gamma / torch.sqrt(running_var + eps)
            weight = conv_layer.weight.clone()
            bn_scale_broadcast = bn_scale.view(1, -1, 1, 1)
            weight = weight * bn_scale_broadcast
            if hasattr(conv_layer, 'bias') and conv_layer.bias is not None:
                bias = conv_layer.bias.clone()
            else:
                bias = torch.zeros(conv_layer.out_channels, dtype=weight.dtype, device=weight.device)
            
            bn_offset = -running_mean * bn_scale + beta
            bias += (weight * bn_offset.view(1, -1, 1, 1)).sum(dim=(1, 2, 3))
            
            return weight, bias
                
        from ttnn.model_preprocessing import fold_batch_norm2d_into_conv2d
        conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.conv0, model.bn0)
        parameters["conv0"] = {
            "weight": ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper),
            "bias": ttnn.from_torch(
                torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            ),
        }
        
        for block_idx in range(model.num_encoder_blocks):
            encoder_block = model.encoder_blocks[block_idx]
            parameters[f"encoder{block_idx}"] = {}
            
            for layer_idx, dense_layer in enumerate(encoder_block.layers):
                parameters[f"encoder{block_idx}"][f"layer{layer_idx}"] = {}
                
                conv_weight, conv_bias = fuse_bn_before_conv(dense_layer.bn1, dense_layer.conv1)
                parameters[f"encoder{block_idx}"][f"layer{layer_idx}"]["bottleneck"] = {
                    "weight": ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper),
                    "bias": ttnn.from_torch(
                        torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
                    ),
                }
                
                conv_weight, conv_bias = fuse_bn_before_conv(dense_layer.bn2, dense_layer.conv2)
                parameters[f"encoder{block_idx}"][f"layer{layer_idx}"]["expansion"] = {
                    "weight": ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper),
                    "bias": ttnn.from_torch(
                        torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
                    ),
                }
        
        for trans_idx in range(model.num_encoder_blocks):
            transition = model.transitions_down[trans_idx]
            
            conv_weight, conv_bias = fuse_bn_before_conv(transition.bn, transition.conv)
            parameters[f"transition_down{trans_idx}"] = {
                "weight": ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper),
                "bias": ttnn.from_torch(
                    torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
                ),
            }
        
        parameters["bottleneck"] = {}
        
        conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.bottleneck[0], model.bottleneck[1])
        parameters["bottleneck"]["conv1"] = {
            "weight": ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper),
            "bias": ttnn.from_torch(
                torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            ),
        }
        
        conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.bottleneck[3], model.bottleneck[4])
        parameters["bottleneck"]["conv2"] = {
            "weight": ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper),
            "bias": ttnn.from_torch(
                torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            ),
        }
        num_decoder_stages = len(model.transitions_up)
        
        for stage_idx in range(num_decoder_stages):
            upconv = model.transitions_up[stage_idx]
            parameters[f"upconv{stage_idx}"] = {
                "weight": ttnn.from_torch(upconv.upconv.weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper),
                "bias": ttnn.from_torch(
                    torch.reshape(upconv.upconv.bias, (1, 1, 1, -1)),
                    dtype=ttnn.bfloat16,
                    mesh_mapper=mesh_mapper,
                ) if upconv.upconv.bias is not None else None,
            }
            
            att_gate = model.attention_gates[stage_idx]
            parameters[f"attention{stage_idx}"] = {
                "theta": {
                    "weight": ttnn.from_torch(att_gate.theta.weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper),
                   "bias": None if att_gate.theta.bias is None else ttnn.from_torch(
                        torch.reshape(att_gate.theta.bias, (1, 1, 1, -1)),
                        dtype=ttnn.bfloat16,
                        mesh_mapper=mesh_mapper,
                    ),
                },
                "phi": {
                    "weight": ttnn.from_torch(att_gate.phi.weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper),
                    "bias": None if att_gate.phi.bias is None else ttnn.from_torch(
                        torch.reshape(att_gate.phi.bias, (1, 1, 1, -1)),
                        dtype=ttnn.bfloat16,
                        mesh_mapper=mesh_mapper,
                    ),
                },
                "psi": {
                    "weight": ttnn.from_torch(att_gate.psi.weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper),
                    "bias": None if att_gate.psi.bias is None else ttnn.from_torch(
                        torch.reshape(att_gate.psi.bias, (1, 1, 1, -1)),
                        dtype=ttnn.bfloat16,
                        mesh_mapper=mesh_mapper,
                    ),
                },
            }
            
            w_conv = att_gate.W[0]
            w_bn = att_gate.W[1]
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(w_conv, w_bn)
            parameters[f"attention{stage_idx}"]["W"] = {
                "weight": ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper),
                "bias": ttnn.from_torch(
                    torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
                ),
            }
            
            decoder_block = model.decoder_blocks[stage_idx]
            parameters[f"decoder{stage_idx}"] = {}
            
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(decoder_block.conv1, decoder_block.bn1)
            parameters[f"decoder{stage_idx}"]["conv1"] = {
                "weight": ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper),
                "bias": ttnn.from_torch(
                    torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
                ),
            }
            
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(decoder_block.conv2, decoder_block.bn2)
            parameters[f"decoder{stage_idx}"]["conv2"] = {
                "weight": ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper),
                "bias": ttnn.from_torch(
                    torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
                ),
            }
        
        parameters["conv_out"] = {
            "weight": ttnn.from_torch(model.conv_out.weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper),
            "bias": ttnn.from_torch(
                torch.reshape(model.conv_out.bias, (1, 1, 1, -1)),
                dtype=ttnn.bfloat16,
                mesh_mapper=mesh_mapper,
            ) if model.conv_out.bias is not None else None,
        }
        
        return parameters
    
    return custom_preprocessor
