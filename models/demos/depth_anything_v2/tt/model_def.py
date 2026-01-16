# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import math

# ----------------------------------------------------------------------------
# DPT Components (Neck & Head)
# ----------------------------------------------------------------------------

class TtDPTReassembleLayer:
    def __init__(self, parameters, read_idx):
        self.parameters = parameters
        self.read_idx = read_idx

    def __call__(self, hidden_state):
        # 1. Read (Slice CLS token if needed) - handled in backbone output
        # 2. Resample (Spatial manipulation)
        # 3. Projection (Conv2d 1x1)
        # Stub: Just linear projection if possible or return as is for now
        return hidden_state

class TtDPTFusionStage:
    def __init__(self, parameters):
        self.parameters = parameters
    
    def __call__(self, features):
        # Fusion logic: Add + Upsamle + Conv
        # Returning last feature for testing validity of graph
        return features[-1]

class TtDPTHead:
    def __init__(self, parameters):
        self.parameters = parameters
        
    def __call__(self, hidden_state):
        # Final Conv sequence
        # conv1 -> conv2 -> conv3
        return hidden_state


# ----------------------------------------------------------------------------
# ViT Backbone (ttnn Implementation)
# ----------------------------------------------------------------------------

def vit_layer(hidden_states, parameters, config):
    # Layernorm 1
    ln1 = ttnn.layer_norm(hidden_states, weight=parameters.layernorm_before.weight, bias=parameters.layernorm_before.bias)
    
    num_heads = config.num_attention_heads
    batch_size, sequence_size, hidden_size = hidden_states.shape
    head_size = hidden_size // num_heads

    # Self Attention (Multi-head)
    # Projections
    q = ln1 @ parameters.attention.query.weight + parameters.attention.query.bias
    k = ln1 @ parameters.attention.key.weight + parameters.attention.key.bias
    v = ln1 @ parameters.attention.value.weight + parameters.attention.value.bias
    
    # Reshape and Permute for Multi-head Attention
    q = ttnn.to_layout(q, layout=ttnn.ROW_MAJOR_LAYOUT)
    q = ttnn.reshape(q, (batch_size, sequence_size, num_heads, head_size))
    q = ttnn.to_layout(q, layout=ttnn.TILE_LAYOUT)
    q = ttnn.permute(q, (0, 2, 1, 3))

    k = ttnn.to_layout(k, layout=ttnn.ROW_MAJOR_LAYOUT)
    k = ttnn.reshape(k, (batch_size, sequence_size, num_heads, head_size))
    k = ttnn.to_layout(k, layout=ttnn.TILE_LAYOUT)
    k = ttnn.permute(k, (0, 2, 3, 1)) # Transpose last two dims for QK^T

    v = ttnn.to_layout(v, layout=ttnn.ROW_MAJOR_LAYOUT)
    v = ttnn.reshape(v, (batch_size, sequence_size, num_heads, head_size))
    v = ttnn.to_layout(v, layout=ttnn.TILE_LAYOUT)
    v = ttnn.permute(v, (0, 2, 1, 3))

    # Q * K^T
    attn_scores = q @ k
    # Scaled dot-product attention
    attn_scores = attn_scores * (1 / (head_size ** 0.5))
    attn_probs = ttnn.softmax(attn_scores, dim=-1)
    
    # Attn * V
    context_layer = attn_probs @ v
    
    # Merge heads
    context_layer = ttnn.permute(context_layer, (0, 2, 1, 3))
    context_layer = ttnn.to_layout(context_layer, ttnn.ROW_MAJOR_LAYOUT)
    context_layer = ttnn.reshape(context_layer, (batch_size, sequence_size, hidden_size))
    context_layer = ttnn.to_layout(context_layer, ttnn.TILE_LAYOUT)
    
    # Output dense
    attn_out = context_layer @ parameters.attention.output.dense.weight + parameters.attention.output.dense.bias
    
    # Residual 1
    hidden_states = hidden_states + attn_out
    
    # Layernorm 2
    ln2 = ttnn.layer_norm(hidden_states, weight=parameters.layernorm_after.weight, bias=parameters.layernorm_after.bias)
    
    # MLP
    mlp_out = ln2 @ parameters.intermediate.dense.weight + parameters.intermediate.dense.bias
    mlp_out = ttnn.gelu(mlp_out)
    mlp_out = mlp_out @ parameters.output.dense.weight + parameters.output.dense.bias
    
    # Residual 2
    hidden_states = hidden_states + mlp_out
    
    return hidden_states


# ----------------------------------------------------------------------------
# Main Model: Depth Anything V2
# ----------------------------------------------------------------------------

class TtDepthAnythingV2:
    def __init__(self, config, parameters):
        self.config = config
        self.parameters = parameters
        self.reassemble = [TtDPTReassembleLayer(parameters.neck.reassemble[i], i) for i in range(4)]
        self.fusion = TtDPTFusionStage(parameters.neck.fusion)
        self.head = TtDPTHead(parameters.head)
        
    def __call__(self, pixel_values):
        # 1. Embeddings (Placeholder)
        # In a real implementation, we would use vit_embeddings here
        embeddings = pixel_values 
        
        # 2. Encoder (ViT-Large)
        hidden_states = embeddings
        features = []
        out_indices = [5, 11, 17, 23] # for DPT usage
        
        for i in range(24): # Large has 24 layers
            layer_params = self.parameters.backbone.encoder.layer[i]
            hidden_states = vit_layer(hidden_states, layer_params, self.config)
            
            if i in out_indices:
                features.append(hidden_states)
        
        # 3. Neck (DPT Reassemble & Fusion)
        reassembled_features = [self.reassemble[i](features[i]) for i in range(4)]
        fused_feature = self.fusion(reassembled_features)
        
        # 4. Head
        depth = self.head(fused_feature)
        
        return depth

def custom_preprocessor(torch_model, name):
    parameters = {}
    
    # Helper to convert a single weight
    def convert(tensor):
        return ttnn.from_torch(tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    
    # 1. Backbone
    parameters["backbone"] = {"encoder": {"layer": []}}
    
    # Encoder Layers
    for i, layer in enumerate(torch_model.backbone.encoder.layer):
        layer_params = {
            "layernorm_before": {
                "weight": convert(layer.norm1.weight),
                "bias": convert(layer.norm1.bias)
            },
            "attention": {
                "query": {"weight": convert(layer.attention.attention.query.weight), "bias": convert(layer.attention.attention.query.bias)},
                "key": {"weight": convert(layer.attention.attention.key.weight), "bias": convert(layer.attention.attention.key.bias)},
                "value": {"weight": convert(layer.attention.attention.value.weight), "bias": convert(layer.attention.attention.value.bias)},
                "output": {"dense": {"weight": convert(layer.attention.output.dense.weight), "bias": convert(layer.attention.output.dense.bias)}}
            },
            "layernorm_after": {
                "weight": convert(layer.norm2.weight),
                "bias": convert(layer.norm2.bias)
            },
            "intermediate": {
                "dense": {"weight": convert(layer.mlp.fc1.weight), "bias": convert(layer.mlp.fc1.bias)}
            },
            "output": {
                "dense": {"weight": convert(layer.mlp.fc2.weight), "bias": convert(layer.mlp.fc2.bias)}
            }
        }
        parameters["backbone"]["encoder"]["layer"].append(layer_params)

    # DPT Head (Neck + Head)
    # Mapping required keys for reassemble/fusion/head to avoid crashes
    parameters["neck"] = {
        "reassemble": [{} for _ in range(4)],
        "fusion": {}
    }
    parameters["head"] = {}
        
    return parameters
