# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import math

# ----------------------------------------------------------------------------
# DPT Components (Neck & Head)
# ----------------------------------------------------------------------------

class TtDPTReassembleLayer(torch.nn.Module):
    def __init__(self, config, parameters, read_idx):
        super().__init__()
        self.parameters = parameters
        self.read_idx = read_idx
        # Simplified: We assume we just need to project. 
        # Real implementation needs Upsample/Conv logic which might be complex in ttnn right now.
        # For this bounty stage (bring up), we maintain the structure.

    def forward(self, hidden_state):
        # 1. Read (Slice CLS token if needed) - handled in backbone output
        # 2. Resample (Spatial manipulation) - Hardest part in ttnn
        # 3. Projection (Conv2d 1x1)
        
        # Stub: Just linear projection if possible or return as is for now
        return hidden_state

class TtDPTFusionStage(torch.nn.Module):
    def __init__(self, config, parameters):
        super().__init__()
        self.parameters = parameters
    
    def forward(self, features):
        # Fusion logic: Add + Upsamle + Conv
        # Returning last feature for testing validity of graph
        return features[-1]

class TtDPTHead(torch.nn.Module):
    def __init__(self, config, parameters):
        super().__init__()
        self.parameters = parameters
        
    def forward(self, hidden_state):
        # Final Conv sequence
        # conv1 -> conv2 -> conv3
        return hidden_state


# ----------------------------------------------------------------------------
# ViT Backbone (Simplified Port)
# ----------------------------------------------------------------------------

def vit_layer(hidden_states, parameters):
    # Layernorm 1
    ln1 = ttnn.layer_norm(hidden_states, weight=parameters.layernorm_before.weight, bias=parameters.layernorm_before.bias)
    
    # Self Attention (Simplified Matmuls)
    # Projections
    q = ln1 @ parameters.attention.query.weight + parameters.attention.query.bias
    k = ln1 @ parameters.attention.key.weight + parameters.attention.key.bias
    v = ln1 @ parameters.attention.value.weight + parameters.attention.value.bias
    
    # Q * K
    attn = q @ ttnn.permute(k, (0, 2, 1, 3)) # Placeholder dim
    attn = ttnn.softmax(attn, dim=-1)
    
    # Attn * V
    attn_out = attn @ v
    
    # Output dense
    attn_out = attn_out @ parameters.attention.output.dense.weight + parameters.attention.output.dense.bias
    
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

class TtDepthAnythingV2(torch.nn.Module):
    def __init__(self, parameters):
        super().__init__()
        self.parameters = parameters
        
    def forward(self, pixel_values):
        # 1. Embeddings (Placeholder)
        embeddings = pixel_values 
        
        # 2. Encoder (ViT-Large)
        hidden_states = embeddings
        features = []
        out_indices = [5, 11, 17, 23] # for DPT usage
        
        # Iterate over layers (assumed list in parameters)
        # Note: parameters object structure is crucial here
        for i in range(24): # Large has 24 layers
            layer_params = self.parameters.backbone.encoder.layer[i]
            hidden_states = vit_layer(hidden_states, layer_params)
            
            if i in out_indices:
                features.append(hidden_states)
        
        # 3. Neck (DPT Reassemble & Fusion)
        # Simplified: just passing features through
        
        # 4. Head
        depth = features[-1] # Stub
        
        return depth

def custom_preprocessor(torch_model, name):
    # Convert PyTorch dictionary to ttnn.Model parameters
    # This function creates the 'parameters' object used in __init__
    
    import ttnn
    
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
    # Stubbing for now to allow 'demo.py' to run without crashing on missing keys
    parameters["neck"] = {}
    parameters["head"] = {}
        
    return parameters
