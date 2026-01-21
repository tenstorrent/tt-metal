# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import math
from transformers import LlamaForCausalLM

class TtLlasaRMSNorm:
    def __init__(self, config, parameters, weight_key="weight", eps_key="rms_norm_eps"):
        self.eps = getattr(config, eps_key, 1e-5)
        self.weight = parameters[weight_key]

    def __call__(self, x):
        return ttnn.rms_norm(x, weight=self.weight, epsilon=self.eps)

class TtLlasaMLP:
    def __init__(self, config, parameters):
        self.gate_proj = parameters.gate_proj
        self.up_proj = parameters.up_proj
        self.down_proj = parameters.down_proj

    def __call__(self, x):
        # SwiGLU: (gate(x) * x) * up(x) -> wait, standard is (swish(gate(x)) * up(x)) ?
        # Llama MLP: down(act(gate(x)) * up(x))
        # where act is SiLU (Swish)
        
        gate = x @ self.gate_proj.weight
        gate = ttnn.silu(gate)
        
        up = x @ self.up_proj.weight
        
        inter = ttnn.mul(gate, up)
        
        down = inter @ self.down_proj.weight
        return down

class TtLlasaAttention:
    def __init__(self, config, parameters):
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        
        self.q_proj = parameters.q_proj
        self.k_proj = parameters.k_proj
        self.v_proj = parameters.v_proj
        self.o_proj = parameters.o_proj
        
        # Scaling factor
        self.scale = 1 / math.sqrt(self.head_dim)

    def __call__(self, x, freqs_cis=None, mask=None):
        batch, seq_len, _ = x.shape
        
        xq = x @ self.q_proj.weight
        xk = x @ self.k_proj.weight
        xv = x @ self.v_proj.weight
        
        # Reshape for multi-head attention
        # [Batch, Seq, Heads, HeadDim]
        xq = ttnn.reshape(xq, (batch, seq_len, self.num_heads, self.head_dim))
        xk = ttnn.reshape(xk, (batch, seq_len, self.num_kv_heads, self.head_dim))
        xv = ttnn.reshape(xv, (batch, seq_len, self.num_kv_heads, self.head_dim))
        
        # RoPE Rotation (Simplified placeholder - assuming inputs are pre-rotated or using ttnn.rope if straightforward)
        # For now, let's assume valid RoPE implementation is needed.
        # This is a critical part.
        
        # Transpose for attention: [Batch, Heads, Seq, HeadDim]
        xq = ttnn.permute(xq, (0, 2, 1, 3))
        xk = ttnn.permute(xk, (0, 2, 1, 3))
        xv = ttnn.permute(xv, (0, 2, 1, 3))
        
        # Repeat KV heads for GQA
        if self.num_kv_groups > 1:
            # Need to repeat k and v
             # ttnn.repeat_interleave equivalent?
             # Or expand?
             # For simpler bringup, assume MQA/GQA handling via repeat
             pass
        
        # Attention Scores: Q @ K^T
        xk_t = ttnn.permute(xk, (0, 1, 3, 2))
        scores = xq @ xk_t
        scores = scores * self.scale
        
        if mask is not None:
            scores = scores + mask
            
        attn_weights = ttnn.softmax(scores, dim=-1)
        
        # Output: Weights @ V
        output = attn_weights @ xv
        
        # Transpose back: [Batch, Seq, Heads, HeadDim]
        output = ttnn.permute(output, (0, 2, 1, 3))
        output = ttnn.reshape(output, (batch, seq_len, self.hidden_size))
        
        # Final Projection
        output = output @ self.o_proj.weight
        
        return output

class TtLlasaDecoderLayer:
    def __init__(self, config, parameters, layer_idx):
        self.input_layernorm = TtLlasaRMSNorm(config, parameters.input_layernorm)
        self.self_attn = TtLlasaAttention(config, parameters.self_attn)
        self.post_attention_layernorm = TtLlasaRMSNorm(config, parameters.post_attention_layernorm)
        self.mlp = TtLlasaMLP(config, parameters.mlp)

    def __call__(self, x, mask=None):
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, mask=mask)
        x = x + residual
        
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = x + residual
        
        return x

class TtLlasaModel:
    def __init__(self, config, parameters, device):
        self.config = config
        self.parameters = parameters
        self.device = device
        
        self.embed_tokens = parameters.model.embed_tokens # Weight tensor
        
        self.layers = [
            TtLlasaDecoderLayer(config, parameters.model.layers[i], i)
            for i in range(config.num_hidden_layers)
        ]
        
        self.norm = TtLlasaRMSNorm(config, parameters.model.norm)
        self.lm_head = parameters.lm_head

    def __call__(self, input_ids):
        # Embedding
        x = ttnn.embedding(input_ids, self.embed_tokens.weight)
        
        # Layers
        for layer in self.layers:
            x = layer(x)
            
        # Final Norm
        x = self.norm(x)
        
        # Head
        logits = x @ self.lm_head.weight
        
        return logits

def custom_preprocessor(torch_model, name):
    parameters = {}
    
    def preprocess_linear(weight, dtype=ttnn.bfloat16):
        return {"weight": ttnn.from_torch(weight.T, dtype=dtype, layout=ttnn.TILE_LAYOUT)}

    if isinstance(torch_model, LlamaForCausalLM): # or Llasa model class
        # Embedding
        parameters["model"] = {"embed_tokens": {}}
        parameters["model"]["embed_tokens"]["weight"] = ttnn.from_torch(
            torch_model.model.embed_tokens.weight, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        
        # Layers
        parameters["model"]["layers"] = []
        for layer in torch_model.model.layers:
            layer_params = {}
            # Self Attn
            layer_params["self_attn"] = {
                "q_proj": preprocess_linear(layer.self_attn.q_proj.weight),
                "k_proj": preprocess_linear(layer.self_attn.k_proj.weight),
                "v_proj": preprocess_linear(layer.self_attn.v_proj.weight),
                "o_proj": preprocess_linear(layer.self_attn.o_proj.weight),
            }
            # MLP
            layer_params["mlp"] = {
                "gate_proj": preprocess_linear(layer.mlp.gate_proj.weight),
                "up_proj": preprocess_linear(layer.mlp.up_proj.weight),
                "down_proj": preprocess_linear(layer.mlp.down_proj.weight),
            }
            # Norms
            layer_params["input_layernorm"] = {"weight": ttnn.from_torch(layer.input_layernorm.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)}
            layer_params["post_attention_layernorm"] = {"weight": ttnn.from_torch(layer.post_attention_layernorm.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)}
            
            parameters["model"]["layers"].append(layer_params)
            
        # Final Norm
        parameters["model"]["norm"] = {"weight": ttnn.from_torch(torch_model.model.norm.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)}
        
        # LM Head
        parameters["lm_head"] = preprocess_linear(torch_model.lm_head.weight)
        
    return parameters
