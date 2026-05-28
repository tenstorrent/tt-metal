import torch
import ttnn
from models.common.lightweightmodule import LightweightModule

class TtPhiMLP(LightweightModule):
    def __init__(self, mesh_device, state_dict, state_dict_prefix, config, dtype):
        super().__init__()
        self.fc1_weight = ttnn.from_torch(state_dict[f"{state_dict_prefix}.fc1.weight"], device=mesh_device, dtype=dtype, layout=ttnn.TILE_LAYOUT)
        self.fc1_bias = ttnn.from_torch(state_dict[f"{state_dict_prefix}.fc1.bias"].reshape(1, 1, 1, -1), device=mesh_device, dtype=dtype, layout=ttnn.TILE_LAYOUT)
        self.fc2_weight = ttnn.from_torch(state_dict[f"{state_dict_prefix}.fc2.weight"], device=mesh_device, dtype=dtype, layout=ttnn.TILE_LAYOUT)
        self.fc2_bias = ttnn.from_torch(state_dict[f"{state_dict_prefix}.fc2.bias"].reshape(1, 1, 1, -1), device=mesh_device, dtype=dtype, layout=ttnn.TILE_LAYOUT)

    def forward(self, x):
        x = ttnn.linear(x, self.fc1_weight, bias=self.fc1_bias)
        x = ttnn.gelu(x)
        x = ttnn.linear(x, self.fc2_weight, bias=self.fc2_bias)
        return x

class TtPhiAttention(LightweightModule):
    def __init__(self, mesh_device, state_dict, state_dict_prefix, config, dtype):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.rotary_dim = int(self.head_dim * config.partial_rotary_factor)
        self.Wqkv_weight = ttnn.from_torch(state_dict[f"{state_dict_prefix}.Wqkv.weight"], device=mesh_device, dtype=dtype, layout=ttnn.TILE_LAYOUT)
        self.Wqkv_bias = ttnn.from_torch(state_dict[f"{state_dict_prefix}.Wqkv.bias"].reshape(1, 1, 1, -1), device=mesh_device, dtype=dtype, layout=ttnn.TILE_LAYOUT)
        self.out_proj_weight = ttnn.from_torch(state_dict[f"{state_dict_prefix}.out_proj.weight"], device=mesh_device, dtype=dtype, layout=ttnn.TILE_LAYOUT)
        self.out_proj_bias = ttnn.from_torch(state_dict[f"{state_dict_prefix}.out_proj.bias"].reshape(1, 1, 1, -1), device=mesh_device, dtype=dtype, layout=ttnn.TILE_LAYOUT)

    def forward(self, x, mask=None, layer_past=None, use_cache=False, rotary_embedding=None):
        qkv = ttnn.linear(x, self.Wqkv_weight, bias=self.Wqkv_bias)
        q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(qkv, num_heads=self.num_heads, num_kv_heads=self.num_heads, transpose_key=True)
        if rotary_embedding is not None:
            q_rot, q_pass = ttnn.split(q, self.rotary_dim, dim=-1)
            k = ttnn.transpose(k, -2, -1)
            k_rot, k_pass = ttnn.split(k, self.rotary_dim, dim=-1)
            q_rot = ttnn.transformer.rotary_embedding(q_rot, rotary_embedding)
            k_rot = ttnn.transformer.rotary_embedding(k_rot, rotary_embedding)
            q = ttnn.concat([q_rot, q_pass], dim=-1)
            k = ttnn.concat([k_rot, k_pass], dim=-1)
        if use_cache and layer_past is not None:
            k_cache, v_cache = layer_past
            k = ttnn.concat([k_cache, k], dim=-2)
            v = ttnn.concat([v_cache, v], dim=-2)
        k_for_sdpa = ttnn.transpose(k, -2, -1)
        attn_output = ttnn.transformer.scaled_dot_product_attention(q, k_for_sdpa, v, is_causal=True if mask is None else False, attn_mask=mask)
        attn_output = ttnn.transformer.concatenate_heads(attn_output)
        out = ttnn.linear(attn_output, self.out_proj_weight, bias=self.out_proj_bias)
        return out, (k, v)

class TtPhiBlock(LightweightModule):
    def __init__(self, mesh_device, state_dict, state_dict_prefix, config, dtype):
        super().__init__()
        self.ln = ttnn.LayerNorm(device=mesh_device, dim=config.hidden_size, eps=config.layer_norm_eps, weight=ttnn.from_torch(state_dict[f"{state_dict_prefix}.ln.weight"], device=mesh_device, dtype=dtype, layout=ttnn.TILE_LAYOUT), bias=ttnn.from_torch(state_dict[f"{state_dict_prefix}.ln.bias"], device=mesh_device, dtype=dtype, layout=ttnn.TILE_LAYOUT))
        self.mixer = TtPhiAttention(mesh_device, state_dict, f"{state_dict_prefix}.mixer", config, dtype)
        self.mlp = TtPhiMLP(mesh_device, state_dict, f"{state_dict_prefix}.mlp", config, dtype)

    def forward(self, x, mask=None, layer_past=None, use_cache=False, rotary_embedding=None):
        residual = x
        x_norm = self.ln(x)
        attn_out, present_key_value = self.mixer(x_norm, mask=mask, layer_past=layer_past, use_cache=use_cache, rotary_embedding=rotary_embedding)
        mlp_out = self.mlp(x_norm)
        tmp = ttnn.add(residual, attn_out)
        out = ttnn.add(tmp, mlp_out)
        return out, present_key_value

class TtPhiModel(LightweightModule):
    def __init__(self, mesh_device, state_dict, config, dtype):
        super().__init__()
        self.embd = ttnn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.hidden_size, weight=ttnn.from_torch(state_dict["embd.weight"], device=mesh_device, dtype=dtype))
        self.layers = [TtPhiBlock(mesh_device, state_dict, f"layers.{i}", config, dtype) for i in range(config.num_hidden_layers)]
        self.final_norm = ttnn.LayerNorm(device=mesh_device, dim=config.hidden_size, eps=config.layer_norm_eps, weight=ttnn.from_torch(state_dict["final_norm.weight"], device=mesh_device, dtype=dtype, layout=ttnn.TILE_LAYOUT), bias=ttnn.from_torch(state_dict["final_norm.bias"], device=mesh_device, dtype=dtype, layout=ttnn.TILE_LAYOUT))
        self.lm_head_weight = ttnn.from_torch(state_dict["lm_head.weight"], device=mesh_device, dtype=dtype, layout=ttnn.TILE_LAYOUT)
        self.lm_head_bias = ttnn.from_torch(state_dict["lm_head.bias"].reshape(1, 1, 1, -1), device=mesh_device, dtype=dtype, layout=ttnn.TILE_LAYOUT)

    def forward(self, x, mask=None, past_key_values=None, use_cache=False, rotary_embedding=None):
        hidden_states = self.embd(x)
        new_past_key_values = []
        for i, layer in enumerate(self.layers):
            layer_past = past_key_values[i] if past_key_values is not None else None
            hidden_states, present_key_value = layer(hidden_states, mask=mask, layer_past=layer_past, use_cache=use_cache, rotary_embedding=rotary_embedding)
            if use_cache: new_past_key_values.append(present_key_value)
        hidden_states = self.final_norm(hidden_states)
        logits = ttnn.linear(hidden_states, self.lm_head_weight, bias=self.lm_head_bias)
        return logits, new_past_key_values
