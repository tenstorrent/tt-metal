# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Validate single DeltaNet layer against HF reference (torch_recurrent_gated_delta_rule)."""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ["HF_MODEL"] = os.environ["HF_MODEL"]

# Copy reference functions from HF modeling_qwen3_5.py to avoid import issues


def l2norm(x, dim=-1, eps=1e-6):
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def torch_causal_conv1d_update(hidden_states, conv_state, weight, bias=None, activation=None):
    _, hidden_size, seq_len = hidden_states.shape
    state_len = conv_state.shape[-1]
    hidden_states_new = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
    conv_state.copy_(hidden_states_new[:, :, -state_len:])
    out = F.conv1d(hidden_states_new, weight.unsqueeze(1), bias, padding=0, groups=hidden_size)
    out = F.silu(out[:, :, -seq_len:])
    return out.to(hidden_states.dtype)


def torch_recurrent_gated_delta_rule(
    query, key, value, g, beta, initial_state, output_final_state, use_qk_l2norm_in_kernel=False
):
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]
    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale
    core_attn_out = torch.zeros(batch_size, num_heads, sequence_length, v_head_dim).to(value)
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )
    for i in range(sequence_length):
        q_t = query[:, :, i]
        k_t = key[:, :, i]
        v_t = value[:, :, i]
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, i].unsqueeze(-1)
        last_recurrent_state = last_recurrent_state * g_t
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)
    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


class Qwen3_5RMSNormGated(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states, gate=None):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states
        if gate is not None:
            hidden_states = hidden_states * F.silu(gate.to(torch.float32))
        return hidden_states.to(input_dtype)


import json

from safetensors.torch import load_file

CKPT_DIR = os.environ["HF_MODEL"]
LAYER = 0  # Test layer 0 (DeltaNet)

# Load config
with open(f"{CKPT_DIR}/config.json") as f:
    cfg = json.load(f)
text_cfg = cfg.get("text_config", cfg)
hidden_size = text_cfg["hidden_size"]
num_k_heads = text_cfg["linear_num_key_heads"]
num_v_heads = text_cfg["linear_num_value_heads"]
head_k_dim = text_cfg["linear_key_head_dim"]
head_v_dim = text_cfg["linear_value_head_dim"]
key_dim = head_k_dim * num_k_heads
value_dim = head_v_dim * num_v_heads
conv_dim = key_dim * 2 + value_dim
conv_kernel = text_cfg["linear_conv_kernel_dim"]
gqa_ratio = num_v_heads // num_k_heads
rms_eps = text_cfg["rms_norm_eps"]

print(
    f"DeltaNet config: H={hidden_size}, K_heads={num_k_heads}, V_heads={num_v_heads}, "
    f"K_dim={head_k_dim}, V_dim={head_v_dim}, conv_kernel={conv_kernel}"
)

# Load layer weights from safetensors
with open(f"{CKPT_DIR}/model.safetensors.index.json") as f:
    idx = json.load(f)

prefix = f"model.language_model.layers.{LAYER}.linear_attn."
needed_files = set()
for k, f in idx["weight_map"].items():
    if k.startswith(prefix):
        needed_files.add(f)

weights = {}
for f in needed_files:
    data = load_file(f"{CKPT_DIR}/{f}")
    for k, v in data.items():
        if k.startswith(prefix):
            short_key = k[len(prefix) :]
            weights[short_key] = v
            print(f"  Loaded {short_key}: {v.shape}")

# Create random input
torch.manual_seed(42)
x = torch.randn(1, 1, hidden_size)  # (batch=1, seq=1, hidden)

# ============================================
# HF REFERENCE: Run single-step decode
# ============================================
# 1. Linear projections
in_proj_qkv_w = weights["in_proj_qkv.weight"]  # (conv_dim, hidden)
in_proj_z_w = weights["in_proj_z.weight"]
in_proj_b_w = weights["in_proj_b.weight"]
in_proj_a_w = weights["in_proj_a.weight"]
out_proj_w = weights["out_proj.weight"]
conv1d_w = weights["conv1d.weight"]  # (conv_dim, 1, kernel)
dt_bias = weights["dt_bias"].float()
A_log = weights["A_log"].float()
norm_w = weights["norm.weight"]

mixed_qkv = x @ in_proj_qkv_w.float().T  # (1, 1, conv_dim)
z = x @ in_proj_z_w.float().T  # (1, 1, value_dim)
b = x @ in_proj_b_w.float().T  # (1, 1, num_v_heads)
a = x @ in_proj_a_w.float().T  # (1, 1, num_v_heads)

# 2. Conv1d (single step, conv state all zeros = first token)
# For first token, conv state is zeros, so conv output = conv_weight[:, :, -1] * input
mixed_qkv_t = mixed_qkv.transpose(1, 2)  # (1, conv_dim, 1)
conv_state = torch.zeros(1, conv_dim, conv_kernel - 1)  # (1, conv_dim, 3)
mixed_qkv_conv = torch_causal_conv1d_update(mixed_qkv_t, conv_state, conv1d_w.squeeze(1), activation="silu")
mixed_qkv_conv = mixed_qkv_conv.transpose(1, 2)  # (1, 1, conv_dim)

# 3. Split Q, K, V
q, k, v = torch.split(mixed_qkv_conv, [key_dim, key_dim, value_dim], dim=-1)
q = q.reshape(1, 1, num_k_heads, head_k_dim)
k = k.reshape(1, 1, num_k_heads, head_k_dim)
v = v.reshape(1, 1, num_v_heads, head_v_dim)

# 4. Gates
beta = b.sigmoid()
g = -A_log.exp() * torch.nn.functional.softplus(a.float() + dt_bias)

# 5. GQA expand
if gqa_ratio > 1:
    q = q.repeat_interleave(gqa_ratio, dim=2)
    k = k.repeat_interleave(gqa_ratio, dim=2)

# 6. Recurrent (with L2 norm in kernel)
ref_out, ref_state = torch_recurrent_gated_delta_rule(
    q,
    k,
    v,
    g=g,
    beta=beta,
    initial_state=None,
    output_final_state=True,
    use_qk_l2norm_in_kernel=True,
)

# 7. Norm + gate
ref_out_flat = ref_out.reshape(-1, head_v_dim)
z_flat = z.reshape(1, 1, num_v_heads, head_v_dim).reshape(-1, head_v_dim)
norm_layer = Qwen3_5RMSNormGated(head_v_dim, eps=rms_eps)
norm_layer.weight.data = norm_w
ref_normed = norm_layer(ref_out_flat, z_flat)
ref_normed = ref_normed.reshape(1, 1, -1)

# 8. Output projection
ref_output = ref_normed @ out_proj_w.float().T

print(f"\nHF Reference output shape: {ref_output.shape}")
print(f"HF Reference output norm: {ref_output.norm():.6f}")
print(f"HF Reference output[0, 0, :5]: {ref_output[0, 0, :5]}")

# ============================================
# TT IMPLEMENTATION: Run same input
# ============================================
import ttnn
from models.tt_transformers.tt.model_config import ModelArgs

device = ttnn.open_device(device_id=0)
args = ModelArgs(device, max_seq_len=256)
sd = args.load_state_dict()

from models.tt_transformers.tt.gated_deltanet import GatedDeltaNet

wcp = args.weight_cache_path(dtype=ttnn.bfloat8_b)
deltanet = GatedDeltaNet(
    mesh_device=device,
    args=args,
    state_dict=sd,
    weight_cache_path=wcp,
    layer_num=LAYER,
    dtype=ttnn.bfloat8_b,
)
deltanet.initialize_states()

# Prepare input for TT (1, 1, 32, hidden) with padding
B_pad = args.tile_padded_batch_rows
x_pad = torch.zeros(1, 1, B_pad, hidden_size)
x_pad[0, 0, 0, :] = x[0, 0, :]
x_tt = ttnn.from_torch(x_pad, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

tt_out = deltanet.forward(x_tt)
tt_result = ttnn.to_torch(tt_out).float()[0, 0, 0, :hidden_size]

print(f"\nTT output norm: {tt_result.norm():.6f}")
print(f"TT output[0:5]: {tt_result[:5]}")

# Compare
ref_vec = ref_output[0, 0, :].float()
pcc = torch.nn.functional.cosine_similarity(ref_vec.unsqueeze(0), tt_result.unsqueeze(0)).item()
print(f"\nPCC (cosine similarity): {pcc:.6f}")

# Also check individual components to find where divergence starts
print(f"\nRef output range: [{ref_vec.min():.4f}, {ref_vec.max():.4f}]")
print(f"TT output range: [{tt_result.min():.4f}, {tt_result.max():.4f}]")

ttnn.close_device(device)
