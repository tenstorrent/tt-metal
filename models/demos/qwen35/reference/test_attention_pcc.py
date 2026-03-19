# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Validate single GatedAttention layer against HF reference."""
import os

import torch
import torch.nn.functional as F

os.environ["HF_MODEL"] = os.environ["HF_MODEL"]

import json

from safetensors.torch import load_file

CKPT_DIR = os.environ["HF_MODEL"]
# Layer 3 is the first full_attention layer (layers 0,1,2 are DeltaNet, layer 3 is full_attention)
# Check layer_types
with open(f"{CKPT_DIR}/config.json") as f:
    cfg = json.load(f)
text_cfg = cfg.get("text_config", cfg)
layer_types = text_cfg["layer_types"]
# Find first full_attention layer
LAYER = next(i for i, t in enumerate(layer_types) if t == "full_attention")
print(f"Testing layer {LAYER} (type: {layer_types[LAYER]})")

hidden_size = text_cfg["hidden_size"]  # 5120
n_heads = text_cfg["num_attention_heads"]  # 24
n_kv_heads = text_cfg["num_key_value_heads"]  # 4
head_dim = text_cfg.get("head_dim", hidden_size // n_heads)  # 256
rms_eps = text_cfg["rms_norm_eps"]
partial_rotary = text_cfg.get("rope_parameters", {}).get("partial_rotary_factor", 1.0)
rope_theta = text_cfg.get("rope_theta", text_cfg.get("rope_parameters", {}).get("rope_theta", 10000))
rotary_dim = int(head_dim * partial_rotary)

print(
    f"Attention: n_heads={n_heads}, n_kv_heads={n_kv_heads}, head_dim={head_dim}, "
    f"rotary_dim={rotary_dim}, rope_theta={rope_theta}"
)

# Load weights from safetensors
with open(f"{CKPT_DIR}/model.safetensors.index.json") as f:
    idx = json.load(f)

prefix = f"model.language_model.layers.{LAYER}.self_attn."
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
            weights[short_key] = v.float()
            print(f"  Loaded {short_key}: {v.shape}")

# Create input + position
torch.manual_seed(42)
x = torch.randn(1, 1, hidden_size)  # (batch=1, seq=1, hidden)
pos = 0  # Use 0 to avoid zero-dilution from empty cache

# ============================================
# HF REFERENCE
# ============================================
q_proj_w = weights["q_proj.weight"]  # (n_heads * head_dim * 2, hidden) = (12288, 5120)
k_proj_w = weights["k_proj.weight"]  # (n_kv_heads * head_dim, hidden) = (1024, 5120)
v_proj_w = weights["v_proj.weight"]  # (1024, 5120)
o_proj_w = weights["o_proj.weight"]  # (n_heads * head_dim, hidden) = (6144, 5120)
q_norm_w = weights["q_norm.weight"]  # (head_dim,) = (256,)
k_norm_w = weights["k_norm.weight"]  # (256,)

# Q projection + split into query and gate
q_out = x @ q_proj_w.T  # (1, 1, 12288)
q_out = q_out.view(1, 1, n_heads, head_dim * 2)
query, gate = torch.chunk(q_out, 2, dim=-1)  # each (1, 1, 24, 256)
gate = gate.reshape(1, 1, -1)  # (1, 1, 6144)

# K, V projections
k_out = (x @ k_proj_w.T).view(1, 1, n_kv_heads, head_dim)  # (1, 1, 4, 256)
v_out = (x @ v_proj_w.T).view(1, 1, n_kv_heads, head_dim)  # (1, 1, 4, 256)


# Q/K norms (Qwen3_5RMSNorm: zero-centered, weight initialized to zeros, applies (1+w))
def qwen_rms_norm(x, w, eps):
    x_f = x.float()
    variance = x_f.pow(2).mean(-1, keepdim=True)
    x_normed = x_f * torch.rsqrt(variance + eps)
    return (x_normed * (1.0 + w.float())).to(x.dtype)


query = qwen_rms_norm(query, q_norm_w, rms_eps)
k_out = qwen_rms_norm(k_out, k_norm_w, rms_eps)

# Transpose for attention: (batch, heads, seq, dim)
query = query.transpose(1, 2)  # (1, 24, 1, 256)
k_out = k_out.transpose(1, 2)  # (1, 4, 1, 256)
v_out = v_out.transpose(1, 2)  # (1, 4, 1, 256)

# Partial RoPE
half_dim = rotary_dim // 2
inv_freq = 1.0 / (rope_theta ** (torch.arange(0, rotary_dim, 2).float() / rotary_dim))
freqs = pos * inv_freq
cos_val = freqs.cos()
sin_val = freqs.sin()


def apply_partial_rope(x):
    x1 = x[..., :half_dim]
    x2 = x[..., half_dim:rotary_dim]
    x_rot = torch.cat([x1 * cos_val - x2 * sin_val, x2 * cos_val + x1 * sin_val], dim=-1)
    return torch.cat([x_rot, x[..., rotary_dim:]], dim=-1)


query = apply_partial_rope(query)
k_out = apply_partial_rope(k_out)

# GQA: repeat KV
n_rep = n_heads // n_kv_heads
k_expanded = k_out.repeat_interleave(n_rep, dim=1)  # (1, 24, 1, 256)
v_expanded = v_out.repeat_interleave(n_rep, dim=1)  # (1, 24, 1, 256)

# Attention (single token, just dot product)
scale = head_dim**-0.5
attn_weights = (query @ k_expanded.transpose(-2, -1)) * scale  # (1, 24, 1, 1)
attn_weights = F.softmax(attn_weights, dim=-1)
attn_output = attn_weights @ v_expanded  # (1, 24, 1, 256)

# Reshape and apply gate
attn_output = attn_output.transpose(1, 2).reshape(1, 1, -1)  # (1, 1, 6144)
attn_output = attn_output * torch.sigmoid(gate)

# Output projection
ref_output = attn_output @ o_proj_w.T  # (1, 1, 5120)

print(f"\nHF Reference output norm: {ref_output.norm():.6f}")
print(f"HF Reference output[0, 0, :5]: {ref_output[0, 0, :5]}")

# ============================================
# TT IMPLEMENTATION
# ============================================
import ttnn
from models.tt_transformers.tt.model_config import ModelArgs

device = ttnn.open_device(device_id=0)
args = ModelArgs(device, max_seq_len=256)
sd = args.load_state_dict()

from models.tt_transformers.tt.gated_attention import GatedAttention
from models.tt_transformers.tt.rope import RotarySetup

wcp = args.weight_cache_path(dtype=ttnn.bfloat8_b)

# Create RotarySetup (needed for transformation_mats even though custom_rope_fn bypasses it)
rope_setup = RotarySetup(
    device=device,
    batch_size=args.max_batch_size,
    head_dim=args.head_dim,
    max_seq_len=args.max_seq_len,
    rope_theta=args.rope_theta,
    rope_scaling=args.rope_scaling,
    use_qk_fused=args.use_qk_fused,
)
trans_mats = rope_setup.get_both_trans_mats()

attn = GatedAttention(
    mesh_device=device,
    tt_ccl=None,
    args=args,
    state_dict=sd,
    weight_cache_path=wcp,
    layer_num=LAYER,
    dtype=ttnn.bfloat8_b,
    transformation_mats=trans_mats,
    configuration=args,
)

# Check if custom_rope_fn was installed
print(f"\ncustom_rope_fn installed: {attn.custom_rope_fn is not None}")
print(f"gate_weight loaded: {attn.gate_weight is not None}")

# Prepare input
B_pad = args.tile_padded_batch_rows
x_pad = torch.zeros(1, 1, B_pad, hidden_size)
x_pad[0, 0, 0, :] = x[0, 0, :]
x_tt = ttnn.from_torch(x_pad, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

tt_pos = ttnn.from_torch(torch.tensor([pos], dtype=torch.int32), dtype=ttnn.int32, device=device)
rot_idxs = ttnn.from_torch(torch.tensor([[pos]], dtype=torch.int64), device=device)
rot_mats = rope_setup.get_rot_mats(rot_idxs)

tt_out = attn.forward(x_tt, current_pos=tt_pos, rot_mats=rot_mats, mode="decode")
tt_result = ttnn.to_torch(tt_out).float()[0, 0, 0, :hidden_size]

print(f"\nTT output norm: {tt_result.norm():.6f}")
print(f"TT output[0:5]: {tt_result[:5]}")

# Compare
ref_vec = ref_output[0, 0, :].float().detach()
pcc = torch.nn.functional.cosine_similarity(ref_vec.unsqueeze(0), tt_result.unsqueeze(0)).item()
print(f"\nPCC (cosine similarity): {pcc:.6f}")
print(f"Ref output range: [{ref_vec.min():.4f}, {ref_vec.max():.4f}]")
print(f"TT output range: [{tt_result.min():.4f}, {tt_result.max():.4f}]")

ttnn.close_device(device)
