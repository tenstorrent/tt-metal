"""Deep diagnostic: compare TT model layer-by-layer with pure CPU reference."""
import torch
import sys
import os
import glob

SNAP = "/home/yito/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"

from safetensors.torch import load_file

# Load only first 2 layers
state_dict = {}
for f in sorted(glob.glob(os.path.join(SNAP, "*.safetensors"))):
    shard = load_file(f)
    for k, t in shard.items():
        nk = k
        if k.startswith("model.language_model."):
            nk = "model." + k[len("model.language_model."):]
        if any(p in nk for p in ["embed_tokens", "lm_head", "model.norm", "layers.0."]):
            state_dict[nk] = t
print(f"Loaded {len(state_dict)} tensors")

# Manually trace through layer 0 (DeltaNet) on CPU
token_ids = torch.tensor([[151644]])  # Single token
embed_w = state_dict["model.embed_tokens.weight"]
hidden = embed_w[token_ids].unsqueeze(0).float()  # [1, 1, 1, 5120]
print(f"Embedding: shape={hidden.shape}, norm={hidden.norm():.4f}")

# RMSNorm
norm_w = state_dict["model.layers.0.input_layernorm.weight"].float()
var = hidden.pow(2).mean(-1, keepdim=True)
normed = (norm_w * (hidden * torch.rsqrt(var + 1e-6)))
print(f"After RMSNorm: shape={normed.shape}, norm={normed.norm():.4f}")

# DeltaNet QKV projection (CPU reference)
qkv_w = state_dict["model.layers.0.linear_attn.in_proj_qkv.weight"].float()
qkv = normed.reshape(1, -1) @ qkv_w.T  # [1, 10240]
print(f"QKV proj (CPU): shape={qkv.shape}, norm={qkv.norm():.4f}, max={qkv.abs().max():.4f}")

# Apply SiLU (no conv state for first token)
qkv_silu = torch.nn.functional.silu(qkv)
print(f"After SiLU: norm={qkv_silu.norm():.4f}")

# Split
key_dim = 128 * 16  # head_k_dim * num_k_heads = 2048
value_dim = 128 * 48  # head_v_dim * num_v_heads = 6144
q_cpu, k_cpu, v_cpu = torch.split(qkv_silu, [key_dim, key_dim, value_dim], dim=-1)
print(f"Q: {q_cpu.shape}, K: {k_cpu.shape}, V: {v_cpu.shape}")

# Expand K heads to V heads (3:1 ratio)
expand_ratio = 48 // 16
q_r = q_cpu.reshape(1, 16, 128).repeat_interleave(expand_ratio, dim=1)  # [1, 48, 128]
k_r = k_cpu.reshape(1, 16, 128).repeat_interleave(expand_ratio, dim=1)  # [1, 48, 128]
v_r = v_cpu.reshape(1, 48, 128)

# L2 normalize
def l2norm(x, dim=-1, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)

q_n = l2norm(q_r) * (128 ** -0.5)
k_n = l2norm(k_r)

# Compute b and a projections
b_w = state_dict["model.layers.0.linear_attn.in_proj_b.weight"].float()
a_w = state_dict["model.layers.0.linear_attn.in_proj_a.weight"].float()
b_proj = normed.reshape(1, -1) @ b_w.T  # [1, 48]
a_proj = normed.reshape(1, -1) @ a_w.T  # [1, 48]
beta = torch.sigmoid(b_proj.float())
A_log = state_dict["model.layers.0.linear_attn.A_log"].float()
dt_bias = state_dict["model.layers.0.linear_attn.dt_bias"].float()
g = -A_log.exp() * torch.nn.functional.softplus(a_proj.flatten().float() + dt_bias)
print(f"beta: mean={beta.mean():.4f}, range=[{beta.min():.4f}, {beta.max():.4f}]")
print(f"g: mean={g.mean():.4f}, range=[{g.min():.4f}, {g.max():.4f}]")
print(f"exp(g): mean={g.exp().mean():.4f}")

# DeltaNet state update
S = torch.zeros(48, 128, 128)
g_t = g.exp().unsqueeze(-1).unsqueeze(-1)  # [48, 1, 1]
beta_t = beta.flatten().unsqueeze(-1)  # [48, 1]
q_t = q_n[0]  # [48, 128]
k_t = k_n[0]  # [48, 128]
v_t = v_r[0]  # [48, 128]

S = S * g_t
kv_mem = (S * k_t.unsqueeze(-1)).sum(dim=-2)  # [48, 128]
delta = (v_t - kv_mem) * beta_t
S = S + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
output_t = (S * q_t.unsqueeze(-1)).sum(dim=-2)  # [48, 128]
print(f"DeltaNet output: shape={output_t.shape}, norm={output_t.norm():.4f}")

# Z projection and gating
z_w = state_dict["model.layers.0.linear_attn.in_proj_z.weight"].float()
z = normed.reshape(1, -1) @ z_w.T  # [1, 6144]
z = z.reshape(48, 128)
norm_weight = state_dict["model.layers.0.linear_attn.norm.weight"].float()
variance = output_t.pow(2).mean(-1, keepdim=True)
out_normed = output_t * torch.rsqrt(variance + 1e-6)
out_normed = norm_weight * out_normed
out_gated = out_normed * torch.nn.functional.silu(z)
print(f"Gated output: shape={out_gated.shape}, norm={out_gated.norm():.4f}")

# Out projection
out_w = state_dict["model.layers.0.linear_attn.out_proj.weight"].float()
out_flat = out_gated.reshape(1, -1) @ out_w.T  # [1, 5120]
print(f"After out_proj (CPU ref): norm={out_flat.norm():.4f}")
print(f"  First 5 values: {out_flat[0, :5]}")

# Now compare with TT model
import ttnn
device = ttnn.open_device(device_id=0)
try:
    from models.demos.qwen36_27b.tt.model_config import Qwen36ModelConfig
    from models.demos.qwen36_27b.tt.deltanet import TtGatedDeltaNet, TtDeltaNetState
    from models.demos.qwen36_27b.tt.decoder import SimpleRMSNorm
    
    config = Qwen36ModelConfig()
    config.num_hidden_layers = 1  # Only 1 layer for testing
    
    # Build the DeltaNet layer
    deltanet = TtGatedDeltaNet(device, state_dict, 0, config, dtype=ttnn.bfloat16)
    ds = TtDeltaNetState(1, ["linear_attention"], device, config)
    
    # Feed the same normed hidden state
    normed_tt = ttnn.from_torch(
        normed.to(torch.bfloat16).reshape(1, 1, 1, -1),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
    )
    
    out_tt = deltanet._decode_step(normed_tt, ds)
    out_tt_cpu = ttnn.to_torch(out_tt).flatten()[:5120]
    
    print(f"\nTT DeltaNet output: norm={out_tt_cpu.float().norm():.4f}")
    print(f"  First 5 values: {out_tt_cpu[:5]}")
    print(f"  Max abs error vs CPU: {(out_tt_cpu.float() - out_flat.flatten()[:5120]).abs().max():.6f}")
    print(f"  Mean abs error vs CPU: {(out_tt_cpu.float() - out_flat.flatten()[:5120]).abs().mean():.6f}")
    
finally:
    ttnn.close_device(device)
