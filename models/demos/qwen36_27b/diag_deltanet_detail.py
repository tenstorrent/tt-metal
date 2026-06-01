"""
Detailed DeltaNet layer comparison: HF reference vs TT implementation.
Step-by-step comparison of intermediate values to find numerical divergence.
"""
import torch
import torch.nn.functional as F
import os

SNAP = "/home/yito/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"

# ---- Step 0: Load HF model with 4 layers (just to get weights) ----
from transformers import AutoConfig
from safetensors.torch import load_file
from pathlib import Path

hf_config = AutoConfig.from_pretrained(SNAP, trust_remote_code=True)

# Load raw weights for layer 0 (a DeltaNet layer)
shard_files = sorted(Path(SNAP).glob("*.safetensors"))
raw_dict = {}
for sf in shard_files:
    shard = load_file(str(sf))
    for k, v in shard.items():
        if "layers.0." in k or "embed_tokens" in k:
            raw_dict[k] = v

def normalize_key(k):
    if k.startswith("model.language_model."):
        return "model." + k[len("model.language_model."):]
    return k

state_dict = {normalize_key(k): v for k, v in raw_dict.items()}
prefix = "model.layers.0.linear_attn"
print("=== Layer 0 DeltaNet weight shapes ===")
for k, v in sorted(state_dict.items()):
    if prefix in k:
        print(f"  {k}: {v.shape} {v.dtype}")

# ---- Step 1: HF-style forward (pure PyTorch, recurrent path) ----
hidden_size = 5120
num_k_heads = 16
num_v_heads = 48
head_k_dim = 128
head_v_dim = 128
conv_kernel = 4
key_dim = num_k_heads * head_k_dim  # 2048
value_dim = num_v_heads * head_v_dim  # 6144
conv_dim = key_dim * 2 + value_dim  # 10240

# Load weights
in_proj_qkv_w = state_dict[f"{prefix}.in_proj_qkv.weight"]  # [conv_dim, hidden_size]
in_proj_z_w = state_dict[f"{prefix}.in_proj_z.weight"]       # [value_dim, hidden_size]
in_proj_b_w = state_dict[f"{prefix}.in_proj_b.weight"]       # [num_v_heads, hidden_size]
in_proj_a_w = state_dict[f"{prefix}.in_proj_a.weight"]       # [num_v_heads, hidden_size]
out_proj_w = state_dict[f"{prefix}.out_proj.weight"]          # [hidden_size, value_dim]
norm_w = state_dict[f"{prefix}.norm.weight"]                  # [head_v_dim]
A_log = state_dict[f"{prefix}.A_log"]                         # [num_v_heads]
dt_bias = state_dict[f"{prefix}.dt_bias"]                     # [num_v_heads]
conv1d_w = state_dict[f"{prefix}.conv1d.weight"]              # [conv_dim, 1, conv_kernel]

# Get a test input: embedding of token 151644
embed_w = state_dict["model.embed_tokens.weight"]
x = embed_w[151644].unsqueeze(0).unsqueeze(0)  # [1, 1, hidden_size]
print(f"\nInput embedding: shape={x.shape}, norm={x.norm():.6f}")

# -- HF reference path (single token, no cache) --
print("\n=== HF Reference (recurrent, float32) ===")
x_f = x.float()

# Projections (nn.Linear: output = input @ W^T)
qkv_proj = (x_f @ in_proj_qkv_w.float().T)  # [1, 1, conv_dim]
z_proj = (x_f @ in_proj_z_w.float().T)       # [1, 1, value_dim]
b_proj = (x_f @ in_proj_b_w.float().T)       # [1, 1, num_v_heads]
a_proj = (x_f @ in_proj_a_w.float().T)       # [1, 1, num_v_heads]

print(f"qkv_proj norm: {qkv_proj.norm():.6f}, first5: {qkv_proj[0,0,:5]}")
print(f"z_proj norm: {z_proj.norm():.6f}")
print(f"b_proj: {b_proj[0,0,:5]}")
print(f"a_proj: {a_proj[0,0,:5]}")

# Conv1d (single token, no cache = zero padding)
# HF: conv1d with padding=3 on a single-token input
qkv_transposed = qkv_proj.transpose(1, 2)  # [1, conv_dim, 1]
# Depthwise conv1d: weight [conv_dim, 1, kernel_size], groups=conv_dim
conv_out = F.conv1d(qkv_transposed, conv1d_w.float(), bias=None,
                    padding=conv_kernel - 1, groups=conv_dim)
conv_out = conv_out[:, :, :1]  # keep only first output
conv_out = F.silu(conv_out)  # [1, conv_dim, 1]
qkv_conv_hf = conv_out.transpose(1, 2)  # [1, 1, conv_dim]

print(f"\nConv1d output (HF): norm={qkv_conv_hf.norm():.6f}, first5: {qkv_conv_hf[0,0,:5]}")

# Manual conv (simulating our TT decode path)
# Zero conv state, insert current input at position -1
conv_state = torch.zeros(1, conv_dim, conv_kernel)  # [1, conv_dim, 4]
conv_state[:, :, -1] = qkv_proj[0, 0]  # [0, 0, 0, x]
conv1d_w_sq = conv1d_w.squeeze(1)  # [conv_dim, conv_kernel]
qkv_conv_manual = (conv_state[0] * conv1d_w_sq.float()).sum(dim=-1)  # [conv_dim]
qkv_conv_manual = F.silu(qkv_conv_manual)  # [conv_dim]

print(f"Conv1d output (manual): norm={qkv_conv_manual.norm():.6f}, first5: {qkv_conv_manual[:5]}")

cos_sim_conv = F.cosine_similarity(qkv_conv_hf.flatten().unsqueeze(0), qkv_conv_manual.flatten().unsqueeze(0))
print(f"Conv cosine sim (HF vs manual): {cos_sim_conv.item():.6f}")

# Split into q, k, v
q_hf, k_hf, v_hf = torch.split(qkv_conv_hf[0, 0], [key_dim, key_dim, value_dim])
q_man, k_man, v_man = torch.split(qkv_conv_manual, [key_dim, key_dim, value_dim])

q_hf = q_hf.reshape(num_k_heads, head_k_dim)
k_hf = k_hf.reshape(num_k_heads, head_k_dim)
v_hf = v_hf.reshape(num_v_heads, head_v_dim)

# Expand heads (k_heads=16 → v_heads=48, ratio=3)
expand_ratio = num_v_heads // num_k_heads  # 3
q_hf_exp = q_hf.repeat_interleave(expand_ratio, dim=0)  # [48, 128]
k_hf_exp = k_hf.repeat_interleave(expand_ratio, dim=0)  # [48, 128]

# L2 norm
def l2norm(x, dim=-1, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)

q_hf_norm = l2norm(q_hf_exp)
k_hf_norm = l2norm(k_hf_exp)

scale = head_k_dim ** -0.5
q_hf_scaled = q_hf_norm * scale

# Gates
beta_hf = b_proj[0, 0].float().sigmoid()  # [num_v_heads]
g_hf = -A_log.float().exp() * F.softplus(a_proj[0, 0].float() + dt_bias.float())  # [num_v_heads]

print(f"\nbeta_hf: min={beta_hf.min():.4f}, max={beta_hf.max():.4f}, mean={beta_hf.mean():.4f}")
print(f"g_hf: min={g_hf.min():.4f}, max={g_hf.max():.4f}, mean={g_hf.mean():.4f}")
print(f"g_hf.exp(): min={g_hf.exp().min():.6f}, max={g_hf.exp().max():.6f}")

# Recurrent step from zero state
S = torch.zeros(num_v_heads, head_k_dim, head_v_dim)
g_t = g_hf.exp().unsqueeze(-1).unsqueeze(-1)  # [H, 1, 1]
beta_t = beta_hf.unsqueeze(-1)  # [H, 1]

S = S * g_t  # zeros
kv_mem = (S * k_hf_norm.unsqueeze(-1)).sum(dim=-2)  # zeros
delta = (v_hf - kv_mem) * beta_t  # v * beta
S = S + k_hf_norm.unsqueeze(-1) * delta.unsqueeze(-2)  # outer(k, v*beta)
output_hf = (S * q_hf_scaled.unsqueeze(-1)).sum(dim=-2)  # [H, Dv]

print(f"\nRecurrent output (HF): shape={output_hf.shape}, norm={output_hf.norm():.6f}")
print(f"  per-head norms: {[f'{output_hf[h].norm():.4f}' for h in range(min(6, num_v_heads))]}")

# Apply gated norm + output projection
z_reshaped = z_proj[0, 0].float().reshape(num_v_heads, head_v_dim)
variance = output_hf.pow(2).mean(-1, keepdim=True)
out_normed = output_hf * torch.rsqrt(variance + 1e-6)
out_normed = norm_w.float() * out_normed
out_gated = out_normed * F.silu(z_reshaped)

out_flat = out_gated.reshape(1, 1, -1)
deltanet_output_hf = (out_flat @ out_proj_w.float().T)  # [1, 1, hidden_size]
print(f"DeltaNet layer output (HF): norm={deltanet_output_hf.norm():.6f}")

# Residual
layer_output_hf = x_f + deltanet_output_hf
print(f"After residual (HF): norm={layer_output_hf.norm():.6f}")


# ---- Step 2: TT path ----
print("\n=== TT Implementation ===")
import ttnn

device = ttnn.open_device(device_id=0)
try:
    from models.demos.qwen36_27b.tt.model_config import Qwen36ModelConfig
    from models.demos.qwen36_27b.tt.load_weights import load_state_dict
    from models.demos.qwen36_27b.tt.deltanet import TtGatedDeltaNet, TtDeltaNetState

    tt_config = Qwen36ModelConfig()
    tt_config.num_hidden_layers = 4
    tt_sd = load_state_dict(tt_config, max_layers=4, model_path=SNAP)

    tt_layer = TtGatedDeltaNet(device, tt_sd, layer_idx=0, config=tt_config)
    state = TtDeltaNetState(4, tt_config.layer_types, device, tt_config)

    # Same input embedding
    x_tt = ttnn.from_torch(x.unsqueeze(0).to(torch.bfloat16),  # [1, 1, 1, H]
                           dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Run projections
    qkv_tt = ttnn.linear(x_tt, tt_layer.in_proj_qkv_w)
    z_tt = ttnn.linear(x_tt, tt_layer.in_proj_z_w)
    b_tt = ttnn.linear(x_tt, tt_layer.in_proj_b_w)
    a_tt = ttnn.linear(x_tt, tt_layer.in_proj_a_w)

    qkv_tt_cpu = ttnn.to_torch(qkv_tt).flatten()
    z_tt_cpu = ttnn.to_torch(z_tt).flatten()
    b_tt_cpu = ttnn.to_torch(b_tt).flatten()[:num_v_heads]
    a_tt_cpu = ttnn.to_torch(a_tt).flatten()[:num_v_heads]

    print(f"qkv_proj (TT) norm: {qkv_tt_cpu[:conv_dim].norm():.6f}, first5: {qkv_tt_cpu[:5]}")
    proj_cos = F.cosine_similarity(qkv_proj.flatten().unsqueeze(0), qkv_tt_cpu[:conv_dim].float().unsqueeze(0))
    print(f"qkv_proj cosine sim (HF vs TT): {proj_cos.item():.6f}")

    # Conv1d step (TT path)
    conv_state_cpu = state.get_conv_state(0).squeeze(0)  # [conv_dim, 4]
    print(f"\nConv state initial: all_zero={conv_state_cpu.abs().sum() == 0}")
    conv_state_cpu = torch.roll(conv_state_cpu, shifts=-1, dims=-1)
    conv_state_cpu[:, -1] = qkv_tt_cpu[:conv_dim]
    qkv_conv_tt = (conv_state_cpu * tt_layer.conv1d_weight.float()).sum(dim=-1)
    qkv_conv_tt = F.silu(qkv_conv_tt)
    print(f"Conv1d output (TT): norm={qkv_conv_tt.norm():.6f}, first5: {qkv_conv_tt[:5]}")
    conv_cos = F.cosine_similarity(qkv_conv_hf.flatten().unsqueeze(0), qkv_conv_tt.float().flatten().unsqueeze(0))
    print(f"Conv cosine sim (HF vs TT): {conv_cos.item():.6f}")

    # Q/K/V split and expand
    q_tt, k_tt, v_tt = torch.split(qkv_conv_tt, [key_dim, key_dim, value_dim])
    q_tt = q_tt.reshape(num_k_heads, head_k_dim)
    k_tt = k_tt.reshape(num_k_heads, head_k_dim)
    v_tt = v_tt.reshape(num_v_heads, head_v_dim)
    q_tt = q_tt.repeat_interleave(expand_ratio, dim=0)
    k_tt = k_tt.repeat_interleave(expand_ratio, dim=0)

    q_tt = l2norm(q_tt.float()) * scale
    k_tt = l2norm(k_tt.float())
    v_tt = v_tt.float()

    # Gates
    beta_tt = b_tt_cpu.float().sigmoid()
    A_log_tt = ttnn.to_torch(tt_layer.A_log).flatten()[:num_v_heads]
    dt_bias_tt = ttnn.to_torch(tt_layer.dt_bias).flatten()[:num_v_heads]
    g_tt = -A_log_tt.float().exp() * F.softplus(a_tt_cpu.float() + dt_bias_tt.float())

    print(f"\nbeta_tt: min={beta_tt.min():.4f}, max={beta_tt.max():.4f}")
    print(f"g_tt: min={g_tt.min():.4f}, max={g_tt.max():.4f}")
    gate_cos = F.cosine_similarity(g_hf.unsqueeze(0), g_tt.unsqueeze(0))
    print(f"Gate cosine sim: beta={F.cosine_similarity(beta_hf.unsqueeze(0), beta_tt.unsqueeze(0)).item():.6f}, g={gate_cos.item():.6f}")

    # Recurrent step
    S_tt = torch.zeros(num_v_heads, head_k_dim, head_v_dim)
    g_t_tt = g_tt.exp().unsqueeze(-1).unsqueeze(-1)
    beta_t_tt = beta_tt.unsqueeze(-1)

    S_tt = S_tt * g_t_tt
    kv_mem_tt = (S_tt * k_tt.unsqueeze(-1)).sum(dim=-2)
    delta_tt = (v_tt - kv_mem_tt) * beta_t_tt
    S_tt = S_tt + k_tt.unsqueeze(-1) * delta_tt.unsqueeze(-2)
    output_tt = (S_tt * q_tt.unsqueeze(-1)).sum(dim=-2)

    print(f"\nRecurrent output (TT): norm={output_tt.norm():.6f}")
    recur_cos = F.cosine_similarity(output_hf.flatten().unsqueeze(0), output_tt.flatten().unsqueeze(0))
    print(f"Recurrent output cosine sim: {recur_cos.item():.6f}")
    print(f"Recurrent output max diff: {(output_hf - output_tt).abs().max():.6e}")

    # Gated norm + output projection
    z_tt_reshaped = z_tt_cpu[:value_dim].float().reshape(num_v_heads, head_v_dim)
    var_tt = output_tt.pow(2).mean(-1, keepdim=True)
    out_normed_tt = output_tt * torch.rsqrt(var_tt + 1e-6)
    out_normed_tt = norm_w.float() * out_normed_tt
    out_gated_tt = out_normed_tt * F.silu(z_tt_reshaped)

    out_flat_tt = out_gated_tt.reshape(1, 1, -1)
    deltanet_output_tt = (out_flat_tt @ out_proj_w.float().T)
    print(f"DeltaNet layer output (TT): norm={deltanet_output_tt.norm():.6f}")
    layer_cos = F.cosine_similarity(deltanet_output_hf.flatten().unsqueeze(0), deltanet_output_tt.flatten().unsqueeze(0))
    print(f"Layer output cosine sim: {layer_cos.item():.6f}")

    # Now run via the actual TT decode path for comparison
    print("\n=== Running actual TT _decode_step ===")
    state2 = TtDeltaNetState(4, tt_config.layer_types, device, tt_config)
    x_tt2 = ttnn.from_torch(x.unsqueeze(0).to(torch.bfloat16),
                            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    actual_out = tt_layer._decode_step(x_tt2, state2)
    actual_out_cpu = ttnn.to_torch(actual_out).float().flatten()[:hidden_size]
    print(f"Actual TT decode output norm: {actual_out_cpu.norm():.6f}")

    # Compare actual TT output with our manual TT computation
    manual_tt_flat = deltanet_output_tt.flatten()[:hidden_size]
    actual_vs_manual = F.cosine_similarity(manual_tt_flat.unsqueeze(0), actual_out_cpu.unsqueeze(0))
    print(f"Actual TT vs manual TT cosine sim: {actual_vs_manual.item():.6f}")

    # Compare actual TT output with HF
    hf_flat = deltanet_output_hf.flatten()[:hidden_size]
    actual_vs_hf = F.cosine_similarity(hf_flat.unsqueeze(0), actual_out_cpu.unsqueeze(0))
    print(f"Actual TT vs HF cosine sim: {actual_vs_hf.item():.6f}")
    print(f"Actual TT vs HF max diff: {(hf_flat - actual_out_cpu).abs().max():.6f}")

finally:
    ttnn.close_device(device)
