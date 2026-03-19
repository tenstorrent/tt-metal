# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Test DeltaNet PCC degradation over multiple sequential steps."""
import os

import torch
import torch.nn.functional as F

os.environ["HF_MODEL"] = os.environ["HF_MODEL"]

import json

from safetensors.torch import load_file

CKPT_DIR = os.environ["HF_MODEL"]
LAYER = 0
NUM_STEPS = 20

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


# Copy reference functions
def l2norm(x, dim=-1, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def torch_causal_conv1d_update(hidden_states, conv_state, weight, bias=None, activation=None):
    _, hidden_size, seq_len = hidden_states.shape
    state_len = conv_state.shape[-1]
    hidden_states_new = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
    conv_state.copy_(hidden_states_new[:, :, -state_len:])
    out = F.conv1d(hidden_states_new, weight.unsqueeze(1), bias, padding=0, groups=hidden_size)
    out = F.silu(out[:, :, -seq_len:])
    return out.to(hidden_states.dtype)


class Qwen3_5RMSNormGated(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states, gate=None):
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states
        if gate is not None:
            hidden_states = hidden_states * F.silu(gate.to(torch.float32))
        return hidden_states


# Load weights
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
            weights[k[len(prefix) :]] = v

# Prepare reference model components
w_qkv = weights["in_proj_qkv.weight"].float()
w_z = weights["in_proj_z.weight"].float()
w_b = weights["in_proj_b.weight"].float()
w_a = weights["in_proj_a.weight"].float()
w_out = weights["out_proj.weight"].float()
conv_w = weights["conv1d.weight"].squeeze(1).float()
dt_bias = weights["dt_bias"].float()
A_log = weights["A_log"].float()
norm_layer = Qwen3_5RMSNormGated(head_v_dim, eps=rms_eps)
norm_layer.weight.data = weights["norm.weight"].float()

# Reference recurrence state
ref_state = torch.zeros(1, num_v_heads, head_k_dim, head_v_dim)
ref_conv_state = torch.zeros(1, conv_dim, conv_kernel - 1)

# TT model
import ttnn
from models.tt_transformers.tt.gated_deltanet import GatedDeltaNet
from models.tt_transformers.tt.model_config import ModelArgs

device = ttnn.open_device(device_id=0)
args = ModelArgs(device, max_seq_len=256)
sd = args.load_state_dict()
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

B_pad = args.tile_padded_batch_rows
torch.manual_seed(42)

print(f"Running {NUM_STEPS} sequential steps for DeltaNet layer {LAYER}...")
for step in range(NUM_STEPS):
    x = torch.randn(1, 1, hidden_size)

    # Reference forward
    mixed_qkv = x @ w_qkv.T
    z = x @ w_z.T
    b = x @ w_b.T
    a = x @ w_a.T

    mixed_qkv_t = mixed_qkv.transpose(1, 2)
    mixed_qkv_conv = torch_causal_conv1d_update(mixed_qkv_t, ref_conv_state, conv_w, activation="silu")
    mixed_qkv_conv = mixed_qkv_conv.transpose(1, 2)

    q, k, v = torch.split(mixed_qkv_conv, [key_dim, key_dim, value_dim], dim=-1)
    q = q.reshape(1, 1, num_k_heads, head_k_dim)
    k = k.reshape(1, 1, num_k_heads, head_k_dim)
    v = v.reshape(1, 1, num_v_heads, head_v_dim)
    beta = b.sigmoid()
    g = -A_log.exp() * F.softplus(a.float() + dt_bias)
    if gqa_ratio > 1:
        q = q.repeat_interleave(gqa_ratio, dim=2)
        k = k.repeat_interleave(gqa_ratio, dim=2)

    # Reference recurrence
    q_n = l2norm(q)
    k_n = l2norm(k)
    scale = 1 / (head_k_dim**0.5)
    q_n = q_n * scale
    q_n, k_n, v_t, beta_t, g_t = [x.transpose(1, 2).contiguous().to(torch.float32) for x in (q_n, k_n, v, beta, g)]
    g_exp = g_t[:, :, 0].exp().unsqueeze(-1).unsqueeze(-1)
    beta_s = beta_t[:, :, 0].unsqueeze(-1)
    ref_state = ref_state * g_exp
    kv_mem = (ref_state * k_n[:, :, 0].unsqueeze(-1)).sum(dim=-2)
    delta = (v_t[:, :, 0] - kv_mem) * beta_s
    ref_state = ref_state + k_n[:, :, 0].unsqueeze(-1) * delta.unsqueeze(-2)
    ref_out_heads = (ref_state * q_n[:, :, 0].unsqueeze(-1)).sum(dim=-2)

    ref_out_flat = ref_out_heads.reshape(-1, head_v_dim).float()
    z_flat = z.reshape(1, 1, num_v_heads, head_v_dim).reshape(-1, head_v_dim).float()
    ref_normed = norm_layer(ref_out_flat, z_flat)
    ref_output = (ref_normed.reshape(1, 1, -1) @ w_out.T)[0, 0]

    # TT forward
    x_pad = torch.zeros(1, 1, B_pad, hidden_size)
    x_pad[0, 0, 0, :] = x[0, 0, :]
    x_tt = ttnn.from_torch(x_pad, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_out = deltanet.forward(x_tt)
    tt_result = ttnn.to_torch(tt_out).float()[0, 0, 0, :hidden_size]

    pcc = F.cosine_similarity(ref_output.unsqueeze(0), tt_result.unsqueeze(0)).item()
    print(
        f"  Step {step:2d}: PCC={pcc:.6f}  ref_norm={ref_output.norm():.3f}  tt_norm={tt_result.norm():.3f}  state_norm={ref_state.norm():.3f}"
    )

ttnn.close_device(device)
