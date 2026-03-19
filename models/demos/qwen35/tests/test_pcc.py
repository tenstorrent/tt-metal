# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
PCC validation tests for Qwen3.5-27B DeltaNet and GatedAttention layers.

Compares single-layer TTNN output against PyTorch reference (HuggingFace model)
to verify numerical correctness. Each test:
  1. Loads raw safetensors weights for one layer
  2. Runs the HF reference forward on host
  3. Runs the TTNN implementation on device
  4. Compares outputs via Pearson Correlation Coefficient (PCC)
"""

import math
import os

import pytest
import torch
import torch.nn.functional as F

import ttnn

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_PATH = os.environ.get("HF_MODEL", "")
if not MODEL_PATH or not os.path.isdir(MODEL_PATH):
    from huggingface_hub import snapshot_download

    MODEL_PATH = snapshot_download(os.environ.get("HF_MODEL", "Qwen/Qwen3.5-27B"))
# Layer 0 is linear_attention (DeltaNet), layer 3 is full_attention (GatedAttention)
DELTANET_LAYER = 0
ATTENTION_LAYER = 3
PCC_THRESHOLD = 0.99

# Qwen3.5-27B architecture constants
HIDDEN_SIZE = 5120
NUM_V_HEADS = 48
NUM_K_HEADS = 16
HEAD_K_DIM = 128
HEAD_V_DIM = 128
KEY_DIM = HEAD_K_DIM * NUM_K_HEADS  # 2048
VALUE_DIM = HEAD_V_DIM * NUM_V_HEADS  # 6144
CONV_DIM = KEY_DIM * 2 + VALUE_DIM  # 10240
CONV_KERNEL_SIZE = 4
GQA_RATIO = NUM_V_HEADS // NUM_K_HEADS  # 3

# Attention layer constants
N_HEADS = 24
N_KV_HEADS = 4
HEAD_DIM = 256
PARTIAL_ROTARY_FACTOR = 0.25
ROTARY_DIM = int(HEAD_DIM * PARTIAL_ROTARY_FACTOR)  # 64
ROPE_THETA = 1000000.0
NORM_EPS = 1e-6


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def compute_pcc(x: torch.Tensor, y: torch.Tensor) -> float:
    """Pearson Correlation Coefficient between two tensors."""
    x_flat = x.flatten().float()
    y_flat = y.flatten().float()
    x_c = x_flat - x_flat.mean()
    y_c = y_flat - y_flat.mean()
    num = (x_c * y_c).sum()
    den = torch.sqrt((x_c**2).sum() * (y_c**2).sum())
    return (num / den).item() if den > 0 else 0.0


def load_layer_weights(layer_idx):
    """Load safetensors weights for a single layer using HF index."""
    import json

    from safetensors.torch import load_file, safe_open

    index_path = os.path.join(MODEL_PATH, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        # Single file fallback
        return load_file(os.path.join(MODEL_PATH, "model.safetensors"))

    with open(index_path) as f:
        index = json.load(f)

    # Find which shard files contain weights for this layer
    prefix = f"model.language_model.layers.{layer_idx}."
    needed_files = set()
    for key, shard_file in index["weight_map"].items():
        if key.startswith(prefix):
            needed_files.add(shard_file)

    weights = {}
    for shard_file in needed_files:
        shard_path = os.path.join(MODEL_PATH, shard_file)
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.startswith(prefix):
                    weights[key] = f.get_tensor(key)

    return weights


# ---------------------------------------------------------------------------
# Reference implementations (pure PyTorch, matching HF exactly)
# ---------------------------------------------------------------------------
def l2norm(x, dim=-1, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def ref_deltanet_forward(weights, hidden_states):
    """
    Single-token DeltaNet forward (decode mode) using PyTorch reference.

    Follows torch_recurrent_gated_delta_rule from HF modeling_qwen3_5.py,
    adapted for single-token (seq_len=1) with zero initial state.

    Args:
        weights: dict of raw HF weights for one DeltaNet layer
        hidden_states: (1, 1, hidden_size) float32 input

    Returns:
        output: (1, 1, hidden_size) float32
    """
    p = "model.language_model.layers.0."  # Will be replaced by caller with correct prefix

    # Projections
    in_proj_qkv_w = weights[f"{p}linear_attn.in_proj_qkv.weight"]
    in_proj_z_w = weights[f"{p}linear_attn.in_proj_z.weight"]
    in_proj_b_w = weights[f"{p}linear_attn.in_proj_b.weight"]
    in_proj_a_w = weights[f"{p}linear_attn.in_proj_a.weight"]
    out_proj_w = weights[f"{p}linear_attn.out_proj.weight"]
    conv1d_w = weights[f"{p}linear_attn.conv1d.weight"]  # (conv_dim, 1, kernel)
    dt_bias = weights[f"{p}linear_attn.dt_bias"]
    A_log = weights[f"{p}linear_attn.A_log"]
    norm_w = weights[f"{p}linear_attn.norm.weight"]

    batch_size, seq_len, _ = hidden_states.shape

    # Linear projections
    mixed_qkv = F.linear(hidden_states, in_proj_qkv_w)  # (1, 1, conv_dim)
    z = F.linear(hidden_states, in_proj_z_w)  # (1, 1, value_dim)
    b = F.linear(hidden_states, in_proj_b_w)  # (1, 1, num_v_heads)
    a = F.linear(hidden_states, in_proj_a_w)  # (1, 1, num_v_heads)

    # Conv1d (single token, zero initial state = just weight[:, :, -1] * silu)
    # For decode with zero conv state, only the last conv weight column matters
    # conv_state starts as zeros, so: conv_out = zeros * w[:,0] + ... + mixed_qkv * w[:,-1]
    conv_w = conv1d_w.squeeze(1)  # (conv_dim, kernel_size)
    conv_out = mixed_qkv.transpose(1, 2) * conv_w[:, -1:].unsqueeze(0)  # (1, conv_dim, 1)
    mixed_qkv = F.silu(conv_out.transpose(1, 2))  # (1, 1, conv_dim)

    # Split QKV
    query, key, value = torch.split(mixed_qkv, [KEY_DIM, KEY_DIM, VALUE_DIM], dim=-1)
    query = query.reshape(batch_size, seq_len, NUM_K_HEADS, HEAD_K_DIM)
    key = key.reshape(batch_size, seq_len, NUM_K_HEADS, HEAD_K_DIM)
    value = value.reshape(batch_size, seq_len, NUM_V_HEADS, HEAD_V_DIM)

    # Gates
    beta = b.sigmoid()  # (1, 1, num_v_heads)
    g = -A_log.float().exp() * F.softplus(a.float() + dt_bias)  # (1, 1, num_v_heads)

    # GQA expansion
    query = query.repeat_interleave(GQA_RATIO, dim=2)
    key = key.repeat_interleave(GQA_RATIO, dim=2)

    # Use torch_recurrent_gated_delta_rule logic
    # Transpose to (batch, heads, seq, dim) for query/key/value
    # and to (batch, heads, seq) for beta/g
    query = query.transpose(1, 2).contiguous().float()  # (1, num_v_heads, 1, head_k_dim)
    key = key.transpose(1, 2).contiguous().float()
    value = value.transpose(1, 2).contiguous().float()
    beta = beta.transpose(1, 2).contiguous().float()  # (1, num_v_heads, 1)
    g = g.transpose(1, 2).contiguous().float()  # (1, num_v_heads, 1)

    # L2 normalize
    query = l2norm(query)
    key = l2norm(key)

    scale = 1.0 / math.sqrt(HEAD_K_DIM)
    query = query * scale

    # Single-step recurrence (seq_len=1, zero initial state)
    q_t = query[:, :, 0]  # (1, H, K)
    k_t = key[:, :, 0]
    v_t = value[:, :, 0]
    g_t = g[:, :, 0].exp().unsqueeze(-1).unsqueeze(-1)  # (1, H, 1, 1)
    beta_t = beta[:, :, 0].unsqueeze(-1)  # (1, H, 1)

    state = torch.zeros(1, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM)
    state = state * g_t
    kv_mem = (state * k_t.unsqueeze(-1)).sum(dim=-2)
    delta = (v_t - kv_mem) * beta_t
    state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
    output = (state * q_t.unsqueeze(-1)).sum(dim=-2)  # (1, H, V)

    # Back to (batch, seq, heads, head_v_dim)
    core_attn_out = output.unsqueeze(2).transpose(1, 2)  # (1, 1, H, V)

    # Gated RMSNorm
    z_reshaped = z.reshape(batch_size, seq_len, NUM_V_HEADS, HEAD_V_DIM)
    core_flat = core_attn_out.reshape(-1, HEAD_V_DIM).float()
    z_flat = z_reshaped.reshape(-1, HEAD_V_DIM).float()
    variance = core_flat.pow(2).mean(-1, keepdim=True)
    normed = core_flat * torch.rsqrt(variance + NORM_EPS)
    normed = norm_w * normed
    gated = normed * F.silu(z_flat)
    core_attn_out = gated.reshape(batch_size, seq_len, -1)

    # Output projection
    output = F.linear(core_attn_out, out_proj_w)
    return output


def ref_gated_attention_forward(weights, hidden_states, position=0, layer_idx=3):
    """
    Single-token GatedAttention forward using PyTorch reference.

    Follows HF Qwen3_5Attention.forward, adapted for single-token decode.

    Args:
        weights: dict of raw HF weights for one attention layer
        hidden_states: (1, 1, hidden_size) float32 input
        position: token position (int)
        layer_idx: layer index

    Returns:
        output: (1, 1, hidden_size) float32
    """
    p = f"model.language_model.layers.{layer_idx}."

    # Load weights
    # q_proj is 2x size: first half query, second half gate (interleaved per head)
    q_proj_w = weights[f"{p}self_attn.q_proj.weight"]  # (n_heads * head_dim * 2, hidden)
    k_proj_w = weights[f"{p}self_attn.k_proj.weight"]
    v_proj_w = weights[f"{p}self_attn.v_proj.weight"]
    o_proj_w = weights[f"{p}self_attn.o_proj.weight"]
    q_norm_w = weights[f"{p}self_attn.q_norm.weight"]
    k_norm_w = weights[f"{p}self_attn.k_norm.weight"]

    batch_size, seq_len, _ = hidden_states.shape

    # QKV projections
    qg = F.linear(hidden_states, q_proj_w)  # (1, 1, n_heads * head_dim * 2)
    k_out = F.linear(hidden_states, k_proj_w)  # (1, 1, n_kv_heads * head_dim)
    v_out = F.linear(hidden_states, v_proj_w)

    # Split Q and gate (interleaved per head)
    qg = qg.view(batch_size, seq_len, N_HEADS, HEAD_DIM * 2)
    query, gate = qg[..., :HEAD_DIM], qg[..., HEAD_DIM:]
    gate = gate.reshape(batch_size, seq_len, -1)  # (1, 1, n_heads * head_dim)

    # RMSNorm on Q and K (per head, zero-centered: weight is initialized to 0, norm = x * (1+w))
    def rms_norm(x, w, eps=1e-6):
        x_f = x.float()
        norm = x_f * torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + eps)
        return norm * (1.0 + w.float())

    query = rms_norm(query, q_norm_w).transpose(1, 2)  # (1, n_heads, 1, head_dim)
    key = rms_norm(k_out.view(batch_size, seq_len, N_KV_HEADS, HEAD_DIM), k_norm_w).transpose(1, 2)
    value = v_out.view(batch_size, seq_len, N_KV_HEADS, HEAD_DIM).transpose(1, 2)

    # Partial RoPE
    inv_freq = 1.0 / (ROPE_THETA ** (torch.arange(0, ROTARY_DIM, 2).float() / ROTARY_DIM))
    freqs = position * inv_freq
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos()
    sin = emb.sin()

    def apply_rope(t):
        t_rot, t_pass = t[..., :ROTARY_DIM], t[..., ROTARY_DIM:]
        t1 = t_rot[..., : ROTARY_DIM // 2]
        t2 = t_rot[..., ROTARY_DIM // 2 :]
        rotated = torch.cat([-t2, t1], dim=-1)
        return torch.cat([t_rot * cos + rotated * sin, t_pass], dim=-1)

    query = apply_rope(query.float())
    key = apply_rope(key.float())

    # GQA expansion
    gqa = N_HEADS // N_KV_HEADS
    key = key.repeat_interleave(gqa, dim=1)
    value = value.float().repeat_interleave(gqa, dim=1)

    # SDPA (single token, no mask needed)
    scale = 1.0 / math.sqrt(HEAD_DIM)
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale
    attn_weights = F.softmax(scores, dim=-1)
    attn_output = torch.matmul(attn_weights, value)

    # Concat heads
    attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, -1)

    # Gate
    attn_output = attn_output * torch.sigmoid(gate.float())

    # Output projection
    output = F.linear(attn_output, o_proj_w)
    return output


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def device():
    dev = ttnn.open_device(device_id=0)
    yield dev
    ttnn.close_device(dev)


class TestDeltaNetPCC:
    """PCC validation for a single DeltaNet (linear attention) layer."""

    @pytest.fixture(scope="class")
    def layer_weights(self):
        return load_layer_weights(DELTANET_LAYER)

    def test_deltanet_single_token(self, device, layer_weights):
        """DeltaNet single-token decode PCC against PyTorch reference."""
        torch.manual_seed(42)
        x_host = torch.randn(1, 1, HIDDEN_SIZE)

        # --- Reference (PyTorch) ---
        ref_out = ref_deltanet_forward(layer_weights, x_host)

        # --- TTNN ---
        # Prepare weights using the same key mapping as ModelArgs.load_state_dict
        from models.tt_transformers.tt.load_checkpoints import convert_hf_to_meta_qwen35

        # Build a minimal state_dict with just this layer's weights
        # Strip "model." prefix to match what convert_hf_to_meta_qwen35 expects
        sd = {k.replace("model.", ""): v for k, v in layer_weights.items()}
        sd = convert_hf_to_meta_qwen35(sd, HEAD_DIM, N_HEADS, N_KV_HEADS)

        # Create a minimal args-like object for GatedDeltaNet
        class MinimalArgs:
            dim = HIDDEN_SIZE
            linear_num_value_heads = NUM_V_HEADS
            linear_num_key_heads = NUM_K_HEADS
            linear_key_head_dim = HEAD_K_DIM
            linear_value_head_dim = HEAD_V_DIM
            linear_conv_kernel_dim = CONV_KERNEL_SIZE
            norm_eps = NORM_EPS
            tile_padded_batch_rows = 32
            dummy_weights = False

            @staticmethod
            def get_state_dict_prefix(module_name, layer_num):
                module_map = {
                    "GatedDeltaNet": "linear_attn",
                    "GatedAttention": "attention",
                }
                return f"layers.{layer_num}.{module_map[module_name]}"

        from models.tt_transformers.tt.gated_deltanet import GatedDeltaNet

        deltanet = GatedDeltaNet(
            mesh_device=device,
            args=MinimalArgs(),
            state_dict=sd,
            weight_cache_path=None,
            layer_num=DELTANET_LAYER,
            dtype=ttnn.bfloat16,
        )
        deltanet.initialize_states()

        # Prepare input: (1, 1, 32, hidden_size) with padding
        B_pad = 32
        x_padded = torch.zeros(1, 1, B_pad, HIDDEN_SIZE)
        x_padded[0, 0, 0, :] = x_host[0, 0, :]
        x_tt = ttnn.from_torch(
            x_padded.bfloat16(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

        # Forward
        out_tt = deltanet.forward(x_tt)
        out_host = ttnn.to_torch(out_tt).float()[0, 0, 0:1, :HIDDEN_SIZE]  # batch 0 only

        # Compare
        ref_flat = ref_out[0, 0, :].float()
        tt_flat = out_host[0, :].float()
        pcc = compute_pcc(ref_flat, tt_flat)
        print(f"DeltaNet PCC: {pcc:.6f}")
        assert pcc >= PCC_THRESHOLD, f"DeltaNet PCC {pcc:.6f} < {PCC_THRESHOLD}"


class TestGatedAttentionPCC:
    """PCC validation for a single GatedAttention (full attention) layer."""

    @pytest.fixture(scope="class")
    def layer_weights(self):
        return load_layer_weights(ATTENTION_LAYER)

    def test_gated_attention_single_token(self, device, layer_weights):
        """GatedAttention single-token decode PCC against PyTorch reference."""
        torch.manual_seed(42)
        x_host = torch.randn(1, 1, HIDDEN_SIZE)
        position = 0

        # --- Reference (PyTorch) ---
        ref_out = ref_gated_attention_forward(layer_weights, x_host, position=position, layer_idx=ATTENTION_LAYER)

        # --- TTNN ---
        from models.tt_transformers.tt.load_checkpoints import convert_hf_to_meta_qwen35

        sd = {k.replace("model.", ""): v for k, v in layer_weights.items()}
        sd = convert_hf_to_meta_qwen35(sd, HEAD_DIM, N_HEADS, N_KV_HEADS)

        # Create a minimal args-like object for GatedAttention
        class MinimalArgs:
            dim = HIDDEN_SIZE
            n_heads = N_HEADS
            n_kv_heads = N_KV_HEADS
            head_dim = HEAD_DIM
            partial_rotary_factor = PARTIAL_ROTARY_FACTOR
            rope_theta = ROPE_THETA
            max_seq_len = 256
            norm_eps = NORM_EPS
            tile_padded_batch_rows = 32
            dummy_weights = False

            @staticmethod
            def get_state_dict_prefix(module_name, layer_num):
                module_map = {
                    "GatedDeltaNet": "linear_attn",
                    "GatedAttention": "attention",
                }
                return f"layers.{layer_num}.{module_map[module_name]}"

        from models.tt_transformers.tt.gated_attention import GatedAttention

        gated_attn = GatedAttention(
            mesh_device=device,
            tt_ccl=None,
            args=MinimalArgs(),
            state_dict=sd,
            weight_cache_path=None,
            layer_num=ATTENTION_LAYER,
            dtype=ttnn.bfloat16,
            transformation_mats=None,
            configuration=MinimalArgs(),
        )

        # Prepare input: (1, 1, 32, hidden_size) with padding
        B_pad = 32
        x_padded = torch.zeros(1, 1, B_pad, HIDDEN_SIZE)
        x_padded[0, 0, 0, :] = x_host[0, 0, :]
        x_tt = ttnn.from_torch(
            x_padded.bfloat16(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

        # Forward
        tt_pos = ttnn.from_torch(
            torch.tensor([position], dtype=torch.int32),
            dtype=ttnn.int32,
            device=device,
        )
        out_tt = gated_attn.forward(x_tt, current_pos=tt_pos)
        out_host = ttnn.to_torch(out_tt).float()[0, 0, 0:1, :HIDDEN_SIZE]

        # Compare
        ref_flat = ref_out[0, 0, :].float()
        tt_flat = out_host[0, :].float()
        pcc = compute_pcc(ref_flat, tt_flat)
        print(f"GatedAttention PCC: {pcc:.6f}")
        assert pcc >= PCC_THRESHOLD, f"GatedAttention PCC {pcc:.6f} < {PCC_THRESHOLD}"
