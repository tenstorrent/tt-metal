# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Tier-1 PCC test for the GPT-OSS prefill GQA attention block vs a self-contained torch reference.

Block: QKV proj (+bias) -> head split (GQA 64 Q / 8 KV) -> full RoPE (YaRN) -> causal SDPA with
per-head attention sinks and optional sliding window -> o_proj (+bias). GPT-OSS-120B dims (from
configs/gpt-oss-120b/config.json): hidden 2880, head_dim 64, rope_theta 150000, YaRN factor 32,
orig_max_pos 4096, sliding_window 128, attention_bias true, no QK-norm.

The reference and the TT module are driven by IDENTICAL random weights. The TT module is fed the
Meta-RoPE-swizzled q/k projections via the production ``convert_hf_qkv_to_meta_format`` helper (and
Meta-format cos/sin), so the TT swizzled-space path matches the HF-convention torch reference. RoPE
cos/sin (with YaRN + mscale) are computed once and shared by both sides, so the test is
self-consistent regardless of the exact YaRN constants. Single card (1x1 mesh, TP=1, SP=1).

Run:
    pytest models/demos/gpt_oss_d_p/tests/unit/test_attention_vs_ref.py
(single Blackhole/Wormhole card; no HF checkpoint or network needed).
"""

import math

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.gpt_oss_d_p.tt.attention import Attention, AttentionConfig, ProgramConfig
from models.demos.gpt_oss_d_p.tt.config import MeshConfig
from models.tt_transformers.tt.common import get_rot_transformation_mat
from models.tt_transformers.tt.load_checkpoints import convert_hf_qkv_to_meta_format

# GPT-OSS-120B attention dims.
HIDDEN = 2880
NQ = 64
NKV = 8
HEAD_DIM = 64
ROPE_THETA = 150000.0
YARN_FACTOR = 32.0
YARN_BETA_FAST = 32.0
YARN_BETA_SLOW = 1.0
YARN_ORIG_MAX_POS = 4096
SLIDING_WINDOW = 128
EPS = 1e-5


# --------------------------------------------------------------------------------------
# RoPE (YaRN) — matches transformers _compute_yarn_parameters. Shared by ref + TT so the
# test is self-consistent (exact HF match is not required for a TT-vs-reference PCC).
# --------------------------------------------------------------------------------------
def _yarn_inv_freq(dim, base, factor, orig_max_pos, beta_fast, beta_slow):
    def find_correction_dim(num_rotations):
        return (dim * math.log(orig_max_pos / (num_rotations * 2 * math.pi))) / (2 * math.log(base))

    low = math.floor(find_correction_dim(beta_fast))
    high = math.ceil(find_correction_dim(beta_slow))
    low = max(low, 0)
    high = min(high, dim - 1)

    pos_freqs = base ** (torch.arange(0, dim, 2).float() / dim)
    inv_freq_extrapolation = 1.0 / pos_freqs
    inv_freq_interpolation = 1.0 / (factor * pos_freqs)

    # linear ramp over dim//2 frequency bands
    lo, hi = low, high
    if lo == hi:
        hi += 0.001
    ramp = (torch.arange(dim // 2).float() - lo) / (hi - lo)
    ramp = ramp.clamp(0, 1)
    inv_freq_extrapolation_factor = 1.0 - ramp

    inv_freq = (
        inv_freq_interpolation * (1.0 - inv_freq_extrapolation_factor)
        + inv_freq_extrapolation * inv_freq_extrapolation_factor
    )
    attention_factor = 0.1 * math.log(factor) + 1.0
    return inv_freq, attention_factor


def _build_cos_sin(seq_len):
    """Return (cos_hf, sin_hf) [S, head_dim] for the reference, and (cos_meta, sin_meta)
    [1,1,S,head_dim] for the TT module. mscale (attention_factor) is folded into both."""
    inv_freq, attn_factor = _yarn_inv_freq(
        HEAD_DIM, ROPE_THETA, YARN_FACTOR, YARN_ORIG_MAX_POS, YARN_BETA_FAST, YARN_BETA_SLOW
    )
    pos = torch.arange(seq_len).float()
    freqs = torch.outer(pos, inv_freq)  # [S, head_dim/2]
    cos_half = torch.cos(freqs) * attn_factor  # [S, head_dim/2]
    sin_half = torch.sin(freqs) * attn_factor

    # HF convention: concat the halves -> rotate_half over the full head.
    cos_hf = torch.cat([cos_half, cos_half], dim=-1)  # [S, head_dim]
    sin_hf = torch.cat([sin_half, sin_half], dim=-1)

    # Meta convention (ttnn rotary_embedding_llama + reverse_permute'd weights): interleave.
    cos_meta = torch.stack([cos_half, cos_half], dim=-1).flatten(-2)[None, None]  # [1,1,S,head_dim]
    sin_meta = torch.stack([sin_half, sin_half], dim=-1).flatten(-2)[None, None]
    return (cos_hf, sin_hf), (cos_meta, sin_meta)


def _rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def _rope_hf(t, cos, sin):
    # cos/sin: [S, head_dim]; t: [B, H, S, head_dim]
    return t * cos + _rotate_half(t) * sin


# --------------------------------------------------------------------------------------
# Torch reference: HF-convention GPT-OSS GQA attention with sinks + sliding window.
# --------------------------------------------------------------------------------------
def _torch_attention(x, w, cos_hf, sin_hf, sliding_window):
    B, S, _ = x.shape
    q = (x @ w["q"].t() + w["q_bias"]).view(B, S, NQ, HEAD_DIM).transpose(1, 2)  # [B, NQ, S, HD]
    k = (x @ w["k"].t() + w["k_bias"]).view(B, S, NKV, HEAD_DIM).transpose(1, 2)
    v = (x @ w["v"].t() + w["v_bias"]).view(B, S, NKV, HEAD_DIM).transpose(1, 2)

    q = _rope_hf(q, cos_hf, sin_hf)
    k = _rope_hf(k, cos_hf, sin_hf)

    rep = NQ // NKV
    k = k.repeat_interleave(rep, dim=1)  # [B, NQ, S, HD]
    v = v.repeat_interleave(rep, dim=1)

    scale = HEAD_DIM**-0.5
    scores = (q @ k.transpose(-1, -2)) * scale  # [B, NQ, S, S]

    # causal mask (+ sliding window: keys older than `sliding_window` are masked)
    mask = torch.triu(torch.full((S, S), float("-inf")), diagonal=1)
    if sliding_window is not None:
        mask += torch.tril(torch.full((S, S), float("-inf")), diagonal=-sliding_window)
    scores = scores + mask

    # attention sinks: append a per-head learned logit as an extra softmax column, then drop it.
    sinks = w["sinks"].view(1, NQ, 1, 1).expand(B, NQ, S, 1)  # [B, NQ, S, 1]
    combined = torch.cat([scores, sinks], dim=-1)  # [B, NQ, S, S+1]
    probs = torch.softmax(combined, dim=-1)[..., :S]  # drop the sink column

    out = probs @ v  # [B, NQ, S, HD]
    out = out.transpose(1, 2).reshape(B, S, NQ * HEAD_DIM)
    return out @ w["o"].t() + w["o_bias"]


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize(
    "layer_idx, layer_types, is_sliding",
    [
        (0, ["sliding_attention", "full_attention"], True),  # sliding-window layer
        (1, ["sliding_attention", "full_attention"], False),  # full-causal layer
    ],
    ids=["sliding_layer", "full_layer"],
)
@pytest.mark.parametrize("seq_len", [256], ids=["s256"])
def test_attention_prefill_vs_ref(mesh_device, layer_idx, layer_types, is_sliding, seq_len, reset_seeds):
    """GPT-OSS GQA prefill attention block vs a self-authored torch reference, random weights.

    (1,1) = single card, TP=1 / SP=1 (no CCL). Both a sliding-window layer (layer_idx 0) and a
    full-causal layer (layer_idx 1) are covered. PCC >= 0.99."""
    torch.manual_seed(0)
    x = torch.randn(1, seq_len, HIDDEN) * 0.1

    # Random projection weights/biases (HF [out, in] layout) + per-head sinks.
    w = {
        "q": torch.randn(NQ * HEAD_DIM, HIDDEN) * 0.02,
        "k": torch.randn(NKV * HEAD_DIM, HIDDEN) * 0.02,
        "v": torch.randn(NKV * HEAD_DIM, HIDDEN) * 0.02,
        "o": torch.randn(HIDDEN, NQ * HEAD_DIM) * 0.02,
        "q_bias": torch.randn(NQ * HEAD_DIM) * 0.02,
        "k_bias": torch.randn(NKV * HEAD_DIM) * 0.02,
        "v_bias": torch.randn(NKV * HEAD_DIM) * 0.02,
        "o_bias": torch.randn(HIDDEN) * 0.02,
        "sinks": torch.randn(NQ) * 0.5,
    }

    (cos_hf, sin_hf), (cos_meta, sin_meta) = _build_cos_sin(seq_len)

    ref_sliding = SLIDING_WINDOW if is_sliding else None
    ref_out = _torch_attention(x.float(), w, cos_hf, sin_hf, ref_sliding)

    # --- TT module from the SAME weights, Meta-RoPE-swizzled (q/k proj weight + bias) ---
    hf_state = {
        "q_proj.weight": w["q"],
        "q_proj.bias": w["q_bias"],
        "k_proj.weight": w["k"],
        "k_proj.bias": w["k_bias"],
        "v_proj.weight": w["v"],
        "v_proj.bias": w["v_bias"],
        "o_proj.weight": w["o"],
        "o_proj.bias": w["o_bias"],
        "sinks": w["sinks"],
    }
    state = convert_hf_qkv_to_meta_format(hf_state, HEAD_DIM)

    mesh_config = MeshConfig(tuple(mesh_device.shape), tp=mesh_device.shape[1])

    attn_config = AttentionConfig(
        hidden_size=HIDDEN,
        num_heads=NQ,
        num_kv_heads=NKV,
        head_dim=HEAD_DIM,
        max_seq_len=max(seq_len, 128),
        sliding_window=SLIDING_WINDOW,  # nulled by Attention for full-attention layers
        rms_norm_eps=EPS,
    )

    # RoPE transformation matrix for the prefill rotary_embedding_llama op.
    trans_mat = ttnn.from_torch(
        get_rot_transformation_mat(),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    attn = Attention(
        mesh_device=mesh_device,
        config=attn_config,
        state_dict=state,
        ccl_manager=None,  # unused at TP=1 / SP=1
        mesh_config=mesh_config,
        program_config=ProgramConfig(),
        layer_idx=layer_idx,
        layer_types=layer_types,
        transformation_mats={"prefill": trans_mat},
        weight_dtype=ttnn.bfloat16,
    )
    assert attn.is_sliding == is_sliding
    assert attn.config.sliding_window == (SLIDING_WINDOW if is_sliding else None)

    def _to_tt_cos_sin(t):
        return ttnn.from_torch(
            t,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    rope_mats = [_to_tt_cos_sin(cos_meta), _to_tt_cos_sin(sin_meta)]

    x_tt = ttnn.from_torch(
        x.reshape(1, 1, seq_len, HIDDEN),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    tt_out = attn(x_tt, rope_mats=rope_mats, position_idx=None, kv_cache=None)
    out = ttnn.to_torch(ttnn.get_device_tensors(tt_out)[0]).reshape(1, seq_len, HIDDEN)

    passing, pcc = comp_pcc(ref_out, out, 0.99)
    logger.info(f"gpt-oss attention prefill vs ref ({'sliding' if is_sliding else 'full'}): {pcc}")
    assert passing, f"attention PCC fail ({'sliding' if is_sliding else 'full'}): {pcc}"
