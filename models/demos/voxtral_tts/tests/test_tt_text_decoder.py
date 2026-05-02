"""
TTNN PCC tests for Voxtral-4B-TTS-2603 text decoder blocks.

Target: N150 (single Wormhole B0 device).
Tests attention, MLP, and full decoder block against reference golden tensors.
PCC > 0.99 required for all blocks.

Run:
  cd tt-metal
  export TT_METAL_HOME=$(pwd) && export PYTHONPATH=$(pwd):$(pwd)/models
  export ARCH_NAME=wormhole_b0
  source python_env/bin/activate
  pytest models/demos/voxtral_tts/tests/test_tt_text_decoder.py -v -s

Environment:
  VOXTRAL_MODEL_DIR  (optional, defaults to HF cache path)
"""

import os
from pathlib import Path

import pytest
import torch

import ttnn

MODEL_DIR = Path(
    os.environ.get(
        "VOXTRAL_MODEL_DIR",
        "/home/ttuser/.cache/huggingface/hub/models--mistralai--Voxtral-4B-TTS-2603/snapshots/b81be46c3777f88621676791b512bb01dc1cb970",
    )
)
GOLDEN_DIR = Path(__file__).parents[1] / "reference" / "golden"
WEIGHTS_PATH = MODEL_DIR / "consolidated.safetensors"

PCC_THRESHOLD = 0.99
P99_THRESHOLD = 0.02

pytestmark = pytest.mark.skipif(
    not WEIGHTS_PATH.exists(),
    reason=f"Model weights not found at {WEIGHTS_PATH}",
)


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float().flatten(), b.float().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def p99_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    diff = (a.float() - b.float()).abs().flatten()
    k = max(1, int(0.99 * diff.numel()))
    return diff.kthvalue(k).values.item()


def verify(ttnn_out: torch.Tensor, ref: torch.Tensor, name: str):
    p = pcc(ttnn_out, ref)
    p99 = p99_diff(ttnn_out, ref)
    print(f"\n  {name}: PCC={p:.6f}, p99_diff={p99:.6f}")
    assert p > PCC_THRESHOLD, f"{name} PCC={p:.4f} < {PCC_THRESHOLD}"
    assert p99 < P99_THRESHOLD, f"{name} p99_diff={p99:.4f} > {P99_THRESHOLD}"
    return p, p99


# ── Module-scoped fixtures ────────────────────────────────────────────────


@pytest.fixture(scope="module")
def device():
    d = ttnn.open_device(device_id=0)
    yield d
    ttnn.close_device(d)


@pytest.fixture(scope="module")
def state_dicts():
    from models.demos.voxtral_tts.tt.load_checkpoint import (
        get_acoustic_transformer_state,
        get_codec_decoder_state,
        get_text_decoder_state,
        load_state_dict,
    )

    sd = load_state_dict(WEIGHTS_PATH)
    return {
        "full": sd,
        "text": get_text_decoder_state(sd),
        "acoustic": get_acoustic_transformer_state(sd),
        "codec": get_codec_decoder_state(sd),
    }


@pytest.fixture(scope="module")
def voxtral_cfg(device):
    from models.demos.voxtral_tts.tt.model_config import VoxtralTTSConfig

    return VoxtralTTSConfig(mesh_device=device)


@pytest.fixture(scope="module")
def transformation_mats(device, voxtral_cfg):
    from models.tt_transformers.tt.common import get_rot_transformation_mat

    mat = get_rot_transformation_mat(dhead=voxtral_cfg.head_dim)
    return ttnn.as_tensor(
        mat,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


@pytest.fixture(scope="module")
def rot_mats_for_seq(device, voxtral_cfg):
    """Returns callable: seq_len → [cos_tt, sin_tt] in HF rotate_half format."""
    from models.tt_transformers.tt.common import precompute_freqs

    cos_raw, sin_raw = precompute_freqs(
        voxtral_cfg.head_dim,
        voxtral_cfg.max_seq_len * 2,
        voxtral_cfg.rope_theta,
        None,
        None,
    )
    # HF concatenated-halves format [max_seq, head_dim]
    cos_hf = torch.cat([cos_raw[: voxtral_cfg.max_seq_len], cos_raw[: voxtral_cfg.max_seq_len]], dim=-1)
    sin_hf = torch.cat([sin_raw[: voxtral_cfg.max_seq_len], sin_raw[: voxtral_cfg.max_seq_len]], dim=-1)

    cos_tt = ttnn.from_torch(
        cos_hf.unsqueeze(0).unsqueeze(0).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    sin_tt = ttnn.from_torch(
        sin_hf.unsqueeze(0).unsqueeze(0).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return lambda seq_len: [cos_tt, sin_tt]


@pytest.fixture(scope="module")
def goldens():
    gd = {}
    for pt_file in GOLDEN_DIR.glob("*.pt"):
        gd[pt_file.stem] = torch.load(pt_file, map_location="cpu", weights_only=True)
    return gd


# ── Tests ─────────────────────────────────────────────────────────────────


def test_text_attention_pcc(device, state_dicts, voxtral_cfg, transformation_mats, rot_mats_for_seq, goldens):
    """TtVoxtralTextAttention PCC > 0.99 vs reference on layer 0."""
    from models.demos.voxtral_tts.reference.functional import build_rope_cache, rms_norm, text_attention
    from models.demos.voxtral_tts.tt.attention import TtVoxtralTextAttention

    sd = state_dicts["text"]
    layer_idx = 0

    x_ref = goldens["text_layer0_input"]  # [1, S, 3072]
    B, S, D = x_ref.shape
    x_normed = rms_norm(x_ref, sd[f"layers.{layer_idx}.attention_norm.weight"])
    cos, sin = build_rope_cache(S, 128, voxtral_cfg.rope_theta, "cpu")
    ref_out, _ = text_attention(x_normed, sd, layer_idx, cos, sin)  # [1, S, D]

    attn = TtVoxtralTextAttention(
        device=device,
        state_dict=sd,
        weight_cache_path=None,
        layer_num=layer_idx,
        dtype=ttnn.bfloat16,
        transformation_mats=transformation_mats,
        configuration=voxtral_cfg,
    )

    x_tt = ttnn.from_torch(
        x_normed.to(torch.bfloat16).reshape(1, 1, S, D),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    rot_mats = rot_mats_for_seq(S)
    out_tt = attn.forward_prefill(x_tt, rot_mats, user_id=0)
    out_torch = ttnn.to_torch(out_tt).reshape(S, D).float()
    ttnn.deallocate(out_tt)

    verify(out_torch, ref_out.reshape(S, D), "text_attention_layer0")


def test_text_mlp_pcc(device, state_dicts, voxtral_cfg, goldens):
    """TtVoxtralTextMLP PCC > 0.99 vs reference on layer 0."""
    from models.demos.voxtral_tts.reference.functional import rms_norm, text_mlp
    from models.demos.voxtral_tts.tt.mlp import TtVoxtralTextMLP

    sd = state_dicts["text"]
    layer_idx = 0

    x_ref = goldens["text_layer0_output"]  # [1, S, 3072]
    B, S, D = x_ref.shape
    x_normed = rms_norm(x_ref, sd[f"layers.{layer_idx}.ffn_norm.weight"])
    ref_out, _ = text_mlp(x_normed, sd, layer_idx)

    mlp = TtVoxtralTextMLP(
        device=device,
        state_dict=sd,
        weight_cache_path=None,
        layer_num=layer_idx,
        dtype=ttnn.bfloat8_b,
        configuration=voxtral_cfg,
    )

    x_tt = ttnn.from_torch(
        x_normed.to(torch.bfloat16).reshape(1, 1, S, D),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    out_tt = mlp.forward(x_tt)
    out_torch = ttnn.to_torch(out_tt).reshape(S, D).float()
    ttnn.deallocate(out_tt)

    verify(out_torch, ref_out.reshape(S, D), "text_mlp_layer0")


def test_decoder_block_pcc(device, state_dicts, voxtral_cfg, transformation_mats, rot_mats_for_seq, goldens):
    """Full decoder block (attention + MLP + residuals) PCC > 0.99 on layer 0."""
    from models.demos.voxtral_tts.reference.functional import build_rope_cache, text_decoder_layer
    from models.demos.voxtral_tts.tt.attention import TtVoxtralTextAttention
    from models.demos.voxtral_tts.tt.mlp import TtVoxtralTextMLP

    sd = state_dicts["text"]
    layer_idx = 0

    x_ref = goldens["text_layer0_input"]  # [1, S, 3072]
    B, S, D = x_ref.shape
    cos, sin = build_rope_cache(S, 128, voxtral_cfg.rope_theta, "cpu")
    ref_out, _ = text_decoder_layer(x_ref, sd, layer_idx, cos, sin)  # [1, S, 3072]

    attn = TtVoxtralTextAttention(
        device=device,
        state_dict=sd,
        weight_cache_path=None,
        layer_num=layer_idx,
        dtype=ttnn.bfloat16,
        transformation_mats=transformation_mats,
        configuration=voxtral_cfg,
    )
    mlp = TtVoxtralTextMLP(
        device=device,
        state_dict=sd,
        weight_cache_path=None,
        layer_num=layer_idx,
        dtype=ttnn.bfloat8_b,
        configuration=voxtral_cfg,
    )

    # RMSNorms
    attn_norm_w = ttnn.from_torch(
        sd[f"layers.{layer_idx}.attention_norm.weight"].to(torch.bfloat16).reshape(1, 1, 1, D),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ffn_norm_w = ttnn.from_torch(
        sd[f"layers.{layer_idx}.ffn_norm.weight"].to(torch.bfloat16).reshape(1, 1, 1, D),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    x_tt = ttnn.from_torch(
        x_ref.to(torch.bfloat16).reshape(1, 1, S, D),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    rot_mats = rot_mats_for_seq(S)

    # Attention residual
    x_normed_tt = ttnn.rms_norm(x_tt, weight=attn_norm_w, epsilon=voxtral_cfg.norm_eps)
    attn_out_tt = attn.forward_prefill(x_normed_tt, rot_mats, user_id=0)
    x_tt = ttnn.add(x_tt, attn_out_tt)
    ttnn.deallocate(attn_out_tt)

    # MLP residual
    x_normed_tt2 = ttnn.rms_norm(x_tt, weight=ffn_norm_w, epsilon=voxtral_cfg.norm_eps)
    mlp_out_tt = mlp.forward(x_normed_tt2)
    x_tt = ttnn.add(x_tt, mlp_out_tt)
    ttnn.deallocate(mlp_out_tt)

    out_torch = ttnn.to_torch(x_tt).reshape(S, D).float()
    ttnn.deallocate(x_tt)

    verify(out_torch, ref_out.reshape(S, D), "decoder_block_layer0")


def test_full_text_decoder_pcc(state_dicts, goldens):
    """26-layer reference decoder matches saved golden (no device needed)."""
    from models.demos.voxtral_tts.reference.functional import text_decoder_forward

    sd = state_dicts["text"]
    input_ids = goldens["text_input_ids"]
    ref_out = goldens["text_decoder_output"]

    out, _ = text_decoder_forward(input_ids, sd)
    verify(out, ref_out, "full_text_decoder_golden_match")
