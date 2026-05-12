# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""TDD tests for models/demos/qwen3_6_galaxy/reference/qwen36.py.

All tests compare our reference module against HF Qwen3Next groundtruth
or internal consistency checks. Each class is tested RED→GREEN.

Run individual tests with:
    python3 -m pytest models/demos/qwen3_6_galaxy/tests/test_reference_qwen36.py::<test_name> -x -s
"""
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, "/home/tt-admin/ssinghal/tt-metal")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SNAPSHOT = Path(
    "/home/tt-admin/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B"
    "/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
)

_CFG_DICT = None


def _get_cfg_dict():
    global _CFG_DICT
    if _CFG_DICT is None:
        with open(SNAPSHOT / "config.json") as f:
            _CFG_DICT = json.load(f)
    return _CFG_DICT


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float().flatten(), b.float().flatten()
    if a.std() < 1e-10 or b.std() < 1e-10:
        return float(torch.allclose(a, b, atol=1e-5))
    return float(nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())


def _load_tensors(keys):
    from safetensors.torch import load_file as load_st

    with open(SNAPSHOT / "model.safetensors.index.json") as f:
        idx = json.load(f)
    weight_map = idx["weight_map"]
    files_needed = sorted({weight_map[k] for k in keys if k in weight_map})
    out = {}
    for fn in files_needed:
        shard = load_st(str(SNAPSHOT / fn))
        for k in keys:
            if k in shard:
                out[k] = shard[k]
    return out


# ---------------------------------------------------------------------------
# Test 1: RMSNorm zero-centered matches HF Qwen3NextRMSNorm
# ---------------------------------------------------------------------------


def test_rmsnorm_zero_centered_matches_hf():
    """RMSNorm(zero_centered=True) must bitwise match HF Qwen3NextRMSNorm (which uses 1+weight)."""
    from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextRMSNorm

    from models.demos.qwen3_6_galaxy.reference.qwen36 import RMSNorm

    torch.manual_seed(42)
    dim = 128
    x = torch.randn(2, 16, dim)
    w = torch.randn(dim) * 0.1  # note: HF inits to zeros; here we use random for sensitivity

    hf_norm = Qwen3NextRMSNorm(dim)
    hf_norm.weight.data.copy_(w)

    our_norm = RMSNorm(dim, zero_centered=True)
    our_norm.weight.data.copy_(w)

    with torch.no_grad():
        hf_out = hf_norm(x)
        our_out = our_norm(x)

    pcc = _pcc(hf_out, our_out)
    print(f"RMSNorm zero-centered PCC: {pcc:.6f}")
    assert pcc > 0.9999, f"PCC too low: {pcc}"
    assert torch.allclose(hf_out.float(), our_out.float(), atol=1e-5), "Not allclose"


# ---------------------------------------------------------------------------
# Test 2: RMSNorm standard (non-zero-centered) matches torch.nn.functional
# ---------------------------------------------------------------------------


def test_rmsnorm_standard_matches_torch():
    """RMSNorm(zero_centered=False) must match w*norm(x) standard convention."""
    from models.demos.qwen3_6_galaxy.reference.qwen36 import RMSNorm

    torch.manual_seed(7)
    dim = 64
    x = torch.randn(3, 10, dim)
    w = torch.ones(dim) + torch.randn(dim) * 0.2

    our_norm = RMSNorm(dim, zero_centered=False)
    our_norm.weight.data.copy_(w)

    with torch.no_grad():
        our_out = our_norm(x)

    # Reference: w * x * rsqrt(mean(x^2) + eps)
    x_f = x.float()
    rms = x_f.pow(2).mean(-1, keepdim=True).add(1e-6).rsqrt()
    ref_out = (w * x_f * rms).to(x.dtype)

    pcc = _pcc(our_out, ref_out)
    print(f"RMSNorm standard PCC: {pcc:.6f}")
    assert pcc > 0.9999, f"PCC too low: {pcc}"
    assert torch.allclose(our_out.float(), ref_out.float(), atol=1e-5), "Not allclose"


# ---------------------------------------------------------------------------
# Test 3: MRoPE text-only matches HF Qwen3NextRotaryEmbedding
# ---------------------------------------------------------------------------


def test_mrope_text_only_matches_hf():
    """build_mrope_cos_sin with text-only positions must PCC > 0.99 vs HF Qwen3NextRotaryEmbedding."""
    from transformers.models.qwen3_next.configuration_qwen3_next import Qwen3NextConfig
    from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextRotaryEmbedding

    from models.demos.qwen3_6_galaxy.reference.qwen36 import Qwen36Config, build_mrope_cos_sin

    cfg_dict = _get_cfg_dict()
    tc = dict(cfg_dict["text_config"])
    tc["model_type"] = "qwen3_next"
    # Pass rope_theta explicitly from rope_parameters so HF uses the correct value
    tc["rope_theta"] = tc["rope_parameters"]["rope_theta"]  # 10_000_000
    hf_cfg = Qwen3NextConfig(**tc)

    our_cfg = Qwen36Config(cfg_dict)

    T = 64
    # HF side
    hf_rope = Qwen3NextRotaryEmbedding(hf_cfg)
    x_dummy = torch.zeros(1, T, 1)
    pos_ids = torch.arange(T).unsqueeze(0).long()
    with torch.no_grad():
        hf_cos, hf_sin = hf_rope(x_dummy, pos_ids)
    print(f"HF cos shape: {hf_cos.shape}")  # [1, T, rotary_dim]

    # Our side: text-only => positions_3d is (T,T,T) collapsed
    # build_mrope_cos_sin signature: (positions_3d, head_dim, partial_rotary_factor, mrope_section, theta)
    rotary_dim = int(our_cfg.head_dim * our_cfg.partial_rotary_factor)
    positions = torch.arange(T).long()
    # For text-only, all three position axes equal the token position
    positions_3d = positions.unsqueeze(0).expand(3, -1)  # [3, T]
    our_cos, our_sin = build_mrope_cos_sin(
        positions_3d,
        our_cfg.head_dim,
        our_cfg.partial_rotary_factor,
        our_cfg.mrope_section,
        our_cfg.rope_theta,
    )
    print(f"Our cos shape: {our_cos.shape}")  # should be [1, T, rotary_dim] or [T, rotary_dim]

    # Ensure shape compatibility [1, T, rotary_dim]
    if our_cos.ndim == 2:
        our_cos = our_cos.unsqueeze(0)
        our_sin = our_sin.unsqueeze(0)

    pcc_cos = _pcc(hf_cos.float(), our_cos.float())
    pcc_sin = _pcc(hf_sin.float(), our_sin.float())
    print(f"MRoPE text-only PCC cos: {pcc_cos:.6f}, sin: {pcc_sin:.6f}")
    assert pcc_cos > 0.99, f"cos PCC too low: {pcc_cos}"
    assert pcc_sin > 0.99, f"sin PCC too low: {pcc_sin}"


# ---------------------------------------------------------------------------
# Test 4: GatedAttention layer-3 PCC vs HF Qwen3NextAttention
# ---------------------------------------------------------------------------


def test_gated_attention_layer3_pcc_vs_hf():
    """GatedAttention with real layer-3 weights must PCC > 0.99 vs HF Qwen3NextAttention."""
    from transformers.models.qwen3_next.configuration_qwen3_next import Qwen3NextConfig
    from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextAttention

    from models.demos.qwen3_6_galaxy.reference.qwen36 import GatedAttention, Qwen36Config, build_mrope_cos_sin

    cfg_dict = _get_cfg_dict()
    tc = dict(cfg_dict["text_config"])
    tc["model_type"] = "qwen3_next"
    tc["rope_theta"] = tc["rope_parameters"]["rope_theta"]  # 10_000_000
    hf_cfg = Qwen3NextConfig(**tc)
    hf_cfg._attn_implementation = "eager"
    our_cfg = Qwen36Config(cfg_dict)

    LAYER_IDX = 3
    B, T = 1, 32

    prefix = f"model.language_model.layers.{LAYER_IDX}.self_attn"
    keys = [
        f"{prefix}.q_proj.weight",
        f"{prefix}.k_proj.weight",
        f"{prefix}.v_proj.weight",
        f"{prefix}.o_proj.weight",
        f"{prefix}.q_norm.weight",
        f"{prefix}.k_norm.weight",
    ]
    weights = _load_tensors(keys)

    # Build HF attention block
    hf_attn = Qwen3NextAttention(hf_cfg, layer_idx=LAYER_IDX).eval()
    hf_sd = {
        "q_proj.weight": weights[f"{prefix}.q_proj.weight"].float(),
        "k_proj.weight": weights[f"{prefix}.k_proj.weight"].float(),
        "v_proj.weight": weights[f"{prefix}.v_proj.weight"].float(),
        "o_proj.weight": weights[f"{prefix}.o_proj.weight"].float(),
        "q_norm.weight": weights[f"{prefix}.q_norm.weight"].float(),
        "k_norm.weight": weights[f"{prefix}.k_norm.weight"].float(),
    }
    hf_attn.load_state_dict(hf_sd, strict=True)

    # Build our GatedAttention
    from models.demos.qwen3_6_galaxy.reference.qwen36 import load_layer_weights_qwen36

    our_attn = GatedAttention(our_cfg).eval()
    our_sd = load_layer_weights_qwen36(str(SNAPSHOT), LAYER_IDX, "full_attention")
    our_attn.load_state_dict(our_sd, strict=True)

    # Build cos/sin
    rotary_dim = int(our_cfg.head_dim * our_cfg.partial_rotary_factor)
    positions = torch.arange(T).long()
    positions_3d = positions.unsqueeze(0).expand(3, -1)
    cos, sin = build_mrope_cos_sin(
        positions_3d, our_cfg.head_dim, our_cfg.partial_rotary_factor, our_cfg.mrope_section, our_cfg.rope_theta
    )
    # cos/sin: [T, rotary_dim] or [1, T, rotary_dim]
    if cos.ndim == 2:
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)

    torch.manual_seed(0)
    x = torch.randn(B, T, our_cfg.hidden_size) * 0.02

    # HF forward
    with torch.no_grad():
        hf_out, _ = hf_attn(x, position_embeddings=(cos, sin), attention_mask=None)

    # Our forward: takes (x, cos, sin, kv_cache=None)
    with torch.no_grad():
        our_out, _ = our_attn(x, cos, sin)

    pcc = _pcc(hf_out, our_out)
    print(f"GatedAttention layer-3 PCC: {pcc:.6f}")
    assert pcc > 0.99, f"PCC too low: {pcc}"


# ---------------------------------------------------------------------------
# Test 5: GatedDeltaNet layer-0 PCC vs HF Qwen3NextGatedDeltaNet
# ---------------------------------------------------------------------------


def _reconstruct_hf_qkvz_ba(weights, prefix, n_k=16, n_v=48, hd_k=128, hd_v=128):
    """Reconstruct HF fused in_proj_qkvz / in_proj_ba from our block-wise split layout."""
    g_ratio = n_v // n_k
    in_proj_qkv = weights[f"{prefix}.in_proj_qkv.weight"].float()  # [10240, H]
    in_proj_z = weights[f"{prefix}.in_proj_z.weight"].float()  # [6144, H]
    in_proj_a = weights[f"{prefix}.in_proj_a.weight"].float()  # [48, H]
    in_proj_b = weights[f"{prefix}.in_proj_b.weight"].float()  # [48, H]

    Q_part = in_proj_qkv[: n_k * hd_k]
    K_part = in_proj_qkv[n_k * hd_k : 2 * n_k * hd_k]
    V_part = in_proj_qkv[2 * n_k * hd_k :]

    rows_qkvz, rows_ba = [], []
    for g in range(n_k):
        Q_g = Q_part[g * hd_k : (g + 1) * hd_k]
        K_g = K_part[g * hd_k : (g + 1) * hd_k]
        V_g = V_part[g * g_ratio * hd_v : (g + 1) * g_ratio * hd_v]
        z_g = in_proj_z[g * g_ratio * hd_v : (g + 1) * g_ratio * hd_v]
        rows_qkvz.append(torch.cat([Q_g, K_g, V_g, z_g], dim=0))
        b_g = in_proj_b[g * g_ratio : (g + 1) * g_ratio]
        a_g = in_proj_a[g * g_ratio : (g + 1) * g_ratio]
        rows_ba.append(torch.cat([b_g, a_g], dim=0))

    in_proj_qkvz = torch.cat(rows_qkvz, dim=0)
    in_proj_ba = torch.cat(rows_ba, dim=0)
    return in_proj_qkvz, in_proj_ba


def test_gated_deltanet_layer0_pcc_vs_hf():
    """GatedDeltaNet with real layer-0 weights must PCC > 0.99 vs HF Qwen3NextGatedDeltaNet."""
    from transformers.models.qwen3_next.configuration_qwen3_next import Qwen3NextConfig
    from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextGatedDeltaNet

    from models.demos.qwen3_6_galaxy.reference.qwen36 import GatedDeltaNet, Qwen36Config, load_layer_weights_qwen36

    cfg_dict = _get_cfg_dict()
    tc = dict(cfg_dict["text_config"])
    tc["model_type"] = "qwen3_next"
    tc["rope_theta"] = tc["rope_parameters"]["rope_theta"]  # 10_000_000
    hf_cfg = Qwen3NextConfig(**tc)
    our_cfg = Qwen36Config(cfg_dict)

    LAYER_IDX = 0
    B, T = 1, 32

    prefix = f"model.language_model.layers.{LAYER_IDX}.linear_attn"
    keys = [
        f"{prefix}.in_proj_qkv.weight",
        f"{prefix}.in_proj_z.weight",
        f"{prefix}.in_proj_a.weight",
        f"{prefix}.in_proj_b.weight",
        f"{prefix}.conv1d.weight",
        f"{prefix}.A_log",
        f"{prefix}.dt_bias",
        f"{prefix}.norm.weight",
        f"{prefix}.out_proj.weight",
    ]
    weights = _load_tensors(keys)

    # Build HF block and load weights
    hf_block = Qwen3NextGatedDeltaNet(hf_cfg, layer_idx=LAYER_IDX).eval()
    qkvz, ba = _reconstruct_hf_qkvz_ba(weights, prefix)
    hf_sd = {
        "in_proj_qkvz.weight": qkvz,
        "in_proj_ba.weight": ba,
        "conv1d.weight": weights[f"{prefix}.conv1d.weight"].float(),
        "A_log": weights[f"{prefix}.A_log"].float(),
        "dt_bias": weights[f"{prefix}.dt_bias"].float(),
        "norm.weight": weights[f"{prefix}.norm.weight"].float(),
        "out_proj.weight": weights[f"{prefix}.out_proj.weight"].float(),
    }
    missing, unexpected = hf_block.load_state_dict(hf_sd, strict=False)
    assert len(missing) == 0, f"HF missing: {missing}"

    # Build our GatedDeltaNet
    our_block = GatedDeltaNet(our_cfg).eval()
    our_sd = load_layer_weights_qwen36(str(SNAPSHOT), LAYER_IDX, "linear_attention")
    our_block.load_state_dict(our_sd, strict=True)

    torch.manual_seed(0)
    x = torch.randn(B, T, our_cfg.hidden_size) * 0.02

    cache_pos = torch.arange(T)
    with torch.no_grad():
        hf_out = hf_block(x, cache_position=cache_pos)
        if isinstance(hf_out, tuple):
            hf_out = hf_out[0]

    with torch.no_grad():
        our_out, _, _ = our_block(x)

    pcc = _pcc(hf_out, our_out)
    print(f"GatedDeltaNet layer-0 PCC: {pcc:.6f}")
    assert pcc > 0.99, f"PCC too low: {pcc}"


# ---------------------------------------------------------------------------
# Test 6: HybridDecoderLayer x4 PCC vs HF
# ---------------------------------------------------------------------------


def test_hybrid_4layer_pcc_vs_hf():
    """HybridDecoderLayer x4 (layers 0-3) must PCC > 0.99 vs HF reference."""
    import torch.nn.functional as F
    from transformers.models.qwen3_next.configuration_qwen3_next import Qwen3NextConfig
    from transformers.models.qwen3_next.modeling_qwen3_next import (
        Qwen3NextAttention,
        Qwen3NextGatedDeltaNet,
        Qwen3NextRMSNorm,
    )

    from models.demos.qwen3_6_galaxy.reference.qwen36 import (
        HybridDecoderLayer,
        Qwen36Config,
        build_mrope_cos_sin,
        load_layer_weights_qwen36,
    )

    cfg_dict = _get_cfg_dict()
    tc = dict(cfg_dict["text_config"])
    tc["model_type"] = "qwen3_next"
    tc["rope_theta"] = tc["rope_parameters"]["rope_theta"]  # 10_000_000
    hf_cfg = Qwen3NextConfig(**tc)
    hf_cfg._attn_implementation = "eager"
    our_cfg = Qwen36Config(cfg_dict)

    B, T = 1, 16
    N_LAYERS = 4  # layers 0,1,2 = linear_attention; layer 3 = full_attention

    # Load weights for layers 0-3
    all_keys = []
    for i in range(N_LAYERS):
        base = f"model.language_model.layers.{i}"
        all_keys += [
            f"{base}.input_layernorm.weight",
            f"{base}.post_attention_layernorm.weight",
            f"{base}.mlp.gate_proj.weight",
            f"{base}.mlp.up_proj.weight",
            f"{base}.mlp.down_proj.weight",
        ]
        if our_cfg.layer_types[i] == "linear_attention":
            pfx = f"{base}.linear_attn"
            all_keys += [
                f"{pfx}.in_proj_qkv.weight",
                f"{pfx}.in_proj_z.weight",
                f"{pfx}.in_proj_a.weight",
                f"{pfx}.in_proj_b.weight",
                f"{pfx}.conv1d.weight",
                f"{pfx}.A_log",
                f"{pfx}.dt_bias",
                f"{pfx}.norm.weight",
                f"{pfx}.out_proj.weight",
            ]
        else:
            pfx = f"{base}.self_attn"
            all_keys += [
                f"{pfx}.q_proj.weight",
                f"{pfx}.k_proj.weight",
                f"{pfx}.v_proj.weight",
                f"{pfx}.o_proj.weight",
                f"{pfx}.q_norm.weight",
                f"{pfx}.k_norm.weight",
            ]

    weights = _load_tensors(all_keys)

    # Build HF 4-layer reference (dense, non-MoE)
    class HFDenseLayer(nn.Module):
        def __init__(self, layer_idx, layer_type):
            super().__init__()
            self.input_ln = Qwen3NextRMSNorm(hf_cfg.hidden_size, eps=hf_cfg.rms_norm_eps)
            self.post_ln = Qwen3NextRMSNorm(hf_cfg.hidden_size, eps=hf_cfg.rms_norm_eps)
            self.gate_proj = nn.Linear(hf_cfg.hidden_size, hf_cfg.intermediate_size, bias=False)
            self.up_proj = nn.Linear(hf_cfg.hidden_size, hf_cfg.intermediate_size, bias=False)
            self.down_proj = nn.Linear(hf_cfg.intermediate_size, hf_cfg.hidden_size, bias=False)
            self.layer_type = layer_type
            if layer_type == "linear_attention":
                self.attn = Qwen3NextGatedDeltaNet(hf_cfg, layer_idx=layer_idx)
            else:
                self.attn = Qwen3NextAttention(hf_cfg, layer_idx=layer_idx)

        def forward(self, x, cos=None, sin=None):
            r = x
            x = self.input_ln(x)
            if self.layer_type == "linear_attention":
                out = self.attn(x, cache_position=torch.arange(x.shape[1]))
                if isinstance(out, tuple):
                    out = out[0]
            else:
                out, _ = self.attn(x, position_embeddings=(cos, sin), attention_mask=None)
            x = r + out
            r = x
            x = self.post_ln(x)
            return r + self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

    hf_layers = []
    for i in range(N_LAYERS):
        lt = our_cfg.layer_types[i]
        layer = HFDenseLayer(i, lt).eval()
        base = f"model.language_model.layers.{i}"
        sd = {
            "input_ln.weight": weights[f"{base}.input_layernorm.weight"].float(),
            "post_ln.weight": weights[f"{base}.post_attention_layernorm.weight"].float(),
            "gate_proj.weight": weights[f"{base}.mlp.gate_proj.weight"].float(),
            "up_proj.weight": weights[f"{base}.mlp.up_proj.weight"].float(),
            "down_proj.weight": weights[f"{base}.mlp.down_proj.weight"].float(),
        }
        if lt == "linear_attention":
            pfx = f"{base}.linear_attn"
            qkvz, ba = _reconstruct_hf_qkvz_ba(weights, pfx)
            sd["attn.in_proj_qkvz.weight"] = qkvz
            sd["attn.in_proj_ba.weight"] = ba
            sd["attn.conv1d.weight"] = weights[f"{pfx}.conv1d.weight"].float()
            sd["attn.A_log"] = weights[f"{pfx}.A_log"].float()
            sd["attn.dt_bias"] = weights[f"{pfx}.dt_bias"].float()
            sd["attn.norm.weight"] = weights[f"{pfx}.norm.weight"].float()
            sd["attn.out_proj.weight"] = weights[f"{pfx}.out_proj.weight"].float()
        else:
            pfx = f"{base}.self_attn"
            for k in [
                "q_proj.weight",
                "k_proj.weight",
                "v_proj.weight",
                "o_proj.weight",
                "q_norm.weight",
                "k_norm.weight",
            ]:
                sd[f"attn.{k}"] = weights[f"{pfx}.{k}"].float()
        missing, _ = layer.load_state_dict(sd, strict=False)
        assert len(missing) == 0, f"Layer {i} missing: {missing}"
        hf_layers.append(layer)

    # Build our 4-layer reference
    our_layers = []
    for i in range(N_LAYERS):
        layer = HybridDecoderLayer(our_cfg, layer_idx=i).eval()
        # Load weights for this layer
        base = f"model.language_model.layers.{i}"
        lt = our_cfg.layer_types[i]
        # Build state dict
        sd = {}
        sd["input_layernorm.weight"] = weights[f"{base}.input_layernorm.weight"].float()
        sd["post_attention_layernorm.weight"] = weights[f"{base}.post_attention_layernorm.weight"].float()
        sd["mlp.gate_proj.weight"] = weights[f"{base}.mlp.gate_proj.weight"].float()
        sd["mlp.up_proj.weight"] = weights[f"{base}.mlp.up_proj.weight"].float()
        sd["mlp.down_proj.weight"] = weights[f"{base}.mlp.down_proj.weight"].float()
        attn_sd = load_layer_weights_qwen36(str(SNAPSHOT), i, lt)
        for k, v in attn_sd.items():
            sd[f"attention.{k}"] = v.float()
        missing, unexpected = layer.load_state_dict(sd, strict=True)
        assert len(missing) == 0, f"Our layer {i} missing: {missing}"
        our_layers.append(layer)

    # Build cos/sin
    positions = torch.arange(T).long()
    positions_3d = positions.unsqueeze(0).expand(3, -1)
    cos, sin = build_mrope_cos_sin(
        positions_3d, our_cfg.head_dim, our_cfg.partial_rotary_factor, our_cfg.mrope_section, our_cfg.rope_theta
    )
    if cos.ndim == 2:
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)

    torch.manual_seed(1)
    x = torch.randn(B, T, our_cfg.hidden_size) * 0.02

    with torch.no_grad():
        hf_x = x.clone()
        for layer in hf_layers:
            hf_x = layer(hf_x, cos, sin)

        our_x = x.clone()
        for layer in our_layers:
            our_x, _, _, _ = layer(our_x, cos, sin)

    pcc = _pcc(hf_x, our_x)
    print(f"HybridDecoderLayer x4 PCC: {pcc:.6f}")
    assert pcc > 0.99, f"PCC too low: {pcc}"


# ---------------------------------------------------------------------------
# Test 7: Full model 4-layer top-1 token matches HF reference
# ---------------------------------------------------------------------------


def test_full_model_4layer_top1_vs_hf():
    """Qwen36TextModel with num_layers=4 must produce same top-1 token as HF reference."""
    import torch.nn.functional as F
    from transformers.models.qwen3_next.configuration_qwen3_next import Qwen3NextConfig
    from transformers.models.qwen3_next.modeling_qwen3_next import (
        Qwen3NextAttention,
        Qwen3NextGatedDeltaNet,
        Qwen3NextRMSNorm,
    )

    from models.demos.qwen3_6_galaxy.reference.qwen36 import (
        Qwen36Config,
        Qwen36TextModel,
        build_mrope_cos_sin,
        load_layer_weights_qwen36,
    )

    cfg_dict = _get_cfg_dict()
    tc = dict(cfg_dict["text_config"])
    tc["model_type"] = "qwen3_next"
    tc["rope_theta"] = tc["rope_parameters"]["rope_theta"]  # 10_000_000
    hf_cfg = Qwen3NextConfig(**tc)
    hf_cfg._attn_implementation = "eager"
    our_cfg = Qwen36Config(cfg_dict)

    N_LAYERS = 4
    B, T = 1, 16

    # Load all weights for 4 layers + embedding + norm + lm_head
    all_keys = [
        "model.language_model.embed_tokens.weight",
        "model.language_model.norm.weight",
        "lm_head.weight",
    ]
    for i in range(N_LAYERS):
        base = f"model.language_model.layers.{i}"
        all_keys += [
            f"{base}.input_layernorm.weight",
            f"{base}.post_attention_layernorm.weight",
            f"{base}.mlp.gate_proj.weight",
            f"{base}.mlp.up_proj.weight",
            f"{base}.mlp.down_proj.weight",
        ]
        if our_cfg.layer_types[i] == "linear_attention":
            pfx = f"{base}.linear_attn"
            all_keys += [
                f"{pfx}.in_proj_qkv.weight",
                f"{pfx}.in_proj_z.weight",
                f"{pfx}.in_proj_a.weight",
                f"{pfx}.in_proj_b.weight",
                f"{pfx}.conv1d.weight",
                f"{pfx}.A_log",
                f"{pfx}.dt_bias",
                f"{pfx}.norm.weight",
                f"{pfx}.out_proj.weight",
            ]
        else:
            pfx = f"{base}.self_attn"
            all_keys += [
                f"{pfx}.q_proj.weight",
                f"{pfx}.k_proj.weight",
                f"{pfx}.v_proj.weight",
                f"{pfx}.o_proj.weight",
                f"{pfx}.q_norm.weight",
                f"{pfx}.k_norm.weight",
            ]

    weights = _load_tensors(all_keys)

    # Build HF 4-layer model (ad-hoc)
    class HFDenseLayer(nn.Module):
        def __init__(self, layer_idx, layer_type):
            super().__init__()
            self.input_ln = Qwen3NextRMSNorm(hf_cfg.hidden_size, eps=hf_cfg.rms_norm_eps)
            self.post_ln = Qwen3NextRMSNorm(hf_cfg.hidden_size, eps=hf_cfg.rms_norm_eps)
            self.gate_proj = nn.Linear(hf_cfg.hidden_size, hf_cfg.intermediate_size, bias=False)
            self.up_proj = nn.Linear(hf_cfg.hidden_size, hf_cfg.intermediate_size, bias=False)
            self.down_proj = nn.Linear(hf_cfg.intermediate_size, hf_cfg.hidden_size, bias=False)
            self.layer_type = layer_type
            if layer_type == "linear_attention":
                self.attn = Qwen3NextGatedDeltaNet(hf_cfg, layer_idx=layer_idx)
            else:
                self.attn = Qwen3NextAttention(hf_cfg, layer_idx=layer_idx)

        def forward(self, x, cos, sin):
            r = x
            x = self.input_ln(x)
            if self.layer_type == "linear_attention":
                out = self.attn(x, cache_position=torch.arange(x.shape[1]))
                if isinstance(out, tuple):
                    out = out[0]
            else:
                out, _ = self.attn(x, position_embeddings=(cos, sin), attention_mask=None)
            x = r + out
            r = x
            x = self.post_ln(x)
            return r + self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

    hf_embed = nn.Embedding(hf_cfg.vocab_size, hf_cfg.hidden_size).eval()
    hf_embed.weight.data.copy_(weights["model.language_model.embed_tokens.weight"].float())
    hf_final_norm = Qwen3NextRMSNorm(hf_cfg.hidden_size, eps=hf_cfg.rms_norm_eps).eval()
    hf_final_norm.weight.data.copy_(weights["model.language_model.norm.weight"].float())
    hf_lm_head = nn.Linear(hf_cfg.hidden_size, hf_cfg.vocab_size, bias=False).eval()
    hf_lm_head.weight.data.copy_(weights["lm_head.weight"].float())

    hf_layers = []
    for i in range(N_LAYERS):
        lt = our_cfg.layer_types[i]
        layer = HFDenseLayer(i, lt).eval()
        base = f"model.language_model.layers.{i}"
        sd = {
            "input_ln.weight": weights[f"{base}.input_layernorm.weight"].float(),
            "post_ln.weight": weights[f"{base}.post_attention_layernorm.weight"].float(),
            "gate_proj.weight": weights[f"{base}.mlp.gate_proj.weight"].float(),
            "up_proj.weight": weights[f"{base}.mlp.up_proj.weight"].float(),
            "down_proj.weight": weights[f"{base}.mlp.down_proj.weight"].float(),
        }
        if lt == "linear_attention":
            pfx = f"{base}.linear_attn"
            qkvz, ba = _reconstruct_hf_qkvz_ba(weights, pfx)
            sd["attn.in_proj_qkvz.weight"] = qkvz
            sd["attn.in_proj_ba.weight"] = ba
            sd["attn.conv1d.weight"] = weights[f"{pfx}.conv1d.weight"].float()
            sd["attn.A_log"] = weights[f"{pfx}.A_log"].float()
            sd["attn.dt_bias"] = weights[f"{pfx}.dt_bias"].float()
            sd["attn.norm.weight"] = weights[f"{pfx}.norm.weight"].float()
            sd["attn.out_proj.weight"] = weights[f"{pfx}.out_proj.weight"].float()
        else:
            pfx = f"{base}.self_attn"
            for k in [
                "q_proj.weight",
                "k_proj.weight",
                "v_proj.weight",
                "o_proj.weight",
                "q_norm.weight",
                "k_norm.weight",
            ]:
                sd[f"attn.{k}"] = weights[f"{pfx}.{k}"].float()
        missing, _ = layer.load_state_dict(sd, strict=False)
        assert len(missing) == 0, f"HF layer {i} missing: {missing}"
        hf_layers.append(layer)

    # Build our model
    our_cfg_4layer = Qwen36Config(cfg_dict)
    our_cfg_4layer.num_hidden_layers = N_LAYERS
    our_model = Qwen36TextModel(our_cfg_4layer).eval()

    # Load weights into our model
    model_sd = {}
    model_sd["tok_embeddings.weight"] = weights["model.language_model.embed_tokens.weight"].float()
    model_sd["norm.weight"] = weights["model.language_model.norm.weight"].float()
    model_sd["lm_head.weight"] = weights["lm_head.weight"].float()

    for i in range(N_LAYERS):
        base = f"model.language_model.layers.{i}"
        lt = our_cfg.layer_types[i]
        model_sd[f"layers.{i}.input_layernorm.weight"] = weights[f"{base}.input_layernorm.weight"].float()
        model_sd[f"layers.{i}.post_attention_layernorm.weight"] = weights[
            f"{base}.post_attention_layernorm.weight"
        ].float()
        model_sd[f"layers.{i}.mlp.gate_proj.weight"] = weights[f"{base}.mlp.gate_proj.weight"].float()
        model_sd[f"layers.{i}.mlp.up_proj.weight"] = weights[f"{base}.mlp.up_proj.weight"].float()
        model_sd[f"layers.{i}.mlp.down_proj.weight"] = weights[f"{base}.mlp.down_proj.weight"].float()
        attn_sd = load_layer_weights_qwen36(str(SNAPSHOT), i, lt)
        for k, v in attn_sd.items():
            model_sd[f"layers.{i}.attention.{k}"] = v.float()

    missing, unexpected = our_model.load_state_dict(model_sd, strict=True)
    assert len(missing) == 0, f"Model missing: {missing}"

    # Fixed input
    torch.manual_seed(42)
    input_ids = torch.randint(0, hf_cfg.vocab_size, (B, T))

    # Build cos/sin
    positions = torch.arange(T).long()
    positions_3d = positions.unsqueeze(0).expand(3, -1)
    cos, sin = build_mrope_cos_sin(
        positions_3d, our_cfg.head_dim, our_cfg.partial_rotary_factor, our_cfg.mrope_section, our_cfg.rope_theta
    )
    if cos.ndim == 2:
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)

    # HF forward
    with torch.no_grad():
        hf_x = hf_embed(input_ids)
        for layer in hf_layers:
            hf_x = layer(hf_x, cos, sin)
        hf_x = hf_final_norm(hf_x)
        hf_logits = hf_lm_head(hf_x)

    # Our forward
    with torch.no_grad():
        our_logits = our_model(input_ids, cos, sin)

    hf_top1 = hf_logits[:, -1].argmax(dim=-1)
    our_top1 = our_logits[:, -1].argmax(dim=-1)
    pcc = _pcc(hf_logits, our_logits)

    print(f"Full model 4-layer logits PCC: {pcc:.6f}")
    print(f"HF top-1: {hf_top1.item()}, Our top-1: {our_top1.item()}")
    assert hf_top1.item() == our_top1.item(), f"Top-1 mismatch: HF={hf_top1.item()}, ours={our_top1.item()}"
    assert pcc > 0.99, f"Logits PCC too low: {pcc}"
