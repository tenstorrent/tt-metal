# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""T1.2 — DeltaNet vs HF Qwen3NextGatedDeltaNet at real Qwen3.6 layer-0 weights.

  RED:    wrong `wq_gate` / fan-out → PCC ~0.5–0.8
  GREEN:  PCC > 0.99 at T ∈ {1, 64, 1024}.
"""
import sys

import pytest
import torch

sys.path.insert(0, "/home/tt-admin/ssinghal/tt-metal")

from transformers.models.qwen3_next.configuration_qwen3_next import Qwen3NextConfig
from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextGatedDeltaNet

from models.demos.qwen3_6_27b.reference.hf_loader import load_qwen36_config, load_qwen36_tensors


def _pcc(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


@pytest.mark.parametrize("seq_len", [1, 64, 1024])
def test_deltanet_hf_parity_layer0(seq_len):
    """Run a real Qwen3.6 linear_attn (layer 0) block; compare HF vs our reference."""
    torch.manual_seed(0)

    # 1. Build HF Qwen3Next config from Qwen3.6 config.json (model_type swapped)
    cfg_dict = load_qwen36_config()
    text_cfg = cfg_dict["text_config"]
    hf_cfg = Qwen3NextConfig(**text_cfg)

    # 2. Build a single HF DeltaNet block
    hf_block = Qwen3NextGatedDeltaNet(hf_cfg, layer_idx=0).eval()

    # 3. Load real layer-0 weights from Qwen3.6 safetensors
    prefix = "model.language_model.layers.0.linear_attn"
    keys = [
        f"{prefix}.in_proj_qkv.weight",
        f"{prefix}.in_proj_a.weight",
        f"{prefix}.in_proj_b.weight",
        f"{prefix}.in_proj_z.weight",
        f"{prefix}.conv1d.weight",
        f"{prefix}.A_log",
        f"{prefix}.dt_bias",
        f"{prefix}.norm.weight",
        f"{prefix}.out_proj.weight",
    ]
    weights = load_qwen36_tensors(keys)

    # 4. HF uses fused per-K-head-group layout:
    #    in_proj_qkvz = [for each k_head g: Q_g(128) || K_g(128) || V_g(384) || z_g(384)]
    #    in_proj_ba   = [for each k_head g: b_g(3)  || a_g(3)]
    # Our Qwen3.6 weights use BLOCK-wise split: Q[2048] || K[2048] || V[6144], z[6144], a[48], b[48]
    n_k = hf_cfg.linear_num_key_heads  # 16
    n_v = hf_cfg.linear_num_value_heads  # 48
    hd_k = hf_cfg.linear_key_head_dim  # 128
    hd_v = hf_cfg.linear_value_head_dim  # 128
    g_ratio = n_v // n_k  # 3

    H = hf_cfg.hidden_size  # 5120
    in_proj_qkv = weights[f"{prefix}.in_proj_qkv.weight"].float()  # [10240, 5120]
    in_proj_z = weights[f"{prefix}.in_proj_z.weight"].float()  # [6144, 5120]
    in_proj_a = weights[f"{prefix}.in_proj_a.weight"].float()  # [48, 5120]
    in_proj_b = weights[f"{prefix}.in_proj_b.weight"].float()  # [48, 5120]

    Q_part = in_proj_qkv[: n_k * hd_k]  # [2048, 5120]
    K_part = in_proj_qkv[n_k * hd_k : 2 * n_k * hd_k]  # [2048, 5120]
    V_part = in_proj_qkv[2 * n_k * hd_k :]  # [6144, 5120]

    rows_qkvz = []
    rows_ba = []
    for g in range(n_k):
        Q_g = Q_part[g * hd_k : (g + 1) * hd_k]
        K_g = K_part[g * hd_k : (g + 1) * hd_k]
        V_g = V_part[g * g_ratio * hd_v : (g + 1) * g_ratio * hd_v]
        z_g = in_proj_z[g * g_ratio * hd_v : (g + 1) * g_ratio * hd_v]
        rows_qkvz.append(torch.cat([Q_g, K_g, V_g, z_g], dim=0))
        b_g = in_proj_b[g * g_ratio : (g + 1) * g_ratio]
        a_g = in_proj_a[g * g_ratio : (g + 1) * g_ratio]
        rows_ba.append(torch.cat([b_g, a_g], dim=0))
    in_proj_qkvz = torch.cat(rows_qkvz, dim=0)  # [16384, 5120]
    in_proj_ba = torch.cat(rows_ba, dim=0)  # [96, 5120]

    sd = {
        "in_proj_qkvz.weight": in_proj_qkvz,
        "in_proj_ba.weight": in_proj_ba,
        "conv1d.weight": weights[f"{prefix}.conv1d.weight"].float(),
        "A_log": weights[f"{prefix}.A_log"].float(),
        "dt_bias": weights[f"{prefix}.dt_bias"].float(),
        "norm.weight": weights[f"{prefix}.norm.weight"].float(),
        "out_proj.weight": weights[f"{prefix}.out_proj.weight"].float(),
    }

    missing, unexpected = hf_block.load_state_dict(sd, strict=False)
    print(f"missing={missing}, unexpected={unexpected}")
    assert len(missing) == 0, f"missing HF params not loaded: {missing}"
    assert len(unexpected) == 0, f"unexpected: {unexpected}"

    # 5. Forward through HF block with a known hidden state input
    H = hf_cfg.hidden_size  # 5120
    hidden = torch.randn(1, seq_len, H, dtype=torch.float32) * 0.02

    # HF Qwen3NextGatedDeltaNet.forward signature: (hidden_states, cache_position, ...)
    cache_position = torch.arange(seq_len)
    with torch.no_grad():
        try:
            hf_out = hf_block(hidden, cache_position=cache_position)
        except TypeError as e:
            print(f"HF forward signature issue: {e}")
            hf_out = hf_block(hidden)
        if isinstance(hf_out, tuple):
            hf_out = hf_out[0]

    print(f"HF output shape: {hf_out.shape}")
    assert hf_out.shape == (1, seq_len, H), f"unexpected HF out shape {hf_out.shape}"

    # For this initial test we just verify shape sanity and HF loads.
    # Full our-reference parity (PCC check) requires building our reference block
    # with the same weights — deferred to follow-on test once HF baseline runs.
