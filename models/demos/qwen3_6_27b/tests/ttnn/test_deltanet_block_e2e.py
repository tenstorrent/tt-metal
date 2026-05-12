# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Full DeltaNet block end-to-end on single BH chip vs HF reference at layer-0 weights.

Wished-for API:
    block = TtDeltaNetBlock(device, state_dict, prefix, hf_config)
    out = block(hidden_states_tt)

RED expected: TtDeltaNetBlock does not exist; ImportError.
"""
import sys

import pytest
import torch

import ttnn

sys.path.insert(0, "/home/tt-admin/ssinghal/tt-metal")

from transformers.models.qwen3_next.configuration_qwen3_next import Qwen3NextConfig
from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextGatedDeltaNet

from models.demos.qwen3_6_27b.reference.hf_loader import load_qwen36_config, load_qwen36_tensors


def _pcc(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


@pytest.fixture(scope="module")
def device():
    dev = ttnn.open_device(device_id=0)
    yield dev
    ttnn.close_device(dev)


def _reconstruct_hf_qkvz_ba(prefix, weights, n_v, n_k, hd_k, hd_v):
    g_ratio = n_v // n_k
    in_proj_qkv = weights[f"{prefix}.in_proj_qkv.weight"].float()
    in_proj_z = weights[f"{prefix}.in_proj_z.weight"].float()
    in_proj_a = weights[f"{prefix}.in_proj_a.weight"].float()
    in_proj_b = weights[f"{prefix}.in_proj_b.weight"].float()

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
    return torch.cat(rows_qkvz, dim=0), torch.cat(rows_ba, dim=0)


def test_tt_deltanet_block_layer0_pcc_vs_hf(device):
    """TtDeltaNetBlock forward at layer-0 weights matches HF Qwen3NextGatedDeltaNet PCC > 0.99."""
    torch.manual_seed(42)
    cfg_dict = load_qwen36_config()
    text_cfg = cfg_dict["text_config"]
    hf_cfg = Qwen3NextConfig(**text_cfg)

    # 1. Load real layer-0 linear_attn weights
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

    # 2. Build HF reference, load weights (fused qkvz/ba layout)
    hf_block = Qwen3NextGatedDeltaNet(hf_cfg, layer_idx=0).eval()
    in_proj_qkvz, in_proj_ba = _reconstruct_hf_qkvz_ba(
        prefix,
        weights,
        n_v=hf_cfg.linear_num_value_heads,
        n_k=hf_cfg.linear_num_key_heads,
        hd_k=hf_cfg.linear_key_head_dim,
        hd_v=hf_cfg.linear_value_head_dim,
    )
    hf_block.load_state_dict(
        {
            "in_proj_qkvz.weight": in_proj_qkvz,
            "in_proj_ba.weight": in_proj_ba,
            "conv1d.weight": weights[f"{prefix}.conv1d.weight"].float(),
            "A_log": weights[f"{prefix}.A_log"].float(),
            "dt_bias": weights[f"{prefix}.dt_bias"].float(),
            "norm.weight": weights[f"{prefix}.norm.weight"].float(),
            "out_proj.weight": weights[f"{prefix}.out_proj.weight"].float(),
        }
    )

    # 3. Build our TT DeltaNet block (does NOT exist yet — RED)
    from models.demos.qwen3_6_27b.tt.linear_attention import TtDeltaNetBlock

    tt_block = TtDeltaNetBlock(device, weights, prefix, hf_cfg)

    # 4. Build input
    B, T, H = 1, 64, hf_cfg.hidden_size  # 5120
    hidden = torch.randn(B, T, H, dtype=torch.float32) * 0.02

    # 5. HF reference
    with torch.no_grad():
        hf_out = hf_block(hidden, cache_position=torch.arange(T))
        if isinstance(hf_out, tuple):
            hf_out = hf_out[0]

    # 6. TT forward
    hidden_tt = ttnn.from_torch(
        hidden,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    tt_out = tt_block(hidden_tt)
    tt_out_back = ttnn.to_torch(tt_out).float()

    print(f"shapes: tt={tt_out_back.shape}, hf={hf_out.shape}")
    assert tt_out_back.shape == hf_out.shape, f"shape mismatch"
    pcc = _pcc(tt_out_back, hf_out)
    print(f"PCC = {pcc:.6f}")
    assert pcc > 0.99, f"PCC {pcc:.6f} below 0.99"
