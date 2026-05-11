# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Smoke test for adaRMSGemma block (PyTorch reference, no device required).

Confirms that with zero-init modulation (scale=shift=gate=0) the AdaRMS
block equals a plain Gemma block plus a zero residual. With non-zero
modulation, outputs differ.
"""

import torch

from models.experimental.pi0.common.configs import GemmaConfig
from models.experimental.pi0.reference.torch_gemma import precompute_freqs_cis
from models.experimental.pi0_5.reference.torch_gemma import AdaRMSGemmaBlock


def _random_block_weights(cfg: GemmaConfig, with_adarms_zero: bool) -> dict:
    g = torch.Generator().manual_seed(0)
    w = {
        "self_attn.q_proj.weight": torch.randn(cfg.num_heads * cfg.head_dim, cfg.width, generator=g) * 0.02,
        "self_attn.k_proj.weight": torch.randn(cfg.num_kv_heads * cfg.head_dim, cfg.width, generator=g) * 0.02,
        "self_attn.v_proj.weight": torch.randn(cfg.num_kv_heads * cfg.head_dim, cfg.width, generator=g) * 0.02,
        "self_attn.o_proj.weight": torch.randn(cfg.width, cfg.num_heads * cfg.head_dim, generator=g) * 0.02,
        "mlp.gate_proj.weight": torch.randn(cfg.mlp_dim, cfg.width, generator=g) * 0.02,
        "mlp.up_proj.weight": torch.randn(cfg.mlp_dim, cfg.width, generator=g) * 0.02,
        "mlp.down_proj.weight": torch.randn(cfg.width, cfg.mlp_dim, generator=g) * 0.02,
    }
    if with_adarms_zero:
        w["input_layernorm.dense.weight"] = torch.zeros(3 * cfg.width, cfg.width)
        w["input_layernorm.dense.bias"] = torch.zeros(3 * cfg.width)
        w["post_attention_layernorm.dense.weight"] = torch.zeros(3 * cfg.width, cfg.width)
        w["post_attention_layernorm.dense.bias"] = torch.zeros(3 * cfg.width)
    else:
        w["input_layernorm.dense.weight"] = torch.randn(3 * cfg.width, cfg.width, generator=g) * 0.02
        w["input_layernorm.dense.bias"] = torch.randn(3 * cfg.width, generator=g) * 0.02
        w["post_attention_layernorm.dense.weight"] = torch.randn(3 * cfg.width, cfg.width, generator=g) * 0.02
        w["post_attention_layernorm.dense.bias"] = torch.randn(3 * cfg.width, generator=g) * 0.02
    return w


def test_adarms_zero_modulation_zeroes_gate():
    """With zero-init modulation, gate=0 -> output equals input (no sublayer contribution)."""
    cfg = GemmaConfig.gemma_300m()
    w = _random_block_weights(cfg, with_adarms_zero=True)
    block = AdaRMSGemmaBlock(cfg, w, layer_idx=0)

    cos, sin = precompute_freqs_cis(cfg.head_dim, 64, cfg.rope_base)
    x = torch.randn(1, 4, cfg.width)
    cond = torch.randn(1, cfg.width)

    out, _ = block.forward(x, cos, sin, adarms_cond=cond)
    # With zero modulation, gate=0 -> residual contribution is zero,
    # so out should equal the input x exactly.
    assert torch.allclose(out, x, atol=1e-5), "Zero-init adaRMS should pass x through unchanged"


def test_adarms_nonzero_changes_output():
    """Non-zero modulation should change the output relative to the identity case."""
    cfg = GemmaConfig.gemma_300m()
    w_zero = _random_block_weights(cfg, with_adarms_zero=True)
    w_nonz = dict(w_zero)
    w_nonz["input_layernorm.dense.weight"] = torch.randn(3 * cfg.width, cfg.width) * 0.02
    w_nonz["post_attention_layernorm.dense.weight"] = torch.randn(3 * cfg.width, cfg.width) * 0.02

    cos, sin = precompute_freqs_cis(cfg.head_dim, 64, cfg.rope_base)
    x = torch.randn(1, 4, cfg.width)
    cond = torch.randn(1, cfg.width)

    out_zero, _ = AdaRMSGemmaBlock(cfg, w_zero, 0).forward(x, cos, sin, adarms_cond=cond)
    out_nonz, _ = AdaRMSGemmaBlock(cfg, w_nonz, 0).forward(x, cos, sin, adarms_cond=cond)
    assert not torch.allclose(out_zero, out_nonz)
