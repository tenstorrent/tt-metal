# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
No-device CPU smoke tests for Pi0.5 adaptive RMSNorm (adaRMS).

Exercises ``adarms_norm`` (the conditioned norm) and a full ``GemmaBlock``
expert layer with ``use_adarms=True`` using random weights. These run on CPU
with no Tenstorrent device and no pretrained weights, so they gate cheaply in
CI before the on-device PCC suite.

Key invariant: with zero-init modulation (dense weight + bias all zero) the
projection produces scale = shift = gate = 0, so
  - ``adarms_norm`` returns pure RMS-normalized ``x`` with an all-zero gate, and
  - a ``GemmaBlock`` becomes an exact identity (both gated residuals are ``*0``).
This pins down the chunk semantics (scale, shift, gate) that the TTNN fused
``ttnn.experimental.fused_adaptive_rms`` op must match.
"""

import pytest
import torch

from models.experimental.pi0_5.common.configs import GemmaConfig
from models.experimental.pi0_5.reference.torch_gemma import (
    GemmaBlock,
    adarms_norm,
    precompute_freqs_cis,
)


def _rms_normed(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Pure Gemma-style RMS normalization without the (weight+1) gain."""
    variance = x.float().pow(2).mean(dim=-1, keepdim=True)
    return (x * torch.rsqrt(variance + eps)).to(x.dtype)


def test_adarms_zero_modulation_is_pure_rmsnorm_and_zero_gate():
    torch.manual_seed(0)
    hidden, cond_dim = 64, 32
    x = torch.randn(2, 5, hidden)
    cond = torch.randn(2, cond_dim)
    dense_weight = torch.zeros(3 * hidden, cond_dim)
    dense_bias = torch.zeros(3 * hidden)

    normed, gate = adarms_norm(x, dense_weight, dense_bias, cond)

    # scale = shift = 0  ->  normed == pure RMSNorm(x);  gate == 0
    assert torch.allclose(normed, _rms_normed(x), atol=1e-5)
    assert torch.count_nonzero(gate) == 0
    assert normed.shape == x.shape
    assert gate.shape == (2, 1, hidden)


def test_adarms_nonzero_modulation_changes_output():
    torch.manual_seed(0)
    hidden, cond_dim = 64, 32
    x = torch.randn(1, 4, hidden)
    cond = torch.randn(1, cond_dim)
    zero_w = torch.zeros(3 * hidden, cond_dim)
    zero_b = torch.zeros(3 * hidden)
    rand_w = torch.randn(3 * hidden, cond_dim) * 0.1

    normed_zero, _ = adarms_norm(x, zero_w, zero_b, cond)
    normed_rand, gate_rand = adarms_norm(x, rand_w, zero_b, cond)

    assert not torch.allclose(normed_zero, normed_rand)
    assert torch.count_nonzero(gate_rand) > 0


@pytest.mark.parametrize("batch,seq", [(1, 1), (2, 8)])
def test_adarms_shapes(batch, seq):
    hidden, cond_dim = 48, 16
    x = torch.randn(batch, seq, hidden)
    cond = torch.randn(batch, cond_dim)
    w = torch.randn(3 * hidden, cond_dim) * 0.1
    b = torch.randn(3 * hidden) * 0.1

    normed, gate = adarms_norm(x, w, b, cond)

    assert normed.shape == (batch, seq, hidden)
    assert gate.shape == (batch, 1, hidden)


def _expert_block_weights(cfg: GemmaConfig, g: torch.Generator, zero_adarms: bool) -> dict:
    w = {
        "self_attn.q_proj.weight": torch.randn(cfg.num_heads * cfg.head_dim, cfg.width, generator=g) * 0.02,
        "self_attn.k_proj.weight": torch.randn(cfg.num_kv_heads * cfg.head_dim, cfg.width, generator=g) * 0.02,
        "self_attn.v_proj.weight": torch.randn(cfg.num_kv_heads * cfg.head_dim, cfg.width, generator=g) * 0.02,
        "self_attn.o_proj.weight": torch.randn(cfg.width, cfg.num_heads * cfg.head_dim, generator=g) * 0.02,
        "mlp.gate_proj.weight": torch.randn(cfg.mlp_dim, cfg.width, generator=g) * 0.02,
        "mlp.up_proj.weight": torch.randn(cfg.mlp_dim, cfg.width, generator=g) * 0.02,
        "mlp.down_proj.weight": torch.randn(cfg.width, cfg.mlp_dim, generator=g) * 0.02,
    }
    cond_dim = cfg.adarms_cond_dim
    if zero_adarms:
        w["input_layernorm.dense.weight"] = torch.zeros(3 * cfg.width, cond_dim)
        w["input_layernorm.dense.bias"] = torch.zeros(3 * cfg.width)
        w["post_attention_layernorm.dense.weight"] = torch.zeros(3 * cfg.width, cond_dim)
        w["post_attention_layernorm.dense.bias"] = torch.zeros(3 * cfg.width)
    else:
        w["input_layernorm.dense.weight"] = torch.randn(3 * cfg.width, cond_dim, generator=g) * 0.02
        w["input_layernorm.dense.bias"] = torch.randn(3 * cfg.width, generator=g) * 0.02
        w["post_attention_layernorm.dense.weight"] = torch.randn(3 * cfg.width, cond_dim, generator=g) * 0.02
        w["post_attention_layernorm.dense.bias"] = torch.randn(3 * cfg.width, generator=g) * 0.02
    return w


def test_gemma_expert_block_zero_adarms_is_identity():
    """gate=0 in both gated residuals -> attention and MLP contribute nothing."""
    cfg = GemmaConfig.gemma_300m(use_adarms=True)
    w = _expert_block_weights(cfg, torch.Generator().manual_seed(0), zero_adarms=True)
    block = GemmaBlock(cfg, w, layer_idx=0)

    seq = 4
    cos, sin = precompute_freqs_cis(cfg.head_dim, seq, cfg.rope_base)
    x = torch.randn(1, seq, cfg.width)
    cond = torch.randn(1, cfg.adarms_cond_dim)

    out, _ = block.forward(x, cos, sin, adarms_cond=cond)
    assert torch.allclose(out, x, atol=1e-5)


def test_gemma_expert_block_nonzero_adarms_changes_output():
    cfg = GemmaConfig.gemma_300m(use_adarms=True)
    w_zero = _expert_block_weights(cfg, torch.Generator().manual_seed(0), zero_adarms=True)
    w_rand = _expert_block_weights(cfg, torch.Generator().manual_seed(0), zero_adarms=False)

    seq = 4
    cos, sin = precompute_freqs_cis(cfg.head_dim, seq, cfg.rope_base)
    x = torch.randn(1, seq, cfg.width)
    cond = torch.randn(1, cfg.adarms_cond_dim)

    out_zero, _ = GemmaBlock(cfg, w_zero, 0).forward(x, cos, sin, adarms_cond=cond)
    out_rand, _ = GemmaBlock(cfg, w_rand, 0).forward(x, cos, sin, adarms_cond=cond)
    assert not torch.allclose(out_zero, out_rand)
