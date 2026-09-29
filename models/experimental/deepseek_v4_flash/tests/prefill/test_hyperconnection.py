# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""PCC tests for the prefill glue around the decoder layer's sublayers.

* ``test_prefill_hyperconnection`` -- ``DeepSeekV4PrefillHyperConnection`` against the reference
  ``DeepseekV4HyperConnection`` (fp32, CPU; a standalone copy of the HF modeling code from
  ``models.demos.deepseek_v3_d_p.reference.deepseek_v4``) with randomised ``fn`` / ``base`` / ``scale``
  at the real hidden size: the ``(post, comb, collapsed)`` triple, at several token counts.
* ``test_prefill_mix_streams`` -- the residual fold ``post * out + comb.T @ streams`` against torch,
  including the ``[1, 1, T, D]`` -> per-token layout change the decoder layer does on a sublayer output.

Run::

    pytest -s models/experimental/deepseek_v4_flash/tests/prefill/test_hyperconnection.py
"""

from __future__ import annotations

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_d_p.reference.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
from models.demos.deepseek_v3_d_p.reference.deepseek_v4.modeling_deepseek_v4 import DeepseekV4HyperConnection
from models.experimental.deepseek_v4_flash.tt.prefill.decoder_layer import mix_streams
from models.experimental.deepseek_v4_flash.tt.prefill.hyperconnection import DeepSeekV4PrefillHyperConnection

_SEED = 1234
_HIDDEN = 4096
_HC = 4
_HC_PCC = 0.99
_MIX_PCC = 0.999


def _config() -> DeepseekV4Config:
    return DeepseekV4Config(
        hidden_size=_HIDDEN,
        num_hidden_layers=1,
        layer_types=["sliding_attention"],
        mlp_layer_types=["moe"],
    )


def _assert_pcc(expected: torch.Tensor, actual: torch.Tensor, floor: float, what: str) -> None:
    expected, actual = expected.to(torch.float32), actual.reshape(expected.shape).to(torch.float32)
    passing, message = comp_pcc(expected, actual, pcc=floor)
    logger.info(f"[{what}] PCC: {message}")
    assert passing, f"{what}: PCC below {floor}: {message}"


def _to_tt(t: torch.Tensor, device) -> ttnn.Tensor:
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)


def _to_host(t: ttnn.Tensor) -> torch.Tensor:
    return ttnn.to_torch(t).to(torch.float32)


# Token counts: a partial tile row of ``fused_w`` (40), a whole number of them (128), and enough to
# make every core of the op loop over several tokens (1024).
@pytest.mark.parametrize("seq_len", (40, 128, 1024))
def test_prefill_hyperconnection(device, reset_seeds, seq_len):
    torch.manual_seed(_SEED + seq_len)
    cfg = _config()
    module = DeepseekV4HyperConnection(cfg).eval()
    with torch.no_grad():
        # Random rather than the model's init (base 0, scale 1) so every term of the mapping matters.
        module.fn.normal_(0.0, 0.02)
        module.base.normal_(0.0, 0.1)
        module.scale.uniform_(0.5, 1.5)
        for p in module.parameters():
            p.copy_(p.to(torch.bfloat16).to(torch.float32))
    streams = torch.randn(1, seq_len, _HC, _HIDDEN).to(torch.bfloat16).to(torch.float32)
    with torch.no_grad():
        ref_post, ref_comb, ref_collapsed = module(streams)

    hc = DeepSeekV4PrefillHyperConnection(cfg, dict(module.state_dict()), device)
    post, comb, collapsed = hc(_to_tt(streams, device))

    assert tuple(post.shape) == (1, seq_len, _HC, 1)
    assert tuple(comb.shape) == (1, seq_len, _HC, _HC)
    assert tuple(collapsed.shape) == (1, seq_len, 1, _HIDDEN)
    _assert_pcc(ref_post, _to_host(post), _HC_PCC, f"hc post, T={seq_len}")
    _assert_pcc(ref_comb, _to_host(comb), _HC_PCC, f"hc comb, T={seq_len}")
    _assert_pcc(ref_collapsed, _to_host(collapsed), _HC_PCC, f"hc collapsed, T={seq_len}")


def test_prefill_hyperconnection_rejects_bad_streams(device, reset_seeds, expect_error):
    cfg = _config()
    module = DeepseekV4HyperConnection(cfg).eval()
    with torch.no_grad():
        module.fn.normal_(0.0, 0.02)
        module.base.zero_()
        module.scale.fill_(1.0)
    hc = DeepSeekV4PrefillHyperConnection(cfg, dict(module.state_dict()), device)
    with expect_error(ValueError, "expected streams"):
        hc(_to_tt(torch.randn(1, 64, _HC + 1, _HIDDEN), device))
    with expect_error(ValueError, "at least 2 tokens"):
        hc(_to_tt(torch.randn(1, 1, _HC, _HIDDEN), device))


@pytest.mark.parametrize("seq_len", (128, 1024))
def test_prefill_mix_streams(device, reset_seeds, seq_len):
    """``post * out + comb.T @ streams`` with the sublayer output in its ``[1, 1, T, D]`` form."""
    torch.manual_seed(_SEED + seq_len)
    post = 2.0 * torch.rand(1, seq_len, _HC, 1)
    comb = torch.softmax(torch.randn(1, seq_len, _HC, _HC), dim=-1)  # not doubly stochastic; any comb will do
    out = torch.randn(1, 1, seq_len, _HIDDEN)
    streams = torch.randn(1, seq_len, _HC, _HIDDEN)
    # The kernel is bf16, so the reference is fed the same rounded inputs.
    post, comb, out, streams = (t.to(torch.bfloat16).to(torch.float32) for t in (post, comb, out, streams))
    expected = post * out.reshape(1, seq_len, 1, _HIDDEN) + comb.transpose(-1, -2) @ streams

    actual = mix_streams(_to_tt(post, device), _to_tt(comb, device), _to_tt(out, device), _to_tt(streams, device))
    assert tuple(actual.shape) == (1, seq_len, _HC, _HIDDEN)
    _assert_pcc(expected, _to_host(actual), _MIX_PCC, f"mix_streams, T={seq_len}")
