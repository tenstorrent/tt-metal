# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only: the first token Kimi-K3's prefill hands the decode side (PREFILL_FIRST_TOKEN_DIR).

The decode side resumes migrated KDA state at prompt_len and cannot derive the first token itself
(re-running prompt_len-1 would overwrite that state), so prefill computes it: the last rank returns its
output AttnRes read without the device final norm, and the adapter applies `model.norm` and `lm_head` on
the host.
"""

from types import SimpleNamespace

import pytest
import torch

from models.demos.common.prefill.adapter import PrefillModelAdapter
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config
from models.demos.deepseek_v3_d_p.tt.kimi_k3 import runtime as k3_runtime
from models.demos.deepseek_v3_d_p.tt.kimi_k3.transformer import TtKimiK3Transformer
from models.demos.deepseek_v3_d_p.tt.runners.adapters.kimi_k3 import KimiK3Adapter, host_norm_head_argmax
from models.demos.deepseek_v3_d_p.tt.tt_prefill_runtime import TtPrefillRuntime

HIDDEN, VOCAB = 64, 1000


def _reference(hidden, norm_weight, lm_head, eps):
    """`KimiRMSNorm` then `lm_head` as the checkpoint's modeling code writes them, logits in fp32."""
    x = hidden.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    x = norm_weight * x.to(norm_weight.dtype)
    return int(torch.argmax(lm_head.float() @ x.float()))


def _weights(seed=0, dtype=torch.bfloat16):
    g = torch.Generator().manual_seed(seed)
    norm_weight = (1 + 0.1 * torch.randn(HIDDEN, generator=g)).to(dtype)
    lm_head = torch.randn(VOCAB, HIDDEN, generator=g).to(dtype)
    hidden = 5 * torch.randn(HIDDEN, generator=g)
    return hidden, norm_weight, lm_head


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("vocab_block", [64, 333, VOCAB, 4 * VOCAB])
def test_blocked_head_matches_the_reference(seed, vocab_block):
    hidden, norm_weight, lm_head = _weights(seed)
    eps = KimiK3Config.RMS_NORM_EPS
    got = host_norm_head_argmax(hidden, norm_weight, lm_head, eps=eps, vocab_block=vocab_block)
    assert got == _reference(hidden, norm_weight, lm_head, eps)


@pytest.mark.parametrize("token", [0, 63, 64, VOCAB - 1])
def test_the_winner_is_found_in_any_block_including_the_partial_last(token):
    hidden, norm_weight, lm_head = _weights()
    lm_head[token] = 100 * torch.sign(hidden).to(lm_head.dtype)  # aligned with the row: the largest logit
    assert host_norm_head_argmax(hidden, norm_weight, lm_head, eps=1e-5, vocab_block=64) == token


def test_a_tie_across_blocks_takes_the_lowest_token_as_argmax_does():
    hidden, norm_weight, lm_head = _weights()
    lm_head[10] = lm_head[900] = 100 * torch.sign(hidden).to(lm_head.dtype)
    assert host_norm_head_argmax(hidden, norm_weight, lm_head, eps=1e-5, vocab_block=64) == 10


def test_only_an_adapter_that_implements_the_head_claims_it(expect_error):
    assert KimiK3Adapter.supports_host_first_token
    assert not PrefillModelAdapter.supports_host_first_token
    with expect_error(NotImplementedError, "does not compute the first token"):
        PrefillModelAdapter.first_token_from_hidden(SimpleNamespace(), torch.zeros(HIDDEN))


def test_the_head_comes_from_the_checkpoint_named_by_the_adapter(monkeypatch, expect_error):
    adapter = KimiK3Adapter.__new__(KimiK3Adapter)
    monkeypatch.delenv(KimiK3Adapter.env_var, raising=False)
    monkeypatch.delenv("PREFILL_HF_MODEL", raising=False)
    with expect_error(RuntimeError, KimiK3Adapter.env_var):
        adapter.first_token_from_hidden(torch.zeros(HIDDEN))


@pytest.mark.parametrize(
    "return_last_hidden, is_last_rank, expected",
    [
        (True, True, k3_runtime._TtKimiK3TransformerHostNorm),
        (True, False, TtKimiK3Transformer),
        (False, True, TtKimiK3Transformer),
    ],
)
def test_only_the_returning_last_rank_drops_the_device_norm(monkeypatch, return_last_hidden, is_last_rank, expected):
    built = []
    monkeypatch.setattr(TtPrefillRuntime, "_build_model", lambda self, state_dict: built.append(self.MODEL_CLS))
    runtime = object.__new__(k3_runtime.TtKimiK3Runtime)
    runtime.config = SimpleNamespace(mtp_levels=0, return_last_hidden=return_last_hidden, is_last_rank=is_last_rank)
    runtime._build_model({})
    assert built == [expected]


def test_the_host_norm_stack_builds_no_tail(monkeypatch):
    seen = {}
    monkeypatch.setattr(TtKimiK3Transformer, "__init__", lambda self, *a, **kw: seen.update(kw))
    k3_runtime._TtKimiK3TransformerHostNorm(num_layers=1)
    assert seen == {"build_tail": False, "num_layers": 1}
    # Whatever the cache holds, the subclass keeps the parent's completeness check.
    assert k3_runtime._TtKimiK3TransformerHostNorm.check_cache_complete is TtKimiK3Transformer.check_cache_complete


def test_return_last_hidden_refuses_a_trimmed_or_extended_last_layer(expect_error):
    runtime = object.__new__(TtPrefillRuntime)
    for flag in ("kv_only_last_layer", "mtp_levels", "dflash_enabled"):
        config = dict(return_last_hidden=True, kv_only_last_layer=False, mtp_levels=0, dflash_enabled=False)
        config[flag] = True
        runtime.config = SimpleNamespace(**config)
        with expect_error(ValueError, "return_last_hidden"):
            runtime._build_model({})
