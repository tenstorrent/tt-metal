#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Smoke test for :class:`NextNSglangStructureDraftAdapter` without loading NextN MoE weights.

The full ``--sglang-draft-structure`` run needs large RAM/GPU; this test mocks the HF stack and
checks fusion → fake decoder → norm → lm_head wiring.

Run from repo root::

  ./python_env/bin/python models/demos/speculative_deepseek_r1_broad/scripts/test_nextn_sglang_structure_smoke.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class _FakeInner(nn.Module):
    def __init__(self, hidden: int, vocab: int) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.layers = nn.ModuleList([_FakeDecoderLayer()])

    def forward(self, *a, **k):  # pragma: no cover
        raise NotImplementedError


class _FakeDecoderLayer(nn.Module):
    """Mimics HF layer return: (hidden_states, (k, v)) for use_cache."""

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        use_cache: bool = False,
        **_,
    ):
        h = hidden_states + 0.01  # tiny nudge so path is non-trivial
        b, t, d = h.shape
        if use_cache:
            # [B, H, S, D]; _hf_past_kv_seq_len((k, v)) uses key.shape[2] as seq len
            k = torch.zeros(b, 2, t, 8, device=h.device, dtype=h.dtype)
            v = torch.zeros_like(k)
            return (h, (k, v))
        return (h,)


class _FakeCausalLM(nn.Module):
    """Same layout as HF CausalLM: ``.model`` = inner stack, ``.lm_head`` = vocab projection."""

    def __init__(self, hidden: int, vocab: int) -> None:
        super().__init__()
        self.model = _FakeInner(hidden, vocab)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)


class _HFAdapterShell:
    """Mimics :class:`NextNFullHuggingfaceDraftAdapter` for tests (only ``.model`` CausalLM is used)."""

    def __init__(self, causal_lm: nn.Module) -> None:
        self.model = causal_lm


def test_hf_past_kv_seq_len_single_layer_tuple() -> None:
    from models.demos.speculative_deepseek_r1_broad.nextn_full_layer_draft import _hf_past_kv_seq_len

    k = torch.zeros(1, 4, 3, 64)  # seq_len == 3
    v = torch.zeros_like(k)
    assert _hf_past_kv_seq_len((k, v)) == 3
    assert _hf_past_kv_seq_len(((k, v),)) == 3


def test_one_step_mocked() -> None:
    from models.demos.speculative_deepseek_r1_broad.nextn_sglang_structure_draft import NextNSglangStructureDraftAdapter

    torch.manual_seed(0)
    H, V = 32, 17
    dev = torch.device("cpu")
    ad = NextNSglangStructureDraftAdapter(device="cpu", torch_dtype="float32")
    fake = _FakeCausalLM(H, V).to(dev).float()
    ad._hf = _HFAdapterShell(fake)
    ad._eh_proj_w = torch.randn(H, 2 * H, device=dev)
    ad._enorm_w = torch.ones(H, device=dev)
    ad._hnorm_w = torch.ones(H, device=dev)
    ad._rms_eps = 1e-6
    ad._hidden_size = H
    ad.bound = True

    h_side = torch.randn(H, device=dev)
    logits, h_next, past = ad._one_step(h_side, token_id=3, past_key_values=None)
    assert logits.shape == (V,)
    assert h_next.shape == (H,)
    assert past is not None and len(past) == 2

    logits2, h_next2, past2 = ad._one_step(h_next, token_id=4, past_key_values=past)
    assert logits2.shape == (V,)
    assert h_next2.shape == (H,)
    assert past2 is not None


def main() -> int:
    print("NextNSglangStructureDraftAdapter mock smoke ...")
    test_hf_past_kv_seq_len_single_layer_tuple()
    print("OK: _hf_past_kv_seq_len single-layer (k,v) vs legacy ((k,v),)")
    test_one_step_mocked()
    print("OK: _one_step (past None + past tuple) passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
