# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""MiniMax-style memory tiling for GPT-OSS HF golden prefill.

GPT-OSS sets ``_supports_sdpa = False`` because attention sinks need the
concat-then-softmax path (an extra virtual key that absorbs probability mass).
Stock eager attention therefore materializes full ``[H, S, S]`` scores — at
S≈55k / Hq=64 / bf16 that alone is ~387 GB and OOMs even on 500GB+ hosts.

These replacements keep **one-shot** prefill math (same causal / sliding-window
mask + sinks + MoE top-k as a single forward) while never allocating the fat
score / expert-activation tensors:

* ``tiled_eager_attention_forward`` — loop query rows in ``ATTN_Q_CHUNK``;
  softmax over full active keys (+ sink column); delete scores each chunk.
  Exact per-row softmax ⇒ bit-identical to unchunked eager.
* ``tiled_experts_forward`` — within each hit expert, process tokens in
  ``FFN_TOKEN_CHUNK`` so ``[N, 2*intermediate]`` never fully materializes.

Install via :func:`install_tiled_ops` before the HF forward. Env knobs match
MiniMax: ``REF_ATTN_Q_CHUNK`` (default 256), ``REF_FFN_TOKEN_CHUNK`` (default 4096).
"""

from __future__ import annotations

import os
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

ATTN_Q_CHUNK = int(os.environ.get("REF_ATTN_Q_CHUNK", "256"))
FFN_TOKEN_CHUNK = int(os.environ.get("REF_FFN_TOKEN_CHUNK", "4096"))


def tiled_eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float | int = 0.0,
    **kwargs,
):
    """Drop-in for HF ``eager_attention_forward`` with query-row tiling.

    Matches transformers ``modeling_gpt_oss.eager_attention_forward`` exactly
    (repeat_kv → scaled QK → optional mask → concat sinks → max-subtract softmax
    → drop sink column → dropout → @V), but scores peak at
    ``[B, H, ATTN_Q_CHUNK, S]`` instead of ``[B, H, S, S]``.
    """
    # Local import keeps this module usable without transformers at import time
    # for lightweight unit checks of the tiling helpers.
    from transformers.models.gpt_oss.modeling_gpt_oss import repeat_kv

    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    batch, num_heads, seq_q, head_dim = query.shape
    assert dropout == 0.0 or not module.training, "tiled eager golden path expects dropout=0"

    attn_output = torch.empty(batch, num_heads, seq_q, head_dim, dtype=value_states.dtype, device=query.device)
    sinks = module.sinks.reshape(1, -1, 1, 1)  # [1, Hq, 1, 1]

    for qs in range(0, seq_q, ATTN_Q_CHUNK):
        qe = min(qs + ATTN_Q_CHUNK, seq_q)
        q = query[:, :, qs:qe, :]
        attn_weights = torch.matmul(q, key_states.transpose(2, 3)) * scaling  # [B, H, C, Sk]
        if attention_mask is not None:
            # HF masks are [B, 1, Sq, Sk] (or broadcastable); slice query rows only.
            attn_weights = attn_weights + attention_mask[:, :, qs:qe, :]

        sinks_exp = sinks.expand(batch, -1, qe - qs, -1)
        combined_logits = torch.cat([attn_weights, sinks_exp], dim=-1)
        # Matches HF: max-subtract before softmax (bf16 overflow guard).
        combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values
        probs = F.softmax(combined_logits, dim=-1, dtype=combined_logits.dtype)
        scores = probs[..., :-1]  # drop sink column
        attn_output[:, :, qs:qe, :] = torch.matmul(scores.to(value_states.dtype), value_states)
        del attn_weights, sinks_exp, combined_logits, probs, scores

    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, None


def tiled_experts_forward(self, hidden_states: torch.Tensor, router_indices=None, routing_weights=None) -> torch.Tensor:
    """Drop-in for ``GptOssExperts.forward`` with per-expert token tiling.

    Same sparse dispatch as HF (one_hot → hit experts → index_add_), but when an
    expert receives more than ``FFN_TOKEN_CHUNK`` tokens the gate_up / down matmuls
    run in token chunks so the ``[N, 2*intermediate]`` activation never fully
    materializes. Chunking is along tokens only ⇒ numerically identical.
    """
    next_states = torch.zeros_like(hidden_states, dtype=hidden_states.dtype, device=hidden_states.device)
    with torch.no_grad():
        expert_mask = torch.nn.functional.one_hot(router_indices, num_classes=self.num_experts)
        expert_mask = expert_mask.permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

    for expert_idx in expert_hit:
        expert_idx = expert_idx[0]
        if expert_idx == self.num_experts:
            continue
        top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
        n = token_idx.numel()
        if n == 0:
            continue

        if n <= FFN_TOKEN_CHUNK:
            current_state = hidden_states[token_idx]
            gate_up = current_state @ self.gate_up_proj[expert_idx] + self.gate_up_proj_bias[expert_idx]
            gated_output = self._apply_gate(gate_up)
            out = gated_output @ self.down_proj[expert_idx] + self.down_proj_bias[expert_idx]
            weighted_output = out * routing_weights[token_idx, top_k_pos, None]
            next_states.index_add_(0, token_idx, weighted_output.to(hidden_states.dtype))
        else:
            for i in range(0, n, FFN_TOKEN_CHUNK):
                j = min(i + FFN_TOKEN_CHUNK, n)
                tok = token_idx[i:j]
                tk = top_k_pos[i:j]
                current_state = hidden_states[tok]
                gate_up = current_state @ self.gate_up_proj[expert_idx] + self.gate_up_proj_bias[expert_idx]
                gated_output = self._apply_gate(gate_up)
                out = gated_output @ self.down_proj[expert_idx] + self.down_proj_bias[expert_idx]
                weighted_output = out * routing_weights[tok, tk, None]
                next_states.index_add_(0, tok, weighted_output.to(hidden_states.dtype))
                del current_state, gate_up, gated_output, out, weighted_output

    return next_states


_STOCK_EAGER: Callable | None = None
_STOCK_EXPERTS: Callable | None = None
_INSTALLED = False


def install_tiled_ops() -> tuple[int, int]:
    """Monkey-patch HF GPT-OSS eager attention + experts for long one-shot prefill.

    Returns ``(ATTN_Q_CHUNK, FFN_TOKEN_CHUNK)`` actually installed. Idempotent.
    """
    global _STOCK_EAGER, _STOCK_EXPERTS, _INSTALLED
    import transformers.models.gpt_oss.modeling_gpt_oss as gpt_oss_mod

    if not _INSTALLED:
        _STOCK_EAGER = gpt_oss_mod.eager_attention_forward
        _STOCK_EXPERTS = gpt_oss_mod.GptOssExperts.forward
        gpt_oss_mod.eager_attention_forward = tiled_eager_attention_forward
        gpt_oss_mod.GptOssExperts.forward = tiled_experts_forward  # type: ignore[method-assign]
        _INSTALLED = True
    return ATTN_Q_CHUNK, FFN_TOKEN_CHUNK


def assert_tiled_matches_eager(
    *,
    seq_len: int = 128,
    num_heads: int = 8,
    num_kv_heads: int = 2,
    head_dim: int = 64,
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
) -> float:
    """Self-check: tiled vs stock HF eager on a tiny random attention problem.

    Must be called **before** :func:`install_tiled_ops`, or after install (uses the
    saved stock reference). Returns max abs diff (expect 0 for fp32).
    """
    import transformers.models.gpt_oss.modeling_gpt_oss as gpt_oss_mod

    stock = _STOCK_EAGER or gpt_oss_mod.eager_attention_forward
    # If install already replaced the module attr with tiled, and we somehow lost
    # the stock ref, comparing tiled-to-tiled is useless — require stock.
    if stock is tiled_eager_attention_forward:
        raise RuntimeError("stock eager unavailable; call assert_tiled_matches_eager before install_tiled_ops")

    g = torch.Generator().manual_seed(seed)
    B, S, Hq, Hkv, D = 1, seq_len, num_heads, num_kv_heads, head_dim
    groups = Hq // Hkv

    class _Mod(nn.Module):
        def __init__(self):
            super().__init__()
            self.num_key_value_groups = groups
            self.sinks = nn.Parameter(torch.randn(Hq, generator=g, dtype=dtype))
            self.training = False

    module = _Mod()
    q = torch.randn(B, Hq, S, D, generator=g, dtype=dtype)
    k = torch.randn(B, Hkv, S, D, generator=g, dtype=dtype)
    v = torch.randn(B, Hkv, S, D, generator=g, dtype=dtype)
    i = torch.arange(S).unsqueeze(1)
    j = torch.arange(S).unsqueeze(0)
    mask = torch.zeros(1, 1, S, S, dtype=dtype)
    mask.masked_fill_(j > i, float("-inf"))
    scaling = D**-0.5

    out_stock, _ = stock(module, q, k, v, mask, scaling, dropout=0.0)
    out_tiled, _ = tiled_eager_attention_forward(module, q, k, v, mask, scaling, dropout=0.0)
    return (out_stock - out_tiled).abs().max().item()


if __name__ == "__main__":
    # Single-chunk (seq <= ATTN_Q_CHUNK) and multi-chunk (partial last tile).
    cases = [
        ("single_chunk", 128),
        ("multi_chunk", ATTN_Q_CHUNK * 2 + 17),
    ]
    for name, seq_len in cases:
        diff = assert_tiled_matches_eager(seq_len=seq_len)
        print(f"tiled vs stock eager [{name} seq={seq_len}] max_diff={diff:.2e}")
        assert diff == 0.0, f"tiled attention not bit-identical to stock ({name}): {diff}"
    install_tiled_ops()
    print(f"installed tiling: ATTN_Q_CHUNK={ATTN_Q_CHUNK} FFN_TOKEN_CHUNK={FFN_TOKEN_CHUNK}")
    print("all checks passed")
