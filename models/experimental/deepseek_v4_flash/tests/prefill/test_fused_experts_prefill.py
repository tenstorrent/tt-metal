# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``ttnn.experimental.deepseek_prefill.fused_experts_prefill``.

The op takes the decode ``fused_experts`` routing arguments (``routing_scores`` plus ``routing_indices``
or ``ranking_scores``, ``top_k``, ``routed_scaling_factor``, ``routing_eps``), a row-major DRAM
``x_tok`` and the DECODE weight layout (bf4, DRAM ND-sharded, gate/up interleaved per 32 columns). Every
one of the 120 worker cores computes the full routing on device; expert ``e`` is owned by group
``e % 15`` (8 cores), which gathers the expert's token rows, tilizes them and runs the FFN. The output is
the routing-weighted sum over each token's top_k experts, [1, 1, T, H].

The reference is a plain torch implementation, per checked token ``t`` with selected experts ``S``:

    w[t, e]  = bf16(routed_scaling_factor * s[t, e] / (sum_{S} s[t, .] + eps))
    act_e    = silu(min(gate, limit)) * clamp(up, -limit, limit),   [gate, up] = x_t @ gate_up_e
    out[t]   = sum_{e in S} (w[t, e] * act_e) @ down_e

using the *dequantized* weights and x (host bf4 / bf8 round trips) and rounding ``w * act`` through bf8
because the op gathers it in bf8. Only the experts selected by the checked tokens keep their host
weights; all the others share one uploaded pair (their outputs feed only unchecked tokens).

Routing scores are built with DISTINCT bf16-exact values (a permutation of 1..E, over 256) with the
chosen experts holding the largest ones, so the top-k is unambiguous for the ranking path too.

Run::

    pytest -s models/experimental/deepseek_v4_flash/tests/prefill/test_fused_experts_prefill.py
"""

from __future__ import annotations

import random

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.experimental.deepseek_v4_flash.tt.decode.moe import _fused_nd_dram_config, _interleave_gate_up

_HIDDEN = 4096
_INTER = 512  # I_local: TP=4 slice of the real I = 2048
_TILE = 32
_LIMIT = 10.0
_SCALING = 1.5
_EPS = 1e-20
_PCC = 0.99


def _quantize(t: torch.Tensor, dtype) -> torch.Tensor:
    """Host round trip through a block-float dtype -> the values the device actually sees."""
    return ttnn.to_torch(ttnn.from_torch(t.float(), dtype=dtype, layout=ttnn.TILE_LAYOUT)).float()


def _quantize_rows(t: torch.Tensor, dtype) -> torch.Tensor:
    """Like ``_quantize`` for an [n, D] matrix with any n (block exponents are per row of 16)."""
    n = t.shape[0]
    pad = (-n) % _TILE
    return _quantize(torch.nn.functional.pad(t.float(), (0, 0, 0, pad)), dtype)[:n]


def _make_routing(num_tokens: int, num_experts: int, top_k: int, hot: list[int], rng: random.Random):
    """Per token: ``top_k`` distinct experts (each hot expert included with prob 0.9) and a score row
    of distinct bf16-exact values in which the selected experts hold the top ``top_k`` values."""
    sel = []
    scores = torch.zeros(num_tokens, num_experts)
    for t in range(num_tokens):
        chosen = [e for e in hot if rng.random() < 0.9][:top_k]
        rest = [e for e in range(num_experts) if e not in chosen]
        chosen += rng.sample(rest, top_k - len(chosen))
        rng.shuffle(chosen)
        others = [e for e in range(num_experts) if e not in chosen]
        low = list(range(1, num_experts - top_k + 1))
        rng.shuffle(low)
        for e, v in zip(others, low):
            scores[t, e] = v
        for j, e in enumerate(chosen):
            scores[t, e] = num_experts - j
        sel.append(chosen)
    return sel, scores / 256.0  # <= 256 distinct integers are exact in bf16; /256 is exact


def _tile_tensor(device, t: torch.Tensor, dtype):
    return ttnn.from_torch(
        t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def _run_case(
    device,
    num_tokens: int,
    num_experts: int,
    top_k: int,
    hot: list[int] = (),
    path: str = "ranking",
    check_every: int = 1,
    x_scale: float = 1.0,
):
    """Run the op and compare tokens ``t % check_every == 0`` (and the last one) to torch."""
    rng = random.Random(0)
    torch.manual_seed(0)
    dram_banks = device.dram_grid_size().x

    sel, scores = _make_routing(num_tokens, num_experts, top_k, list(hot), rng)
    x = (torch.randn(num_tokens, _HIDDEN) * x_scale).to(torch.bfloat16)

    checked_tokens = [t for t in range(num_tokens) if t % check_every == 0 or t == num_tokens - 1]
    needed = sorted({e for t in checked_tokens for e in sel[t]})

    counts = [0] * num_experts
    for row in sel:
        for e in row:
            counts[e] += 1
    logger.info(
        f"T={num_tokens} E={num_experts} k={top_k}: max tokens on one expert {max(counts)}, checking {len(needed)} experts"
    )

    # ---- weights: quantize on the host, upload the interleaved copy in the decode layout ----
    gate_up_nd = _fused_nd_dram_config(_HIDDEN, 2 * _INTER, 2 * _TILE, dram_banks)
    down_nd = _fused_nd_dram_config(_INTER, _HIDDEN, _HIDDEN // 64, dram_banks)
    gate_up_q, down_q, gate_up_tt, down_tt = {}, {}, [], []
    dummy = None

    def _upload(gate, up, down):
        gu_q = _quantize(torch.cat([gate, up], dim=1), ttnn.bfloat4_b)
        dn_q = _quantize(down, ttnn.bfloat4_b)
        gu_t = ttnn.from_torch(
            _interleave_gate_up(gu_q, _TILE),
            dtype=ttnn.bfloat4_b,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=gate_up_nd,
        )
        dn_t = ttnn.from_torch(
            dn_q, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=down_nd
        )
        return gu_q, dn_q, gu_t, dn_t

    needed_set = set(needed)
    for e in range(num_experts):
        if e not in needed_set and dummy is not None:
            gate_up_tt.append(dummy[0])
            down_tt.append(dummy[1])
            continue
        gate = torch.randn(_HIDDEN, _INTER) / _HIDDEN**0.5
        up = torch.randn(_HIDDEN, _INTER) / _HIDDEN**0.5
        down = torch.randn(_INTER, _HIDDEN) / _INTER**0.5
        gu_q, dn_q, gu_t, dn_t = _upload(gate, up, down)
        if e in needed_set:
            gate_up_q[e], down_q[e] = gu_q, dn_q
        else:
            dummy = (gu_t, dn_t)
        gate_up_tt.append(gu_t)
        down_tt.append(dn_t)

    x_tt = ttnn.from_torch(
        x.reshape(1, 1, num_tokens, _HIDDEN),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    scores_tt = _tile_tensor(device, scores.reshape(1, 1, num_tokens, num_experts), ttnn.bfloat16)
    indices_tt = ranking_tt = None
    if path == "ranking":
        ranking_tt = scores_tt
    elif path == "indices":
        ids = torch.tensor(sel, dtype=torch.int32).reshape(1, 1, num_tokens, top_k)
        indices_tt = _tile_tensor(device, ids, ttnn.uint16)
    elif path == "indices_bf16":
        ids = torch.tensor(sel, dtype=torch.float32).reshape(1, 1, num_tokens, top_k)
        indices_tt = _tile_tensor(device, ids, ttnn.bfloat16)
    else:
        raise ValueError(path)

    out_tt = ttnn.experimental.deepseek_prefill.fused_experts_prefill(
        x_tt,
        scores_tt,
        gate_up_tt,
        down_tt,
        _INTER,
        _LIMIT,
        top_k,
        _SCALING,
        _EPS,
        routing_indices=indices_tt,
        ranking_scores=ranking_tt,
    )
    out = ttnn.to_torch(out_tt).reshape(num_tokens, _HIDDEN).float()

    # ---- reference, per checked token ----
    ref_rows, got_rows = [], []
    x_q = _quantize_rows(x[checked_tokens].float(), ttnn.bfloat8_b)  # what the tilize + bf8 x block sees
    for i, t in enumerate(checked_tokens):
        s_sel = torch.tensor([scores[t, e] for e in sel[t]], dtype=torch.float32)
        w = (s_sel * (_SCALING / (s_sel.sum() + _EPS))).to(torch.bfloat16).float()
        ref = torch.zeros(_HIDDEN)
        for j, e in enumerate(sel[t]):
            gate_up = x_q[i : i + 1] @ gate_up_q[e]
            gate, up = gate_up[:, :_INTER], gate_up[:, _INTER:]
            act = torch.nn.functional.silu(gate.clamp(max=_LIMIT)) * up.clamp(-_LIMIT, _LIMIT)
            act_q = _quantize_rows(act * w[j], ttnn.bfloat8_b)  # the op gathers (w * act) in bf8
            ref += (act_q @ down_q[e])[0]
        ref_rows.append(ref)
        got_rows.append(out[t])
        passed, msg = comp_pcc(ref, out[t], _PCC)
        assert passed, f"token {t} (experts {sel[t]}): {msg}"
    _, msg = comp_pcc(torch.stack(ref_rows), torch.stack(got_rows), _PCC)
    logger.info(f"overall: {msg}")


# (T, E, top_k, hot experts, path). Covers: one tile row (short M-block), fewer experts than core groups,
# several experts per group (e, e + 15, ...), odd numbers of tile rows, a hot expert that every token
# selects (T tokens on one expert -> 2 chunks for T = 480), and the decode index formats.
_CASES = {
    "t32_e16_k2_ranking": (32, 16, 2, [], "ranking"),
    "t32_e16_k2_indices": (32, 16, 2, [], "indices"),
    "t64_e40_k4_indices_bf16": (64, 40, 4, [], "indices_bf16"),
    "t96_e32_k6_hot0": (96, 32, 6, [0, 17], "ranking"),
    "t480_e64_k6_hot": (480, 64, 6, [3, 21], "ranking"),
}


@pytest.mark.parametrize("case", list(_CASES), ids=list(_CASES))
def test_fused_experts_prefill(device, reset_seeds, case):
    num_tokens, num_experts, top_k, hot, path = _CASES[case]
    _run_case(device, num_tokens, num_experts, top_k, hot, path, check_every=1 if num_tokens <= 96 else 13)


def test_fused_experts_prefill_clamp(device, reset_seeds):
    """Large activations so gate is clamped at the limit and up at +-limit."""
    _run_case(device, 64, 32, 4, [], "ranking", x_scale=8.0)


def test_fused_experts_prefill_256_experts_480_tokens(device, reset_seeds):
    """Full scale: 480 tokens, 256 experts, top-6 (2880 (token, expert) pairs, ~11 tokens per expert)."""
    _run_case(device, 480, 256, 6, [], "ranking", check_every=29)


def test_fused_experts_prefill_256_experts_480_tokens_hot(device, reset_seeds):
    """Full scale with three hot experts that ~90% of the tokens select (chunked, unbalanced groups)."""
    _run_case(device, 480, 256, 6, [5, 100, 255], "indices", check_every=29)
