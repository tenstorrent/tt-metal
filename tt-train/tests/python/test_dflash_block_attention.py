# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Parity tests for DFlash anchor-block attention on the square fused SDPA.

DFlash attention is rectangular: Q is ``n_anchors * block_size`` synthetic block
slots, and KV is ``[packed context (T) | all block slots]``. The fused tt-train SDPA
only supports square ``(1, 1, S, S)`` masks, so ``ttml.models.dflash`` runs it over
``X = [context ; blocks]`` with dummy context queries and slices the block rows back
out.

The golden here is the rectangular reference: a verbatim copy of the speculators
``create_anchor_block_mask_mod`` (speculators ``src/speculators/models/dflash/
attention.py``) materialized densely, run through torch SDPA over only the valid
anchors. Padded anchors (``anchor_valid == 0``) must not affect valid outputs and
must receive exactly zero gradient.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import ttnn
import ttml
from ttml.models.dflash import build_dflash_square_mask, dflash_square_attention


SEED = 2026
NUM_Q_HEADS = 64
NUM_KV_HEADS = 8
HEAD_DIM = 128
BLOCK_SIZE = 8


@dataclass(frozen=True)
class Case:
    name: str
    doc_lengths: tuple[int, ...]  # remainder of T is padding (document id -1)
    total_seq_len: int
    anchor_budget: int
    num_valid_anchors: int
    sliding_window: int | None = None
    sliding_window_non_causal: bool = False
    sliding_window_per_query: bool = False


SMALL = dict(doc_lengths=(100, 90, 50), total_seq_len=256, anchor_budget=16, num_valid_anchors=13)
MEDIUM = dict(doc_lengths=(700, 600, 500, 200), total_seq_len=2048, anchor_budget=128, num_valid_anchors=120)

CASES = [
    Case("small-full", **SMALL),
    Case("small-sw-causal-block", **SMALL, sliding_window=64),
    Case("small-sw-noncausal", **SMALL, sliding_window=64, sliding_window_non_causal=True),
    Case("small-sw-per-query", **SMALL, sliding_window=64, sliding_window_non_causal=True, sliding_window_per_query=True),
    Case("medium-full", **MEDIUM),
    Case("medium-sw", **MEDIUM, sliding_window=512, sliding_window_non_causal=True),
]
CASE_IDS = [c.name for c in CASES]


# ---------------------------------------------------------------------------
# Reference (rectangular) mask and attention
# ---------------------------------------------------------------------------


def _reference_mask(
    document_ids: torch.Tensor,
    anchor_positions: torch.Tensor,
    block_size: int,
    sliding_window: int | None,
    sliding_window_non_causal: bool,
    sliding_window_per_query: bool,
) -> torch.Tensor:
    """Dense bool [n*B, T + n*B] mask; logic copied from speculators create_anchor_block_mask_mod."""
    total_seq_len = document_ids.numel()
    non_causal = sliding_window is None or sliding_window_non_causal
    q_len = anchor_positions.numel() * block_size
    query_anchor_positions = torch.repeat_interleave(anchor_positions, block_size)

    q_idx = torch.arange(q_len).unsqueeze(1)
    kv_idx = torch.arange(total_seq_len + q_len).unsqueeze(0)

    # base_prefix_mod
    q_anchor = query_anchor_positions[q_idx]
    q_doc = document_ids[q_anchor]
    kv_is_base = kv_idx < total_seq_len
    kv_base_pos = torch.remainder(kv_idx, total_seq_len)
    kv_doc = document_ids[kv_base_pos]
    same_doc = (q_doc == kv_doc) & (q_doc != -1)
    before_anchor = kv_base_pos < q_anchor
    if sliding_window is None:
        in_window = True
    elif sliding_window_per_query:
        query_position = q_anchor + q_idx % block_size
        in_window = kv_base_pos >= query_position - sliding_window + 1
    else:
        in_window = kv_base_pos >= q_anchor - sliding_window
    base = kv_is_base & same_doc & before_anchor & in_window

    # same_block_mod
    q_block = q_idx // block_size
    kv_is_block = kv_idx >= total_seq_len
    kv_block = (kv_idx - total_seq_len) // block_size
    same = kv_is_block & (q_block == kv_block)
    if not non_causal:
        same = same & (kv_idx <= q_idx + total_seq_len)
    if sliding_window_per_query and sliding_window is not None:
        same = same & (kv_idx >= q_idx + total_seq_len - sliding_window + 1)

    return base | same


@dataclass
class Inputs:
    document_ids: np.ndarray  # [T] int64
    anchor_positions: np.ndarray  # [n] int64 (padded slots hold a dummy position)
    anchor_valid: np.ndarray  # [n] bool
    q_blk: torch.Tensor  # [1, Hq, n*B, D]
    k_ctx: torch.Tensor  # [1, Hkv, T, D]
    v_ctx: torch.Tensor
    k_blk: torch.Tensor  # [1, Hkv, n*B, D]
    v_blk: torch.Tensor
    grad_out: torch.Tensor  # [1, Hq, n*B, D], zero on padded anchors


def _bf16(x: torch.Tensor) -> torch.Tensor:
    """Round to bf16 and back so the reference sees exactly the device inputs."""
    return x.to(torch.bfloat16).to(torch.float32)


def _make_inputs(case: Case) -> Inputs:
    rng = np.random.default_rng(SEED)
    T, B = case.total_seq_len, BLOCK_SIZE
    document_ids = np.full(T, -1, dtype=np.int64)
    start = 0
    for doc, length in enumerate(case.doc_lengths):
        document_ids[start : start + length] = doc
        start += length
    assert start <= T

    # Eligible anchors: the whole block fits inside one (non-padding) document.
    eligible = [
        p for p in range(T - B + 1) if document_ids[p] != -1 and document_ids[p + B - 1] == document_ids[p]
    ]
    valid = rng.choice(eligible, size=case.num_valid_anchors, replace=False)
    n_pad = case.anchor_budget - case.num_valid_anchors
    # Padded slots point at a real position on purpose: the mask must ignore them anyway.
    anchor_positions = np.concatenate([valid, np.full(n_pad, valid[0])]).astype(np.int64)
    anchor_valid = np.concatenate([np.ones(case.num_valid_anchors, bool), np.zeros(n_pad, bool)])
    # Interleave padded slots instead of leaving them all at the end.
    perm = rng.permutation(case.anchor_budget)
    anchor_positions, anchor_valid = anchor_positions[perm], anchor_valid[perm]

    g = torch.Generator().manual_seed(SEED)
    nB = case.anchor_budget * B

    def randn(*shape):
        return _bf16(torch.randn(*shape, generator=g))

    grad_out = randn(1, NUM_Q_HEADS, nB, HEAD_DIM)
    row_valid = torch.from_numpy(np.repeat(anchor_valid, B))
    grad_out[:, :, ~row_valid] = 0.0
    return Inputs(
        document_ids=document_ids,
        anchor_positions=anchor_positions,
        anchor_valid=anchor_valid,
        q_blk=randn(1, NUM_Q_HEADS, nB, HEAD_DIM),
        k_ctx=randn(1, NUM_KV_HEADS, T, HEAD_DIM),
        v_ctx=randn(1, NUM_KV_HEADS, T, HEAD_DIM),
        k_blk=randn(1, NUM_KV_HEADS, nB, HEAD_DIM),
        v_blk=randn(1, NUM_KV_HEADS, nB, HEAD_DIM),
        grad_out=grad_out,
    )


def _valid_rows(inputs: Inputs) -> torch.Tensor:
    return torch.from_numpy(np.repeat(inputs.anchor_valid, BLOCK_SIZE))


def _reference(case: Case, inputs: Inputs) -> dict[str, torch.Tensor]:
    """Rectangular DFlash attention over the valid anchors only, in fp32.

    Returns the output and gradients scattered back to the full anchor-budget layout
    (zeros on padded anchors).
    """
    rows = _valid_rows(inputs)
    anchors = torch.from_numpy(inputs.anchor_positions[inputs.anchor_valid])
    mask = _reference_mask(
        torch.from_numpy(inputs.document_ids),
        anchors,
        BLOCK_SIZE,
        case.sliding_window,
        case.sliding_window_non_causal,
        case.sliding_window_per_query,
    )
    q = inputs.q_blk[:, :, rows].clone().requires_grad_(True)
    k_ctx = inputs.k_ctx.clone().requires_grad_(True)
    v_ctx = inputs.v_ctx.clone().requires_grad_(True)
    k_blk = inputs.k_blk[:, :, rows].clone().requires_grad_(True)
    v_blk = inputs.v_blk[:, :, rows].clone().requires_grad_(True)
    out = F.scaled_dot_product_attention(
        q, torch.cat([k_ctx, k_blk], dim=2), torch.cat([v_ctx, v_blk], dim=2), attn_mask=mask, enable_gqa=True
    )
    out.backward(inputs.grad_out[:, :, rows])

    def scatter(t: torch.Tensor) -> torch.Tensor:
        full = torch.zeros(t.shape[0], t.shape[1], rows.numel(), t.shape[3])
        full[:, :, rows] = t.detach()
        return full

    return {
        "out": scatter(out),
        "dq_blk": scatter(q.grad),
        "dk_ctx": k_ctx.grad,
        "dv_ctx": v_ctx.grad,
        "dk_blk": scatter(k_blk.grad),
        "dv_blk": scatter(v_blk.grad),
    }


def _square_mask(case: Case, inputs: Inputs) -> np.ndarray:
    return build_dflash_square_mask(
        inputs.document_ids,
        inputs.anchor_positions,
        inputs.anchor_valid,
        BLOCK_SIZE,
        sliding_window=case.sliding_window,
        sliding_window_non_causal=case.sliding_window_non_causal,
        sliding_window_per_query=case.sliding_window_per_query,
    )


def _assert_close(name: str, actual: torch.Tensor, expected: torch.Tensor, rel_limit: float, cos_limit: float) -> None:
    a = actual.reshape(-1).double()
    e = expected.reshape(-1).double()
    rel = float((a - e).norm() / (e.norm() + 1e-12))
    cos = float(torch.dot(a, e) / (a.norm() * e.norm() + 1e-12))
    assert rel <= rel_limit and cos >= cos_limit, f"{name}: rel_err={rel:.3e} (<= {rel_limit}), cos={cos:.6f} (>= {cos_limit})"


def _assert_padded_rows_zero(name: str, t: torch.Tensor, inputs: Inputs) -> None:
    pad = ~_valid_rows(inputs)
    if pad.any():
        worst = float(t[:, :, pad].abs().max())
        assert worst == 0.0, f"{name}: padded-anchor rows must get exactly zero, max |x| = {worst}"


# ---------------------------------------------------------------------------
# Host tier: mask construction + square-padding math (no device)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_square_mask_matches_reference(case: Case):
    """Block rows of the square mask equal the reference mask; other rows are self-only."""
    inputs = _make_inputs(case)
    T, nB = case.total_seq_len, case.anchor_budget * BLOCK_SIZE
    mask = np.asarray(_square_mask(case, inputs))
    assert mask.shape == (T + nB, T + nB)
    assert mask.shape[0] % 32 == 0
    assert mask.dtype == np.bool_

    # Every row attends to at least one key, so the softmax stays finite.
    assert mask.any(axis=1).all()

    # Context rows and padded-anchor rows attend only to themselves.
    rows = _valid_rows(inputs).numpy()
    dummy = np.concatenate([np.ones(T, bool), ~rows])
    np.testing.assert_array_equal(mask[dummy], np.eye(T + nB, dtype=bool)[dummy])

    # Valid block rows equal the reference mask, remapped onto the full anchor budget.
    ref = _reference_mask(
        torch.from_numpy(inputs.document_ids),
        torch.from_numpy(inputs.anchor_positions),
        BLOCK_SIZE,
        case.sliding_window,
        case.sliding_window_non_causal,
        case.sliding_window_per_query,
    ).numpy()
    np.testing.assert_array_equal(mask[T:][rows], ref[rows])


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_square_padding_matches_reference_torch(case: Case):
    """The square-padding transform is exact in fp32 (torch SDPA stands in for the kernel)."""
    inputs = _make_inputs(case)
    ref = _reference(case, inputs)
    T = case.total_seq_len
    mask = torch.from_numpy(np.asarray(_square_mask(case, inputs)))

    q_blk = inputs.q_blk.clone().requires_grad_(True)
    k_ctx = inputs.k_ctx.clone().requires_grad_(True)
    v_ctx = inputs.v_ctx.clone().requires_grad_(True)
    k_blk = inputs.k_blk.clone().requires_grad_(True)
    v_blk = inputs.v_blk.clone().requires_grad_(True)
    q = torch.cat([torch.zeros(1, NUM_Q_HEADS, T, HEAD_DIM), q_blk], dim=2)
    out = F.scaled_dot_product_attention(
        q, torch.cat([k_ctx, k_blk], dim=2), torch.cat([v_ctx, v_blk], dim=2), attn_mask=mask, enable_gqa=True
    )[:, :, T:]
    out.backward(inputs.grad_out)

    got = {
        "dq_blk": q_blk.grad,
        "dk_ctx": k_ctx.grad,
        "dv_ctx": v_ctx.grad,
        "dk_blk": k_blk.grad,
        "dv_blk": v_blk.grad,
    }
    rows = _valid_rows(inputs)
    _assert_close("out", out.detach()[:, :, rows], ref["out"][:, :, rows], 1e-5, 0.99999)
    for name, grad in got.items():
        _assert_close(name, grad, ref[name], 1e-5, 0.99999)
    for name in ("dq_blk", "dk_blk", "dv_blk"):
        _assert_padded_rows_zero(name, got[name], inputs)


# ---------------------------------------------------------------------------
# Device tier: ttml fused SDPA
# ---------------------------------------------------------------------------


def _to_ttml(x: torch.Tensor, requires_grad: bool = True) -> ttml.autograd.Tensor:
    t = ttml.autograd.Tensor.from_numpy(
        x.numpy().astype(np.float32), layout=ttnn.Layout.TILE, new_type=ttnn.DataType.BFLOAT16
    )
    t.set_requires_grad(requires_grad)
    return t


def _value(t: ttml.autograd.Tensor) -> torch.Tensor:
    return torch.from_numpy(np.asarray(t.to_numpy(ttnn.DataType.FLOAT32), dtype=np.float32))


def _grad(t: ttml.autograd.Tensor) -> torch.Tensor:
    return torch.from_numpy(np.asarray(t.get_grad_tensor().to_numpy(ttnn.DataType.FLOAT32), dtype=np.float32))


@pytest.fixture
def device_ctx():
    ctx = ttml.autograd.AutoContext.get_instance()
    ctx.get_device()
    ctx.set_gradient_mode(ttml.autograd.GradMode.ENABLED)
    yield ctx
    ctx.reset_graph()
    ctx.close_device()


@pytest.mark.requires_device
@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_dflash_square_attention_matches_reference_on_device(case: Case, device_ctx):
    """Forward output and dQ/dK/dV of the fused-SDPA path match the rectangular reference."""
    inputs = _make_inputs(case)
    ref = _reference(case, inputs)

    mask = _square_mask(case, inputs)
    mask_t = _to_ttml(torch.from_numpy(np.asarray(mask, dtype=np.float32)).reshape(1, 1, *mask.shape), False)
    q_blk, k_ctx, v_ctx, k_blk, v_blk = (
        _to_ttml(t) for t in (inputs.q_blk, inputs.k_ctx, inputs.v_ctx, inputs.k_blk, inputs.v_blk)
    )

    out = dflash_square_attention(q_blk, k_ctx, v_ctx, k_blk, v_blk, mask_t)
    assert list(out.get_value().shape) == list(inputs.q_blk.shape)

    # mean(out * (numel * dO)) == sum(out * dO): upstream gradient is exactly grad_out.
    numel = inputs.grad_out.numel()
    loss = ttml.ops.unary.mean(ttml.ops.binary.mul(out, _to_ttml(inputs.grad_out * numel, False)))
    loss.backward(False)
    ttnn.synchronize_device(device_ctx.get_device())

    rows = _valid_rows(inputs)
    _assert_close("out", _value(out)[:, :, rows], ref["out"][:, :, rows], 0.02, 0.999)
    got = {
        "dq_blk": _grad(q_blk),
        "dk_ctx": _grad(k_ctx),
        "dv_ctx": _grad(v_ctx),
        "dk_blk": _grad(k_blk),
        "dv_blk": _grad(v_blk),
    }
    for name, grad in got.items():
        _assert_close(name, grad, ref[name], 0.05, 0.995)
    for name in ("dq_blk", "dk_blk", "dv_blk"):
        _assert_padded_rows_zero(name, got[name], inputs)
