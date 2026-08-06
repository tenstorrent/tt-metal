# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only equivalence proof for the scattered-domain row-parallel bias.

`HY_DIT_RS_DOMAIN_BIAS=1` moves the row-parallel bias add of `to_out`,
`to_add_out` and the two FF down-projections from after the all-gather to
between the reduce-scatter and the all-gather.

These tests simulate the TP rank decomposition in torch (no device) and show
the transformed dataflow is *bit-identical*, not merely close: the reduction
itself is untouched, the bias is a per-column constant that is outside it, and
an all-gather is a pure copy.  Bit-identity is asserted in bfloat16 as well as
float64, because bfloat16 is where a reassociation would show up first.
"""

import pytest
import torch

TP = 4
M = 96  # tokens
K = 3072  # model width (contraction dim, sharded row-wise)
N = 3072  # output width


def _row_parallel_partials(x, w, dtype):
    """Per-device partial products of a Megatron row-parallel projection.

    Device `d` owns rows `[d*K/TP, (d+1)*K/TP)` of `w` and the matching columns
    of `x`, and produces a full-width (M, N) partial that still has to be summed
    over the TP axis.
    """
    kc = K // TP
    return [(x[:, d * kc : (d + 1) * kc].to(dtype) @ w[d * kc : (d + 1) * kc, :].to(dtype)) for d in range(TP)]


def _reduce_scatter(partials, dtype):
    """Ring reduce-scatter over the last dim: device `d` gets column chunk `d`.

    The sum for a given column is formed from exactly the same operands in
    exactly the same order as the all-reduce this replaces -- reduce_scatter is
    the first half of the all-reduce the block already runs.
    """
    nc = N // TP
    out = []
    for d in range(TP):
        acc = partials[0][:, d * nc : (d + 1) * nc].clone().to(dtype)
        for e in range(1, TP):
            acc = acc + partials[e][:, d * nc : (d + 1) * nc].to(dtype)
        out.append(acc)
    return out


def _all_gather(shards):
    return torch.cat(shards, dim=-1)


def _legacy(x, w, b, dtype):
    """all_gather(reduce_scatter(partials)) then add the replicated bias."""
    y = _all_gather(_reduce_scatter(_row_parallel_partials(x, w, dtype), dtype))
    return y + b.to(dtype)


def _rs_domain(x, w, b, dtype):
    """Add the TP-fractured bias to the reduce-scatter output, then all-gather."""
    nc = N // TP
    shards = _reduce_scatter(_row_parallel_partials(x, w, dtype), dtype)
    biased = [s + b[..., d * nc : (d + 1) * nc].to(dtype) for d, s in enumerate(shards)]
    return _all_gather(biased)


def _mmrs_then_rs_domain(x, w, b, dtype):
    """Same, for the fused matmul+reduce-scatter path (`HY_DIT_MMRS_OVERLAP`).

    MMRS streams matmul blocks into the reduce-scatter but computes the same
    per-device partials, so it lands on the identical scattered tensor.
    """
    return _rs_domain(x, w, b, dtype)


@pytest.fixture(scope="module")
def operands():
    torch.manual_seed(0)
    x = torch.randn(M, K, dtype=torch.float64)
    w = torch.randn(K, N, dtype=torch.float64) / K**0.5
    b = torch.randn(1, N, dtype=torch.float64)
    return x, w, b


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.bfloat16])
def test_scattered_domain_bias_is_bit_identical(operands, dtype):
    x, w, b = operands
    legacy = _legacy(x, w, b, dtype)
    moved = _rs_domain(x, w, b, dtype)
    assert legacy.dtype == moved.dtype == dtype
    assert torch.equal(legacy, moved), (legacy - moved).abs().max().item()


@pytest.mark.parametrize("dtype", [torch.float64, torch.bfloat16])
def test_scattered_domain_bias_is_bit_identical_under_mmrs(operands, dtype):
    x, w, b = operands
    assert torch.equal(_legacy(x, w, b, dtype), _mmrs_then_rs_domain(x, w, b, dtype))


def test_reduction_itself_is_unchanged_by_the_transform(operands):
    """The bias move must not touch the summation order of the reduction.

    Strip the bias from both paths and require the pre-bias tensors to match
    exactly; this is what makes the bias-add position irrelevant.
    """
    x, w, b = operands
    zero = torch.zeros_like(b)
    for dtype in (torch.float64, torch.bfloat16):
        assert torch.equal(_legacy(x, w, zero, dtype), _rs_domain(x, w, zero, dtype))


def test_rank_decomposition_reproduces_the_single_device_projection(operands):
    """Sanity check on the simulation itself, at float64 tolerance."""
    x, w, b = operands
    reference = x @ w + b
    assert torch.allclose(reference, _legacy(x, w, b, torch.float64), atol=1e-12, rtol=0)
    assert torch.allclose(reference, _rs_domain(x, w, b, torch.float64), atol=1e-12, rtol=0)


def test_fractured_bias_chunking_matches_the_all_gather_column_order(operands):
    """Chunk `d` of the bias must land on device `d`.

    `_mapper(-1)` shards the bias last-dim-major and `all_gather(dim=-1)`
    concatenates device-major, so a mismatch here would silently permute the
    bias across the output width -- a failure mode PCC would catch late.
    """
    _x, _w, b = operands
    nc = N // TP
    per_device = [b[..., d * nc : (d + 1) * nc] for d in range(TP)]
    assert torch.equal(_all_gather(per_device), b)


def test_bias_is_not_multiplied_by_tp():
    """Guard the failure mode the transform must avoid.

    Adding the replicated bias on every rank *before* the reduction (i.e.
    passing bias into the fused MMRS epilogue) scales it by TP.  This test
    documents why `_row_linear_mmrs_start` passes `bias=None`.
    """
    torch.manual_seed(1)
    x = torch.randn(M, K, dtype=torch.float64)
    w = torch.randn(K, N, dtype=torch.float64) / K**0.5
    b = torch.ones(1, N, dtype=torch.float64)
    partials = _row_parallel_partials(x, w, torch.float64)
    wrong = _all_gather(_reduce_scatter([p + b for p in partials], torch.float64))
    right = _rs_domain(x, w, b, torch.float64)
    assert torch.allclose(wrong - right, torch.full_like(right, TP - 1.0), atol=1e-9, rtol=0)
