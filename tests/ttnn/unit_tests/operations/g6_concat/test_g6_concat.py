# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for g6_concat (Phase 0: bfloat16, TILE, DRAM in and out).

IMMUTABLE SPEC — the implementer must not modify this file.

g6_concat moves bytes and computes nothing, so the tolerance is the golden suite's
exact-copy gate: every logical output element is bit-identical to its source
element (compared as int16 bit patterns, so NaN payloads and -0.0 count).

Cases are limited to production shapes (eval/golden_tests/g6_concat/feature_spec.py
INPUTS inside Phase 0) plus the cases the spec's rules require (BB-22):
  - both regimes of op_design.md pinned (page_relocation / row_gather),
  - one row_gather case whose pieces straddle face halves and tile edges,
  - poisoned input padding (no logical output depends on padding),
  - dim spellings (-1 and 3 are the same case),
  - Python-side validation errors.
"""

import pytest
import torch

import ttnn
from ttnn.operations.g6_concat import g6_concat


def pytorch_g6_concat(tensors, dim=0):
    return torch.cat([t.to(torch.float32) for t in tensors], dim=dim).to(tensors[0].dtype)


def _bits(t):
    return t.contiguous().view(torch.int16)


def _make_inputs(shapes):
    torch.manual_seed(42)
    return [torch.randn(list(s), dtype=torch.float32).to(torch.bfloat16) for s in shapes]


def _to_device(torch_inputs, device):
    return [
        ttnn.from_torch(
            t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        for t in torch_inputs
    ]


def _check(output, torch_inputs, dim):
    expected = pytorch_g6_concat(torch_inputs, dim)
    assert list(output.shape) == list(expected.shape), f"shape {list(output.shape)} != {list(expected.shape)}"
    assert output.dtype == ttnn.bfloat16
    assert output.layout == ttnn.TILE_LAYOUT
    mc = output.memory_config()
    assert mc.buffer_type == ttnn.BufferType.DRAM, f"output buffer {mc.buffer_type}, expected DRAM"
    assert mc.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
    got = ttnn.to_torch(output).to(torch.bfloat16)
    mismatch = _bits(got) != _bits(expected)
    assert not mismatch.any(), (
        f"{int(mismatch.sum())} of {mismatch.numel()} elements differ; "
        f"first at {mismatch.nonzero()[0].tolist()}: got {got[tuple(mismatch.nonzero()[0])]} "
        f"want {expected[tuple(mismatch.nonzero()[0])]}"
    )


# (shapes, dim, id). Production cases from feature_spec.py INPUTS (Phase 0 cells) unless marked.
CASES = [
    # single tile row, two inputs, last dim (Qwen 157-shaped, DRAM): page_relocation, "w"
    ([[1, 1, 1, 128]] * 2, -1, "w_1x128x2"),
    # 31 rows, non-square, last dim: page_relocation, "w", h_non_aligned
    ([[1, 6, 31, 64], [1, 6, 31, 192]], -1, "w_6x31_64+192"),
    # multi-tile, tile-aligned, last dim, multi-batch
    ([[1, 6, 256, 64], [1, 6, 256, 192]], -1, "w_6x256_64+192"),
    # four inputs, one tile-row tall, 1940 tile columns: page_relocation, "w"
    ([[1, 1, 16, 16384]] * 3 + [[1, 1, 16, 12928]], -1, "w_16rows_4in"),
    # row dim, tile-aligned offsets: must take page_relocation (regime-pinned)
    ([[1, 1, 2048, 128], [1, 1, 256, 128]], 2, "h_aligned_rank4"),
    # row dim of a rank-3 tensor, tile-aligned
    ([[1, 2048, 3072], [1, 256, 3072]], 1, "h_aligned_rank3"),
    # row dim, height-1 inputs: rows of 2 / 16 inputs share one output tile (row_gather, regime-pinned)
    ([[1, 1, 1, 62080]] * 2, 2, "h_gather_2in"),
    ([[1, 1, 1, 62080]] * 16, 2, "h_gather_16in"),
    # leading dim, sixteen inputs, multi-slice
    ([[1, 6, 128, 256]] * 16, 0, "outer_16x6x128x256"),
    # leading dim, sixteen large inputs (Qwen 19)
    ([[1, 4096, 1536]] * 16, 0, "outer_16x4096x1536"),
    # leading dim, two inputs (Qwen 91)
    ([[1, 1, 128, 256]] * 2, 0, "outer_2in"),
    # RULE-REQUIRED (not production): row_gather pieces cross face halves and input-tile edges
    ([[1, 1, 31, 64], [1, 1, 33, 64], [1, 1, 1, 64]], 2, "h_gather_straddle"),
]


@pytest.mark.parametrize("shapes, dim", [(c[0], c[1]) for c in CASES], ids=[c[2] for c in CASES])
def test_g6_concat(device, shapes, dim):
    torch_inputs = _make_inputs(shapes)
    tt_inputs = _to_device(torch_inputs, device)
    output = g6_concat(tt_inputs, dim)
    _check(output, torch_inputs, dim)


def test_g6_concat_dim_spellings(device):
    """dim=-1 and dim=3 of a rank-4 input are the same case."""
    shapes = [[1, 6, 31, 64], [1, 6, 31, 192]]
    torch_inputs = _make_inputs(shapes)
    tt_inputs = _to_device(torch_inputs, device)
    out_neg = ttnn.to_torch(g6_concat(tt_inputs, -1)).to(torch.bfloat16)
    out_pos = ttnn.to_torch(g6_concat(tt_inputs, 3)).to(torch.bfloat16)
    expected = pytorch_g6_concat(torch_inputs, 3)
    assert torch.equal(_bits(out_neg), _bits(expected))
    assert torch.equal(_bits(out_pos), _bits(expected))


@pytest.mark.parametrize(
    "shapes, dim, poison",
    [
        ([[1, 6, 31, 64], [1, 6, 31, 192]], -1, 1000.0),
        ([[1, 1, 1, 62080]] * 2, 2, float("nan")),
        ([[1, 1, 31, 64], [1, 1, 33, 64], [1, 1, 1, 64]], 2, float("nan")),
    ],
    ids=["w_31rows_poison1000", "h_gather_poison_nan", "h_gather_straddle_poison_nan"],
)
def test_g6_concat_poisoned_padding(device, shapes, dim, poison):
    """No logical output element may depend on the content of input tile padding."""
    torch_inputs = _make_inputs(shapes)
    tt_inputs = [
        ttnn.fill_implicit_tile_padding(t, poison, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for t in _to_device(torch_inputs, device)
    ]
    output = g6_concat(tt_inputs, dim)
    _check(output, torch_inputs, dim)


def test_g6_concat_special_values(device):
    """NaN, +-inf, -0.0 and subnormals move bit for bit (row_gather path)."""
    shapes = [[1, 1, 1, 62080]] * 2
    torch_inputs = _make_inputs(shapes)
    specials = torch.tensor([float("nan"), float("inf"), float("-inf"), -0.0, 1e-39, -1e-39], dtype=torch.float32).to(
        torch.bfloat16
    )
    for t in torch_inputs:
        t.view(-1)[: specials.numel()] = specials
        t.view(-1)[-specials.numel() :] = specials
    tt_inputs = _to_device(torch_inputs, device)
    # Reference = the values the device holds (the op is graded, not the host conversion).
    held = [ttnn.to_torch(t).to(torch.bfloat16) for t in tt_inputs]
    output = g6_concat(tt_inputs, 2)
    _check(output, held, 2)


def test_g6_concat_repeat_is_deterministic(device):
    """The same call twice returns bit-identical output (BB-06)."""
    shapes = [[1, 1, 1, 62080]] * 16
    torch_inputs = _make_inputs(shapes)
    tt_inputs = _to_device(torch_inputs, device)
    first = ttnn.to_torch(g6_concat(tt_inputs, 2)).to(torch.bfloat16)
    second = ttnn.to_torch(g6_concat(tt_inputs, 2)).to(torch.bfloat16)
    assert torch.equal(_bits(first), _bits(second))
    assert torch.equal(_bits(first), _bits(pytorch_g6_concat(torch_inputs, 2)))


def _dev(device, shape, dtype=ttnn.bfloat16):
    return ttnn.from_torch(torch.randn(shape), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


@pytest.mark.parametrize(
    "build, dim",
    [
        (lambda d: [], 0),  # empty list
        (lambda d: [_dev(d, (1, 1, 32, 32)), _dev(d, (1, 32, 32))], 0),  # rank mismatch
        (lambda d: [_dev(d, (1, 1, 32, 32)), _dev(d, (1, 1, 32, 32), ttnn.float32)], 0),  # dtype mismatch
        (lambda d: [_dev(d, (1, 1, 32, 32)), _dev(d, (1, 1, 32, 64))], 2),  # size mismatch off dim
        (lambda d: [_dev(d, (1, 1, 32, 32))] * 2, 4),  # dim out of range
        (lambda d: [_dev(d, (1, 1, 32, 32))] * 2, -5),  # dim out of range
        (lambda d: [_dev(d, (32, 32))] * 2, 0),  # rank 2
        (lambda d: [_dev(d, (1, 1, 1, 32, 32))] * 2, 0),  # rank 5
    ],
    ids=["empty", "rank_mismatch", "dtype_mismatch", "size_mismatch", "dim_too_big", "dim_too_small", "rank2", "rank5"],
)
def test_g6_concat_validation(device, expect_error, build, dim):
    tensors = build(device)
    with expect_error((ValueError, RuntimeError), ""):
        g6_concat(tensors, dim)
