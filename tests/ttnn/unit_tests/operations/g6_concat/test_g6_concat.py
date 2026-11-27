# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for g6_concat, Phase 0: bfloat16, TILE_LAYOUT, DRAM interleaved in and out.

Shapes are production calls (eval/golden_tests/g6_concat/feature_spec.py INPUTS) plus the cases
the rules require: regime pins for page_copy / row_gather, poisoned padding, special values,
input order, dim spellings, repeat calls, program-cache keys and validation errors.

The op moves data and computes nothing, so every check is bit-exact on the logical values.
"""

import pytest
import torch
import ttnn

from ttnn.operations.g6_concat import g6_concat


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(42)


def _bits(t):
    t = t.contiguous()
    return t.view(torch.int16) if t.element_size() == 2 else t.view(torch.int32)


def torch_reference(torch_inputs, dim):
    return torch.cat(torch_inputs, dim=dim)


def to_device(t, device, *, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(t, dtype=dtype, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def random_inputs(shapes):
    return [torch.randn(shape, dtype=torch.float32).to(torch.bfloat16) for shape in shapes]


def check_output(output, expected):
    assert output.dtype == ttnn.bfloat16
    assert output.layout == ttnn.TILE_LAYOUT
    mc = output.memory_config()
    assert mc.buffer_type == ttnn.BufferType.DRAM
    assert mc.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
    got = ttnn.to_torch(output)
    assert list(got.shape) == list(expected.shape), f"shape {list(got.shape)} != {list(expected.shape)}"
    got = got.to(expected.dtype)
    mismatch = _bits(got) != _bits(expected)
    bad = int(mismatch.sum())
    assert bad == 0, f"{bad} of {expected.numel()} elements differ in bits; first at {mismatch.nonzero()[:4].tolist()}"


def run_case(shapes, dim, device):
    torch_inputs = random_inputs(shapes)
    expected = torch_reference(torch_inputs, dim)
    output = g6_concat([to_device(t, device) for t in torch_inputs], dim)
    check_output(output, expected)


PRODUCTION_CASES = [
    # page_copy, last-dim joins
    pytest.param([(1, 1, 1, 128)] * 2, -1, id="flux23_one_tile_row_w"),
    pytest.param([(1, 1, 64, 64), (1, 1, 64, 192)], -1, id="qwen104_aligned_w"),
    pytest.param([(1, 6, 31, 64), (1, 6, 31, 192)], -1, id="qwen109_31rows_w"),
    pytest.param([(1, 1, 33, 64), (1, 1, 33, 192)], -1, id="qwen112_33rows_w"),
    pytest.param([(1, 16, 1, 64), (1, 16, 1, 192)], -1, id="qwen74_height1_batched_w"),
    pytest.param([(1, 1, 3844, 64), (1, 1, 3844, 192)], -1, id="qwen25_3844rows_w"),
    pytest.param([(1, 1, 16, 16384)] * 3 + [(1, 1, 16, 12928)], -1, id="qwen80_wide_16rows_w"),
    pytest.param([(1, 1, 1, 16384)] * 3 + [(1, 1, 1, 12928)], -1, id="qwen160_wide_1row_w"),
    pytest.param([(1, 16, 512, 64)] * 2, -1, id="flux18_w"),
    # page_copy, leading-dim joins
    pytest.param([(1, 1, 128, 256)] * 16, 0, id="qwen81_16x_outer"),
    pytest.param([(1, 6, 128, 256)] * 2, 0, id="qwen92_outer"),
    pytest.param([(1, 128, 1536)] * 16, 0, id="qwen85_rank3_16x_outer"),
    pytest.param([(1, 1024, 1536)] * 8, 0, id="qwen136_rank3_multiblock_outer"),
    pytest.param([(1, 4096, 1536)] * 8, 0, id="qwen23_p4_outer"),
    # page_copy, row-dim joins on tile boundaries
    pytest.param([(1, 1, 2048, 128), (1, 1, 256, 128)], 2, id="flux21_aligned_h"),
    pytest.param([(1, 2048, 3072), (1, 256, 3072)], 1, id="flux24_rank3_aligned_h"),
    # row_gather: rows of several height-1 inputs share one output tile
    pytest.param([(1, 1, 1, 62080)] * 2, 2, id="qwen89_2x_height1_h"),
    pytest.param([(1, 1, 1, 62080)] * 8, 2, id="qwen143_8x_height1_h"),
    pytest.param([(1, 1, 1, 62080)] * 16, 2, id="qwen79_16x_height1_h"),
]


@pytest.mark.parametrize("shapes, dim", PRODUCTION_CASES)
def test_g6_concat(shapes, dim, device):
    run_case(shapes, dim, device)


REGIME_PIN_CASES = [
    # h join, only the last input unaligned: page_copy, output's last tile row has one valid row
    pytest.param([(1, 1, 64, 128), (1, 1, 33, 128)], 2, id="page_copy_last_input_unaligned_h"),
    # row_gather: shift 31 crosses both face halves and both tile rows of the second input
    pytest.param([(1, 2, 31, 64), (1, 2, 33, 64), (1, 2, 7, 64)], 2, id="row_gather_straddle_h"),
    # row_gather, rank 3: shift 5, two input tile rows land in two output tile rows
    pytest.param([(2, 5, 64), (2, 40, 64)], 1, id="row_gather_rank3_h"),
    # K = 1 is a copy
    pytest.param([(1, 6, 31, 64)], -1, id="single_input"),
]


@pytest.mark.parametrize("shapes, dim", REGIME_PIN_CASES)
def test_g6_concat_regime_pins(shapes, dim, device):
    run_case(shapes, dim, device)


@pytest.mark.parametrize(
    "shapes, dim, poison",
    [
        pytest.param([(1, 6, 31, 64), (1, 6, 31, 192)], -1, 1000.0, id="qwen109_w"),
        pytest.param([(1, 1, 33, 64), (1, 1, 33, 192)], -1, float("nan"), id="qwen112_w"),
        pytest.param([(1, 1, 1, 62080)] * 2, 2, float("nan"), id="qwen89_h"),
        pytest.param([(1, 2, 31, 64), (1, 2, 33, 64), (1, 2, 7, 64)], 2, 1000.0, id="row_gather_straddle_h"),
    ],
)
def test_g6_concat_poisoned_padding(shapes, dim, poison, device):
    """No logical output element may depend on what an input's tile padding holds."""
    torch_inputs = random_inputs(shapes)
    expected = torch_reference(torch_inputs, dim)
    tt_inputs = [
        ttnn.fill_implicit_tile_padding(to_device(t, device), poison, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for t in torch_inputs
    ]
    output = g6_concat(tt_inputs, dim)
    check_output(output, expected)


def _special_values():
    # NaN with payload, +inf, -inf, -0.0, smallest +/- subnormal (bfloat16 bit patterns)
    bits = [0x7FC1, 0x7F80, -0x0080, -0x8000, 0x0001, -0x7FFF]
    return torch.tensor(bits, dtype=torch.int16).view(torch.bfloat16)


@pytest.mark.parametrize(
    "shapes, dim",
    [
        pytest.param([(1, 6, 31, 64), (1, 6, 31, 192)], -1, id="qwen109_w"),
        pytest.param([(1, 1, 1, 62080)] * 2, 2, id="qwen89_h"),
    ],
)
def test_g6_concat_special_values(shapes, dim, device):
    """NaN payloads, infinities, -0.0 and subnormals reach the output bit for bit."""
    specials = _special_values()
    torch_inputs = []
    for t in random_inputs(shapes):
        flat = t.view(-1)
        n = min(len(specials), flat.numel())
        flat[:n] = specials[:n]
        flat[flat.numel() - n :] = specials.flip(0)[:n]
        torch_inputs.append(t)
    tt_inputs = [to_device(t, device) for t in torch_inputs]
    for src, t in zip(torch_inputs, tt_inputs):
        if not torch.equal(_bits(ttnn.to_torch(t).to(torch.bfloat16)), _bits(src)):
            pytest.skip("harness: the torch-to-device path changed special values")
    output = g6_concat(tt_inputs, dim)
    check_output(output, torch_reference(torch_inputs, dim))


@pytest.mark.parametrize(
    "shapes, dim",
    [
        pytest.param([(1, 1, 1, 62080)] * 16, 2, id="qwen79_h"),
        pytest.param([(1, 1, 128, 256)] * 16, 0, id="qwen81_outer"),
        pytest.param([(1, 1, 16, 16384)] * 3 + [(1, 1, 16, 12928)], -1, id="qwen80_w"),
    ],
)
def test_g6_concat_input_order(shapes, dim, device):
    """Input k holds the constant k + 1: every output slice comes from its own input, in order."""
    torch_inputs = [torch.full(shape, float(k + 1), dtype=torch.bfloat16) for k, shape in enumerate(shapes)]
    output = g6_concat([to_device(t, device) for t in torch_inputs], dim)
    check_output(output, torch_reference(torch_inputs, dim))


@pytest.mark.parametrize(
    "shapes, negative_dim, positive_dim",
    [
        pytest.param([(1, 1, 64, 64), (1, 1, 64, 192)], -1, 3, id="qwen104_w"),
        pytest.param([(1, 1, 1, 62080)] * 2, -2, 2, id="qwen89_h"),
        pytest.param([(1, 128, 512)] * 2, -3, 0, id="qwen96_rank3_outer"),
    ],
)
def test_g6_concat_dim_spellings(shapes, negative_dim, positive_dim, device):
    """A negative dim and its canonical positive spelling are the same case."""
    torch_inputs = random_inputs(shapes)
    tt_inputs = [to_device(t, device) for t in torch_inputs]
    expected = torch_reference(torch_inputs, positive_dim)
    check_output(g6_concat(tt_inputs, negative_dim), expected)
    check_output(g6_concat(tt_inputs, positive_dim), expected)


@pytest.mark.parametrize(
    "shapes, dim",
    [
        pytest.param([(1, 1, 1, 62080)] * 16, 2, id="qwen79_h"),
        pytest.param([(1, 1, 64, 64), (1, 1, 64, 192)], -1, id="qwen104_w"),
    ],
)
def test_g6_concat_repeat_call(shapes, dim, device):
    """Three calls on the same inputs return bit-identical logical outputs."""
    torch_inputs = random_inputs(shapes)
    tt_inputs = [to_device(t, device) for t in torch_inputs]
    expected = torch_reference(torch_inputs, dim)
    for _ in range(3):
        check_output(g6_concat(tt_inputs, dim), expected)


def test_g6_concat_cache_key(device):
    """Same output shape from different input splits, back to back: no stale program."""
    for shapes in ([(1, 1, 128, 64), (1, 1, 128, 192)], [(1, 1, 128, 192), (1, 1, 128, 64)]):
        run_case(shapes, -1, device)
    # row_gather then page_copy (K = 1) with the same output shape
    for shapes in ([(1, 1, 1, 62080)] * 2, [(1, 1, 2, 62080)]):
        run_case(shapes, 2, device)


def _bf16(shape, device, *, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return to_device(torch.randn(shape, dtype=torch.float32).to(torch.bfloat16), device, dtype=dtype, layout=layout)


@pytest.mark.parametrize(
    "case",
    [
        "empty_list",
        "rank_2",
        "rank_5",
        "rank_mismatch",
        "dtype_mismatch",
        "layout_mismatch",
        "size_mismatch_outside_dim",
        "dim_too_large",
        "dim_too_small",
    ],
)
def test_g6_concat_validation(case, device, expect_error):
    if case == "empty_list":
        tensors, dim = [], 0
    elif case == "rank_2":
        tensors, dim = [_bf16((32, 64), device)] * 2, -1
    elif case == "rank_5":
        tensors, dim = [_bf16((1, 1, 1, 32, 64), device)] * 2, -1
    elif case == "rank_mismatch":
        tensors, dim = [_bf16((1, 1, 32, 64), device), _bf16((1, 32, 64), device)], -1
    elif case == "dtype_mismatch":
        tensors, dim = [_bf16((1, 1, 32, 64), device), _bf16((1, 1, 32, 64), device, dtype=ttnn.float32)], -1
    elif case == "layout_mismatch":
        tensors, dim = [
            _bf16((1, 1, 32, 64), device),
            _bf16((1, 1, 32, 64), device, layout=ttnn.ROW_MAJOR_LAYOUT),
        ], -1
    elif case == "size_mismatch_outside_dim":
        tensors, dim = [_bf16((1, 1, 32, 64), device), _bf16((1, 1, 64, 64), device)], -1
    elif case == "dim_too_large":
        tensors, dim = [_bf16((1, 1, 32, 64), device)] * 2, 4
    else:
        tensors, dim = [_bf16((1, 1, 32, 64), device)] * 2, -5
    # The refusal text is the implementer's; only the exception type is the contract.
    with expect_error((ValueError, RuntimeError), ""):
        g6_concat(tensors, dim)
