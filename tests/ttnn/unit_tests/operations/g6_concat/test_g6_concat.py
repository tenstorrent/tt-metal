# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for g6_concat, Phase 0: bfloat16, TILE_LAYOUT, DRAM interleaved in and out.

Every shape is a production call of eval/golden_tests/g6_concat/feature_spec.py (BB-22) or a
case the op's rules require. The op only moves data, so every logical output element must be
bit-identical to its source element (NaN payloads and signed zeros included).
"""

import pytest
import torch
import ttnn

from ttnn.operations.g6_concat import g6_concat

L1_BUDGET_BYTES = 1347776

# (id, input shapes, dim, regime the op_design.md selection function assigns)
CASES = [
    # page_run, concat_dim "w"
    ("flux23_w_rows1", [[1, 1, 1, 128]] * 2, -1, "page_run"),
    ("qwen108_w_rows31", [[1, 1, 31, 64], [1, 1, 31, 192]], -1, "page_run"),
    ("qwen109_w_rows31_6ch", [[1, 6, 31, 64], [1, 6, 31, 192]], -1, "page_run"),
    ("qwen113_w_rows33_6ch", [[1, 6, 33, 64], [1, 6, 33, 192]], -1, "page_run"),
    ("qwen74_w_rows1_16ch", [[1, 16, 1, 64], [1, 16, 1, 192]], -1, "page_run"),
    ("qwen80_w_rows16_4in", [[1, 1, 16, 16384]] * 3 + [[1, 1, 16, 12928]], -1, "page_run"),
    ("qwen160_w_rows1_4in", [[1, 1, 1, 16384]] * 3 + [[1, 1, 1, 12928]], -1, "page_run"),
    ("qwen25_w_rows3844", [[1, 1, 3844, 64], [1, 1, 3844, 192]], -1, "page_run"),
    ("flux18_w_16ch", [[1, 16, 512, 64]] * 2, -1, "page_run"),
    ("flux20_w_rank3_3in", [[1, 512, 5120]] * 3, -1, "page_run"),
    # page_run, concat_dim "outer"
    ("qwen81_outer_16in", [[1, 1, 128, 256]] * 16, 0, "page_run"),
    ("qwen82_outer_16in_6ch", [[1, 6, 128, 256]] * 16, 0, "page_run"),
    ("qwen85_outer_rank3_16in", [[1, 128, 1536]] * 16, 0, "page_run"),
    # 49152 output pages: more pages per core than one L1-capped block, so the last block is ragged.
    ("qwen22_outer_multiblock", [[1, 6, 4096, 256]] * 8, 0, "page_run"),
    # page_run, concat_dim "h" with tile-aligned heights
    ("flux21_h_aligned", [[1, 1, 2048, 128], [1, 1, 256, 128]], 2, "page_run"),
    ("flux24_h_rank3", [[1, 2048, 3072], [1, 256, 3072]], 1, "page_run"),
    # row_gather: rows of several height-1 inputs share each output tile
    ("qwen89_h_rows1_2in", [[1, 1, 1, 62080]] * 2, 2, "row_gather"),
    ("qwen143_h_rows1_8in", [[1, 1, 1, 62080]] * 8, 2, "row_gather"),
    ("qwen79_h_rows1_16in", [[1, 1, 1, 62080]] * 16, 2, "row_gather"),
]


def _params(cases):
    return [pytest.param(shapes, dim, regime, id=case_id) for case_id, shapes, dim, regime in cases]


def _bits(t):
    return t.contiguous().view(torch.int16)


def _make_inputs(shapes):
    torch.manual_seed(42)
    # Successive draws give each input its own values, so a swapped, repeated or dropped input shows.
    return [torch.randn(shape, dtype=torch.float32).to(torch.bfloat16) for shape in shapes]


def _to_device(torch_inputs, device):
    return [
        ttnn.from_torch(
            t,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        for t in torch_inputs
    ]


def _reference(torch_inputs, dim):
    return torch.cat(torch_inputs, dim=dim)


def _check(output, expected, tt_inputs):
    assert list(output.shape) == list(expected.shape)
    assert output.dtype == ttnn.bfloat16
    assert output.layout == ttnn.TILE_LAYOUT
    mc = output.memory_config()
    assert mc.buffer_type == ttnn.BufferType.DRAM
    assert mc.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
    for k, t in enumerate(tt_inputs):
        assert t.memory_config().buffer_type == ttnn.BufferType.DRAM, f"input {k} changed placement"
    got = ttnn.to_torch(output)
    mismatch = _bits(got) != _bits(expected)
    bad = int(mismatch.sum())
    assert bad == 0, f"{bad} of {expected.numel()} elements differ in bits; first at {mismatch.nonzero()[0].tolist()}"
    return got


def _special_values():
    # NaN with payload, +inf, -inf, -0.0, smallest +/- subnormal (bfloat16 bit patterns).
    bits = [0x7FC1, 0x7F80, -0x0080, -0x8000, 0x0001, -0x7FFF]
    return torch.tensor(bits, dtype=torch.int16).view(torch.bfloat16)


@pytest.mark.parametrize("shapes,dim,regime", _params(CASES))
def test_g6_concat(device, shapes, dim, regime):
    from ttnn.operations.g6_concat import plan_of

    torch_inputs = _make_inputs(shapes)
    tt_inputs = _to_device(torch_inputs, device)

    plan = plan_of(tt_inputs, dim)
    assert plan["regime"] == regime
    assert plan["l1_bytes_per_core"] <= L1_BUDGET_BYTES

    output = g6_concat(tt_inputs, dim)
    _check(output, _reference(torch_inputs, dim), tt_inputs)


@pytest.mark.parametrize(
    "shapes,dim,canonical_dim",
    [
        pytest.param([[1, 16, 512, 64]] * 2, 3, -1, id="flux18_dim3_is_last"),
        pytest.param([[1, 1, 1, 62080]] * 2, -2, 2, id="qwen89_dim_minus2_is_h"),
    ],
)
def test_g6_concat_dim_canonicalization(device, shapes, dim, canonical_dim):
    torch_inputs = _make_inputs(shapes)
    tt_inputs = _to_device(torch_inputs, device)
    output = g6_concat(tt_inputs, dim)
    _check(output, _reference(torch_inputs, canonical_dim), tt_inputs)


def test_g6_concat_same_tensor_repeated(device):
    # g6_concat([x] * 16, 0): one buffer feeds sixteen output segments.
    (x,) = _make_inputs([[1, 1, 128, 256]])
    (x_tt,) = _to_device([x], device)
    output = g6_concat([x_tt] * 16, 0)
    _check(output, _reference([x] * 16, 0), [x_tt])


@pytest.mark.parametrize(
    "shapes,dim,poison",
    [
        pytest.param([[1, 6, 31, 64], [1, 6, 31, 192]], -1, 1000.0, id="qwen109_poison_1000"),
        pytest.param([[1, 1, 1, 62080]] * 2, 2, float("nan"), id="qwen89_poison_nan"),
        pytest.param([[1, 1, 1, 62080]] * 16, 2, 1000.0, id="qwen79_poison_1000"),
    ],
)
def test_g6_concat_ignores_input_padding(device, shapes, dim, poison):
    torch_inputs = _make_inputs(shapes)
    tt_inputs = [
        ttnn.fill_implicit_tile_padding(t, poison, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for t in _to_device(torch_inputs, device)
    ]
    output = g6_concat(tt_inputs, dim)
    _check(output, _reference(torch_inputs, dim), tt_inputs)


@pytest.mark.parametrize(
    "shapes,dim",
    [
        pytest.param([[1, 6, 31, 64], [1, 6, 31, 192]], -1, id="qwen109_w"),
        pytest.param([[1, 1, 1, 62080]] * 16, 2, id="qwen79_h"),
    ],
)
def test_g6_concat_special_values(device, shapes, dim):
    specials = _special_values()
    n = len(specials)
    torch_inputs = []
    for t in _make_inputs(shapes):
        flat = t.view(-1)
        flat[:n] = specials
        flat[flat.numel() - n :] = specials.flip(0)
        torch_inputs.append(t)
    tt_inputs = _to_device(torch_inputs, device)
    for k, (src, t) in enumerate(zip(torch_inputs, tt_inputs)):
        if not torch.equal(_bits(ttnn.to_torch(t)), _bits(src)):
            pytest.skip(f"host-to-device path changed special values of input {k}")
    output = g6_concat(tt_inputs, dim)
    _check(output, _reference(torch_inputs, dim), tt_inputs)


def test_g6_concat_repeat_call_is_bit_identical(device):
    shapes, dim = [[1, 1, 1, 62080]] * 16, 2
    torch_inputs = _make_inputs(shapes)
    tt_inputs = _to_device(torch_inputs, device)
    expected = _reference(torch_inputs, dim)
    first = _check(g6_concat(tt_inputs, dim), expected, tt_inputs)
    second = _check(g6_concat(tt_inputs, dim), expected, tt_inputs)
    assert torch.equal(_bits(first), _bits(second))


def test_g6_concat_structures_sharing_a_program_hash(device):
    # FLUX 23 and Qwen 108 both produce 8 output pages on 8 cores with one page per block, but
    # differ in run lengths and logical rows per tile: a cache hit must not keep the other's values.
    sequence = [
        ([[1, 1, 1, 128]] * 2, -1),
        ([[1, 1, 31, 64], [1, 1, 31, 192]], -1),
        ([[1, 1, 1, 128]] * 2, -1),
    ]
    for shapes, dim in sequence:
        torch_inputs = _make_inputs(shapes)
        tt_inputs = _to_device(torch_inputs, device)
        _check(g6_concat(tt_inputs, dim), _reference(torch_inputs, dim), tt_inputs)


def _device_tensor(shape, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        torch.zeros(shape, dtype=torch.float32),
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def test_g6_concat_rejects_empty_list(device, expect_error):
    with expect_error((ValueError, RuntimeError), "g6_concat"):
        g6_concat([], 0)


@pytest.mark.parametrize(
    "case",
    ["rank2", "rank5", "rank_mismatch", "dtype_mismatch", "layout_mismatch", "size_mismatch", "dim_high", "dim_low"],
)
def test_g6_concat_rejects_invalid_inputs(device, expect_error, case):
    if case == "rank2":
        tensors, dim = [_device_tensor([32, 64], device)] * 2, 0
    elif case == "rank5":
        tensors, dim = [_device_tensor([1, 1, 1, 32, 64], device)] * 2, 0
    elif case == "rank_mismatch":
        tensors, dim = [_device_tensor([1, 32, 64], device), _device_tensor([1, 1, 32, 64], device)], 0
    elif case == "dtype_mismatch":
        tensors, dim = [
            _device_tensor([1, 1, 32, 64], device),
            _device_tensor([1, 1, 32, 64], device, dtype=ttnn.float32),
        ], -1
    elif case == "layout_mismatch":
        tensors, dim = [
            _device_tensor([1, 1, 32, 64], device),
            _device_tensor([1, 1, 32, 64], device, layout=ttnn.ROW_MAJOR_LAYOUT),
        ], -1
    elif case == "size_mismatch":
        tensors, dim = [_device_tensor([1, 1, 32, 64], device), _device_tensor([1, 1, 64, 64], device)], -1
    elif case == "dim_high":
        tensors, dim = [_device_tensor([1, 1, 32, 64], device)] * 2, 4
    else:
        tensors, dim = [_device_tensor([1, 1, 32, 64], device)] * 2, -5
    with expect_error((ValueError, RuntimeError), "g6_concat"):
        g6_concat(tensors, dim)
