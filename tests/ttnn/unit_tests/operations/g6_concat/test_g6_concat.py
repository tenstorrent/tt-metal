# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for g6_concat, Phase 0 (bfloat16, TILE, DRAM interleaved in and out).

Cases are the Phase 0 production calls of eval/golden_tests/g6_concat/feature_spec.py
plus the cases the op's rules require: input padding never reaches a logical row,
rows of several inputs share an output tile for any heights, the plan a call selects
is reportable, the single program is reused on a repeat call.

The op moves data and computes nothing, so every logical value is compared bit for
bit (as int16 for bfloat16): NaN payloads, signed zeros and subnormals included.
"""

import pytest
import torch
import ttnn

import ttnn.operations.g6_concat as g6_concat_module
from ttnn.operations.g6_concat import g6_concat

BF16 = ttnn.bfloat16
TILE = ttnn.TILE_LAYOUT

# (id, input shapes, dim) -- production calls inside Phase 0 (feature_spec.py INPUTS).
PRODUCTION_CASES = [
    ("flux23_1row_w", [[1, 1, 1, 128]] * 2, -1),
    ("qwen109_31rows_w", [[1, 6, 31, 64], [1, 6, 31, 192]], -1),
    ("qwen113_33rows_w", [[1, 6, 33, 64], [1, 6, 33, 192]], -1),
    ("qwen151_w", [[1, 1, 4096, 64], [1, 1, 4096, 192]], -1),
    ("qwen160_1row_wide_w", [[1, 1, 1, 16384]] * 3 + [[1, 1, 1, 12928]], -1),
    ("qwen80_16rows_wide_w", [[1, 1, 16, 16384]] * 3 + [[1, 1, 16, 12928]], -1),
    ("flux20_rank3_w", [[1, 512, 5120]] * 3, -1),
    ("qwen79_16x1row_h", [[1, 1, 1, 62080]] * 16, 2),
    ("qwen89_2x1row_h", [[1, 1, 1, 62080]] * 2, 2),
    ("qwen143_8x1row_h", [[1, 1, 1, 62080]] * 8, 2),
    ("flux21_aligned_h", [[1, 1, 2048, 128], [1, 1, 256, 128]], 2),
    ("flux24_rank3_aligned_h", [[1, 2048, 3072], [1, 256, 3072]], 1),
    ("qwen81_16x_outer", [[1, 1, 128, 256]] * 16, 0),
    ("qwen92_multibatch_outer", [[1, 6, 128, 256]] * 2, 0),
    ("qwen147_rank3_outer", [[1, 128, 1536]] * 8, 0),
    ("qwen136_rank3_outer", [[1, 1024, 1536]] * 8, 0),
]

# Required by the rule "output row o_k + j comes from row j of input k" for any heights:
# runs that straddle source and output tiles, both row parities, several images, rank 3.
RULE_ROW_GATHER_CASES = [
    ("rule_h_mixed_heights", [[1, 2, 31, 64], [1, 2, 33, 64], [1, 2, 1, 64], [1, 2, 64, 64]], 2),
    ("rule_h_rank3_rows", [[2, 5, 96], [2, 7, 96], [2, 40, 96]], 1),
]

# Predicate edge: only the LAST input is not tile-aligned, so every offset is aligned.
RULE_LAST_MISALIGNED = ("rule_h_last_misaligned", [[1, 1, 64, 64], [1, 1, 31, 64]], 2)

ALL_VALUE_CASES = PRODUCTION_CASES + RULE_ROW_GATHER_CASES + [RULE_LAST_MISALIGNED]

# Regime the plan must select (op_design.md, Regimes).
REGIME_CASES = [
    ("qwen79", [[1, 1, 1, 62080]] * 16, 2, "row_gather"),
    ("qwen89_dim_neg2", [[1, 1, 1, 62080]] * 2, -2, "row_gather"),
    ("rule_h_mixed_heights", RULE_ROW_GATHER_CASES[0][1], 2, "row_gather"),
    ("rule_h_rank3_rows", RULE_ROW_GATHER_CASES[1][1], 1, "row_gather"),
    ("rule_h_last_misaligned", RULE_LAST_MISALIGNED[1], 2, "page_copy"),
    ("flux21", [[1, 1, 2048, 128], [1, 1, 256, 128]], 2, "page_copy"),
    ("qwen109", [[1, 6, 31, 64], [1, 6, 31, 192]], -1, "page_copy"),
    ("qwen81", [[1, 1, 128, 256]] * 16, 0, "page_copy"),
]


def _ids(cases):
    return [c[0] for c in cases]


def _make_inputs(shapes, seed=42):
    torch.manual_seed(seed)
    return [torch.randn(shape, dtype=torch.float32).to(torch.bfloat16) for shape in shapes]


def _to_device(torch_inputs, device):
    return [
        ttnn.from_torch(t, dtype=BF16, layout=TILE, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for t in torch_inputs
    ]


def _reference(torch_inputs, dim):
    # torch.cat on bfloat16 is a pure byte move: the exact expected bits.
    return torch.cat(torch_inputs, dim=dim)


def _assert_bit_exact(got, expected, label=""):
    assert list(got.shape) == list(expected.shape), f"{label} shape {list(got.shape)} != {list(expected.shape)}"
    got = got.to(torch.bfloat16).contiguous()
    expected = expected.contiguous()
    mismatch = got.view(torch.int16) != expected.view(torch.int16)
    bad = int(mismatch.sum())
    if bad:
        first = mismatch.nonzero()[:4].tolist()
        detail = "; ".join(
            f"{tuple(i)}: got {got[tuple(i)].item()!r} want {expected[tuple(i)].item()!r}" for i in first
        )
        pytest.fail(f"{label} {bad} of {expected.numel()} elements differ in bits; first: {detail}")


def _assert_output_tensor(out, expected_shape):
    assert list(out.shape) == list(expected_shape)
    assert out.dtype == BF16
    assert out.layout == TILE
    mc = out.memory_config()
    assert mc.buffer_type == ttnn.BufferType.DRAM
    assert mc.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED


def _run_and_check(torch_inputs, dim, device, label="", **kwargs):
    tt_inputs = _to_device(torch_inputs, device)
    expected = _reference(torch_inputs, dim)
    out = g6_concat(tt_inputs, dim, **kwargs)
    _assert_output_tensor(out, expected.shape)
    _assert_bit_exact(ttnn.to_torch(out), expected, label)
    return out


@pytest.mark.parametrize("case_id,shapes,dim", ALL_VALUE_CASES, ids=_ids(ALL_VALUE_CASES))
def test_values_bit_exact(case_id, shapes, dim, device):
    _run_and_check(_make_inputs(shapes), dim, device, label=case_id)


def test_single_input_is_a_copy(device):
    _run_and_check(_make_inputs([[1, 1, 4096, 64]]), -1, device, label="K=1")


def test_explicit_dram_memory_config(device):
    shapes = [[1, 1, 4096, 64], [1, 1, 4096, 192]]
    _run_and_check(_make_inputs(shapes), -1, device, label="memory_config=DRAM", memory_config=ttnn.DRAM_MEMORY_CONFIG)


def test_dim_spellings_agree(device):
    shapes = [[1, 1, 1, 62080]] * 2
    torch_inputs = _make_inputs(shapes)
    out_pos = ttnn.to_torch(_run_and_check(torch_inputs, 2, device, label="dim 2"))
    out_neg = ttnn.to_torch(_run_and_check(torch_inputs, -2, device, label="dim -2"))
    _assert_bit_exact(out_neg, out_pos, "dim -2 vs dim 2")


@pytest.mark.parametrize(
    "case_id,shapes,dim,poison",
    [
        ("qwen109_31rows_w", [[1, 6, 31, 64], [1, 6, 31, 192]], -1, 1000.0),
        ("qwen160_1row_wide_w", [[1, 1, 1, 16384]] * 3 + [[1, 1, 1, 12928]], -1, 1000.0),
        ("qwen89_2x1row_h", [[1, 1, 1, 62080]] * 2, 2, float("nan")),
        ("rule_h_mixed_heights", RULE_ROW_GATHER_CASES[0][1], 2, float("nan")),
    ],
    ids=["qwen109", "qwen160", "qwen89", "rule_h_mixed_heights"],
)
def test_input_padding_never_reaches_logical_rows(case_id, shapes, dim, poison, device):
    torch_inputs = _make_inputs(shapes)
    tt_inputs = [
        ttnn.fill_implicit_tile_padding(t, poison, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for t in _to_device(torch_inputs, device)
    ]
    expected = _reference(torch_inputs, dim)
    out = g6_concat(tt_inputs, dim)
    _assert_output_tensor(out, expected.shape)
    _assert_bit_exact(ttnn.to_torch(out), expected, f"{case_id} poisoned padding")


def _special_values_bf16():
    # NaN with payload, +inf, -inf, -0.0, smallest +subnormal, smallest -subnormal.
    bits = [0x7FC1, 0x7F80, -0x0080, -0x8000, 0x0001, -0x7FFF]
    return torch.tensor(bits, dtype=torch.int16).view(torch.bfloat16)


def _place_special_values(torch_inputs):
    placed = []
    for t in torch_inputs:
        t = t.clone()
        flat = t.view(-1)
        specials = _special_values_bf16()
        n = min(len(specials), flat.numel())
        flat[:n] = specials[:n]
        flat[flat.numel() - n :] = specials.flip(0)[:n]
        placed.append(t)
    return placed


@pytest.mark.parametrize(
    "case_id,shapes,dim",
    [
        ("qwen109_31rows_w", [[1, 6, 31, 64], [1, 6, 31, 192]], -1),
        ("rule_h_mixed_heights", RULE_ROW_GATHER_CASES[0][1], 2),
    ],
    ids=["qwen109", "rule_h_mixed_heights"],
)
def test_special_values_move_bit_for_bit(case_id, shapes, dim, device):
    torch_inputs = _place_special_values(_make_inputs(shapes))
    tt_inputs = _to_device(torch_inputs, device)
    for k, (src, t) in enumerate(zip(torch_inputs, tt_inputs)):
        back = ttnn.to_torch(t).to(torch.bfloat16)
        if not torch.equal(back.view(torch.int16), src.view(torch.int16)):
            pytest.skip(f"harness: the torch-to-device path changed special values of input {k}")
    out = g6_concat(tt_inputs, dim)
    _assert_bit_exact(ttnn.to_torch(out), _reference(torch_inputs, dim), f"{case_id} special values")


def test_rows_of_sixteen_inputs_keep_their_order(device):
    shapes = [[1, 1, 1, 62080]] * 16
    torch_inputs = [torch.full(s, float(k + 1), dtype=torch.bfloat16) for k, s in enumerate(shapes)]
    out = ttnn.to_torch(_run_and_check(torch_inputs, 2, device, label="qwen79 constants"))
    for k in range(len(shapes)):
        assert bool((out[0, 0, k, :] == float(k + 1)).all()), f"output row {k} is not input {k}"


@pytest.mark.parametrize("case_id,shapes,dim,regime", REGIME_CASES, ids=_ids(REGIME_CASES))
def test_plan_selects_regime(case_id, shapes, dim, regime, device):
    torch_inputs = _make_inputs(shapes)
    tt_inputs = _to_device(torch_inputs, device)
    plan = g6_concat_module.select_plan(tt_inputs, dim)
    assert plan["regime"] == regime, f"{case_id}: plan {plan}"
    assert plan["block_pages"] >= 1
    out = g6_concat(tt_inputs, dim)
    _assert_bit_exact(ttnn.to_torch(out), _reference(torch_inputs, dim), case_id)


@pytest.mark.parametrize(
    "case_id,shapes,dim",
    [
        ("qwen79_16x1row_h", [[1, 1, 1, 62080]] * 16, 2),
        ("qwen160_1row_wide_w", [[1, 1, 1, 16384]] * 3 + [[1, 1, 1, 12928]], -1),
    ],
    ids=["qwen79", "qwen160"],
)
def test_plan_spreads_over_full_grid(case_id, shapes, dim, device):
    # One output tile row by 1940 tile columns must not collapse onto one core.
    tt_inputs = _to_device(_make_inputs(shapes), device)
    plan = g6_concat_module.select_plan(tt_inputs, dim)
    grid = device.compute_with_storage_grid_size()
    assert plan["num_cores"] == grid.x * grid.y, f"{case_id}: plan {plan}"


def test_repeat_call_reuses_program_with_new_buffers(device):
    shapes, dim = [[1, 1, 4096, 64], [1, 1, 4096, 192]], -1
    first = _make_inputs(shapes, seed=1)
    first_tt = _to_device(first, device)
    out_first = g6_concat(first_tt, dim)
    _assert_bit_exact(ttnn.to_torch(out_first), _reference(first, dim), "first call")
    entries = device.num_program_cache_entries()

    # New buffers while the old ones are still allocated: a replayed stale address shows here.
    second = _make_inputs(shapes, seed=2)
    second_tt = _to_device(second, device)
    out_second = g6_concat(second_tt, dim)
    _assert_bit_exact(ttnn.to_torch(out_second), _reference(second, dim), "second call")
    assert device.num_program_cache_entries() == entries, "a repeat call with equal shapes built a new program"


def _tt(shape, device, dtype=BF16, layout=TILE):
    torch.manual_seed(42)
    t = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    return ttnn.from_torch(t, dtype=dtype, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)


@pytest.mark.parametrize(
    "case_id",
    [
        "empty",
        "rank_mismatch",
        "dtype_mismatch",
        "layout_mismatch",
        "size_mismatch",
        "dim_too_large",
        "dim_too_small",
        "rank_2",
        "rank_5",
    ],
)
def test_validation_rejects(case_id, device, expect_error):
    if case_id == "empty":
        tensors, dim = [], 0
    elif case_id == "rank_mismatch":
        tensors, dim = [_tt([1, 1, 32, 64], device), _tt([1, 32, 64], device)], -1
    elif case_id == "dtype_mismatch":
        tensors, dim = [_tt([1, 1, 32, 64], device), _tt([1, 1, 32, 64], device, dtype=ttnn.float32)], -1
    elif case_id == "layout_mismatch":
        tensors, dim = [_tt([1, 1, 32, 64], device), _tt([1, 1, 32, 64], device, layout=ttnn.ROW_MAJOR_LAYOUT)], -1
    elif case_id == "size_mismatch":
        tensors, dim = [_tt([1, 1, 32, 64], device), _tt([1, 1, 64, 64], device)], 3
    elif case_id == "dim_too_large":
        tensors, dim = [_tt([1, 1, 32, 64], device)] * 2, 4
    elif case_id == "dim_too_small":
        tensors, dim = [_tt([1, 1, 32, 64], device)] * 2, -5
    elif case_id == "rank_2":
        tensors, dim = [_tt([32, 64], device)] * 2, 0
    else:
        tensors, dim = [_tt([1, 1, 1, 32, 64], device)] * 2, 0
    # Host-side refusals: the message text is the implementer's; only the type is the contract.
    with expect_error((ValueError, RuntimeError), ""):
        g6_concat(tensors, dim)
