# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for g6_permute regime nhwc_tile_bfp8_transpose (Refinement 2) and for buffer "l1"
on the two earlier regimes.

Cell: bfloat8_b, TILE, DRAM or L1 interleaved, rank 4, dims (0, 3, 1, 2): out[n, c, h, w] = x[n, h, w, c].
Graded with the golden suite's bfloat8_b gate: every logical value within one quantization step
2^(E - 133) of the host rendering of the permuted logical input, E the largest exponent of its output
group of 16 along w. Besides Panoptic p3, the regime must hold for every shape validate() accepts:
H not a multiple of 32 (short last group), C not a multiple of 32 (output tiles skipped), W not a
multiple of 32 (input padding rows land in live output groups, so they are poisoned here), N > 1,
several groups per core.
"""

import importlib

import pytest
import torch
import ttnn

from eval.golden_tests.g6_permute.helpers import (
    _bf8_host_rendering,
    check_bf8_one_step,
    check_bf8_shared_exponent,
    make_input,
)
from models.common.utility_functions import comp_pcc
from ttnn.operations._op_contract import ExcludedCell
from ttnn.operations.g6_permute import g6_permute, g6_permute_plan

op_module = importlib.import_module("ttnn.operations.g6_permute.g6_permute")
nhwc_module = importlib.import_module("ttnn.operations.g6_permute.g6_permute_nhwc_tile_bfp8")

NHWC = (0, 3, 1, 2)
ROWS = (0, 2, 1, 3)
BLOCKS = (0, 1, 4, 2, 5, 3)

MEMORY_CONFIG = {"dram": ttnn.DRAM_MEMORY_CONFIG, "l1": ttnn.L1_MEMORY_CONFIG}
BUFFER_TYPE = {"dram": ttnn.BufferType.DRAM, "l1": ttnn.BufferType.L1}

PANOPTIC_P3 = (1, 128, 256, 32)

# id, shape [N, H, W, C], buffer, poison for the input's tile padding (None: from_torch's zeros)
CELL_SHAPES = [
    ("one_group", (1, 32, 32, 32), "dram", None),
    ("h_tail_batch2_c64", (2, 40, 64, 64), "dram", None),  # short last group, 2 channel tiles
    ("c_non_aligned", (1, 32, 64, 40), "l1", None),  # tag w_non_aligned: 8 live channels in tile 1
    ("w_non_aligned_poison", (1, 32, 40, 32), "dram", 1.0e4),  # tag h_non_aligned: rows 8-31 padding
    ("w_tail_above_16_poison", (1, 8, 56, 32), "l1", float("inf")),  # padding rows only in faces 2/3
    ("both_non_aligned_poison", (1, 5, 40, 72), "dram", -3.0e4),  # tag w_non_aligned, W padding too
    ("many_groups", (2, 128, 512, 64), "dram", None),  # 256 groups: several per core, depth 2
]


def _shape_id(shape):
    return "x".join(str(d) for d in shape)


def _check_placement(ttnn_output, shape, dtype, layout, buffer):
    assert list(ttnn_output.shape) == list(shape)
    assert ttnn_output.dtype == dtype
    assert ttnn_output.layout == layout
    memory_config = ttnn_output.memory_config()
    assert memory_config.buffer_type == BUFFER_TYPE[buffer]
    assert memory_config.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED


def _bf8_input(torch_input, device, buffer, poison=None):
    ttnn_input = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=MEMORY_CONFIG[buffer]
    )
    if poison is not None:
        ttnn_input = ttnn.fill_implicit_tile_padding(ttnn_input, poison, memory_config=ttnn_input.memory_config())
    return ttnn_input


def _permute_bf8_and_check(ttnn_input, buffer, call_dims=NHWC, shared_exponent=False):
    logical = ttnn.to_torch(ttnn_input).to(torch.float32)
    permuted = torch.permute(logical, NHWC).contiguous()
    expected = _bf8_host_rendering(permuted)
    ttnn_output = g6_permute(ttnn_input, call_dims)
    _check_placement(ttnn_output, permuted.shape, ttnn.bfloat8_b, ttnn.TILE_LAYOUT, buffer)
    actual = ttnn.to_torch(ttnn_output).to(torch.float32)
    check_bf8_one_step(actual, expected, permuted)
    if shared_exponent:
        check_bf8_shared_exponent(actual, permuted)
    return actual


@pytest.mark.parametrize("buffer", ["l1", "dram"])
def test_panoptic_p3(buffer, device):
    torch.manual_seed(53)
    torch_input = make_input(PANOPTIC_P3, NHWC, "randn")
    _permute_bf8_and_check(_bf8_input(torch_input, device, buffer), buffer)


@pytest.mark.parametrize("buffer", ["l1", "dram"])
def test_panoptic_p3_precision_record(buffer, device):
    """PCC and relative RMS of p3 against the host bfloat8_b rendering and against the permuted
    logical input, printed for the changelog; the gate is the one-step check."""
    torch.manual_seed(53)
    torch_input = make_input(PANOPTIC_P3, NHWC, "randn")
    ttnn_input = _bf8_input(torch_input, device, buffer)
    actual = _permute_bf8_and_check(ttnn_input, buffer)
    permuted = torch.permute(ttnn.to_torch(ttnn_input).to(torch.float32), NHWC).contiguous()
    for name, reference in (("host_rendering", _bf8_host_rendering(permuted)), ("permuted_input", permuted)):
        _, pcc = comp_pcc(reference, actual, pcc=0.0)
        rel_rms = float((actual - reference).pow(2).mean().sqrt() / reference.std())
        exact = float((actual == reference).float().mean())
        print(f"\nPRECISION p3 {buffer} vs {name}: pcc={pcc} rel_rms={rel_rms:.3e} exact_fraction={exact:.4f}")


@pytest.mark.parametrize("input_gen", ["w_scaled", "bf8_odd_grid"])
def test_panoptic_p3_adversarial_groups(input_gen, device):
    torch.manual_seed(53)
    torch_input = make_input(PANOPTIC_P3, NHWC, input_gen)
    _permute_bf8_and_check(_bf8_input(torch_input, device, "l1"), "l1", shared_exponent=input_gen == "bf8_odd_grid")


@pytest.mark.parametrize("shape, buffer, poison", [c[1:] for c in CELL_SHAPES], ids=[c[0] for c in CELL_SHAPES])
def test_other_shapes_of_the_cell(shape, buffer, poison, device):
    torch.manual_seed(42)
    torch_input = make_input(shape, NHWC, "randn")
    _permute_bf8_and_check(_bf8_input(torch_input, device, buffer, poison), buffer)


def test_odd_grid_with_poisoned_w_padding(device):
    """Exponent pinning on a W non-aligned shape: a padding value that leaked into a live group would
    raise its exponent and lose the odd probe steps."""
    torch.manual_seed(5)
    shape = (1, 32, 40, 32)
    torch_input = make_input(shape, NHWC, "bf8_odd_grid")
    _permute_bf8_and_check(_bf8_input(torch_input, device, "dram", 1.0e4), "dram", shared_exponent=True)


def test_tiny_exponent_group(device):
    """Groups whose shared exponent sits at the bottom of the range unpack to bfloat16 denormals."""
    torch.manual_seed(9)
    shape = (1, 32, 32, 32)
    torch_input = (torch.randn(shape) * 2.0**-124).to(torch.bfloat16)
    _permute_bf8_and_check(_bf8_input(torch_input, device, "dram"), "dram")


def test_negative_dims_equal_canonical(device):
    torch.manual_seed(42)
    torch_input = make_input((1, 32, 64, 32), NHWC, "randn")
    ttnn_input = _bf8_input(torch_input, device, "dram")
    canonical = _permute_bf8_and_check(ttnn_input, "dram")
    negative = _permute_bf8_and_check(ttnn_input, "dram", call_dims=(0, -1, 1, -2))
    assert torch.equal(canonical, negative)


def test_plan_reports_regime(device):
    torch.manual_seed(42)
    ttnn_input = _bf8_input(make_input(PANOPTIC_P3, NHWC, "randn"), device, "l1")
    plan = g6_permute_plan(ttnn_input, NHWC)
    assert plan.regime == "nhwc_tile_bfp8_transpose"
    assert plan.tensor_groups == 1 * 4 * 8 * 1
    assert plan.num_cores == len(plan.assignment)
    assert sum(n for _, _, n in plan.assignment) == plan.tensor_groups
    assert plan.out_l1_bytes_per_core > 0
    assert plan.l1_bytes_per_core + plan.out_l1_bytes_per_core <= nhwc_module.L1_OP_BUDGET_BYTES


@pytest.mark.parametrize(
    "dtype, dims, shape",
    [
        (ttnn.bfloat16, NHWC, (1, 32, 64, 32)),
        (ttnn.bfloat8_b, ROWS, (1, 6, 16, 64)),
        (ttnn.bfloat8_b, BLOCKS, (1, 4, 6, 32, 2, 2)),
    ],
    ids=["bf16_tile_nhwc", "bf8_rows", "bf8_blocks"],
)
def test_cells_without_regime_are_excluded(dtype, dims, shape, device, expect_error):
    torch_input = torch.randn(shape).to(torch.bfloat16)
    ttnn_input = ttnn.from_torch(
        torch_input, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    with expect_error(ExcludedCell, ""):
        g6_permute(ttnn_input, dims)


def test_repeat_call_reuses_program(device):
    torch.manual_seed(42)
    shape = (1, 32, 64, 32)
    first = _bf8_input(make_input(shape, NHWC, "randn"), device, "dram")
    second = _bf8_input(make_input(shape, NHWC, "randn"), device, "dram")
    out_first = _permute_bf8_and_check(first, "dram")
    entries = device.num_program_cache_entries()
    out_again = _permute_bf8_and_check(first, "dram")
    _permute_bf8_and_check(second, "dram")
    assert device.num_program_cache_entries() == entries
    assert torch.equal(out_again, out_first)


@pytest.fixture
def group_depth(monkeypatch):
    def force(depth):
        monkeypatch.setattr(nhwc_module, "GROUP_DEPTH", depth)

    op_module._PROGRAM_CACHE.clear()
    yield force
    op_module._PROGRAM_CACHE.clear()


@pytest.mark.parametrize("depth", [1, 3])
def test_forced_group_depth(depth, device, group_depth):
    group_depth(depth)
    torch.manual_seed(7)
    shape = (2, 128, 512, 64)
    ttnn_input = _bf8_input(make_input(shape, NHWC, "randn"), device, "dram")
    plan = g6_permute_plan(ttnn_input, NHWC)
    assert plan.groups_per_core_g1 > 1
    assert plan.group_depth == min(depth, plan.groups_per_core_g1)
    _permute_bf8_and_check(ttnn_input, "dram")


@pytest.fixture
def read_chunk(monkeypatch):
    def force(tiles):
        monkeypatch.setattr(nhwc_module, "READ_CHUNK_TILES", tiles)

    op_module._PROGRAM_CACHE.clear()
    yield force
    op_module._PROGRAM_CACHE.clear()


@pytest.mark.parametrize("tiles", [1, 32])
@pytest.mark.parametrize(
    "shape, poison",
    [((2, 40, 64, 64), None), ((1, 5, 40, 72), -3.0e4)],
    ids=["h_tail_8_live", "h_tail_5_live_w_poison"],
)
def test_forced_read_chunk(tiles, shape, poison, device, read_chunk):
    """The reader's push granularity is a live knob: per-tile chunks and whole-group pushes give
    the same output, including chunks past a short last tile row and zeroed W padding rows."""
    read_chunk(tiles)
    torch.manual_seed(11)
    ttnn_input = _bf8_input(make_input(shape, NHWC, "randn"), device, "dram", poison)
    assert g6_permute_plan(ttnn_input, NHWC).read_chunk_tiles == tiles
    _permute_bf8_and_check(ttnn_input, "dram")


# --- buffer "l1" on the bfloat16 regimes (bit-exact) ----------------------------------------------


def _bits(t):
    return t.to(torch.bfloat16).contiguous().view(torch.int16)


@pytest.mark.parametrize(
    "shape, layout, dims",
    [
        ((1, 6, 16, 64), ttnn.TILE_LAYOUT, ROWS),  # Qwen 303
        ((2, 34, 47, 96), ttnn.TILE_LAYOUT, ROWS),
        ((1, 32, 64, 32, 2, 2), ttnn.ROW_MAJOR_LAYOUT, BLOCKS),  # FLUX 73
        ((1, 3, 5, 7, 3, 3), ttnn.ROW_MAJOR_LAYOUT, BLOCKS),  # 6 B input pages, 14 B output rows
    ],
    ids=["qwen_303", "rows_every_axis_past_a_tile", "flux_73", "blocks_odd_j_odd_w"],
)
def test_bf16_regimes_in_l1(shape, layout, dims, device):
    torch.manual_seed(42)
    torch_input = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    expected = torch.permute(torch_input, dims).contiguous()
    ttnn_input = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    plan = g6_permute_plan(ttnn_input, dims)
    ttnn_output = g6_permute(ttnn_input, dims)
    _check_placement(ttnn_output, expected.shape, ttnn.bfloat16, layout, "l1")
    actual = ttnn.to_torch(ttnn_output)
    mismatch = _bits(actual) != _bits(expected)
    assert not bool(mismatch.any()), f"{int(mismatch.sum())} of {mismatch.numel()} values differ ({plan.regime})"
