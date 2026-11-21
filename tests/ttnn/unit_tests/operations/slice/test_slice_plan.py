# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests the design requires beyond the acceptance test (op_design.md, Work Distribution):

- every thread of the plan ran: two calls in one session on different inputs, each exact; the plan turned
  through the private `_slice(threads=...)` so the strided dealing is exercised;
- session order: a tt-metal program before slice, and slice before a tt-metal program (`to_layout`);
- R7: the output sits in the input's buffer, interleaved;
- regime selection (Refinement 1): a TILE block narrowing W stays `page_relay`, a row-major one is
  `rm_column_cut`, each pinned per placement, and the column cut accepts any byte offset into a row.
"""

import pytest
import torch

import ttnn
from ttnn.operations.slice.slice import _dispatch, _slice
from ttnn.operations.slice.slice_program_artifacts import (
    PAGE_RELAY,
    RM_COLUMN_CUT,
    regime_for,
    uses_explicit_sync,
)

# (shape, start, end): the YOLO calls and a ragged multi-page case per layout.
CASES = [
    ((1, 144, 400), (0, 64, 0), (1, 144, 400)),
    ((1, 144, 8400), (0, 0, 0), (1, 64, 8400)),
    ((4, 128, 47), (0, 64, 0), (4, 100, 47)),
    ((2, 512, 1024), (0, 256, 0), (2, 512, 1024)),
    # Row-major pages over one 8 KiB NoC packet take the ring's explicit protocol: several blocks per
    # cluster with a ragged last one (144 rows: 4 or 5 per cluster; 320 rows: 10 per cluster).
    ((1, 144, 8400), (0, 0, 0), (1, 144, 8400)),
    ((1, 352, 4128), (0, 32, 0), (1, 352, 4128)),
    # W narrowed (Refinement 1): TILE stays page_relay; row-major is rm_column_cut (explicit sync, a byte
    # range of every row), one with several blocks per cluster and one whose input rows are 8 KiB.
    ((4, 128, 512), (0, 32, 96), (4, 96, 352)),
    ((1, 32, 4096), (0, 0, 1024), (1, 32, 3072)),
]
PLANS = [(1, 1), (2, 2), (2, 1), (3, 3), (4, 2)]  # W > R is refused (slice_program_artifacts.py)


def _to_device(device, t, layout, mem):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=mem)


def _expected(t, start, end):
    return t[tuple(slice(s, e) for s, e in zip(start, end))]


def _explicit_sync(x, start, end):
    out_shape = [e - s for s, e in zip(start, end)]
    return uses_explicit_sync(x.buffer_aligned_page_size(), regime_for(x, list(start), out_shape))


def _exact(out, expected):
    got = ttnn.to_torch(out)
    assert list(got.shape) == list(expected.shape), (got.shape, expected.shape)
    assert torch.equal(got, expected), (got.float() - expected.float()).abs().max()


@pytest.mark.parametrize("threads", PLANS, ids=lambda p: "R%dW%d" % p)
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["tile", "rm"])
@pytest.mark.parametrize("case", CASES, ids=lambda c: "x".join(map(str, c[0])))
def test_every_thread_runs(device, case, layout, threads):
    """Two different inputs of the same shape into fresh outputs: a hart that did not run would leave its
    pages unwritten (or stale from the first call), and the second result would not be exact."""
    shape, start, end = case
    probe = _to_device(device, torch.zeros(shape, dtype=torch.bfloat16), layout, ttnn.DRAM_MEMORY_CONFIG)
    if _explicit_sync(probe, start, end) and threads[0] != threads[1]:
        with pytest.raises(ValueError, match="R == W"):
            _slice(probe, start, end, threads=threads)
        return
    for seed in (0, 1):
        torch.manual_seed(seed)
        t = torch.randn(shape, dtype=torch.bfloat16)
        x = _to_device(device, t, layout, ttnn.DRAM_MEMORY_CONFIG)
        _exact(_slice(x, start, end, threads=threads), _expected(t, start, end))


def test_a_tt_metal_program_precedes(device):
    """tt-metal's own program (to_layout: tilize) runs first, then slice, in one session."""
    torch.manual_seed(2)
    t = torch.randn((4, 128, 512), dtype=torch.bfloat16)
    x = ttnn.to_layout(_to_device(device, t, ttnn.ROW_MAJOR_LAYOUT, ttnn.L1_MEMORY_CONFIG), ttnn.TILE_LAYOUT)
    _exact(ttnn.operations.slice.slice(x, [0, 32, 0], [4, 96, 512]), t[:, 32:96, :])


def test_a_tt_metal_program_follows(device):
    """slice (implicit-sync tagged transfers) then tt-metal's untilize, whose plain transfers would inherit a
    ring's tag if a hart did not hand it back."""
    torch.manual_seed(3)
    t = torch.randn((1, 144, 400), dtype=torch.bfloat16)
    x = _to_device(device, t, ttnn.TILE_LAYOUT, ttnn.L1_MEMORY_CONFIG)
    out = ttnn.operations.slice.slice(x, [0, 64, 0], [1, 144, 400])
    _exact(ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT), t[:, 64:144, :])
    # And once more after that, for an RM input in DRAM.
    t2 = torch.randn((1, 144, 8400), dtype=torch.bfloat16)
    x2 = _to_device(device, t2, ttnn.ROW_MAJOR_LAYOUT, ttnn.DRAM_MEMORY_CONFIG)
    _exact(ttnn.operations.slice.slice(x2, (0, 0, 0), (1, 64, 8400)), t2[:, 0:64, :])


@pytest.mark.parametrize("threads", [(1, 1), (2, 2)], ids=lambda p: "R%dW%d" % p)
def test_a_tt_metal_program_follows_explicit_sync(device, threads):
    """The explicit-sync path (row-major pages over 8 KiB: YOLO's traced [1, 144, 8400] call) followed by
    tt-metal's tilize and untilize in the same session: its user-trid transfers must leave nothing behind."""
    torch.manual_seed(4)
    t = torch.randn((1, 144, 8400), dtype=torch.bfloat16)
    x = _to_device(device, t, ttnn.ROW_MAJOR_LAYOUT, ttnn.DRAM_MEMORY_CONFIG)
    assert uses_explicit_sync(x.buffer_aligned_page_size())
    out = _slice(x, [0, 64, 0], [1, 144, 8400], threads=threads)
    tiled = ttnn.to_layout(out, ttnn.TILE_LAYOUT)
    _exact(ttnn.to_layout(tiled, ttnn.ROW_MAJOR_LAYOUT), t[:, 64:144, :])


@pytest.mark.parametrize("mem", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG], ids=["dram", "l1"])
def test_output_in_the_inputs_buffer(device, mem):
    """R7: the output is interleaved in the input's buffer type."""
    t = torch.randn((1, 96, 128), dtype=torch.bfloat16)
    x = _to_device(device, t, ttnn.TILE_LAYOUT, mem)
    out = ttnn.operations.slice.slice(x, [0, -64, 0], [1, -32, 128])
    assert out.memory_config().buffer_type == mem.buffer_type
    assert out.memory_config().memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
    _exact(out, t[:, 32:64, :])


MEMS = [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG]


@pytest.mark.parametrize("mem", MEMS, ids=["dram", "l1"])
@pytest.mark.parametrize(
    "layout,regime",
    [(ttnn.TILE_LAYOUT, PAGE_RELAY), (ttnn.ROW_MAJOR_LAYOUT, RM_COLUMN_CUT)],
    ids=["tile_page_relay", "rm_column_cut"],
)
def test_w_narrowed_regime(device, layout, regime, mem):
    """Regime pin (Refinement 1): rows and columns narrowed from a tile boundary. TILE whole tiles relay as
    pages (the page map's W tile axis); row-major rows are cut to their byte range."""
    shape, start, end = (4, 128, 512), (0, 32, 96), (4, 96, 352)
    torch.manual_seed(5)
    t = torch.randn(shape, dtype=torch.bfloat16)
    x = _to_device(device, t, layout, mem)
    assert regime_for(x, list(start), [e - s for s, e in zip(start, end)]) == regime
    _exact(ttnn.operations.slice.slice(x, list(start), list(end)), _expected(t, start, end))


# Row cuts whose byte offset (and length) is not a multiple of the 16-B L1 / 64-B DRAM alignment: start
# inside a tile, which Refinement 2 adds to SUPPORTED on top of this path, so they go through the program
# directly (_dispatch takes normalized bounds and skips the support gate).
UNALIGNED_ROW_CUTS = [
    ((1, 32, 50), (0, 0, 17), (1, 32, 50)),  # 34-B offset, 66-B rows
    ((4, 8, 32, 47), (0, 0, 0, 7), (4, 8, 32, 39)),  # 14-B offset, 64-B rows of 94-B input rows
    ((1, 1, 64, 1024), (0, 0, 0, 0), (1, 1, 64, 1000)),  # offset 0, 2,000-B rows (ResNet-50)
    ((1, 32, 8400), (0, 0, 3), (1, 32, 8397)),  # 6-B offset into 16,800-B rows
]


@pytest.mark.parametrize("threads", [(1, 1), (2, 2)], ids=lambda p: "R%dW%d" % p)
@pytest.mark.parametrize("mem", MEMS, ids=["dram", "l1"])
@pytest.mark.parametrize("case", UNALIGNED_ROW_CUTS, ids=lambda c: "x".join(map(str, c[0])) + "_w%d" % c[1][-1])
def test_rm_column_cut_any_byte_offset(device, case, mem, threads):
    shape, start, end = case
    torch.manual_seed(6)
    t = torch.randn(shape, dtype=torch.bfloat16)
    x = _to_device(device, t, ttnn.ROW_MAJOR_LAYOUT, mem)
    assert regime_for(x, list(start), [e - s for s, e in zip(start, end)]) == RM_COLUMN_CUT
    _exact(_dispatch(x, start, end, threads), _expected(t, start, end))


def test_rm_column_cut_between_tt_metal_programs(device):
    """Session order for rm_column_cut (explicit sync, user trid 0): tt-metal's untilize runs before it, and
    its tilize and untilize after it, in one session."""
    torch.manual_seed(7)
    t = torch.randn((4, 128, 512), dtype=torch.bfloat16)
    x = ttnn.to_layout(_to_device(device, t, ttnn.TILE_LAYOUT, ttnn.L1_MEMORY_CONFIG), ttnn.ROW_MAJOR_LAYOUT)
    out = ttnn.operations.slice.slice(x, [0, 32, 96], [4, 96, 352])
    tiled = ttnn.to_layout(out, ttnn.TILE_LAYOUT)
    _exact(ttnn.to_layout(tiled, ttnn.ROW_MAJOR_LAYOUT), t[:, 32:96, 96:352])


def test_slice_step_argument(device):
    """R9 / the specification's stub: `slice_step` is the fourth parameter. None is the Phase 0 block;
    a step other than 1 reads as the `step` value "other" and is refused by name (NotImplementedError)."""
    from ttnn.operations.slice import slice as slice_op

    torch_input = torch.randn(1, 64, 128, dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)
    out = slice_op(tt_input, [0, 32, 0], [1, 64, 128], None)
    assert torch.equal(ttnn.to_torch(out), torch_input[:, 32:64, :])
    with pytest.raises(NotImplementedError, match="step"):
        slice_op(tt_input, [0, 32, 0], [1, 64, 128], [1, 2, 1])


# Refinement 10: the rank-4 calls of the other three Quasar models of tt-metal's merge gate, as they make them
# (the public entry point, bounds as lists, `slice_step` as the fourth positional argument where they pass it).
# (name, shape, layout, start, end, slice_step)
MODEL_CALLS = [
    ("qwen_tile_row", (1, 1, 128, 512), ttnn.TILE_LAYOUT, [0, 0, 64, 0], [1, 1, 96, 512], None),
    ("qwen_whole_step", (1, 1, 1, 128), ttnn.TILE_LAYOUT, [0, 0, 0, 0], [1, 1, 1, 128], [1, 1, 1, 1]),
    ("llama_user_step", (1, 32, 1, 64), ttnn.TILE_LAYOUT, [0, 0, 0, 0], [1, 1, 1, 64], [1, 1, 1, 1]),
    ("resnet_logits", (1, 1, 32, 1024), ttnn.ROW_MAJOR_LAYOUT, [0, 0, 0, 0], [1, 1, 32, 1000], None),
    ("resnet_hw_unpad", (1, 1, 64, 64), ttnn.ROW_MAJOR_LAYOUT, [0, 0, 0, 0], [1, 1, 30, 62], None),
]


@pytest.mark.parametrize("mem", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG], ids=["dram", "l1"])
def test_model_calls(device, mem):
    """Llama's head cut (every head of an [8, 8, 32, 64] TILE KV cache, in one session), Qwen's tile-row,
    ResNet-50's two row-major unpaddings, and the calls that pass `slice_step=[1, 1, 1, 1]`: each exact."""
    from ttnn.operations.slice import slice as slice_op

    cache = torch.randn((8, 8, 32, 64), dtype=torch.bfloat16)
    x = _to_device(device, cache, ttnn.TILE_LAYOUT, mem)
    for k in range(8):
        _exact(slice_op(x, [0, k, 0, 0], [8, k + 1, 32, 64]), cache[:, k : k + 1])
    for name, shape, layout, start, end, step in MODEL_CALLS:
        t = torch.randn(shape, dtype=torch.bfloat16)
        out = slice_op(_to_device(device, t, layout, mem), start, end, step)
        assert out.layout == layout, name
        _exact(out, _expected(t, start, end))
