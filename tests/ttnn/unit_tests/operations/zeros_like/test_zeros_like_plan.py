# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests op_design.md requires beyond the golden cells.

- Every page written: a freshly allocated buffer on the simulator may already hold zeros, so the golden cells
  do not show that every core (and every writer thread) wrote its pages. Here the output is pre-filled with a
  non-zero value and handed to the program directly.
- Blocks: a small PAGES_PER_BARRIER turns the block knob so cores run several blocks and a ragged last one.
- One device session: an implicit-sync tt-metal Quasar program before, a tt-metal program after (to_layout).
- R2: the input keeps its values and the result is a buffer of its own.
"""

import pathlib
import sys

import pytest
import torch

import ttnn
from ttnn.operations.zeros_like import zeros_like
from ttnn.operations.zeros_like import zeros_like_program_artifacts as artifacts

FILL = 7.0
# tt-metal root / .claude/references/quasar_op_examples (the toy_thread_plan example)
EXAMPLES_DIR = pathlib.Path(__file__).resolve().parents[5] / ".claude" / "references" / "quasar_op_examples"


def _device_tensor(device, shape, value=None, seed=0):
    torch.manual_seed(seed)
    t = torch.full(shape, value, dtype=torch.bfloat16) if value is not None else torch.randn(shape, dtype=torch.bfloat16)
    return t, ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)


def _fill_prefilled(device, shape, thread_plan=None):
    _, out = _device_tensor(device, shape, FILL)
    assert float(ttnn.to_torch(out).float().abs().min()) == FILL  # really pre-filled
    spec, run_args, indices = artifacts.create_program_artifacts(out, thread_plan=thread_plan)
    result = ttnn.generic_op([out], spec, run_args, indices)
    return ttnn.to_torch(result).float()


def _assert_zeros(got, shape):
    assert tuple(got.shape) == tuple(shape)
    nonzero = torch.count_nonzero(got).item()
    assert nonzero == 0, f"{nonzero} non-zero elements left (pages not written)"


# (70,512): 48 pages, ragged split over 32 clusters; (1,1,2048,256): 512 pages, 16 per cluster;
# (2,1,100,47): both non-aligned; (1024,1024): 1024 pages, 32 per cluster.
SHAPES = [(70, 512), (1, 1, 2048, 256), (2, 1, 100, 47), (1024, 1024)]


@pytest.mark.parametrize("writers", [1, 2], ids=lambda w: f"W{w}")
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_every_page_written(device, shape, writers):
    """Every page of a pre-filled output becomes zero: every cluster and every writer thread ran.

    W = 1 is the design's plan; W = 2 (the perf lamp) deals pages t, t+W, ... so a missing thread leaves
    every other page non-zero."""
    plan = {**artifacts.THREAD_PLAN, "writer_threads": writers}
    _assert_zeros(_fill_prefilled(device, shape, plan), shape)


@pytest.mark.parametrize("pages_per_barrier", [1, 3])
def test_several_blocks_per_core(device, monkeypatch, pages_per_barrier):
    """The block knob turned down: 16 pages per cluster in blocks of 1 or 3 (ragged last block of 1)."""
    monkeypatch.setattr(artifacts, "PAGES_PER_BARRIER", pages_per_barrier)
    shape = (1, 1, 2048, 256)
    _, _, ppb = artifacts.block_parameters(2048, 1)
    assert ppb == pages_per_barrier
    _assert_zeros(_fill_prefilled(device, shape), shape)


def test_input_kept_and_result_is_new(device):
    """R2: the result is a new buffer; the input keeps its values. R3/R4: placement, shape, dtype, layout."""
    shape = (70, 512)
    t, x = _device_tensor(device, shape)
    out = zeros_like(x)
    assert out.buffer_address() != x.buffer_address()
    assert torch.equal(ttnn.to_torch(x), t)
    assert out.dtype == x.dtype and out.layout == x.layout and tuple(out.shape) == shape
    assert out.memory_config() == x.memory_config()
    _assert_zeros(ttnn.to_torch(out).float(), shape)


def test_after_an_implicit_sync_program(device):
    """tt-metal's own Quasar softmax (implicit sync: it leaves a ring's tag on its harts), then zeros_like in the
    same session: every write of ours is tagged with the user id.

    Not followed by a tt-metal program here."""
    shape = (1, 1, 64, 256)
    t, x = _device_tensor(device, shape)
    sm = ttnn.experimental.quasar.softmax(x, dim=-1)
    assert torch.allclose(ttnn.to_torch(sm).float(), torch.softmax(t.float(), dim=-1), atol=0.02)
    _assert_zeros(_fill_prefilled(device, (1024, 1024)), (1024, 1024))
    _assert_zeros(ttnn.to_torch(zeros_like(sm)).float(), shape)


def test_one_device_session(device):
    """An implicit-sync program before (the toy_thread_plan example, which hands its tag back), then zeros_like,
    then a tt-metal program after it (to_layout: untagged transfers), and the
    implicit-sync program again, all on one open device."""
    sys.path.insert(0, str(EXAMPLES_DIR))
    from toy_thread_plan import toy_thread_plan

    shape = (1, 3, 320, 320)  # 300 pages: every cluster
    ta, a = _device_tensor(device, shape, seed=1)
    tb, b = _device_tensor(device, shape, seed=2)
    prod = toy_thread_plan(a, b, threads=(4, 4, 2), sync="implicit")
    assert torch.allclose(ttnn.to_torch(prod).float(), (ta * tb).float(), atol=0.05)

    _, prefilled = _device_tensor(device, shape, FILL)
    spec, run_args, indices = artifacts.create_program_artifacts(prefilled)
    out = ttnn.generic_op([prefilled], spec, run_args, indices)
    rm = ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT)
    _assert_zeros(ttnn.to_torch(ttnn.from_device(rm)).float(), shape)

    out2 = zeros_like(prod)
    rm2 = ttnn.to_layout(out2, ttnn.ROW_MAJOR_LAYOUT)
    _assert_zeros(ttnn.to_torch(ttnn.from_device(rm2)).float(), shape)
    prod2 = toy_thread_plan(a, b, threads=(4, 4, 2), sync="explicit")
    assert torch.allclose(ttnn.to_torch(prod2).float(), (ta * tb).float(), atol=0.05)


@pytest.mark.parametrize(
    "kind", ["float32", "row_major", "l1"],
)
def test_refuses_outside_phase0(device, kind):
    shape = (32, 64)
    t = torch.randn(shape)
    if kind == "float32":
        x = ttnn.from_torch(t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    elif kind == "row_major":
        x = ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    else:
        x = ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
                            memory_config=ttnn.L1_MEMORY_CONFIG)
    with pytest.raises(NotImplementedError):
        zeros_like(x)


@pytest.mark.parametrize("sharding", ["height", "width"])
def test_refuses_sharded_input_by_name(device, sharding):
    """R5: the placement axes come from eval.memory.memory_axes, so a sharded input is refused by name. The golden
    suite builds interleaved tensors only (`sharding` has one value), so this test is the op's own (spec R5)."""
    from eval import memory
    from ttnn.operations._op_contract import UnsupportedAxisValue

    shape = (64, 64)
    config = memory.place(memory.build("L1", sharding), shape, ttnn.TILE_LAYOUT, device)
    x = ttnn.from_torch(torch.randn(shape), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
                        memory_config=config)
    assert memory.memory_axes(x) == {"memory_config": "L1", "sharding": sharding}
    with pytest.raises(UnsupportedAxisValue, match="sharding"):
        zeros_like(x)
