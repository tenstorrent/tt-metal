# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""The tests op_design.md's Work Distribution requires of pad, beyond the acceptance test.

- coverage: value 1.0 through the program entry (validate() bypassed: the kernels are value-general), so a
  missing pad write or a missing thread's rows show (simulator DRAM starts zeroed, which hides value 0.0);
- regime pin: block_rows_override forces rows_streamed (several blocks per core, a ring that wraps, a ragged
  last block), the margins of later laps reused;
- session: pad after another program and before one of tt-metal's own, in one device session;
- rejections X1-X4 of the specification.
"""

import pytest
import torch

import ttnn
from ttnn.operations.pad import pad
from ttnn.operations.pad.pad import _run
from ttnn.operations.pad.pad_program_artifacts import PAD_THREADS


def _reference(t, padding, value):
    flat = [amount for pair in reversed(list(padding)) for amount in pair]
    return torch.nn.functional.pad(t, flat, mode="constant", value=value)


def _input(device, shape, seed=0):
    torch.manual_seed(seed)
    t = torch.randn(shape, dtype=torch.bfloat16)
    return t, ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)


def _assert_exact(got, expected):
    assert got.shape == expected.shape, (got.shape, expected.shape)
    bad = (got != expected).sum().item()
    assert bad == 0, f"{bad} elements differ"


CASES = [
    ((2, 4, 128, 512), ((0, 1), (0, 0), (0, 0), (0, 0))),  # N: a third of the output is pad rows
    ((4, 8, 47, 256), ((0, 2), (0, 3), (0, 1), (0, 4))),  # every dimension
    ((1, 3, 20, 20), ((0, 0), (0, 13), (0, 0), (0, 0))),  # YOLO stem
    ((1, 1, 64, 17), ((0, 0), (0, 0), (0, 3), (0, 5))),  # H and W, odd bf16 tail offset
]


def _case_id(case):
    shape, padding = case
    return "x".join(map(str, shape)) + "_p" + "_".join(f"{b}+{a}" for b, a in padding)


@pytest.mark.parametrize("threads", [PAD_THREADS, (1, 2), (2, 1), (2, 2)], ids=lambda t: "R%dW%d" % t)
@pytest.mark.parametrize("case", CASES, ids=_case_id)
def test_coverage_value_one(device, case, threads):
    """Every pad row, every margin and every thread's data rows are written: value 1.0 makes a missing write
    visible, and data rows dealt to a thread that did not run would stay zero."""
    shape, padding = case
    t, x = _input(device, shape)
    got = ttnn.to_torch(_run(x, padding, 1.0, threads=threads))
    _assert_exact(got, _reference(t, padding, 1.0))


@pytest.mark.parametrize("threads", [PAD_THREADS, (1, 2), (2, 1)], ids=lambda t: "R%dW%d" % t)
@pytest.mark.parametrize("block_rows", [1, 3])
@pytest.mark.parametrize("case", CASES[:3], ids=_case_id)
def test_rows_streamed_regime(device, case, block_rows, threads):
    """block_rows_override forces rows_streamed: ring_blocks = BUFFER_DEPTH, several blocks per core, the ring
    wraps (margins filled on the first lap only are reused) and the last block per core is ragged."""
    shape, padding = case
    t, x = _input(device, shape)
    got = ttnn.to_torch(_run(x, padding, 1.0, block_rows_override=block_rows, threads=threads))
    _assert_exact(got, _reference(t, padding, 1.0))


def test_pad_rows_in_several_barriers(device):
    """pad_rows_per_barrier below the core's pad count: several write barriers per core."""
    shape, padding = (2, 4, 128, 512), ((0, 1), (0, 0), (0, 0), (0, 0))
    t, x = _input(device, shape)
    got = ttnn.to_torch(_run(x, padding, 1.0, pad_rows_per_barrier=3))
    _assert_exact(got, _reference(t, padding, 1.0))


def test_session_between_programs(device):
    """In one device session: a tt-metal program, then pad, then a tt-metal program on pad's output.

    The writer ends on implicit (ring-tagged) writes; it hands the tag register back. The other programs are tt-metal's
    tilize / untilize (`ttnn.to_layout`), which run on this part (a Metal 1 op such as ttnn.add does not).
    """
    torch.manual_seed(1)
    a = torch.randn(1, 1, 64, 64, dtype=torch.bfloat16)
    rm = ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    _assert_exact(ttnn.to_torch(ttnn.to_layout(rm, ttnn.TILE_LAYOUT)), a)  # a program before pad

    for seed, (shape, padding) in enumerate([CASES[0], CASES[2], ((1, 1, 32, 64), ((0, 0), (0, 0), (0, 0), (0, 32)))]):
        t, x = _input(device, shape, seed)
        y = pad(x, padding, 0.0)
        expected = _reference(t, padding, 0.0)
        _assert_exact(ttnn.to_torch(y), expected)
        # tt-metal's own program on pad's output, then back
        tiled = ttnn.to_layout(y, ttnn.TILE_LAYOUT)
        _assert_exact(ttnn.to_torch(ttnn.to_layout(tiled, ttnn.ROW_MAJOR_LAYOUT)), expected)

    _assert_exact(ttnn.to_torch(ttnn.to_layout(ttnn.to_layout(rm, ttnn.TILE_LAYOUT), ttnn.ROW_MAJOR_LAYOUT)), a)


@pytest.mark.parametrize(
    "shape,padding,raises",
    [
        ((3, 20, 20), ((0, 0), (0, 13), (0, 0), (0, 0)), ValueError),  # X1: more pairs than dims
        ((1, 3, 20, 20), ((0, 13), (0, 0), (0, 0)), ValueError),  # X2: fewer pairs
        ((1, 3, 20, 20), ((0, 0), (0, -1), (0, 0), (0, 0)), ValueError),  # X3: negative
        ((1, 3, 20, 20), (0, 0, 0, 0, 0, 13, 0, 0), TypeError),  # X4: torch's flat form
    ],
    ids=["X1", "X2", "X3", "X4"],
)
def test_rejections(device, shape, padding, raises):
    _, x = _input(device, shape)
    with pytest.raises(raises):
        pad(x, padding, 0.0)


def test_outside_phase0_is_refused(device):
    """Valid calls outside SUPPORTED raise NotImplementedError (R7)."""
    _, x = _input(device, (1, 1, 32, 64))
    with pytest.raises(NotImplementedError):
        pad(x, ((0, 0), (0, 0), (1, 0), (0, 0)), 0.0)  # padding before
    with pytest.raises(NotImplementedError):
        pad(x, ((0, 0), (0, 0), (0, 1), (0, 0)), 1.0)  # value 1.0


# Entries wider than one NoC packet (NOC_PACKET_BYTES): drained by the explicit-sync writer.
WIDE_CASES = [
    ((1, 1, 128, 4097), ((0, 0), (0, 1), (0, 2), (0, 3))),  # bf16 entry 8208 B: just past the packet
    ((1, 1, 128, 4096), ((0, 0), (0, 0), (0, 0), (0, 0))),  # bf16 entry 8192 B: implicit, the boundary itself
]


@pytest.mark.parametrize("threads", [PAD_THREADS, (1, 2), (2, 2)], ids=lambda t: "R%dW%d" % t)
@pytest.mark.parametrize("block_rows", [None, 1])
@pytest.mark.parametrize("case", WIDE_CASES, ids=_case_id)
def test_rows_wider_than_a_packet(device, case, block_rows, threads):
    shape, padding = case
    t, x = _input(device, shape)
    got = ttnn.to_torch(_run(x, padding, 1.0, block_rows_override=block_rows, threads=threads))
    _assert_exact(got, _reference(t, padding, 1.0))


def test_rows_wider_than_a_packet_refuse_more_readers(device):
    """An explicit consumer rotates per pop (a whole reader op), so the wide-row path needs W >= R."""
    _, x = _input(device, WIDE_CASES[0][0])
    with pytest.raises(ValueError):
        _run(x, WIDE_CASES[0][1], 1.0, threads=(2, 1))


def test_sharded_input_is_refused_by_name(device):
    """R6: the placement axes come from eval.memory.memory_axes and are gated like any other axis, so a sharded
    input is refused with the pipeline's refusal (a NotImplementedError naming the axis) before any kernel work."""
    torch.manual_seed(0)
    t = torch.randn((1, 1, 32, 64), dtype=torch.bfloat16)
    shard = ttnn.ShardSpec(
        ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))]),
        [32, 64],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard)
    x = ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mc)
    from eval.memory import memory_axes
    from ttnn.operations.pad.pad import SUPPORTED

    assert memory_axes(x)["sharding"] == "height"
    assert "height" not in SUPPORTED["sharding"]
    # validate() stops at the first unsupported axis: memory_config=L1 until L1 is supported, sharding after.
    with pytest.raises(NotImplementedError, match="sharding|memory_config"):
        pad(x, ((0, 0), (0, 0), (0, 0), (0, 32)), 0.0)
