# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Refinement 3 regression net: cross-M-block WEIGHT RESIDENCY must be scheduling-only.

Refinement 3's win is that the three bfp4 weight streams are read on M-block 0 ONLY. Every weight
read in the kernels is a pure function of this core's `kstart`/`hstart`/`jstart` with no M-block
index in it, so blocks b > 0 were re-reading bytes that are still sitting in the same L1 slot. The
reader/writer therefore keep the full reserve/push handshake (compute is untouched, bit-for-bit) and
skip only the `BR::read` DRAM loops.

WHY THIS NEEDS ITS OWN TEST — the failure mode is silent, and invisible to every other test here.
Residency rests on an invariant that NOTHING in the type system or the CB machinery enforces:

    a weight CB slot re-reserved on a later M-block still holds the bytes block 0 read into it

That holds today because (a) `cb_pop_front` only advances a read pointer, it never clears the
bytes, (b) each weight CB has exactly ONE producer, and (c) the reserve/push cycle returns the write
pointer to the same slot every M-block — `cb_w_gate`/`cb_w_up` hold one block, and `cb_w_down` holds
exactly `HGROUPS` K-blocks against `HGROUPS` pushes per M-block, so K-block r always lands in slot r.
Break (c) — a different `depth_wd`, a changed per-block push count, a second producer — and M-blocks
b > 0 matmul against the WRONG weight block. There is no hang and no compile error: the output is
simply wrong, and ONLY on the multi-M-block path, which is `count > 256`. A single-M-block test
(which is every graded correctness cell except `count 512`) passes straight through it, because
b == 0 always reads.

So: run each shape with residency OFF and ON and require BIT-IDENTICAL output. Residency moves only
WHEN bytes are fetched, never which numbers are multiplied, so bit-identity is the correct and
sharpest assertion — a PCC gate would hide a wrong-weight-block bug behind this op's bfp4 noise
floor (the whole op sits within 6e-4 of a 0.9797 format ceiling, so "wrong" and "right" are ~2e-2
apart while the gate has ~5e-3 of slack).

Shapes deliberately span 2, 2-with-a-shrunk-tail, and 4 M-blocks: two blocks proves the slot is
re-read, but only a run of MORE than two proves the `cb_w_down` cycle actually CLOSES each block
rather than drifting by a slot per block.

`DEPTH_X` gets the same treatment for a different reason: a second resident-x slot makes the x
multicast's landing address ALTERNATE between two L1 offsets, and mcast_pipe requires that address
to be identical on every core in the grid row. It is (the write pointer is a pure function of the
mailbox words, so all 110 cores compute the same one) — but that is an argument, and this test is
the check.

The knobs are module-level names read inside `create_program_descriptor`, so the test rebinds them
directly (the env vars they default from are only read at import).
"""

import pytest
import torch

import ttnn

from ttnn.operations.moe_fused_swiglu import moe_fused_swiglu
from ttnn.operations.moe_fused_swiglu import moe_fused_swiglu_program_descriptor as pd

HIDDEN = 2048
NUM_GLOBAL_EXPERTS, NUM_LOCAL_EXPERTS, LOCAL_EXPERT_ID, GLOBAL_EXPERT_ID = 256, 8, 3, 137
PADDING_SENTINEL = 100.0

_FORMATS = {"bf16_rm": (ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT), "bfp8_tile": (ttnn.bfloat8_b, ttnn.TILE_LAYOUT)}

# Every shape here MUST span more than one M-block — residency is a no-op at m_blocks == 1, so a
# single-block shape would assert nothing at all. M_BLOCK is 8 tile-rows = 256 tokens.
SHAPES = [
    (7168, 1024, 288, "M_t=9 -> 2 blocks, the second a SHRUNK tail (m_eff 8 then 1)"),
    (7168, 1024, 512, "M_t=16 -> 2 FULL blocks"),
    (6144, 1024, 1024, "M_t=32 -> 4 full blocks: proves the cb_w_down slot cycle CLOSES per block"),
]

# The residency-OFF arms, as knob OVERRIDE GROUPS rather than single knobs, because the L1 budget
# couples them: gate/up residency is what collapses `DEPTH_W` 2 -> 1 and FREES the 155 KB that funds
# `DEPTH_X`'s resident-x slot. Turning W_RESIDENT off while leaving DEPTH_X at 2 asks for both, which
# genuinely does not fit at emb 7168 (measured: 1 692 928 B against a 1 572 864 B budget) and throws
# at program build — the L1 arithmetic in Refinement 3's verifier notes, confirmed on device. So the
# off-arm restores the whole pre-refinement configuration, which is also the more meaningful A/B.
# The W_down arm needs no such pairing: turning it off SHRINKS cb_w_down (depth 11 -> 5).
RESIDENCY_OFF = [
    (
        "gate_up",
        {"W_RESIDENT": 0, "DEPTH_X": 1},
        "re-read W_gate/W_up every M-block (DEPTH_W back to 2, which un-funds the resident-x slot)",
    ),
    ("w_down", {"WD_RESIDENT": 0}, "re-read the phase-2 W_down K stream every M-block (depth_wd back to 5)"),
]


def _build(emb, capacity, count, input_format, device):
    torch.manual_seed(42)
    x = torch.randn((1, 1, capacity, emb), dtype=torch.float32)
    if count < capacity:
        x[:, :, count:, :] = PADDING_SENTINEL  # hostile padding: a leak into a real row is visible
    dt, lay = _FORMATS[input_format]
    tt_x = ttnn.from_torch(
        x.to(torch.bfloat16), dtype=dt, layout=lay, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    tt_w = [
        ttnn.from_torch(
            torch.randn(s, dtype=torch.float32).to(torch.bfloat16),
            dtype=ttnn.bfloat4_b,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        for s in ((emb, HIDDEN), (emb, HIDDEN), (HIDDEN, emb))
    ]
    counts = torch.zeros(NUM_GLOBAL_EXPERTS, dtype=torch.int32)
    counts[GLOBAL_EXPERT_ID] = count
    idx = torch.tensor([(11 + 37 * i) % NUM_GLOBAL_EXPERTS for i in range(NUM_LOCAL_EXPERTS)], dtype=torch.int32)
    idx[LOCAL_EXPERT_ID] = GLOBAL_EXPERT_ID
    to_dev = lambda t: ttnn.from_torch(  # noqa: E731
        t, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    return tt_x, tt_w, to_dev(counts), to_dev(idx)


def _run(args, count):
    tt_x, tt_w, tt_counts, tt_idx = args
    out = moe_fused_swiglu(tt_x, tt_w[0], tt_w[1], tt_w[2], tt_counts, tt_idx, LOCAL_EXPERT_ID)
    return ttnn.to_torch(out)[0, 0, :count, :].float()


@pytest.mark.parametrize("emb, capacity, count, why_shape", SHAPES)
@pytest.mark.parametrize("input_format", list(_FORMATS))
@pytest.mark.parametrize("stream, overrides, what", RESIDENCY_OFF)
def test_weight_residency_is_bit_identical(
    device, emb, capacity, count, why_shape, input_format, stream, overrides, what
):
    """Residency (shipped ON) must match the re-reading path BIT-FOR-BIT on a multi-M-block run."""
    args = _build(emb, capacity, count, input_format, device)
    resident = _run(args, count)  # shipped configuration

    original = {k: getattr(pd, k) for k in overrides}
    try:
        for k, v in overrides.items():
            setattr(pd, k, v)
        rereading = _run(args, count)
    finally:
        for k, v in original.items():
            setattr(pd, k, v)

    assert torch.equal(resident, rereading), (
        f"{stream} residency changed the output on emb={emb} capacity={capacity} count={count} "
        f"({why_shape}) {input_format}. Turning it off {what}, so a mismatch means a later M-block "
        f"read a weight slot that no longer held block 0's bytes: "
        f"max|delta| = {(resident - rereading).abs().max().item()}"
    )


@pytest.mark.parametrize("emb, capacity, count, why_shape", SHAPES)
@pytest.mark.parametrize("input_format", list(_FORMATS))
def test_resident_x_double_buffer_is_bit_identical(device, emb, capacity, count, why_shape, input_format):
    """DEPTH_X=2 alternates the x multicast's landing slot; every core must still agree on it."""
    args = _build(emb, capacity, count, input_format, device)
    doubled = _run(args, count)  # shipped configuration

    original = pd.DEPTH_X
    try:
        pd.DEPTH_X = 1
        single = _run(args, count)
    finally:
        pd.DEPTH_X = original

    assert torch.equal(doubled, single), (
        f"DEPTH_X=2 changed the output on emb={emb} capacity={capacity} count={count} ({why_shape}) "
        f"{input_format}: the alternating cb_x_tiles landing address is no longer identical across "
        f"the grid row. max|delta| = {(doubled - single).abs().max().item()}"
    )
