# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Refinement 2 regression net: every PARKED perf knob must still be CORRECT when turned.

Refinement 2 measured four blocking/dataflow levers and kept all of them as live knobs, three of
them parked at a byte-identical default because they did not pay TODAY:

    REDUCE_SLOTS_CAP      concurrent child landing slots in the reduce tree (lever 1) — parked at 1
    HN_BLOCK              gate/up in1 sub-block width (lever 2)                       — parked at HN_PAD
    OUT_SUBBLOCK_H_DN_MAX `down` output sub-block height (lever 3)                    — parked at 1
    WD_AHEAD              phase-2 W_down prefetch depth                               — shipped at 1

A parked knob is exactly the thing that ROTS: the shipped path never touches it, so the day a later
refinement turns it (Refinement 3 frees the L1 that lever 1 wants, and the deferred read barrier
makes WD_AHEAD live for the first time) it can be silently broken — and each of these knobs changes
a CROSS-CORE protocol, where "broken" means a hang or run-to-run garbage rather than a compile
error. `REDUCE_SLOTS >= 2` switches the reduce tree from one-child-at-a-time to invite WAVES with a
per-child slot stride; `WD_AHEAD >= 2` changes which K-block the deferred `wd_pending` barrier
carries across a round boundary. Neither is exercised by any other test in this directory.

So: run the op at every non-default knob value and require the output to be BIT-IDENTICAL to the
default. These knobs are scheduling-only — they move when bytes are read and how many landing slots
exist, never which numbers are multiplied — so bit-identity is the correct, sharpest assertion.

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

# (knob attribute, value, why it must be exercised)
KNOB_SETTINGS = [
    ("REDUCE_SLOTS_CAP", 2, "lever 1: wave invites + per-child slot stride in the reduce tree"),
    ("HN_BLOCK", 3, "lever 2: 2 in1 sub-blocks, incl. the ragged column's narrowed last sub-block"),
    ("OUT_SUBBLOCK_H_DN_MAX", 4, "lever 3: `down` sub-block height 2 (emb 7168) / 4 (emb 6144)"),
    ("WD_AHEAD", 2, "the deferred read barrier's `wd_pending` carried across a round boundary"),
    ("WD_AHEAD", 3, "same, two blocks of prefetch depth"),
]

# count 288 spans TWO M-blocks with a SHRUNK tail (m_eff 8 then 1), so one dispatch covers the
# multi-block path (the writer's deferred output barrier) and the m_eff shrink at once.
SHAPES = [(7168, 1024, 288), (6144, 1024, 255)]


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


@pytest.mark.parametrize("emb, capacity, count", SHAPES)
@pytest.mark.parametrize("input_format", list(_FORMATS))
@pytest.mark.parametrize("knob, value, why", KNOB_SETTINGS)
def test_parked_knob_is_numerics_invariant(device, emb, capacity, count, input_format, knob, value, why):
    """Turning a parked scheduling knob must not change one bit of the output (nor hang)."""
    args = _build(emb, capacity, count, input_format, device)
    baseline = _run(args, count)

    original = getattr(pd, knob)
    try:
        setattr(pd, knob, value)
        turned = _run(args, count)
    finally:
        setattr(pd, knob, original)

    assert torch.equal(turned, baseline), (
        f"{knob}={value} ({why}) changed the output on "
        f"emb={emb} capacity={capacity} count={count} {input_format}: "
        f"max|delta| = {(turned - baseline).abs().max().item()}"
    )
