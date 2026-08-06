# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""The three fusion knobs: `output=`, `expert_region_offsets=`, `read_x_at_offset=`.

Same three as `ttnn.experimental.deepseek_prefill.unified_routed_expert_ffn`, and the point of them
is the same: a routed-expert MoE dispatch hands every expert a SLICE of one shared token buffer, so
the standalone op had to be wrapped in `ttnn.extract` (copy my rows out) and `ttnn.insert` (copy my
rows back), two whole DRAM round-trips per expert. With the offsets tensor the reader rebases its x
reads and the writer rebases its output writes, and both copies disappear.

WHAT EACH TEST PINS, because a region bug is an off-by-one that a PCC check alone will not see:

  * `test_preallocated_output` — the buffer handed in IS the buffer returned and written, and its
    contents match the allocating path. Both output dtypes.
  * `test_region_fusion` — E experts in ONE shared x buffer and ONE shared output buffer, looped.
    Three independent assertions per expert:
      1. numerics on [start, start+count) against torch (did I read the right rows?),
      2. every output row OUTSIDE this expert's written prefix still holds the pre-fill SENTINEL
         (did I write only my rows?) — this is the assertion that catches a wrong offset, because a
         wrong offset lands somewhere and that somewhere is checked,
      3. neighbouring x regions carry a hostile sentinel, so reading past a region edge shows up as
         a numerics failure rather than as luck.
  * `test_region_fusion_is_deterministic` — the same dispatch three times, bit-identical. This op
    has a documented history of SILENT ordering races (see test_moe_fused_swiglu_m_tiles.py), and a
    fusion that moves DRAM addresses is exactly the kind of change that would wake one up. A
    single-shot PCC test would not see it.
  * `test_offset_zero_still_rebases` — expert 0 sits at offset 0, where a broken rebase is
    invisible; it is therefore NOT the only expert any test exercises.
  * `test_validation` — the host-side gates, including the in-place refusal.

`insert`-only fusion (a shared output, a per-expert x) is covered by giving expert 0 the whole x
buffer with `read_x_at_offset=False` in `test_insert_only_fusion`.
"""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.operations.moe_fused_swiglu import moe_fused_swiglu

TILE = 32
HIDDEN = 2048
EMB = 7168

NUM_GLOBAL_EXPERTS = 256
NUM_LOCAL_EXPERTS = 4
# Deliberately NOT the identity: an op that indexes counts/start by local id instead of by
# idx[local_id] gets a zero count and an offset belonging to somebody else.
GLOBAL_IDS = [137, 9, 200, 61]

# One region per expert. 1024 rows = 32 tile-rows, the smallest supported capacity.
REGION_ROWS = 1024
REGION_TILES = REGION_ROWS // TILE
# Per-expert token counts: a tile-aligned one, a NON-tile-aligned one (the phantom-row seam), a
# zero (must not hang, must write nothing), and a full region.
COUNTS = [32, 255, 0, REGION_ROWS]

X_PAD_SENTINEL = 100.0  # x rows past a region's count, and every row of a neighbour region
OUT_SENTINEL = -7.5  # pre-filled into the shared output; must survive everywhere unwritten

try:
    from eval.golden_tests.moe_fused_swiglu.feature_spec import _PCC_GATE as PCC_GATE
except ImportError:  # a checkout without the eval harness
    PCC_GATE = 0.975

_FORMATS = {"bf16_rm": (ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT), "bfp8_tile": (ttnn.bfloat8_b, ttnn.TILE_LAYOUT)}


def _reference(x_rows, w_gate, w_up, w_down):
    xf = x_rows.to(torch.float32)
    h = torch.nn.functional.silu(torch.matmul(xf, w_gate.to(torch.float32)))
    h = h * torch.matmul(xf, w_up.to(torch.float32))
    return torch.matmul(h, w_down.to(torch.float32))


def _u32(t, device):
    return ttnn.from_torch(
        t, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def _aux_tensors(device, counts_per_expert):
    """`counts`, `idx` and `start` — all three device-resident, all indexed by GLOBAL expert id."""
    counts = torch.zeros(NUM_GLOBAL_EXPERTS, dtype=torch.int32)
    start = torch.zeros(NUM_GLOBAL_EXPERTS, dtype=torch.int32)
    idx = torch.tensor([(11 + 37 * i) % NUM_GLOBAL_EXPERTS for i in range(NUM_LOCAL_EXPERTS)], dtype=torch.int32)
    for local, g in enumerate(GLOBAL_IDS):
        idx[local] = g
        counts[g] = counts_per_expert[local]
        start[g] = local * REGION_ROWS  # region base in TOKEN rows, tile-aligned
    return _u32(counts, device), _u32(idx, device), _u32(start, device)


def _weights(device, seed):
    torch.manual_seed(seed)
    w = [torch.randn(s, dtype=torch.float32) for s in ((EMB, HIDDEN), (EMB, HIDDEN), (HIDDEN, EMB))]
    tt = [
        ttnn.from_torch(
            t.to(torch.bfloat16),
            dtype=ttnn.bfloat4_b,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        for t in w
    ]
    return w, tt


def _shared_x(device, counts_per_expert, input_format):
    """One buffer holding NUM_LOCAL_EXPERTS regions back to back, as the dispatch produces it."""
    torch.manual_seed(7)
    rows = NUM_LOCAL_EXPERTS * REGION_ROWS
    x = torch.randn((1, 1, rows, EMB), dtype=torch.float32)
    for local, count in enumerate(counts_per_expert):
        base = local * REGION_ROWS
        # Hostile padding INSIDE the region: a leak into a real row is visible in PCC.
        if count < REGION_ROWS:
            x[:, :, base + count : base + REGION_ROWS, :] = X_PAD_SENTINEL
    dtype, layout = _FORMATS[input_format]
    tt_x = ttnn.from_torch(
        x.to(torch.bfloat16), dtype=dtype, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    return x, tt_x


def _filled_output(device, rows, dtype=ttnn.bfloat8_b):
    """A shared output pre-filled with a sentinel, so "never written" is a CHECKABLE property."""
    t = torch.full((1, 1, rows, EMB), OUT_SENTINEL, dtype=torch.float32)
    return ttnn.from_torch(
        t.to(torch.bfloat16), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


# ---------------------------------------------------------------------------------------------
# 1. `output=` on its own
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("out_dtype", [ttnn.bfloat8_b, ttnn.bfloat16])
def test_preallocated_output(device, out_dtype):
    count = 255
    w, tt_w = _weights(device, seed=42)
    torch.manual_seed(11)
    x = torch.randn((1, 1, REGION_ROWS, EMB), dtype=torch.float32)
    x[:, :, count:, :] = X_PAD_SENTINEL
    tt_x = ttnn.from_torch(
        x.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_counts, tt_idx, _ = _aux_tensors(device, [count] + [0] * (NUM_LOCAL_EXPERTS - 1))

    out = _filled_output(device, REGION_ROWS, out_dtype)
    addr = out.buffer_address()
    returned = moe_fused_swiglu(tt_x, tt_w[0], tt_w[1], tt_w[2], tt_counts, tt_idx, 0, output=out)

    # The op writes into the buffer it was handed and hands the same one back — no allocation.
    assert returned.buffer_address() == addr
    assert returned.dtype == out_dtype and returned.layout == ttnn.TILE_LAYOUT
    assert list(returned.shape) == [1, 1, REGION_ROWS, EMB]

    expected = _reference(x[0, 0, :count, :].to(torch.bfloat16), *w)
    actual = ttnn.to_torch(returned)[0, 0, :count, :].to(torch.float32)
    assert_with_pcc(expected, actual, PCC_GATE)

    # Rows past ceil_tile(count) are never written, so the sentinel must still be there. This is
    # the documented contract for a pre-allocated output: the op does not zero the padding.
    tail = ttnn.to_torch(returned)[0, 0, ((count + TILE - 1) // TILE) * TILE :, :].to(torch.float32)
    assert torch.allclose(tail, torch.full_like(tail, OUT_SENTINEL), atol=0.2), "the op wrote past its rows"


def test_preallocated_output_matches_allocating_path(device):
    """The two paths must produce the same bytes — `output=` is placement, not numerics."""
    count = 128
    w, tt_w = _weights(device, seed=42)
    torch.manual_seed(11)
    x = torch.randn((1, 1, REGION_ROWS, EMB), dtype=torch.float32)
    x[:, :, count:, :] = X_PAD_SENTINEL
    tt_x = ttnn.from_torch(
        x.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_counts, tt_idx, _ = _aux_tensors(device, [count] + [0] * (NUM_LOCAL_EXPERTS - 1))

    allocated = moe_fused_swiglu(tt_x, tt_w[0], tt_w[1], tt_w[2], tt_counts, tt_idx, 0)
    provided = moe_fused_swiglu(
        tt_x, tt_w[0], tt_w[1], tt_w[2], tt_counts, tt_idx, 0, output=_filled_output(device, REGION_ROWS)
    )
    a = ttnn.to_torch(allocated)[0, 0, :count, :]
    b = ttnn.to_torch(provided)[0, 0, :count, :]
    assert torch.equal(a, b), "pre-allocated output changed the numerics"


# ---------------------------------------------------------------------------------------------
# 2. extract + insert fusion over a shared buffer
# ---------------------------------------------------------------------------------------------
def _run_all_experts(device, tt_x, tt_w, tt_counts, tt_idx, tt_start, out):
    """The loop a MoE block runs: every expert reads and writes its own region of one buffer."""
    for local in range(NUM_LOCAL_EXPERTS):
        moe_fused_swiglu(
            tt_x,
            tt_w[0],
            tt_w[1],
            tt_w[2],
            tt_counts,
            tt_idx,
            local,
            input_m_tiles=REGION_TILES,
            output=out,
            expert_region_offsets=tt_start,
            read_x_at_offset=True,
        )
    return out


@pytest.mark.parametrize("input_format", ["bf16_rm", "bfp8_tile"])
def test_region_fusion(device, input_format):
    rows = NUM_LOCAL_EXPERTS * REGION_ROWS
    w, tt_w = _weights(device, seed=42)
    x, tt_x = _shared_x(device, COUNTS, input_format)
    tt_counts, tt_idx, tt_start = _aux_tensors(device, COUNTS)

    out = _run_all_experts(device, tt_x, tt_w, tt_counts, tt_idx, tt_start, _filled_output(device, rows))
    got = ttnn.to_torch(out)[0, 0].to(torch.float32)

    written = torch.zeros(rows, dtype=torch.bool)
    for local, count in enumerate(COUNTS):
        base = local * REGION_ROWS
        if count == 0:
            continue  # a zero-count expert writes nothing at all
        expected = _reference(x[0, 0, base : base + count, :].to(torch.bfloat16), *w)
        assert_with_pcc(expected, got[base : base + count, :], PCC_GATE)
        assert torch.isfinite(got[base : base + count, :]).all()
        written[base : base + ((count + TILE - 1) // TILE) * TILE] = True

    # THE OFFSET ASSERTION: everything the op should not have touched still holds the sentinel. A
    # wrong region base writes SOMEWHERE, and that somewhere is inside this mask.
    untouched = got[~written]
    assert torch.allclose(
        untouched, torch.full_like(untouched, OUT_SENTINEL), atol=0.2
    ), "an expert wrote outside its own region"


def test_offset_zero_still_rebases(device):
    """Expert 0 is at offset 0. Run expert 2 ALONE, so the only correct destination is 2048.."""
    rows = NUM_LOCAL_EXPERTS * REGION_ROWS
    counts = [0, 0, 512, 0]
    w, tt_w = _weights(device, seed=42)
    x, tt_x = _shared_x(device, counts, "bf16_rm")
    tt_counts, tt_idx, tt_start = _aux_tensors(device, counts)

    out = _filled_output(device, rows)
    moe_fused_swiglu(
        tt_x,
        tt_w[0],
        tt_w[1],
        tt_w[2],
        tt_counts,
        tt_idx,
        2,
        input_m_tiles=REGION_TILES,
        output=out,
        expert_region_offsets=tt_start,
        read_x_at_offset=True,
    )
    got = ttnn.to_torch(out)[0, 0].to(torch.float32)
    base = 2 * REGION_ROWS
    expected = _reference(x[0, 0, base : base + 512, :].to(torch.bfloat16), *w)
    assert_with_pcc(expected, got[base : base + 512, :], PCC_GATE)
    # Region 0 must be untouched — that is what distinguishes a real rebase from a no-op.
    head = got[:REGION_ROWS]
    assert torch.allclose(head, torch.full_like(head, OUT_SENTINEL), atol=0.2), "wrote at row 0, not at its region"


def test_insert_only_fusion(device):
    """Offsets WITHOUT read_x_at_offset: x is this expert's own tensor, the output is shared."""
    count = 255
    rows = NUM_LOCAL_EXPERTS * REGION_ROWS
    w, tt_w = _weights(device, seed=42)
    torch.manual_seed(11)
    x = torch.randn((1, 1, REGION_ROWS, EMB), dtype=torch.float32)
    x[:, :, count:, :] = X_PAD_SENTINEL
    tt_x = ttnn.from_torch(
        x.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    counts = [0, count, 0, 0]
    tt_counts, tt_idx, tt_start = _aux_tensors(device, counts)

    out = _filled_output(device, rows)
    moe_fused_swiglu(tt_x, tt_w[0], tt_w[1], tt_w[2], tt_counts, tt_idx, 1, output=out, expert_region_offsets=tt_start)
    got = ttnn.to_torch(out)[0, 0].to(torch.float32)
    base = REGION_ROWS  # expert 1's region, though x was read from row 0
    expected = _reference(x[0, 0, :count, :].to(torch.bfloat16), *w)
    assert_with_pcc(expected, got[base : base + count, :], PCC_GATE)
    head = got[:REGION_ROWS]
    assert torch.allclose(head, torch.full_like(head, OUT_SENTINEL), atol=0.2), "insert offset not applied"


def test_region_fusion_is_deterministic(device):
    """Bit-identical across repeats. The guard against a SILENT ordering race, not accuracy."""
    rows = NUM_LOCAL_EXPERTS * REGION_ROWS
    _, tt_w = _weights(device, seed=42)
    _, tt_x = _shared_x(device, COUNTS, "bf16_rm")
    tt_counts, tt_idx, tt_start = _aux_tensors(device, COUNTS)

    runs = []
    for _ in range(3):
        out = _run_all_experts(device, tt_x, tt_w, tt_counts, tt_idx, tt_start, _filled_output(device, rows))
        runs.append(ttnn.to_torch(out)[0, 0].clone())
    for i, r in enumerate(runs[1:], start=1):
        assert torch.equal(runs[0], r), f"run {i} differs from run 0 — a race, not a numerics issue"


# ---------------------------------------------------------------------------------------------
# 3. host-side gates
# ---------------------------------------------------------------------------------------------
def test_validation(device, expect_error):
    count = 32
    w, tt_w = _weights(device, seed=42)
    torch.manual_seed(11)
    x = torch.randn((1, 1, REGION_ROWS, EMB), dtype=torch.float32)
    tt_x = ttnn.from_torch(
        x.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_counts, tt_idx, tt_start = _aux_tensors(device, [count] * NUM_LOCAL_EXPERTS)
    call = lambda **kw: moe_fused_swiglu(tt_x, tt_w[0], tt_w[1], tt_w[2], tt_counts, tt_idx, 0, **kw)  # noqa: E731

    with expect_error(ValueError, "read_x_at_offset requires expert_region_offsets"):
        call(read_x_at_offset=True)

    with expect_error(ValueError, "requires `output`"):
        call(expert_region_offsets=tt_start)

    # A shared destination smaller than one region.
    with expect_error(ValueError, "must be >= input rows"):
        call(expert_region_offsets=tt_start, output=_filled_output(device, REGION_ROWS - TILE))

    # No offsets => the output is per-expert and its rows must match x's exactly.
    with expect_error(ValueError, "must equal input rows"):
        call(output=_filled_output(device, 2 * REGION_ROWS))

    # start and counts index the same global-expert space, so their lengths must agree.
    short_start = _u32(torch.zeros(NUM_GLOBAL_EXPERTS // 2, dtype=torch.int32), device)
    with expect_error(ValueError, "must equal counts length"):
        call(expert_region_offsets=short_start, output=_filled_output(device, REGION_ROWS))

    with expect_error(ValueError, "must be uint32 ROW_MAJOR"):
        call(
            expert_region_offsets=ttnn.from_torch(
                torch.zeros(NUM_GLOBAL_EXPERTS, dtype=torch.int32),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            ),
            output=_filled_output(device, REGION_ROWS),
        )

    with expect_error(ValueError, "contradicts the supplied output"):
        call(output=_filled_output(device, REGION_ROWS), dtype=ttnn.bfloat16)

    # In place is refused: the reader prefetches the next M-block with no ordering against this
    # block's write-back, so an aliased output is a read-after-write race, not a saving.
    tt_x_tiled = ttnn.from_torch(
        x.to(torch.bfloat16),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    with expect_error(ValueError, "must not alias"):
        moe_fused_swiglu(tt_x_tiled, tt_w[0], tt_w[1], tt_w[2], tt_counts, tt_idx, 0, output=tt_x_tiled)
