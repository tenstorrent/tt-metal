# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Refinement 1 regression net: the runtime `m_eff` shrink must be DETERMINISTIC.

The op works `m_eff = m_tiles_eff(M_t, b, M_BLOCK, M_EFF_MIN)` token tile-rows per M-block — a
power of two <= M_BLOCK, derived on device from the runtime token count — instead of a constant
M_BLOCK. Shrinking the block shortens both collectives (fewer x-multicast rounds, smaller h
payload), which REMOVED the matmul latency that had been masking a real ordering bug in
mcast_pipe's rotating-sender Flag protocol:

    send() resets its own data-ready cell to INVALID behind a `fence_()` that is
    `async_writes_flushed()` — SENT, not LANDED. On a LOOPBACK (`src != dst`) multicast the
    sender's own cell is one of the destinations, so the in-flight VALID can land AFTER the
    reset. The sender's next receive() then returns on a stale flag, every later round shifts one
    early, and the block's last round is consumed before its data arrives.

Both sends therefore land their own copy locally first and multicast IN PLACE (`src == dst`,
EXCLUDE-source). The bug was SILENT — correct output most runs, garbage some runs — so the guard
here is REPEAT-DETERMINISM, not just accuracy: run the same input several times and require
bit-identical output. A single-shot accuracy test would have passed throughout.

Every count below pins a distinct m_eff regime; keep them all if this file is ever edited.
"""

import pytest
import torch

import ttnn

from ttnn.operations.moe_fused_swiglu import moe_fused_swiglu

TILE = 32
HIDDEN = 2048
NUM_GLOBAL_EXPERTS, NUM_LOCAL_EXPERTS, LOCAL_EXPERT_ID, GLOBAL_EXPERT_ID = 256, 8, 3, 137
PADDING_SENTINEL = 100.0

# The measured bfloat4_b format floor is ~0.9797 (changelog §3), so this only asserts that the
# op is still in its normal precision regime — the sharp assertion in this file is determinism.
PCC_FLOOR = 0.975

_FORMATS = {"bf16_rm": (ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT), "bfp8_tile": (ttnn.bfloat8_b, ttnn.TILE_LAYOUT)}

# (capacity, count, m_eff, why) — M_t = ceil(count/32), m_eff = pow2_ceil(tail M_t) capped at M_BLOCK=8
M_TILES_CASES = [
    (1024, 32, 1, "M_t=1 -> m_eff=1: the smallest block, 1 x-round"),
    (1024, 64, 2, "M_t=2 -> m_eff=2"),
    (1024, 96, 4, "M_t=3 -> m_eff=4: m_eff EXCEEDS M_t, so the tail tile-row is padding"),
    (2048, 128, 4, "M_t=4 -> m_eff=4: the graded balanced count, exact fit"),
    (1024, 255, 8, "M_t=8 -> m_eff=8: full block, byte-identical to pre-refinement behaviour"),
    (2048, 512, 8, "M_t=16 -> 2 full blocks: m_eff must NOT shrink a non-final block"),
    (2048, 288, 8, "M_t=9 -> block 0 m_eff=8, block 1 m_eff=1: a SHRUNK TAIL after a full block"),
]


def _count_tensors(count, device):
    counts = torch.zeros(NUM_GLOBAL_EXPERTS, dtype=torch.int32)
    counts[GLOBAL_EXPERT_ID] = count
    idx = torch.tensor([(11 + 37 * i) % NUM_GLOBAL_EXPERTS for i in range(NUM_LOCAL_EXPERTS)], dtype=torch.int32)
    idx[LOCAL_EXPERT_ID] = GLOBAL_EXPERT_ID
    to_dev = lambda t: ttnn.from_torch(  # noqa: E731
        t, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    return to_dev(counts), to_dev(idx)


def _build(emb, capacity, count, input_format, device):
    torch.manual_seed(42)
    x = torch.randn((1, 1, capacity, emb), dtype=torch.float32)
    if count < capacity:
        x[:, :, count:, :] = PADDING_SENTINEL  # hostile padding: a leak into a real row is visible
    w = [torch.randn(s, dtype=torch.float32) for s in ((emb, HIDDEN), (emb, HIDDEN), (HIDDEN, emb))]

    dt, lay = _FORMATS[input_format]
    tt_x = ttnn.from_torch(
        x.to(torch.bfloat16), dtype=dt, layout=lay, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    tt_w = [
        ttnn.from_torch(
            t.to(torch.bfloat16),
            dtype=ttnn.bfloat4_b,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        for t in w
    ]
    tt_counts, tt_idx = _count_tensors(count, device)

    xr = x[0, 0, :count, :].to(torch.bfloat16).to(torch.float32)
    h = torch.nn.functional.silu(torch.matmul(xr, w[0])) * torch.matmul(xr, w[1])
    return tt_x, tt_w, tt_counts, tt_idx, torch.matmul(h, w[2])


def _pcc(a, b):
    a = a.flatten().to(torch.float64) - a.mean().to(torch.float64)
    b = b.flatten().to(torch.float64) - b.mean().to(torch.float64)
    return (a @ b / (a.norm() * b.norm())).item()


@pytest.mark.parametrize("capacity, count, m_eff, why", M_TILES_CASES)
@pytest.mark.parametrize("input_format", ["bf16_rm", "bfp8_tile"])
def test_m_tiles_regime_is_deterministic(device, capacity, count, m_eff, why, input_format):
    """Same input, 3 dispatches: the defined rows must be BIT-IDENTICAL every time.

    This is the assertion that catches a racing collective. Accuracy is only sanity-checked —
    the m_eff shrink must not change the numerics at all, and a race shows up here as a spread.
    """
    emb = 7168
    tt_x, tt_w, tt_counts, tt_idx, expected = _build(emb, capacity, count, input_format, device)

    runs = []
    for _ in range(3):
        out = moe_fused_swiglu(tt_x, tt_w[0], tt_w[1], tt_w[2], tt_counts, tt_idx, LOCAL_EXPERT_ID)
        got = ttnn.to_torch(out)[0, 0, :count, :].to(torch.float32)
        assert torch.isfinite(got).all(), f"non-finite value in a defined row ({why})"
        runs.append(got)

    for i, got in enumerate(runs[1:], start=1):
        n_diff = int((got != runs[0]).sum())
        assert n_diff == 0, (
            f"run {i} differs from run 0 in {n_diff} elements — a collective is RACING at "
            f"m_eff={m_eff} ({why}). Check that BOTH mcast_pipe sends are src==dst "
            f"(EXCLUDE-source); a src!=dst loopback send lets the rotating-sender flag reset race "
            f"the sender's own in-flight VALID."
        )

    pcc = _pcc(expected, runs[0])
    assert pcc > PCC_FLOOR, f"pcc {pcc} below the format floor at m_eff={m_eff} ({why})"


def test_m_tiles_shrink_does_not_change_numerics(device):
    """A count whose M_t is 4 must give the SAME defined rows as sizing the op to 8 tile-rows.

    `input_m_tiles` caps M_t, so it is the one host-visible handle on the block size: with
    input_m_tiles >= M_t the runtime m_eff is identical either way, and the output must be too.
    That pins "m_eff only removes work on UNDEFINED rows" — the whole correctness premise of the
    refinement.
    """
    emb, capacity, count = 7168, 1024, 128
    tt_x, tt_w, tt_counts, tt_idx, _ = _build(emb, capacity, count, "bf16_rm", device)

    ref = None
    for m_tiles in (capacity // TILE, 8, 4):  # all >= M_t = 4
        out = moe_fused_swiglu(
            tt_x, tt_w[0], tt_w[1], tt_w[2], tt_counts, tt_idx, LOCAL_EXPERT_ID, input_m_tiles=m_tiles
        )
        got = ttnn.to_torch(out)[0, 0, :count, :].to(torch.float32)
        if ref is None:
            ref = got
        else:
            assert torch.equal(ref, got), f"input_m_tiles={m_tiles} changed the defined rows"


def test_zero_count_still_skips_every_collective(device):
    """count == 0 -> m_blocks == 0 on all 110 cores. The m_eff derivation must not make this
    core-dependent (that would enter a collective on some cores only, i.e. hang)."""
    emb, capacity = 7168, 1024
    tt_x, tt_w, tt_counts, tt_idx, _ = _build(emb, capacity, 0, "bf16_rm", device)
    for _ in range(2):
        out = moe_fused_swiglu(tt_x, tt_w[0], tt_w[1], tt_w[2], tt_counts, tt_idx, LOCAL_EXPERT_ID)
        assert list(out.shape) == [1, 1, capacity, emb]
