# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""onorm cross-core re-tile (Refinement 2) — correctness + measurement harness.

DO NOT DELETE.

`RETILE_GROUP_CORES` splits ONE token-block across `group_cores` cores on two
axes at once (tokens for the normalize half, flat output columns for the gate
half), joined by a row-major all-to-all exchange through `cb_rm_flat_rows`.  This
file is what proves the knob is live and, crucially, that the exchange is
**numerically inert**: the same tokens are normalized by the same arithmetic on
whichever core owns them, so every group size must return the SAME bytes as the
single-core-per-block path.

Three jobs:

1. **Correctness across the whole legal group-size range** (1..32) on the shapes
   that motivated the refinement, plus a core-saturated shape.
2. **Bit-equality against `group_cores = 1`** — the exchange moves bytes, it does
   not compute, so any difference at all is a bug (not a tolerance question).
3. **Measurement vehicle**: `test_group_trial` is a trial-major interleaved
   device-ns sweep over group sizes, the only reproducible timing shape for this
   op (see op_requirements.md "Measurement discipline").
"""

import pytest
import torch

import ttnn
import ttnn.operations.onorm.onorm_program_descriptor as pd
from ttnn.operations.onorm import default_compute_kernel_config, onorm

from tests.ttnn.utils_for_testing import assert_with_pcc

HV = 32
V = 128
FLAT = HV * V
PCC = 0.995

# The legal group sizes at the default TOKENS_PER_BLOCK=32 / flat_tiles=128: a
# group size must divide BOTH (it is the token slice AND the column slice).
GROUP_SIZES = [1, 2, 4, 8, 16, 32]

# (batch, tokens) -> token-blocks at TOKENS_PER_BLOCK=32.
#   (1, 32)  -> 1 block   : the refinement's headline case (1 of 110 cores busy)
#   (1, 128) -> 4 blocks  : the short-prefill case
#   (1, 640) -> 20 blocks : partially filled
#   (8, 640) -> 160 blocks: core-saturated; "auto" must leave this at group 1
SHAPES = [(1, 32), (1, 128), (1, 640)]


@pytest.fixture
def restore_knobs():
    saved = {k: getattr(pd, k) for k in ("RETILE_GROUP_CORES", "MAX_RETILE_GROUP_CORES", "RM_LOCAL_DEPTH")}
    yield
    for k, v in saved.items():
        setattr(pd, k, v)


def _inputs(batch, tokens, seed=42):
    torch.manual_seed(seed)
    return (
        torch.randn(batch, tokens, HV, V, dtype=torch.bfloat16),
        torch.randn(batch, tokens, FLAT, dtype=torch.bfloat16),
        torch.randn(1, 1, 1, V, dtype=torch.bfloat16),
    )


def _reference(t_o, t_gate, t_w, eps=1e-5):
    batch, tokens = t_o.shape[0], t_o.shape[1]
    f = t_o.to(torch.float32)
    ref = f * torch.rsqrt(f.pow(2).mean(dim=-1, keepdim=True) + eps)
    ref = ref * t_w.to(torch.float32).reshape(1, 1, 1, V)
    return ref.reshape(batch, tokens, FLAT) * torch.sigmoid(t_gate.to(torch.float32))


def _run(device, batch, tokens, seed=42):
    t_o, t_gate, t_w = _inputs(batch, tokens, seed)
    o = ttnn.from_torch(t_o, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    gate = ttnn.from_torch(t_gate, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    w = ttnn.from_torch(t_w, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = onorm(o, gate, w, compute_kernel_config=default_compute_kernel_config())
    assert list(out.shape) == [batch, tokens, FLAT]
    return ttnn.to_torch(out).to(torch.float32), _reference(t_o, t_gate, t_w)


# ---------------------------------------------------------------------------
# 1. correctness at every legal group size
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("group_cores", GROUP_SIZES, ids=lambda g: f"g{g}")
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: f"b{s[0]}t{s[1]}")
def test_group_size_correct(device, restore_knobs, group_cores, shape):
    """Every legal cross-core re-tile group size must be numerically correct."""
    pd.RETILE_GROUP_CORES = group_cores
    got, ref = _run(device, *shape)
    assert_with_pcc(ref, got, PCC)


def test_group_size_correct_core_saturated(device, restore_knobs):
    """A shape with more token-blocks than cores, forced into a split anyway."""
    pd.RETILE_GROUP_CORES = 4
    got, ref = _run(device, 8, 640)
    assert_with_pcc(ref, got, PCC)


# ---------------------------------------------------------------------------
# 2. the exchange must be numerically INERT
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("group_cores", [g for g in GROUP_SIZES if g > 1], ids=lambda g: f"g{g}")
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: f"b{s[0]}t{s[1]}")
def test_group_size_is_bit_identical_to_single_core(device, restore_knobs, group_cores, shape):
    """A cross-core exchange moves bytes; it must not perturb a single bit.

    Which core normalizes a token, and which core gates an output column, changes
    — the arithmetic applied to either does not.  So this is an exact-equality
    test, not a tolerance test: any difference is a layout/indexing bug.
    """
    pd.RETILE_GROUP_CORES = 1
    base, _ = _run(device, *shape)
    pd.RETILE_GROUP_CORES = group_cores
    split, _ = _run(device, *shape)
    assert torch.equal(base, split), (
        f"group_cores={group_cores} changed the output on shape {shape}: "
        f"max |diff| = {(base - split).abs().max().item()}, "
        f"{(base != split).sum().item()} of {base.numel()} elements differ"
    )


# ---------------------------------------------------------------------------
# 3. the "auto" dispatch policy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape, num_token_blocks",
    [((1, 32), 1), ((1, 128), 4), ((1, 640), 20), ((8, 640), 160)],
    ids=lambda v: str(v),
)
def test_auto_policy_matches_spare_capacity(device, restore_knobs, shape, num_token_blocks):
    """`auto` spends only cores that a per-block split would leave idle.

    The policy is: largest legal power of two <= (total_cores // token_blocks),
    capped by MAX_RETILE_GROUP_CORES.  A core-saturated shape must therefore land
    on group 1 — the byte-identical, exchange-free path.
    """
    grid = device.compute_with_storage_grid_size()
    total = grid.x * grid.y
    pd.RETILE_GROUP_CORES = "auto"
    got = pd._retile_group_cores(device, num_token_blocks, pd.TOKENS_PER_BLOCK, FLAT // 32)

    expected = 1
    ceiling = min(total // num_token_blocks, pd.MAX_RETILE_GROUP_CORES)
    while expected * 2 <= ceiling and pd.TOKENS_PER_BLOCK % (expected * 2) == 0:
        expected *= 2
    assert got == expected
    if num_token_blocks >= total:
        assert got == 1, "a core-saturated shape must not pay for an exchange"

    # ...and it must still be correct at whatever the policy picked.
    out, ref = _run(device, *shape)
    assert_with_pcc(ref, out, PCC)


def test_illegal_group_size_is_rejected_with_guidance(device, restore_knobs):
    """A group size that does not divide both split axes fails the host guard."""
    pd.RETILE_GROUP_CORES = 3  # divides neither TOKENS_PER_BLOCK=32 nor flat_tiles=128
    try:
        _run(device, 1, 128)
    except AssertionError as exc:
        msg = str(exc)
        assert "RETILE_GROUP_CORES" in msg and "TOKENS_PER_BLOCK" in msg and "flat_tiles" in msg, msg
        return
    pytest.fail("expected the host assert to reject an indivisible RETILE_GROUP_CORES")


def test_repeated_dispatch_is_stable(device, restore_knobs):
    """The exchange semaphores are monotone counters, never reset by a kernel.

    That only works if the HOST re-initialises them to 0 on every program launch.
    If it did not, the second dispatch's `wait_min` targets would already be
    satisfied by the first dispatch's leftovers and the flow control would
    silently vanish (a race, not an error).  So: run the same op several times in
    one process and demand an identical answer every time.
    """
    pd.RETILE_GROUP_CORES = 8
    first, ref = _run(device, 1, 128)
    assert_with_pcc(ref, first, PCC)
    for _ in range(4):
        again, _ = _run(device, 1, 128)
        assert torch.equal(first, again), "repeated dispatch diverged — semaphore state leaked across launches"


# ---------------------------------------------------------------------------
# 4. measurement (trial-major interleaved; run under --profile)
# ---------------------------------------------------------------------------

TRIAL_SHAPES = [(1, 32), (1, 128), (1, 640), (8, 640)]
TRIAL_GROUPS = [1, 2, 4, 8, 16, 32]


@pytest.mark.parametrize("trials", [5])
def test_group_trial(device, restore_knobs, trials):
    """Trial-major interleaved sweep: group size x shape, one process.

    Interleaving by TRIAL (not by candidate) is what makes onorm's numbers
    comparable — a 248 vs 102 us swing on identical config is on record for
    candidate-major runs.  Read `DEVICE KERNEL DURATION [ns]` per row from the
    profiler CSV; the row order is (trial, shape, group).
    """
    for _ in range(trials):
        for shape in TRIAL_SHAPES:
            for group_cores in TRIAL_GROUPS:
                pd.RETILE_GROUP_CORES = group_cores
                got, ref = _run(device, *shape)
                assert_with_pcc(ref, got, PCC)
