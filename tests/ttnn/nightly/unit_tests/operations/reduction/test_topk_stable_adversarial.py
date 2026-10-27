# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Adversarial input classes (A1-A7) for ttnn.topk(stable=True).

Engine routing -- stable=True routes by shape:
* W=64 (a single tile row) runs the single-core COMPARATOR engine
  (16-bit DEST).
* W=8192 with bf16 values + uint16 indices runs the multicore FUSED-KEY
  engine (32-bit DEST; each element sorts as one fused {value, index} word).

SILICON-CHARACTERIZED semantics (Blackhole, 2026-08-18): both engines behave
BYTE-IDENTICALLY on special values, in both largest modes.  The device
computes

    device_topk_stable(x) == stable-argsort prefix over canon(x)

where canon(v) is applied INSIDE the compute (FPU widening) -- the
input/output I/O path itself is bit-exact:

  * +-0.0 and ALL bf16 denormals (exponent field 0, either sign, e.g.
    0x0001 / 0x007F / 0x8001) -> +0.0 (bits 0x0000);
  * NaN with sign bit 0 (0x7FC0, 0x7F81, ...) -> +inf (bits 0x7F80) and
    NaN with sign bit 1 (0xFFC0, ...) -> -inf (bits 0xFF80): a NaN genuinely
    TIES with a same-sign real infinity -- there is no strict order between
    them;
  * everything else (normal values, +-inf) passes through bit-exact.

Output INDICES are exactly torch.argsort(canon(x), dim=-1, stable=True,
descending=largest)[..., :k] -- every tie, including the whole +-0/denormal
flush group and NaN-vs-same-sign-inf, breaks index-ascending.  Output VALUES
are bit-exact equal to canon(x) gathered at those indices.  Runs are
deterministic.

Consequences the classes below pin down:
  * a -0.0 or denormal input yields output value bits 0x0000;
  * a NaN input never surfaces its payload -- it lands as +-inf bits;
  * torch's NaN-is-greatest placement does NOT hold on device: a -NaN sorts
    to the BOTTOM (as -inf), so it appears in bottom-k, never in top-k.

canon is the identity on normal values and +-inf, so the classes without
zeros/denormals/NaNs (A1, A2, A4, A6 and the normal A7 variant) reduce to
plain torch-stable parity.  torch.topk itself is NOT stable, so all goldens
come from a stable argsort.

All tests use shape [1, 1, 32, W] with k=32 (one output tile).
"""

import pytest

pytestmark = pytest.mark.use_module_device

import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_equal

K = 32  # top-k size for every class: exactly one output tile column
H = 32  # independent adversarial rows per tensor: one tile row
# Routing note: the multicore fused-key path requires W to be a power of two
# and >= 8192 (bf16 values + uint16 indices); smaller widths stay on the
# single-core comparator path.  W=64 is a single tile row.
W_COMPARATOR = 64
W_FUSED = 8192
SEED = 2026

both_engines = pytest.mark.parametrize("W", [W_COMPARATOR, W_FUSED], ids=["W64_comparator", "W8192_fused"])
both_directions = pytest.mark.parametrize("largest", [True, False], ids=["largest", "smallest"])


def _seed(cls, W, largest):
    """Deterministic per-testcase seed so every (class, W, largest) combination
    gets distinct -- but reproducible -- shuffles and fillers."""
    return SEED + 1000 * cls + W + int(largest)


def bits(u16):
    """Raw uint16 bit patterns -> bfloat16 tensor, bit-exact.

    Accepts a python list of ints or an integer tensor (any shape).  This is
    the only sanctioned way to build the special values (-0.0, denormals,
    NaN payloads, inf) -- never build them through float arithmetic, which
    may canonicalize them on the host before they ever reach the device.
    """
    if isinstance(u16, torch.Tensor):
        u16 = u16.to(torch.uint16)
    else:
        u16 = torch.tensor(u16, dtype=torch.uint16)
    return u16.view(torch.bfloat16)


def bf16_bits(x):
    """bfloat16 tensor -> its raw bit patterns as int64 in [0, 0xFFFF].

    Used for bit-exact comparisons where float semantics would lie:
    NaN != NaN would hide payload changes and -0.0 == +0.0 would hide sign
    canonicalization.
    """
    return x.contiguous().view(torch.int16).to(torch.int64) & 0xFFFF


def canon(x):
    """Host-side model of the device's compute canonicalization (silicon-
    characterized on both engines -- see module docstring), applied on the
    raw uint16 view of a bf16 tensor:

        exponent field == 0                    -> 0x0000  (+-0 and all
                                                           denormals flush
                                                           to +0.0)
        exponent field == 0xFF, mantissa != 0  -> sign | 0x7F80  (NaN
                                                           collapses to
                                                           same-sign inf)
        otherwise                              -> unchanged

    The result contains no NaNs, no -0.0 and no denormals, so ordinary IEEE
    float comparison on it is total up to exact-value ties.
    """
    b = bf16_bits(x)
    exp = (b >> 7) & 0xFF
    mant = b & 0x7F
    b = torch.where(exp == 0, torch.zeros_like(b), b)
    b = torch.where((exp == 0xFF) & (mant != 0), (b & 0x8000) | 0x7F80, b)
    return bits(b)


def torch_stable_golden(input, k, largest):
    """Golden for the normals-only classes: (values, indices) of a STABLE
    top-k.  torch.topk is not stable, so the golden is a stable-argsort
    prefix: exact-value ties resolve to the lowest original index first.
    On inputs without zeros/denormals/NaNs canon is the identity, so this
    equals canon_stable_golden there.
    """
    order = torch.argsort(input, dim=-1, descending=largest, stable=True)[..., :k]
    values = torch.gather(input, -1, order)
    return values, order


def canon_stable_golden(input, k, largest):
    """THE golden for classes containing specials, valid for BOTH engines:
    a stable-argsort prefix over canon(input).

    canon output has no NaNs (they became same-sign infs) and no
    -0/denormals (they became +0.0), so the float argsort order is exactly
    the device's value order; stable=True breaks every tie -- the whole
    +-0/denormal flush group and NaN-vs-same-sign-inf alike -- by lowest
    original index.  Golden values are canon(input) gathered at the golden
    indices, gathered on the bit view for bit-exactness.
    """
    c = canon(input)
    order = torch.argsort(c, dim=-1, descending=largest, stable=True)[..., :k]
    values = torch.gather(c.contiguous().view(torch.int16), -1, order).view(torch.bfloat16)
    return values, order


def unique_normal_fillers(n, negative=False, start=0x0100):
    """n distinct NORMAL bf16 values with magnitudes in (2^-125, 0.5).

    Consecutive bf16 bit patterns are distinct values of strictly increasing
    magnitude, so this is guaranteed tie-free filler.  Starting at 0x0100
    (exponent field 2) stays clear of bf16 denormals (exponent field 0),
    which the device flushes to +0.0 and which would therefore create
    unintended ties; capping below 0x3F00 keeps every filler under 0.5 so
    the deliberately-tied levels of each class always outrank the filler.
    """
    assert start >= 0x0100 and start + n <= 0x3F00, "fillers must stay normal and below 0.5"
    b = start + torch.arange(n)
    if negative:
        b = b + 0x8000
    return bits(b)


def shuffled_rows(pool, num_rows, seed):
    """Tile a 1-D bf16 pool into [1, 1, num_rows, W] with an independent seeded
    permutation per row.

    The shuffle runs on the int16 bit view (gather is a bitwise move there),
    so -0.0 signs, denormal bits and NaN payloads survive bit-exactly; a
    float-path shuffle could not be trusted to preserve them.
    """
    g = torch.Generator().manual_seed(seed)
    w = pool.numel()
    perms = torch.argsort(torch.rand(num_rows, w, generator=g), dim=-1)
    rows = pool.contiguous().view(torch.int16).expand(num_rows, w).gather(-1, perms)
    return rows.view(torch.bfloat16).reshape(1, 1, num_rows, w).contiguous()


def run_stable_topk(input, largest, device):
    """Upload, run ttnn.topk(stable=True, sorted=True), download.

    Shapes here are always tile-aligned ([1, 1, 32, 64] or [1, 1, 32, 8192]),
    so no implicit-padding fill is needed.  Returns (bf16 values, int64
    indices) as torch tensors.
    """
    ttnn_input = ttnn.from_torch(input, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)
    ttnn_values, ttnn_indices = ttnn.topk(ttnn_input, K, dim=-1, largest=largest, sorted=True, stable=True)
    return ttnn.to_torch(ttnn_values), ttnn.to_torch(ttnn_indices).to(torch.int64)


@both_directions
@both_engines
def test_a1_all_negative_ties(W, largest, device):
    """A1: rows made ONLY of six distinct negative levels, so several exact-tie
    groups land inside the top-k and one group straddles the k=32 cut in each
    direction.

    Why this must fail if the fused sign-conditional index complement were
    dropped: the fused-key engine packs {value bits, index} into a single
    32-bit sort word, and for NEGATIVE values the index bits must be stored
    COMPLEMENTED so that the key-descending sort still emits ties of a
    negative value in index-ASCENDING order.  Every top-k element here is a
    tied negative, so dropping the complement would flip every tie group to
    index-descending and the strict index assert would fail on every row.
    The comparator engine is silicon-characterized to produce the identical
    order, so one torch-stable golden serves both engines (canon is the
    identity on these normal values).
    """
    torch.manual_seed(SEED)
    levels = [-8.0, -4.0, -2.0, -1.0, -0.5, -0.25]
    # Bulk out the middle level so both directions see two full tie groups in
    # the top-k and then a cut INSIDE the third group.
    counts = [12, 12, W - 60, 12, 12, 12]
    pool = torch.cat([torch.full((c,), v, dtype=torch.bfloat16) for v, c in zip(levels, counts)])
    input = shuffled_rows(pool, H, _seed(1, W, largest))

    values, indices = run_stable_topk(input, largest, device)
    golden_values, golden_indices = torch_stable_golden(input, K, largest)
    assert_equal(golden_values, values)
    assert_equal(golden_indices, indices)


@both_directions
@both_engines
def test_a2_mixed_sign_ties(W, largest, device):
    """A2: interleaved positive AND negative tie levels in the same row, so the
    fused engine's sign-conditional key construction must keep index-ascending
    tie order on both sides of zero simultaneously (a per-row sign-mix bug
    that A1's all-negative rows cannot see).  On the fused width the k=32 cut
    additionally lands inside the +-0.5 tie group.  Distinct sub-0.5 fillers
    pad the row without ever displacing a tied level from the top/bottom.
    Strict torch-stable parity on values and indices, both engines.
    """
    torch.manual_seed(SEED)
    reps = 8 if W == W_COMPARATOR else 12
    levels = torch.tensor([-2.0, -1.0, -0.5, 0.5, 1.0, 2.0], dtype=torch.bfloat16)
    tied = levels.repeat_interleave(reps)
    fillers = unique_normal_fillers(W - tied.numel())  # unique, in (0, 0.5)
    input = shuffled_rows(torch.cat([tied, fillers]), H, _seed(2, W, largest))

    values, indices = run_stable_topk(input, largest, device)
    golden_values, golden_indices = torch_stable_golden(input, K, largest)
    assert_equal(golden_values, values)
    assert_equal(golden_indices, indices)


@both_directions
@both_engines
def test_a3_signed_zero(W, largest, device):
    """A3: a 32-element FLUSH-TO-ZERO tie block per row -- 8x +0.0, 8x -0.0
    (raw 0x8000) and 16 denormals of both signs and payloads (0x0001, 0x007F,
    0x8001, 0x807F) -- plus 16 unique fillers on the winning side and unique
    opposite-sign fillers everywhere else, so the k=32 cut lands INSIDE the
    flush block (16 winners + 16 flush members kept).

    Silicon-characterized, both engines: the compute canonicalizes the entire
    +-0/denormal set to +0.0 BEFORE the sort, so the whole block forms ONE
    tie that breaks index-ascending.  Indices must match the canon-golden
    stable argsort exactly, and every emitted value at a flush position must
    carry bit pattern 0x0000 (a -0.0 or denormal input comes back as +0.0).
    """
    torch.manual_seed(SEED)
    flush_block = bits([0x8000, 0x0000] * 8 + [0x0001, 0x007F, 0x8001, 0x807F] * 4)
    if largest:
        winners = bits(0x3F81 + torch.arange(16))  # 16 unique in (1, 2)
        losers = unique_normal_fillers(W - 48, negative=True)  # unique in (-0.5, 0)
    else:
        winners = bits(0xBF81 + torch.arange(16))  # 16 unique in (-2, -1)
        losers = unique_normal_fillers(W - 48)  # unique in (0, 0.5)
    input = shuffled_rows(torch.cat([winners, flush_block, losers]), H, _seed(3, W, largest))

    values, indices = run_stable_topk(input, largest, device)
    golden_values, golden_indices = canon_stable_golden(input, K, largest)
    assert_equal(golden_indices, indices)
    assert_equal(bf16_bits(golden_values), bf16_bits(values))
    # Redundant with the bit-exact golden compare, but documents the flush
    # explicitly: every kept flush-block member returns as +0.0 bits.
    zero_mask = golden_values == 0
    assert zero_mask.any(), "test construction error: the cut must land inside the flush block"
    assert (
        bf16_bits(values)[zero_mask] == 0x0000
    ).all(), "every -0.0/denormal output must be canonicalized to +0.0 (bits 0x0000)"


@both_directions
@both_engines
def test_a4_infinities(W, largest, device):
    """A4: 8 copies of +inf (0x7F80) and 8 of -inf (0xFF80) per row (>= 4 each;
    8 keeps the whole winning inf tie group inside k=32) among unique finite
    fillers of both signs.  canon is the identity on +-inf and normals, and
    this class is deliberately NaN-free (NaN-vs-inf ties live in A5), so
    torch orders +-inf as ordinary extremes and stable-ties equal infs by
    index -- one strict torch-stable golden serves both engines.

    The asserts are deliberately strict on raw values AND indices: if the
    fused engine's fp32 widening or fused-key arithmetic (e.g. an
    ELWADD-based index fuse stepping an inf bit pattern into the NaN space)
    mangled infinities, assert_equal on the values surfaces it immediately.
    """
    torch.manual_seed(SEED)
    infs = bits([0x7F80] * 8 + [0xFF80] * 8)
    n = W - 16
    fillers = torch.cat(
        [
            unique_normal_fillers(n - n // 2),  # unique positives in (0, 0.5)
            unique_normal_fillers(n // 2, negative=True),  # unique negatives in (-0.5, 0)
        ]
    )
    input = shuffled_rows(torch.cat([infs, fillers]), H, _seed(4, W, largest))

    values, indices = run_stable_topk(input, largest, device)
    golden_values, golden_indices = torch_stable_golden(input, K, largest)
    assert_equal(golden_values, values)
    assert_equal(golden_indices, indices)


@both_directions
@both_engines
def test_a5_nan_payloads(W, largest, device):
    """A5: NaN canonicalization, NaN-vs-inf ties, determinism, containment.

    Even rows carry two copies each of qNaN 0x7FC0, sNaN-payload 0x7F81,
    -NaN 0xFFC0, real +inf 0x7F80 and real -inf 0xFF80 among unique finite
    fillers; odd rows are NaN-free.

    Silicon-characterized, both engines: a NaN collapses to the SAME-SIGN
    infinity inside the compute (payloads never survive) and genuinely TIES
    with a real infinity of that sign, the tie breaking index-ascending.  So
    for largest=True the top-k head is the 6-way +inf tie (4 NaN-provenance
    + 2 real, all value bits 0x7F80) and for largest=False the bottom-k head
    is the 4-way -inf tie (2x -NaN + 2 real -inf, all value bits 0xFF80).
    Note torch's NaN-is-greatest placement is deliberately NOT asserted: on
    device a -NaN sorts to the BOTTOM, appearing in bottom-k as 0xFF80.

    Asserts, both engines:
      * full-tensor bit-exact parity (raw value bits + indices) against the
        canon golden;
      * two back-to-back runs on the same device tensor agree bit-exactly
        (values compared on the raw bit view, so NaN != NaN could not hide a
        difference) -- determinism;
      * the CLEAN (odd) rows additionally meet plain torch-stable parity:
        specials in neighbouring rows must not leak into NaN-free rows.
    """
    torch.manual_seed(SEED)
    specials = bits([0x7FC0, 0x7FC0, 0x7F81, 0x7F81, 0xFFC0, 0xFFC0, 0x7F80, 0x7F80, 0xFF80, 0xFF80])
    m = W - specials.numel()
    special_pool = torch.cat(
        [
            specials,
            unique_normal_fillers(m - m // 2),
            unique_normal_fillers(m // 2, negative=True),
        ]
    )
    clean_pool = torch.cat(
        [
            unique_normal_fillers(W - W // 2),
            unique_normal_fillers(W // 2, negative=True),
        ]
    )
    input = shuffled_rows(special_pool, H, _seed(5, W, largest))
    clean_rows = shuffled_rows(clean_pool, H, _seed(5, W, largest) + 1)
    input[:, :, 1::2, :] = clean_rows[:, :, 1::2, :]  # odd rows: NaN-free

    ttnn_input = ttnn.from_torch(input, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)
    v1, i1 = ttnn.topk(ttnn_input, K, dim=-1, largest=largest, sorted=True, stable=True)
    v2, i2 = ttnn.topk(ttnn_input, K, dim=-1, largest=largest, sorted=True, stable=True)
    values1, indices1 = ttnn.to_torch(v1), ttnn.to_torch(i1).to(torch.int64)
    values2, indices2 = ttnn.to_torch(v2), ttnn.to_torch(i2).to(torch.int64)

    # Bit-exact determinism between the two runs.
    assert_equal(bf16_bits(values1), bf16_bits(values2))
    assert_equal(indices1, indices2)

    # Exact canon-golden parity on the full tensor, NaN rows included.
    golden_values, golden_indices = canon_stable_golden(input, K, largest)
    assert_equal(golden_indices, indices1)
    assert_equal(bf16_bits(golden_values), bf16_bits(values1))

    # Clean (odd) rows also satisfy plain torch-stable parity (canon is the
    # identity there); this pins containment independently of canon().
    torch_values, torch_indices = torch_stable_golden(input, K, largest)
    assert_equal(torch_values[:, :, 1::2, :], values1[:, :, 1::2, :])
    assert_equal(torch_indices[:, :, 1::2, :], indices1[:, :, 1::2, :])


@both_directions
@both_engines
def test_a6_tie_straddling_k(W, largest, device):
    """A6: the k=32 cut falls INSIDE a 48-element exact-tie group.

    Per row, with winning sign s (= +1 for largest, -1 for smallest):
    s*4.0 x8 (a tie level landing fully inside the cut), s*2.0 x48 scattered
    at seeded positions (more tied copies than fit: only 24 of 48 survive),
    and unique sub-0.5 fillers elsewhere.  Stability demands the survivors be
    EXACTLY the 24 lowest-index copies of the straddling level, emitted
    index-ascending right after the 8 (also index-ascending) inner-level
    copies.  Strict torch-stable golden, both engines.
    """
    torch.manual_seed(SEED)
    s = 1.0 if largest else -1.0
    inner = torch.full((8,), s * 4.0, dtype=torch.bfloat16)
    straddle = torch.full((48,), s * 2.0, dtype=torch.bfloat16)
    fillers = unique_normal_fillers(W - 56)
    input = shuffled_rows(torch.cat([inner, straddle, fillers]), H, _seed(6, W, largest))

    values, indices = run_stable_topk(input, largest, device)
    golden_values, golden_indices = torch_stable_golden(input, K, largest)
    assert_equal(golden_values, values)
    assert_equal(golden_indices, indices)

    # Redundant with the golden, but documents the cut semantics directly:
    # the kept copies of the straddling level are the 24 lowest of its 48
    # scattered positions, sitting index-ascending at output slots 8..31.
    straddle_val = torch.tensor(s * 2.0, dtype=torch.bfloat16)
    for r in range(H):
        row_positions = (input[0, 0, r] == straddle_val).nonzero().flatten()
        assert_equal(row_positions[: K - 8], indices[0, 0, r, 8:])


@both_directions
@both_engines
def test_a7_index_extremes(W, largest, device):
    """A7 (normal-value variant): an exact tie pair at column 0 and column
    W-1 -- the two index extremes -- must come back as (0, W-1), in that
    order, at the head of the output.  For a power-of-two W, column W-1
    carries the all-ones index payload and column 0 the all-zero one, so this
    exercises both index-bit extremes through the fused {value, index} word
    (and through the comparator's index tiebreak).  30 unique runner-up
    values scattered at seeded interior positions fill the rest of the
    top-k.  Strict torch-stable parity, both engines.
    """
    torch.manual_seed(SEED)
    s = 1.0 if largest else -1.0
    g = torch.Generator().manual_seed(_seed(7, W, largest))
    winner_bits = int(torch.tensor(s * 5.0, dtype=torch.bfloat16).view(torch.int16))
    # 30 unique runner-ups with |v| in (2, 4) and sign s: consecutive bit patterns.
    runner_bits = bits((0x4001 if largest else 0xC001) + torch.arange(30)).view(torch.int16)

    rows = unique_normal_fillers(W).view(torch.int16).expand(H, W).clone()
    rows[:, 0] = winner_bits
    rows[:, W - 1] = winner_bits
    for r in range(H):
        interior = torch.randperm(W - 2, generator=g)[:30] + 1  # never columns 0 / W-1
        rows[r, interior] = runner_bits
    input = rows.view(torch.bfloat16).reshape(1, 1, H, W).contiguous()

    values, indices = run_stable_topk(input, largest, device)
    golden_values, golden_indices = torch_stable_golden(input, K, largest)
    assert_equal(golden_values, values)
    assert_equal(golden_indices, indices)

    # Implied by the golden; stated explicitly for readability: the tied
    # winners appear lowest-index-first, i.e. (0, W-1).
    assert (indices[..., 0] == 0).all() and (indices[..., 1] == W - 1).all()


@both_engines
def test_a7_zero_at_index_zero_smallest(W, device):
    """A7 (+-0 variant, largest=False only): +0.0 pinned at column 0 with 15
    more +0.0s and 16 -0.0s scattered per row; everything else is a unique
    positive, so the bottom-32 is exactly the 32 zeros and index 0 is in the
    top-k.

    The element {value=+0.0, index=0} is the fused engine's ALL-ZERO sort
    word 0x00000000 -- the denormal-flush canary: any stage that flushes,
    special-cases, or drops the all-zero word (as a denormal-flush or "empty
    slot" sentinel would) breaks the strict asserts below.

    Silicon-characterized, both engines: the whole +-0 set forms one tie
    breaking index-ascending, so index 0 must be the FIRST output on BOTH
    engines, and every output value bit pattern must be 0x0000 (the 16 -0.0
    inputs return as +0.0).  Asserted bit-exactly against the canon golden.
    """
    largest = False
    torch.manual_seed(SEED)
    g = torch.Generator().manual_seed(_seed(8, W, largest))
    NEG_ZERO_I16 = bits([0x8000]).view(torch.int16)[0]  # -0.0 bit pattern as int16

    rows = unique_normal_fillers(W).view(torch.int16).expand(H, W).clone()
    for r in range(H):
        rows[r, 0] = 0x0000  # +0.0 pinned at index 0
        scattered = (torch.randperm(W - 1, generator=g)[:31] + 1).sort().values
        signs = torch.zeros(31, dtype=torch.int16)  # 15x +0.0 ...
        signs[0::2] = NEG_ZERO_I16  # ... interleaved with 16x -0.0
        rows[r, scattered] = signs
    input = rows.view(torch.bfloat16).reshape(1, 1, H, W).contiguous()

    values, indices = run_stable_topk(input, largest, device)
    golden_values, golden_indices = canon_stable_golden(input, K, largest)
    assert_equal(golden_indices, indices)
    assert_equal(bf16_bits(golden_values), bf16_bits(values))
    assert (indices[..., 0] == 0).all(), "the all-zero fused word {+0.0, idx 0} must head the zero tie"
    assert (bf16_bits(values) == 0x0000).all(), "all 32 outputs are zeros and must return as +0.0 bits"
