#!/usr/bin/env python3
"""laneMO — stratified operand-A sample generator for high-coverage silicon
differential sampling of the cross-lane / multi-input ops.

This is the DEVICE-INDEPENDENT heart of the sampler (the analog of
``fp32_stream_lib`` for laneMK): given a deterministic seed and a chunk index,
it emits the exact raw little-endian uint32 bytes for one dispatch's operand-A
tile stream. The SAME (op, seed) reproduces the SAME stream bit-for-bit, so the
sem leg and the hand leg are fed byte-identical inputs and their output bytes
can be compared directly (per-leg streaming SHA + diff count).

WHY sampling, not exhaustion: these ops read a whole tile jointly (cross-lane
reduce / scan / sort / broadcast), so the input space is 2^(32*lanes) — far
beyond exhaustion. We stratify operand A's value space instead and sample a
defensible budget; a verdict from this file is SAMPLED-CONSISTENT (N samples,
0 diffs) — a DISTINCT, WEAKER class than a proof over all inputs. Never call a
sampled op "verified" or "proven".

Stratification (per the laneJN structured-sampling spec, single-operand form):
  * SPECIALS  — whole tiles of {+0,-0,+inf,-inf,qNaN,sNaN,denorm-min,denorm-max,
                max-normal,1.0,-1.0} and a specials-mix tile. Deterministic,
                emitted first so every run covers the corner set.
  * EXP-GRID  — for each biased exponent 0..255, a tile at that exponent with
                mantissa corners {0, 0x7fffff, mid, random} and both signs
                (exponent sweep = the "256-cell exponent grid" for one operand).
  * STRUCTURE — cross-lane-structure tiles the reduce/scan/sort/top-k paths care
                about: ascending, descending, all-equal, tie-heavy, single-hot,
                two-value. (A reduce over an all-equal tile, a sort over a
                reversed tile, ties in a top-k — the shapes that exercise the
                cross-lane fold order.)
  * RANDOM    — the bulk of the budget: uniform-random uint32 tiles from a
                seeded SplitMix64 PRNG (subnormals / NaN payloads delivered
                exactly — no host float() ever touches the bits).

Only operand A is sampled: the tt-llk stimuli harness exposes raw injection for
operand A only (``lanejn_raw_a``); operand B (when an op has one) keeps its
fixed generated stimulus. For genuinely binary ops that is PARTIAL sampling
(A varies, B fixed) and the sweep records it honestly as such.
"""
from __future__ import annotations

import struct

MASK32 = 0xFFFFFFFF
MASK64 = (1 << 64) - 1

# Corner fp32 bit-patterns (as uint32).
_SPECIAL_WORDS = [
    0x00000000,  # +0
    0x80000000,  # -0
    0x7F800000,  # +inf
    0xFF800000,  # -inf
    0x7FC00000,  # qNaN
    0x7F800001,  # sNaN
    0x00000001,  # smallest positive denormal
    0x007FFFFF,  # largest denormal
    0x7F7FFFFF,  # max positive normal
    0xFF7FFFFF,  # max negative normal
    0x3F800000,  # 1.0
    0xBF800000,  # -1.0
]


def splitmix64(state: int) -> tuple[int, int]:
    """One SplitMix64 step: returns (next_state, output). Pure, deterministic."""
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    z = z ^ (z >> 31)
    return state, z


def _seed_for(op: str, base_seed: int) -> int:
    """Stable per-op seed = base_seed folded with the op name bytes."""
    s = base_seed & MASK64
    for b in op.encode("utf-8"):
        s = (s * 1099511628211 + b) & MASK64  # FNV-1a-ish fold
    return s or 1


def _exp_tile(words: list[int], exp: int, prng_state: int) -> tuple[list[int], int]:
    """Fill `words` in place-style (returns new list) at biased exponent `exp`,
    cycling mantissa corners {0, max, mid, random} and both signs."""
    corners = [0x000000, 0x7FFFFF, 0x400000]
    out = []
    for i in range(len(words)):
        sign = (i & 1) << 31
        if i % 4 == 3:
            prng_state, r = splitmix64(prng_state)
            mant = r & 0x7FFFFF
        else:
            mant = corners[i % 3]
        out.append(sign | (exp << 23) | mant)
    return out, prng_state


def _structure_tiles(n: int, prng_state: int) -> tuple[list[list[int]], int]:
    """Cross-lane-structure tiles of width n."""
    prng_state, r = splitmix64(prng_state)
    hot = 0x3F800000  # 1.0
    tiles = [
        list(range(n)),  # ascending int-pattern ramp
        list(range(n - 1, -1, -1)),  # descending ramp
        [0x40490FDB] * n,  # all-equal (pi)
        ([0x3F800000, 0x40000000] * (n // 2 + 1))[:n],  # two-value alternating
        [hot] + [0] * (n - 1),  # single-hot lane 0
        [0] * (n - 1) + [hot],  # single-hot last lane
        ([0x3F800000] * (n // 2) + [0xBF800000] * (n - n // 2)),  # tie-heavy +/-1
    ]
    return tiles, prng_state


def iter_tiles(op: str, per_run: int, n_tiles: int, base_seed: int):
    """Yield exactly `n_tiles` operand-A tiles (each a list of `per_run` uint32),
    deterministically for (op, base_seed). Strata come first (specials, exp-grid,
    structure), then uniform-random fill for the remainder of the budget."""
    if per_run <= 0 or n_tiles <= 0:
        return
    state = _seed_for(op, base_seed)
    emitted = 0

    # 1. SPECIALS — one whole-tile per special word, then a mixed-specials tile.
    for w in _SPECIAL_WORDS:
        if emitted >= n_tiles:
            return
        yield [w] * per_run
        emitted += 1
    if emitted < n_tiles:
        mix = [_SPECIAL_WORDS[i % len(_SPECIAL_WORDS)] for i in range(per_run)]
        yield mix
        emitted += 1

    # 2. EXP-GRID — biased exponent sweep 0..255.
    for exp in range(256):
        if emitted >= n_tiles:
            return
        tile, state = _exp_tile([0] * per_run, exp, state)
        yield tile
        emitted += 1

    # 3. STRUCTURE — cross-lane fold-order shapes.
    structs, state = _structure_tiles(per_run, state)
    for tile in structs:
        if emitted >= n_tiles:
            return
        yield tile
        emitted += 1

    # 4. RANDOM — uniform uint32 fill for the rest of the budget.
    while emitted < n_tiles:
        tile = []
        for _ in range(per_run):
            state, r = splitmix64(state)
            tile.append(r & MASK32)
        yield tile
        emitted += 1


def chunk_bytes(
    op: str, per_run: int, chunk_index: int, tiles_per_chunk: int, base_seed: int
) -> tuple[bytes, int]:
    """Raw little-endian uint32 bytes for one dispatch chunk = `tiles_per_chunk`
    tiles starting at tile (chunk_index * tiles_per_chunk). Returns (bytes,
    n_tiles_in_chunk). Deterministic; sem and hand replay the same stream.

    NOTE: reproduces the whole prefix up to the chunk (the strata generator is a
    stream) — callers stream forward chunk 0,1,2,... so this is O(total), not
    O(chunk^2), when driven in order via `iter_tiles`. `chunk_bytes` is the
    random-access helper the selftest uses; the device sweep uses `iter_tiles`.
    """
    first = chunk_index * tiles_per_chunk
    last = first + tiles_per_chunk
    words: list[int] = []
    n = 0
    for idx, tile in enumerate(iter_tiles(op, per_run, last, base_seed)):
        if idx < first:
            continue
        words.extend(tile)
        n += 1
    return struct.pack(f"<{len(words)}I", *words), n


def stream_sha_and_count(op: str, per_run: int, n_tiles: int, base_seed: int, out_fn):
    """Fold every sampled input tile through `out_fn(raw_tile_bytes) -> out_bytes`
    (a leg: sem or hand), returning (output_sha256_hex, tiles, out_bytes, in_sha).
    Used by both the device sweep and the selftest's model legs."""
    import hashlib

    osha = hashlib.sha256()
    isha = hashlib.sha256()
    tiles = 0
    obytes = 0
    for tile in iter_tiles(op, per_run, n_tiles, base_seed):
        raw = struct.pack(f"<{per_run}I", *tile)
        isha.update(raw)
        out = out_fn(raw)
        osha.update(out)
        obytes += len(out)
        tiles += 1
    return osha.hexdigest(), tiles, obytes, isha.hexdigest()
