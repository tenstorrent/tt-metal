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


def _fields(elem_bits: int) -> tuple[int, int, int, int]:
    """(sign_shift, exp_shift, mant_mask, word_mask) for an IEEE-ish float of
    `elem_bits`. fp32 = 32 (1+8+23); bf16 = 16 (1+8+7). Both keep the 8-bit
    biased exponent, so the exponent grid 0..255 maps directly across widths."""
    if elem_bits == 32:
        return 31, 23, 0x7FFFFF, MASK32
    if elem_bits == 16:
        return 15, 7, 0x7F, 0xFFFF  # bf16 = high 16 bits of fp32
    raise ValueError(f"unsupported elem_bits {elem_bits}")


def _special_words(elem_bits: int) -> list[int]:
    if elem_bits == 32:
        return list(_SPECIAL_WORDS)
    return [w >> 16 for w in _SPECIAL_WORDS]  # bf16 = fp32 truncated to high 16


def _canon(word32: int, elem_bits: int) -> int:
    """A canonical fp32 constant re-expressed at `elem_bits` (bf16 = >>16)."""
    return word32 if elem_bits == 32 else (word32 >> 16)


def _exp_tile(per_run: int, exp: int, prng_state: int, elem_bits: int):
    """A tile at biased exponent `exp`, cycling mantissa corners + both signs."""
    sign_sh, exp_sh, mant_mask, _ = _fields(elem_bits)
    corners = [0, mant_mask, mant_mask // 2 + 1]
    out = []
    for i in range(per_run):
        sign = (i & 1) << sign_sh
        if i % 4 == 3:
            prng_state, r = splitmix64(prng_state)
            mant = r & mant_mask
        else:
            mant = corners[i % 3]
        out.append(sign | (exp << exp_sh) | mant)
    return out, prng_state


def _structure_tiles(n: int, prng_state: int, elem_bits: int):
    """Cross-lane-structure tiles of width n (format-correct constants)."""
    _, _, _, wmask = _fields(elem_bits)
    one = _canon(0x3F800000, elem_bits)  # 1.0
    negone = _canon(0xBF800000, elem_bits)  # -1.0
    two = _canon(0x40000000, elem_bits)  # 2.0
    pi = _canon(0x40490FDB, elem_bits)  # pi
    tiles = [
        [i & wmask for i in range(n)],  # ascending int-pattern ramp
        [(n - 1 - i) & wmask for i in range(n)],  # descending ramp
        [pi] * n,  # all-equal
        ([one, two] * (n // 2 + 1))[:n],  # two-value alternating
        [one] + [0] * (n - 1),  # single-hot lane 0
        [0] * (n - 1) + [one],  # single-hot last lane
        ([one] * (n // 2) + [negone] * (n - n // 2)),  # tie-heavy +/-1
    ]
    return tiles, prng_state


def iter_tiles(
    op: str, per_run: int, n_tiles: int, base_seed: int, elem_bits: int = 32
):
    """Yield exactly `n_tiles` operand-A tiles (each a list of `per_run` values of
    `elem_bits` each), deterministically for (op, base_seed, elem_bits). Strata
    come first (specials, exp-grid, structure), then uniform-random fill for the
    remainder of the budget. elem_bits = 32 for fp32 operands, 16 for bf16."""
    if per_run <= 0 or n_tiles <= 0:
        return
    _, _, _, wmask = _fields(elem_bits)
    specials = _special_words(elem_bits)
    state = _seed_for(op, base_seed)
    emitted = 0

    # 1. SPECIALS — one whole-tile per special word, then a mixed-specials tile.
    for w in specials:
        if emitted >= n_tiles:
            return
        yield [w] * per_run
        emitted += 1
    if emitted < n_tiles:
        yield [specials[i % len(specials)] for i in range(per_run)]
        emitted += 1

    # 2. EXP-GRID — biased exponent sweep 0..255.
    for exp in range(256):
        if emitted >= n_tiles:
            return
        tile, state = _exp_tile(per_run, exp, state, elem_bits)
        yield tile
        emitted += 1

    # 3. STRUCTURE — cross-lane fold-order shapes.
    structs, state = _structure_tiles(per_run, state, elem_bits)
    for tile in structs:
        if emitted >= n_tiles:
            return
        yield tile
        emitted += 1

    # 4. RANDOM — uniform fill for the rest of the budget.
    while emitted < n_tiles:
        tile = []
        for _ in range(per_run):
            state, r = splitmix64(state)
            tile.append(r & wmask)
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


def stream_on_device(configuration, location, spec, op, wait_to):
    """Shared persistent-session device leg for the LANEMO_SAMPLE hooks (unary +
    blaze). ONE open device session: per dispatch inject the next stratified
    operand-A sample tile (raw bits, subnormals/NaN exact), run the certified
    kernel, read Res, fold output SHA-256 + checkpoint SHAs. Only operand A is
    sampled (harness has no lanejn_raw_b); an op's operand B keeps its fixed
    generated stimulus.

    spec = "n_tiles,seed,ckpt,outfile". Emits one LANEMO_SAMPLE_RESULT line.
    Object identity is preserved by construction (same `configuration`/ELF).
    """
    import hashlib
    import time

    n_s, seed_s, ckpt_s, outfile = spec.split(",", 3)
    n_tiles = int(n_s, 0)
    seed = int(seed_s, 0)
    ckpt = max(1, int(ckpt_s, 0))
    st = configuration.variant_stimuli
    per_run = int(st.tile_count_A) * 1024
    # Element byte-width from the operand's tile size (1024 elements per tile):
    # 4 => fp32, 2 => bf16. The raw-injection payload must match this exactly.
    elem_bytes = int(st.tile_size_A_bytes) // 1024
    elem_bits = elem_bytes * 8
    pack_code = {4: "I", 2: "H"}.get(elem_bytes)
    if pack_code is None:
        raise RuntimeError(f"unsupported operand-A elem_bytes {elem_bytes}")

    configuration.prepare()
    configuration.write_runtimes_to_L1()

    osha = hashlib.sha256()
    isha = hashlib.sha256()
    ckpt_shas = []
    ck = hashlib.sha256()
    tiles = 0
    obytes = 0
    t0 = time.time()
    for tile in iter_tiles(op, per_run, n_tiles, seed, elem_bits):
        raw = struct.pack(f"<{per_run}{pack_code}", *tile)
        st.lanejn_raw_a = raw
        st.write(location)
        st.clear_result_buffer(location)
        configuration.run_elf_files()
        configuration.wait_for_tensix_operations_finished(timeout=wait_to)
        res = st.collect_raw_result_bytes(location)
        want = int(st.buf_res_tile_size) * int(st.tile_count_res)
        if len(res) < want:
            raise RuntimeError(f"sample {tiles}: got {len(res)} result bytes < {want}")
        out = res[:want]
        osha.update(out)
        isha.update(raw)
        ck.update(out)
        obytes += want
        tiles += 1
        if tiles % ckpt == 0:
            ckpt_shas.append(ck.hexdigest())
            ck = hashlib.sha256()
    if tiles % ckpt != 0:
        ckpt_shas.append(ck.hexdigest())
    dt = time.time() - t0

    line = (
        "LANEMO_SAMPLE_RESULT,"
        f"op={op},n_samples={tiles},per_run_elems={per_run},seed=0x{seed:016x},"
        f"ckpt={ckpt},wall_s={dt:.3f},"
        f"per_run_ms={(1000.0 * dt / tiles) if tiles else 0:.3f},"
        f"input_sha256={isha.hexdigest()},output_sha256={osha.hexdigest()}"
    )
    print(line, flush=True)
    with open(outfile, "w") as fh:
        fh.write(line + "\n")
        fh.write("checkpoint_shas\t" + ",".join(ckpt_shas) + "\n")
