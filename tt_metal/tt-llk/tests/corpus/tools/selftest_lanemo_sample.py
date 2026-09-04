#!/usr/bin/env python3
"""laneMO selftest — device-independent core of the stratified sampler.

Mandated cases (must pass before any galaxy run):
  1. DETERMINISM  : same (op, seed) reproduces the identical sample stream; a
                    different seed (or op) changes it.
  2. STRATA       : the stream begins with the special-value corner tiles and
                    covers the biased-exponent grid (0..255) and the cross-lane
                    structure shapes.
  3. KNOWN-EQUAL  : two model legs computing the SAME function of the input give
                    equal per-leg output SHA -> SAMPLED-CONSISTENT (0 diffs).
  4. SEEDED-DIVERGENT : a model leg that perturbs exactly one sample gives a
                    different output SHA -> SAMPLED-DIVERGENT, and the checkpoint
                    SHAs localize the diverging window.
"""
import hashlib
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lanemo_sample_gen as G

PER_RUN = 64  # small tile for the selftest (device uses tile_count_A*1024)


def _leg(op, per_run, n, seed, perturb=None):
    """Model leg: out = f(in). perturb=(sample_index, xor_word) flips one sample."""
    osha = hashlib.sha256()
    ck = hashlib.sha256()
    ckpts = []
    ckpt = 8
    for i, tile in enumerate(G.iter_tiles(op, per_run, n, seed)):
        vals = [
            (w * 2654435761) & 0xFFFFFFFF for w in tile
        ]  # arbitrary deterministic map
        if perturb and i == perturb[0]:
            vals[0] ^= perturb[1]
        out = struct.pack(f"<{per_run}I", *vals)
        osha.update(out)
        ck.update(out)
        if (i + 1) % ckpt == 0:
            ckpts.append(ck.hexdigest())
            ck = hashlib.sha256()
    if n % ckpt != 0:
        ckpts.append(ck.hexdigest())
    return osha.hexdigest(), ckpts


def test_determinism():
    a = list(G.iter_tiles("welford", PER_RUN, 300, 0x1))
    b = list(G.iter_tiles("welford", PER_RUN, 300, 0x1))
    assert a == b, "same (op,seed) must reproduce the identical stream"
    c = list(G.iter_tiles("welford", PER_RUN, 300, 0x2))
    assert a != c, "different seed must change the stream"
    d = list(G.iter_tiles("cumsum", PER_RUN, 300, 0x1))
    assert a != d, "different op must change the stream"
    print("PASS determinism (stream is a pure function of (op, seed))")


def test_strata():
    tiles = list(G.iter_tiles("welford", PER_RUN, 300, 0x1))
    # specials come first: tile 0 is all +0, tile 2 is all +inf.
    assert tiles[0] == [0x00000000] * PER_RUN, "first special tile must be +0"
    assert tiles[2] == [0x7F800000] * PER_RUN, "third special tile must be +inf"
    n_special = len(G._SPECIAL_WORDS) + 1
    # exponent grid follows: some tile at exponent 200 exists.
    exp_tiles = tiles[n_special : n_special + 256]
    assert any(
        ((t[0] >> 23) & 0xFF) == 200 for t in exp_tiles
    ), "exp-grid must cover exp=200"
    # structure tiles include an all-equal (pi) tile.
    struct_tiles = tiles[n_special + 256 :]
    assert any(
        len(set(t)) == 1 and t[0] == 0x40490FDB for t in struct_tiles
    ), "structure stratum must include the all-equal tile"
    print(
        f"PASS strata (specials-first, exp-grid 0..255, structure shapes; "
        f"{n_special} specials + 256 exp + structure)"
    )


def test_known_equal():
    sem, sem_ck = _leg("welford", PER_RUN, 500, 0x1)
    hand, hand_ck = _leg("welford", PER_RUN, 500, 0x1)
    assert sem == hand, "identical model legs must give equal output SHA"
    assert sem_ck == hand_ck
    verdict = "SAMPLED-CONSISTENT" if sem == hand else "SAMPLED-DIVERGENT"
    assert verdict == "SAMPLED-CONSISTENT"
    print(f"PASS known-equal (identical legs -> {verdict}, 0 diffs; sha={sem[:16]}...)")


def test_seeded_divergent():
    n, ckpt = 500, 8
    perturbed_sample = 123  # falls in checkpoint window 123//8 = 15
    sem, sem_ck = _leg("welford", PER_RUN, n, 0x1)
    hand, hand_ck = _leg(
        "welford", PER_RUN, n, 0x1, perturb=(perturbed_sample, 0xDEADBEEF)
    )
    assert sem != hand, "a one-sample perturbation must change the output SHA"
    first = next((i for i, (a, b) in enumerate(zip(sem_ck, hand_ck)) if a != b), -1)
    assert (
        first == perturbed_sample // ckpt
    ), f"checkpoint localization found window {first}, expected {perturbed_sample // ckpt}"
    lo, hi = first * ckpt, first * ckpt + ckpt
    assert lo <= perturbed_sample < hi
    print(
        f"PASS seeded-divergent (SAMPLED-DIVERGENT; witness in checkpoint "
        f"window [{lo},{hi}) containing sample {perturbed_sample})"
    )


if __name__ == "__main__":
    test_determinism()
    test_strata()
    test_known_equal()
    test_seeded_divergent()
    print("\nALL SELFTESTS PASS")
