#!/usr/bin/env python3
"""laneMQ selftest — device-independent core of the two-operand 2^32 sem-vs-hand streamer.

Mandated cases (must pass before any silicon run):
  1. ENUMERATION   : dispatches tile [start, start+count) with no gap/overlap; joint
                     sum64/xor32 match an independent full recomputation; whole-dispatch
                     alignment is enforced (partial dispatch rejected).
  2. PAYLOAD       : the interleaved buffer_A payload has the right length; even tiles are
                     a constant base pattern, odd tiles are a consecutive exponent run; and
                     concatenating a base slice's dispatches covers each (base,exp) pair in
                     that slice exactly once (a 2^32-in-miniature full-cover check).
  3. KNOWN-EQUAL   : two identical output streams over a band -> equal per-leg SHA -> BIT-EXACT.
  4. DIVERGENT     : streams differ at exactly one known joint index -> witness bisection
                     recovers that joint index; per-leg SHA differ -> DIVERGENT.
  5. TEXT-GATE     : the reused laneMK object-identity gate passes on match, refuses on
                     mismatch/absence.
"""
import os
import struct
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import binary_stream_lib as B

ELEMS = B.ELEMS_PER_TILE  # 1024


def _model_leg(band_start, band_count, pairs, out_of_pattern):
    """A fake device leg: result bytes = f(joint_index). out_of_pattern maps joint->override.

    The model result region per dispatch is 4 bytes per covered joint (an arbitrary but
    deterministic stand-in for the real packed result region); enough to exercise the
    digest/coverage folding and witness bisection without a device.
    """
    acc = B.LegAccumulator()
    dispatches = 0
    joint_per = pairs * ELEMS
    for ds, span in B.enum_dispatches(band_start, band_count, joint_per):
        acc.update_input(ds, pairs)
        buf = bytearray()
        for j in range(ds, ds + span):
            val = out_of_pattern.get(j, (j * 2654435761) & 0xFFFFFFFF)
            buf += struct.pack("<I", val)
        acc.update_output(bytes(buf))
        dispatches += 1
    return acc, dispatches


def test_enumeration_and_checksums():
    pairs = 16
    joint_per = pairs * ELEMS  # 16384
    start, count = joint_per * 3, joint_per * 40
    tiled = list(B.enum_dispatches(start, count, joint_per))
    assert tiled[0][0] == start
    cur = start
    total = 0
    for ds, span in tiled:
        assert ds == cur, "gap/overlap in dispatch tiling"
        assert span == joint_per
        cur += span
        total += span
    assert total == count and cur == start + count
    # independent checksum recompute over joint indices
    sum64 = sum(range(start, start + count)) & ((1 << 64) - 1)
    xor32 = 0
    for j in range(start, start + count):
        xor32 ^= j
    acc, _ = _model_leg(start, count, pairs, {})
    assert acc.sum64 == sum64 and acc.xor32 == xor32 and acc.joints == count
    # whole-dispatch alignment is enforced
    for bad in ((joint_per + 1, joint_per), (joint_per, joint_per + 1)):
        try:
            list(B.enum_dispatches(bad[0], bad[1], joint_per))
            raise AssertionError("expected alignment rejection")
        except ValueError:
            pass
    print(
        "PASS enumeration+checksums (exact cover, sum64/xor32 match, alignment enforced)"
    )


def test_payload_structure_and_full_cover():
    pairs = 16
    elems = ELEMS
    # Sweep one full base value's entire exponent range and assert every (base,exp) is
    # produced exactly once -- the 2^32 argument in miniature (one base row of the joint
    # grid). base value chosen arbitrarily.
    base_val = (
        0x3F80  # bf16 1.0 (any 16-bit pattern works; value is irrelevant to cover)
    )
    slice_start = base_val << 16
    joint_per = pairs * elems
    seen = {}
    for ds in range(slice_start, slice_start + (1 << 16), joint_per):
        payload = B.interleaved_payload(ds, pairs)
        # length: 2*pairs tiles, each elems bf16 (2 bytes)
        assert len(payload) == 2 * pairs * elems * 2
        vals = struct.unpack(f"<{2 * pairs * elems}H", payload)
        for p in range(pairs):
            even = vals[(2 * p) * elems : (2 * p + 1) * elems]
            odd = vals[(2 * p + 1) * elems : (2 * p + 2) * elems]
            # even tile: constant base
            assert len(set(even)) == 1, "even tile is not a constant base pattern"
            assert even[0] == base_val
            # odd tile: consecutive exponent run
            assert list(odd) == list(
                range(odd[0], odd[0] + elems)
            ), "odd tile not a consecutive exp run"
            for e in odd:
                key = (base_val, e)
                assert key not in seen, f"duplicate joint {key}"
                seen[key] = True
    assert len(seen) == (1 << 16), f"base slice cover {len(seen)} != 65536"
    print(
        f"PASS payload+full-cover (base slice covers all {len(seen)} exponents exactly once)"
    )


def test_known_equal():
    pairs = 16
    joint_per = pairs * ELEMS
    start, count = 0, joint_per * 30
    sem, _ = _model_leg(start, count, pairs, {})
    hand, _ = _model_leg(start, count, pairs, {})
    v = B.band_verdict(sem.sha.hexdigest(), hand.sha.hexdigest())
    assert v == "BIT-EXACT-ALL-INPUTS", v
    assert sem.out_bytes == count * 4 == hand.out_bytes
    print(
        f"PASS known-equal (identical streams -> {v}; sha={sem.sha.hexdigest()[:16]}...)"
    )


def test_divergent_and_witness():
    pairs = 16
    joint_per = pairs * ELEMS
    start, count = 0, joint_per * 50
    witness = joint_per * 17 + 12345  # some joint index mid-band
    sem, _ = _model_leg(start, count, pairs, {})
    hand, _ = _model_leg(start, count, pairs, {witness: 0xDEADBEEF})
    assert B.band_verdict(sem.sha.hexdigest(), hand.sha.hexdigest()) == "DIVERGENT"

    # witness bisection reuses the laneMK first_divergence (4-byte element granularity on
    # the model result stream). run_leg returns the element-aligned XOR-diff mask.
    def run_leg(lo, n):
        sb = _raw(lo, n, {})
        hb = _raw(lo, n, {witness: 0xDEADBEEF})
        return bytes(a ^ b for a, b in zip(sb, hb))

    hit = B.first_divergence(start, count, run_leg, coarse_chunk=1 << 16)
    assert hit == witness, f"bisection found {hit}, expected {witness}"
    print(f"PASS divergent+witness (first differing joint index localized = {hit})")


def _raw(lo, n, override):
    buf = bytearray()
    for j in range(lo, lo + n):
        val = override.get(j, (j * 2654435761) & 0xFFFFFFFF)
        buf += struct.pack("<I", val)
    return bytes(buf)


def test_texthash_gate():
    with tempfile.NamedTemporaryFile("w", suffix=".tsv", delete=False) as fh:
        fh.write("row\tleg\tmath_text_sha256\n")
        fh.write("binarypow\tsem\t" + "ab" * 32 + "\n")
        fh.write("binarypow\thand\t" + "cd" * 32 + "\n")
        path = fh.name
    ok, _ = B.texthash_gate("binarypow", "sem", "ab" * 32, path)
    assert ok, "matching hash should pass"
    ok2, _ = B.texthash_gate("binarypow", "sem", "00" * 32, path)
    assert not ok2, "mismatching hash must refuse"
    ok3, msg = B.texthash_gate("nonexistent", "sem", "x", path)
    assert not ok3 and "NO-CERTIFIED-HASH" in msg, "absent row must refuse"
    os.unlink(path)
    print("PASS text-gate (match passes; mismatch and absent both refuse)")


if __name__ == "__main__":
    test_enumeration_and_checksums()
    test_payload_structure_and_full_cover()
    test_known_equal()
    test_divergent_and_witness()
    test_texthash_gate()
    print("\nALL SELFTESTS PASS")
