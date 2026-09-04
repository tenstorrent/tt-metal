#!/usr/bin/env python3
"""laneMK selftest — device-independent core of the 2^32 sem-vs-hand streamer.

Mandated cases (must pass before any galaxy run):
  1. KNOWN-EQUAL   : two identical output streams over a band -> equal per-leg SHA -> BIT-EXACT.
  2. DIVERGENT     : streams differ at exactly one known input pattern -> witness bisection
                     recovers that pattern; per-leg SHA differ -> DIVERGENT.
  3. TEXT-GATE     : matching .text sha -> pass; absent or mismatching -> refuse.
Also checks: chunk enumeration tiles the band with no gap/overlap and correct coverage
checksums (sum64/xor32) against an independent full recomputation.
"""
import os
import struct
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fp32_stream_lib as L


def _model_leg(band_start, band_count, out_of_pattern):
    """A fake leg: output bytes = f(input_bits). out_of_pattern maps input_bit->override."""
    acc = L.LegAccumulator()
    chunks = 0
    for cs, vc in L.enum_chunks(band_start, band_count, 4096):
        acc.update_input(cs, vc)
        buf = bytearray()
        for b in range(cs, cs + vc):
            val = out_of_pattern.get(
                b, (b * 2654435761) & 0xFFFFFFFF
            )  # arbitrary deterministic map
            buf += struct.pack("<I", val)
        acc.update_output(bytes(buf), vc)
        chunks += 1
    return acc, chunks


def test_enumeration_and_checksums():
    start, count = 1000, 100000
    tiled = list(L.enum_chunks(start, count, 4096))
    # no gap / no overlap / exact cover
    assert tiled[0][0] == start
    cur = start
    total = 0
    for cs, vc in tiled:
        assert cs == cur, "gap/overlap in tiling"
        cur += vc
        total += vc
    assert total == count and cur == start + count
    # independent checksum recompute
    sum64 = sum(range(start, start + count)) & ((1 << 64) - 1)
    xor32 = 0
    for b in range(start, start + count):
        xor32 ^= b
    acc, _ = _model_leg(start, count, {})
    assert acc.sum64 == sum64 and acc.xor32 == xor32 and acc.patterns == count
    print(
        "PASS enumeration+checksums (exact cover, sum64/xor32 match independent recompute)"
    )


def test_known_equal():
    start, count = 0, 200000
    sem, cs_ = _model_leg(start, count, {})
    hand, _ = _model_leg(start, count, {})
    v = L.band_verdict(sem.sha.hexdigest(), hand.sha.hexdigest())
    assert v == "BIT-EXACT-ALL-INPUTS", v
    assert sem.out_bytes == count * 4 == hand.out_bytes
    print(
        f"PASS known-equal (identical streams -> {v}; sha={sem.sha.hexdigest()[:16]}...)"
    )


def test_divergent_and_witness():
    start, count = 0, 500000
    witness = 314159
    sem, _ = _model_leg(start, count, {})
    hand, _ = _model_leg(start, count, {witness: 0xDEADBEEF})
    assert L.band_verdict(sem.sha.hexdigest(), hand.sha.hexdigest()) == "DIVERGENT"

    # witness bisection: run_leg returns element-aligned XOR-diff mask for a sub-band.
    def run_leg(lo, n):
        s, _ = _model_leg(lo, n, {})
        h, _ = _model_leg(
            lo, n, {witness: 0xDEADBEEF} if lo <= witness < lo + n else {}
        )
        # rebuild raw output byte streams to diff (the model stores only sha; recompute here)
        sb = _raw(lo, n, {})
        hb = _raw(lo, n, {witness: 0xDEADBEEF})
        return bytes(a ^ b for a, b in zip(sb, hb))

    hit = L.first_divergence(start, count, run_leg, coarse_chunk=1 << 16)
    assert hit == witness, f"bisection found {hit}, expected {witness}"
    print(f"PASS divergent+witness (first differing input pattern localized = {hit})")


def _raw(lo, n, override):
    buf = bytearray()
    for b in range(lo, lo + n):
        val = override.get(b, (b * 2654435761) & 0xFFFFFFFF)
        buf += struct.pack("<I", val)
    return bytes(buf)


def test_texthash_gate():
    with tempfile.NamedTemporaryFile("w", suffix=".tsv", delete=False) as fh:
        fh.write("row\tleg\tmath_text_sha256\n")
        fh.write(
            "sign\tsem\te81335acfc1f851b928e7fe002aa344df7e0835f31161a13865342dbabdaabae\n"
        )
        fh.write(
            "sign\thand\t9f875162e5b47499e9cab50894cc16c82363cc9c9b1613f9b80cb68b4ef0654d\n"
        )
        path = fh.name
    ok, exp = L.texthash_gate(
        "sign",
        "sem",
        "e81335acfc1f851b928e7fe002aa344df7e0835f31161a13865342dbabdaabae",
        path,
    )
    assert ok, "matching hash should pass"
    ok2, _ = L.texthash_gate("sign", "sem", "deadbeef" * 8, path)
    assert not ok2, "mismatching hash must refuse"
    ok3, msg = L.texthash_gate("nonexistent", "sem", "x", path)
    assert not ok3 and "NO-CERTIFIED-HASH" in msg, "absent row must refuse"
    os.unlink(path)
    print("PASS text-gate (match passes; mismatch and absent both refuse)")


if __name__ == "__main__":
    test_enumeration_and_checksums()
    test_known_equal()
    test_divergent_and_witness()
    test_texthash_gate()
    print("\nALL SELFTESTS PASS")
