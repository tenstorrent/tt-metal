#!/usr/bin/env python3
"""laneMK — sound 2^32 sem-vs-hand streaming-equivalence core (object-identity preserving).

This module holds the DEVICE-INDEPENDENT logic of the persistent-session streamer:
  * chunk enumeration over a [start_bit, end_bit) band of the 2^32 raw-uint32 space,
  * raw little-endian input-pattern bytes per chunk (no host float conversion, so
    subnormals / NaN payloads are delivered exactly — matches the fitter producer),
  * per-leg streaming SHA-256 over the fp32 OUTPUT bytes in ascending input order
    (equality between the two legs' digests over the SAME band tiling = BIT-EXACT),
  * input sum64 / xor32 checksums (coverage attestation, matches the fitter),
  * first-divergence witness localization by band bisection,
  * the .text object-identity gate against the corpus TEXTHASHES.

The device leg (open ONE session, per chunk: write raw A to L1, clear Res, run_elf,
wait, read Res, sha.update) lives in the harness (test_config.run_fp32_stream) and the
orchestrator; this file is the part that can be unit-verified with no device, and is
what the selftest exercises for the known-equal / divergent-witness / .text-gate cases.
"""
from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, field
from typing import Callable, Iterator, Optional

TWO32 = 1 << 32


def enum_chunks(
    start_bit: int, count: int, chunk_patterns: int
) -> Iterator[tuple[int, int]]:
    """Yield (chunk_start, valid_count) contiguous chunks tiling [start_bit, start_bit+count)."""
    if start_bit < 0 or count < 0 or start_bit + count > TWO32:
        raise ValueError(f"band [{start_bit},{start_bit+count}) out of [0,2^32)")
    if chunk_patterns <= 0:
        raise ValueError("chunk_patterns must be positive")
    done = 0
    while done < count:
        valid = min(chunk_patterns, count - done)
        yield start_bit + done, valid
        done += valid


def chunk_input_bytes(chunk_start: int, valid_count: int) -> bytes:
    """Raw little-endian uint32 pattern bytes for a chunk: [chunk_start, chunk_start+valid)."""
    # struct little-endian; no float() ever touches the bits (subnormals/NaN exact).
    return struct.pack(
        f"<{valid_count}I", *range(chunk_start, chunk_start + valid_count)
    )


@dataclass
class LegAccumulator:
    """Per-leg streaming digest + coverage checksums, folded chunk by chunk in order."""

    sha: "hashlib._Hash" = field(default_factory=hashlib.sha256)
    patterns: int = 0
    out_bytes: int = 0
    sum64: int = 0
    xor32: int = 0

    def update_input(self, chunk_start: int, valid_count: int) -> None:
        for b in range(chunk_start, chunk_start + valid_count):
            self.sum64 = (self.sum64 + b) & ((1 << 64) - 1)
            self.xor32 ^= b
        self.patterns += valid_count

    def update_output(self, out_bytes: bytes, valid_count: int) -> None:
        want = valid_count * 4
        if len(out_bytes) < want:
            raise ValueError(f"leg produced {len(out_bytes)} < {want} output bytes")
        self.sha.update(out_bytes[:want])
        self.out_bytes += want

    def producer_line(self, tag: str, start_bit: int, end_bit: int, chunks: int) -> str:
        return (
            f"FP32_STREAM_PRODUCER_COVERAGE,{tag},"
            f"complete=true,start_bit={start_bit},count={self.patterns},"
            f"end_bit_exclusive={end_bit},chunks={chunks},"
            f"expected_bytes={self.patterns*4},observed_bytes={self.out_bytes},"
            f"input_bit_sum_mod_2_64={self.sum64},input_bit_xor={self.xor32},"
            f"output_sha256={self.sha.hexdigest()}"
        )


def band_verdict(sem_sha: str, hand_sha: str) -> str:
    return "BIT-EXACT-ALL-INPUTS" if sem_sha == hand_sha else "DIVERGENT"


def first_divergence(
    start_bit: int,
    count: int,
    run_leg: Callable[[int, int], bytes],
    coarse_chunk: int = 1 << 20,
) -> Optional[int]:
    """Locate the first input bit-pattern where the two legs' outputs differ.

    run_leg(chunk_start, valid_count) -> concatenated (sem_out || hand_out) is NOT the
    contract; instead run_leg returns the XOR-diff mask bytes so this stays leg-agnostic:
    it must return b"\\x00"*(valid*4) where the legs agree and non-zero where they differ,
    element-aligned. Bisection narrows to the first differing 4-byte element and returns
    its input bit pattern, or None if the whole band agrees.
    """

    def first_diff_in(lo: int, n: int) -> Optional[int]:
        mask = run_leg(lo, n)
        for i in range(n):
            if mask[i * 4 : i * 4 + 4] != b"\x00\x00\x00\x00":
                return lo + i
        return None

    done = 0
    while done < count:
        n = min(coarse_chunk, count - done)
        hit = first_diff_in(start_bit + done, n)
        if hit is not None:
            return hit
        done += n
    return None


def texthash_gate(
    row: str,
    leg: str,
    measured_text_sha256: str,
    texthashes_tsv: str,
) -> tuple[bool, str]:
    """Object-identity gate: measured kernel .text sha256 must equal the certified corpus one.

    Returns (ok, expected). Refuse (ok=False) if the row/leg is absent OR mismatches — a
    verdict on a different ELF is not a verdict on the certified corpus kernel.
    """
    expected = None
    with open(texthashes_tsv) as fh:
        header = fh.readline()
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3 and parts[0] == row and parts[1] == leg:
                expected = parts[2].strip()
                break
    if expected is None:
        return False, f"NO-CERTIFIED-HASH row={row} leg={leg}"
    return (measured_text_sha256.strip() == expected), expected
