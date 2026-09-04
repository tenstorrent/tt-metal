#!/usr/bin/env python3
"""laneMQ — sound 2^32 sem-vs-hand streaming-equivalence core for a TWO-operand op.

This is the laneMK single-operand streamer widened by exactly one dimension: instead
of sweeping one raw uint32 per element, it sweeps the JOINT bf16 x bf16 space of a
binary op whose two operands are each an independent 16-bit draw. The joint index

    J in [0, 2^32),   base16 = J >> 16,   exp16 = J & 0xFFFF

enumerates every (base, exponent) bf16 bit-pattern pair exactly once. base16/exp16 are
raw bf16 BIT PATTERNS (no host float() ever touches them), so subnormals / inf / NaN
payloads are delivered exactly, the same discipline as the laneMK raw-A path.

DEVICE ABI (SFPU binary pow / xlogy / ...): the kernel reads operand 0 from the EVEN
tile of each tile pair and operand 1 from the ODD tile, both out of buffer_A (see
sources/sfpu_binary_test.cpp: `for (tile = 0; tile < N; tile += 2)` consuming tile and
tile+1). So the two-operand injection is a single interleaved buffer_A payload
(even = base, odd = exponent) driven through the EXISTING validated laneJN raw-A path
-- no new device write path is needed for this ABI. (For a binary op that instead keeps
operand B in a separate buffer_B, helpers/stimuli_config.py grows a mirror `lanejn_raw_b`
path; this module targets the interleaved-A ABI, which is what binary pow uses.)

This file holds the DEVICE-INDEPENDENT logic (enumeration, interleaved-payload packing,
coverage checksums, per-leg digest, band verdict). The device leg (open ONE session,
per dispatch: write raw A, clear Res, run_elf, wait, read Res, sha.update) lives in the
harness hook (test_sfpu_binary._lanemk_run_binary_stream) and the orchestrator
(binary_stream_sweep.py). The selftest exercises exactly the functions here.
"""
from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, field
from typing import Iterator

# Reuse the laneMK object-identity gate and first-divergence helpers verbatim; only the
# input enumeration / payload packing differs for the two-operand sweep.
from fp32_stream_lib import first_divergence, texthash_gate  # noqa: F401

TWO32 = 1 << 32
ELEMS_PER_TILE = 1024  # a 32x32 bf16 tile is 1024 elements
_U64 = (1 << 64) - 1


def joint_base_exp(j: int) -> tuple[int, int]:
    """Split a joint index into (base16, exp16) raw bf16 bit patterns."""
    return (j >> 16) & 0xFFFF, j & 0xFFFF


def enum_dispatches(
    start: int, count: int, joint_per_dispatch: int
) -> Iterator[tuple[int, int]]:
    """Yield (dispatch_start, joint_per_dispatch) tiling [start, start+count).

    The two-operand sweep requires whole dispatches: `start` and `count` must both be
    multiples of `joint_per_dispatch`. This is always satisfiable because the joint
    space is 2^32 and both band sizes and joint_per_dispatch are chosen as powers of two,
    so there is never a partial final dispatch to pad -- which lets the device leg hash
    the whole (cleared) result region without excluding padding bytes.
    """
    if start < 0 or count < 0 or start + count > TWO32:
        raise ValueError(f"band [{start},{start+count}) out of [0,2^32)")
    if joint_per_dispatch <= 0:
        raise ValueError("joint_per_dispatch must be positive")
    if start % joint_per_dispatch or count % joint_per_dispatch:
        raise ValueError(
            f"two-operand sweep needs whole dispatches: start={start} count={count} "
            f"must both be multiples of joint_per_dispatch={joint_per_dispatch}"
        )
    done = 0
    while done < count:
        yield start + done, joint_per_dispatch
        done += joint_per_dispatch


def interleaved_payload(
    dispatch_start: int, pairs: int, elems_per_tile: int = ELEMS_PER_TILE
) -> bytes:
    """Interleaved buffer_A payload for one dispatch of `pairs` tile-pairs.

    Tile layout written to L1 (concatenated, tile 0..2*pairs-1):
        even tile 2p  = operand 0 (base)     : `elems_per_tile` copies of one bf16 pattern
        odd  tile 2p+1= operand 1 (exponent) : `elems_per_tile` consecutive bf16 patterns

    Pair p covers the joint run [x, x+elems_per_tile) with x = dispatch_start + p*elems.
    Within that run base16 = x>>16 is constant and exp16 = (x&0xFFFF)+i sweeps a
    contiguous block of bf16 patterns. This is exact and gap-free provided a run never
    crosses a 2^16 base boundary, which holds whenever `elems_per_tile` divides 2^16 and
    `dispatch_start` is a multiple of elems_per_tile (both guaranteed by the caller:
    elems_per_tile=1024 divides 65536, and dispatches are elems-aligned). The intra-tile
    face permutation of the 1024 slots is irrelevant to the verdict: both legs receive
    the identical payload, so it only permutes which output slot holds which joint point,
    and coverage is a set property.
    """
    if 65536 % elems_per_tile:
        raise ValueError(f"elems_per_tile={elems_per_tile} must divide 65536")
    if dispatch_start % elems_per_tile:
        raise ValueError(
            f"dispatch_start={dispatch_start} must be a multiple of elems_per_tile"
        )
    buf = bytearray()
    for p in range(pairs):
        x = dispatch_start + p * elems_per_tile
        base16 = (x >> 16) & 0xFFFF
        lo = x & 0xFFFF
        if lo + elems_per_tile > 65536:
            raise ValueError(f"pair {p} run crosses a 2^16 base boundary (lo={lo})")
        buf += struct.pack(f"<{elems_per_tile}H", *([base16] * elems_per_tile))
        buf += struct.pack(f"<{elems_per_tile}H", *range(lo, lo + elems_per_tile))
    return bytes(buf)


def covered_joints(
    dispatch_start: int, pairs: int, elems_per_tile: int = ELEMS_PER_TILE
) -> range:
    """The contiguous joint-index range a dispatch covers: exactly pairs*elems points."""
    return range(dispatch_start, dispatch_start + pairs * elems_per_tile)


@dataclass
class LegAccumulator:
    """Per-leg streaming digest + joint-coverage checksums, folded dispatch by dispatch.

    Unlike the laneMK single-operand accumulator (which hashes valid*4 output bytes and
    excludes padding), the two-operand sweep runs only whole dispatches, so the whole
    (cleared) result region is hashed each dispatch: any result tile the kernel does not
    write stays at the 0xA5 clear sentinel in BOTH legs and therefore cannot manufacture
    a spurious divergence, while the tiles it does write carry the only place the two
    kernels can differ.
    """

    sha: "hashlib._Hash" = field(default_factory=hashlib.sha256)
    joints: int = 0
    out_bytes: int = 0
    sum64: int = 0
    xor32: int = 0

    def update_input(
        self, dispatch_start: int, pairs: int, elems_per_tile: int = ELEMS_PER_TILE
    ) -> None:
        for j in covered_joints(dispatch_start, pairs, elems_per_tile):
            self.sum64 = (self.sum64 + j) & _U64
            self.xor32 ^= j
        self.joints += pairs * elems_per_tile

    def update_output(self, result_region_bytes: bytes) -> None:
        self.sha.update(result_region_bytes)
        self.out_bytes += len(result_region_bytes)

    def producer_line(
        self, tag: str, start: int, end_exclusive: int, dispatches: int
    ) -> str:
        return (
            f"BINARY_STREAM_PRODUCER_COVERAGE,{tag},"
            f"complete=true,start={start},joints={self.joints},"
            f"end_exclusive={end_exclusive},dispatches={dispatches},"
            f"observed_result_bytes={self.out_bytes},"
            f"joint_sum_mod_2_64={self.sum64},joint_xor={self.xor32},"
            f"output_sha256={self.sha.hexdigest()}"
        )


def band_verdict(sem_sha: str, hand_sha: str) -> str:
    return "BIT-EXACT-ALL-INPUTS" if sem_sha == hand_sha else "DIVERGENT"
