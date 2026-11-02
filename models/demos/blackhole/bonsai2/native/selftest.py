# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only checks with no device and no ttnn: run on any machine with numpy.

These are the parts of the harness that must be right before a device run is
worth doing, so they run fast and often:

  hadamard   H orthogonal, fwht matches the dense product, H/32 tiles exact
  pack       BFP2 tile packing round-trips, exponent section correct
  layout     the streaming DRAM layout inverts back to the input codes

``check_against_ttnn`` (the byte-exact comparison with tt-metal's C++ packer)
needs a built tt-metal, so it is opt-in via ``--ttnn``.

Usage::

    python -m models.demos.blackhole.bonsai2.native.selftest            # numpy only
    python -m models.demos.blackhole.bonsai2.native.selftest --ttnn     # + C++ packer
"""

from __future__ import annotations

import argparse

import numpy as np


def check_hadamard() -> list[str]:
    """``H`` orthogonal, ``fwht`` matches it, and the rotated basis is exactly inverted."""
    from .hadamard import BlockHadamard, fwht, sylvester

    fails = []
    h = sylvester(1024)
    if not np.allclose(h @ h.T, np.eye(1024), atol=1e-12):
        fails.append("sylvester(1024) is not orthogonal")
    x = np.random.default_rng(0).standard_normal((3, 1024))
    if not np.allclose(fwht(x) / np.sqrt(1024), x @ h, atol=1e-9):
        fails.append("fwht does not match the dense product")

    # H D is orthogonal but *not* symmetric or an involution, so the property to
    # check is orthogonality, plus the pair identity that actually matters:
    # folding the rotation into the weights must reproduce rotating the input.
    signs = {5120: np.where(np.arange(5120) % 3 == 0, -1.0, 1.0).astype(np.float32)}
    had = BlockHadamard(1024, signs)
    xs = np.random.default_rng(1).standard_normal((2, 5120))
    w = np.random.default_rng(2).standard_normal((16, 5120))
    if not np.allclose(had.unrotate(w) @ xs.T, w @ had.rotate(xs).T, atol=1e-9):
        fails.append("unrotate(w) @ x != w @ rotate(x), so the folded weight is not equivalent")
    # Rotate must preserve norms (that is what orthogonality buys).
    if not np.allclose(np.linalg.norm(had.rotate(xs), axis=1), np.linalg.norm(xs, axis=1), atol=1e-9):
        fails.append("rotate does not preserve the row norms, so it is not orthogonal")

    from .bfp2 import codes_to_trits, unpack_tile_codes

    tiles = had.h_tiles_bfp2()
    trits = codes_to_trits(unpack_tile_codes(tiles[0, 0]))
    if not np.all(np.abs(trits) == 1):
        fails.append("H/32 tile has a zero entry, which a Hadamard matrix must not")
    # And it round-trips back to the same sign pattern.
    if not np.array_equal(trits, np.sign(h[:32, :32]).astype(np.int8)):
        fails.append("H tile signs do not match the dense H block")
    return fails


def check_reader(path: str, names: list[str] | None = None, sample: int = 0) -> list[str]:
    """Reader self-consistency: shapes, group divisibility, trit range, dequant.

    ``sample`` limits the check to that many tensors spread across the file
    (first, last and evenly between) when the full sweep would take minutes.
    """
    from . import gguf_ternary as gt

    fails = []
    gguf = gt.read_gguf(path)
    have = sorted(n for n, t in gguf.tensors.items() if t.ggml_type in gt.TERNARY_TYPES)
    if names:
        want = set(names)
        have = [n for n in have if n in want]
    if not have:
        return [f"{path}: no ternary tensors found"]
    if sample and len(have) > sample:
        idx = np.unique(np.linspace(0, len(have) - 1, sample).astype(int))
        have = [have[i] for i in idx]
    for name in have:
        t = gt.read_ternary(gguf, name)
        n, k = t.codes.shape
        if k % t.group:
            fails.append(f"{name}: K={k} is not a multiple of G={t.group}")
        if t.scales.shape != (n, k // t.group):
            fails.append(f"{name}: scale shape {t.scales.shape} != {(n, k // t.group)}")
        lo, hi = int(t.codes.min()), int(t.codes.max())
        if lo < -1 or hi > 1:
            fails.append(f"{name}: codes span [{lo}, {hi}], outside {{-1,0,+1}}")
        # Dequant + finiteness on one row-block, which is where a bad scale shows
        # up; materializing every 17408x5120 tensor would cost gigabytes.
        rows = t.codes[:64].astype(np.float32) * t.expanded_scales()[:64]
        if not np.all(np.isfinite(rows)):
            fails.append(f"{name}: dequantized weight has non-finite entries")
    return fails


def check_pack() -> list[str]:
    """BFP2 pack/unpack round trip over the sparse and dense extremes."""
    from .bfp2 import (
        BFP2_SHARED_EXP_TILE_BYTES,
        BFP2_TILE_BYTES,
        codes_to_trits,
        pack_tiles,
        trits_to_codes,
        unpack_tile_codes,
    )

    fails = []
    rng = np.random.default_rng(7)
    cases = {
        "random": rng.integers(-1, 2, size=(16, 32, 32)).astype(np.int8),
        "all_zero": np.zeros((2, 32, 32), dtype=np.int8),
        "all_one": np.ones((2, 32, 32), dtype=np.int8),
        "alternating": ((np.indices((2, 32, 32)).sum(axis=0) % 2) * 2 - 1).astype(np.int8),
    }
    for label, trits in cases.items():
        codes = trits_to_codes(trits)
        packed = pack_tiles(codes)
        want_shape = (trits.shape[0], BFP2_TILE_BYTES)
        if packed.shape != want_shape:
            fails.append(f"{label}: packed shape {packed.shape} != {want_shape}")
            continue
        for i in range(trits.shape[0]):
            back = codes_to_trits(unpack_tile_codes(packed[i]))
            if not np.array_equal(back, trits[i]):
                fails.append(f"{label}[{i}]: round trip mismatch at {int((back != trits[i]).sum())} positions")
        # Exponent section: nonzero exactly where the 16-element face row is nonzero.
        exps = packed[:, :64].reshape(trits.shape[0], 2, 2, 16)  # tr, tc, row
        want = trits.reshape(trits.shape[0], 2, 16, 2, 16).any(axis=(2, 4))  # [n, tr, tc]
        got = (exps > 0).any(axis=3)  # [n, tr, tc]
        if not np.array_equal(got, want):
            fails.append(f"{label}: exponent section does not flag the nonzero face rows")

    trits = rng.integers(-1, 2, size=(4, 32, 32)).astype(np.int8)
    codes = trits_to_codes(trits)
    full = pack_tiles(codes, shared_exp=False)
    shed = pack_tiles(codes, shared_exp=True)
    if shed.shape != (4, BFP2_SHARED_EXP_TILE_BYTES):
        fails.append(f"shared_exp: shape {shed.shape} != {(4, BFP2_SHARED_EXP_TILE_BYTES)}")
    elif not np.array_equal(shed, full[:, 64:]):
        fails.append("shared_exp tile is not the data section of the standard tile")
    for i in range(4):
        if not np.array_equal(codes_to_trits(unpack_tile_codes(shed[i], shared_exp=True)), trits[i]):
            fails.append(f"shared_exp[{i}]: round trip mismatch")

    # 0b10 must be rejected, never silently mapped.
    bad = np.zeros(320, dtype=np.uint8)
    bad[64] = 0b10  # first data byte, lowest 2 bits
    try:
        codes_to_trits(unpack_tile_codes(bad))
        fails.append("codes_to_trits accepted the 0b10 datum, which decodes as -Inf")
    except ValueError:
        pass
    return fails


def check_layout() -> list[str]:
    """The streaming layout is a permutation a matmul can undo, with zero padding."""
    from . import pack
    from .bfp2 import TILE

    fails = []
    rng = np.random.default_rng(11)
    for n_banks, n, k in ((8, 96, 128), (8, 256, 256), (10, 1024, 512)):
        trits = rng.integers(-1, 2, size=(n, k)).astype(np.int8)
        n_pad = pack.padded_n(n, n_banks)
        tiles = pack.tile_column_major(trits, n_pad, n_banks)
        per_bank_tiles = (n_pad // n_banks) // 32
        kt = k // 32
        # Undo the walk: tile (bank, j) holds K-tile j % kt of column
        # bank*per_bank + j // kt, as one full 32x32 block.
        back = np.zeros((n_pad, k), dtype=np.int8)
        for b in range(n_banks):
            for j in range(per_bank_tiles * kt):
                ct, t = divmod(j, kt)  # tile-column inside the bank, K-tile
                col0 = (b * per_bank_tiles + ct) * TILE
                back[col0 : col0 + TILE, t * TILE : (t + 1) * TILE] = tiles[b, j].T
        if not np.array_equal(back[:n], trits):
            fails.append(f"layout n={n} k={k} banks={n_banks}: walk does not invert to the input codes")
        if np.any(back[n:] != 0):
            fails.append(f"layout n={n} k={k}: padding rows are not zero")

        flat, n_pad2 = pack.pack_for_streaming(trits, n_banks)
        # One 320 B tile per (output tile-column, K-tile) pair.
        want = (n_pad2 // 32) * (k // 32) * 320
        if flat.size != want:
            fails.append(f"pack_for_streaming n={n} k={k}: {flat.size} bytes, expected {want}")
    return fails


def check_ttnn_pack() -> list[str]:
    """The byte-exact check against tt-metal's C++ BFP2 packer."""
    from .bfp2 import check_against_ttnn

    try:
        check_against_ttnn(seed=0)
        check_against_ttnn(seed=5)
    except AssertionError as e:
        return [str(e)]
    except ImportError as e:
        return [f"ttnn not importable: {e}"]
    return []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", default="")
    ap.add_argument("--ttnn", action="store_true", help="also compare against the C++ packer")
    ap.add_argument("--reader-sample", type=int, default=12, help="how many tensors the reader sweep touches")
    args = ap.parse_args()

    checks = [("hadamard", check_hadamard), ("pack", check_pack), ("layout", check_layout)]
    if args.gguf:
        checks.insert(0, ("reader", lambda: check_reader(args.gguf, sample=args.reader_sample)))
    if args.ttnn:
        checks.append(("ttnn_pack", check_ttnn_pack))

    bad = 0
    for name, fn in checks:
        print(f"{name} ...", flush=True)
        fails = fn()
        print(f"{name:12s} {'ok' if not fails else str(len(fails)) + ' FAIL'}", flush=True)
        for f in fails[:8]:
            print(f"    {f}", flush=True)
        bad += len(fails)
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
