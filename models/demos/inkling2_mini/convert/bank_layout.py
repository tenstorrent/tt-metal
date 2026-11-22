# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0

"""Bank layout for the bank-streamed MoE expert step (tt/fused/moe_bank.py), host only.

Like the stick layout (convert/stick.py) this is a reorder of whole tiles of a TILE expert tensor [1, E, K, N] into a
[1, 1, P*32, 32] TILE tensor that sits DRAM interleaved: page p is in bank p % n_banks at slot p // n_banks, so the
pages of one bank with consecutive slots are one contiguous byte run. Unlike sticks, every output column always lives
in the same bank, so a reader core next to that bank reads only it (the gpt-oss fused decode streams,
models/demos/gpt_oss/tt/fused_decode/stream.py, read at ~96 % of DRAM peak this way).

gate_up [E, H, 2I] (gate columns j < I/32, up columns I/32 + j): pair j = (gate col j, up col j) stays in one bank.
  With P = I/32 pairs and B banks, pair j < B*(P//B) is in bank j % B, slot k = j // B ("base" pairs, every
  expert). The P % B leftover pairs ("extra", here pair 35 of 36 over 7 banks) go to bank extra_bank(e, x), which
  rotates over the banks per expert so every bank holds about the same bytes. Per bank, expert-major: expert e's
  pairs in slot order, each pair = gate column (Kt tiles) then up column (Kt tiles).
down [E, I, H]: output column n is in bank n % B, column slot n // B; per bank, expert-major with
  ceil(N/32 / B) column slots per expert (banks with fewer columns leave their last slot empty).
"""

from __future__ import annotations

import numpy as np

N_BANKS = 7


def gu_pair_counts(P: int, n_banks: int = N_BANKS) -> tuple[int, int]:
    """(base pairs per bank, extra pairs per expert)."""
    return P // n_banks, P % n_banks


def extra_bank(e: int, x: int, n_banks: int = N_BANKS) -> int:
    """Bank of extra pair x of expert e. The extras skip banks 0 .. n_extra_dn-1, which carry one more down column
    than the rest when N/32 % n_banks != 0 (72 columns over 7 banks: banks 0, 1 have 11, the others 10)."""
    lo = 2  # shortcut: tuned to 72 down columns over 7 banks; upgrade with the shapes if another model uses this
    return lo + (e * 1 + x) % (n_banks - lo)


def gu_bank_units(E: int, P: int, n_banks: int = N_BANKS) -> list[list[tuple[int, int]]]:
    """Per bank: the (expert, pair) units in stored order."""
    base, n_extra = gu_pair_counts(P, n_banks)
    out = [[] for _ in range(n_banks)]
    for e in range(E):
        for b in range(n_banks):
            out[b] += [(e, b + n_banks * k) for k in range(base)]
            out[b] += [(e, base * n_banks + x) for x in range(n_extra) if extra_bank(e, x, n_banks) == b]
    return out


def gu_slot(e: int, b: int, k: int, P: int, n_banks: int = N_BANKS) -> int:
    """First pair slot of unit k (k < base: base pair b + n_banks k; k == base: the extra) of expert e in bank b.
    Same formula as moe_bank_reader.cpp (one extra pair per expert)."""
    base, n_extra = gu_pair_counts(P, n_banks)
    assert n_extra == 1
    lo = 2
    before = base * e + (((e + (n_banks - lo) - 1 - (b - lo)) // (n_banks - lo)) if b >= lo else 0)
    return before + k


def bank_pages_gu(E: int, kt: int, nt: int, n_banks: int = N_BANKS) -> np.ndarray:
    """Stored page -> source tile index in (e, k, n) order (-1: zero padding), gate_up [E, kt*32, nt*32]."""
    P = nt // 2
    units = gu_bank_units(E, P, n_banks)
    per_bank = []
    for b in range(n_banks):
        tiles = []
        for e, j in units[b]:
            for n in (j, P + j):
                tiles.append(e * kt * nt + np.arange(kt) * nt + n)
        per_bank.append(np.concatenate(tiles))
    return _interleave(per_bank, n_banks)


def dn_slots(nt: int, n_banks: int = N_BANKS) -> int:
    return -(-nt // n_banks)


def bank_pages_dn(E: int, kt: int, nt: int, n_banks: int = N_BANKS) -> np.ndarray:
    """Stored page -> source tile index, down [E, kt*32, nt*32]: column n in bank n % n_banks, slot n // n_banks."""
    S = dn_slots(nt, n_banks)
    per_bank = []
    for b in range(n_banks):
        tiles = np.full((E, S, kt), -1)
        for s in range(S):
            n = b + n_banks * s
            if n < nt:
                tiles[:, s] = np.arange(E)[:, None] * kt * nt + np.arange(kt)[None, :] * nt + n
        per_bank.append(tiles.reshape(-1))
    return _interleave(per_bank, n_banks)


def _interleave(per_bank: list[np.ndarray], n_banks: int) -> np.ndarray:
    """Bank b's tile list -> pages p = slot * n_banks + b (shorter banks padded with -1)."""
    m = max(len(t) for t in per_bank)
    out = np.full((m, n_banks), -1)
    for b, t in enumerate(per_bank):
        out[: len(t), b] = t
    return out.reshape(-1)


def bank_pages(name: str, shape) -> np.ndarray:
    _, E, K, N = shape
    f = bank_pages_gu if name.endswith("w_gate_up") else bank_pages_dn
    return f(E, K // 32, N // 32)


def write_bank(src, dst, entry: dict) -> dict:
    """Write the bank layout of the TILE expert tensor file ``src`` (manifest ``entry``) to ``dst`` atomically, check
    it, and return the manifest entry for ``dst`` (layout "BANK", tile_shape = the original [1, E, K, N])."""
    import os
    from pathlib import Path

    from models.demos.inkling2_mini.convert.stick import TILE_BYTES, _read, header

    if entry["layout"] != "TILE" or entry["dtype"] not in TILE_BYTES:
        raise ValueError(f"{entry['name']} L{entry['layer']}: {entry['dtype']} {entry['layout']} has no bank layout")
    pages = bank_pages(entry["name"], entry["shape"])
    _, tiles = _read(src, entry["dtype"])
    out = np.zeros((len(pages), tiles.shape[1]), np.uint8)
    out[pages >= 0] = tiles[pages[pages >= 0]]
    dst = Path(dst)
    tmp = dst.with_name(dst.name.removesuffix(".tensorbin") + ".tmp.tensorbin")
    with open(tmp, "wb") as f:
        f.write(header(len(pages) * 32, entry["dtype"]))
        f.write(out.data)
        f.flush()
        os.fsync(f.fileno())
    del out, tiles
    new = dict(entry, layout="BANK", shape=[1, 1, len(pages) * 32, 32], tile_shape=entry["shape"], bank_banks=N_BANKS)
    check_bank(src, tmp, new)
    os.replace(tmp, dst)
    return new


def check_bank(tile_file, bank_file, entry: dict):
    """The bank file holds exactly the tiles of ``tile_file`` (padding pages zero) and ttnn loads it as stored."""
    import ttnn
    from models.demos.inkling2_mini.convert.stick import _TTNN, _read

    pages = bank_pages(entry["name"], entry["tile_shape"])
    _, orig = _read(tile_file, entry["dtype"])
    _, got = _read(bank_file, entry["dtype"])
    if len(got) != len(pages) or got[pages < 0].any() or not np.array_equal(got[pages >= 0], orig[pages[pages >= 0]]):
        raise ValueError(f"{bank_file}: tiles differ from {tile_file}")
    del orig, got
    t = ttnn.load_tensor(bank_file)
    if list(t.shape) != entry["shape"] or t.dtype != _TTNN[entry["dtype"]]:
        raise ValueError(f"{bank_file}: ttnn loads {list(t.shape)} {t.dtype}")
