# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0

"""Stick layout for the fused MoE expert weights (tt/fused/moe_expert.py), host only.

A TILE expert tensor [1, E, K, N] stores its tiles in (e, k, n) order. The stick layout stores the SAME tiles
(byte for byte) as a [1, 1, P*32, 32] TILE tensor whose page order puts each "stick" (one output column n of one
expert e, its Kt tiles in K order) contiguous in one DRAM bank: stick q = e*Nt + n lies in bank q % n_banks, and
page p = n_banks*(m*Kt + i) + b holds tile i of stick q = m*n_banks + b. Interleaved DRAM puts page p in bank
p % n_banks, so a core reads a stick in a few long bursts. Up to n_banks-1 all-zero padding sticks end the tensor.

Tiles move whole, so the re-layout is a reorder of tile records in the ``ttnn.dump_tensor`` file; the header is a
stored ttnn header of the same dtype with its row count and data size patched. Host only: no device is opened.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

import ttnn

N_BANKS = 7  # live DRAM banks of a Blackhole p100a (dram_grid_size().x)
TILE_BYTES = {"BFP4": 576, "BFP8": 1088}  # 64 B exponents + 32x32 mantissas
_TTNN = {"BFP4": ttnn.bfloat4_b, "BFP8": ttnn.bfloat8_b}


def stick_pages(E: int, kt: int, nt: int, n_banks: int = N_BANKS) -> np.ndarray:
    """Stored page p -> index of the source tile in the (e, k, n) TILE order, or -1 for a zero padding tile."""
    q = np.arange(E * nt)
    src = (q // nt)[:, None] * (kt * nt) + np.arange(kt)[None, :] * nt + (q % nt)[:, None]  # [stick, i]
    pad = (-len(q)) % n_banks
    src = np.concatenate([src, np.full((pad, kt), -1)])
    return src.reshape(-1, n_banks, kt).transpose(0, 2, 1).reshape(-1)


def stored_shape(shape, n_banks: int = N_BANKS) -> list[int]:
    _, E, K, N = shape
    return [1, 1, len(stick_pages(E, K // 32, N // 32, n_banks)) * 32, 32]


def _read(path: Path, dtype: str) -> tuple[bytes, np.ndarray]:
    """(header bytes, tile records [T, tile_bytes] uint8) of a dumped TILE tensor."""
    raw = Path(path).read_bytes()
    off = 8 + int.from_bytes(raw[:8], "little")
    tb = TILE_BYTES[dtype]
    if (len(raw) - off) % tb:
        raise ValueError(f"{path}: {len(raw) - off} data bytes is not a whole number of {dtype} tiles")
    return raw[:off], np.frombuffer(raw, np.uint8, offset=off).reshape(-1, tb)


def _patch(buf: bytearray, old: int, new: int, width: int, what: str):
    o, n = old.to_bytes(width, "little"), new.to_bytes(width, "little")
    at = [i for i in range(len(buf) - width + 1) if buf[i : i + width] == o]
    if len(at) != 1:
        raise RuntimeError(f"tensor header: {what} {old} found {len(at)} times, expected once")
    buf[at[0] : at[0] + width] = n


# ``ttnn.dump_tensor`` header of a [1, 1, R0, 32] TILE host tensor of zeros, per dtype. Making it with
# ttnn.from_torch needs the card (BFP conversion opens the device driver), so it is stored here; test_cache.py
# checks it against a fresh dump and check_sticks() loads every written file back through ttnn.
R0 = 32 * 37  # distinctive row count so the patched fields are unambiguous
_HEADER_R0 = {
    "BFP4": bytes.fromhex(
        "b80100000000000014000000000000000c001600040008000c0010000c000000dc0000006c0000007c00000010000000"
        "00000a001000040008000c000a00000040000000240000000400000001000000040000007affffff0400000002000000"
        "0000000000000000010000000c000000080008000000040008000000ffffffffa6ffffff040000000100000001000000"
        "b6ffffff04000000020000000100000001000000010000001000000000000a001000070008000c000a00000000000001"
        "080000001c00000000000000000000004053000000000000000006000800040006000000040000000200000000000000"
        "0000000008000e000400080008000000740000001400000000000e0014000600070008000c0010000e00000000000402"
        "1c0000000c000000400000000400060004000000000006000a000400060000001400000000000e001400040008000000"
        "0c0010000e00000020000000200000001000000010000000020000002000000020000000040000000100000001000000"
        "a00400002000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
        "00000000000000000000000000000000"
    ),
    "BFP8": bytes.fromhex(
        "b80100000000000014000000000000000c001600040008000c0010000c000000dc0000006c0000007c00000010000000"
        "00000a001000040008000c000a00000040000000240000000400000001000000040000007affffff0400000002000000"
        "0000000000000000010000000c000000080008000000040008000000ffffffffa6ffffff040000000100000001000000"
        "b6ffffff04000000020000000100000001000000010000001000000000000a001000070008000c000a00000000000001"
        "080000001c0000000000000000000000409d000000000000000006000800040006000000040000000200000000000000"
        "0000000008000e000400080008000000740000001400000000000e0014000600070008000c0010000e00000000000302"
        "1c0000000c000000400000000400060004000000000006000a000400060000001400000000000e001400040008000000"
        "0c0010000e00000020000000200000001000000010000000020000002000000020000000040000000100000001000000"
        "a00400002000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
        "00000000000000000000000000000000"
    ),
}


def header(rows: int, dtype: str) -> bytes:
    """ttnn file header of a [1, 1, rows, 32] TILE host tensor in ``dtype``."""
    h = bytearray(_HEADER_R0[dtype])
    tb = TILE_BYTES[dtype]
    _patch(h, R0 // 32 * tb, rows // 32 * tb, 8, "data size")
    _patch(h, R0, rows, 4, "row count")
    return bytes(h)


def write_sticks(src: Path, dst: Path, entry: dict, n_banks: int = N_BANKS) -> dict:
    """Write the stick layout of the TILE expert tensor file ``src`` (manifest ``entry``) to ``dst`` (atomically,
    so src == dst is fine), check it, and return the manifest entry for ``dst``."""
    if entry["layout"] != "TILE" or entry["dtype"] not in TILE_BYTES:
        raise ValueError(f"{entry['name']} L{entry['layer']}: {entry['dtype']} {entry['layout']} has no stick layout")
    _, E, K, N = entry["shape"]
    pages = stick_pages(E, K // 32, N // 32, n_banks)
    _, tiles = _read(src, entry["dtype"])
    if len(tiles) != E * (K // 32) * (N // 32):
        raise ValueError(f"{src}: {len(tiles)} tiles != shape {entry['shape']}")
    out = np.zeros((len(pages), tiles.shape[1]), np.uint8)
    out[pages >= 0] = tiles[pages[pages >= 0]]
    dst = Path(dst)
    tmp = dst.with_name(dst.name.removesuffix(".tensorbin") + ".tmp.tensorbin")  # ttnn wants the suffix
    with open(tmp, "wb") as f:
        f.write(header(len(pages) * 32, entry["dtype"]))
        f.write(out.data)
        f.flush()
        os.fsync(f.fileno())  # the manifest that names this file is written later; survive a crash in between
    del out, tiles  # keep peak RSS to ~3 copies of one tensor during the check
    check_sticks(src, tmp, entry, n_banks)
    os.replace(tmp, dst)
    new = dict(entry, layout="STICK", shape=stored_shape(entry["shape"], n_banks))
    new.update(tile_shape=entry["shape"], stick_banks=n_banks)
    return new


def unstick(path: Path, entry: dict) -> np.ndarray:
    """Tile records [T, tile_bytes] of a stick file back in the (e, k, n) TILE order of ``entry['tile_shape']``
    (or ``entry['shape']`` for a TILE entry); checks the padding tiles are zero."""
    _, E, K, N = entry.get("tile_shape", entry["shape"])
    pages = stick_pages(E, K // 32, N // 32, entry.get("stick_banks", N_BANKS))
    _, tiles = _read(path, entry["dtype"])
    if len(tiles) != len(pages):
        raise ValueError(f"{path}: {len(tiles)} pages != {len(pages)}")
    if tiles[pages < 0].any():
        raise ValueError(f"{path}: nonzero padding tile")
    back = np.empty((E * (K // 32) * (N // 32), tiles.shape[1]), np.uint8)
    back[pages[pages >= 0]] = tiles[pages >= 0]
    return back


def check_sticks(tile_file: Path, stick_file: Path, entry: dict, n_banks: int = N_BANKS):
    """Un-sticking ``stick_file`` gives back the tile bytes of ``tile_file`` exactly, and ttnn loads it with the
    stored shape and dtype."""
    _, orig = _read(tile_file, entry["dtype"])
    back = unstick(stick_file, dict(entry, tile_shape=entry["shape"], stick_banks=n_banks))
    if not np.array_equal(orig, back):
        raise ValueError(f"{stick_file}: un-stuck tiles differ from {tile_file}")
    del orig, back
    t = ttnn.load_tensor(Path(stick_file))
    if list(t.shape) != stored_shape(entry["shape"], n_banks) or t.dtype != _TTNN[entry["dtype"]]:
        raise ValueError(f"{stick_file}: ttnn loads {list(t.shape)} {t.dtype}")
