# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0

"""PF layout for the bank-streamed MoE expert step (tt/fused/moe_bank.py pf_weights), host only.

Each expert tensor [1, E, K, N] becomes two tensors with the same tiles, byte for byte:
  main: [1, E, K, n_pf*32] TILE, column m = source column pf_columns(name)[m] (gate_up: the gate (m // 7 even) or up
    column of base pair m % 7 + 7 * (m // 14); down: column m). The loader places it ND-sharded one 32-column shard
    per bank in turn, so ring position m's column of every expert sits in bank m % 7, receiver-contiguous: the
    grouped-weight layout the tensor prefetcher streams by expert mask.
  extra ("<name>.x"): what is left, DRAM interleaved [1, 1, P*32, 32] (page p in bank p % 7): gate_up's extra pair 35
    as the bank layout stores it (bank 2 + e % 5, slot (e // 5) * 2 Kt, gate column then up column); down's columns
    70, 71 in banks 0, 1 at slot e * Kt (banks 2..6 hold zero pages there).

    python -m models.demos.inkling2_mini.convert.pf_layout --src cache/v1-bank --tiles cache/v1 --dst cache/v1-pf

Every file but the expert tensors is hard-linked from --src; the expert tiles are read from the TILE cache --tiles.
Each written tensor is checked against its source tiles and loaded back through ttnn.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from models.demos.inkling2_mini.convert.relayout_sticks import link_or_copy
from models.demos.inkling2_mini.convert.stick import _TTNN, N_BANKS, TILE_BYTES, _read, header

X_LO = 2  # first bank holding gate_up extra pairs (convert/bank_layout.extra_bank)

# ``ttnn.dump_tensor`` header of a [1, 37, 1184, 96] TILE host tensor of zeros (made on the mock device); the data
# size (8 bytes at 208) and dims 1..3 (4 bytes at 380, 384, 388) are patched.
_HDR = {
    "BFP4": bytes.fromhex(
        "b80100000000000014000000000000000c001600040008000c0010000c000000dc0000006c0000007c00000010000000"
        "00000a001000040008000c000a00000040000000240000000400000001000000040000007affffff0400000002000000"
        "0000000000000000010000000c000000080008000000040008000000ffffffffa6ffffff040000000100000001000000"
        "b6ffffff04000000020000000100000001000000010000001000000000000a001000070008000c000a00000000000001"
        "080000001c0000000000000000000000c018240000000000000006000800040006000000040000000200000000000000"
        "0000000008000e000400080008000000740000001400000000000e0014000600070008000c0010000e00000000000402"
        "1c0000000c000000400000000400060004000000000006000a000400060000001400000000000e001400040008000000"
        "0c0010000e00000020000000200000001000000010000000020000002000000020000000040000000100000025000000"
        "a00400006000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
        "00000000000000000000000000000000"
    ),
    "BFP8": bytes.fromhex(
        "b80100000000000014000000000000000c001600040008000c0010000c000000dc0000006c0000007c00000010000000"
        "00000a001000040008000c000a00000040000000240000000400000001000000040000007affffff0400000002000000"
        "0000000000000000010000000c000000080008000000040008000000ffffffffa6ffffff040000000100000001000000"
        "b6ffffff04000000020000000100000001000000010000001000000000000a001000070008000c000a00000000000001"
        "080000001c0000000000000000000000c02e440000000000000006000800040006000000040000000200000000000000"
        "0000000008000e000400080008000000740000001400000000000e0014000600070008000c0010000e00000000000302"
        "1c0000000c000000400000000400060004000000000006000a000400060000001400000000000e001400040008000000"
        "0c0010000e00000020000000200000001000000010000000020000002000000020000000040000000100000025000000"
        "a00400006000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
        "00000000000000000000000000000000"
    ),
}


def nd_header(shape, dtype: str) -> bytes:
    h = bytearray(_HDR[dtype])
    _, E, K, N = shape
    assert int.from_bytes(h[380:392], "little") == 37 | 1184 << 32 | 96 << 64, "header template"
    h[208:216] = (E * K * N // 1024 * TILE_BYTES[dtype]).to_bytes(8, "little")
    for off, v in ((380, E), (384, K), (388, N)):
        h[off : off + 4] = v.to_bytes(4, "little")
    return bytes(h)


def pf_columns(name: str, P: int, nt_dn: int, nb: int = N_BANKS) -> list[int]:
    """Source column tile of ring position m (same as tt/fused/moe_bank.py pf_columns)."""
    if name.endswith("w_gate_up"):
        base = P // nb
        out = []
        for m in range(nb * 2 * base):
            b, i = m % nb, m // nb
            j = b + nb * (i // 2)
            out.append(j if i % 2 == 0 else P + j)
        return out
    return list(range(nb * (nt_dn // nb)))


def pf_pages(name: str, shape, nb: int = N_BANKS) -> tuple[np.ndarray, np.ndarray]:
    """(main, extra) stored page -> source tile index in (e, k, n) order (-1: zero padding)."""
    _, E, K, N = shape
    kt, nt = K // 32, N // 32
    gu = name.endswith("w_gate_up")
    P = nt // 2
    cols = np.array(pf_columns(name, P, nt, nb))
    ek = np.arange(E)[:, None] * kt + np.arange(kt)[None, :]  # [E, kt]
    main = (ek[:, :, None] * nt + cols[None, None, :]).reshape(-1)
    if gu:
        per = nb - X_LO
        slots = -(-E // per) * 2 * kt
        extra = np.full((slots, nb), -1)
        for e in range(E):
            b, s0 = X_LO + e % per, (e // per) * 2 * kt
            extra[s0 : s0 + kt, b] = ek[e] * nt + (P - 1)
            extra[s0 + kt : s0 + 2 * kt, b] = ek[e] * nt + (2 * P - 1)
    else:
        lo = len(cols)
        extra = np.full((E * kt, nb), -1)
        for c in range(nt - lo):
            extra[:, c] = ek.reshape(-1) * nt + lo + c
    return main, extra.reshape(-1)


def _write(path: Path, hdr: bytes, tiles: np.ndarray, pages: np.ndarray):
    out = np.zeros((len(pages), tiles.shape[1]), np.uint8)
    out[pages >= 0] = tiles[pages[pages >= 0]]
    tmp = path.with_name(path.name.removesuffix(".tensorbin") + ".tmp.tensorbin")
    with open(tmp, "wb") as f:
        f.write(hdr)
        f.write(out.data)
        f.flush()
        os.fsync(f.fileno())
    return tmp


def check(tile_file, path, pages, entry):
    import ttnn

    _, orig = _read(tile_file, entry["dtype"])
    _, got = _read(path, entry["dtype"])
    if len(got) != len(pages) or got[pages < 0].any() or not np.array_equal(got[pages >= 0], orig[pages[pages >= 0]]):
        raise ValueError(f"{path}: tiles differ from {tile_file}")
    t = ttnn.load_tensor(path)
    if list(t.shape) != entry["shape"] or t.dtype != _TTNN[entry["dtype"]]:
        raise ValueError(f"{path}: ttnn loads {list(t.shape)} {t.dtype}, want {entry['shape']}")


def write_pf(tile_file: Path, dst: Path, entry: dict) -> list[dict]:
    """Write the PF main and extra files of the TILE expert tensor ``tile_file`` (its v1 manifest ``entry``) next
    to ``dst``; returns their manifest entries."""
    shape = entry["shape"]
    main, extra = pf_pages(entry["name"], shape)
    _, E, K, N = shape
    _, tiles = _read(tile_file, entry["dtype"])
    e_main = dict(entry, layout="PF", shape=[1, E, K, len(main) // (E * K // 32) * 32], tile_shape=shape,
                  pf_banks=N_BANKS)  # fmt: skip
    xfile = entry["file"].removesuffix(".tensorbin") + ".x.tensorbin"
    e_x = dict(entry, name=entry["name"] + ".x", file=xfile, layout="PF_X", shape=[1, 1, len(extra) * 32, 32],
               tile_shape=shape, pf_banks=N_BANKS)  # fmt: skip
    t1 = _write(dst, nd_header(e_main["shape"], entry["dtype"]), tiles, main)
    xdst = dst.with_name(Path(xfile).name)
    t2 = _write(xdst, header(len(extra) * 32, entry["dtype"]), tiles, extra)
    del tiles
    check(tile_file, t1, main, e_main)
    check(tile_file, t2, extra, e_x)
    os.replace(t1, dst)
    os.replace(t2, xdst)
    return [e_main, e_x]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", type=Path, required=True, help="cache whose non-expert files are linked (e.g. v1-bank)")
    ap.add_argument("--tiles", type=Path, required=True, help="cache with the TILE expert tensors (e.g. v1)")
    ap.add_argument("--dst", type=Path, required=True)
    args = ap.parse_args(argv)
    if (args.dst / "manifest.json").exists():
        raise SystemExit(f"{args.dst} already has a manifest")
    m = json.loads((args.src / "manifest.json").read_text())
    tm = {(e["layer"], e["name"]): e for e in json.loads((args.tiles / "manifest.json").read_text())["tensors"]}
    t0, out = time.time(), []
    for e in m["tensors"]:
        dst = args.dst / e["file"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        if e["name"].startswith("experts."):
            te = tm[(e["layer"], e["name"])]
            assert te["layout"] == "TILE" and te["dtype"] == e["dtype"], te
            t1 = time.time()
            out += write_pf(args.tiles / te["file"], dst, te)
            print(f"L{e['layer']:02d} {e['name']} {e['dtype']}: {time.time() - t1:.1f}s", flush=True)
        else:
            if dst.exists():
                dst.unlink()
            link_or_copy(args.src / e["file"], dst)
            out.append(e)
    new = dict(m, tensors=out, pf_source=str(args.tiles), pf_banks=N_BANKS)
    new["expert_layout"] = {
        f"L{e['layer']:02d}/{e['name']}": e["layout"] for e in out if e["name"].startswith("experts.")
    }
    tmp = args.dst / "manifest.tmp.json"
    tmp.write_text(json.dumps(new, indent=1))
    os.replace(tmp, args.dst / "manifest.json")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
