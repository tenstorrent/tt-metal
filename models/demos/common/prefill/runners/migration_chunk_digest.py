# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Byte-exact P->D check of a KV migration, one host at a time.

digest: hash every chunk of the chosen configs for positions [0, P) of one slot, on this host's chips, addressed
through a published KV chunk table and this host's device map (``mesh chip unique_id`` lines), read with
``read_dram_umd``. Keys are (config name, table row, position), so a prefill table and a decode table with
different layouts still name the same logical chunk the same way.

compare: merge every host's digest per side and compare the two sides key by key. It fails on any byte difference,
on any chunk that only one side hashed, and on a side whose merged digests hold fewer chunks than its table names
(a host left out, which the key comparison cannot see when both sides lose the same chunks), on two sides
digested with different arguments, and on positions below P that digest left unchecked. Two files of one side
that hash the same chunk differently, or that were taken with different arguments, are an error.

Run digest after a P->D run with the KV Manager fleet down and before any board reset: device DRAM keeps its
contents after the processes exit. Decode writes slot 1 only at positions >= P after the migration, so [0, P)
still holds the migrated bytes. The KDA configs are left out: decode advances that state after the handoff.

A chunk that P ends inside also holds positions >= P, so only its bytes below P are hashed. Where those bytes
sit depends on how the cache is stored, which the table does not record, so ``--layouts`` names it per config:
``rows`` (row-major, one page per position) or ``bfp8`` (32x32 bfloat8_b tiles, one tile row per position).
For Kimi-K3.1 that is ``00=rows,01=bfp8``. A config with no layout has that chunk skipped, and digest prints how
many positions that leaves unchecked.

  python -m models.demos.common.prefill.runners.migration_chunk_digest digest --table T.pb --device-map /tmp/x.txt --slot S --positions P --configs 00,01 --layouts 00=rows,01=bfp8 --out F
  python -m models.demos.common.prefill.runners.migration_chunk_digest compare --src 'prefill_*.json' --dst 'decode_*.json'
"""

import argparse
import glob
import hashlib
import json
import socket
import sys
import time


def _device_map(path: str) -> dict:
    out = {}
    for line in open(path):
        parts = line.split()
        if len(parts) == 3:
            out[(int(parts[0]), int(parts[1]))] = int(parts[2])
    return out


def _node_key(node):
    """(mesh, chip) of a fabric node id, whatever the binding calls the fields."""
    for mesh_attr, chip_attr in (("mesh_id", "chip_id"), ("mesh", "chip")):
        if hasattr(node, mesh_attr) and hasattr(node, chip_attr):
            mesh = getattr(node, mesh_attr)
            mesh = int(mesh.get()) if hasattr(mesh, "get") else int(mesh)
            return mesh, int(getattr(node, chip_attr))
    if isinstance(node, (tuple, list)) and len(node) == 2:
        return int(node[0]), int(node[1])
    raise TypeError(f"cannot read (mesh, chip) from fabric node id {node!r}")


_TILE_ROWS = 32
_BFP8_TILE_BYTES = 1088  # 64 exponent bytes, then 1024 mantissa bytes
_FACE_ROWS = _FACE_COLS = 16


def _bfp8_tile_row_spans(row: int) -> list:
    """(offset, length) of the bytes one tile row occupies in a bfloat8_b tile.

    A tile is four 16x16 faces in the order top-left, top-right, bottom-left, bottom-right. It opens with one
    exponent byte per face row (64), followed by the faces' mantissas, each face row-major (pack_as_bfp_tiles).
    """
    spans = []
    for face_col in range(_TILE_ROWS // _FACE_COLS):
        face = (row // _FACE_ROWS) * 2 + face_col
        face_row = row % _FACE_ROWS
        spans.append((face * _FACE_ROWS + face_row, 1))
        spans.append((64 + face * _FACE_ROWS * _FACE_COLS + face_row * _FACE_COLS, _FACE_COLS))
    return spans


def _bytes_below(raw: bytes, layout: str, n_tokens: int, chunk_n_tokens: int) -> bytes:
    """The bytes of a chunk that hold its first ``n_tokens`` positions."""
    if layout == "rows":
        if len(raw) % chunk_n_tokens:
            raise SystemExit(f"rows layout: a {len(raw)}-byte chunk is not {chunk_n_tokens} equal pages")
        return raw[: len(raw) // chunk_n_tokens * n_tokens]
    if layout == "bfp8":
        if chunk_n_tokens != _TILE_ROWS or len(raw) % _BFP8_TILE_BYTES:
            raise SystemExit(
                f"bfp8 layout: a {len(raw)}-byte chunk of {chunk_n_tokens} positions is not whole "
                f"{_TILE_ROWS}-row tiles of {_BFP8_TILE_BYTES} bytes"
            )
        spans = sorted(span for row in range(n_tokens) for span in _bfp8_tile_row_spans(row))
        return b"".join(
            raw[tile + off : tile + off + length]
            for tile in range(0, len(raw), _BFP8_TILE_BYTES)
            for off, length in spans
        )
    raise SystemExit(f"unknown layout {layout!r}; use rows or bfp8")


def _parse_layouts(text: str) -> dict:
    layouts = dict(item.split("=", 1) for item in text.split(",") if item)
    for name, layout in layouts.items():
        if layout not in ("rows", "bfp8"):
            raise SystemExit(f"--layouts {name}={layout}: unknown layout; use rows or bfp8")
    return layouts


def digest(args) -> None:
    import ttnn

    dis = ttnn.experimental.disaggregation
    table = dis.import_from_protobuf_file(args.table)
    local = _device_map(args.device_map)
    names = {table.config_name(i): i for i in range(table.num_configs())}
    wanted = [names[n] for n in args.configs.split(",")]
    layouts = _parse_layouts(args.layouts)
    out, skipped_remote, unchecked, partial, t0 = {}, 0, {}, {}, time.monotonic()
    for cfg_id in wanted:
        cfg = table.config(cfg_id)
        name = table.config_name(cfg_id)
        step = int(cfg.chunk_n_tokens)
        # A chunk that P ends inside also holds positions >= P, which decode has written since the migration.
        # Only its bytes below P are compared, and only when the config's layout says where they are.
        whole = args.positions // step * step
        tail = args.positions - whole
        end = whole
        if tail and name in layouts:
            end, partial[name] = args.positions, tail
        elif tail:
            unchecked[name] = tail
        for row in range(int(cfg.num_layers)):
            if table.lookup(row, 0, args.slot, cfg_id).size_bytes == 0:
                continue
            for pos in range(0, end, step):
                loc = table.lookup(row, pos, args.slot, cfg_id)
                nodes = table.get_device_group(loc.device_group_index).fabric_node_ids
                uid = next((local[_node_key(n)] for n in nodes if _node_key(n) in local), None)
                if uid is None:
                    skipped_remote += 1
                    continue
                raw = bytes(dis.read_dram_umd(uid, loc.noc_addr, loc.size_bytes))
                if pos == whole:
                    raw = _bytes_below(raw, layouts[name], tail, step)
                out[f"{name}/{row}/{pos}"] = hashlib.sha1(raw).hexdigest()
    # Every host walks the whole table, so each one knows how many chunks its side holds across all hosts.
    scope = {
        "slot": args.slot,
        "positions": args.positions,
        "configs": sorted(args.configs.split(",")),
        "layouts": {n: layouts[n] for n in sorted(partial)},
        "unchecked_positions": unchecked,
    }
    json.dump(
        {"host": socket.gethostname(), "scope": scope, "expected_chunks": len(out) + skipped_remote, "chunks": out},
        open(args.out, "w"),
    )
    print(
        f"{socket.gethostname()}: {len(out)} chunk(s) hashed, {skipped_remote} on other hosts, "
        f"{time.monotonic() - t0:.1f} s -> {args.out}"
    )
    for name, n in partial.items():
        print(f"  config {name}: last chunk hashed for its first {n} position(s) only ({layouts[name]} layout)")
    for name, n in unchecked.items():
        print(
            f"  config {name}: --positions {args.positions} ends inside a chunk and --layouts does not name "
            f"this config; last {n} position(s) NOT checked"
        )


def compare(args) -> int:
    def load(pattern):
        """One side's merged chunks, and how many its table names for the digested scope."""
        merged, origin, stated = {}, {}, None
        for f in sorted(glob.glob(pattern)):
            doc = json.load(open(f))
            if "expected_chunks" not in doc or "scope" not in doc:
                raise SystemExit(f"{f}: no expected_chunks/scope; take the digest again with this version")
            this = (doc["scope"], int(doc["expected_chunks"]))
            if stated is not None and this != stated[0]:
                raise SystemExit(f"{stated[1]} and {f} differ in scope or table: {stated[0]} vs {this}")
            stated = stated or (this, f)
            for key, digest in doc["chunks"].items():
                if key in merged and merged[key] != digest:
                    raise SystemExit(f"{key}: {origin[key]} and {f} hashed different bytes for the same chunk")
                merged[key], origin[key] = digest, f
        return merged, (stated[0][1] if stated else 0), (stated[0][0] if stated else {})

    (src, src_expected, src_scope), (dst, dst_expected, dst_scope) = load(args.src), load(args.dst)
    both = sorted(set(src) & set(dst))
    bad = [k for k in both if src[k] != dst[k]]
    per_cfg = {}
    for k in both:
        cfg = k.split("/")[0]
        n, b = per_cfg.get(cfg, (0, 0))
        per_cfg[cfg] = (n + 1, b + (src[k] != dst[k]))
    print(f"src {len(src)} chunk(s), dst {len(dst)}, compared {len(both)}, mismatched {len(bad)}")
    for cfg, (n, b) in sorted(per_cfg.items()):
        print(f"  config {cfg}: {n - b}/{n} identical")
    only = sorted(set(src) ^ set(dst))
    if only:
        print(f"  {len(only)} key(s) on one side only, e.g. {only[:5]}")
    short = [
        (side, len(got), want)
        for side, got, want in (("src", src, src_expected), ("dst", dst, dst_expected))
        if len(got) != want
    ]
    for side, got, want in short:
        print(f"  {side}: {got} chunk(s) hashed but its table names {want}; a host's digest is missing")
    if src_expected != dst_expected:
        print(f"  the two tables name different chunk counts: src {src_expected}, dst {dst_expected}")
    if bad:
        layers = sorted({int(k.split("/")[1]) for k in bad})
        print(f"  mismatched layers: {layers[:30]}; first: {bad[:5]}")
    # The slot differs between the sides by design; what was hashed must not.
    hashed = [{k: v for k, v in scope.items() if k != "slot"} for scope in (src_scope, dst_scope)]
    if hashed[0] != hashed[1]:
        print(f"  the two sides were digested differently: src {hashed[0]}, dst {hashed[1]}")
    unchecked = {**src_scope.get("unchecked_positions", {}), **dst_scope.get("unchecked_positions", {})}
    for cfg, n in sorted(unchecked.items()):
        print(f"  config {cfg}: last {n} position(s) of each layer NOT checked; name its layout in digest --layouts")
    failed = bad or only or short or src_expected != dst_expected or hashed[0] != hashed[1] or unchecked or not both
    return 1 if failed else 0


def main() -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("digest")
    d.add_argument("--table", required=True)
    d.add_argument("--device-map", required=True)
    d.add_argument("--slot", type=int, required=True)
    d.add_argument("--positions", type=int, required=True)
    d.add_argument("--configs", default="00,01")
    d.add_argument("--layouts", default="", help="per config, e.g. 00=rows,01=bfp8; see the module docstring")
    d.add_argument("--out", required=True)
    c = sub.add_parser("compare")
    c.add_argument("--src", required=True)
    c.add_argument("--dst", required=True)
    args = p.parse_args()
    if args.cmd == "digest":
        digest(args)
        return 0
    return compare(args)


if __name__ == "__main__":
    sys.exit(main())
