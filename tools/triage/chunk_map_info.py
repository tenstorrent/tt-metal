#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Lightweight KvChunkAddressTable inspector — no protobuf / ttnn dependency.

Reads the proto wire format directly (varint/length-delimited decoding over
~60 lines), so it runs on ANY host with any Python 3, against a chunk-map .pb
written by ANY version of the serializer — unknown fields are reported, not
misparsed, which is exactly what you want when eyeballing forward/backward
compatibility across mixed-version deployments.

Usage:
    chunk_map_info.py TABLE.pb [--entries] [--runs] [--json]

Prints: format_version, origin host, per-config dims + compression tag,
slot/layer/position-chunk counts, payload form (entries vs runs), and device
groups with their hostnames. --entries/--runs dump individual records.
"""

import json
import struct
import sys

# --- minimal proto3 wire reader (varint / fixed64 / length-delimited only) ---


def _read_varint(buf, pos):
    result = 0
    shift = 0
    while True:
        if pos >= len(buf):
            raise ValueError(f"truncated varint at offset {pos}")
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not b & 0x80:
            return result, pos
        shift += 7


def _unzigzag(v):
    return (v >> 1) ^ -(v & 1)  # sint64


def _fields(buf):
    """Yield (field_number, wire_type, value) for every field in a message buffer."""
    pos = 0
    while pos < len(buf):
        tag, pos = _read_varint(buf, pos)
        num, wt = tag >> 3, tag & 7
        if wt == 0:
            val, pos = _read_varint(buf, pos)
        elif wt == 1:
            val = struct.unpack_from("<Q", buf, pos)[0]
            pos += 8
        elif wt == 2:
            n, pos = _read_varint(buf, pos)
            if pos + n > len(buf):
                raise ValueError(f"truncated length-delimited field at offset {pos} (want {n} B, {len(buf) - pos} left)")
            val = buf[pos : pos + n]
            pos += n
        elif wt == 5:
            if pos + 4 > len(buf):
                raise ValueError(f"truncated fixed32 at offset {pos}")
            val = struct.unpack_from("<I", buf, pos)[0]
            pos += 4
        else:
            raise ValueError(f"unsupported wire type {wt} for field {num}")
        yield num, wt, val


def _s(b):
    return b.decode("utf-8", errors="replace")


# --- schema (field numbers from kv_chunk_address_table.proto) ---

CONFIG_NAMES = {
    1: "name",
    2: "num_layers",
    3: "max_sequence_length",
    4: "num_slots",
    5: "chunk_n_tokens",
    6: "chunk_size_bytes",
    7: "compression",
}
COMPRESSION = {0: "UNROLLED", 1: "STRIDED_ROWS"}


def _parse_config(buf):
    cfg = {}
    for num, _wt, val in _fields(buf):
        if num == 1:
            cfg["name"] = _s(val)
        elif num in CONFIG_NAMES:
            cfg[CONFIG_NAMES[num]] = val
    return cfg


def _parse_group(buf):
    nodes = []
    for num, _wt, val in _fields(buf):
        if num == 1:  # repeated FabricNodeId
            mesh, chip = 0, 0
            for fnum, _wt, fval in _fields(val):
                if fnum == 1:
                    mesh = fval
                elif fnum == 2:
                    chip = fval
            nodes.append((mesh, chip))
    return nodes


def _parse_host(buf):
    mesh = chip = 0
    host = ""
    for num, _wt, val in _fields(buf):
        if num == 1:
            mesh = val
        elif num == 2:
            chip = val
        elif num == 3:
            host = _s(val)
    return (mesh, chip), host


def inspect(path, show_entries=False, show_runs=False):
    data = open(path, "rb").read()
    out = {
        "file": path,
        "bytes": len(data),
        "format_version": 0,  # absent == legacy v1
        "origin_host": None,
        "legacy_scalars": {},
        "configs": [],
        "device_groups": [],
        "hosts": {},
        "entries": 0,
        "runs": 0,
        "unknown_fields": {},
    }
    entries, runs = [], []
    for num, _wt, val in _fields(data):
        if num in (1, 2, 3, 4, 5):  # legacy single-config scalars
            out["legacy_scalars"][
                {1: "num_layers", 2: "max_sequence_length", 3: "num_slots", 4: "chunk_n_tokens", 5: "chunk_size_bytes"}[
                    num
                ]
            ] = val
        elif num == 6:
            out["device_groups"].append(_parse_group(val))
        elif num == 7:
            node, host = _parse_host(val)
            out["hosts"][f"{node[0]}.{node[1]}"] = host
        elif num == 8:
            out["entries"] += 1
            if show_entries:
                entries.append({f: v for f, _wt2, v in _fields(val)})
        elif num == 9:
            out["configs"].append(_parse_config(val))
        elif num == 11:
            out["runs"] += 1
            if show_runs:
                rec = {f: v for f, _wt2, v in _fields(val)}
                if 8 in rec:  # addr_stride is sint64 (zigzag on the wire)
                    rec[8] = _unzigzag(rec[8])
                runs.append(rec)
        elif num == 12:
            out["format_version"] = val
        elif num == 13:
            out["origin_host"] = _s(val)
        else:
            out["unknown_fields"][num] = out["unknown_fields"].get(num, 0) + 1
    if show_entries:
        out["entry_records"] = entries
    if show_runs:
        out["run_records"] = runs
    return out


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    if len(args) != 1:
        sys.exit(__doc__)
    info = inspect(args[0], show_entries="--entries" in flags, show_runs="--runs" in flags)
    if "--json" in flags:
        print(json.dumps(info, indent=2))
        return

    print(f"file: {info['file']}  ({info['bytes']} bytes)")
    print(
        f"format_version: {info['format_version']}"
        + ("  (absent: legacy pre-tag format)" if info["format_version"] == 0 else "")
    )
    print(f"origin_host:    {info['origin_host'] or '(not recorded)'}")
    if info["unknown_fields"]:
        print(f"unknown fields: {info['unknown_fields']}  (written by a NEWER format; ignored here)")
    print(f"payload: {info['entries']} unrolled entries, {info['runs']} strided runs")
    print(f"device groups: {len(info['device_groups'])}")
    for i, nodes in enumerate(info["device_groups"]):
        hosts = [info["hosts"].get(f"{m}.{c}", "?") for m, c in nodes]
        print(f"  group {i}: {len(nodes)} nodes {nodes} on {sorted(set(hosts))}")
    for i, c in enumerate(info["configs"]):
        comp = COMPRESSION.get(c.get("compression", 0), f"?{c.get('compression')}")
        npc = -(-c.get("max_sequence_length", 0) // max(c.get("chunk_n_tokens", 1), 1))
        print(
            f"  config {i} '{c.get('name', '?')}': {c.get('num_slots')} slots x {c.get('num_layers')} layers "
            f"x {npc} position chunks ({c.get('chunk_n_tokens')} tok/chunk, {c.get('chunk_size_bytes')} B/chunk) "
            f"[{comp}]"
        )
    for rec in info.get("entry_records", []):
        print(f"  entry {rec}")
    for rec in info.get("run_records", []):
        print(f"  run {rec}")


if __name__ == "__main__":
    main()
