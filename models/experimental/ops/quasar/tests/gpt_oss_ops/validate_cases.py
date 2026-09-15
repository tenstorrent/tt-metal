#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Offline validation of a generated graph_ops suite — no ttnn, no device needed.

    python models/experimental/ops/quasar/tests/gpt_oss_ops/validate_cases.py

Run this after regenerating (and after editing ``graph_case.py``) to catch capture
parsing regressions before spending emulator time on them. It cross-checks the
generated ``CASES`` data against ``graph_case.py``'s own tables, so the two cannot
drift apart silently:

  * ``CASES`` is a pure literal, ids are unique, no unreconstructible spec leaked in;
  * every dtype / layout / memory layout / buffer type / shard orientation name is
    in the corresponding ``graph_case`` table;
  * every program-config kind and every one of its fields is handled by
    ``graph_case._PROGRAM_CONFIG_FIELDS`` (a field the runtime would silently drop
    is an error — that is how a stale field mapping shows up);
  * shard arithmetic: the shard shape times the core count actually covers the
    tensor (allowing DRAM bank padding and tile padding, rejecting a shard grid
    that cannot hold the tensor at all — the signature of a bad parse);
  * bfloat8_b/bfloat4_b tensors are TILE layout;
  * ``GOLDEN`` / ``INDEX_VALUES`` entries refer to ops the capture actually called.
"""

from __future__ import annotations

import ast
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TILE = 32

_graph_case_src = (HERE / "graph_case.py").read_text()
_tree = ast.parse(_graph_case_src)


def _assignment(name):
    for node in _tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == name:
            value = node.value
            if isinstance(value, ast.Call):  # frozenset({...})
                value = value.args[0]
            return value
    raise AssertionError(f"{name} not found in graph_case.py")


def _keys(name):
    """Key set of a dict whose *values* are ttnn objects (not literals)."""
    return {ast.literal_eval(k) for k in _assignment(name).keys}


def _field_map(name):
    node = _assignment(name)
    return {ast.literal_eval(k): set(ast.literal_eval(v)) for k, v in zip(node.keys, node.values)}


DTYPE = _keys("DTYPE")
LAYOUT = _keys("LAYOUT")
MEM_LAYOUT = _keys("MEM_LAYOUT")
BUFFER_TYPE = _keys("BUFFER_TYPE")
ORIENTATION = _keys("ORIENTATION")
PROGRAM_CONFIG_FIELDS = _field_map("_PROGRAM_CONFIG_FIELDS")
DROPPED_FIELDS = _field_map("_DROPPED_FIELDS")
GOLDEN_OPS = set(re.findall(r'^\s+"(ttnn\.[^"]+)": _ref', _graph_case_src, re.M))
POSTCONDITION_OPS = set(re.findall(r'^\s+"(ttnn\.[^"]+)": _check_', _graph_case_src, re.M))
INDEX_OPS = {op for op, _ in re.findall(r'\("(ttnn\.[^"]+)", "([^"]+)"\)', _graph_case_src)}

KINDS = {"t", "tlist", "mem", "cfg", "dtype", "layout", "acts", "device", "lit", "slices", "tile"}

errors: list[str] = []
notes: list[str] = []


def _rows_and_width(shape):
    return math.prod(shape[:-1]) if len(shape) > 1 else 1, shape[-1]


def check_memory_config(mem, where, shape=None):
    if mem is None:
        return
    if mem["layout"] not in MEM_LAYOUT:
        errors.append(f"{where}: memory layout {mem['layout']!r} missing from graph_case.MEM_LAYOUT")
    if mem["buffer"] not in BUFFER_TYPE:
        errors.append(f"{where}: buffer type {mem['buffer']!r} missing from graph_case.BUFFER_TYPE")

    shard = mem.get("shard")
    if shard is None:
        if mem["layout"] != "INTERLEAVED":
            # Real and reproducible: the model passes a sharded memory config with no
            # shard spec and lets the op derive it from the program config.
            notes.append(f"{where}: {mem['layout']} with no shard spec (op derives it)")
        return

    if shard["orientation"] not in ORIENTATION:
        errors.append(f"{where}: shard orientation {shard['orientation']!r} missing from graph_case.ORIENTATION")
    cores = sum((x1 - x0 + 1) * (y1 - y0 + 1) for x0, y0, x1, y1 in shard["grid"])
    if cores <= 0:
        errors.append(f"{where}: empty shard grid {shard['grid']}")
        return
    if shape is None:
        return

    rows, width = _rows_and_width(shape)
    sh, sw = shard["shape"]
    padded_rows = math.ceil(rows / TILE) * TILE

    if mem["layout"] == "WIDTH_SHARDED":
        # width split over cores (padded up per bank/core), full height per shard
        if sw * cores < width:
            errors.append(
                f"{where}: WIDTH_SHARDED shard {sh}x{sw} over {cores} cores cannot cover "
                f"width {width} of shape {shape} — likely a bad capture parse"
            )
        elif sw * (cores - 1) >= width:
            # DRAM/L1 shard widths are rounded up to a tile multiple, so the last
            # bank(s) can end up entirely padding. Normal, but worth seeing.
            notes.append(
                f"{where}: WIDTH_SHARDED shard {sh}x{sw} over {cores} cores leaves "
                f"{cores - math.ceil(width / sw)} core(s) unused for width {width}"
            )
        if sh < min(rows, padded_rows):
            errors.append(f"{where}: WIDTH_SHARDED shard height {sh} < tensor rows {rows} of shape {shape}")
    elif mem["layout"] == "HEIGHT_SHARDED":
        if sh * cores < min(rows, padded_rows):
            errors.append(
                f"{where}: HEIGHT_SHARDED shard {sh}x{sw} over {cores} cores cannot cover "
                f"{rows} rows of shape {shape}"
            )
        if sw < width:
            errors.append(f"{where}: HEIGHT_SHARDED shard width {sw} < tensor width {width} of shape {shape}")
    elif mem["layout"] == "BLOCK_SHARDED":
        if sh * sw * cores < padded_rows * width:
            errors.append(f"{where}: BLOCK_SHARDED shard {sh}x{sw} over {cores} cores cannot cover shape {shape}")


def check_tensor(spec, where):
    if spec["dtype"] not in DTYPE:
        errors.append(f"{where}: dtype {spec['dtype']!r} missing from graph_case.DTYPE")
    if spec["layout"] not in LAYOUT:
        errors.append(f"{where}: layout {spec['layout']!r} missing from graph_case.LAYOUT")
    if not spec["shape"]:
        # Everything below indexes the shape (check_memory_config -> shape[-1]).
        errors.append(f"{where}: empty shape")
        return
    if spec["dtype"] in ("BFLOAT8_B", "BFLOAT4_B") and spec["layout"] != "TILE":
        errors.append(f"{where}: {spec['dtype']} requires TILE layout, capture says {spec['layout']}")
    check_memory_config(spec.get("mem"), where, spec["shape"])


def check_spec(spec, where):
    kind = spec.get("k")
    if kind == "skip":
        errors.append(f"{where}: unreconstructible spec leaked into a generated case: {spec.get('repr')}")
        return
    if kind not in KINDS:
        errors.append(f"{where}: unknown spec kind {kind!r} (graph_case._build_value would raise)")
        return

    if kind == "t":
        check_tensor(spec, where)
    elif kind == "tlist":
        for i, sub in enumerate(spec["tensors"]):
            check_tensor(sub, f"{where}[{i}]")
    elif kind == "mem":
        check_memory_config(spec, where)
    elif kind == "cfg":
        config_kind, fields = spec["kind"], spec["fields"]
        allowed = PROGRAM_CONFIG_FIELDS.get(config_kind)
        if allowed is None:
            errors.append(f"{where}: program config {config_kind!r} has no mapping in graph_case")
            return
        known = allowed | DROPPED_FIELDS.get(config_kind, set()) | {"fused_activation"}
        for field in fields:
            if field not in known:
                errors.append(
                    f"{where}: {config_kind} field {field!r} is not in graph_case._PROGRAM_CONFIG_FIELDS "
                    f"— the runtime would drop it"
                )
        for field in allowed:
            if field not in fields:
                notes.append(f"{where}: {config_kind} has no {field!r} in the capture (ctor default used)")
    elif kind == "dtype" and spec["v"] not in DTYPE:
        errors.append(f"{where}: dtype {spec['v']!r} missing from graph_case.DTYPE")
    elif kind == "layout" and spec["v"] not in LAYOUT:
        errors.append(f"{where}: layout {spec['v']!r} missing from graph_case.LAYOUT")


def main():
    files = sorted(HERE.glob("test_*.py"))
    if not files:
        print(f"no test_*.py in {HERE} — run generate_from_graph_capture.py first")
        return 1

    ops_seen, n_cases, n_calls, n_shards = set(), 0, 0, 0
    for path in files:
        tree = ast.parse(path.read_text(), filename=str(path))
        cases = None
        for node in tree.body:
            if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "CASES":
                try:
                    cases = ast.literal_eval(node.value)
                except ValueError as exc:
                    errors.append(f"{path.name}: CASES is not a pure literal ({exc})")
        if cases is None:
            errors.append(f"{path.name}: no CASES list found")
            continue

        ids = [c["id"] for c in cases]
        if len(set(ids)) != len(ids):
            errors.append(f"{path.name}: duplicate case ids: {sorted(ids)}")

        for case in cases:
            n_cases += 1
            n_calls += case["count"]
            ops_seen.add(case["op"])
            for i, spec in enumerate(case["args"]):
                check_spec(spec, f"{path.name}:{case['id']}:arg{i}")
                n_shards += 1 if spec.get("k") == "t" and (spec.get("mem") or {}).get("shard") else 0
            for name, spec in case["kwargs"].items():
                check_spec(spec, f"{path.name}:{case['id']}:{name}")
            for i, out in enumerate(case["outs"]):
                if out is not None:
                    check_tensor(out, f"{path.name}:{case['id']}:out{i}")

    for op in sorted(GOLDEN_OPS - ops_seen):
        notes.append(f"graph_case.GOLDEN has a reference for {op}, which this capture never called")
    for op in sorted(INDEX_OPS - ops_seen):
        notes.append(f"graph_case.INDEX_VALUES targets {op}, which this capture never called")
    for op in sorted(POSTCONDITION_OPS - ops_seen):
        notes.append(f"graph_case.POSTCONDITION has a check for {op}, which this capture never called")

    print(
        f"{len(files)} op file(s), {n_cases} case(s) covering {n_calls} captured call(s), "
        f"{len(ops_seen)} op(s), {n_shards} sharded input(s)"
    )
    print(f"\n{len(errors)} error(s)")
    for e in errors:
        print(f"  ERROR {e}")
    print(f"\n{len(notes)} note(s)")
    for n in notes:
        print(f"  note  {n}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
