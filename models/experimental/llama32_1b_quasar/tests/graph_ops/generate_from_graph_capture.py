#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Generate a per-op pytest suite from a ttnn graph capture.

    python models/experimental/llama32_1b_quasar/tests/graph_ops/generate_from_graph_capture.py \
        --capture generated/ttnn/reports/<run>/graph_capture.python_io.json \
        --out    models/experimental/llama32_1b_quasar/tests/graph_ops

The capture (produced by running the model demo under ``ttnn.graph.begin_graph_capture``)
records **every python-level ttnn call the model actually made**, with the full repr
of each argument: tensor shapes, dtypes, layouts, memory configs (shard grids
included), program configs and scalars.

This script turns that into one ``test_<op>.py`` per op, each holding the list of
*distinct* calls to that op. Compared with hand-authoring op tests from a static
read of the model source, the capture gives you:

  * the real memory configs and program configs (sharded grids, DRAM-sharded
    weights, per-layer block sizes) instead of an idealized interleaved stand-in,
  * every call site including ones static reading misses (implicit relayouts,
    ``to_memory_config`` inserted by helpers, per-layer config differences),
  * exact call counts, so you can see which shape actually dominates a run,
  * and it is *rerunnable*: point it at the next model's capture and you have that
    model's op suite.

What it cannot give you: tensor values (so inputs are random and index tensors get
semantic values from ``graph_case.INDEX_VALUES``), ``compute_kernel_config`` fields
(recorded only as an object address), and golden outputs (a torch reference is used
where one is unambiguous — see ``graph_case.GOLDEN``).

The generated files are pure data plus a two-line test body; all interpretation
lives in ``graph_case.py``, so improving fidelity does not require regeneration.
"""

from __future__ import annotations

import argparse
import ast
import collections
import json
import math
import re
import subprocess
import sys
from pathlib import Path

# =============================================================================
# Capture parsing
# =============================================================================

# ttnn.Tensor(shape=Shape([...]), dtype=DataType.X, layout=Layout.Y, memory_config=MemoryConfig(...), storage_type=...
_TENSOR_RE = re.compile(
    r"^ttnn\.Tensor\(shape=Shape\(\[(?P<shape>[^\]]*)\]\), "
    r"dtype=DataType\.(?P<dtype>\w+), layout=Layout\.(?P<layout>\w+), "
    r"memory_config=(?P<mem>MemoryConfig\(.*?\)), storage_type="
)
# An element of a python list of tensors, as summarized by ttnn.graph._safe_arg_str:
# same fields as a bare tensor argument, so the memory config comes through too.
# Unanchored so finditer can walk every element of the list.
_TENSOR_SUMMARY_RE = re.compile(_TENSOR_RE.pattern.lstrip("^"))
# Legacy captures (before ttnn.graph._safe_arg_str learned to summarize sequences)
# print each element's *values*, then shape/dtype/layout in C++ spelling, and carry no
# memory config for list elements.
_TENSOR_IN_LIST_RE = re.compile(
    r"shape=Shape\(\[(?P<shape>[^\]]*)\]\), dtype=DataType::(?P<dtype>\w+), layout=Layout::(?P<layout>\w+)"
)
# NOTE: the ShardSpec body contains nested JSON braces (grid=[{"start":{"x":0,…}}]),
# so it must be anchored on its trailing orientation field — a lazy `\{.*?\}` stops
# at the first inner brace and silently loses the whole shard spec.
_MEM_RE = re.compile(
    r"MemoryConfig\(memory_layout=TensorMemoryLayout::(?P<layout>\w+),"
    r"buffer_type=BufferType::(?P<buffer>\w+),"
    r"shard_spec=(?P<shard>std::nullopt|ShardSpec\{.*?orientation=ShardOrientation::\w+\})"
)
_SHARD_RANGE_RE = re.compile(r'"start":\{"x":(\d+),"y":(\d+)\},"end":\{"x":(\d+),"y":(\d+)\}')
_SHARD_SHAPE_RE = re.compile(r"shape=\[(\d+), *(\d+)\]")
_SHARD_ORIENT_RE = re.compile(r"orientation=ShardOrientation::(\w+)")
_CONFIG_RE = re.compile(r"^(?P<kind>[A-Z]\w*(?:ProgramConfig|Config))\((?P<body>.*)\)$")
_TENSOR_ID_RE = re.compile(r"tensor_id=(\d+)")
# ttnn.Tile's repr, e.g. gpt-oss's sparse_matmul(output_tile=ttnn.Tile([32, 32])).
_TILE_RE = re.compile(r"^Tile with shape: \[(?P<h>\d+), (?P<w>\d+)\]$")
# An unquoted string argument: the capture prints a str's value, not its repr.
_BARE_STRING_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
# ttnn.graph._safe_arg_str' marker for a sequence it summarized only in part.
_ELIDED_SEQUENCE_RE = re.compile(r"\.\.\. (\+\d+ more|\d+ element\(s\) below the summary depth limit)")
_GRID_RE = re.compile(r"^(\d+)-(\d+)$")

_DTYPE_TAG = {
    "BFLOAT16": "bf16",
    "BFLOAT8_B": "bf8",
    "BFLOAT4_B": "bf4",
    "FLOAT32": "f32",
    "UINT32": "u32",
    "INT32": "i32",
    "UINT16": "u16",
    "UINT8": "u8",
}
_MEM_TAG = {"INTERLEAVED": "int", "HEIGHT_SHARDED": "hs", "WIDTH_SHARDED": "ws", "BLOCK_SHARDED": "bs"}


# A captured argument repr is arbitrary text, so the literals in it are parsed and
# rebuilt by hand rather than handed to ``ast.literal_eval``: the value is built from
# an explicit whitelist of literal node types, and the input is bounded first, so a
# pathologically long or deeply nested repr cannot cost unbounded memory/recursion.
# Every literal worth having here is a scalar, a small tuple or a short list.
_LITERAL_MAX_LEN = 4096
_LITERAL_MAX_DEPTH = 8
_LITERAL_MAX_NODES = 512

_LITERAL_NUMBER = (int, float, complex)


def _literal_value(node):
    """Build the value of one literal AST node; ValueError on anything else."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Tuple):
        return tuple(_literal_value(e) for e in node.elts)
    if isinstance(node, ast.List):
        return [_literal_value(e) for e in node.elts]
    if isinstance(node, ast.Set):
        return {_literal_value(e) for e in node.elts}
    if isinstance(node, ast.Dict):
        if any(k is None for k in node.keys):  # {**other}: not a literal
            raise ValueError("dict unpacking is not a literal")
        return {_literal_value(k): _literal_value(v) for k, v in zip(node.keys, node.values)}
    # A negative number parses as USub applied to a constant (dim=-1, scale=-0.5).
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        operand = node.operand
        if isinstance(operand, ast.Constant) and isinstance(operand.value, _LITERAL_NUMBER):
            return operand.value if isinstance(node.op, ast.UAdd) else -operand.value
    raise ValueError(f"not a literal: {type(node).__name__}")


def parse_literal(text: str):
    """Parse ``text`` as a bounded python literal (no evaluation of any kind).

    Raises ``ValueError`` when the input exceeds a size limit or is not a literal,
    so callers treat it the same way they treat any unparsed repr.
    """
    if len(text) > _LITERAL_MAX_LEN:
        raise ValueError("literal too long")

    # Check bracket nesting on the raw text: ast.parse itself recurses, so the
    # depth limit has to be enforced before parsing, not on the tree.
    depth = 0
    for ch in text:
        if ch in "([{":
            depth += 1
            if depth > _LITERAL_MAX_DEPTH:
                raise ValueError("literal too deeply nested")
        elif ch in ")]}":
            depth -= 1

    tree = ast.parse(text, mode="eval")
    nodes = 0
    for _ in ast.walk(tree):
        nodes += 1
        if nodes > _LITERAL_MAX_NODES:
            raise ValueError("literal too large")
    return _literal_value(tree.body)


def parse_memory_config(text: str):
    """Parse a captured MemoryConfig; None when it does not parse *completely*.

    Failing closed matters here: a partially parsed config is indistinguishable from
    a real one. A ShardSpec that stopped matching would leave ``shard: None``, which
    is also what a genuine "sharded layout, no shard spec, the op derives it" capture
    looks like — and the case would then run interleaved while claiming to reproduce
    a sharded call. Callers turn None into an unreconstructible argument instead, so
    the case is dropped and reported rather than silently weakened.
    """
    m = _MEM_RE.search(text)
    if not m:
        return None
    spec = {"layout": m.group("layout"), "buffer": m.group("buffer"), "shard": None}
    shard = m.group("shard")
    if shard != "std::nullopt":
        ranges = [[int(v) for v in r] for r in _SHARD_RANGE_RE.findall(shard)]
        shape = _SHARD_SHAPE_RE.search(shard)
        orient = _SHARD_ORIENT_RE.search(shard)
        if not (ranges and shape and orient):
            return None  # an explicit ShardSpec we could not read
        spec["shard"] = {
            "grid": ranges,
            "shape": [int(shape.group(1)), int(shape.group(2))],
            "orientation": orient.group(1),
        }
    return spec


def _shape(text: str):
    return [int(v) for v in text.split(",") if v.strip()]


def parse_config(text: str):
    """MatmulMultiCoreReuseMultiCast…(in0_block_w=2,…) -> {"kind":…, "fields": {…}}."""
    m = _CONFIG_RE.match(text.strip())
    if not m:
        return None
    fields = {}
    # Split on commas that are not inside brackets/braces.
    depth, token, tokens = 0, "", []
    for ch in m.group("body"):
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            tokens.append(token)
            token = ""
        else:
            token += ch
    tokens.append(token)

    for tok in tokens:
        if "=" not in tok:
            continue
        key, _, raw = tok.partition("=")
        key, raw = key.strip(), raw.strip()
        grid = _GRID_RE.match(raw)
        if grid:  # compute_with_storage_grid_size=8-4
            fields[key] = [int(grid.group(1)), int(grid.group(2))]
        elif raw == "std::nullopt":
            fields[key] = None
        elif raw in ("true", "false"):
            fields[key] = raw == "true"
        else:
            try:
                fields[key] = parse_literal(raw)
            except (ValueError, SyntaxError):
                fields[key] = raw
    return {"kind": m.group("kind"), "fields": fields}


def parse_argument(text: str):
    """One captured argument repr -> a spec dict understood by graph_case."""
    m = _TENSOR_RE.match(text)
    if m:
        mem = parse_memory_config(m.group("mem"))
        if mem is None:
            # The repr carries a MemoryConfig we could not read; running the case with
            # a default placement would not be the captured call (see parse_memory_config).
            return {"k": "skip", "repr": text}
        return {
            "k": "t",
            "shape": _shape(m.group("shape")),
            "dtype": m.group("dtype"),
            "layout": m.group("layout"),
            "mem": mem,
        }

    if text.startswith("[ttnn.Tensor(") or text.startswith("(ttnn.Tensor("):
        # Every element must be accounted for: a scan over a list whose element repr
        # changed would quietly return a subset, and a concat case would then run with
        # fewer operands than the captured call and still pass.
        # An elided sequence is exactly that subset: _safe_arg_str summarizes at most
        # _MAX_SEQUENCE_ELEMENTS entries and appends "... +N more", which the per-element counts
        # below cannot see. Fail closed instead of emitting a case with the operands that fit.
        if _ELIDED_SEQUENCE_RE.search(text):
            return {"k": "skip", "repr": text}
        expected = text.count("ttnn.Tensor(")
        summaries = list(_TENSOR_SUMMARY_RE.finditer(text))
        if len(summaries) == expected:
            tensors = []
            for m in summaries:
                mem = parse_memory_config(m.group("mem"))
                if mem is None:
                    return {"k": "skip", "repr": text}
                tensors.append(
                    {
                        "k": "t",
                        "shape": _shape(m.group("shape")),
                        "dtype": m.group("dtype"),
                        "layout": m.group("layout"),
                        "mem": mem,
                    }
                )
            return {"k": "tlist", "tensors": tensors}
        found = _TENSOR_IN_LIST_RE.findall(text)
        if not found or len(found) != expected:
            return {"k": "skip", "repr": text}
        tensors = [{"k": "t", "shape": _shape(sh), "dtype": dt, "layout": lay, "mem": None} for sh, dt, lay in found]
        return {"k": "tlist", "tensors": tensors}

    if "ttnn.Tensor(" in text and (text.startswith("[") or text.startswith("(")):
        # A nested sequence -- [[tensor], [tensor]] -- which _safe_arg_str now summarizes rather
        # than str()-ing. graph_case rebuilds a "tlist" as one flat list of operands, so parsing
        # this would flatten the structure and call the op with a different argument shape than
        # the capture recorded. Fail closed; the generated file's docstring reports the drop.
        return {"k": "skip", "repr": text}

    if text.startswith("MemoryConfig("):
        spec = parse_memory_config(text)
        return dict(spec, k="mem") if spec else {"k": "skip", "repr": text}

    if text.startswith("DataType."):
        return {"k": "dtype", "v": text.split(".", 1)[1]}
    if text.startswith("Layout."):
        return {"k": "layout", "v": text.split(".", 1)[1]}
    if text.startswith("MeshDevice("):
        return {"k": "device"}
    tile = _TILE_RE.match(text)
    if tile:
        return {"k": "tile", "shape": [int(tile.group("h")), int(tile.group("w"))]}
    if text.startswith("[UnaryOpType."):
        return {"k": "acts", "v": re.findall(r"UnaryOpType\.(\w+)", text)}
    if text.startswith("slice(") or text.startswith("(slice("):
        keys = re.findall(r"slice\(([^)]*)\)", text)
        parsed = []
        for k in keys:
            parts = [None if p.strip() == "None" else int(p) for p in k.split(",")]
            parsed.append(parts)
        return {"k": "slices", "v": parsed} if parsed else {"k": "skip", "repr": text}

    cfg = parse_config(text)
    if cfg:
        return dict(cfg, k="cfg")

    if text.startswith("<") or text.startswith("torch.Tensor("):
        # Object repr (compute_kernel_config, mesh mapper) or a host torch tensor:
        # not reconstructible. Strip the heap address, otherwise two identical
        # calls from different layers look like different cases.
        return {"k": "skip", "repr": re.sub(r" at 0x[0-9a-f]+", "", text)}

    try:
        return {"k": "lit", "v": parse_literal(text)}
    except (ValueError, SyntaxError):
        # A bare identifier is an unquoted string argument -- ttnn.linear(activation="gelu")
        # reaches the capture as `gelu`, which literal_eval cannot read. Without this branch the
        # whole call is dropped: 84 of the Qwen3-VL capture's linear calls (the vision MLP and
        # patch-merger gelu fusions) were unreconstructible for want of it, and with them the only
        # calls that launch the standalone activation program (matmul.cpp:355 runs a fused
        # activation as a separate unary_chain when no core_coord is given).
        # True/False/None never reach here -- literal_eval takes them above. nan/inf/infinity do
        # reach here (literal_eval rejects them as bare identifiers) and must stay floats: a
        # captured `epsilon=inf` reconstructed as the string "inf" would change the argument's
        # type for any op reading a scalar kwarg.
        try:
            return {"k": "lit", "v": float(text)}
        except ValueError:
            pass
        if _BARE_STRING_RE.match(text):
            return {"k": "lit", "v": text}
        return {"k": "skip", "repr": text}


def iter_records(path: Path):
    """Stream the top-level array one record at a time (the file is ~100 MB)."""
    blob = path.read_text()
    decoder = json.JSONDecoder()
    i = blob.index("[") + 1
    while True:
        while i < len(blob) and blob[i] in " \n\r\t,":
            i += 1
        if i >= len(blob) or blob[i] == "]":
            return
        record, i = decoder.raw_decode(blob, i)
        yield record


# =============================================================================
# Case building
# =============================================================================

# Host-side plumbing: no device op of its own to iterate on. from_torch / as_tensor
# are exercised implicitly — every generated case uploads its inputs with from_torch
# using the captured dtype/layout/memory config.
SKIP_OPS = {
    "ttnn.deallocate",  # frees a buffer; nothing to assert
    "ttnn.from_torch",  # covered by every case's input setup
    "ttnn.as_tensor",  # from_torch + on-disk weight cache
    "ttnn.load_tensor",  # reads a model_cache/ file that only the demo produces
    "ttnn.to_torch",  # device -> host readback, used by every assertion already
}

# Keyword arguments dropped before cases are deduped, because the capture records
# only an object address for them — keeping them would make every layer's call look
# distinct while carrying no reconstructible information. The op falls back to its
# own default (documented in README.md / graph_case.py).
DROP_KWARGS = {"compute_kernel_config"}

# The captured name doubles as the callable expression (``ttnn.Tensor.__getitem__``
# is bound in ttnn/__init__.py:478, so even that one needs no wrapper). Ops whose
# python entry point differs from the captured name go here.
#
# Quasar-uplifted ops: the model now calls the ``ttnn.experimental.quasar``
# fork of the Metal 2.0 port, so the generated test's callable (``_OP``) must target
# the fork. Only ``_OP`` is remapped; ``case["op"]`` stays the captured current-gen
# name so it still keys into graph_case.py's INDEX_VALUES / GOLDEN tables (the op is
# semantically identical, and its index-tensor generators are keyed by that name).
OP_EXPR: dict[str, str] = {
    "ttnn.transformer.scaled_dot_product_attention_decode": "ttnn.experimental.quasar.transformer.scaled_dot_product_attention_decode",
    "ttnn.transformer.paged_scaled_dot_product_attention_decode": "ttnn.experimental.quasar.transformer.paged_scaled_dot_product_attention_decode",
    "ttnn.add": "ttnn.experimental.quasar.add",
    "ttnn.multiply": "ttnn.experimental.quasar.multiply",
}

FILE_NAME = {"ttnn.Tensor.__getitem__": "test_tensor_getitem"}


def file_stem(op: str) -> str:
    if op in FILE_NAME:
        return FILE_NAME[op]
    return "test_" + op.split(".")[-1]


def merge_outs(old, new, conflicts, op):
    """Combine two observations of the same call's outputs, position by position.

    Identical calls appear many times in a capture and their outputs are not equally
    well observed: an output is only recoverable if that tensor is used again later,
    so the same call can yield a spec on one occurrence and None on another. Keeping
    whichever came first would make the generated suite depend on capture order, so
    the strongest observation wins instead. Two *different* non-None observations of
    the same position are contradictory (the same call cannot produce two placements),
    so that position is dropped to None and reported rather than silently picked.
    """
    merged = []
    for i in range(max(len(old), len(new))):
        a = old[i] if i < len(old) else None
        b = new[i] if i < len(new) else None
        if a is None or a == b:
            merged.append(b if a is None else a)
        elif b is None:
            merged.append(a)
        else:
            conflicts[op] += 1
            merged.append(None)
    return merged


def canonical(spec) -> str:
    """Dedup key: everything except volatile identity (tensor_id, addresses)."""
    return json.dumps(spec, sort_keys=True)


def case_id(index: int, case: dict) -> str:
    """Short, greppable id: leading shape + dtype + memory layout of input 0."""
    first = next((a for a in case["args"] if a["k"] == "t"), None)
    if first is None:
        first = next((t for a in case["args"] if a["k"] == "tlist" for t in a["tensors"]), None)
    if first is None:
        return f"{index:02d}"

    dims = first["shape"] or [1]
    while len(dims) > 2 and dims[0] == 1:
        dims = dims[1:]
    tag = "x".join(str(d) for d in dims)
    dtype = _DTYPE_TAG.get(first["dtype"], first["dtype"].lower())
    mem = first.get("mem")
    mem_tag = f"{_MEM_TAG.get(mem['layout'], '?')}-{mem['buffer'].lower()}" if mem else "host"
    return f"{index:02d}_{tag}_{dtype}_{mem_tag}"


def build_cases(capture: Path, verbose=False):
    """Collect deduped cases per op, plus a tensor_id -> spec map for outputs."""
    tensor_specs: dict[int, str] = {}  # id -> canonical spec (None when ambiguous)
    per_op: dict[str, dict[str, dict]] = collections.defaultdict(dict)
    order: list[str] = []
    raw_counts = collections.Counter()
    dropped_kwargs: dict[str, set] = collections.defaultdict(set)
    out_conflicts = collections.Counter()  # same call, contradictory output observations

    records = list(iter_records(capture))

    # Pass 1: every tensor repr anywhere in the capture teaches us that tensor's
    # spec, which is how an op's *output* spec is recovered (it shows up as some
    # later op's input).
    for record in records:
        for text in record["arguments"].values():
            m = _TENSOR_ID_RE.search(text)
            if not m or not text.startswith("ttnn.Tensor("):
                continue
            spec = parse_argument(text)
            if spec["k"] != "t":
                continue
            tid = int(m.group(1))
            key = canonical(spec)
            if tid in tensor_specs and tensor_specs[tid] != key:
                tensor_specs[tid] = None  # id reused with a different spec: unusable
            else:
                tensor_specs.setdefault(tid, key)

    # Pass 2: build one case per distinct call.
    for record in records:
        op = record["name"]
        raw_counts[op] += 1
        if op in SKIP_OPS:
            continue

        args, kwargs = [], {}
        for key, text in record["arguments"].items():
            if key in DROP_KWARGS:
                dropped_kwargs[op].add(key)
                continue
            spec = parse_argument(text)
            if key.isdigit():
                args.append((int(key), spec))
            else:
                kwargs[key] = spec
        args = [spec for _, spec in sorted(args)]

        # One spec per returned tensor, positionally aligned with what the op
        # returns — nlp_create_qkv_heads records three ids (Q, K, V) and each gets
        # its own shape/dtype check. An entry is None when that output is never
        # consumed again in the capture, so its spec was never observed; the entry
        # is kept anyway so the list length still pins the op's output count.
        out_ids = record.get("output_tensor_ids") or []
        outs = [json.loads(tensor_specs[tid]) if tensor_specs.get(tid) else None for tid in out_ids]

        case = {"op": op, "args": args, "kwargs": kwargs, "outs": outs}
        dedup = canonical({"args": args, "kwargs": kwargs})
        bucket = per_op[op]
        if op not in order:
            order.append(op)
        if dedup in bucket:
            bucket[dedup]["count"] += 1
            bucket[dedup]["outs"] = merge_outs(bucket[dedup]["outs"], outs, out_conflicts, op)
        else:
            case["count"] = 1
            bucket[dedup] = case

    if verbose:
        for op in order:
            print(f"  {op:<58} {raw_counts[op]:>6} calls -> {len(per_op[op]):>3} distinct")
    for op, n in sorted(out_conflicts.items()):
        print(f"  NOTE {op}: {n} output(s) observed with two different specs for the same call; left unchecked")

    return {op: list(per_op[op].values()) for op in order}, raw_counts, dropped_kwargs


def drop_unreconstructible(cases, op):
    """Split cases into runnable ones and ones whose arguments we cannot rebuild.

    ``DROP_KWARGS`` is already gone by this point. Anything else that parsed to a
    ``skip`` spec (a host torch tensor, a mesh-mapper object, an unparsed repr)
    makes the case unusable, and is reported in the generated file's docstring
    rather than silently dropped.
    """
    keep, dropped = [], []
    for case in cases:
        bad = [k for k, v in case["kwargs"].items() if v["k"] == "skip"]
        bad += [str(i) for i, v in enumerate(case["args"]) if v["k"] == "skip"]
        if bad:
            dropped.append((case, bad))
        else:
            keep.append(case)
    return keep, dropped


# =============================================================================
# Emission
# =============================================================================

# Marker line every generated file carries; also how a stale file from a previous
# capture is recognized as safe to delete (see main()).
_GENERATED_MARKER = "GENERATED FILE - do not edit by hand."

_HEADER = '''# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# ---------------------------------------------------------------------------
# GENERATED FILE - do not edit by hand.
# Regenerate with:
#   python {generator} \\
#       --capture {capture} --out {out}
# Source capture: {capture}
# ---------------------------------------------------------------------------
"""
Per-op test: ``{op}`` — every distinct call the model made, as captured.

{summary}

Each CASES entry is one distinct call: the exact input shapes / dtypes / layouts /
memory configs, the keyword arguments (memory_config, program_config, scalars) and
one captured output spec per tensor the op returned. ``count`` is how many times
that exact call occurred in the captured run. See ``graph_case.py`` for how a case
is materialized and checked, and README.md for the fidelity caveats (random inputs,
no compute_kernel_config).
"""

import pytest

import ttnn
from {runtime_module} import graph_case as G

_OP = {op_expr}

CASES = {cases}


@G.with_default_mesh()
@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def {test_name}(ttnn_mesh_device, reset_seeds, case):
    G.run_case(_OP, case, ttnn_mesh_device)
'''


def _fmt(value, indent=0):
    """Compact, diff-friendly repr: one dict per line, nested dicts inline."""
    pad = " " * indent
    if isinstance(value, dict):
        inner = ", ".join(f'"{k}": {_fmt(v)}' for k, v in value.items())
        return "{" + inner + "}"
    if isinstance(value, list):
        return "[" + ", ".join(_fmt(v) for v in value) + "]"
    return repr(value)


def render_cases(cases) -> str:
    lines = ["["]
    for case in cases:
        lines.append("    {")
        lines.append(f'        "id": {case["id"]!r},')
        lines.append(f'        "op": {case["op"]!r},')
        lines.append(f'        "count": {case["count"]},')
        lines.append(f'        "args": [')
        for spec in case["args"]:
            lines.append(f"            {_fmt(spec)},")
        lines.append("        ],")
        lines.append('        "kwargs": {')
        for name, spec in case["kwargs"].items():
            lines.append(f'            "{name}": {_fmt(spec)},')
        lines.append("        },")
        lines.append('        "outs": [')
        for spec in case["outs"]:
            lines.append(f"            {_fmt(spec)},")
        lines.append("        ],")
        lines.append("    },")
    lines.append("]")
    return "\n".join(lines)


def _tensor_specs(case):
    for spec in list(case["args"]) + list(case["kwargs"].values()):
        if spec.get("k") == "t":
            yield spec
        elif spec.get("k") == "tlist":
            yield from spec["tensors"]


def _rows(shape):
    return math.prod(shape[:-1]) if len(shape) > 1 else 1


def _partial_shard(spec):
    """Logical shape does not fill its shard region (see graph_case._is_partial_shard)."""
    mem = spec.get("mem")
    if not mem or not mem.get("shard"):
        return False
    shard = mem["shard"]
    cores = sum((x1 - x0 + 1) * (y1 - y0 + 1) for x0, y0, x1, y1 in shard["grid"])
    capacity = shard["shape"][0] * (cores if mem["layout"] == "HEIGHT_SHARDED" else 1)
    return _rows(spec["shape"]) < capacity


def _undefinable_output(case):
    """An output has more elements than every input combined, so part of it is untouched."""
    in_elems = sum(math.prod(s["shape"]) for s in _tensor_specs(case))
    return any(math.prod(out["shape"]) > in_elems for out in case["outs"] if out is not None)


def summarize(op, cases, raw_count, dropped, dropped_kwargs) -> str:
    total = sum(c["count"] for c in cases)
    lines = [
        f"Captured {raw_count} call(s) to this op; {len(cases)} distinct signature(s) covering {total} of them.",
    ]

    caveats = []
    if dropped_kwargs:
        caveats.append(
            f"{', '.join(sorted(dropped_kwargs))} is recorded only as an object address in the "
            f"capture, so it is dropped and the op's default is used (can shift PCC, not shapes)"
        )
    if any(a["k"] == "tlist" for case in cases for a in case["args"]):
        caveats.append(
            "one argument is a python list of tensors; that repr carries shape/dtype/layout but "
            "NOT memory configs, so the list elements are uploaded as DRAM interleaved"
        )
    if any(s["dtype"] in ("UINT32", "INT32") for case in cases for s in _tensor_specs(case)):
        caveats.append(
            "an integer index tensor is involved; the capture holds no values, so it is filled by "
            "graph_case.INDEX_VALUES (page ids, positions, token ids) instead of random data"
        )
    if any(_partial_shard(s) for case in cases for s in _tensor_specs(case)):
        caveats.append(
            "an input's logical shape does not fill its shard (e.g. 8 rows in a 32-row shard), so it is "
            "built interleaved and relaid out — handing that memory config straight to from_torch would "
            "pad the logical shape up to the shard and change what the op computes"
        )
    if any(_undefinable_output(case) for case in cases):
        caveats.append(
            "the output has more elements than all inputs combined (a batch-padded decode tensor), so the "
            "op cannot write all of it; finiteness is asserted over the portion the inputs can account for"
        )
    if caveats:
        lines.append("")
        lines.append("Fidelity notes for this op:")
        lines += [f"  * {c}" for c in caveats]

    if dropped:
        lines.append("")
        lines.append("NOT generated (arguments not reconstructible from the capture):")
        for case, bad in dropped:
            lines.append(f"  * {case['count']} call(s): argument(s) {', '.join(bad)}")
    return "\n".join(lines)


def repo_relative(path: Path) -> Path:
    """``path`` relative to the repo root, so the header holds a runnable command."""
    resolved = path.resolve()
    for root in resolved.parents:
        if (root / ".git").exists():
            return resolved.relative_to(root)
    return resolved


def runtime_module(out_dir: Path, explicit: str | None = None) -> str:
    """Dotted import path of the package that holds ``graph_case.py``.

    Derived from ``--out``, not hardcoded: pointing the generator at another model's
    ``tests/graph_ops/`` must produce files that import *that* directory's
    ``graph_case.py``, since the runtime's ``op_utils`` import (assert_pcc, from_tt,
    the mesh fixture) is model-specific. ``--runtime-module`` overrides the guess.
    """
    if explicit:
        return explicit
    resolved = out_dir.resolve()
    for root in resolved.parents:
        if (root / ".git").exists():
            return ".".join(resolved.relative_to(root).parts)
    raise SystemExit(f"cannot derive an import path for --out {out_dir} (no repo root above it); pass --runtime-module")


def write_op_file(out_dir: Path, capture: Path, op: str, cases, raw_count, dropped, dropped_kwargs, module: str):
    for i, case in enumerate(cases):
        case["id"] = case_id(i, case)

    stem = file_stem(op)
    text = _HEADER.format(
        generator=repo_relative(Path(__file__)),
        capture=capture,
        out=out_dir,
        runtime_module=module,
        op=op,
        summary=summarize(op, cases, raw_count, dropped, dropped_kwargs),
        op_expr=OP_EXPR.get(op, op),
        cases=render_cases(cases),
        test_name=stem,
    )
    path = out_dir / f"{stem}.py"
    path.write_text(text)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capture", required=True, type=Path, help="graph_capture.python_io.json")
    ap.add_argument("--out", required=True, type=Path, help="output directory for test_<op>.py files")
    ap.add_argument("--max-cases", type=int, default=32, help="cap on distinct cases per op (most frequent kept)")
    ap.add_argument("--no-format", action="store_true", help="skip running black on the generated files")
    ap.add_argument(
        "--runtime-module",
        help="dotted import path of the package holding graph_case.py (default: derived from --out)",
    )
    args = ap.parse_args()

    print(f"reading {args.capture} ({args.capture.stat().st_size / 1e6:.0f} MB)")
    per_op, raw_counts, dropped_kwargs = build_cases(args.capture, verbose=True)
    args.out.mkdir(parents=True, exist_ok=True)

    module = runtime_module(args.out, args.runtime_module)
    print(f"generated tests will import their runtime from {module}.graph_case")
    if not (args.out / "graph_case.py").exists():
        print(f"  NOTE {args.out}/graph_case.py does not exist — copy it there and point its op_utils import at")
        print("       this model's tests/ops/op_utils.py (assert_pcc / from_tt / the mesh fixture)")

    written = []
    for op, cases in per_op.items():
        cases, dropped = drop_unreconstructible(cases, op)
        cases.sort(key=lambda c: (-c["count"], canonical(c["args"])))
        if len(cases) > args.max_cases:
            print(
                f"  NOTE {op}: {len(cases)} distinct cases, keeping the {args.max_cases} "
                f"most frequent ({sum(c['count'] for c in cases[args.max_cases:])} calls dropped)"
            )
            cases = cases[: args.max_cases]
        if not cases:
            print(f"  SKIP {op}: no reconstructible cases ({len(dropped)} dropped)")
            continue
        path = write_op_file(args.out, args.capture, op, cases, raw_counts[op], dropped, dropped_kwargs[op], module)
        written.append(path)
        note = f"  ({len(dropped)} case(s) not reconstructible)" if dropped else ""
        print(f"  wrote {path.name:<52} {len(cases):>3} cases{note}")

    # An op that the new capture never called (or that lost its last reconstructible
    # case) would otherwise keep its file from the previous run, leaving the suite a
    # mix of two captures while claiming to be this one.
    # Only files this generator wrote are removed (the marker line), so a hand-written
    # test that happens to live in --out is left alone.
    generated = {p.resolve() for p in written}
    for stale in sorted(args.out.glob("test_*.py")):
        if stale.resolve() in generated or _GENERATED_MARKER not in stale.read_text():
            continue
        stale.unlink()
        print(f"  removed stale {stale.name} (not in this capture)")

    for op in sorted(SKIP_OPS & set(raw_counts)):
        print(f"  skipped {op} ({raw_counts[op]} calls) — host-side plumbing, see SKIP_OPS")

    if not args.no_format and written:
        try:
            subprocess.run(
                [sys.executable, "-m", "black", "-l", "120", *[str(p) for p in written]],
                check=True,
                capture_output=True,
            )
            print(f"formatted {len(written)} file(s) with black")
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            print(f"black not run ({exc}); files are written unformatted")

    print(f"\n{len(written)} op file(s) in {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
