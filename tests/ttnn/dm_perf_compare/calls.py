# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The call-list file: one op and the calls to time.

    {"schema": "dm-perf-calls:v1", "op": "repeat",
     "calls": [{"shape": [1, 1, 32, 64], "dtype": "bfloat16", "layout": "tile",
                "memory_config": "DRAM", "args": {"repeat_dims": [2, 1, 1, 1]}}]}

`memory_config` places the input: "DRAM", "L1", or a sharded spec
`{"layout": "HEIGHT_SHARDED", "buffer": "L1", "shard_shape": [h, w], "grid": [[x0, y0], [x1, y1]],
"orientation": "ROW_MAJOR"}`. In `args`, the strings "DRAM"/"L1" become interleaved memory configs
and `{"tensor": {"shape", "dtype", "layout", "randint": [lo, hi]}}` becomes a device tensor.
"""

import json

SCHEMA = "dm-perf-calls:v1"
DTYPES = ("bfloat16", "bfloat8_b", "bfloat4_b", "float32", "uint32", "int32", "uint16")
LAYOUTS = ("tile", "row_major")


class CallsError(ValueError):
    pass


def _compact(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def call_id(call):
    memory_config = call["memory_config"]
    if not isinstance(memory_config, str):
        memory_config = _compact(memory_config)
    shape = "x".join(str(d) for d in call["shape"])
    return "|".join([shape, call["dtype"], call["layout"], memory_config, _compact(call["args"])])


def _normalize(index, raw):
    if not isinstance(raw, dict):
        raise CallsError(f"call {index}: not an object")
    unknown = set(raw) - {"shape", "dtype", "layout", "memory_config", "args"}
    if unknown:
        raise CallsError(f"call {index}: unknown field(s) {sorted(unknown)}")
    shape = raw.get("shape")
    if not (isinstance(shape, list) and shape and all(isinstance(d, int) and d > 0 for d in shape)):
        raise CallsError(f"call {index}: shape must be a non-empty list of positive ints")
    if raw.get("dtype") not in DTYPES:
        raise CallsError(f"call {index}: dtype must be one of {DTYPES}")
    if raw.get("layout") not in LAYOUTS:
        raise CallsError(f"call {index}: layout must be one of {LAYOUTS}")
    args = raw.get("args", {})
    if not isinstance(args, dict):
        raise CallsError(f"call {index}: args must be an object")
    call = {
        "shape": shape,
        "dtype": raw["dtype"],
        "layout": raw["layout"],
        "memory_config": raw.get("memory_config", "DRAM"),
        "args": args,
    }
    call["id"] = call_id(call)
    return call


def parse_calls(doc):
    if doc.get("schema") != SCHEMA:
        raise CallsError(f"schema must be {SCHEMA!r}")
    op = doc.get("op")
    if not isinstance(op, str) or not op:
        raise CallsError("op must be a non-empty string")
    calls = [_normalize(i, raw) for i, raw in enumerate(doc.get("calls", []))]
    if not calls:
        raise CallsError("no calls")
    ids = [c["id"] for c in calls]
    if len(set(ids)) != len(ids):
        raise CallsError("duplicate calls")
    return {"op": op, "calls": calls}


def load_calls(path):
    with open(path) as f:
        return parse_calls(json.load(f))
