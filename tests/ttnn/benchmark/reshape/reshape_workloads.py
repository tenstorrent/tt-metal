# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Reshape workload records: the ``ttnn::reshape`` calls harvested by graph capture, and how to replay them.

A record is the JSON that ``ReshapeCallRecord`` attaches to every ``ttnn::reshape`` graph node::

    {"input":   {"logical_shape", "padded_shape", "dtype", "layout", "tile", "storage", "memory_config", "topology"},
     "request": {"logical_shape", "padded_shape", "memory_config", "pad_value", "sub_core_grid", "skip_padding_fill"}}

``memory_config`` and ``sub_core_grid`` are the lossless ``ttsl::json`` forms, so ``ttnn.MemoryConfig.from_json``
rebuilds the exact spec, shard grid included. A workload file holds deduplicated records::

    {"version": 1, "workloads": [{"key", "input", "request", "count", "sources"}]}
"""

import hashlib
import json
from pathlib import Path

import torch

import ttnn

WORKLOAD_FILE_VERSION = 1
RESHAPE_FUNCTION_NAME = "ttnn::reshape"
MAX_SOURCES_PER_WORKLOAD = 8

_TORCH_DTYPES = {
    "BFLOAT16": torch.bfloat16,
    "FLOAT32": torch.float32,
    "BFLOAT8_B": torch.float32,
    "BFLOAT4_B": torch.float32,
    "INT32": torch.int32,
    "UINT32": torch.int32,
    "UINT16": torch.int32,
    "UINT8": torch.uint8,
    "INT8": torch.int8,
}

_INTEGER_RANGES = {
    "INT32": (-(2**31), 2**31 - 1),
    "UINT32": (0, 2**31 - 1),
    "UINT16": (0, 2**16 - 1),
    "UINT8": (0, 2**8 - 1),
    "INT8": (-(2**7), 2**7 - 1),
}


def workload_key(record):
    """Identity of a call: everything that changes what the reshape does (the input topology included)."""
    canonical = json.dumps({"input": record["input"], "request": record["request"]}, sort_keys=True)
    return hashlib.sha1(canonical.encode()).hexdigest()


def is_reshape_record(value):
    return isinstance(value, dict) and "input" in value and "request" in value


def reshape_records_from_graph(graph):
    """Records of the outermost ``ttnn::reshape`` calls in a captured graph.

    Reshapes nested inside another reshape (the legacy implementation recurses, and compare mode runs both
    implementations) are implementation details of the outer call, so they are not workloads of their own.
    """
    records = []
    open_functions = []
    for node in graph:
        node_type = node["node_type"]
        if node_type == "function_start":
            name = node["params"]["name"]
            if name == RESHAPE_FUNCTION_NAME and RESHAPE_FUNCTION_NAME not in open_functions:
                record = _record_from_arguments(node.get("arguments", []))
                if record is not None:
                    records.append(record)
            open_functions.append(name)
        elif node_type == "function_end" and open_functions:
            open_functions.pop()
    return records


def _record_from_arguments(arguments):
    for argument in arguments:
        if not argument.startswith("{"):
            continue
        try:
            value = json.loads(argument)
        except json.JSONDecodeError:
            continue
        if is_reshape_record(value):
            return value
    return None


class WorkloadSet:
    """Deduplicated workloads with call counts and a few source test ids each."""

    def __init__(self):
        self._by_key = {}

    def __len__(self):
        return len(self._by_key)

    def add(self, record, source, count=1):
        key = workload_key(record)
        entry = self._by_key.get(key)
        if entry is None:
            entry = {"key": key, "input": record["input"], "request": record["request"], "count": 0, "sources": []}
            self._by_key[key] = entry
        entry["count"] += count
        if source is not None and source not in entry["sources"] and len(entry["sources"]) < MAX_SOURCES_PER_WORKLOAD:
            entry["sources"].append(source)

    def merge_file(self, path):
        for entry in load_workloads(path):
            for source in entry["sources"] or [None]:
                self.add(entry, source, count=0)
            self._by_key[workload_key(entry)]["count"] += entry["count"]

    def save(self, path):
        workloads = sorted(self._by_key.values(), key=lambda entry: (-entry["count"], entry["key"]))
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"version": WORKLOAD_FILE_VERSION, "workloads": workloads}, indent=1) + "\n")


def load_workloads(path):
    data = json.loads(Path(path).read_text())
    if data.get("version") != WORKLOAD_FILE_VERSION:
        raise ValueError(f"{path}: workload file version {data.get('version')}, expected {WORKLOAD_FILE_VERSION}")
    return data["workloads"]


def describe(entry):
    inp, req = entry["input"], entry["request"]
    layout = _memory_config(inp["memory_config"]).memory_layout
    return (
        f"{inp['dtype']} {inp['layout']} {layout} {inp['logical_shape']} -> {req['logical_shape']}"
        f"{' (padded ' + str(req['padded_shape']) + ')' if req['padded_shape'] != req['logical_shape'] else ''}"
    )


def _memory_config(value):
    return None if value is None else ttnn.MemoryConfig.from_json(json.dumps(value))


def _pad_value(value):
    if value is None:
        return None
    return float(value["float"]) if "float" in value else int(value["uint"])


def special_values(torch_dtype):
    """Bit patterns a byte mover must carry unchanged: NaN payloads, signed zero, subnormals, extremes."""
    if torch_dtype == torch.float32:
        bits = [0x7FC00001, 0xFFC12345, 0x80000000, 0x00000001, 0x807FFFFF, 0x7F7FFFFF, 0xFF800000]
        return torch.tensor(bits, dtype=torch.int64).to(torch.int32).view(torch.float32)
    if torch_dtype == torch.bfloat16:
        bits = [0x7FC1, 0xFFA5, 0x8000, 0x0001, 0x807F, 0x7F7F, 0xFF80]
        return torch.tensor(bits, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    return None


def make_torch_input(entry, seed=0):
    """Random input for the record, with special bit patterns sprinkled in for exactly stored float dtypes."""
    inp = entry["input"]
    shape = inp["logical_shape"]
    torch_dtype = _TORCH_DTYPES[inp["dtype"]]
    generator = torch.Generator().manual_seed(seed)
    if inp["dtype"] in _INTEGER_RANGES:
        low, high = _INTEGER_RANGES[inp["dtype"]]
        return torch.randint(low, high, shape, generator=generator, dtype=torch.int64).to(torch_dtype)
    values = torch.randn(shape, generator=generator, dtype=torch.float32).to(torch_dtype)
    specials = special_values(torch_dtype) if inp["dtype"] in ("BFLOAT16", "FLOAT32") else None
    if specials is not None and values.numel() > 0:
        flat = values.view(-1)
        positions = torch.randint(0, flat.numel(), (min(flat.numel(), 4 * specials.numel()),), generator=generator)
        flat[positions] = specials.repeat(positions.numel() // specials.numel() + 1)[: positions.numel()]
    return values


def make_input(entry, device, torch_input=None):
    """Device (or host) tensor with the record's spec: dtype, layout, tile, padded shape and memory config."""
    inp = entry["input"]
    if torch_input is None:
        torch_input = make_torch_input(entry)
    layout = getattr(ttnn.Layout, inp["layout"])
    dtype = getattr(ttnn.DataType, inp["dtype"])
    tile = ttnn.Tile(tuple(inp["tile"])) if layout == ttnn.Layout.TILE else None
    tensor = ttnn.from_torch(torch_input, dtype=dtype, layout=layout, tile=tile)
    if inp["padded_shape"] is not None and list(tensor.padded_shape) != inp["padded_shape"]:
        tensor = ttnn.reshape(tensor, ttnn.Shape(inp["logical_shape"]), ttnn.Shape(inp["padded_shape"]))
    if inp["storage"] == "device":
        tensor = ttnn.to_device(tensor, device, memory_config=_memory_config(inp["memory_config"]))
    return tensor


def request_arguments(entry):
    """Positional and keyword arguments for ``reshape_legacy``/``reshape_new`` reproducing the recorded call."""
    req = entry["request"]
    sub_core_grid = req["sub_core_grid"]
    return (
        (ttnn.Shape(req["logical_shape"]), ttnn.Shape(req["padded_shape"])),
        {
            "memory_config": _memory_config(req["memory_config"]),
            "pad_value": _pad_value(req["pad_value"]),
            "sub_core_grids": None if sub_core_grid is None else ttnn.CoreRangeSet.from_json(json.dumps(sub_core_grid)),
            "skip_padding_fill": req["skip_padding_fill"],
        },
    )


def is_multi_device(entry):
    distribution_shape = entry["input"]["topology"]["distribution_shape"]
    devices = 1
    for extent in distribution_shape:
        devices *= extent
    return devices > 1
