# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Harvest the ``ttnn.reshape`` calls real workloads make, for ``bench_reshape_policies.py`` and the C++ benchmark.

As a pytest plugin, every test runs under ``ttnn.graph.begin_graph_capture`` and each outermost ``ttnn::reshape``
node -- views and no-ops included, which leave no device-operation node -- is deduplicated into a workload file::

    PYTHONPATH=tests/ttnn/benchmark/reshape pytest -p capture_reshape_workloads \\
        --reshape-capture=reshape_workloads.json models/demos/.../test_model.py

An existing file is merged into rather than overwritten, so several runs accumulate. As a script, it converts the
``ttnn.reshape`` entries of a model-tracer master JSON (``model_tracer/traced_operations``)::

    python tests/ttnn/benchmark/reshape/capture_reshape_workloads.py import-master \\
        ttnn_operations_master.json -o reshape_workloads.json

Master JSON records lack the padded shapes and the lossless memory configs graph capture records, so imported
workloads use default padding (``padded_shape`` null), and shard specs are rebuilt from the traced grid, shape and
orientation.
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path

import pytest

import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reshape_workloads import WorkloadSet, reshape_records_from_graph  # noqa: E402

_CAPTURE_OPTION = "--reshape-capture"


def pytest_addoption(parser):
    parser.addoption(
        _CAPTURE_OPTION,
        action="store",
        default=None,
        metavar="PATH",
        help="Capture every ttnn.reshape call of the run into the reshape workload file PATH (merged if it exists).",
    )


def pytest_configure(config):
    path = config.getoption(_CAPTURE_OPTION)
    if path is not None:
        config.pluginmanager.register(_ReshapeCapture(Path(path)), "reshape-capture")


class _ReshapeCapture:
    def __init__(self, path):
        self.path = path
        self.workloads = WorkloadSet()
        if path.exists():
            self.workloads.merge_file(path)
        self.captured_calls = 0

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_call(self, item):
        ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
        try:
            yield
        finally:
            graph = ttnn.graph.end_graph_capture()
            for record in reshape_records_from_graph(graph):
                self.workloads.add(record, item.nodeid)
                self.captured_calls += 1

    def pytest_sessionfinish(self, session):
        self.workloads.save(self.path)

    def pytest_terminal_summary(self, terminalreporter):
        terminalreporter.write_line(
            f"reshape capture: {self.captured_calls} calls this run, {len(self.workloads)} distinct workloads "
            f"in {self.path}"
        )


# --- model-tracer master JSON import ---------------------------------------------------------------------------


def _enum_name(value):
    return str(value).rsplit(".", 1)[-1].upper()


def _int_list(value):
    if isinstance(value, dict):
        value = value.get("value", value.get("data", value.get("shape")))
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    return [int(v) for v in re.findall(r"-?\d+", str(value))]


def _resolve_inferred_dim(shape, volume):
    if -1 not in shape:
        return shape
    known = math.prod(d for d in shape if d != -1)
    return [volume // known if d == -1 else d for d in shape]


def _memory_config_from_traced(traced):
    """The tracer's memory config dict (``{"memory_layout": "TensorMemoryLayout::HEIGHT_SHARDED", ...}``)."""
    if not isinstance(traced, dict):
        return ttnn.DRAM_MEMORY_CONFIG
    traced = traced.get("data", traced)
    buffer_type = ttnn.BufferType.L1 if "L1" in str(traced.get("buffer_type", "")) else ttnn.BufferType.DRAM
    memory_layout = next(
        (
            getattr(ttnn.TensorMemoryLayout, name)
            for name in ("HEIGHT_SHARDED", "WIDTH_SHARDED", "BLOCK_SHARDED")
            if name in str(traced.get("memory_layout", ""))
        ),
        ttnn.TensorMemoryLayout.INTERLEAVED,
    )
    shard_spec = traced.get("shard_spec")
    if memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED or not isinstance(shard_spec, dict):
        return ttnn.MemoryConfig(memory_layout, buffer_type)
    grid = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(
                ttnn.CoreCoord(r["start"]["x"], r["start"]["y"]), ttnn.CoreCoord(r["end"]["x"], r["end"]["y"])
            )
            for r in shard_spec["grid"]
        ]
    )
    orientation = (
        ttnn.ShardOrientation.COL_MAJOR
        if str(shard_spec.get("orientation", "")) in ("COL_MAJOR", "ShardOrientation::COL_MAJOR", "1")
        else ttnn.ShardOrientation.ROW_MAJOR
    )
    return ttnn.MemoryConfig(
        memory_layout, buffer_type, ttnn.ShardSpec(grid, _int_list(shard_spec["shape"]), orientation)
    )


def _lossless_memory_config(traced):
    return json.loads(_memory_config_from_traced(traced).to_json())


def _topology(tensor_placement):
    if not tensor_placement:
        return {"distribution_shape": [1], "placements": ["replicate"]}
    placements = []
    for match in re.finditer(r"Shard\((-?\d+)\)|Replicate", str(tensor_placement.get("placement", ""))):
        placements.append({"shard": int(match.group(1))} if match.group(1) is not None else "replicate")
    distribution_shape = _int_list(tensor_placement.get("distribution_shape", "[1]")) or [1]
    return {"distribution_shape": distribution_shape, "placements": placements or ["replicate"]}


def record_from_master_configuration(configuration):
    arguments = configuration["arguments"]
    tensor = arguments["arg0"]
    logical = _int_list(tensor["original_shape"])
    layout = _enum_name(tensor.get("layout", "TILE"))
    target = _resolve_inferred_dim(_int_list(arguments.get("arg1", arguments.get("shape"))), math.prod(logical))
    requested_padded = arguments.get("arg2")
    requested_memory_config = arguments.get("memory_config")
    pad_value = arguments.get("pad_value")
    return {
        "input": {
            "logical_shape": logical,
            "padded_shape": None,
            "dtype": _enum_name(tensor.get("original_dtype", tensor.get("dtype"))),
            "layout": layout,
            "tile": [32, 32],
            "storage": "host" if "HOST" in str(tensor.get("storage_type", "")).upper() else "device",
            "memory_config": _lossless_memory_config(tensor.get("memory_config")),
            "topology": _topology(tensor.get("tensor_placement")),
        },
        "request": {
            "logical_shape": target,
            "padded_shape": _int_list(requested_padded) if isinstance(requested_padded, (list, dict)) else target,
            "memory_config": (
                _lossless_memory_config(requested_memory_config) if isinstance(requested_memory_config, dict) else None
            ),
            "pad_value": None if pad_value in (None, "None") else {"float": float(pad_value)},
            "sub_core_grid": None,
            "skip_padding_fill": False,
        },
    }


def import_master(master_path, output_path, operation_names=("ttnn.reshape",)):
    master = json.loads(Path(master_path).read_text())
    workloads = WorkloadSet()
    if Path(output_path).exists():
        workloads.merge_file(output_path)
    imported, skipped = 0, 0
    for name in operation_names:
        for configuration in master.get("operations", {}).get(name, {}).get("configurations", []):
            try:
                record = record_from_master_configuration(configuration)
            except (KeyError, ValueError, TypeError) as error:
                skipped += 1
                print(f"skipping config {configuration.get('config_id')}: {error!r}", file=sys.stderr)
                continue
            executions = configuration.get("executions", [])
            count = sum(int(e.get("count", 1)) for e in executions) or 1
            sources = [e.get("source") for e in executions if e.get("source")] or [f"master:{Path(master_path).name}"]
            workloads.add(record, sources[0], count=count)
            for source in sources[1:]:
                workloads.add(record, source, count=0)
            imported += 1
    workloads.save(output_path)
    print(
        f"imported {imported} configurations ({skipped} skipped); {len(workloads)} distinct workloads in {output_path}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    importer = commands.add_parser("import-master", help="convert ttnn.reshape entries of a model-tracer master JSON")
    importer.add_argument("master", type=Path)
    importer.add_argument("-o", "--output", type=Path, default=Path("reshape_workloads.json"))
    importer.add_argument(
        "--operation", action="append", dest="operations", help="operation name(s) to import (default ttnn.reshape)"
    )
    args = parser.parse_args()
    if args.command == "import-master":
        import_master(args.master, args.output, tuple(args.operations or ("ttnn.reshape",)))


if __name__ == "__main__":
    main()
