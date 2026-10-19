#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Regenerate configs/sweeps/<op>.json: a fixed-seed sample of each op's sweep grid as call lists.

    python tests/ttnn/dm_perf_compare/sample_sweeps.py [--ops repeat,untilize] [--count 40] [--seed 0]

The sweep modules are imported with ttnn replaced by a symbolic stub, so this needs neither a
device nor a built tree. Vectors the sweep itself rejects (`invalidate_vector`) are dropped, and so
are sharded inputs and dtypes/layouts the call-list schema cannot express. Duplicates collapse
before sampling, so the sample is over distinct calls.
"""

import argparse
import importlib.util
import itertools
import json
import os
import random
import re
import sys
import types
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
SWEEPS = os.path.join(REPO, "tests", "sweep_framework", "sweeps")
OUT_DIR = os.path.join(HERE, "configs", "sweeps")

sys.path.insert(0, HERE)
import calls as calls_mod  # noqa: E402

# Sweep modules import helpers that need a device or torch; the grids themselves are plain data.
STUBBED_MODULES = (
    "tests.ttnn.utils_for_testing",
    "models.common.utility_functions",
    "tests.tt_eager.python_api_testing.sweep_tests.generation_funcs",
    "tests.sweep_framework.sweep_utils.mesh_tensor_utils",
    "tests.sweep_framework.master_config_loader_v2",
    "tests.sweep_framework.sweep_utils.op_kwargs_utils",
    "tests.sweep_framework.sweep_utils.sharding_utils",
    "tests.sweep_framework.sweep_utils.shard_utils",
    "loguru",
)


class Sym:
    """Stands for a ttnn attribute or call; compares and prints by its dotted name."""

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return self.name

    def __eq__(self, other):
        return isinstance(other, Sym) and other.name == self.name

    def __hash__(self):
        return hash(self.name)

    def __getattr__(self, attr):
        if attr.startswith("__"):
            raise AttributeError(attr)
        return Sym(f"{self.name}.{attr}")

    def __call__(self, *args, **kwargs):
        shown = [repr(a) for a in args] + [f"{k}={v!r}" for k, v in kwargs.items()]
        return Sym(f"{self.name}({', '.join(shown)})")


class StubTtnn(types.ModuleType):
    def __getattr__(self, attr):
        if attr.startswith("__"):
            raise AttributeError(attr)
        return Sym(f"ttnn.{attr}")


class StubModule(types.ModuleType):
    __path__ = []

    def __getattr__(self, attr):
        if attr.startswith("__"):
            raise AttributeError(attr)
        value = mock.MagicMock(name=attr)
        setattr(self, attr, value)
        return value


def install_stubs():
    sys.modules["ttnn"] = StubTtnn("ttnn")
    sys.path.insert(0, REPO)
    for name in STUBBED_MODULES:
        sys.modules[name] = StubModule(name)


def load_sweep(relpath):
    path = os.path.join(SWEEPS, relpath)
    spec = importlib.util.spec_from_file_location("sweep_" + re.sub(r"\W", "_", relpath), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dtype_name(value):
    if isinstance(value, (tuple, list)):
        return next(filter(None, (dtype_name(v) for v in value)), None)
    match = re.search(r"ttnn\.(?:DataType\.)?(\w+?)\b", str(value))
    return match.group(1) if match and match.group(1) in calls_mod.DTYPES else None


def layout_name(value):
    text = str(value)
    return "tile" if "TILE" in text else "row_major" if "ROW_MAJOR" in text else None


def memory_name(value):
    """'DRAM', 'L1' or 'SHARDED'; a callable config is classified by its source name."""
    text = str(value)
    if "SHARD" in text.upper():
        return "SHARDED"
    if "L1_MEMORY_CONFIG" in text:
        return "L1"
    if "DRAM_MEMORY_CONFIG" in text:
        return "DRAM"
    if callable(value):
        name = getattr(value, "__name__", "").lower()
        return "SHARDED" if "shard" in name else "L1" if "l1" in name else "DRAM"
    return None


def output_l1(value, key):
    return {key: "L1"} if memory_name(value) == "L1" else {}


def make_call(shape, dtype, layout, memory, args):
    return {
        "shape": [int(d) for d in shape],
        "dtype": dtype_name(dtype),
        "layout": layout_name(layout),
        "memory_config": memory,
        "args": args,
    }


# Each converter maps one sweep vector to (call, input_is_sharded).
def repeat(v):
    return make_call(v["shape"], v["dtype"], v["layout"], "DRAM", {"repeat_dims": list(v["repeat_shape"])}), False


def repeat_pytorch2(v):
    spec = v["repeat_specs"]
    return make_call(spec["shape"], v["dtype"], v["layout"], "DRAM", {"repeat_dims": spec["repeats"]}), False


def repeat_interleave(v):
    shape = list(v["input_shape"])
    if layout_name(v["input_a_layout"]) == "row_major" and shape[-1] % 2 == 1:
        shape[-1] += 1  # the op rejects odd row-major widths, which would leave the call refused everywhere
    memory = memory_name(v["input_a_memory_config"])
    args = {"repeats": v["repeats"], "dim": v["dim"], **output_l1(v["output_memory_config"], "memory_config")}
    return make_call(shape, v["input_a_dtype"], v["input_a_layout"], memory, args), memory == "SHARDED"


def permute(v):
    memory = memory_name(v["input_a_memory_config"])
    args = {"dims": list(v["permute_spec"]), **output_l1(v["output_memory_config"], "memory_config")}
    return make_call(v["input_shape"], v["input_a_dtype"], v["input_a_layout"], memory, args), memory == "SHARDED"


def permute_pytorch2(v):
    spec = v["permute_specs"]
    return make_call(spec["shape"], v["dtype"], v["layout"], "DRAM", {"dims": spec["dims"]}), False


def pad(v):
    args = {"padding": [list(p) for p in v["padding"]], "value": v["value"]}
    return make_call(v["shape"], v["dtype"], v["layout"], "DRAM", args), False


def untilize(v):
    memory = memory_name(v["input_a_memory_config"])
    args = output_l1(v["output_memory_config"], "memory_config")
    return make_call(v["input_shape"], v["input_a_dtype"], v["input_a_layout"], memory, args), memory == "SHARDED"


OPS = {
    "repeat": [
        ("data_movement/repeat/repeat.py", repeat),
        ("data_movement/repeat/repeat_pytorch2.py", repeat_pytorch2),
    ],
    "repeat_interleave": [("data_movement/repeat_interleave/repeat_interleave.py", repeat_interleave)],
    "permute": [
        ("data_movement/permute/permute.py", permute),
        ("data_movement/permute/permute_pytorch2_rm.py", permute_pytorch2),
        ("data_movement/permute/permute_pytorch2_tiled.py", permute_pytorch2),
    ],
    "pad": [("data_movement/pad/pad.py", pad)],
    "untilize": [("untilize.py", untilize)],
}


def vectors(parameters):
    keys = list(parameters)
    for combo in itertools.product(*(parameters[k] for k in keys)):
        yield dict(zip(keys, combo))


def grid_calls(op):
    """Distinct calls of the op's sweep grids, ordered by id so the sample does not depend on grid order."""
    found = {}
    for relpath, convert in OPS[op]:
        module = load_sweep(relpath)
        invalid = getattr(module, "invalidate_vector", None)
        for parameters in module.parameters.values():
            for vector in vectors(parameters):
                if invalid and invalid(vector)[0]:
                    continue
                call, sharded = convert(vector)
                if sharded or call["dtype"] is None or call["layout"] is None:
                    continue
                found[calls_mod.call_id(call)] = call
    return [found[key] for key in sorted(found)]


def write_calls(path, op, calls):
    with open(path, "w") as f:
        f.write('{"schema": "%s", "op": "%s", "calls": [\n' % (calls_mod.SCHEMA, op))
        f.write(",\n".join(json.dumps(c, sort_keys=True) for c in calls))
        f.write("\n]}\n")


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ops", default=",".join(OPS))
    parser.add_argument("--count", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default=OUT_DIR)
    args = parser.parse_args(argv)

    install_stubs()
    os.makedirs(args.out_dir, exist_ok=True)
    for op in args.ops.split(","):
        grid = grid_calls(op)
        sample = random.Random(args.seed).sample(grid, min(args.count, len(grid)))
        write_calls(os.path.join(args.out_dir, f"{op}.json"), op, sorted(sample, key=calls_mod.call_id))
        print(f"{op}: {len(sample)} of {len(grid)} distinct calls")


if __name__ == "__main__":
    main(sys.argv[1:])
