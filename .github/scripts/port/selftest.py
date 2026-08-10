#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Harness self-test: runs on a laptop, with ttnn and the sweep module stubbed.

The pieces covered here are the ones that cannot be exercised on a developer machine any other way --
`ledger.py` imports a sweep module that imports ttnn and torch, so the only way to test its
classification logic without a device is to hand it a fake sweep module and a fake ttnn.

Every check below corresponds to a defect that reached the workflow and was found by reading rather
than by running: a manifest naming several sweep suites raised `TypeError: unhashable type: 'list'`
before anything ran; `port_scope` was declared by a manifest and honoured by nobody, which would have
force-graded a large slice of out-of-scope cases as in-scope; and a case whose kwargs hold a live
ttnn object had those objects flattened to strings by the ledger's JSON, so every `ttnn` call built
from them raised. `pad` has none of those three shapes, which is why it never noticed.

Run it directly: `python3 .github/scripts/port/selftest.py`.
"""

import json
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

failures = []


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  -- {detail}" if detail and not cond else ""))
    if not cond:
        failures.append(name)


# ---------------------------------------------------------------- stub ttnn
class MemoryConfig:
    """Stands in for ttnn's MemoryConfig: unhashable-by-value repr, no source form."""

    def __init__(self, tag):
        self.tag = tag

    def __eq__(self, other):
        return isinstance(other, MemoryConfig) and other.tag == self.tag

    def __repr__(self):
        return f"MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::{self.tag})"


class DataType:
    def __init__(self, n):
        self.n = n

    def __repr__(self):
        return f"DataType.{self.n}"


ttnn = types.ModuleType("ttnn")
ttnn.DRAM_MEMORY_CONFIG = MemoryConfig("DRAM")
ttnn.L1_MEMORY_CONFIG = MemoryConfig("L1")
ttnn.bfloat16 = DataType("BFLOAT16")
ttnn.bfloat8_b = DataType("BFLOAT8_B")
ttnn.TILE_LAYOUT = "Layout.TILE"
ttnn.untilize = lambda *a, **k: None          # callable -> must be skipped as a "constant"
ttnn.MemoryConfig = MemoryConfig              # a type -> must be skipped too
MemoryConfig.__module__ = "ttnn._ttnn.tensor"
DataType.__module__ = "ttnn._ttnn.tensor"
sys.modules["ttnn"] = ttnn

# --------------------------------------------------- stub the sweep module
nightly = {
    "input_shape": [[1, 1, 32, 32], [1, 1, 33, 64], [1, 1, 64, 64], [1, 1, 65, 128]],
    "input_a_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
    "input_a_layout": [ttnn.TILE_LAYOUT],
    "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
}
broaden = {
    # [1,1,32,32] duplicates a nightly point exactly -> must dedupe away.
    "input_shape": [[1, 1, 32, 32], [1, 1, 128, 256]],
    "input_a_dtype": [ttnn.bfloat8_b],
    "input_a_layout": [ttnn.TILE_LAYOUT],
    "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
    "shard_strategy": ["none"],          # a key nightly does not have
}
sweep = types.ModuleType("fake_sweep")
sweep.parameters = {"nightly": nightly, "broaden_suite": broaden}
sweep.invalidate_vector = lambda v: (False, None)
sys.modules["fake_sweep"] = sweep

import ledger  # noqa: E402
import scaffold  # noqa: E402

MANIFEST = {
    "op": "untilize",
    "sweep_module": "fake_sweep",
    "sweep_suite": ["nightly", "broaden_suite"],
    "vector_map": {
        "shape": "input_shape",
        "dtype": "input_a_dtype",
        "layout": "input_a_layout",
        "kwargs": {"memory_config": "output_memory_config"},
    },
    "coverage": {"dtypes": ["bfloat16", "bfloat8_b"], "layouts": ["tile"]},
    "port_scope": {"layouts": ["tile"], "dtypes": ["bfloat16", "bfloat8_b"], "tile_aligned": ["bfloat8_b"]},
}

# ============================================ 1. multi-suite union + dedupe
cases = ledger.build_ledger(MANIFEST)
check("list sweep_suite does not raise", True)

expected = (4 * 2 * 2) + (2 * 1 * 2) - 2  # nightly 16, broaden 4, minus 2 exact duplicates
check("union dedupes overlapping points", len(cases) == expected, f"{len(cases)} != {expected}")

ids = [c["case_id"] for c in cases]
check("case ids stay unique and contiguous", ids == [f"fake_sweep[{i}]" for i in range(len(cases))])
check("suite provenance recorded", {c["suite"] for c in cases} == {"nightly", "broaden_suite"})

repeat = ledger.build_ledger(MANIFEST)
check("expansion is deterministic", [c["case_id"] for c in repeat] == ids)
check(
    "same cases in same order",
    [(c["shape"], c["dtype"], c["suite"]) for c in repeat] == [(c["shape"], c["dtype"], c["suite"]) for c in cases],
)

# =========================================================== 2. port_scope
def scope_of(dtype, shape):
    for c in cases:
        if c["dtype"] == dtype and c["shape"] == shape:
            return c["scope"], c.get("reason")
    return None, None

check("aligned bfloat8_b is in scope", scope_of("bfloat8_b", [1, 1, 32, 32])[0] == "in")
s, r = scope_of("bfloat8_b", [1, 1, 33, 64])
check("non-aligned bfloat8_b is out of scope", s == "out", f"got {s}")
check("and says why", bool(r) and "tile-aligned" in r, str(r))
check("non-aligned bfloat16 stays in scope", scope_of("bfloat16", [1, 1, 33, 64])[0] == "in")
check("aligned bfloat16 stays in scope", scope_of("bfloat16", [1, 1, 64, 64])[0] == "in")

n_out = sum(1 for c in cases if c["scope"] == "out")
check("out-of-scope set is non-empty", n_out > 0, f"{n_out}")

# port_scope absent must narrow nothing (this is what keeps pad unchanged)
bare = dict(MANIFEST)
bare.pop("port_scope")
check("no port_scope -> nothing out of scope", all(c["scope"] == "in" for c in ledger.build_ledger(bare)))

# ============================== 3. kwargs keep live ttnn objects, not strings
kw = cases[0]["kwargs"]["memory_config"]
check("kwargs hold the live object", isinstance(kw, MemoryConfig), type(kw).__name__)
# The reason measure.py re-expands the ledger in process instead of reading the JSON beside it. If
# this ever stops being a string, that indirection can go.
round_tripped = json.loads(json.dumps({"cases": cases}, default=str))["cases"][0]["kwargs"]["memory_config"]
check("ledger JSON still cannot carry a ttnn object", isinstance(round_tripped, str))

# ==================================== 4. emitter renders MemoryConfig kwargs
check("_literal names a ttnn constant", scaffold._literal(ttnn.DRAM_MEMORY_CONFIG) == "ttnn.DRAM_MEMORY_CONFIG")
check("_literal distinguishes the two", scaffold._literal(ttnn.L1_MEMORY_CONFIG) == "ttnn.L1_MEMORY_CONFIG")
check("_literal renders an equal-but-new object", scaffold._literal(MemoryConfig("DRAM")) == "ttnn.DRAM_MEMORY_CONFIG")
try:
    scaffold._literal(MemoryConfig("WEIRD"))
    check("_literal refuses an unnameable ttnn object", False, "no raise")
except TypeError:
    check("_literal refuses an unnameable ttnn object", True)

src = scaffold.render_routing_test("untilize", "data_movement", cases, year=2026)
check("emitted routing test is parseable Python", True)
try:
    compile(src, "test_untilize_codegen_routing.py", "exec")
except SyntaxError as exc:
    check("emitted routing test is parseable Python", False, str(exc))
check("emitted test carries the memory_config", "ttnn.DRAM_MEMORY_CONFIG" in src)
check("emitted test has no object addresses", "0x" not in src and "object at" not in src)
check("emitted test is stable across renders", src == scaffold.render_routing_test("untilize", "data_movement", cases, year=2026))

# ==================================== 5. empty out-of-scope set stays valid
allin = [dict(c, scope="in") for c in cases]
empty = scaffold.render_routing_test("untilize", "data_movement", allin, year=2026)
try:
    compile(empty, "t.py", "exec")
    check("empty out-of-scope set still emits valid Python", True)
except SyntaxError as exc:
    check("empty out-of-scope set still emits valid Python", False, str(exc))

print()
print(f"{len(failures)} failure(s)" + (": " + ", ".join(failures) if failures else ""))
sys.exit(1 if failures else 0)
