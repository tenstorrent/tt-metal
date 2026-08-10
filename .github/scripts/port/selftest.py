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
# One reason for the whole class, not one per shape: it is the routing test's grouping key.
check("the reason does not name the shape", bool(r) and "33" not in r, str(r))
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

# ============== 6. kernel globs land in the glob call, not in target_sources
# Mirrors data_movement/CMakeLists.txt: the op appears both as a glob and as explicit FILES entries.
CMAKE = """include(sources.cmake)

file(
    GLOB_RECURSE kernels
    pad/device/kernels/*.cpp
    tilize/device/kernels/*.cpp
    untilize/device/kernels/*.cpp
    untilize_with_unpadding/device/kernels/*.cpp
)
target_sources(
    ttnn_op_data_movement
    PUBLIC
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES
            ${kernels}
            untilize/device/kernels/dataflow/reader_unary_start_id.cpp
            untilize/device/kernels/dataflow/writer_unary_stick_layout_split_rows_multi_core.cpp
            tilize_with_val_padding/device/kernels/dataflow/reader_unary_pad_multicore_both_dims.cpp
    PRIVATE
        ${TTNN_OP_DATA_MOVEMENT_SRCS}
)
"""

import tempfile  # noqa: E402

for op, expect_beside in (("untilize", True), ("permute", False)):
    with tempfile.TemporaryDirectory() as tmp:
        category = Path(tmp)
        (category / "CMakeLists.txt").write_text(CMAKE)
        (category / "sources.cmake").write_text(f"    {op}/{op}.cpp\n")
        kernels = category / op / "codegen" / "kernels"
        kernels.mkdir(parents=True)
        (kernels / "k.cpp").write_text("")
        (kernels / "k.h").write_text("")

        added = scaffold.register_kernel_globs(category, op, kernels)
        out = (category / "CMakeLists.txt").read_text()
        lines = out.splitlines()
        start, close = scaffold.glob_block(out)
        placed = [i for i, ln in enumerate(lines) if ln.strip().startswith(f"{op}/codegen/kernels/")]

        check(f"[{op}] both extensions added", added == [f"{op}/codegen/kernels/*.cpp", f"{op}/codegen/kernels/*.h"])
        check(f"[{op}] globs land inside the glob call", placed and all(start < i < close for i in placed),
              f"placed={[i + 1 for i in placed]} block={start + 1}..{close + 1}")
        check(f"[{op}] target_sources FILES list untouched",
              out.split("target_sources")[1] == CMAKE.split("target_sources")[1])
        check(f"[{op}] indentation matches its neighbours",
              all(lines[i].startswith("    ") and not lines[i].startswith("     ") for i in placed))
        if expect_beside:
            check(f"[{op}] sits beside the op's own native glob",
                  lines[placed[0] - 1].strip() == f"{op}/device/kernels/*.cpp")
        else:
            check(f"[{op}] with no native glob, goes in last", placed[-1] == close - 1)

        # The check that would have caught the bug: it must object when a glob sits outside the call.
        broken = CMAKE.replace(
            "            untilize/device/kernels/dataflow/reader_unary_start_id.cpp\n",
            f"            untilize/device/kernels/dataflow/reader_unary_start_id.cpp\n    {op}/codegen/kernels/*.cpp\n",
        )
        (category / "CMakeLists.txt").write_text(broken)
        op_dir = category / op
        (op_dir / "codegen").mkdir(parents=True, exist_ok=True)
        for c in scaffold.COMPONENTS:
            for ext in ("hpp", "cpp"):
                (op_dir / "codegen" / f"{op}_codegen_{c}.{ext}").write_text("x")
        (category / "sources.cmake").write_text(
            "".join(f"    {op}/codegen/{op}_codegen_{c}.cpp\n" for c in scaffold.COMPONENTS)
        )
        errors = scaffold.verify(op_dir, op, [])
        check(f"[{op}] verify rejects a glob outside the call",
              any("outside file(GLOB_RECURSE" in e for e in errors), str(errors))

# ====== 7. strata labels agree across the JSON boundary, and are readable
import strata  # noqa: E402

axes, _ = strata.choose_axes(cases, 24)
check("memory_config is an axis", "kwargs.memory_config" in axes, str(axes))

live_labels = [strata.stratum_key(c, axes) for c in cases]
# What gate.py sees: the ledger after ledger.py has written and reloaded it.
reloaded = json.loads(json.dumps({"cases": cases}, indent=2, default=str))["cases"]
json_labels = [strata.stratum_key(c, axes) for c in reloaded]
check("measure-side and gate-side labels agree", live_labels == json_labels,
      f"{live_labels[0]!r} != {json_labels[0]!r}")
check("labels name the constant instead of dumping its repr",
      all("DRAM_MEMORY_CONFIG" in l or "L1_MEMORY_CONFIG" in l for l in live_labels), live_labels[0])
check("labels carry no C++ field dump", not any("memory_layout=" in l for l in live_labels), live_labels[0])

# ============ 8. a leg may span several op codes across cases, but not within one
import gate  # noqa: E402

def attribute(rows, order):
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as fh:
        fh.write("OP CODE,DEVICE KERNEL DURATION [NS]\n")
        for code, ns in rows:
            fh.write(f"{code},{ns}\n")
        name = fh.name
    return gate.attribute_device_rows(Path(name), order)

# untilize's real shape: native is one device op on an aligned case and another on a non-aligned one.
order, rows = [], []
for case_id, native_code in (("c0", "UntilizeDeviceOperation"), ("c1", "UntilizeWithUnpaddingDeviceOperation")):
    for leg, code in (("native", native_code), ("ported", "UntilizeCodegenDeviceOperation")):
        order.append({"case_id": case_id, "leg": f"{leg}:warmup", "rep": -1, "error": None})
        rows.append((code, 100.0))
    for rep in range(2):
        for leg, code in (("native", native_code), ("ported", "UntilizeCodegenDeviceOperation")):
            order.append({"case_id": case_id, "leg": leg, "rep": rep, "error": None})
            rows.append((code, 200.0 if leg == "native" else 150.0))

samples, notes = attribute(rows, order)
check("a leg spanning two op codes across cases is accepted", bool(samples), str(notes))
check("both op codes are reported", any("op codes:" in n and "UntilizeWithUnpadding" in n for n in notes), str(notes))
check("samples are attributed per case and leg",
      sorted(samples) == [("c0", "native"), ("c0", "ported"), ("c1", "native"), ("c1", "ported")])

# A demoted case sends the ported leg to the native op: same code on two legs, still fine.
demoted_rows = [(c if l.startswith("native") or "warmup" in l else "UntilizeDeviceOperation", ns)
                for (c, ns), l in zip(rows, [o["leg"] for o in order])]
samples, notes = attribute(demoted_rows, order)
check("a demoted case sharing native's op code is accepted", bool(samples), str(notes))

# But a slipped positional join shows up as two op codes for one case and leg.
slipped = list(rows)
slipped[4] = ("SomethingElseDeviceOperation", 200.0)
samples, notes = attribute(slipped, order)
check("a slip within one case and leg is still caught", not samples and any("inconclusive" in n for n in notes),
      str(notes))

print()
print(f"{len(failures)} failure(s)" + (": " + ", ".join(failures) if failures else ""))
sys.exit(1 if failures else 0)
