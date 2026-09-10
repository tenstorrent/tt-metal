# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""What block stacks does this model have -- answered BEFORE the perf test is written.

THE GENERATOR ASKS AND NOBODY ANSWERED. generate_perf_test takes a `stacks` argument, documents it,
and carries a whole multi-stack branch behind it: given more than one stack it replaces the single
`_pl` / `PERF_LAYERS` lines with one env var per stack so each depth can be capped separately. The
parameter defaults to None and not one production caller passes it -- before_loop.py:602 and
model_files.py:258/272 all omit it -- so that branch has never run outside its own unit test. Every
perf test this tool has ever generated was written as if the model had exactly one stack.

WHAT THAT COSTS, measured on Voxtral-Mini-3B 2026-08-13. The test reads only TT_PERF_LAYERS. The
depth bridge later discovers two stacks and sets TT_PERF_STACK0_LAYERS / TT_PERF_STACK1_LAYERS --
names the already-written test does not read. So one depth reaches every stack, and the run must pick
it with max(): stack0=2, stack2=32, stack3=3 -> 32. The encoder IS 32 deep, so capping to 32 changes
no work, and the tool concluded the depth knob never reached the builder. A correct window was
discarded and the run refused, on a model whose knobs were wired and working.

WHY THIS CAN RUN FIRST. Discovery needed a built model and got one by running the generated perf
test -- so the test had to exist before the walk, and the walk therefore could not inform the test.
That is circular, and the way out is that the PCC gate is not generated: it is supplied by the
operator (--pcc-test), it exists before anything is written, and it builds the model. Walking it
costs one build and answers the question the generator has been asking all along.

The probe is unchanged and node-agnostic -- it runs whatever node it is given -- so this is a
different INPUT to the same machinery, not a second implementation of it.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


LAST_REASON = [""]


class Stack:
    """One discovered stack, in the shape generate_perf_test expects (.path and .count)."""

    __slots__ = ("path", "count")

    def __init__(self, path: str, count: int):
        self.path, self.count = str(path), int(count)

    def __repr__(self) -> str:  # pragma: no cover - debug only
        return "Stack(%r, %d)" % (self.path, self.count)


def stacks_from_census(rows) -> list:
    """The DEVICE stacks a census reports, deepest first.

    Reference stacks are excluded for the same reason the probe excludes them from sizing: they are
    torch modules held for weight loading and never dispatch a ttnn op, so a depth variable aimed at
    one would cap nothing. Deepest first only to make the generated prompt read predictably.
    """
    out = [
        Stack(r.get("path") or "?", int(r.get("blocks") or 0))
        for r in rows or []
        if isinstance(r, dict) and r.get("kind") == "device" and int(r.get("blocks") or 0) >= 2
    ]
    return sorted(out, key=lambda s: (-s.count, s.path))


def model_id_from_source(model_root) -> str:
    """The hub repo id the model's own source names, so nothing has to be configured or passed.

    ONE IMPLEMENTATION, NOT TWO. The contract already extracts this to decide whether a model's
    weights are present, and it scans EVERY source file with module constants resolved -- which is
    what a real model needs: Voxtral calls from_pretrained(HF_REPO_ID) in tt/reference.py while
    HF_REPO_ID is defined in tt/inputs.py and imported. A second, narrower copy that looked only at
    the calling file returned "" and silently disabled the checkpoint witness, sending the survey off
    to build the model instead -- the exact failure it exists to prevent.

    Nothing here imports transformers or touches a network: it reads source text and, later, the
    NAMES inside a weight file. "HF" is only where the file happens to be cached.
    """
    try:
        from .model_contract import Source, _hf_repo_ids

        ids = _hf_repo_ids(Source.load(model_root))
    except Exception:  # noqa: BLE001
        return ""
    # A pipeline commonly names several repos -- weights plus a sample-input dataset. Prefer one that
    # actually has weight keys behind it.
    try:
        from .checkpoint_sections import declared_sections
    except Exception:  # noqa: BLE001
        return next(iter(sorted(ids)), "")
    for rid in sorted(ids):
        if declared_sections(model_root, rid):
            return rid
    return ""


def survey_model(repo_root, model_root, env=None, timeout_s: int = 1800, python_bin=None, model_id: str = "") -> list:
    """Build the model and walk it. No test involved.

    This is the answer to the dependency that broke the test-based survey: running a test and waiting
    for it to call build_pipeline works only for tests that build the model that way, and the
    correctness gate does not. The contract guarantees the factory, so call it.
    """
    repo_root, model_root = Path(repo_root), Path(model_root)
    # ASK THE WEIGHTS FIRST -- IT COSTS NOTHING AND NEEDS NO DEVICE. A repeated block prints its index
    # into every key it owns, so grouping the checkpoint's key names gives one entry per section
    # without building anything. That matters here more than anywhere: this runs BEFORE the knob
    # repair, so the factory has no depth argument yet and `layers=2` is swallowed by **kwargs -- a
    # "shallow" build is a FULL build, which on Voxtral is 30+ minutes to learn a number the weight
    # file states in milliseconds.
    try:
        from .checkpoint_sections import declared_sections as _ckpt_sections

        _ck = _ckpt_sections(model_root, model_id or model_id_from_source(model_root))
        if _ck:
            return [Stack(name, depth) for name, depth in sorted(_ck.items(), key=lambda kv: (-kv[1], kv[0]))]
    except Exception:  # noqa: BLE001 -- an unreadable checkpoint just means we build instead
        pass
    probe = repo_root / "models" / "experimental" / "perf_automation" / "cc_optimize" / "_stack_probe.py"
    if not probe.is_file() or not (model_root / "tt" / "pipeline.py").is_file():
        _why("no build_pipeline to call at %s" % model_root)
        return []
    if python_bin is None:
        _cand = repo_root / "python_env" / "bin" / "python"
        python_bin = str(_cand) if _cand.is_file() else sys.executable
    run_env = dict(os.environ)
    run_env.update(env or {})
    run_env.pop("TT_PERF_LAYERS", None)  # walk at FULL depth: a capped build hides short stacks
    try:
        proc = subprocess.run(
            [str(python_bin), str(probe), str(model_root), str(repo_root)],
            cwd=str(repo_root),
            env=run_env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except Exception as exc:  # noqa: BLE001
        _why("stack probe did not run: %s" % str(exc)[:160])
        return []
    found = stacks_from_census(_census_rows(proc.stdout or ""))
    if not found:
        err = ""
        for line in (proc.stdout or "").splitlines():
            if line.startswith("PERF_STACK_PROBE_ERROR="):
                err = line.split("=", 1)[1]
        _why("build+walk produced no census (rc=%s): %s" % (proc.returncode, err or _tail(proc.stderr)))
    return found


def survey(repo_root, node, env=None, timeout_s: int = 3600, python_bin=None) -> list:
    """Walk the model by running `node`, and return its device stacks. [] when it cannot be walked.

    Deliberately failure-tolerant: an empty answer puts generation exactly where it is today, so a
    model this cannot walk is never worse off than before.
    """
    repo_root = Path(repo_root)
    probe = repo_root / "models" / "experimental" / "perf_automation" / "cc_optimize" / "_op_sig_probe.py"
    if not node or not probe.is_file():
        _why("no node to walk" if not node else "probe script missing")
        return []
    path, _, case = str(node).partition("::")
    # THE NODE MUST BE RESOLVABLE FROM THE DIRECTORY THE PROBE RUNS IN. resolve_pcc_node returns a
    # MODEL-ROOT-relative node ("tests/e2e/test_e2e_pipeline.py::..."), and this runs pytest from the
    # REPO root -- so handing it the relative form gives pytest a path that does not exist, collection
    # fails, no census is printed, and an empty list comes back looking exactly like a model with no
    # stacks. That is precisely what happened on 2026-08-13: the survey reported "no block stacks
    # discovered" while the walk three steps later found two.
    if not Path(path).is_absolute():
        path = str((repo_root / path).resolve())
    if not Path(path).is_file():
        _why("node file not found: %s" % path)
        return []
    # The repo's own interpreter, exactly as cc_optimize.run._python_bin resolves it: the system
    # python has no ttnn, and a probe that dies importing it prints PERF_OP_SIGS=[] and exits 0 --
    # indistinguishable from a model with no stacks.
    if python_bin is None:
        _cand = repo_root / "python_env" / "bin" / "python"
        python_bin = str(_cand) if _cand.is_file() else sys.executable
    cmd = [str(python_bin), str(probe), path]
    if case:
        cmd.append(case)
    run_env = dict(os.environ)
    run_env.update(env or {})
    # The walk needs the model at FULL depth: capping shrinks every stack, and a stack of one element
    # is not a stack -- so a capped build reports structure the model does not have.
    run_env.pop("TT_PERF_LAYERS", None)
    run_env["TT_PERF_OSL_TOKENS"] = "1"
    try:
        proc = subprocess.run(cmd, cwd=str(repo_root), env=run_env, capture_output=True, text=True, timeout=timeout_s)
    except Exception as exc:  # noqa: BLE001 -- an unwalkable model degrades to today's blind generation
        _why("probe did not run: %s" % str(exc)[:160])
        return []
    found = stacks_from_census(_census_rows(proc.stdout or ""))
    if not found:
        # SAY WHY IT IS EMPTY. A survey that cannot walk the model and a model with genuinely no
        # stacks produce the same empty list, and only the first is a defect -- silence here is what
        # let a wrong path read as a finding.
        _why("no census in probe output (rc=%s); last stderr: %s" % (proc.returncode, _tail(proc.stderr)))
    return found


def _why(reason: str) -> None:
    """Record why a survey came back empty, so a failure is never mistaken for a finding."""
    LAST_REASON[0] = str(reason)


def _tail(text, n: int = 240) -> str:
    lines = [ln for ln in (text or "").splitlines() if ln.strip()]
    return lines[-1][:n] if lines else "(none)"


def _census_rows(raw: str) -> list:
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from agent.stack_visibility import parse_census

        return parse_census(raw)
    except Exception:  # noqa: BLE001
        return []


def describe(stacks) -> str:
    """One line for the run log, so a single-stack answer is on record rather than assumed."""
    if not stacks:
        return "no block stacks discovered (%s) -- the perf test gets one depth variable" % (
            LAST_REASON[0] or "no reason recorded"
        )
    return "%d block stack(s): %s" % (len(stacks), ", ".join("%s(%d)" % (s.path, s.count) for s in stacks))


def as_json(stacks) -> str:
    return json.dumps([{"path": s.path, "count": s.count} for s in stacks])
