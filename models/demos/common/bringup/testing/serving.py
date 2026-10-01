# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The serving contract a bring-up builds to (agents/serving-contract.md writes it at the start, before the plan):

    <bringup dir>/serving_contract.md    the how-to for the bring-up engineer, one `## <section>` per part of the model
    <bringup dir>/contract_tests.yaml    tests: [{test, checks, section, gates, kind?}], gates = a component step or
                                         "adapter"; kind: runner_smoke marks the runner smoke (records smoke_runner_ok)

The orchestrator puts a step's how-to section in its brief, ledger_gen adds a step's contract tests to its gate, and
`python -m models.demos.common.bringup.testing.serving` is the SC.1 gate (the files exist and agree)."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

SECTIONS = ("Input", "KV cache", "Attention and cache writes", "Acks", "Adapter and table", "Deployment")
OWNER = "Questions for the owner"
ADAPTER = "adapter"


def paths(spec) -> tuple[Path, Path]:
    b = Path(spec.bringup_dir)
    return b / "serving_contract.md", b / "contract_tests.yaml"


def sections(spec) -> dict[str, str]:
    """`## <name>` -> its text (empty if the how-to does not exist yet)."""
    md, _ = paths(spec)
    if not md.exists():
        return {}
    out, name = {}, None
    for line in md.read_text().splitlines():
        m = re.match(r"^##\s+(.+?)\s*$", line)
        if m:
            name = m.group(1)
            out[name] = ""
        elif name is not None:
            out[name] += line + "\n"
    return {k: v.strip() for k, v in out.items()}


def tests(spec) -> list[dict]:
    _, y = paths(spec)
    if not y.exists():
        return []
    return list((yaml.safe_load(y.read_text()) or {}).get("tests") or [])


def tests_for(spec, gate: str) -> list[dict]:
    return [t for t in tests(spec) if t.get("gates") == gate]


RUNNER_SMOKE = "runner_smoke"


def runner_smoke(spec, gate: str | None = None) -> dict | None:
    """The contract test marked ``kind: runner_smoke`` (for this gate, if given), or None."""
    return next(
        (t for t in tests(spec) if t.get("kind") == RUNNER_SMOKE and (gate is None or t.get("gates") == gate)), None
    )


def brief_text(spec, gate: str) -> str:
    """The how-to sections and contract tests for one gate (a component step name, or "adapter")."""
    ts = tests_for(spec, gate)
    if not ts:
        return ""
    secs = sections(spec)
    names = list(dict.fromkeys(t.get("section") for t in ts if t.get("section")))
    out = [
        "## Serving contract for this step",
        "",
        "Build it this way from the start; these frozen tests are part of"
        " your gate (from `serving_contract.md`, which the serving-contract agent wrote from tt-d-gen's code).",
        "",
    ]
    for n in names:
        out += [f"### {n}", secs.get(n, "(section missing in serving_contract.md)"), ""]
    out += ["Contract tests in your gate:"] + [f"- `{t['test']}`: {t.get('checks', '')}" for t in ts] + [""]
    return "\n".join(out)


def check(spec, repo: Path) -> list[str]:
    """Problems with the serving contract files (empty = the SC.1 gate passes)."""
    md, y = paths(spec)
    errs = []
    if not md.exists():
        return [f"missing {md}"]
    if not y.exists():
        return [f"missing {y}"]
    text = md.read_text()
    secs = sections(spec)
    for s in SECTIONS + (OWNER,):
        if s not in secs or not secs[s]:
            errs.append(f"serving_contract.md: section '## {s}' missing or empty")
    if not re.search(r"tt-d-gen\b.*@\s*[0-9a-f]{7,40}", text):
        errs.append("serving_contract.md: no 'tt-d-gen @ <sha>' line naming the commit it was read from")
    ts = tests(spec)
    if not ts:
        errs.append("contract_tests.yaml: no tests")
    if not any(t.get("gates") == ADAPTER for t in ts):
        errs.append("contract_tests.yaml: no runner test (gates: adapter)")
    for t in ts:
        p = repo / str(t.get("test", "")).split("::")[0]
        if not t.get("test") or not p.exists():
            errs.append(f"contract_tests.yaml: test file missing: {t.get('test')}")
        if t.get("section") not in secs:
            errs.append(
                f"contract_tests.yaml: {t.get('test')}: section '{t.get('section')}' not in serving_contract.md"
            )
        if not t.get("gates"):
            errs.append(f"contract_tests.yaml: {t.get('test')}: no 'gates'")
        if t.get("kind") not in (None, RUNNER_SMOKE):
            errs.append(f"contract_tests.yaml: {t.get('test')}: unknown kind '{t.get('kind')}' (known: {RUNNER_SMOKE})")
    return errs


def run_tests(spec, repo: Path, gate: str = "all", runner=None) -> list[str]:
    """Run the contract tests for one gate (or all of them) with the safe runner; the ones that failed. For a model
    whose steps were built before its serving contract existed: one contract step runs them all."""
    import subprocess

    def default(cmd):
        return subprocess.run(cmd, cwd=repo, shell=True).returncode

    runner = runner or default
    failed = []
    for t in tests(spec):
        if gate != "all" and t.get("gates") != gate:
            continue
        rc = runner(f"scripts/run_safe_pytest.sh --no-precompile --run-all {t['test']}")
        print(f"{'PASS' if rc == 0 else 'FAIL'} {t['test']} ({t.get('checks', '')})", flush=True)
        if rc != 0:
            failed.append(t["test"])
    return failed


def main(argv=None) -> int:
    import argparse

    from models.demos.common.bringup.core import metrics
    from models.demos.common.bringup.reference.golden import load_spec

    ap = argparse.ArgumentParser()
    ap.add_argument("--spec")
    ap.add_argument("--run", metavar="GATE", help="run the contract tests of GATE (a step name, adapter, or all)")
    a = ap.parse_args(argv)
    spec = load_spec(a.spec)
    if a.run:
        failed = run_tests(spec, Path(spec.repo), a.run)
        ran = [t["test"] for t in tests(spec) if a.run == "all" or t.get("gates") == a.run]
        metrics.record_json("contract", {"run": a.run, "results": {t: t not in failed for t in ran}})
        metrics.record("contract_tests_run", len([t for t in tests(spec) if a.run == "all" or t.get("gates") == a.run]))
        metrics.record("contract_tests_failed", len(failed))
        return 0 if not failed else 1
    errs = check(spec, Path(spec.repo))
    for e in errs:
        print(f"FAIL {e}")
    metrics.record("serving_contract_errors", len(errs))
    metrics.record("serving_contract_tests", len(tests(spec)))
    print(f"serving contract: {len(tests(spec))} tests, {len(errs)} problems")
    return 0 if not errs else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
