# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Run forks' model-case tests (ttnn/ttnn/bringup/<fork>/tests/test_*.py) on whatever cards this box has.

Every case in a fork's ``tests/*cases.py`` names the mesh it runs on (``mesh: [rows, cols]``), and the tests ask for
that exact device count (``require_exact_physical_num_devices``), so on a box with more cards they would skip. This
runner collects the tests, groups them by their case's mesh, and runs each group through scripts/run_safe_pytest.sh
with only the cards that form that mesh visible (``cards.py``: probed per box and shape, cached). A test whose case it
cannot find, and a mesh that needs every card, run with every card visible. A TT_VISIBLE_DEVICES set by the caller
wins (every group then runs under it, as before).

F198: every test is accounted for. Per fork it counts passed, failed, errors, skipped (with pytest's reason) and not
run (collected but never reported: a crash or hang mid-group), from a junit report per group; a test file that fails to
import is a collection error, and a fork whose test files collect nothing is one too. Any failed, error, not run or
collection error fails the run; skips are listed, never folded into the passes.

    python -m models.demos.common.bringup.testing.fork_tests --fork rms_norm_ttnn [--fork sdpa ...]
"""

from __future__ import annotations

import argparse
import os
import runpy
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
FORKS = REPO / "ttnn" / "ttnn" / "bringup"


def case_meshes(fork: str) -> dict[str, tuple[int, int]]:
    """Case id -> mesh, over every ``*cases.py`` of the fork's tests."""
    out = {}
    for p in sorted((FORKS / fork / "tests").glob("*cases.py")):
        for c in runpy.run_path(str(p)).get("CASES", []):
            if "id" in c and "mesh" in c:
                out[c["id"]] = tuple(int(x) for x in c["mesh"])
    return out


def collect(files: list[str]) -> tuple[list[str], list[str]]:
    """(node ids, collection errors): a file that fails to import is an error, never dropped."""
    if not files:
        return [], []
    r = subprocess.run(
        # pytest.ini's addopts carry -vvs (a tree, not node ids) and a junit path; keep only the import mode
        [sys.executable, "-m", "pytest", "-o", "addopts=--import-mode=importlib", "--collect-only", "-q", *files],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    lines = r.stdout.splitlines()
    nodes = [ln.strip() for ln in lines if "::" in ln and not ln.startswith(" ")]
    errors = [ln.strip() for ln in lines if ln.startswith("ERROR ")]
    if r.returncode not in (0, 5) and not errors:  # (5: nothing collected)
        errors = [f"pytest --collect-only exit {r.returncode}: {(r.stdout + r.stderr).strip().splitlines()[-1:]}"]
    if not nodes and not errors:
        errors = [f"no tests collected from {files}"]
    return nodes, errors


def mesh_of(nodeid: str, meshes: dict[str, tuple[int, int]]) -> tuple[int, int] | None:
    """The mesh of the case a node id is parametrized with (its id is the last part of the [...] param string)."""
    params = nodeid.rsplit("[", 1)[-1].rstrip("]") if "[" in nodeid else ""
    best = None
    for cid, m in meshes.items():
        if params == cid or params.endswith("-" + cid) or params.endswith(cid):
            if best is None or len(cid) > len(best[0]):
                best = (cid, m)
    return best[1] if best else None


def test_files(fork: str) -> list[str]:
    return [str(p.relative_to(REPO)) for p in sorted((FORKS / fork / "tests").glob("test_*.py"))]


def groups(fork: str) -> tuple[dict[tuple[int, int] | None, list[str]], list[str]]:
    """({mesh: node ids}, collection errors)."""
    meshes = case_meshes(fork)
    nodes, errors = collect(test_files(fork))
    out: dict[tuple[int, int] | None, list[str]] = {}
    for n in nodes:
        out.setdefault(mesh_of(n, meshes), []).append(n)
    return out, errors


def junit_outcomes(path: Path) -> dict[str, tuple[str, str]]:
    """{test name: (outcome, message)} from a junit report; outcome passed / failed / error / skipped."""
    out = {}
    if not path.exists() or path.stat().st_size == 0:
        return out
    for tc in ET.parse(path).getroot().iter("testcase"):
        name = f"{tc.get('classname', '')}::{tc.get('name', '')}"
        kind, msg = "passed", ""
        for child in tc:
            if child.tag in ("failure", "error", "skipped"):
                kind = {"failure": "failed", "error": "error", "skipped": "skipped"}[child.tag]
                msg = (child.get("message") or "").strip().splitlines()[0][:300] if child.get("message") else ""
                if kind != "skipped":
                    break
        prev = out.get(name, ("passed", ""))[0]
        if prev in ("failed", "error") and kind in ("passed", "skipped"):
            continue  # (a teardown error after a pass: the error stands)
        out[name] = (kind, msg)
    return out


def same_test(nodeid: str, junit_key: str) -> bool:
    """A node id (path/test_x.py::[Class::]test[p]) and a junit key (<module.dotted>[.Class]::test[p]) name one test."""
    file_, *rest = nodeid.split("::")
    want_cls = ".".join([Path(file_).stem, *rest[:-1]])
    cls, _, name = junit_key.rpartition("::")
    return name == rest[-1] and (cls == want_cls or cls.endswith("." + want_cls))


def empty_tally() -> dict:
    return {"passed": 0, "failed": 0, "error": 0, "skipped": 0, "not_run": 0, "collect_errors": 0, "problems": []}


def bad(t: dict) -> int:
    """Tests that count against the run: failed, errors, not run, collection errors."""
    return t["failed"] + t["error"] + t["not_run"] + t["collect_errors"]


def run_group(nodes: list[str], env: dict, extra: list[str], runner=None) -> tuple[int, dict[str, tuple[str, str]]]:
    with tempfile.TemporaryDirectory() as d:
        xml = Path(d) / "junit.xml"
        # --no-precompile: the up-front collect pass would run each test's host-side input building twice.
        cmd = ["scripts/run_safe_pytest.sh", "--run-all", "--no-precompile", *nodes, *extra, f"--junitxml={xml}"]
        rc = (runner or (lambda c, e: subprocess.run(c, cwd=REPO, env=e).returncode))(cmd, env)
        return rc, junit_outcomes(xml)


def box_error(rc: int, got: dict[str, tuple[str, str]]) -> bool:
    """A group that failed with nothing passing, failing or skipping: every test errored at setup or none reported
    (a dead Ethernet core, a device left open), not a verdict on any test."""
    return rc != 0 and all(k == "error" for k, _ in got.values())


def tally_group(t: dict, fork: str, nodes: list[str], env: dict, extra: list[str], runner, where: str) -> None:
    """Run one group of node ids and add every test's outcome to the tally ``t`` (a box error: once more)."""
    rc, got = run_group(nodes, env, extra, runner)
    if box_error(rc, got):  # (run_safe_pytest reset the cards after the run: one more try, said out loud)
        print(f"[{fork}] every test errored or none reported (exit {rc}): box error, retrying once", flush=True)
        t["problems"].append(f"box error on {where} (exit {rc}): the group ran twice")
        rc, got = run_group(nodes, env, extra, runner)
    seen = set()
    for n in nodes:
        key = next((k for k in got if same_test(n, k)), None)
        if key is None or key in seen:
            t["not_run"] += 1
            t["problems"].append(f"not run: {n} (group exit {rc})")
            continue
        seen.add(key)
        kind, msg = got[key]
        t[kind] += 1
        if kind != "passed":
            t["problems"].append(f"{kind}: {n}" + (f": {msg}" if msg else ""))


def run(forks: list[str], extra: list[str] = (), runner=None, grouper=None) -> dict[str, dict]:
    """{fork: tally}. ``runner(cmd, env) -> rc`` and ``grouper(fork)`` are the selftests' fakes."""
    from models.demos.common.bringup.testing.cards import visible_devices

    user_cards = os.environ.get("TT_VISIBLE_DEVICES")
    tallies = {}
    for fork in forks:
        t = tallies[fork] = empty_tally()
        by_mesh, errors = (grouper or groups)(fork)
        t["collect_errors"] = len(errors)
        t["problems"] += [f"collection error: {e}" for e in errors]
        for mesh, nodes in by_mesh.items():
            env = dict(os.environ)
            cards = user_cards if user_cards is not None else (visible_devices(mesh) if mesh else None)
            if cards is None:
                env.pop("TT_VISIBLE_DEVICES", None)
            else:
                env["TT_VISIBLE_DEVICES"] = cards
            name = f"{mesh[0]}x{mesh[1]}" if mesh else "no case"
            print(f"[{fork}] {len(nodes)} test(s) on mesh {name}, TT_VISIBLE_DEVICES={cards or 'all'}", flush=True)
            tally_group(t, fork, nodes, env, list(extra), runner, f"mesh {name}")
    return tallies


def report(tallies: dict[str, dict]) -> None:
    for fork, t in tallies.items():
        print(
            f"[{fork}] passed {t['passed']}, failed {t['failed']}, errors {t['error']}, skipped {t['skipped']}, "
            f"not run {t['not_run']}, collection errors {t['collect_errors']}"
        )
        for p in t["problems"]:
            print(f"    {p}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fork", action="append", required=True)
    a, extra = ap.parse_known_args(argv)
    tallies = run(a.fork, extra)
    report(tallies)
    return 1 if any(bad(t) for t in tallies.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
