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

F256 (owner, 2026-11-06: "the box cases should just check and run on a correct box"): with a box (the spec's
``box.mesh``), a test whose mesh is not the box's is SKIPPED, never run, with the reason "box: needs mesh RxC (this box
RxC)", counted under skipped and under skipped_box. A test's mesh is its case's ``mesh`` (model cases), else its
``mesh_device`` pytest param, read at collection (``collect_needs``: a (rows, cols) tuple; an int N is a 1xN line,
matched by device count). The transpose is the same box (open_mesh_device rotates). A smaller mesh is a box skip too (a
2x2 / 1x4 / 4x2 case on an 8x4 Galaxy fails the fabric handshake with every card visible, and a case of a 2x2 capture is
that box's to run), unless the test declares it runs on a submesh of a bigger box: ``submesh_ok: True`` in its case, or
``BRINGUP_SUBMESH_OK = True`` in a unit test module; then it runs on the cards that form its mesh, as before. A
``mesh_device`` param that is not a tuple or an int is not guessed: the test runs as before and is listed ("box
unknown"). Without a box (no spec) nothing changes.

F198: every test is accounted for. Per fork it counts passed, failed, errors, skipped (with pytest's reason) and not
run (collected but never reported: a crash or hang mid-group), from a junit report per group; a test file that fails to
import is a collection error, and a fork whose test files collect nothing is one too. Any failed, error, not run or
collection error fails the run; skips are listed, never folded into the passes.

    python -m models.demos.common.bringup.testing.fork_tests --fork rms_norm_ttnn [--fork sdpa ...]
"""

from __future__ import annotations

import argparse
import json
import os
import runpy
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
FORKS = REPO / "ttnn" / "ttnn" / "bringup"
ROOT = REPO  # (where this package imports from: the collect plugin's PYTHONPATH, also when a selftest moves REPO)
NEEDS_ENV = "BRINGUP_FORK_NEEDS"  # the collect plugin writes {node id: need} here
SUBMESH = "submesh"  # a need's flag: runs on a submesh of a bigger box (submesh_ok / BRINGUP_SUBMESH_OK)
COUNT = "count"  # a need's flag: an int mesh_device param, N devices in a line, matched by device count
PARAM = "param"  # a group key's flag: the mesh came from the mesh_device param (no case); no box, all cards as before


def box_of(spec) -> tuple[int, int] | None:
    """F256: the box a run is for: the spec's box.mesh (a Spec or a spec path); None without one."""
    if spec is None:
        return None
    if isinstance(spec, (str, Path)):
        from models.demos.common.bringup.core.spec import Spec

        spec = Spec.load(spec)
    mesh = spec.get("box.mesh")
    return (int(mesh[0]), int(mesh[1])) if mesh and len(mesh) == 2 else None


def box_reason(need, box: tuple[int, int] | None) -> str | None:
    """F256: why a test that needs ``need`` ((rows, cols), with an optional SUBMESH / COUNT flag) does not run on
    ``box``; None when it runs: no box, no known need, the box's mesh or its transpose, a declared submesh that fits."""
    if box is None or not need:
        return None
    r, c, *flags = need
    R, C = box
    if (r, c) in ((R, C), (C, R)):
        return None
    if COUNT in flags and r * c == R * C:
        return None
    if SUBMESH in flags and ((r <= R and c <= C) or (r <= C and c <= R)):
        return None
    return f"box: needs mesh {r}x{c} (this box {R}x{C})"


def param_need(item):
    """The mesh a collected test's ``mesh_device`` param asks for (conftest.mesh_device): [rows, cols] (+ SUBMESH when
    its module sets BRINGUP_SUBMESH_OK), [1, N, COUNT] for an int, a "?..." string when it cannot be read, None when
    the test has no such param."""
    cs = getattr(item, "callspec", None)
    if cs is None or "mesh_device" not in cs.params:
        return None
    p = cs.params["mesh_device"]
    is_int = lambda x: isinstance(x, int) and not isinstance(x, bool)  # noqa: E731
    if isinstance(p, tuple) and len(p) == 2 and all(is_int(x) for x in p):
        return [p[0], p[1]] + ([SUBMESH] if getattr(getattr(item, "module", None), "BRINGUP_SUBMESH_OK", False) else [])
    if is_int(p):
        return [1, p, COUNT]
    return f"? mesh_device={p!r}"[:160]


def pytest_collection_finish(session):
    """The collect plugin (``-p models.demos.common.bringup.testing.fork_tests``): every item's need, to NEEDS_ENV."""
    out = os.environ.get(NEEDS_ENV)
    if out:
        Path(out).write_text(json.dumps({it.nodeid: param_need(it) for it in session.items}))


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
    nodes, errors, _ = collect_needs(files)
    return nodes, errors


def collect_needs(files: list[str]) -> tuple[list[str], list[str], dict]:
    """(node ids, collection errors, {node id: need}): the need is what ``param_need`` read off the collected item."""
    if not files:
        return [], [], {}
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(x for x in (str(ROOT), env.get("PYTHONPATH")) if x)
    with tempfile.TemporaryDirectory() as d:
        env[NEEDS_ENV] = str(Path(d) / "needs.json")
        r = subprocess.run(
            # pytest.ini's addopts carry -vvs (a tree, not node ids) and a junit path; keep only the import mode
            [sys.executable, "-m", "pytest", "-o", "addopts=--import-mode=importlib", "--collect-only", "-q"]
            + ["-p", "models.demos.common.bringup.testing.fork_tests", *files],
            cwd=REPO,
            capture_output=True,
            text=True,
            env=env,
        )
        try:
            needs = json.loads(Path(env[NEEDS_ENV]).read_text())
        except (OSError, ValueError):
            needs = {}
    lines = r.stdout.splitlines()
    # F258: a node id is "<path>.py::<name>" (not any line with "::": ttnn's Config dump prints "std::nullopt")
    nodes = [
        ln.strip() for ln in lines if "::" in ln and ln.split("::", 1)[0].endswith(".py") and not ln.startswith(" ")
    ]
    errors = [ln.strip() for ln in lines if ln.startswith("ERROR ")]
    if r.returncode not in (0, 5) and not errors:  # (5: nothing collected)
        errors = [f"pytest --collect-only exit {r.returncode}: {(r.stdout + r.stderr).strip().splitlines()[-1:]}"]
    if not nodes and not errors:
        errors = [f"no tests collected from {files}"]
    return nodes, errors, needs


def case_of(nodeid: str, ids) -> str | None:
    """The case id a node id is parametrized with (the last part of the [...] param string; the longest id wins)."""
    params = nodeid.rsplit("[", 1)[-1].rstrip("]") if "[" in nodeid else ""
    hits = [cid for cid in ids if params == cid or params.endswith("-" + cid) or params.endswith(cid)]
    return max(hits, key=len) if hits else None


def mesh_of(nodeid: str, meshes: dict[str, tuple[int, int]]) -> tuple[int, int] | None:
    """The mesh of the case a node id is parametrized with."""
    cid = case_of(nodeid, meshes)
    return meshes[cid] if cid else None


def test_files(fork: str) -> list[str]:
    return [str(p.relative_to(REPO)) for p in sorted((FORKS / fork / "tests").glob("test_*.py"))]


def case_submesh_ok(fork: str) -> set[str]:
    """F256: ids of the cases that declare ``submesh_ok: True`` (they run on a submesh of a bigger box)."""
    out = set()
    for p in sorted((FORKS / fork / "tests").glob("*cases.py")):
        out |= {c["id"] for c in runpy.run_path(str(p)).get("CASES", []) if c.get("submesh_ok") and "id" in c}
    return out


def groups(fork: str) -> tuple[dict[tuple | None, list[str]], list[str]]:
    """({mesh: node ids}, collection errors). The mesh is the case's ((rows, cols), plus SUBMESH for a case with
    submesh_ok), else the node's mesh_device param (F256: with PARAM, so without a box it runs as before: every card),
    else None (run as before)."""
    meshes = case_meshes(fork)
    ok = case_submesh_ok(fork)
    nodes, errors, needs = collect_needs(test_files(fork))
    out: dict[tuple | None, list[str]] = {}
    for n in nodes:
        cid = case_of(n, meshes)
        if cid is not None:
            mesh = (*meshes[cid], SUBMESH) if cid in ok else meshes[cid]
        else:
            need = needs.get(n)
            if isinstance(need, str):
                print(f"[{fork}] box unknown: {n}: {need}: run as before", flush=True)
            mesh = (*need, PARAM) if isinstance(need, list) else None
        out.setdefault(mesh, []).append(n)
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
    return {
        "passed": 0,
        "failed": 0,
        "error": 0,
        "skipped": 0,
        "skipped_box": 0,  # F256: of the skipped, those for another box (never run)
        "not_run": 0,
        "collect_errors": 0,
        "problems": [],
    }


def skip_for_box(t: dict, nodes: list[str], reason: str) -> None:
    """F256: count tests for another box as skipped (and skipped_box) with the box as the reason; they never run."""
    t["skipped"] += len(nodes)
    t["skipped_box"] = t.get("skipped_box", 0) + len(nodes)
    t["problems"] += [f"skipped: {n}: {reason}" for n in nodes]


def box_filter(t: dict, fork: str, nodes: list[str], needs: dict, box, where: str) -> list[str]:
    """F256, for tests run as one group (a unit file): skip those for another box, list those whose mesh param cannot
    be read (they run as before); returns the node ids to run."""
    run, skip = [], {}
    for n in nodes:
        need = needs.get(n)
        if isinstance(need, str):
            if box is not None:
                t["problems"].append(f"box unknown: {n}: {need}: run as before")
            need = None
        why = box_reason(need, box)
        (skip.setdefault(why, []) if why else run).append(n)
    for why, ns in skip.items():
        print(f"[{fork}] {where}: {len(ns)} test(s) skipped, {why}", flush=True)
        skip_for_box(t, ns, why)
    return run


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


def run(forks: list[str], extra: list[str] = (), runner=None, grouper=None, box=None) -> dict[str, dict]:
    """{fork: tally}. ``runner(cmd, env) -> rc`` and ``grouper(fork)`` are the selftests' fakes. ``box`` (rows, cols):
    F256, a group whose mesh is not the box's is skipped, never run."""
    from models.demos.common.bringup.testing.cards import visible_devices

    user_cards = os.environ.get("TT_VISIBLE_DEVICES")
    tallies = {}
    for fork in forks:
        t = tallies[fork] = empty_tally()
        by_mesh, errors = (grouper or groups)(fork)
        t["collect_errors"] = len(errors)
        t["problems"] += [f"collection error: {e}" for e in errors]
        for mesh, nodes in by_mesh.items():
            name = f"{mesh[0]}x{mesh[1]}" if mesh else "no case"
            why = box_reason(mesh, box)
            if why:
                print(f"[{fork}] {len(nodes)} test(s) on mesh {name}: skipped, {why}", flush=True)
                skip_for_box(t, nodes, why)
                continue
            env = dict(os.environ)
            show = mesh and not (box is None and PARAM in mesh)  # (no box: a param-mesh group sees every card)
            cards = user_cards if user_cards is not None else (visible_devices(tuple(mesh[:2])) if show else None)
            if cards is None:
                env.pop("TT_VISIBLE_DEVICES", None)
            else:
                env["TT_VISIBLE_DEVICES"] = cards
            print(f"[{fork}] {len(nodes)} test(s) on mesh {name}, TT_VISIBLE_DEVICES={cards or 'all'}", flush=True)
            tally_group(t, fork, nodes, env, list(extra), runner, f"mesh {name}")
    return tallies


def report(tallies: dict[str, dict]) -> None:
    for fork, t in tallies.items():
        print(
            f"[{fork}] passed {t['passed']}, failed {t['failed']}, errors {t['error']}, skipped {t['skipped']} "
            f"(for another box {t.get('skipped_box', 0)}), not run {t['not_run']}, "
            f"collection errors {t['collect_errors']}"
        )
        for p in t["problems"]:
            print(f"    {p}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fork", action="append", required=True)
    ap.add_argument("--spec", help="F256: the box (box.mesh) the tests are for; default $BRINGUP_SPEC, else none")
    a, extra = ap.parse_known_args(argv)
    tallies = run(a.fork, extra, box=box_of(a.spec or os.environ.get("BRINGUP_SPEC")))
    report(tallies)
    return 1 if any(bad(t) for t in tallies.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
