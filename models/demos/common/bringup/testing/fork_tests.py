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

    python -m models.demos.common.bringup.testing.fork_tests --fork rms_norm_ttnn [--fork sdpa ...]
"""

from __future__ import annotations

import argparse
import os
import runpy
import subprocess
import sys
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


def collect(files: list[str]) -> list[str]:
    r = subprocess.run(
        # pytest.ini's addopts carry -vvs (a tree, not node ids) and a junit path; keep only the import mode
        [sys.executable, "-m", "pytest", "-o", "addopts=--import-mode=importlib", "--collect-only", "-q", *files],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    return [ln.strip() for ln in r.stdout.splitlines() if "::" in ln and not ln.startswith(" ")]


def mesh_of(nodeid: str, meshes: dict[str, tuple[int, int]]) -> tuple[int, int] | None:
    """The mesh of the case a node id is parametrized with (its id is the last part of the [...] param string)."""
    params = nodeid.rsplit("[", 1)[-1].rstrip("]") if "[" in nodeid else ""
    best = None
    for cid, m in meshes.items():
        if params == cid or params.endswith("-" + cid) or params.endswith(cid):
            if best is None or len(cid) > len(best[0]):
                best = (cid, m)
    return best[1] if best else None


def groups(fork: str) -> dict[tuple[int, int] | None, list[str]]:
    files = [str(p.relative_to(REPO)) for p in sorted((FORKS / fork / "tests").glob("test_*.py"))]
    meshes = case_meshes(fork)
    out: dict[tuple[int, int] | None, list[str]] = {}
    for n in collect(files):
        out.setdefault(mesh_of(n, meshes), []).append(n)
    return out


def run(forks: list[str], extra: list[str] = ()) -> int:
    from models.demos.common.bringup.testing.cards import visible_devices

    user_cards = os.environ.get("TT_VISIBLE_DEVICES")
    rc = 0
    for fork in forks:
        for mesh, nodes in groups(fork).items():
            env = dict(os.environ)
            cards = user_cards if user_cards is not None else (visible_devices(mesh) if mesh else None)
            if cards is None:
                env.pop("TT_VISIBLE_DEVICES", None)
            else:
                env["TT_VISIBLE_DEVICES"] = cards
            name = f"{mesh[0]}x{mesh[1]}" if mesh else "no case"
            print(f"[{fork}] {len(nodes)} test(s) on mesh {name}, TT_VISIBLE_DEVICES={cards or 'all'}", flush=True)
            # --no-precompile: the up-front collect pass would run each test's host-side input building twice.
            cmd = ["scripts/run_safe_pytest.sh", "--run-all", "--no-precompile", *nodes, *extra]
            rc |= subprocess.run(cmd, cwd=REPO, env=env).returncode
    return rc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fork", action="append", required=True)
    a, extra = ap.parse_known_args(argv)
    return 1 if run(a.fork, extra) else 0


if __name__ == "__main__":
    sys.exit(main())
