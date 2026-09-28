# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Run the source op's own tests against a derived bring-up op, in place, and keep a per-case baseline.

A fork's ``tests/source.yaml`` names a best-effort selection of the upstream tests of the op it was forked from (or
replaces), and the names to swap while they run:

    upstream_sha: <sha the selection was made at>
    swap:                                   # original name -> fork name, set for the whole pytest session
      ttnn.rms_norm: ttnn.bringup.rms_norm
    tests:                                  # pytest paths, optionally with a -k filter (mixed files)
      - {path: tests/ttnn/unit_tests/operations/fused/test_rms_norm.py}
      - {path: tests/ttnn/nightly/unit_tests/operations/fused/test_layernorm.py, k: "RMSN", why: "RMS cases only"}
      - {path: ..., unskip: "unfeasible on the given hardware"}   # drop collection-time skips with this reason

    python -m models.demos.common.bringup.testing.fork_source --fork rms_norm_ttnn --record   # baseline
    python -m models.demos.common.bringup.testing.fork_source --fork rms_norm_ttnn            # check vs baseline

``--record`` runs the selection twice, on the original op (no swap) and on the fork, and writes
``tests/source_baseline.json`` (per test: the original's and the fork's outcome). A check runs it on the fork and
fails if a test that passed on the fork at baseline no longer does; it lists tests that newly pass. Where the original
passes and the fork fails at baseline, the fork does not cover that case (the report lists them); that is known, not a
regression.

The swap is a pytest plugin (``-p models.demos.common.bringup.testing.fork_source``, env ``BRINGUP_FORK_SWAP``: a
JSON dict): each original attribute (``ttnn.<...>``) is replaced by the fork's object for the session, so tests and the
Python wrappers they call reach the fork without being edited. Enum types can be swapped the same way.

An entry's ``unskip`` (env ``BRINGUP_FORK_UNSKIP``) removes the collection-time skip marks whose reason contains that
text, in both runs. Use it only where a conftest's hardware table is known to be too conservative for this box (the
manifest's ``why:`` says why the cases run here); a real hardware limit then shows as a failure in both runs.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[5]
FORKS = REPO / "ttnn" / "ttnn" / "bringup"
ENV = "BRINGUP_FORK_SWAP"
UNSKIP_ENV = "BRINGUP_FORK_UNSKIP"
_SAVED: list[tuple[object, str, object]] = []


# ---- pytest plugin: swap the original names for the fork's
def _resolve(path: str):
    """'ttnn.experimental.deepseek_prefill.dispatch' -> (parent object, attribute name)."""
    parts = path.split(".")
    obj = importlib.import_module(parts[0])
    for p in parts[1:-1]:
        obj = getattr(obj, p)
    return obj, parts[-1]


_GOLDEN = ("golden_function", "preprocess_golden_function_inputs", "postprocess_golden_function_outputs")


def _with_source_golden(new, old):
    """The source op's tests take their reference from ttnn.get_golden_function(<source op>). A fork has none of its
    own, so the swapped-in object carries the source's golden (the reference the fork must match)."""
    import dataclasses

    fields = {
        f: getattr(old, f) for f in _GOLDEN if getattr(old, f, None) is not None and getattr(new, f, None) is None
    }
    if not fields:
        return new
    if dataclasses.is_dataclass(new):
        return dataclasses.replace(new, **fields)
    for f, v in fields.items():
        setattr(new, f, v)
    return new


def pytest_configure(config):
    swap = json.loads(os.environ.get(ENV) or "{}")
    for orig, new in swap.items():
        parent, name = _resolve(orig)
        nparent, nname = _resolve(new)
        old = getattr(parent, name)
        _SAVED.append((parent, name, old))
        setattr(parent, name, _with_source_golden(getattr(nparent, nname), old))
    if swap:
        print(f"FORK_SWAP: {len(swap)} name(s) point at the fork: {sorted(swap)}")


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    """Drop skip marks whose reason contains BRINGUP_FORK_UNSKIP (after every conftest has added its own)."""
    text = os.environ.get(UNSKIP_ENV)
    if not text:
        return
    n = 0
    for item in items:
        keep = [m for m in item.own_markers if not (m.name == "skip" and text in str(m.kwargs.get("reason", "")))]
        n += len(item.own_markers) - len(keep)
        item.own_markers = keep
    print(f"FORK_UNSKIP: removed {n} skip mark(s) with reason containing {text!r}")


def pytest_unconfigure(config):
    while _SAVED:
        parent, name, value = _SAVED.pop()
        setattr(parent, name, value)


# ---- runner
def manifest(fork: str) -> dict:
    p = FORKS / fork / "tests" / "source.yaml"
    if not p.exists():
        raise SystemExit(f"{p.relative_to(REPO)}: no source-test manifest (see the bringup-fork-op skill)")
    return yaml.safe_load(p.read_text()) or {}


def _outcomes(xml: Path) -> dict[str, str]:
    out = {}
    if not xml.exists():
        return out
    for tc in ET.parse(xml).getroot().iter("testcase"):
        key = f"{tc.get('classname')}::{tc.get('name')}"
        kinds = {c.tag for c in tc}
        out[key] = "failed" if kinds & {"failure", "error"} else "skipped" if "skipped" in kinds else "passed"
    return out


def run(m: dict, swap: bool, tag: str, scratch: Path, only: list[int] | None = None) -> dict[str, str]:
    """Every manifest entry (or the ones in ``only``) through run_safe_pytest, one session per entry, so a -k filter
    applies to its file."""
    env = dict(os.environ)
    if swap:
        env[ENV] = json.dumps(m.get("swap") or {})
    else:
        env.pop(ENV, None)
    results: dict[str, str] = {}
    for i, t in enumerate(m.get("tests") or []):
        if only is not None and i not in only:
            continue
        xml = scratch / f"{tag}_{i}.xml"
        cmd = ["scripts/run_safe_pytest.sh", "--run-all", "--no-precompile", t["path"], f"--junitxml={xml}"]
        # the plugin swaps only when ENV is set; it is loaded in both runs so an entry's unskip applies to both
        cmd += ["-p", "models.demos.common.bringup.testing.fork_source"]
        env.pop(UNSKIP_ENV, None)
        if t.get("unskip"):
            env[UNSKIP_ENV] = t["unskip"]
        if t.get("k"):
            cmd += ["-k", t["k"]]
        subprocess.run(cmd, cwd=REPO, env=env)
        got = _outcomes(xml)
        if not got:
            results[f"{t['path']}::<no results>"] = "failed"
        results.update(got)
    return results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fork", required=True, help="folder under ttnn/ttnn/bringup")
    ap.add_argument("--record", action="store_true", help="run on the original and on the fork, write the baseline")
    ap.add_argument(
        "--entry",
        type=int,
        action="append",
        help="only these manifest entries (0-based; repeatable). With --record their results merge into the baseline",
    )
    a = ap.parse_args(argv)

    m = manifest(a.fork)
    scratch = REPO / "generated" / "bringup_fork_source" / a.fork
    scratch.mkdir(parents=True, exist_ok=True)
    base_path = FORKS / a.fork / "tests" / "source_baseline.json"
    fork = run(m, True, "fork", scratch, a.entry)
    if a.record:
        orig = run(m, False, "original", scratch, a.entry)
        keys = sorted(set(orig) | set(fork))
        base = json.loads(base_path.read_text())["tests"] if (a.entry and base_path.exists()) else {}
        base.update({k: {"original": orig.get(k, "missing"), "fork": fork.get(k, "missing")} for k in keys})
        base_path.write_text(json.dumps({"upstream_sha": m.get("upstream_sha"), "tests": base}, indent=1) + "\n")
        gaps = [k for k, v in base.items() if v["original"] == "passed" and v["fork"] != "passed"]
        n = {s: sum(v["fork"] == s for v in base.values()) for s in ("passed", "failed", "skipped")}
        print(f"baseline {base_path.relative_to(REPO)}: {len(base)} tests; fork {n}")
        print(f"original passes, fork does not ({len(gaps)}): the fork's known gaps")
        for k in gaps[:40]:
            print(f"  {k}: {base[k]['fork']}")
        return 0
    if not base_path.exists():
        raise SystemExit(f"no baseline {base_path.relative_to(REPO)}: run with --record first")
    base = json.loads(base_path.read_text())["tests"]
    if a.entry:  # only the tests of the entries that ran
        base = {k: v for k, v in base.items() if k in fork}
    broke = [k for k, v in base.items() if v["fork"] == "passed" and fork.get(k) != "passed"]
    fixed = [k for k, v in base.items() if v["fork"] != "passed" and fork.get(k) == "passed"]
    print(f"{a.fork}: {len(fork)} tests; regressions vs baseline {len(broke)}; newly passing {len(fixed)}")
    for k in broke:
        print(f"  REGRESSION {k}: {fork.get(k, 'missing')}")
    for k in fixed[:20]:
        print(f"  newly passing {k}")
    return 1 if broke else 0


if __name__ == "__main__":
    sys.exit(main())
