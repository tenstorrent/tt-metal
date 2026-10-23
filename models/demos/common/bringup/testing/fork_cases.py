# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Do the derived bring-up ops' tests cover every call this model makes? Optionally run those tests.

    python -m models.demos.common.bringup.testing.fork_cases --capture <bringup>/results/fork_calls.json \
        [--spec <spec.yaml> | --model <name>] [--run-tests]

- ``--capture``: the file ``testing/fork_capture.py`` wrote, one entry per distinct call signature (``sig``).
- Test cases: each fork ``ttnn/ttnn/bringup/<fork>/tests/cases.py`` defines ``CASES``, a list of dicts, each with at
  least ``model`` and ``sig``: the model that made the call, and the captured signature the case reproduces.
- Uncovered: a call this model makes that no case of its fork carries (same model, same sig).
- ``--run-tests``: runs the tests of every fork the model uses through ``fork_tests.py`` (each case's mesh on the cards
  of this box that form it), every model's cases included (the top-level tests/test_*.py; a fork's own unit suite in tests/unit/ is not run).
Records forks_used, fork_calls, fork_calls_uncovered and fork_tests_failed (a gate wants 0 of each of the last two)
and prints what is missing.

F139 (owner rules, as F80 per step): a math case's limit is never a number read off the device's own error.
``noise_limit(fn_ref, inputs, dtypes)`` gives it: the torch reference at the op's real input formats (bf16 / bfp8 /
bfp4 rounded on the CPU, fp32 sums) and output dtype vs the same reference in fp64, times calib. A case declares its
``noise`` (the input dtypes and output dtype), not a ``pcc`` / ``rtol``; data-movement cases stay exact
(``exact: True``). ``case_lint`` (``--lint``) refuses a math case with a numeric tolerance and no ``noise``.
"""

from __future__ import annotations

import argparse
import json
import re
import runpy
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
FORKS = REPO / "ttnn" / "ttnn" / "bringup"
BIND = re.compile(r'bind_function<\s*"(\w+)"\s*,\s*"ttnn\.bringup\."\s*>')


def op_to_fork() -> dict[str, str]:
    """ttnn.bringup.<op> -> fork folder: the C++ forks from their nanobind sources, the Python forks from PYTHON_OPS in
    ttnn/ttnn/bringup/__init__.py."""
    import ast

    out = {}
    for f in sorted(FORKS.glob("*/*_nanobind.cpp")):
        for op in BIND.findall(f.read_text()):
            out[f"ttnn.bringup.{op}"] = f.parent.name
    init = FORKS / "__init__.py"
    if init.exists():
        for node in ast.parse(init.read_text()).body:
            if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "PYTHON_OPS" for t in node.targets):
                for name, (folder, _fn) in ast.literal_eval(node.value).items():
                    out[f"ttnn.bringup.{name}"] = folder
    return out


CALIB = 2.0  # as per step (component_checks.COMPONENT_DEFAULTS component_calib) and end to end (e2e.calib)
MANT = {"bf16": 7, "bfp8": 7, "bfp4": 3}
TOLERANCE_KEYS = ("pcc", "rtol", "atol", "rel", "max_rel")


def round_to(t, dtype: str):
    """t as the device holds it: bf16, or a block format (bfp8 / bfp4: one shared exponent per 16 elements along the
    last dim, 7 / 3 mantissa bits; the tile face rows ttnn packs), fp32 kept; a CPU model of the format, in fp32."""
    import torch

    x = t.float()
    if dtype == "fp32":
        return x
    if dtype == "bf16":
        return x.to(torch.bfloat16).float()
    if dtype not in MANT:
        raise ValueError(f"dtype {dtype!r}: one of fp32, bf16, bfp8, bfp4")
    n = x.shape[-1]
    pad = (-n) % 16
    xb = torch.nn.functional.pad(x, (0, pad)).reshape(*x.shape[:-1], -1, 16)
    e = torch.floor(torch.log2(xb.abs().amax(-1, keepdim=True).clamp_min(1e-38)))
    step = torch.pow(2.0, e - MANT[dtype] + 1)
    q = torch.round(xb / step) * step
    return q.reshape(*x.shape[:-1], -1)[..., :n]


def noise_limit(fn_ref, inputs: list, dtypes: list[str], out_dtype: str = "bf16", calib: float = CALIB) -> dict:
    """F139: {"rel": calib x the rel L2 of the reference at the op's real formats (inputs rounded to ``dtypes``, fp32
    arithmetic, the output rounded to ``out_dtype``) vs the reference in fp64; "pcc": 1 - calib^2 x its miss}. A
    case checks the device against the fp64 reference with these (``noise`` in the case: its dtypes)."""
    import torch

    want = fn_ref(*[x.double() if isinstance(x, torch.Tensor) and x.is_floating_point() else x for x in inputs])
    xs = [
        round_to(x, d) if isinstance(x, torch.Tensor) and x.is_floating_point() else x for x, d in zip(inputs, dtypes)
    ]
    real = round_to(fn_ref(*xs).float(), out_dtype).double()
    w = want.double()
    rel = float((real - w).norm() / w.norm().clamp_min(1e-300))
    a, b = (real - real.mean()).flatten(), (w - w.mean()).flatten()
    pcc = float(a @ b / (a.norm() * b.norm()).clamp_min(1e-300))
    return {"rel": calib * rel, "pcc": 1.0 - calib * calib * (1.0 - pcc), "noise_rel": rel}


def case_lint(case: dict) -> list[str]:
    """F139: a math case declares its noise floor's formats (``noise``), never a numeric tolerance of its own."""
    if case.get("exact") or "noise" in case:
        return []
    nums = [k for k in TOLERANCE_KEYS if isinstance(case.get(k), (int, float, dict))]
    return (
        [f"{case.get('id', '?')}: tolerance {nums} with no `noise` (F139: the limit is noise_limit's)"] if nums else []
    )


def cases(fork: str) -> list[dict]:
    p = FORKS / fork / "tests" / "cases.py"
    return list(runpy.run_path(str(p)).get("CASES", [])) if p.exists() else []


def check(capture: Path | list[Path], model: str) -> tuple[dict, dict[str, list[dict]]]:
    calls, seen = [], set()
    for f in capture if isinstance(capture, list) else [capture]:  # e.g. a model's real and scaled runs (F64)
        for c in json.loads(Path(f).read_text())["calls"]:
            if (c["op"], c["sig"]) not in seen:
                seen.add((c["op"], c["sig"]))
                calls.append(c)
    owner = op_to_fork()
    by_fork: dict[str, list[dict]] = {}
    for c in calls:
        fork = owner.get(c["op"])
        if fork is None:
            raise SystemExit(f"{c['op']}: no fork binds this op (INDEX.md)")
        by_fork.setdefault(fork, []).append(c)
    missing: dict[str, list[dict]] = {}
    for fork, cs in by_fork.items():
        have = {k["sig"] for k in cases(fork) if k.get("model") == model}
        gap = [c for c in cs if c["sig"] not in have and c.get("sig_full") not in have]  # F179: old cases keep matching
        if gap:
            missing[fork] = gap
    stats = {
        "forks_used": len(by_fork),
        "fork_calls": len(calls),
        "fork_calls_uncovered": sum(len(v) for v in missing.values()),
    }
    return {"stats": stats, "forks": sorted(by_fork)}, missing


def run_tests(forks: list[str]) -> int:
    """The model-case tests of each fork: the top-level tests/test_*.py (an op's own unit suite in tests/unit/ is not
    part of the pass)."""
    files = {f: sorted((FORKS / f / "tests").glob("test_*.py")) for f in forks}
    missing = sorted(f for f, v in files.items() if not v)
    if missing:
        print(f"forks without tests/test_*.py: {missing}")
        return 1
    if not any(files.values()):
        return 0
    # Grouped by each case's mesh and run on the cards of this box that form it (fork_tests.py / cards.py).
    from models.demos.common.bringup.testing.fork_tests import run

    return run(sorted(files))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capture", required=True, nargs="+", help="one or more capture files (merged by call signature)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--spec")
    g.add_argument("--model")
    ap.add_argument("--run-tests", action="store_true")
    ap.add_argument("--lint", action="store_true", help="F139: also lint this model's cases of the forks it uses")
    a = ap.parse_args(argv)

    spec = None
    if a.model is None:
        from models.demos.common.bringup.reference.golden import load_spec

        spec = load_spec(a.spec)
        model = spec.model_dir.name
    else:
        model = a.model
    res, missing = check([Path(c) for c in a.capture], model)
    stats = res["stats"]
    print(f"{model}: {stats['fork_calls']} distinct call(s) to {stats['forks_used']} fork(s): {res['forks']}")
    for fork, gap in missing.items():
        print(f"  {fork}: {len(gap)} call(s) with no case for {model}:")
        for c in gap:
            print(f"    sig {c['sig']} (x{c['count']}): {c['op']}")
    failed = 0
    if a.lint:
        bad = [e for f in res["forks"] for c in cases(f) if c.get("model") == model for e in case_lint(c)]
        for e in bad:
            print(f"FAIL case lint: {e}")
        stats["fork_cases_lint"] = len(bad)
    if a.run_tests and res["forks"]:
        failed = int(run_tests(res["forks"]) != 0)
    stats["fork_tests_failed"] = failed
    if spec is not None:
        from models.demos.common.bringup.core import metrics

        for k, v in stats.items():
            metrics.record(k, v)
    print(json.dumps(stats))
    return 1 if stats["fork_calls_uncovered"] or failed else 0


if __name__ == "__main__":
    sys.exit(main())
