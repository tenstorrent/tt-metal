# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""F49 proof, CPU only: do the framework's per-step swap checks catch what the hand-reviewed swap tests catch?

    python models/demos/common/bringup/dev/f49_mutation_proof.py [--spec S] [--all] [--reviewed F ...] [--out MD]

For each chosen frozen swap test (BLOCK_TYPE / SWAPPED parsed with ast, never imported for this) and each mutation
kind of testing/mutate.py that applies to the last swapped step's output (float kinds for float outputs, idxshift for
integer outputs), with that step mutated (BRINGUP_IMPL=mutate:<kind>, BRINGUP_MUTATE_STEP=<last step>) and every
other swapped step the plain CPU reference, mesh None:
    old       run_swap_test(..., checks=None)     the block-out gate the template had before F49
    new       run_swap_test(..., checks="steps")  the F49 template
    reviewed  the frozen reviewed test's own test_swap(None), for the tests given with --reviewed (default: the four
              Hy4 tests the review agents extended by hand); AssertionError = caught
plus one unmutated reference run of each (BRINGUP_IMPL=reference), which must pass. Requirement: new catches every
mutation a reviewed test catches.

The reference is built once per layer and shared by every run (hooks.reference is wrapped with a cache): each run
still gets a fresh state and context from the golden. A layer that reuses another layer's top-k (Hy4 moe_shared)
cannot run in one block without it; when the hooks lack ``swap_context`` and the reference has ``cfg.topk_source``,
this script supplies the golden source-layer top-k in both contexts, as the reviewed Hy4 moe_shared tests do.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import importlib.util
import io
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO))

from models.demos.common.bringup.core import metrics  # noqa: E402
from models.demos.common.bringup.core.runs import IMPL_ENV  # noqa: E402
from models.demos.common.bringup.core.spec import Spec  # noqa: E402
from models.demos.common.bringup.testing import mutate as MU  # noqa: E402
from models.demos.common.bringup.testing.component import run_swap_test  # noqa: E402
from models.demos.common.bringup.testing.templates import tests_dir  # noqa: E402

DEFAULT_SPEC = REPO / "models/demos/hy4_preview_d_p/bringup/spec.yaml"
HY4_REVIEWED = [
    "test_swap_dense_full_01_attn_hc.py",
    "test_swap_dense_full_02_attn_hc_pre.py",
    "test_swap_moe_full_07_attn_residual.py",
    "test_swap_moe_full_14_moe_combine.py",
]


def literals(path: Path) -> dict:
    out = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                out[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    return out


def swap_tests(spec) -> list[dict]:
    rows = []
    for p in sorted(tests_dir(spec).glob("test_swap_*.py")):
        v = literals(p)
        if "BLOCK_TYPE" in v and "SWAPPED" in v:
            bt, sw = v["BLOCK_TYPE"], v["SWAPPED"]
            rows.append({"id": f"S.{bt}.{len(sw):02d}", "path": p, "block_type": bt, "swapped": sw})
    return rows


def install_caches(spec):
    """One reference per layer set for every run; the golden top-k of a shared-index layer's source layer."""
    hooks = spec.hooks()
    real = hooks.reference
    cache = {}

    def cached(s, layers=None, dtype=None):
        key = (tuple(layers) if layers is not None else None, dtype)
        if key not in cache:
            cache.clear()  # one layer's weights at a time (Hy4 MoE layers are ~39 GB each in fp32)
            cache[key] = real(s, layers=layers, dtype=dtype)
        return cache[key]

    hooks.reference = cached
    if not hasattr(hooks, "swap_context"):

        def swap_context(s, ref, layer, g, c, rctx, dctx):
            cfg = getattr(ref, "cfg", None)
            if cfg is not None and hasattr(cfg, "topk_source") and not cfg.is_full(layer):
                tk = g.layer(c, cfg.topk_source(layer))["topk"]
                rctx.extra["shared_topk"] = dctx.extra["shared_topk"] = tk

        hooks.swap_context = swap_context
        return hooks, True
    return hooks, False


def run(fn, impl: str, step: str | None) -> dict:
    """fn() under BRINGUP_IMPL=impl; caught = returned False or raised AssertionError. stdout kept for the reasons."""
    os.environ[IMPL_ENV] = impl
    if step:
        os.environ[MU.MUTATE_STEP_ENV] = step
    else:
        os.environ.pop(MU.MUTATE_STEP_ENV, None)
    buf, t0 = io.StringIO(), time.time()
    res = {"error": None}
    with contextlib.redirect_stdout(buf):
        try:
            ok = fn()
            res["passed"] = ok is not False
        except AssertionError as e:
            res["passed"] = False
            print(f"FAIL assert: {str(e)[:300]}")
        except Exception as e:  # noqa: BLE001
            res["passed"] = None
            res["error"] = f"{type(e).__name__}: {e}"
            traceback.print_exc(file=buf)
    res["seconds"] = time.time() - t0
    res["fails"] = [m.group(1) for m in re.finditer(r"^FAIL (\S+?):?\s", buf.getvalue(), re.M)]
    res["log"] = buf.getvalue()
    return res


def load_test(path: Path):
    spec_ = importlib.util.spec_from_file_location(f"f49_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(mod)
    return mod


def verdict(r) -> str:
    if r is None:
        return "-"
    if r["passed"] is None:
        return "ERROR"
    return "pass" if r["passed"] else "CAUGHT"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--spec", default=str(DEFAULT_SPEC))
    ap.add_argument("--all", action="store_true", help="every frozen swap test (default: reviewed + last per type)")
    ap.add_argument("--reviewed", nargs="*", default=None, help="reviewed test file names (default: the Hy4 four)")
    ap.add_argument("--only", nargs="*", default=None, help="task ids to run (overrides the selection)")
    ap.add_argument("--kinds", nargs="*", default=list(MU.KINDS + MU.CONTROL_KINDS))
    ap.add_argument("--out", default=str(Path(__file__).with_name("f49_mutation_proof.md")))
    ap.add_argument("--run-old", action="store_true", help="run checks=None separately (default: read its gate line)")
    ap.add_argument("--merge", nargs="*", default=None, help="write --out from these result JSON files and exit")
    ap.add_argument("--log", default=None, help="full stdout of every run (default: next to --out, .log)")
    a = ap.parse_args(argv)

    if a.merge is not None:
        spec = Spec.load(a.spec)
        results = [r for f in a.merge for r in json.loads(Path(f).read_text())["results"]]
        secs = sum(json.loads(Path(f).read_text())["seconds"] for f in a.merge)
        write_md(a, spec, None, results, None, True, secs)
        return 0
    torch.set_num_threads(int(os.environ.get("F49_THREADS") or metrics.cpu_threads()))
    os.environ["BRINGUP_SPEC"] = str(Path(a.spec).resolve())
    os.environ[IMPL_ENV] = "reference"  # the reviewed tests parametrize their mesh at import: no device
    spec = Spec.load(a.spec)
    reviewed_names = HY4_REVIEWED if a.reviewed is None else a.reviewed
    tests = swap_tests(spec)
    last = {}
    for t in sorted(tests, key=lambda t: len(t["swapped"])):
        last[t["block_type"]] = t  # the longest swap order of each block type
    if a.only:
        chosen = [t for t in tests if t["id"] in a.only]
    elif a.all:
        chosen = tests
    else:
        chosen = [t for t in tests if t["path"].name in reviewed_names or t is last[t["block_type"]]]
    chosen.sort(key=lambda t: (list(spec.data["block_types"]).index(t["block_type"]), len(t["swapped"])))
    hooks, shimmed = install_caches(spec)
    log = open(a.log or Path(a.out).with_suffix(".log"), "w")
    results, t_start = [], time.time()
    for t in chosen:
        bt, sw = t["block_type"], t["swapped"]
        step = sw[-1]
        reviewed = load_test(t["path"]).test_swap if t["path"].name in reviewed_names else None
        ref = hooks.reference(spec, layers=[spec.representative_layer(bt)], dtype=torch.float32)
        out_name = next(st.output for st in ref.block_graph(spec.representative_layer(bt)) if st.name == step)
        from models.demos.common.bringup.testing.harness import component_golden

        g, c = component_golden(spec)
        is_float = g.layer(c, spec.representative_layer(bt))[out_name].is_floating_point()
        plan = [("reference", None)]
        plan += [(k, step) for k in a.kinds if k in MU.KINDS and (k in MU.FLOAT_KINDS) == is_float]
        plan += [(k, m) for k in a.kinds if k in MU.CONTROL_KINDS for m in (step, "*")]
        for kind, mstep in plan:
            impl = "reference" if kind == "reference" else f"mutate:{kind}"
            row = {"task": t["id"], "step": "every swapped step" if mstep == "*" else step, "kind": kind}
            row["control"] = kind in MU.CONTROL_KINDS or kind == "reference"
            env_step = None if mstep in (None, "*") else mstep
            row["new"] = run(lambda: run_swap_test(spec, bt, sw, None, None, "steps"), impl, env_step)
            if a.run_old:
                row["old"] = run(lambda: run_swap_test(spec, bt, sw, None, None, None), impl, env_step)
            else:  # checks=None gates exactly this line of the same run (the checks leave the block unchanged)
                m = re.search(r"^(ok  |FAIL) pcc_swap_out:", row["new"]["log"], re.M)
                row["old"] = {
                    "passed": None if m is None else m.group(1) == "ok  ",
                    "error": None,
                    "fails": [],
                    "log": "",
                    "seconds": 0.0,
                }
            row["reviewed"] = run(lambda: reviewed(None), impl, env_step) if reviewed else None
            results.append(row)
            for k in ("old", "new", "reviewed"):
                if row[k]:
                    log.write(
                        f"===== {t['id']} {kind} {row['step']} {k}: {verdict(row[k])} ({row[k]['seconds']:.1f} s)\n"
                    )
                    log.write(row[k]["log"] + "\n")
            log.flush()
            print(
                f"{t['id']:<16} {row['step'][:14]:<14} {kind:<11} old={verdict(row['old']):<6} "
                f"new={verdict(row['new']):<6} reviewed={verdict(row['reviewed']):<6} "
                f"new fails: {','.join(row['new']['fails'][:6])} [{time.time() - t_start:.0f} s]",
                flush=True,
            )
        write_md(a, spec, chosen, results, reviewed_names, shimmed, time.time() - t_start)  # partial results
        dump(a, results)
    write_md(a, spec, chosen, results, reviewed_names, shimmed, time.time() - t_start)
    bad = [r for r in results if r["control"] and any(r[k] and r[k]["passed"] is not True for k in ("old", "new"))]
    bad += [r for r in results if r["kind"] == "reference" and r["reviewed"] and r["reviewed"]["passed"] is not True]
    bad += [
        r
        for r in results
        if not r["control"] and r["reviewed"] and r["reviewed"]["passed"] is False and r["new"]["passed"] is not False
    ]
    print(f"done in {time.time() - t_start:.0f} s; violations: {len(bad)}")
    return 1 if bad else 0


def dump(a, results):
    rows = [
        {k: ({kk: vv for kk, vv in v.items() if kk != "log"} if isinstance(v, dict) else v) for k, v in r.items()}
        for r in results
    ]
    secs = sum(r[k]["seconds"] for r in results for k in ("old", "new", "reviewed") if r.get(k))
    Path(a.out).with_suffix(".json").write_text(json.dumps({"results": rows, "seconds": secs}, indent=1))


def write_md(a, spec, chosen, results, reviewed_names, shimmed, seconds):
    def n(rows, k):
        return sum(1 for r in rows if r[k] and r[k]["passed"] is False)

    muts = [r for r in results if not r["control"]]
    rev = [r for r in muts if r["reviewed"]]
    refs = [r for r in results if r["kind"] == "reference"]
    ctrl = [r for r in results if r["control"] and r["kind"] != "reference"]
    lines = [
        "# F49 mutation proof",
        "",
        f"Generated by `models/demos/common/bringup/dev/f49_mutation_proof.py` (CPU only, mesh None) on "
        f"{time.strftime('%Y-%m-%d %H:%M')}, spec `{Path(a.spec).resolve().relative_to(REPO)}`, golden rung "
        f"`{spec.get('tests.component_rung') or 'first with full_dumps'}`, torch threads {torch.get_num_threads()}, "
        f"{seconds / 60:.0f} min.",
        "",
        "Each row: the frozen swap test's BLOCK_TYPE / SWAPPED with its last swapped step mutated "
        "(`BRINGUP_IMPL=mutate:<kind>`, `BRINGUP_MUTATE_STEP=<step>`), every other swapped step the plain CPU "
        "reference. CAUGHT = the test fails, pass = the mutation slips through. old = `run_swap_test(checks=None)` "
        '(block out PCC only), new = `checks="steps"` (F49), reviewed = the frozen hand-reviewed test\'s own '
        "`test_swap(None)`. `new caught by` lists the new checks that failed (`swap_<step>` = the step's float "
        "limits vs the CPU step on the same inputs, `swap_<step>_out` = the block out with vs without the step's "
        "error, `_vs_cpu` / `_vs_golden` = the component compare mode and threshold).",
        "",
        "Controls must pass old and new: `reference` (no change) and, when run (`--kinds ... bf16`), `bf16` (the step on bf16-rounded inputs, its "
        "output rounded to bf16: a correct step's precision, not the device's; on the last step, and on every "
        "swapped step at once). The step limits are calibrated from the Hy4 device component gates "
        "(`bringup/results/C.<block>.<step>.json`, `pcc_<step>_L<layer>`), as they would be in a run.",
        "",
        f"Selection: {'every frozen swap test' if a.all else 'only ' + ', '.join(a.only) if a.only else 'the reviewed tests plus the last swap test of each block type (a full sweep of all 42 would take several hours)'}.",
    ]
    if shimmed:
        lines.append(
            "The hooks have no `swap_context`; the script supplied the golden source-layer top-k for shared-index "
            "layers (Hy4 moe_shared), as the reviewed moe_shared tests do."
        )
    ran = lambda rows, k: [r for r in rows if r[k]]  # noqa: E731
    passed = lambda rows, k: sum(1 for r in rows if r[k] and r[k]["passed"] is True)  # noqa: E731
    ctrl_line = (
        [
            f"- bf16 controls pass: old {passed(ctrl, 'old')}/{len(ctrl)}, new {passed(ctrl, 'new')}/{len(ctrl)}, "
            f"reviewed {passed(ctrl, 'reviewed')}/{len(ran(ctrl, 'reviewed'))}."
        ]
        if ctrl
        else []
    )
    lines += [
        "",
        "## Summary",
        "",
        f"- Unmutated reference runs pass: old {passed(refs, 'old')}/{len(refs)}, new {passed(refs, 'new')}/"
        f"{len(refs)}, reviewed {passed(refs, 'reviewed')}/{len(ran(refs, 'reviewed'))}.",
        *ctrl_line,
        f"- Mutations: {len(muts)}; old catches {n(muts, 'old')}, new catches {n(muts, 'new')}.",
        f"- On the reviewed tests ({len(rev)} mutations): reviewed catches {n(rev, 'reviewed')}, old {n(rev, 'old')}, "
        f"new {n(rev, 'new')}; caught by reviewed but not by new: "
        f"{sum(1 for r in rev if r['reviewed']['passed'] is False and r['new']['passed'] is not False)}; "
        f"caught by new but not by reviewed: "
        f"{sum(1 for r in rev if r['new']['passed'] is False and r['reviewed']['passed'] is not False)}.",
        f"- Errors (a run raised): {sum(1 for r in results for k in ('old', 'new', 'reviewed') if r[k] and r[k]['passed'] is None)}.",
        "",
        "## Results",
        "",
        "| task | mutated step | kind | old | new | reviewed | new caught by |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        trail = lambda f: f.startswith("pcc_swap_") and f != "pcc_swap_out"  # noqa: E731 (ungated diagnosis lines)
        fails = ", ".join(f for f in r["new"]["fails"] if f != "assert" and not trail(f))
        fails = fails or ("-" if r["new"]["passed"] else "")
        err = f" ({r['new']['error'][:80]})" if r["new"]["error"] else ""
        lines.append(
            f"| {r['task']} | {r['step']} | {r['kind']} | {verdict(r['old'])} | {verdict(r['new'])}{err} | "
            f"{verdict(r['reviewed'])} | {fails} |"
        )
    Path(a.out).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    sys.exit(main())
