# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""F56 proof, CPU only: do the built-in component checks (checks="auto") catch what the hand-reviewed component
tests catch?

    python models/demos/common/bringup/dev/f56_mutation_proof.py [--spec S] [--only C.x.y ...] [--out MD]
    python models/demos/common/bringup/dev/f56_mutation_proof.py --merge a.json b.json --out MD

For each chosen frozen component test (STEP / LAYER parsed with ast) and each mistake, mesh None:
    old       run_component_test(..., checks=None)    the template's gate before F56 (PCC / match vs golden)
    new       run_component_test(..., checks="auto")  the F56 template
    reviewed  the frozen reviewed test's own test_component(None); AssertionError = caught
Mistakes: every kind of testing/mutate.py that applies to the output (BRINGUP_IMPL=mutate:<kind>), plus the known
hard cases below (a wrong norm epsilon, iHC streams swapped at layer 0, a small addend scaled, no SwiGLU clamp, no
router bias, iHC post gates halved), made by patching the module under test. Controls must pass everywhere:
reference, bf16 (mutate:bf16) and, for top-k outputs, each row's positions shuffled (the device returns them
unsorted). Also the freeze sweep (BRINGUP_IMPL=mutations) of each test: PASS = the task would freeze without review.
Requirement: new catches every mistake the reviewed test catches; every control passes new.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import os
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO))

from models.demos.common.bringup.core import metrics  # noqa: E402
from models.demos.common.bringup.core.spec import Spec  # noqa: E402
from models.demos.common.bringup.dev.f49_mutation_proof import (  # noqa: E402
    install_caches,
    literals,
    load_test,
    run,
    verdict,
)
from models.demos.common.bringup.testing import component as COMP  # noqa: E402
from models.demos.common.bringup.testing import mutate as MU  # noqa: E402
from models.demos.common.bringup.testing.component import run_component_test  # noqa: E402
from models.demos.common.bringup.testing.harness import component_golden  # noqa: E402
from models.demos.common.bringup.testing.templates import tests_dir  # noqa: E402

DEFAULT_SPEC = REPO / "models/demos/hy4_preview_d_p/bringup/spec.yaml"
# one per output kind and every known hard case (knowledge/known_issues.md)
DEFAULT_ONLY = [
    "C.dense_full.attn_hc",
    "C.dense_full.attn_hc_pre",
    "C.dense_full.attn_norm",
    "C.dense_full.q_a",
    "C.dense_full.indexer",
    "C.dense_full.attention",
    "C.dense_full.attn_residual",
    "C.moe_full.router",
    "C.moe_full.experts",
    "C.moe_full.moe_combine",
    "C.moe_shared.topk_shared",
]


# ---------------------------------------------------------------- hard cases (Hy4 reference internals)
def _hy4():
    return importlib.import_module("models.demos.hy4_preview_d_p.reference.hy4_ref")


@contextlib.contextmanager
def _setattr(obj, name, value):
    old = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, old)


def _swap_streams(t, n=4):
    h = t.shape[-1] // n
    return torch.cat([t[..., h : 2 * h], t[..., :h], t[..., 2 * h :]], -1)


def hard_cases(step: str) -> dict:
    """name -> make(ref, layer, cpu) -> fn(ctx, *x): a wrong module for this step."""
    hc = {}
    if step in ("attn_norm", "ffn_norm"):

        def eps(ref, layer, cpu):
            with _setattr(ref.cfg, "rms_norm_eps", 1e-6):
                return ref.component(layer, step)  # eps is captured when the step is built

        hc["eps1e-6"] = eps
    if step == "q_a":

        def eps_lat(ref, layer, cpu):
            def fn(ctx, *x):
                with _setattr(_hy4(), "LATENT_NORM_EPS", 1e-5):
                    return cpu(ctx, *x)

            return fn

        hc["eps1e-5"] = eps_lat
    if step in ("attn_hc_pre", "ffn_hc_pre"):
        hc["streamswap"] = lambda ref, layer, cpu: lambda ctx, s, gt: cpu(ctx, _swap_streams(s), gt)
    if step in ("attn_residual", "ffn_residual"):
        hc["addend1.02"] = lambda ref, layer, cpu: lambda ctx, s, gt, y: cpu(ctx, s, gt, y * 1.02)
    if step == "moe_combine":
        hc["addend1.02"] = lambda ref, layer, cpu: lambda ctx, a, b: cpu(ctx, a * 1.02, b)
    if step == "experts":

        def noclamp(ref, layer, cpu):
            def fn(ctx, *x):
                with _setattr(ref.cfg, "swiglu_limit", float("inf")):
                    return ref.component(layer, step)(ctx, *x)

            return fn

        hc["noclamp"] = noclamp
    if step == "router":

        def nobias(ref, layer, cpu):
            def fn(ctx, *x):
                w = ref.w[layer]
                with _setattr(w, "router_bias", torch.zeros_like(w.router_bias)):
                    return ref.component(layer, step)(ctx, *x)

            return fn

        hc["nobias"] = nobias
    if step in ("attn_hc", "ffn_hc"):

        def post_half(ref, layer, cpu):
            def fn(ctx, *x):
                y = cpu(ctx, *x).clone()
                n = y.shape[-1]
                y[..., n // 2 :] *= 0.5
                return y

            return fn

        hc["post_half"] = post_half
    return hc


def shuffle_rows(t):
    g = torch.Generator().manual_seed(0)
    perm = torch.argsort(torch.rand(t.shape, generator=g), dim=-1)
    return torch.gather(t, -1, perm)


@contextlib.contextmanager
def patched_module(step, make, reviewed_mod):
    """module_under_test returns make(ref, layer, cpu) for ``step`` (every caller: the auto path, the reviewed test)."""
    real = COMP.module_under_test

    def fake(s, ref, mesh, layer, name, mode=None):
        if name != step:
            return real(s, ref, mesh, layer, name, "reference")
        fn = make(ref, layer, ref.component(layer, name))
        return lambda rctx, dctx, *x: fn(rctx, *x)

    COMP.module_under_test = fake
    if reviewed_mod is not None and hasattr(reviewed_mod, "module_under_test"):
        reviewed_mod.module_under_test = fake
    try:
        yield
    finally:
        COMP.module_under_test = real
        if reviewed_mod is not None and hasattr(reviewed_mod, "module_under_test"):
            reviewed_mod.module_under_test = real


def component_tests(spec) -> list[dict]:
    rows = []
    for p in sorted(tests_dir(spec).glob("test_c_*.py")):
        v = literals(p)
        if "STEP" in v and "LAYER" in v:
            bt = spec.block_type_of(v["LAYER"])
            rows.append(
                {
                    "id": f"C.{bt}.{v['STEP']}",
                    "path": p,
                    "step": v["STEP"],
                    "layer": v["LAYER"],
                    "compare": v.get("COMPARE"),
                    "thr": v.get("THRESHOLD"),
                }
            )
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--spec", default=str(DEFAULT_SPEC))
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--kinds", nargs="*", default=None, help="standard kinds (default all that apply)")
    ap.add_argument("--no-sweep", action="store_true")
    ap.add_argument("--out", default=str(Path(__file__).with_name("f56_mutation_proof.md")))
    ap.add_argument("--merge", nargs="*", default=None)
    a = ap.parse_args(argv)
    if a.merge is not None:
        res = [r for f in a.merge for r in json.loads(Path(f).read_text())["results"]]
        secs = sum(json.loads(Path(f).read_text())["seconds"] for f in a.merge)
        write_md(a, res, secs)
        return 0
    torch.set_num_threads(int(os.environ.get("F56_THREADS") or metrics.cpu_threads()))
    os.environ["BRINGUP_SPEC"] = str(Path(a.spec).resolve())
    os.environ["BRINGUP_IMPL"] = "reference"
    spec = Spec.load(a.spec)
    hooks, _ = install_caches(spec)
    tests = component_tests(spec)
    only = DEFAULT_ONLY if a.only is None and not a.all else a.only
    chosen = [t for t in tests if only is None or t["id"] in only]
    g, c = component_golden(spec)
    log = open(Path(a.out).with_suffix(".log"), "w")
    results, t0 = [], time.time()
    for t in chosen:
        step, layer = t["step"], t["layer"]
        mod = load_test(t["path"])
        want = g.layer(c, layer)[
            next(
                st.output
                for st in hooks.reference(spec, layers=[layer], dtype=torch.float32).block_graph(layer)
                if st.name == step
            )
        ]
        is_float = want.is_floating_point()
        plan = [("reference", "reference", None)]
        plan += [(k, f"mutate:{k}", None) for k in (a.kinds or MU.KINDS) if (k in MU.FLOAT_KINDS) == is_float]
        plan += [(k, "reference", m) for k, m in hard_cases(step).items()]
        if is_float:
            plan.append(("bf16", "mutate:bf16", None))
        else:
            plan.append(("shuffle", "reference", lambda ref, layer, cpu: lambda ctx, *x: shuffle_rows(cpu(ctx, *x))))
        if not a.no_sweep:
            plan.append(("sweep", "mutations", None))
        for kind, impl, make in plan:
            control = kind in ("reference", "bf16", "shuffle", "sweep")
            row = {"task": t["id"], "kind": kind, "control": control}
            args = (spec, step, layer, None, t["compare"], t["thr"])
            ctx = patched_module(step, make, mod) if make else contextlib.nullcontext()
            with ctx:
                row["new"] = run(
                    lambda: run_component_test(*args, checks="auto"), impl, step if impl.startswith("mutate:") else None
                )
                if kind != "sweep":
                    row["old"] = run(
                        lambda: run_component_test(*args), impl, step if impl.startswith("mutate:") else None
                    )
                    row["reviewed"] = run(
                        lambda: mod.test_component(None), impl, step if impl.startswith("mutate:") else None
                    )
                else:
                    row["old"] = row["reviewed"] = None
            for k in ("old", "new", "reviewed"):
                if row[k]:
                    log.write(f"===== {t['id']} {kind} {k}: {verdict(row[k])} ({row[k]['seconds']:.1f} s)\n")
                    log.write(row[k]["log"] + "\n")
            log.flush()
            results.append(row)
            print(
                f"{t['id']:<26} {kind:<12} old={verdict(row['old']):<6} new={verdict(row['new']):<6} "
                f"reviewed={verdict(row['reviewed']):<6} [{time.time() - t0:.0f} s]",
                flush=True,
            )
        dump(a, results, time.time() - t0)
        write_md(a, results, time.time() - t0)
    bad = violations(results)
    print(f"done in {time.time() - t0:.0f} s; violations: {len(bad)}")
    for r in bad:
        print("  ", r["task"], r["kind"])
    return 1 if bad else 0


def violations(results):
    bad = []
    for r in results:
        if r["kind"] == "sweep":
            continue
        if r["control"]:
            if r["new"]["passed"] is not True or (
                r["reviewed"] and r["reviewed"]["passed"] is not True and r["kind"] == "reference"
            ):
                bad.append(r)
        elif r["reviewed"] and r["reviewed"]["passed"] is False and r["new"]["passed"] is not False:
            bad.append(r)
    return bad


def dump(a, results, secs):
    rows = [
        {k: ({kk: vv for kk, vv in v.items() if kk != "log"} if isinstance(v, dict) else v) for k, v in r.items()}
        for r in results
    ]
    Path(a.out).with_suffix(".json").write_text(json.dumps({"results": rows, "seconds": secs}, indent=1))


def write_md(a, results, seconds):
    muts = [r for r in results if not r["control"]]
    n = lambda rows, k: sum(1 for r in rows if r[k] and r[k]["passed"] is False)  # noqa: E731
    ctrl = [r for r in results if r["control"] and r["kind"] != "sweep"]
    sweeps = [r for r in results if r["kind"] == "sweep"]
    miss = [r for r in muts if r["reviewed"] and r["reviewed"]["passed"] is False and r["new"]["passed"] is not False]
    extra = [r for r in muts if r["new"]["passed"] is False and r["reviewed"] and r["reviewed"]["passed"] is not False]
    lines = [
        "# F56 mutation proof",
        "",
        f"Generated by `models/demos/common/bringup/dev/f56_mutation_proof.py` (CPU only, mesh None) on "
        f"{time.strftime('%Y-%m-%d %H:%M')}, {seconds / 60:.0f} min of runs.",
        "",
        "Each row: a frozen hand-reviewed Hy4 component test's STEP / LAYER with the module under test replaced by a "
        "mistake. CAUGHT = the test fails, pass = the mistake slips through. old = `run_component_test(checks=None)` "
        '(the template\'s PCC / match gate), new = `checks="auto"` (F56), reviewed = the frozen reviewed test. '
        "Standard kinds come from `testing/mutate.py`; hard cases patch the Hy4 reference (eps, streamswap, "
        "addend1.02, noclamp, nobias, post_half). Controls (reference, bf16, shuffle) must pass. `sweep` = the freeze "
        "sweep (`BRINGUP_IMPL=mutations`): pass = the task freezes without a review.",
        "",
        "## Summary",
        "",
        f"- Controls: new passes {sum(1 for r in ctrl if r['new']['passed'] is True)}/{len(ctrl)}, reviewed "
        f"{sum(1 for r in ctrl if r['reviewed'] and r['reviewed']['passed'] is True)}/{len(ctrl)}, old "
        f"{sum(1 for r in ctrl if r['old'] and r['old']['passed'] is True)}/{len(ctrl)}.",
        f"- Mistakes: {len(muts)}; reviewed catches {n(muts, 'reviewed')}, old {n(muts, 'old')}, new {n(muts, 'new')}.",
        f"- Caught by reviewed but not by new: {len(miss)}"
        + (f" ({', '.join(r['task'] + ' ' + r['kind'] for r in miss)})" if miss else "")
        + f"; caught by new but not by reviewed: {len(extra)}.",
        f"- Freeze sweeps pass (freeze without review): {sum(1 for r in sweeps if r['new']['passed'] is True)}"
        f"/{len(sweeps)}"
        + (f"; to review: {', '.join(r['task'] for r in sweeps if r['new']['passed'] is not True)}" if sweeps else ""),
        f"- Errors (a run raised): {sum(1 for r in results for k in ('old', 'new', 'reviewed') if r[k] and r[k]['passed'] is None)}.",
        "",
        "## Results",
        "",
        "| task | mistake | old | new | reviewed | new caught by |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        fails = ", ".join(f for f in r["new"]["fails"] if f != "assert")[:200] or ("-" if r["new"]["passed"] else "")
        err = f" ({r['new']['error'][:80]})" if r["new"].get("error") else ""
        lines.append(
            f"| {r['task']} | {r['kind']} | {verdict(r['old'])} | {verdict(r['new'])}{err} | "
            f"{verdict(r['reviewed'])} | {fails} |"
        )
    Path(a.out).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    sys.exit(main())
