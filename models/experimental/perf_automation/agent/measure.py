"""REMEASURE measurement (PLAN 8.7) — re-profile the edited model on hardware.

measure_runs() is the injectable default (ctx.deps["measure_runner"]); exercised
live, not in unit tests. It reuses the SAME tracy_tool + make_run_profiled path
the Before Loop used for the baseline. TBD(noise-N): returns ONE profile for now
(tracy_tool medians internally); return N separate profiles for true variance.
"""

from __future__ import annotations

from . import gitio


def measure_runs(ctx) -> list[dict]:
    from .probes import make_run_profiled
    from .tracy_tool import tracy_tool

    m = ctx.manifest
    perf = m["perf_test_resolved"]["path"]
    case = m["perf_test_resolved"].get("case")
    cfg = m.get("config", {})
    env_facts = m.get("env", {})

    xenv: dict[str, str] = {}
    vd = cfg.get("visible_devices")
    if vd is not None:
        xenv["TT_VISIBLE_DEVICES"] = str(vd)
        xenv["TT_METAL_VISIBLE_DEVICES"] = str(vd)

    factory = make_run_profiled(
        str(gitio.repo_root(ctx.model_root())),
        perf,
        case,
        timeout_s=cfg.get("timeout", 1800),
        extra_env=xenv,
    )
    profile = tracy_tool(
        pcc_path=perf,
        batch_size=cfg.get("batch_size", 1),
        seq_len=cfg.get("seq_len", 0),
        runs=cfg.get("runs", 3),
        profiles_dir=str(ctx.run.profiles_dir),
        start_signpost=cfg.get("start_signpost"),
        end_signpost=cfg.get("end_signpost"),
        arch=env_facts.get("arch"),
        available_cores=env_facts.get("worker_cores", 64),
        run_profiled=factory,
    )
    return [profile]
