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
    from .tracy_tool import profile_model

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
        timeout_s=cfg.get("timeout", 10800),
        extra_env=xenv,
    )
    profile = profile_model(
        perf_test=perf,
        config=cfg,
        env=env_facts,
        profiles_dir=str(ctx.run.profiles_dir),
        run_profiled=factory,
    )
    return [profile]
