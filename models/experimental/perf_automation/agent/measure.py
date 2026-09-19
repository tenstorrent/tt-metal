"""REMEASURE measurement (PLAN 8.7) — re-profile the edited model on hardware.

measure_runs() is the injectable default (ctx.deps["measure_runner"]); exercised
live, not in unit tests. It reuses the SAME tracy_tool + make_run_profiled path
the Before Loop used for the baseline. TBD(noise-N): returns ONE profile for now
(tracy_tool medians internally); return N separate profiles for true variance.
"""

from __future__ import annotations

import os

from . import gitio


def _capacity_scaled_osl(ctx, repo_root, node, case, declared_osl: int):
    """Shrink the declared OSL for THIS profiling capture when the model's own coverage probe
    already measured enough op invocations per decode step that running the full declared OSL would
    flood tracy. Returns (osl_str, flush_every_str) or None when there is no signal or no shrink is
    needed -- the declared OSL is left completely alone for every model that never triggers this.

    THE SIGNAL ALREADY EXISTS, MEASURED ONCE. The coverage probe that sizes TT_PERF_LAYERS already
    runs one full decode step (no tracy) to enumerate op signatures, and its op SEQUENCE -- not just
    the distinct kinds -- is what run.py's coverage cache now also stores. Reading it here costs
    nothing: no second probe, no device time, no new signal invented for this.

    nvidia_nemotron_3_5_lightning_30b_a3b_bf16 (2026-09-12): coverage depth 6 measured 27,577 op
    invocations in ONE decode step (its dense 128-expert MoE block dominates). At the declared
    OSL=128 that is ~3.5M profiled invocations, which produced a 27+ GB tracy_ops_times.csv and
    OOM'd mid-round. Measured on the same hardware, holding coverage depth fixed and varying only
    OSL: OSL=128 never finished (tracy start/end marker imbalance -- the SAME per-core marker-buffer
    overflow TT_PERF_FLUSH_EVERY's periodic drain exists to prevent, just not frequently enough for
    this many op invocations); OSL=6 completed at ~24 GB combined; OSL=2 completed at ~9 GB combined.
    _OP_BUDGET is picked so nemotron's 27,577/step lands at the OSL=2 floor with headroom, while a
    normal model (a few hundred op invocations per step, same order as gemma-3-12b's ~32x-under-count
    incident this OSL convention was built to prevent) clears the budget at its full declared OSL and
    is left untouched.
    """
    from .probes import _cc_optimize

    try:
        ops_per_step = _cc_optimize("run").coverage_cache_get_ops_per_step(repo_root, node, case)
    except Exception:  # noqa: BLE001 -- a signal that cannot be read must not block the profile
        ops_per_step = None
    if not ops_per_step or ops_per_step <= 0:
        return None
    _OP_BUDGET = 60_000  # total profiled op invocations this capture should not exceed
    if ops_per_step * declared_osl <= _OP_BUDGET:
        return None  # cheap enough already at the declared unit -- leave it alone
    from .layer_depth import MIN_TOKEN_WINDOW

    osl = max(MIN_TOKEN_WINDOW, _OP_BUDGET // ops_per_step)
    # The overflow that made OSL=128 fail to even finish was NOT primarily size -- it was
    # TT_PERF_FLUSH_EVERY=32 draining too rarely for this many op invocations between drains.
    # Shrinking only kicks in here, alongside the token cap: a model that never trips the budget
    # keeps the original default untouched.
    return str(osl), "4"


def measure_runs(ctx) -> list[dict]:
    from .probes import make_run_profiled
    from .tracy_tool import profile_model

    m = ctx.manifest
    perf = m["perf_test_resolved"]["path"]
    case = m["perf_test_resolved"].get("case")
    cfg = m.get("config", {})
    env_facts = m.get("env", {})

    xenv: dict[str, str] = {}
    # THE PROFILE MUST SAMPLE THE REQUEST IT RANKS AGAINST. gap_ms is summed over an op's invocations
    # INSIDE the capture, so a window of N decode steps counts every decode op N times while prefill's
    # single pass counts once. Profile 4 steps of a 128-step request and decode is under-counted 32x:
    # on gemma-3-12b-it that put prefill matmuls at the top of the ranking from the first round, and
    # 72 of 138 shaped attempts went to prefill -- work that improves TTFT and cannot move tok/s/u.
    #
    # Nothing set this here, so the generated test fell back to its own literal (4 in every test
    # written before today). The declared OSL is the unit the run reports, so it is the unit the
    # profile measures -- one variable, so the two can never disagree again. A probe that wants a
    # cheaper unit sets TT_PERF_OSL_TOKENS itself (the op-signature probe caps at 1 on purpose; 128
    # there would be pure waste), and the test then reports the unit it actually ran.
    #
    # THE COST IS REAL: 128 decode steps is ~32x the markers and ~32x the eager time of 4, every
    # round. The drain (TT_PERF_FLUSH_EVERY) keeps that safe rather than fast. PERF_MCP_PROFILE_TOKENS
    # buys the old behaviour back for anyone who would rather have a quick, skewed ranking -- and a
    # model whose OWN coverage probe already measured a per-step op count too high for the declared
    # OSL gets the same relief automatically (_capacity_scaled_osl), sized off what that model
    # measured rather than a number typed for one model.
    _explicit_osl = os.environ.get("TT_PERF_OSL_TOKENS") or os.environ.get("PERF_MCP_PROFILE_TOKENS")
    _scaled = None if _explicit_osl else _capacity_scaled_osl(ctx, gitio.repo_root(ctx.model_root()), perf, case, 128)
    xenv["TT_PERF_OSL_TOKENS"] = (
        os.environ.get("TT_PERF_OSL_TOKENS")
        or os.environ.get("PERF_MCP_PROFILE_TOKENS")
        or (_scaled[0] if _scaled else None)
        or "128"
    )
    if _scaled:
        xenv["TT_PERF_FLUSH_EVERY"] = _scaled[1]
    from .mesh_descriptor import apply_scope

    apply_scope(xenv, cfg)

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
