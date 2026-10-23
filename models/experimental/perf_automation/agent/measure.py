"""REMEASURE measurement (PLAN 8.7) — re-profile the edited model on hardware.

measure_runs() is the injectable default (ctx.deps["measure_runner"]); exercised
live, not in unit tests. It reuses the SAME tracy_tool + make_run_profiled path
the Before Loop used for the baseline. TBD(noise-N): returns ONE profile for now
(tracy_tool medians internally); return N separate profiles for true variance.
"""

from __future__ import annotations

import os

from . import gitio


_OP_BUDGET = 60_000  # total profiled op invocations one capture should not exceed


def _coverage_ops_per_step(repo_root, node, case):
    """The op invocations this model's own coverage probe already counted, or None.

    Shared by the OSL bound and the heavy-axis bound so one cache read is spelled one way; a signal
    that cannot be read must never block the profile, so every failure answers None.
    """
    from .probes import _cc_optimize

    try:
        return _cc_optimize("run").coverage_cache_get_ops_per_step(repo_root, node, case, allow_stale=True)
    except Exception:  # noqa: BLE001
        return None


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
    ops_per_step = _coverage_ops_per_step(repo_root, node, case)
    if not ops_per_step or ops_per_step <= 0:
        return None
    if ops_per_step * declared_osl <= _OP_BUDGET:
        return None  # cheap enough already at the declared unit -- leave it alone
    from .layer_depth import MIN_TOKEN_WINDOW

    osl = max(MIN_TOKEN_WINDOW, _OP_BUDGET // ops_per_step)
    # The drain interval comes from the profiler buffer's own size (profiler_drain.capacity_cadence),
    # the rule the per-stage pass already reads at. This was a fixed "4": on a 32-chip WH Galaxy
    # (2026-09-28) reading every chip every 4 ops kept a depth-capped forward running past 1 h 38 m,
    # where the same stages at the buffer-sized interval took 10 minutes with no marker dropped. A
    # model whose ops dispatch more programs than the interval assumes drops markers on repeated
    # reads, and the heal cuts the interval by _HEAL_GROWTH per attempt (probes.choose_marker_drop_remedy)
    # -- nemotron (2026-09-12, above) overflowed at an interval of 32, so a model like it pays one
    # re-profile to get there instead of keeping every model at 4.
    # Only models that trip the budget get it: everyone else keeps the test's own default.
    from .profiler_drain import capacity_cadence

    return str(osl), str(capacity_cadence())


def _read_perf_src(ctx, perf) -> str:
    """The generated perf test's source, or "" when it cannot be read (then nothing is resized)."""
    from pathlib import Path

    try:
        return (Path(gitio.repo_root(ctx.model_root())) / perf).read_text()
    except Exception:  # noqa: BLE001
        return ""


def _declared_stages(model_root) -> tuple:
    """The stage names THE MODEL declares, via the contract's own reader (PIPELINE_STAGES).

    Needed so a per-stage depth/trace knob is recognised as the tool's, not as an invented axis.
    Read from the model rather than listed here: a model that renames a stage keeps working, and
    there is no second copy of the spelling to drift.
    """
    try:
        from .model_contract import declared_stage_names

        return tuple(declared_stage_names(model_root) or ())
    except Exception:  # noqa: BLE001
        return ()


def heavy_axis_scaled(repo_root, node, case, perf_src: str, stages=()):
    """Size THIS model's own heavy axis to the same op budget the OSL bound uses.

    THE PROFILE MUST SAMPLE THE REQUEST IT RANKS AGAINST -- the rule _capacity_scaled_osl's comment
    states for tokens, applied to the axis a non-token model actually scales on. The tool drives
    TT_PERF_OSL_TOKENS and nothing else, so a diffusion/audio/video test's own axis kept the small
    literal the generator gave it: Qwen-Image-Edit profiled 2 of 50 denoise steps, under-counting
    denoise 25x, and 77% of its attempts went at the three encoder stages worth 0.8% of per-image
    time while denoise (98.1%) took 14.5% -- and won 17 of 19, the best rate of any stage.

    Returns (var, value_str) or None. None means "changed nothing", for every case that is not an
    unambiguous win: no invented axis (every LLM), more than one candidate, or no coverage count.

    CONSERVATIVE BY CONSTRUCTION. The cached count is the probe's WHOLE capture at the test's own
    default, not the per-unit cost, so the budget multiple is applied as if EVERY op in it scaled
    with the axis. Nothing in a real model does -- its fixed stages don't repeat -- so the value
    this returns is an under-estimate of what the budget allows. It can only under-sample, never
    overrun the capture the budget exists to bound. The axis is never lowered below the default the
    test chose for itself.

    KNOWN BOUND: the request LENGTH is not discoverable here -- a schedule lives in the model's own
    inputs, under no name the tool can assume -- so this sizes to the op budget, not to the request.
    A model cheap enough per unit can therefore be given more units than its real request has, which
    costs profile time but cannot breach the budget (it is priced as if every op scaled). Sampling
    past the request is wasteful; sampling a fraction of it is what mis-ranks the run, so the bound
    errs on the side the budget already bounds.
    """
    from .perf_test_gen import model_heavy_axis

    axis = model_heavy_axis(perf_src or "", stages)
    if not axis:
        return None
    var, default = axis
    ops_per_step = _coverage_ops_per_step(repo_root, node, case)
    if not ops_per_step or ops_per_step <= 0:
        return None  # withhold rather than invent, exactly as the OSL bound does
    value = max(int(default), int(default) * max(1, _OP_BUDGET // ops_per_step))
    return (var, str(value)) if value > int(default) else None


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
    # THE SAME RULE ON THE AXIS THIS MODEL ACTUALLY SCALES ON. Above drives the token axis the tool
    # owns; a model whose heavy axis is anything else had nobody driving it, so the generated test
    # kept its own literal and the capture sampled a fraction of the request it ranks against. An
    # operator who set the var keeps it: explicit always wins, as it does for the OSL above.
    try:
        _axis = heavy_axis_scaled(
            gitio.repo_root(ctx.model_root()),
            perf,
            case,
            _read_perf_src(ctx, perf),
            _declared_stages(ctx.model_root()),
        )
        if _axis and not os.environ.get(_axis[0]):
            xenv[_axis[0]] = _axis[1]
    except Exception:  # noqa: BLE001 -- sizing is an optimisation; never block the profile
        pass
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
