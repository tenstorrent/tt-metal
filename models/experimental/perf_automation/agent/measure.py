"""REMEASURE measurement (PLAN 8.7) — re-profile the edited model on hardware.

measure_runs() is the injectable default (ctx.deps["measure_runner"]); exercised
live, not in unit tests. It reuses the SAME tracy_tool + make_run_profiled path
the Before Loop used for the baseline. TBD(noise-N): returns ONE profile for now
(tracy_tool medians internally); return N separate profiles for true variance.
"""

from __future__ import annotations

import os

from . import gitio


def _coverage_sized_osl(ctx) -> str | None:
    """The declared OSL is too expensive to profile at full width on a model dense enough per layer:
    nemotron-3.5-lightning-30b's per-op tracy capture (6-7 coverage layers x 128 decode steps x its
    attention/mamba/MoE/MLP/norm op mix) produced a 27 GB tracy_ops_times.csv and OOM'd mid-round.

    Reuses layer_depth.depth_in_force -- the tool's one existing answer to "what depth is this build
    capped at", already correct for both single-stack (TT_PERF_LAYERS) and multi-stack
    (TT_PERF_STACK{i}_LAYERS / TT_PERF_<STAGE>_LAYERS) models -- and layer_depth.token_window, the
    same coverage-depth-to-token-count rule _bridge_depth_env already applies elsewhere. No new
    number, no model name: a model whose coverage depth is small profiles with a small window
    instead of always paying for 128; a model with no active depth cap ("all") is left alone, since
    there is no coverage number to size the window from and 128 is what the run declared."""
    from . import layer_depth

    depth = layer_depth.depth_in_force(model_root=ctx.model_root())
    if depth == "all":
        return None
    return str(layer_depth.token_window(depth))


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
    # buys the old behaviour back for anyone who would rather have a quick, skewed ranking -- and
    # below that, a model dense enough per layer gets the same relief automatically, sized off its
    # own coverage depth rather than a number typed for one model.
    xenv["TT_PERF_OSL_TOKENS"] = (
        os.environ.get("TT_PERF_OSL_TOKENS")
        or os.environ.get("PERF_MCP_PROFILE_TOKENS")
        or _coverage_sized_osl(ctx)
        or "128"
    )
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
