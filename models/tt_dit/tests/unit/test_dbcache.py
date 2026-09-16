# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Host-only tests for the DBCache decision bookkeeping (no device needed)."""

import pytest

from models.tt_dit.utils.dbcache import DBCacheConfig, DBCacheContext, steps_mask


def _run(ctx: DBCacheContext, diffs: list[float], branch: int = 0) -> list[str]:
    """Drive one branch through `len(diffs)` steps; returns 'C' (cached) / 'x' (computed) per step."""
    out = []
    for diff in diffs:
        ctx.mark_step_begin()
        gate = ctx.gate(branch)
        use_cache = ctx.decide(branch, diff) if gate == "dynamic" else gate == "cache"
        if use_cache:
            ctx.add_cached_step(branch)
            out.append("C")
        else:
            ctx.add_computed_step(branch)
            out.append("x")
    return out


def test_warmup_never_caches():
    ctx = DBCacheContext(DBCacheConfig(max_warmup_steps=3, max_continuous_cached_steps=-1))
    assert _run(ctx, [0.0] * 6) == ["x", "x", "x", "C", "C", "C"]


def test_threshold_gates_dynamic_steps():
    ctx = DBCacheContext(DBCacheConfig(max_warmup_steps=1, residual_diff_threshold=0.1, max_continuous_cached_steps=-1))
    assert _run(ctx, [1.0, 0.05, 0.5, 0.01]) == ["x", "C", "x", "C"]
    assert ctx.branch(0).residual_diffs == {1: 0.05, 2: 0.5, 3: 0.01}


def test_max_continuous_cached_steps_forces_compute():
    ctx = DBCacheContext(DBCacheConfig(max_warmup_steps=1, max_continuous_cached_steps=2))
    # After two consecutive cached steps the third must compute, then caching resumes.
    assert _run(ctx, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) == ["x", "C", "C", "x", "C", "C", "x"]


def test_max_cached_steps_cap():
    ctx = DBCacheContext(DBCacheConfig(max_warmup_steps=1, max_cached_steps=2, max_continuous_cached_steps=-1))
    assert _run(ctx, [1.0, 0.0, 0.0, 0.0, 0.0]) == ["x", "C", "C", "x", "x"]


def test_branches_are_independent_but_share_steps():
    ctx = DBCacheContext(DBCacheConfig(max_warmup_steps=1, max_continuous_cached_steps=-1))
    for cond_diff, uncond_diff in [(1.0, 1.0), (0.0, 1.0), (1.0, 0.0)]:
        ctx.mark_step_begin()
        for branch, diff in ((0, cond_diff), (1, uncond_diff)):
            gate = ctx.gate(branch)
            if gate == "dynamic" and ctx.decide(branch, diff):
                ctx.add_cached_step(branch)
            else:
                ctx.add_computed_step(branch)
    assert ctx.branch(0).cached_steps == [1]
    assert ctx.branch(1).cached_steps == [2]
    assert ctx.executed_steps == 3


def test_cfg_diff_shared_reuses_conditional_decision():
    ctx = DBCacheContext(
        DBCacheConfig(max_warmup_steps=1, cfg_diff_compute_separate=False, max_continuous_cached_steps=-1)
    )
    # Step 0: warmup, both compute.
    ctx.mark_step_begin()
    for b in (0, 1):
        assert ctx.gate(b) == "compute"
        ctx.add_computed_step(b)
    # Step 1: conditional decides to cache -> unconditional follows without measuring a diff.
    ctx.mark_step_begin()
    assert ctx.gate(0) == "dynamic"
    assert ctx.decide(0, 0.0) is True
    ctx.add_cached_step(0)
    assert ctx.gate(1) == "cache"


def test_static_mask_caches_without_diff():
    mask = steps_mask(compute_bins=[2, 1], cache_bins=[1, 2])
    assert mask == (1, 1, 0, 1, 0, 0)
    ctx = DBCacheContext(
        DBCacheConfig(max_warmup_steps=8, steps_computation_mask=mask, steps_computation_policy="static")
    )
    # Warmup is truncated to the leading run of 1s in the mask (2 steps).
    assert ctx.warmup_steps == [0, 1]
    gates = []
    for _ in mask:
        ctx.mark_step_begin()
        g = ctx.gate(0)
        gates.append(g)
        if g == "cache":
            ctx.add_cached_step(0)
        else:
            ctx.add_computed_step(0)
    # Static masked steps cache unconditionally (the continuous-cache cap is not consulted, as in cache-dit).
    assert gates == ["compute", "compute", "cache", "compute", "cache", "cache"]


def test_dynamic_mask_gates_with_threshold():
    ctx = DBCacheContext(
        DBCacheConfig(
            max_warmup_steps=8,
            residual_diff_threshold=0.1,
            steps_computation_mask=(1, 0, 0, 0),
            max_continuous_cached_steps=-1,
        )
    )
    assert _run(ctx, [9.0, 0.05, 0.5, 0.05]) == ["x", "C", "x", "C"]


def test_reset_clears_state():
    ctx = DBCacheContext(DBCacheConfig(max_warmup_steps=1, max_continuous_cached_steps=-1))
    _run(ctx, [1.0, 0.0])
    assert ctx.num_cached_steps() == 1
    ctx.reset()
    assert ctx.executed_steps == 0
    assert ctx.num_cached_steps() == 0
    assert ctx.branch(0).has_fn_buffer is False


def test_config_validation():
    with pytest.raises(ValueError):
        DBCacheConfig(Fn_compute_blocks=0)
    with pytest.raises(ValueError):
        DBCacheConfig(steps_computation_policy="sometimes")
    assert DBCacheConfig(residual_diff_threshold=0.0).enabled is False
    assert DBCacheConfig().replace(residual_diff_threshold=0.2).residual_diff_threshold == 0.2
    assert "F1B0" in DBCacheConfig().strify()
    assert DBCacheConfig(taylorseer_order=2).strify().endswith("_T1O2")
    with pytest.raises(ValueError):
        DBCacheConfig(taylorseer_order=-1)
