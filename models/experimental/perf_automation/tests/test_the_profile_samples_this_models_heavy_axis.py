# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""The profile must sample the request it ranks against -- on THIS model's axis, not just tokens.

measure_runs drove TT_PERF_OSL_TOKENS and nothing else, so a model whose heavy axis is anything
else kept the small literal its generated test was written with. Qwen-Image-Edit profiled 2 of 50
denoise steps: denoise was under-counted 25x, and 77% of the run's attempts went at the three
encoder stages worth 0.8% of per-image time while denoise -- 98.1% of it -- took 14.5% and won 17
of 19, the best rate of any stage.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_DIFFUSION = """
PERF_FLUSH_EVERY = int(os.environ.get("TT_PERF_FLUSH_EVERY", "32"))
PERF_OSL_TOKENS = int(os.environ.get("TT_PERF_OSL_TOKENS", "128"))
_EAGER = min(PERF_OSL_TOKENS, int(os.environ.get("TT_PERF_EAGER_OSL_TOKENS", "8")))
PERF_BATCH = int(os.environ.get("TT_PERF_BATCH", "0"))
PERF_DENOISE_STEPS = max(1, int(os.environ.get("TT_PERF_DENOISE_STEPS", "2")))
"""


def test_the_skeletons_own_knobs_are_not_reported_as_invented():
    """TT_PERF_FLUSH_EVERY and TT_PERF_EAGER_OSL_TOKENS come from the skeleton, so every generated
    test has them; reporting them as the model's invention was noise the docstring had to excuse."""
    from agent.perf_test_gen import invented_workload_vars, skeleton_env_vars

    assert "TT_PERF_FLUSH_EVERY" in skeleton_env_vars()
    assert "TT_PERF_EAGER_OSL_TOKENS" in skeleton_env_vars()
    assert invented_workload_vars(_DIFFUSION) == [("TT_PERF_DENOISE_STEPS", 2)]


def test_the_models_own_axis_is_discovered_not_named():
    """Nothing here types the axis: it is whatever survives after the tool's own knobs."""
    from agent.perf_test_gen import model_heavy_axis

    assert model_heavy_axis(_DIFFUSION) == ("TT_PERF_DENOISE_STEPS", 2)


def test_an_llm_has_no_invented_axis_so_nothing_changes():
    """The tool already drives the token axis; a test that invents none must be left alone."""
    from agent.perf_test_gen import model_heavy_axis

    src = 'PERF_OSL_TOKENS = int(os.environ.get("TT_PERF_OSL_TOKENS", "128"))\n'
    assert model_heavy_axis(src) is None


def test_two_candidates_are_never_guessed_between():
    """No static rule separates a work-shaping knob from an incidental count, so a model with two
    keeps exactly today's behaviour rather than having one picked for it."""
    from agent.perf_test_gen import model_heavy_axis

    src = _DIFFUSION + 'N = int(os.environ.get("TT_PERF_CLIPS", "4"))\n'
    assert model_heavy_axis(src) is None


def test_a_per_stage_knob_is_the_tools_not_the_models():
    """Per-stage depth knobs are derived from the model's own PIPELINE_STAGES and the tool sets
    them, so they must not look like an invented axis and must not create false ambiguity."""
    from agent.layer_depth import stage_layers_var
    from agent.perf_test_gen import model_heavy_axis

    src = _DIFFUSION + '%s = int(os.environ.get("%s", "2"))\n' % (
        stage_layers_var("denoise").replace("TT_PERF_", "PERF_"),
        stage_layers_var("denoise"),
    )
    assert model_heavy_axis(src, stages=("denoise",)) == ("TT_PERF_DENOISE_STEPS", 2)


def test_a_cheap_model_gets_its_axis_raised_toward_the_budget():
    from agent import measure as m

    orig = m._coverage_ops_per_step
    try:
        m._coverage_ops_per_step = lambda *a: 1000  # 60 budget multiples
        assert m.heavy_axis_scaled("r", "n", "c", _DIFFUSION) == ("TT_PERF_DENOISE_STEPS", "120")
    finally:
        m._coverage_ops_per_step = orig


def test_a_model_already_over_the_budget_keeps_its_own_default():
    """Density is why the budget exists. A capture that cannot even be doubled is left exactly as
    the test wrote it -- the axis is never lowered, and never raised past what the budget prices."""
    from agent import measure as m

    orig = m._coverage_ops_per_step
    try:
        m._coverage_ops_per_step = lambda *a: m._OP_BUDGET + 1
        assert m.heavy_axis_scaled("r", "n", "c", _DIFFUSION) is None
    finally:
        m._coverage_ops_per_step = orig


def test_a_dense_model_is_raised_only_as_far_as_the_budget_prices():
    """nemotron's measured 27,577 per capture buys exactly one doubling, not the full schedule."""
    from agent import measure as m

    orig = m._coverage_ops_per_step
    try:
        m._coverage_ops_per_step = lambda *a: 27_577  # 60_000 // 27_577 == 2
        assert m.heavy_axis_scaled("r", "n", "c", _DIFFUSION) == ("TT_PERF_DENOISE_STEPS", "4")
    finally:
        m._coverage_ops_per_step = orig


def test_no_coverage_signal_withholds_rather_than_invents():
    from agent import measure as m

    orig = m._coverage_ops_per_step
    try:
        m._coverage_ops_per_step = lambda *a: None
        assert m.heavy_axis_scaled("r", "n", "c", _DIFFUSION) is None
    finally:
        m._coverage_ops_per_step = orig


def test_the_budget_is_stated_once():
    """The heavy-axis bound and the OSL bound must price a capture the same; two copies of the
    budget is how the two would come to disagree."""
    src = Path(__file__).resolve().parents[1].joinpath("agent/measure.py").read_text()
    assert src.count("_OP_BUDGET = ") == 1


def test_the_baseline_sizes_the_axis_through_the_same_function():
    """A baseline sized differently from the candidates measures a different workload, and every
    delta after it is against the wrong reference -- the rule the capacity bridge already follows."""
    src = Path(__file__).resolve().parents[1].joinpath("agent/before_loop.py").read_text()
    assert "from .measure import heavy_axis_scaled" in src, "the baseline re-implements the sizing"


def test_an_operators_explicit_value_always_wins():
    for src in (
        Path(__file__).resolve().parents[1].joinpath("agent/measure.py").read_text(),
        Path(__file__).resolve().parents[1].joinpath("agent/before_loop.py").read_text(),
    ):
        i = src.rindex("heavy_axis_scaled(")  # the CALL site, not the definition above it
        assert "os.environ.get(_axis[0])" in src[i : i + 900], "an explicitly set axis is overridden"
