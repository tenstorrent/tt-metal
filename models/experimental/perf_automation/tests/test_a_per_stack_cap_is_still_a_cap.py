# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Every reader of depth looked at ONE variable, and a multi-stack model is capped by others.

read_depth() reads TT_PERF_LAYERS. A model with two towers is capped per stack --
TT_PERF_STACK0_LAYERS, TT_PERF_STACK1_LAYERS, and the per-stage TT_PERF_<STAGE>_LAYERS the
generated test binds -- while TT_PERF_LAYERS stays unset. So every reader concluded "all layers" for
a build that had two of thirty.

RUN 11, 2026-08-19, is the measurement. The run's own log shows the caps in force:

    coverage-sized profiling window (multi-stack): TT_PERF_STACK0_LAYERS=2, TT_PERF_STACK1_LAYERS=2

and the census, added the same day to refuse exactly this, reported depth=all and pinned 1.247 B
parameters of a 4.676 B model. Its sections say what it actually saw: `rest`, which holds layers
3..30, came back at 114 MB -- about two layers.

The refusal was correct and blind. It asked the one variable that was not set.

Matched on the SHAPE of the name rather than a list: the tool derives these per stage and per stack,
so a list would need extending for every stage a model declares that nobody anticipated.
"""

import sys
from pathlib import Path

_PA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PA))


def _clear(monkeypatch):
    import os

    for k in list(os.environ):
        if k.startswith("TT_PERF_") and k.endswith("LAYERS"):
            monkeypatch.delenv(k, raising=False)


def test_a_per_stack_cap_is_seen(monkeypatch):
    """HOW MANY STACKS TO LOOK FOR IS THE MODEL'S ANSWER, not a bound. This probed range(8) -- a
    number I picked -- which would miss the ninth cap on a model with nine stacks and ask after
    seven names that cannot exist on a model with one. declared_sections counts the repeated-block
    stacks off the checkpoint's own tensor names."""
    import agent.layer_depth as LD

    _clear(monkeypatch)
    monkeypatch.setenv("TT_PERF_STACK0_LAYERS", "2")
    monkeypatch.setenv("TT_PERF_STACK1_LAYERS", "2")
    monkeypatch.setattr(LD, "_declared_stack_count", lambda root: 2)

    assert LD.active_depth_caps(model_root="/anything") == {
        "TT_PERF_STACK0_LAYERS": 2,
        "TT_PERF_STACK1_LAYERS": 2,
    }


def test_the_stack_count_comes_from_the_model(monkeypatch):
    import agent.layer_depth as LD

    _clear(monkeypatch)
    monkeypatch.setenv("TT_PERF_STACK8_LAYERS", "2")
    monkeypatch.setattr(LD, "_declared_stack_count", lambda root: 9)
    assert LD.active_depth_caps(model_root="/anything") == {
        "TT_PERF_STACK8_LAYERS": 2
    }, "a ninth stack's cap is missed by a fixed bound"

    monkeypatch.setattr(LD, "_declared_stack_count", lambda root: 1)
    assert LD.active_depth_caps(model_root="/anything") == {}


def test_a_per_stage_cap_is_seen_when_the_model_declares_that_stage(monkeypatch):
    """The names come FROM THE MODEL. test_one_depth_vocabulary states the rule the repair, the
    generator and the bridge already share -- the knob for stage X is TT_PERF_X_LAYERS -- and
    stack_knob_repair.stage_names() reads PIPELINE_STAGES from the model's source. So a stage nobody
    anticipated is covered by asking, not by a pattern over the environment."""
    from agent.layer_depth import active_depth_caps

    _clear(monkeypatch)
    monkeypatch.setenv("TT_PERF_VOCODE_LAYERS", "4")

    assert active_depth_caps(stages=["vocode"]) == {"TT_PERF_VOCODE_LAYERS": 4}
    assert active_depth_caps() == {}, "a stage no model declares is not a cap this tool set"


def test_nothing_set_is_full_depth(monkeypatch):
    from agent.layer_depth import active_depth_caps
    from agent.weight_census import census_depth

    _clear(monkeypatch)
    assert active_depth_caps() == {}
    assert census_depth() == "all"


def test_zero_is_the_no_cap_sentinel_not_a_cap(monkeypatch):
    """set_depth expresses "all layers" by REMOVING the variable, and the gate writes 0."""
    from agent.layer_depth import active_depth_caps
    from agent.weight_census import census_depth

    _clear(monkeypatch)
    monkeypatch.setenv("TT_PERF_LAYERS", "0")
    assert active_depth_caps() == {}
    assert census_depth() == "all"


def test_the_census_refuses_a_per_stack_capped_build(monkeypatch):
    """The case run 11 let through: TT_PERF_LAYERS unset, the stacks capped at 2."""
    from agent.weight_census import census_depth

    import agent.layer_depth as LD

    _clear(monkeypatch)
    monkeypatch.setenv("TT_PERF_STACK0_LAYERS", "2")
    monkeypatch.setenv("TT_PERF_STACK1_LAYERS", "2")
    monkeypatch.setattr(LD, "_declared_stack_count", lambda root: 2)

    got = census_depth()
    assert got != "all", "a per-stack capped census still reports itself as the whole model"
    assert "STACK" in got and "2" in got, got


def test_the_marker_names_the_knob_that_shrank_the_build(monkeypatch):
    from agent.weight_census import census_depth

    import agent.layer_depth as LD

    _clear(monkeypatch)
    monkeypatch.setenv("TT_PERF_STACK0_LAYERS", "8")
    monkeypatch.setenv("TT_PERF_STACK1_LAYERS", "2")
    monkeypatch.setattr(LD, "_declared_stack_count", lambda root: 2)

    assert census_depth() == "TT_PERF_STACK1_LAYERS=2", "the tightest cap in force is not named"
