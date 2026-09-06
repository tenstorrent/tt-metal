# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The depth knobs have ONE set of names, and the model declares them.

THREE COMPONENTS, THREE VOCABULARIES, AND ONLY ONE OF THEM WAS RIGHT.

    knob repair       reads PIPELINE_STAGES, creates `encode_layers` / `prefill_layers` / ...
    perf test gen     was handed a positional id and a stack path, and let the LLM choose
    depth bridge      exported TT_PERF_STACK0_LAYERS / TT_PERF_STACK1_LAYERS

Given the weight-file section names "audio_tower.layers" and "language_model.model.layers", the
generator wrote a test that passed `audio_layers=` and `text_layers=` -- names the factory does not
accept. They vanished into **kwargs, the model built every layer, and the bridge measured the cap
achieving nothing: 18729 dispatched ops before, 18729 after. Every earlier step had worked; the run
went on to optimize while profiling the whole model.

Nothing had to be invented. `stack_knob_repair.stage_names()` already reads PIPELINE_STAGES out of
the model's own source -- no build, no device, no execution -- and its docstring already says "the
override names come from the model itself". The generator simply never imported it, because it was
written before the repair existed and took the walk's output as its only input.

THE RULE, now derived in all three places: the depth knob for stage X is the build argument
`X_layers`, set by the environment variable TT_PERF_X_LAYERS.
"""

import sys
from pathlib import Path

_PA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PA))


def _model(tmp, stages='["encode", "prefill", "decode"]'):
    d = Path(tmp)
    (d / "tt").mkdir(parents=True, exist_ok=True)
    (d / "tt" / "pipeline.py").write_text(
        "PIPELINE_STAGES = %s\n\n\ndef build_pipeline(device, model=None, **kwargs):\n    return None\n" % stages
    )
    return d


def test_the_repair_and_the_generator_ask_for_the_same_names(tmp_path):
    """THE BUG: the two ends named the same knob differently, so the cap reached nothing."""
    from agent.perf_test_gen import _component_prompt  # noqa: F401  (import guard only)
    from agent.stack_knob_repair import missing_knobs, stage_names

    root = _model(tmp_path)
    stages = stage_names(root)
    assert stages == ["encode", "prefill", "decode"]

    stage_map = {st: ["stack%d" % i] for i, st in enumerate(stages)}
    wanted = missing_knobs(root, len(stages), stage_map)
    assert wanted == ["layers", "decode_layers", "encode_layers", "prefill_layers"] or set(wanted) == {
        "layers",
        "encode_layers",
        "prefill_layers",
        "decode_layers",
    }, wanted

    prompt = _multi_stack_prompt(root, stages)
    for st in stages:
        assert "%s_layers" % st in prompt, "the generator does not ask for the repair's argument name %r" % st
        assert "TT_PERF_%s_LAYERS" % st.upper() in prompt


def _multi_stack_prompt(root, stages):
    """The per-stage block the generator appends, exercised through the real function."""
    from agent.perf_test_gen import generate_perf_test

    captured = {}

    class _Stack:
        def __init__(self, path, count):
            self.path, self.count = path, count

    def _runner(prompt):
        captured["prompt"] = prompt
        return ""  # invalid output: generation fails, we only want the prompt

    (Path(root) / "demo").mkdir(exist_ok=True)
    (Path(root) / "demo" / "demo.py").write_text("def main():\n    pass\n")
    try:
        generate_perf_test(
            root,
            "main",
            "demo/demo.py",
            runner=_runner,
            force=True,
            stacks=[_Stack("audio_tower.layers", 32), _Stack("language_model.model.layers", 30)],
        )
    except Exception:  # noqa: BLE001 -- a failed generation still captured the prompt
        pass
    return captured.get("prompt", "")


def test_the_generator_no_longer_invents_names_from_stack_paths(tmp_path):
    """`audio_tower.layers` must not become `audio_layers`: the factory never accepts that, and a
    kwarg it does not accept is swallowed by **kwargs rather than refused."""
    from agent.stack_knob_repair import stage_names

    root = _model(tmp_path)
    prompt = _multi_stack_prompt(root, stage_names(root))
    assert prompt, "no prompt captured"
    assert "PERF_STACK0_LAYERS" not in prompt, "the positional vocabulary is still used for a staged model"
    assert (
        "audio_tower.layers" not in prompt.split("PER-STAGE DEPTH OVERRIDE")[-1]
    ), "the per-stage block still exposes weight-file paths the LLM can name arguments from"


def test_a_model_declaring_no_stages_keeps_the_old_form(tmp_path):
    """No stages means no shared vocabulary to derive from. Those models are no worse off than
    before -- but they must not silently get nothing."""
    root = _model(tmp_path, stages="[]")
    prompt = _multi_stack_prompt(root, [])
    assert "MULTI-STACK DEPTH OVERRIDE" in prompt
    assert "PERF_STACK0_LAYERS" in prompt


def test_the_bridge_exports_the_stage_names_too():
    """The third component. It set TT_PERF_STACK{i}_LAYERS while the test read something else, so a
    correct per-stack window was exported into variables nothing consumed."""
    src = (_PA / "cc_optimize" / "run.py").read_text()
    i = src.index("def _bridge_depth_env(")
    body = src[i : src.index("\ndef ", i + 1)]
    # Spelled by layer_depth.stage_layers_var now: the rule this file states was written out
    # longhand in four places, and the bridge was one of them. What matters is that the bridge
    # exports a PER-STAGE name, not which module formats the string.
    assert "_stage_layers_var(_stage)" in body, "the bridge does not export stage names"
    assert "stage_depths" in body, "the bridge is not given the stage mapping"
    assert "_stack_layers_var(" in body, "the positional fallback was removed with nothing to replace it"


def test_the_per_stage_mapping_is_published_not_just_printed():
    """It was computed for a log line and dropped, while the bridge invented its own names."""
    src = (_PA / "cc_optimize" / "run.py").read_text()
    assert 'facts["per_stage"] = dict(_per_stage)' in src, "the per-stage depths never leave the coverage step"
    bl = (_PA / "agent" / "before_loop.py").read_text()
    assert 'stage_depths=(_bl_facts or {}).get("per_stage")' in bl, "the bridge is not handed them"


def test_stage_names_come_from_the_model_not_from_the_tool():
    """One authority: PIPELINE_STAGES, read from the model's own source with no build and no device."""
    src = (_PA / "agent" / "stack_knob_repair.py").read_text()
    i = src.index("def stage_names(")
    body = src[i : src.index("\ndef ", i + 1)]
    assert "PIPELINE_STAGES" in body
    gen = (_PA / "agent" / "perf_test_gen.py").read_text()
    assert "from .stack_knob_repair import stage_names" in gen, "the generator still does not consult the model"


def test_the_generated_variable_names_are_valid_python():
    """A stage named with a space or a dash would produce a syntactically broken test."""
    from agent.perf_test_gen import generate_perf_test  # noqa: F401

    src = (_PA / "agent" / "perf_test_gen.py").read_text()
    i = src.index("PER-STAGE DEPTH OVERRIDE")
    window = src[max(0, i - 1500) : i]
    assert "isidentifier()" in window, "a stage name that is not an identifier would break the test it writes"
