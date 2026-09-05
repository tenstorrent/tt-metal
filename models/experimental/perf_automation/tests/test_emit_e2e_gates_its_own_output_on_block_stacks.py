# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""emit-e2e refuses to hand over a model whose stacks the profiler cannot see.

PROSE IN A SPEC IS NOT A GUARANTEE. emit_e2e's prompt told the author to cap every repeated stack
and keep each one discoverable -- and Voxtral-Mini-3B was emitted violating both. Its per-layer
wrappers shared no base, so find_all_stacks saw ONE stack for a three-section model: full_blocks came
back 0, the 2/4/8/16 ladder was climbed to recover a depth the markers supply free, and a single
depth capped the text decoder while both 32-layer audio encoders ran whole. That cost a day of runs,
and every gate in emit_e2e passed the entire time.

An instruction that nothing checks is a suggestion. This gate is the check.

IT NEEDS NO NAMING CONVENTION AND NO MARKER IN THE MODEL. Two independent facts settle it: the HF
config declares a block depth per section (transformers has already parsed it, no device required),
and building the model and walking it says how many stacks are actually discoverable. Fewer stacks
than sections means structure is hidden, and hidden structure is inferred rather than measured for
the entire life of every run that follows.

THE BUILD IS CHEAP AND ITS FAILURE IS ALSO A FINDING. The probe builds at layers=2 in a subprocess:
7.1s on Voxtral against 30+ minutes at full depth. A model that dies building shallow fails the gate
too -- Voxtral crashed in the argmax reshape at depth 2 because capping left its aggregate sub-block
holding zero layers, which is exactly the "capped build must remain a MODEL, not a fragment" clause
the spec states and nothing enforced.
"""

from pathlib import Path

_EMIT = Path(__file__).resolve().parents[4] / "scripts/tt_hw_planner/commands/emit_e2e.py"


def _src() -> str:
    return _EMIT.read_text()


def test_the_gate_exists_and_the_runner_calls_it():
    src = _src()
    assert "def _block_stack_gate(" in src, "emit-e2e does not check its own block stacks"
    i = src.index("def _run_deterministic_gates(")
    body = src[i:]
    assert "_block_stack_gate(" in body, "the gate is defined but never run"
    assert body.index("_block_stack_gate(") < body.index(
        "return (len(reasons) == 0)"
    ), "the gate runs after the verdict"


def test_it_compares_against_the_config_not_a_convention():
    src = _src()
    i = src.index("def _block_stack_gate(")
    body = src[i : i + 4000]
    assert "declared_section_depths" in body, "the gate does not read the declared sections"
    assert "len(sections)" in body, "the gate does not compare counts"
    # no naming heuristics anywhere in the decision
    assert "layer|block" not in body and "re.search" not in body, "the gate decides by name"


def test_a_model_that_cannot_build_shallow_fails_the_gate():
    """Capping must leave a runnable model. Voxtral's cap left an aggregate sub-block with zero
    layers and died in the argmax reshape -- a fragment, not a model."""
    src = _src()
    i = src.index("def _block_stack_gate(")
    body = src[i : i + 4000]
    assert "could not be BUILT at layers=2" in body, "a failed shallow build is not reported"


def test_the_probe_runs_out_of_process():
    """This command must not be taken down by the model it is checking."""
    src = _src()
    i = src.index("def _block_stack_gate(")
    body = src[i : i + 4000]
    assert "subprocess.run(" in body, "the build runs in-process"
    assert "timeout=" in body, "the probe can hang the emit"


def test_single_section_models_are_left_alone():
    """One declared section means one stack is the whole story; there is nothing to compare."""
    src = _src()
    i = src.index("def _block_stack_gate(")
    body = src[i : i + 4000]
    assert "len(sections) < 2" in body, "single-section models are gated on a comparison that cannot fail"


def test_multi_stack_models_must_expose_a_knob_per_stack():
    """ONE NUMBER CANNOT DESCRIBE A MULTI-SECTION MODEL.

    optimize sizes a coverage window PER stack -- the smallest depth in which every distinct op type
    of that stack appears -- and those numbers differ: an audio encoder of one repeated conformer
    block saturates at 2, a decoder interleaving attention and MLP variants may need 8. With a single
    `layers` argument the tool has nowhere to put the second number, so it collapses them with max()
    and every section is profiled at the deepest one.

    Voxtral-Mini-3B is the worked example: a 32-layer text decoder and TWO 32-layer audio encoder
    stacks behind one argument. Capping at 2 built 2 text layers behind 64 encoder layers; once the
    encoders were capped too, all three sections were forced to the same depth whether it suited them
    or not.

    The override names come from the model's own PIPELINE_STAGES, so no new convention is invented,
    and it is read from the SIGNATURE -- **kwargs is exactly what swallowed `layers` silently.
    """
    src = _src()
    assert "def _missing_stack_knobs(" in src, "nothing checks for per-stack depth knobs"
    i = src.index("def _missing_stack_knobs(")
    body = src[i : i + 2600]
    assert "PIPELINE_STAGES" in body, "the override names are invented rather than taken from the model"
    assert "kwonlyargs" in body, "the check does not read the factory signature"
    assert "n_stacks < 2" in body, "single-stack models are asked for overrides they do not need"
    gate = src[src.index("def _block_stack_gate(") :]
    assert "_missing_stack_knobs(" in gate, "the check is defined but never run by the gate"
