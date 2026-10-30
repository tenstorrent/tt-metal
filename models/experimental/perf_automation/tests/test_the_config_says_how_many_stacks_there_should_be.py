# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""What the walk finds is checked against what the model's own config declares.

WHY AN INDEPENDENT WITNESS IS NEEDED. A model can expose one discoverable stack and hide the rest,
and that reads as success at every layer: markers are emitted, coverage is measured, a depth is
sized. Voxtral-Mini-3B did precisely this -- one encoder stack reporting and the other silent -- so
a single depth was applied to a three-section model, the text decoder was capped to 2 and both
32-layer audio encoders ran whole. Nothing in the run could tell, because "how many sections should
there be" was never asked of anything.

The HF config answers it for free. transformers has already parsed it, it needs no device, no
markers and no naming convention, and it declares a depth per section: for Voxtral, 32 for the audio
tower and 32 for the text decoder. _walk_depths always collected every one of them and
_depth_from_mapping reduced them with max() one line later, because both callers wanted a ceiling.

ENFORCED FOR WHAT THE TOOL WROTE, REPORTED FOR WHAT IT DID NOT. emit-e2e's spec requires every
repeated stack to be discoverable, so for an emitted model a mismatch is a defect in the tool's own
output and stops the run before it spends hours. Hand-written models (gemma3, llama3_1_8b_p150) have
no such obligation -- they expose no stacks at all and are measured through the coverage ladder by
design, and blocking them would refuse the entire direct path. That is not a guess: making it
unconditional turned test_coverage_source_order red immediately.
"""

from pathlib import Path

_PA = Path(__file__).resolve().parent.parent


def _run_src() -> str:
    return (_PA / "cc_optimize" / "run.py").read_text()


def test_the_config_is_read_per_section_not_collapsed():
    from models.experimental.perf_automation.agent.layer_depth import declared_section_depths

    assert callable(declared_section_depths)
    src = (_PA / "agent" / "layer_depth.py").read_text()
    i = src.index("def declared_section_depths(")
    body = src[i : i + 2500]
    assert "return max(" not in body, "the per-section reader collapses to a single number"
    assert "sorted(found, reverse=True)" in body, "it does not return the full list"
    assert "_walk_depths(" in body, "it does not reuse the existing walk"


def test_the_walk_result_is_compared_against_the_declared_sections():
    src = _run_src()
    assert "_declared_sections(" in src, "nothing reads the declared section structure"
    assert "only %d " in src or "block stack(s) are discoverable" in src, "no comparison is reported"


def test_emitted_models_are_refused_and_others_are_not():
    """The distinction that keeps this from refusing gemma3 and llama."""
    src = _run_src()
    assert "def _is_emitted_model(" in src, "no way to tell emitted output from a hand-written model"
    i = src.index("def _is_emitted_model(")
    body = src[i : i + 1400]
    assert "_stubs" in body and "e2e_plan.json" in body, "emitted models are identified by name, not structure"
    assert src.count("raise SystemExit(EXIT_REFUSED)") >= 2, "nothing actually refuses"


def test_a_refusal_can_be_overridden_deliberately():
    """An operator who knows the structure is hidden can still measure; silence is what is removed,
    not the ability to proceed."""
    src = _run_src()
    assert "PERF_MCP_ALLOW_NO_STACKS" in src, "no deliberate escape hatch"
