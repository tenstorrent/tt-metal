"""Pin: emit-e2e refuses to finish a model whose declared stages do not state what one call of
each retires.

WHY A GATE AND NOT AN INSTRUCTION. `<stage>_trace_items()` was added to every CONSUMER in August --
adapter, marker, parser, renderer -- and to no producer, so no model ever emitted it and the stage
compute ceiling silently fell back to ONE item for every stage of every model. That is right for a
recurring step and ~1500x wrong for an encoder over 1500 frames, and nothing downstream can tell the
two apart. Listing the seam in the emit-e2e prompt asks for it; a prompt is a request, and nothing
failed when it was ignored. This is what makes it real.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.tt_hw_planner.commands.emit_e2e import _stage_items_gate  # noqa: E402


def _model(tmp_path, pipeline_src: str) -> Path:
    root = tmp_path / "m"
    (root / "tt").mkdir(parents=True)
    (root / "tt" / "pipeline.py").write_text(pipeline_src)
    return root


def test_a_stage_that_states_nothing_is_refused(tmp_path):
    # The stage names are the MODEL'S OWN and deliberately unlike anything the tool could have
    # guessed: the gate must find them through PIPELINE_STAGES, never by matching a known word.
    reason = _stage_items_gate(_model(tmp_path, "PIPELINE_STAGES = ['zzz_first', 'zzz_second']\n"))
    assert reason, "a stage with no item count passes the gate"
    assert "zzz_first" in reason and "zzz_second" in reason, "the gate did not report the model's own names"
    assert "zzz_first_trace_items" in reason, "the remedy does not name the seam to add"


def test_a_model_that_states_every_count_passes(tmp_path):
    root = _model(
        tmp_path,
        "PIPELINE_STAGES = ['zzz_first', 'zzz_second']\n" "def zzz_first_trace_items(): return 1500\n"
        # ONE IS A STATEMENT, NOT A DEFAULT. A recurring stage must say so, or it cannot be told
        # apart from a stage nobody measured -- which is the whole failure this gate exists for.
        "def zzz_second_trace_items(): return 1\n",
    )
    assert _stage_items_gate(root) is None


def test_only_the_stages_that_are_missing_are_named(tmp_path):
    root = _model(
        tmp_path,
        "PIPELINE_STAGES = ['zzz_first', 'zzz_second']\ndef zzz_first_trace_items(): return 8\n",
    )
    reason = _stage_items_gate(root) or ""
    assert "zzz_second" in reason and "zzz_first'" not in reason, reason


def test_a_model_declaring_no_stages_is_not_refused(tmp_path):
    """The direct path: a model with no PIPELINE_STAGES has no stage to price, and a gate that
    refused it would refuse every hand-written model that never went through emit-e2e."""
    assert _stage_items_gate(_model(tmp_path, "def build_pipeline(device, **kw): ...\n")) is None


def test_the_gate_never_takes_emission_down(tmp_path):
    """A gate that raises is worse than the gap it looks for: emission must fail loudly on the
    model's shape, never on the checker's."""
    assert _stage_items_gate(tmp_path / "does-not-exist") is None
    assert _stage_items_gate(_model(tmp_path, "PIPELINE_STAGES = [\n")) is None
