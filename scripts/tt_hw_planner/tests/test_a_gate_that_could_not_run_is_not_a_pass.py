# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""A correctness gate that could not run used to report the same thing as one that passed.

Both engines returned None when the HF reference would not load -- the evidence engine called it a
"soft pass" in as many words -- and None is also what the operator's own --no-strict-pcc produces.
So the run stamped SUCCESS on a model nothing had compared, and the one class of model this hit
hardest was the class most likely to be wrong: models shipping their own code, whose reference load
was refused by a default that disagreed with the rest of the tool.

These pin the three pieces: the trust decision, the fail-closed verdict, and the local-checkpoint
probe that was skipping its architecture step for the same "it returned early" reason.
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.tt_hw_planner.output_validation import (
    UNVERIFIED,
    ValidationResult,
    is_unverified,
    resolve_trust_remote_code,
    unverified,
)

_ENV = "HF_TRUST_REMOTE_CODE"


# --------------------------------------------------------------------------------------
# One decision about running a model's own code
# --------------------------------------------------------------------------------------


def test_a_model_that_ships_its_own_code_can_be_loaded_by_default(monkeypatch) -> None:
    """The rest of bring-up already runs that code minutes earlier; refusing here protected nothing."""
    monkeypatch.delenv(_ENV, raising=False)
    assert resolve_trust_remote_code() is True


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "OFF", " False "])
def test_the_refusal_is_still_available_to_anyone_who_wants_it(monkeypatch, value) -> None:
    monkeypatch.setenv(_ENV, value)
    assert resolve_trust_remote_code() is False


@pytest.mark.parametrize("value", ["1", "true", "yes", ""])
def test_anything_that_is_not_a_refusal_leaves_it_on(monkeypatch, value) -> None:
    monkeypatch.setenv(_ENV, value)
    assert resolve_trust_remote_code() is True


def test_a_caller_that_states_its_own_answer_is_not_overridden(monkeypatch) -> None:
    monkeypatch.setenv(_ENV, "0")
    assert resolve_trust_remote_code(True) is True
    monkeypatch.delenv(_ENV, raising=False)
    assert resolve_trust_remote_code(False) is False


def test_the_decision_is_made_in_exactly_one_place() -> None:
    """Three copies of this env read disagreed with the rest of the tool; one is enough."""
    body = (Path(__file__).resolve().parents[1] / "output_validation.py").read_text()
    assert body.count(f'os.environ.get(_TRUST_REMOTE_CODE_ENV, "")') == 1
    assert 'environ.get("HF_TRUST_REMOTE_CODE"' not in body


# --------------------------------------------------------------------------------------
# Could-not-compare is not the same as compared-and-agreed
# --------------------------------------------------------------------------------------


def test_a_gate_that_could_not_compare_reports_a_failure_not_a_blank() -> None:
    v = unverified("HF reference generation failed (OSError: no cache)")
    assert v.ok is False
    assert v.reason.startswith(UNVERIFIED)
    assert is_unverified(v)


def test_a_real_mismatch_is_not_mistaken_for_an_unrunnable_gate() -> None:
    """The two are both ok=False and call for opposite responses, so they must stay separable."""
    mismatch = ValidationResult(ok=False, reason="LOGIT-PCC FAIL: 0.42")
    assert is_unverified(mismatch) is False


def test_a_passing_gate_and_an_absent_one_are_neither_of_them_unverified() -> None:
    assert is_unverified(ValidationResult(ok=True, reason="32/32 match")) is False
    assert is_unverified(None) is False


# --------------------------------------------------------------------------------------
# Both engines, on the path that actually failed
# --------------------------------------------------------------------------------------


def _gate_inputs():
    return {
        "model_id": "some/model",
        "captured_output": "==USER 0 - OUTPUT\nhello world\n",
        "args": None,
    }


def test_a_reference_that_will_not_load_fails_the_gate_closed() -> None:
    """This is the whole bug: the load throws, and the gate used to hand back a blank."""
    from scripts.tt_hw_planner.cli import _run_pcc_gate

    with (
        patch("scripts.tt_hw_planner.output_validation.extract_demo_user_output", return_value="hello"),
        patch("scripts.tt_hw_planner.output_validation.load_demo_first_prompt", return_value="a prompt"),
        patch(
            "scripts.tt_hw_planner.output_validation.generate_hf_reference",
            side_effect=OSError("needs trust_remote_code"),
        ),
    ):
        result, prompt = _run_pcc_gate(**_gate_inputs())

    assert result is not None, "a reference that would not load still reported nothing to judge"
    assert is_unverified(result), result.reason
    assert prompt == "a prompt"


def test_the_evidence_engine_no_longer_soft_passes_either() -> None:
    """It named the defect itself: 'Skipping the gate (soft pass)'."""
    from scripts.tt_hw_planner.correctness.engine import run_evidence_gate

    with (
        patch("scripts.tt_hw_planner.output_validation.extract_demo_user_output", return_value="hello"),
        patch("scripts.tt_hw_planner.output_validation.load_demo_first_prompt", return_value="a prompt"),
        patch(
            "scripts.tt_hw_planner.output_validation.generate_hf_reference",
            side_effect=OSError("needs trust_remote_code"),
        ),
    ):
        result, _prompt = run_evidence_gate(category="LLM", **_gate_inputs())

    assert result is not None and is_unverified(result), getattr(result, "reason", result)


def test_a_tokenizer_that_will_not_reload_also_fails_closed() -> None:
    """The second way the comparison can fail to happen, on the same path."""
    from scripts.tt_hw_planner.cli import _run_pcc_gate

    class _Ref:
        text = "hello"
        token_ids = [1, 2]
        truncated = False
        step0_logits = None

    with (
        patch("scripts.tt_hw_planner.output_validation.extract_demo_user_output", return_value="hello"),
        patch("scripts.tt_hw_planner.output_validation.load_demo_first_prompt", return_value="a prompt"),
        patch("scripts.tt_hw_planner.output_validation.generate_hf_reference", return_value=_Ref()),
        patch(
            "scripts.tt_hw_planner.output_validation.tokenize_text_for_compare",
            side_effect=OSError("needs trust_remote_code"),
        ),
    ):
        result, _prompt = _run_pcc_gate(**_gate_inputs())

    assert result is not None and is_unverified(result), getattr(result, "reason", result)


# --------------------------------------------------------------------------------------
# The verdict has to reach the outcome line, without starting a repair loop
# --------------------------------------------------------------------------------------


def _cli_source() -> str:
    return (Path(__file__).resolve().parents[1] / "cli.py").read_text()


def test_an_unverified_gate_does_not_launch_the_repair_loop() -> None:
    """There is no divergence to localize and no component to repair when nothing was compared.

    Escalating would spend a repair budget on what is usually a missing cache entry, so both gate
    call sites must exclude it from the escalation branch.
    """
    src = _cli_source()
    escalation_guards = src.count("and not _pcc_result.ok and not _is_unverified(_pcc_result)")
    assert escalation_guards == 2, "both gate call sites must exclude unverified from escalation"
    assert "and not _pcc_result.ok:\n" not in src, "an unguarded escalation branch is left"


def test_a_run_that_verified_nothing_no_longer_calls_itself_a_success() -> None:
    """OUTCOME_UNVERIFIED existed for exactly this and was never passed to the banner."""
    src = _cli_source()
    assert "_gate_outcome = OUTCOME_UNVERIFIED" in src
    assert "outcome=_gate_outcome," in src
    assert "outcome=OUTCOME_UNVERIFIED if _cold_note else None," in src


def test_both_call_sites_word_the_verdict_the_same_way() -> None:
    """One sentence, written once. Two ends of a check that drift apart is the bug being fixed."""
    from scripts.tt_hw_planner.cli import _unverified_banner_note

    assert _unverified_banner_note(ValidationResult(ok=True, reason="matched")) is None
    assert _unverified_banner_note(ValidationResult(ok=False, reason="LOGIT-PCC FAIL: 0.42")) is None
    note = _unverified_banner_note(unverified("HF reference generation failed"))
    assert note is not None and "NOT a confirmed SUCCESS" in note
    assert _cli_source().count("PCC correctness gate could NOT verify this model") == 1


def test_switching_the_gate_off_is_still_your_own_business() -> None:
    """An operator who passed --no-strict-pcc gets the old label; only a FAILED gate downgrades."""
    src = _cli_source()
    unverified_branch = src[src.index("_gate_outcome = None") : src.index("outcome=_gate_outcome,")]
    none_branch = unverified_branch[unverified_branch.index("if _pcc_result is None:") :]
    assert "OUTCOME_UNVERIFIED" not in none_branch.split("elif")[0]


# --------------------------------------------------------------------------------------
# A local checkpoint gets the planning a hub id gets
# --------------------------------------------------------------------------------------


# A model_type no name table carries. The planning must come from the config's SHAPE, not from
# recognising a family, or a local checkpoint of anything new is back to getting no plan at all.
_LOCAL_CONFIG = {
    "model_type": "zzz-not-a-known-family",
    "architectures": ["ZzzForCausalLM"],
    "hidden_size": 2048,
    "num_hidden_layers": 16,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "intermediate_size": 5632,
    "vocab_size": 32000,
    "max_position_embeddings": 4096,
    "torch_dtype": "bfloat16",
}


@pytest.fixture()
def local_checkpoint(tmp_path):
    """A directory that looks like a downloaded checkpoint: a config and something weighing bytes."""
    import json

    (tmp_path / "config.json").write_text(json.dumps(_LOCAL_CONFIG))
    (tmp_path / "model.safetensors").write_bytes(b"\0" * 4096)
    return str(tmp_path)


def test_a_local_checkpoint_is_planned_like_a_named_one(local_checkpoint) -> None:
    """Same model, same config, pointed at by path instead of by hub id.

    Without the architecture step this returned arch_spec=None, and every consumer downstream reads
    that as "not a transformer": weights-only plan, no KV math, mesh defaulted to one chip.
    """
    from scripts.tt_hw_planner import probe as probe_mod

    with patch.object(probe_mod, "_agent_classify_category", return_value=None):
        p = probe_mod.probe_model(local_checkpoint)

    assert p.arch_spec is not None, "local checkpoint still skips the architecture step"
    assert p.memory_model is not None, "local checkpoint still has no memory model to plan against"
    assert p.arch_spec.num_layers == _LOCAL_CONFIG["num_hidden_layers"]
    assert p.arch_spec.hidden_size == _LOCAL_CONFIG["hidden_size"]


def test_the_two_probe_paths_run_the_same_architecture_step(local_checkpoint) -> None:
    """One helper, called by both -- so the next field added to it cannot reach only one path."""
    body = (Path(__file__).resolve().parents[1] / "probe.py").read_text()
    assert body.count("return _apply_transformer_config(probe, cfg, total_params, weight_bytes)") == 2


def test_a_directory_without_a_readable_config_is_still_returned(tmp_path) -> None:
    """A folder that is not a checkpoint must not start raising where it used to return."""
    from scripts.tt_hw_planner import probe as probe_mod

    (tmp_path / "config.json").write_text("{}")
    with patch.object(probe_mod, "_agent_classify_category", return_value=None):
        p = probe_mod.probe_model(str(tmp_path))
    assert p is not None
    assert p.arch_spec is None
