"""Smoke tests for the agentic generic-discovery primitives.

These tests are deliberately offline: no HF model download, no TT
device. They exercise the pairing, divergence, signature, and
convergence math against synthetic records so the primitives stay
correct even without GPU access."""

from __future__ import annotations

from scripts.tt_hw_planner.agentic.convergence import (
    is_stagnant,
    predict_convergence,
    progress_score,
)
from scripts.tt_hw_planner.agentic.diverge import (
    DEFAULT_REL_TOL,
    ModulePair,
    _normalize_qn,
    _rel_err,
    compute_divergence,
    format_divergence_block,
)
from scripts.tt_hw_planner.agentic.learnings import compute_arch_signature
from scripts.tt_hw_planner.agentic.probe import HFModuleStats, HFProbeResult


def _mk_hf_record(qn: str, cls: str, step: int = 0, mean: float = 1.0) -> HFModuleStats:
    return HFModuleStats(
        qualified_name=qn,
        class_name=cls,
        step=step,
        shape=(1, 1, 16),
        dtype="bf16",
        mean=mean,
        std=0.5,
        l2=4.0,
        abs_max=2.0,
    )


def _mk_tt_record(qn: str, cls: str, step: int = 0, mean: float = 1.0, l2: float = 4.0) -> dict:
    return {
        "qualified_name": qn,
        "class_name": cls,
        "step": step,
        "shape": [1, 1, 16],
        "dtype": "bf16",
        "mean": mean,
        "std": 0.5,
        "l2": l2,
        "abs_max": 2.0,
    }


def test_normalize_qn_strips_model_prefix():
    assert _normalize_qn("model.layers.0.attention") == "layers.0.attention"
    assert _normalize_qn("model.language_model.layers.0.attention") == "layers.0.attention"


def test_normalize_qn_maps_synonyms():
    assert _normalize_qn("model.layers.0.self_attn") == _normalize_qn("model.layers.0.attention")
    assert _normalize_qn("model.layers.5.mlp") == _normalize_qn("model.layers.5.feedforward")


def test_rel_err_handles_small_values():
    assert _rel_err(1e-4, 2e-4) == 0.0

    assert abs(_rel_err(1.0, 2.0) - 0.5) < 1e-9

    assert abs(_rel_err(-1.0, 1.0) - 2.0) < 1e-9


def test_compute_divergence_no_diverge_when_identical():
    hf = HFProbeResult(
        model_id="x",
        records=[
            _mk_hf_record("model.layers.0.self_attn", "Attention", step=0, mean=1.0),
            _mk_hf_record("model.layers.0.mlp", "MLP", step=0, mean=0.5),
        ],
    )
    tt = [
        _mk_tt_record("model.layers.0.attention", "Attention", step=0, mean=1.0),
        _mk_tt_record("model.layers.0.feedforward", "MLP", step=0, mean=0.5),
    ]
    report = compute_divergence(hf, tt)
    assert report.paired == 2
    assert report.first_diverging is None


def test_compute_divergence_pinpoints_first_diverging():
    hf = HFProbeResult(
        model_id="x",
        records=[
            _mk_hf_record("model.layers.0.attn", "Attention", step=0, mean=1.0),
            _mk_hf_record("model.layers.1.attn", "Attention", step=0, mean=1.0),
            _mk_hf_record("model.layers.2.attn", "Attention", step=0, mean=1.0),
        ],
    )

    tt = [
        _mk_tt_record("model.layers.0.attn", "Attention", step=0, mean=1.0),
        _mk_tt_record("model.layers.1.attn", "Attention", step=0, mean=5.0, l2=20.0),
        _mk_tt_record("model.layers.2.attn", "Attention", step=0, mean=5.0, l2=20.0),
    ]
    report = compute_divergence(hf, tt)
    assert report.first_diverging is not None
    assert report.first_diverging.qualified_name == "model.layers.1.attn"


def test_compute_divergence_handles_no_pairs():
    hf = HFProbeResult(
        model_id="x",
        records=[
            _mk_hf_record("model.layers.0.self_attn", "Attention"),
        ],
    )
    tt = [_mk_tt_record("entirely.different.path", "Other")]
    report = compute_divergence(hf, tt)
    assert report.paired == 0
    assert report.note == "no-pairs-aligned"


def test_format_divergence_block_renders_first_diverging_marker():
    hf = HFProbeResult(
        model_id="x",
        records=[
            _mk_hf_record("model.layers.0.attn", "Attention", mean=1.0),
            _mk_hf_record("model.layers.1.attn", "Attention", mean=1.0),
        ],
    )
    tt = [
        _mk_tt_record("model.layers.0.attn", "Attention", mean=1.0),
        _mk_tt_record("model.layers.1.attn", "Attention", mean=5.0, l2=20.0),
    ]
    report = compute_divergence(hf, tt)
    block = format_divergence_block(report)
    assert "FIRST DIVERGENCE" in block
    assert "layers.1.attn" in block


def test_arch_signature_stable_across_runs():
    cfg = {"model_type": "llama", "num_hidden_layers": 32, "hidden_size": 4096}
    s1 = compute_arch_signature(cfg)
    s2 = compute_arch_signature(cfg)
    assert s1 == s2
    assert len(s1) == 16


def test_arch_signature_distinguishes_sizes():
    a = compute_arch_signature({"model_type": "gemma3", "num_hidden_layers": 34, "hidden_size": 3072})
    b = compute_arch_signature({"model_type": "gemma3", "num_hidden_layers": 62, "hidden_size": 5376})
    assert a != b


def test_arch_signature_reads_nested_text_config():
    cfg = {
        "model_type": "gemma3",
        "text_config": {
            "num_hidden_layers": 34,
            "hidden_size": 3072,
            "model_type": "gemma3_text",
        },
    }
    s = compute_arch_signature(cfg)
    assert s


def test_arch_signature_empty_config_returns_empty():
    assert compute_arch_signature({}) == ""


def test_progress_score_positive_when_decreasing():
    score = progress_score([1.0, 0.8, 0.6, 0.4])
    assert score > 0.5


def test_progress_score_negative_when_increasing():
    score = progress_score([0.4, 0.6, 0.8, 1.0])
    assert score < -0.5


def test_is_stagnant_detects_flat_history():
    assert is_stagnant([0.5, 0.5, 0.5, 0.5])
    assert not is_stagnant([0.5, 0.3, 0.1])


def test_predict_convergence_extrapolates_iters_to_zero():
    v = predict_convergence([1.0, 0.75, 0.5, 0.25], iters_remaining=5)
    assert v.predicted_iters_to_zero is not None
    assert 1 <= v.predicted_iters_to_zero <= 3


def test_predict_convergence_returns_none_when_no_progress():
    v = predict_convergence([0.5, 0.5, 0.5], iters_remaining=10)
    assert v.predicted_iters_to_zero is None


def test_package_exports_run_iteration():
    from scripts.tt_hw_planner.agentic import (
        AgenticIterationResult,
        run_iteration,
    )

    assert run_iteration is not None
    assert AgenticIterationResult is not None


def test_correctness_gate_demotes_category_demo_mismatch_to_loud_fail():
    """Regression guard for the Qwen3-Embedding-8B false-green.

    Scenario: probe classifies the model as ``Embed`` so the Embed
    comparator is dispatched, but the planner routed the model to
    ``simple_text_demo`` which emits ``==USER 0 - OUTPUT`` (text)
    not ``==EMBED 0 - OUTPUT``. Before the fix, the dispatcher
    soft-skipped on ``evidence.ok=False`` and the bring-up reported
    SUCCESS while the model was emitting garbage tokens. After the
    fix, the dispatcher recognises completed-demo markers from the
    wrong category and synthesises a hard-fail ValidationResult.
    """
    from scripts.tt_hw_planner.correctness.engine import (
        _looks_like_completed_demo,
        _run_via_comparator,
    )
    from scripts.tt_hw_planner.correctness.registry import get_comparator

    captured = "Some pytest preamble...\n" "==USER 0 - OUTPUT\n" "!!!!!!!!!!!!!!!!\n" "=== Performance metrics ===\n"
    assert _looks_like_completed_demo(captured)

    embed_cmp = get_comparator("Embed", "Qwen/Qwen3-Embedding-8B")
    assert embed_cmp is not None, "Embed comparator must be registered for this regression " "test to be meaningful"

    result, prompt = _run_via_comparator(
        comparator=embed_cmp,
        model_id="Qwen/Qwen3-Embedding-8B",
        captured_output=captured,
    )
    assert result is not None, (
        "MISMATCH must produce a hard-fail ValidationResult, not None " "(None is treated as a soft-pass by cmd_up)"
    )
    assert result.ok is False, "synthesised result must report ok=False"
    assert "CATEGORY/DEMO MISMATCH" in result.reason
    assert prompt is None, (
        "prompt must be None so cmd_up demotes to _PCC_FAIL_RC instead "
        "of entering the repair loop (which can't fix a routing bug)"
    )


def test_correctness_gate_soft_skips_when_no_demo_ran():
    """Complement to the previous test: a genuinely empty/cold pytest
    output (no terminal markers from any demo) should still soft-skip,
    not loud-fail. Otherwise we'd promote 'demo crashed before
    producing output' to a PCC failure -- the wrong layer of the stack
    to be reporting that error."""
    from scripts.tt_hw_planner.correctness.engine import (
        _looks_like_completed_demo,
        _run_via_comparator,
    )
    from scripts.tt_hw_planner.correctness.registry import get_comparator

    empty_captured = "pytest started\nimport errors here\n"
    assert not _looks_like_completed_demo(empty_captured)

    embed_cmp = get_comparator("Embed", "Qwen/Qwen3-Embedding-8B")
    assert embed_cmp is not None

    result, prompt = _run_via_comparator(
        comparator=embed_cmp,
        model_id="Qwen/Qwen3-Embedding-8B",
        captured_output=empty_captured,
    )

    assert result is None
    assert prompt is None


def test_lookup_fix_accepts_richer_inloop_kwargs(tmp_path):
    from scripts.tt_hw_planner.agentic.learnings import lookup_fix, register_fix

    log = tmp_path / "learned_fixes.json"
    lock = tmp_path / ".learned_fixes.lock"
    assert register_fix(
        arch_signature="sig123",
        first_diverging_qn="encoder_stack",
        diff="--- a\n+++ b\n",
        diff_files=["x.py"],
        source_model_id="m",
        log_path=log,
        lock_path=lock,
    )

    hit = lookup_fix(
        arch_signature="sig123",
        first_diverging_qn="encoder_stack",
        log_path=log,
        failure_class="API_SIGNATURE",
        error_extract="got multiple values",
        component_kind="ADAPT",
    )
    assert hit is not None
    assert hit.source_model_id == "m"

    miss = lookup_fix(
        arch_signature="sig123",
        first_diverging_qn="other",
        log_path=log,
        failure_class="API_SIGNATURE",
    )
    assert miss is None
