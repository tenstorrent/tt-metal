"""Tests for the kernel_constraints → iter-prompt / OUTCOME-banner bridge.

Background — the Phi-3.5 bring-up Run 3 (2026-06-01) hit a TT_FATAL on
``rotary_embedding_hf`` because the LLM iter-5 agent wrote
``ModelArgs(..., use_hf_rope=True, ...)`` in its canonical-wrapper.
The kernel_constraints catalog already produced a WARN for this exact
case (head_dim=96, "If you've explicitly set use_hf_rope=True, unset
it"), but the warning was never surfaced to the iter prompt — it lived
in a separate static-analysis layer that the LLM never saw.

The fix is a small bridge:

1. ``BringUpPlan.kernel_findings`` now carries the serialised
   WARN+BLOCKER findings.
2. ``collect_bringup_plan_files`` writes
   ``<demo_dir>/kernel_findings.json``.
3. ``build_constraint_block`` reads it and appends a "MODEL-WIDE
   KERNEL CONSTRAINTS" block to every iter prompt.
4. ``_final_outcome_banner`` auto-discovers the demo_dir for the
   model and prints findings as risks in the final banner.

These tests cover all four hops.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


# ─── KernelFinding round-trip + collect_actionable_findings ─────────


def test_kernel_finding_to_dict_and_back():
    from scripts.tt_hw_planner.kernel_constraints import KernelFinding, Severity

    f = KernelFinding(
        op="op",
        field="head_dim",
        value=96,
        constraint="must be %64",
        passes=False,
        severity=Severity.WARN,
        fix="unset use_hf_rope=True",
        source="kernels.cpp",
    )
    d = f.to_dict()
    assert d["severity"] == "warn"
    assert d["passes"] is False
    assert d["value"] == 96

    f2 = KernelFinding.from_dict(d)
    assert f2.op == f.op
    assert f2.severity == Severity.WARN
    assert f2.fix == f.fix


def test_collect_actionable_excludes_ok_and_info():
    from scripts.tt_hw_planner.kernel_constraints import (
        KernelFinding,
        KernelReport,
        Severity,
        collect_actionable_findings,
    )

    rep = KernelReport(
        findings_by_tp={
            1: [
                KernelFinding("op_a", "f", 1, "ok", passes=True, severity=Severity.BLOCKER),
                KernelFinding("op_b", "f", 2, "info_msg", passes=False, severity=Severity.INFO),
                KernelFinding("op_c", "f", 3, "warning", passes=False, severity=Severity.WARN),
                KernelFinding("op_d", "f", 4, "blocker", passes=False, severity=Severity.BLOCKER),
            ],
        },
        tp_grid=[1],
    )
    out = collect_actionable_findings(rep)
    ops = {f.op for f in out}
    assert ops == {"op_c", "op_d"}, f"expected only WARN/BLOCKER non-passing, got {ops}"


def test_collect_actionable_dedups_across_tp():
    """Most kernel constraints are TP-invariant (e.g. head_dim). The
    same finding appears in findings_by_tp[1], [2], [4], ... — collect
    must dedup by (op, field, value, constraint)."""
    from scripts.tt_hw_planner.kernel_constraints import (
        KernelFinding,
        KernelReport,
        Severity,
        collect_actionable_findings,
    )

    dup = KernelFinding("op", "head_dim", 96, "head_dim%64", passes=False, severity=Severity.WARN)
    rep = KernelReport(
        findings_by_tp={1: [dup], 2: [dup], 4: [dup], 8: [dup]},
        tp_grid=[1, 2, 4, 8],
    )
    out = collect_actionable_findings(rep)
    assert len(out) == 1


# ─── Phi-3.5 end-to-end: catalog catches head_dim=96 ────────────────


PHI_3_5_LIKE_CONFIG = {
    "model_type": "phi3",
    "hidden_size": 3072,
    "num_attention_heads": 32,
    "num_key_value_heads": 32,
    "intermediate_size": 8192,
    "num_hidden_layers": 32,
    "vocab_size": 32064,
    "max_position_embeddings": 131072,
    "rope_theta": 10000.0,
}


def test_phi_3_5_head_dim_warning_is_collected():
    """Regression: the LLM iter-5 agent wrote use_hf_rope=True for
    Phi-3.5 because this warning was never surfaced. Verify the catalog
    DOES produce it for head_dim=96."""
    from scripts.tt_hw_planner.kernel_constraints import (
        collect_actionable_findings,
        evaluate_kernels,
    )

    report = evaluate_kernels(PHI_3_5_LIKE_CONFIG)
    findings = collect_actionable_findings(report)
    rotary = [f for f in findings if "rotary_embedding_hf" in f.op]
    assert rotary, f"no rotary warning for head_dim=96; got ops {[f.op for f in findings]}"
    f0 = rotary[0]
    assert f0.value == 96
    assert (
        "use_hf_rope" in (f0.fix or "").lower()
    ), f"fix text must explicitly mention use_hf_rope so the LLM knows what to unset; got {f0.fix!r}"


# ─── load_kernel_findings — file-level reads ────────────────────────


def test_load_kernel_findings_missing_returns_empty(tmp_path):
    from scripts.tt_hw_planner._cli_helpers.kernel_findings import load_kernel_findings

    assert load_kernel_findings(tmp_path) == []


def test_load_kernel_findings_malformed_returns_empty(tmp_path):
    from scripts.tt_hw_planner._cli_helpers.kernel_findings import load_kernel_findings

    (tmp_path / "kernel_findings.json").write_text("{not json")
    assert load_kernel_findings(tmp_path) == []


def test_load_kernel_findings_round_trip(tmp_path):
    from scripts.tt_hw_planner._cli_helpers.kernel_findings import load_kernel_findings

    payload = {
        "findings": [
            {
                "op": "ttnn.x",
                "field": "head_dim",
                "value": 96,
                "constraint": "head_dim%64",
                "passes": False,
                "severity": "warn",
                "fix": "unset use_hf_rope=True",
                "source": "kernel.cpp",
            },
        ],
    }
    (tmp_path / "kernel_findings.json").write_text(json.dumps(payload))
    findings = load_kernel_findings(tmp_path)
    assert len(findings) == 1
    assert findings[0]["op"] == "ttnn.x"
    assert findings[0]["fix"] == "unset use_hf_rope=True"


# ─── format helpers ────────────────────────────────────────────────


def test_format_for_prompt_empty_returns_empty_string():
    from scripts.tt_hw_planner._cli_helpers.kernel_findings import (
        format_kernel_findings_for_prompt,
    )

    assert format_kernel_findings_for_prompt([]) == ""


def test_format_for_prompt_surfaces_fix_text_and_warning_header():
    """The prompt block must include both the constraint AND the fix
    text, with a header strong enough that the LLM doesn't reason past
    it (the Phi-3.5 iter-5 agent justified use_hf_rope=True for
    'rotary correctness' — softer wording would lose to that
    reasoning)."""
    from scripts.tt_hw_planner._cli_helpers.kernel_findings import (
        format_kernel_findings_for_prompt,
    )

    findings = [
        {
            "op": "ttnn.experimental.rotary_embedding_hf",
            "field": "head_dim",
            "value": 96,
            "constraint": "head_dim must be divisible by 64",
            "passes": False,
            "severity": "warn",
            "fix": "If you've explicitly set use_hf_rope=True, unset it.",
        }
    ]
    text = format_kernel_findings_for_prompt(findings)

    assert "MODEL-WIDE KERNEL CONSTRAINTS" in text
    assert "do NOT violate" in text or "DO NOT" in text.upper()
    assert "head_dim must be divisible by 64" in text
    assert "use_hf_rope=True" in text
    assert "[warn]" in text


def test_format_for_banner_empty_returns_empty_list():
    from scripts.tt_hw_planner._cli_helpers.kernel_findings import (
        format_kernel_findings_for_banner,
    )

    assert format_kernel_findings_for_banner([]) == []


def test_format_for_banner_includes_each_finding():
    from scripts.tt_hw_planner._cli_helpers.kernel_findings import (
        format_kernel_findings_for_banner,
    )

    findings = [
        {
            "op": "op_w",
            "field": "x",
            "value": 1,
            "constraint": "rule W",
            "passes": False,
            "severity": "warn",
            "fix": "fix W",
        },
        {
            "op": "op_b",
            "field": "y",
            "value": 2,
            "constraint": "rule B",
            "passes": False,
            "severity": "blocker",
            "fix": "",
        },
    ]
    lines = format_kernel_findings_for_banner(findings)
    flat = "\n".join(lines)
    assert "Static-analysis kernel constraints" in lines[0]
    assert "[warn]" in flat
    assert "[BLOCKER]" in flat
    assert "op_w" in flat and "op_b" in flat
    assert "fix W" in flat


# ─── bringup_plan integration: kernel_findings is populated + emitted


def test_build_bringup_plan_populates_kernel_findings():
    """build_bringup_plan now evaluates kernels off the HF config and
    stores serialised findings on the plan. For Phi-3.5-like config the
    head_dim=96 finding must appear."""
    from scripts.tt_hw_planner.bringup_plan import build_bringup_plan
    from scripts.tt_hw_planner.family_backends import FamilyBackend

    fake_backend = FamilyBackend(
        category="LLM",
        name="phi3-test",
        demo_path="models/tt_transformers/demo/simple_text_demo.py",
        routing_mode="generic",
        canonical_hf_id="microsoft/Phi-3-mini-4k-instruct",
    )

    plan = build_bringup_plan(
        new_model_id="microsoft/Phi-3.5-mini-instruct",
        new_cfg=PHI_3_5_LIKE_CONFIG,
        backend=fake_backend,
        repo_root=Path("."),
    )

    assert isinstance(plan.kernel_findings, list)
    ops = {f.get("op") for f in plan.kernel_findings}
    assert any(
        "rotary_embedding_hf" in str(op) for op in ops
    ), f"head_dim=96 rotary warning missing from plan.kernel_findings; got {ops}"


def test_collect_bringup_plan_files_emits_kernel_findings_json(tmp_path):
    from scripts.tt_hw_planner.bringup_plan import BringUpPlan, collect_bringup_plan_files

    plan = BringUpPlan(
        new_model_id="x/y",
        new_model_type="phi3",
        sibling_hf_id=None,
        sibling_model_type=None,
        backend_name="phi3",
        backend_demo_path="models/x.py",
        kernel_findings=[
            {
                "op": "ttnn.rope_hf",
                "field": "head_dim",
                "value": 96,
                "constraint": "head_dim%64",
                "passes": False,
                "severity": "warn",
                "fix": "unset use_hf_rope=True",
                "source": "kernels.cpp",
            },
        ],
    )
    files = collect_bringup_plan_files(plan=plan, new_demo_dir_rel=tmp_path)
    emitted_paths = [str(p) for p, _content, _label in files]
    assert any(
        p.endswith("kernel_findings.json") for p in emitted_paths
    ), f"kernel_findings.json not emitted; got {emitted_paths}"
    findings_blob = next(content for p, content, _ in files if str(p).endswith("kernel_findings.json"))
    parsed = json.loads(findings_blob.decode("utf-8"))
    assert parsed["findings"][0]["fix"] == "unset use_hf_rope=True"


def test_collect_bringup_plan_files_no_kernel_findings_no_file(tmp_path):
    """If there are no findings, don't emit an empty file — cleaner."""
    from scripts.tt_hw_planner.bringup_plan import BringUpPlan, collect_bringup_plan_files

    plan = BringUpPlan(
        new_model_id="x/y",
        new_model_type=None,
        sibling_hf_id=None,
        sibling_model_type=None,
        backend_name="b",
        backend_demo_path="d",
        kernel_findings=[],
    )
    files = collect_bringup_plan_files(plan=plan, new_demo_dir_rel=tmp_path)
    emitted = [str(p) for p, _, _ in files]
    assert not any(p.endswith("kernel_findings.json") for p in emitted)


# ─── build_constraint_block surfaces findings into iter prompt ──────


def test_build_constraint_block_surfaces_kernel_findings(tmp_path):
    """Critical regression test: build_constraint_block is the function
    feeding the LLM iter prompt. Without this bridge it returned ""
    for Phi-3.5 attention even though the head_dim=96 warning was sitting
    in the catalog. Now it must return a block that includes the
    rotary warning when kernel_findings.json is present."""
    from scripts.tt_hw_planner._cli_helpers.iter_prompt import build_constraint_block

    # Simulate the scaffold-time persistence step
    payload = {
        "findings": [
            {
                "op": "ttnn.experimental.rotary_embedding_hf",
                "field": "head_dim",
                "value": 96,
                "constraint": "head_dim must be divisible by 64",
                "passes": False,
                "severity": "warn",
                "fix": "If you've explicitly set use_hf_rope=True, unset it.",
            },
        ],
    }
    (tmp_path / "kernel_findings.json").write_text(json.dumps(payload))

    out = build_constraint_block(demo_dir=tmp_path, target_component="attention")
    assert out != "", "build_constraint_block must surface kernel findings even when component-level catalog is empty"
    assert "head_dim must be divisible by 64" in out
    assert "use_hf_rope=True" in out
    assert "MODEL-WIDE KERNEL CONSTRAINTS" in out


def test_build_constraint_block_empty_when_no_findings_file(tmp_path):
    """If neither component catalog nor kernel_findings.json have
    anything actionable, the block stays empty (don't pollute the
    prompt with empty headers)."""
    from scripts.tt_hw_planner._cli_helpers.iter_prompt import build_constraint_block

    out = build_constraint_block(demo_dir=tmp_path, target_component="some_component")
    assert out == ""


# ─── OUTCOME banner surfaces findings ──────────────────────────────


def test_outcome_banner_surfaces_kernel_findings(tmp_path, capsys):
    """End-to-end: when a demo_dir contains kernel_findings.json and
    _final_outcome_banner is invoked with it, the banner output must
    contain the warning lines so a multi-hour run's operator sees the
    connection between an early static-analysis warning and a late
    failure."""
    from scripts.tt_hw_planner.cli import _final_outcome_banner

    payload = {
        "findings": [
            {
                "op": "ttnn.experimental.rotary_embedding_hf",
                "field": "head_dim",
                "value": 96,
                "constraint": "head_dim must be divisible by 64",
                "passes": False,
                "severity": "warn",
                "fix": "Unset use_hf_rope=True for this model.",
            },
        ],
    }
    (tmp_path / "kernel_findings.json").write_text(json.dumps(payload))

    _final_outcome_banner(
        rc=1,
        model_id="microsoft/Phi-3.5-mini-instruct",
        path_label="test",
        demo_dir=tmp_path,
    )
    out = capsys.readouterr().out
    assert "Static-analysis kernel constraints" in out
    assert "head_dim must be divisible by 64" in out
    assert "use_hf_rope=True" in out


def test_outcome_banner_no_findings_no_block(tmp_path, capsys):
    """Empty findings → no banner block, no spurious header."""
    from scripts.tt_hw_planner.cli import _final_outcome_banner

    _final_outcome_banner(
        rc=0,
        model_id="x/y",
        path_label="t",
        demo_dir=tmp_path,
    )
    out = capsys.readouterr().out
    assert "Static-analysis kernel constraints" not in out
