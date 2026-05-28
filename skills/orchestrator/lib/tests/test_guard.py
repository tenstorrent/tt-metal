# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Tests for skills.orchestrator.lib.guard — static lint + traced-op assertions."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from skills.orchestrator.lib.guard import (
    BlockVerdict,
    HostResidentSubOp,
    KIND_REQUIRED_KERNELS,
    LintViolation,
    PerfArtifactVerdict,
    UseCaseVerdict,
    assert_traced_ops,
    cross_check_reference,
    lint_block,
    verify_block,
    verify_optimization_artifact,
    verify_use_case,
)


# ---------------------------------------------------------------------------
# lint_block
# ---------------------------------------------------------------------------


def _write(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    return path


def test_lint_rejects_cpu_in_forward(tmp_path):
    """RED: .cpu() inside a forward method is forbidden."""
    src = "class Block:\n" "    def forward(self, x):\n" "        return x.cpu()\n"
    f = _write(tmp_path / "block.py", src)
    violations = lint_block(f)
    assert len(violations) == 1, violations
    v = violations[0]
    assert isinstance(v, LintViolation)
    assert v.pattern == ".cpu()"
    assert v.line == 3
    assert "cpu" in v.snippet


def test_lint_rejects_numpy_in_forward(tmp_path):
    src = "class Block:\n" "    def forward(self, x):\n" "        return x.numpy()\n"
    f = _write(tmp_path / "block.py", src)
    violations = lint_block(f)
    assert len(violations) == 1
    assert violations[0].pattern == ".numpy()"


def test_lint_rejects_torch_functional(tmp_path):
    src = "import torch\n" "class Block:\n" "    def forward(self, x):\n" "        return torch.nn.functional.relu(x)\n"
    f = _write(tmp_path / "block.py", src)
    violations = lint_block(f)
    assert len(violations) == 1
    assert violations[0].pattern == "torch.nn.functional"


def test_lint_rejects_torch_matmul_in_forward(tmp_path):
    src = "import torch\n" "class Block:\n" "    def forward(self, a, b):\n" "        return torch.matmul(a, b)\n"
    f = _write(tmp_path / "block.py", src)
    violations = lint_block(f)
    assert len(violations) == 1
    assert violations[0].pattern == "torch.matmul"


def test_lint_rejects_todo_comment(tmp_path):
    src = "class Block:\n" "    def forward(self, x):\n" "        # TODO: move to ttnn\n" "        return x\n"
    f = _write(tmp_path / "block.py", src)
    violations = lint_block(f)
    assert len(violations) == 1
    assert violations[0].pattern == "# TODO: move to ttnn"


def test_lint_accepts_cpu_in_test_function(tmp_path):
    """A function named test_* is exempt — pytest helpers may legitimately .cpu()."""
    src = "def test_something():\n" "    x = object()\n" "    x.cpu()\n"
    f = _write(tmp_path / "block.py", src)
    assert lint_block(f) == []


def test_lint_accepts_cpu_in_tests_path(tmp_path):
    """Files under */tests/* are exempt entirely."""
    src = "class Block:\n" "    def forward(self, x):\n" "        return x.cpu()\n"
    f = _write(tmp_path / "tests" / "foo.py", src)
    assert lint_block(f) == []


def test_lint_real_qwen3_tts_rms_norm_is_clean():
    """Smoke: known-good TTNN block file lints to empty list.

    The plan spells the file as rms_norm.py; the repo has rmsnorm.py.
    Accept either; skip if neither exists.
    """
    candidates = [
        Path("models/demos/qwen3_tts/tt/rms_norm.py"),
        Path("models/demos/qwen3_tts/tt/rmsnorm.py"),
    ]
    chosen = next((p for p in candidates if p.exists()), None)
    if chosen is None:
        pytest.skip("no qwen3_tts rmsnorm file in repo")
    assert lint_block(chosen) == []


# ---------------------------------------------------------------------------
# assert_traced_ops
# ---------------------------------------------------------------------------


def test_assert_traced_ops_norm_satisfied():
    assert assert_traced_ops(["ttnn.rms_norm"], "norm") == []


def test_assert_traced_ops_attention_one_missing():
    missing = assert_traced_ops(["ttnn.matmul"], "attention")
    assert len(missing) == 1
    assert "softmax" in missing[0]


def test_assert_traced_ops_unknown_kind_raises():
    with pytest.raises(ValueError):
        assert_traced_ops([], "nonsense")


def test_assert_traced_ops_returns_human_readable():
    missing = assert_traced_ops([], "attention")
    # First option set is {ttnn.linear, ttnn.matmul} → sorted, comma-space.
    assert "any of {ttnn.linear, ttnn.matmul}" in missing


# ---------------------------------------------------------------------------
# cross_check_reference
# ---------------------------------------------------------------------------


def test_cross_check_allows_same_op_in_both(tmp_path):
    block = _write(
        tmp_path / "block.py",
        "class B:\n" "    def forward(self, x):\n" "        return x.cpu()\n",
    )
    ref = _write(
        tmp_path / "ref.py",
        "def helper(x):\n" "    return x.cpu()\n",
    )
    assert cross_check_reference(block, ref) == []


def test_cross_check_flags_new_host_op(tmp_path):
    block = _write(
        tmp_path / "block.py",
        "class B:\n" "    def forward(self, x):\n" "        return x.cpu()\n",
    )
    ref = _write(
        tmp_path / "ref.py",
        "def helper(x):\n" "    return x + 1\n",
    )
    result = cross_check_reference(block, ref)
    assert len(result) == 1
    assert isinstance(result[0], HostResidentSubOp)
    assert result[0].op == "cpu"


def test_cross_check_reference_directory(tmp_path):
    block = _write(
        tmp_path / "block" / "block.py",
        "class B:\n" "    def forward(self, x):\n" "        return x.cpu()\n",
    )
    ref_dir = tmp_path / "ref"
    _write(ref_dir / "a.py", "def f(x):\n    return x + 1\n")
    _write(ref_dir / "b.py", "def g(x):\n    return x.cpu()\n")
    assert cross_check_reference(block, ref_dir) == []


def test_cross_check_block_clean_returns_empty(tmp_path):
    block = _write(
        tmp_path / "block.py",
        "class B:\n" "    def forward(self, x):\n" "        return x + 1\n",
    )
    ref = _write(tmp_path / "ref.py", "pass\n")
    assert cross_check_reference(block, ref) == []


# ---------------------------------------------------------------------------
# KIND_REQUIRED_KERNELS shape sanity
# ---------------------------------------------------------------------------


def test_kind_required_kernels_shape():
    """Each value must be a list of (possibly empty) sets of strings.

    The empty-list case (kind="other") is allowed: it expresses "no
    traced-op requirement for this kind." See guard.py for the kinds
    table; conv + other were added when speech / audio models needed
    looser requirements.
    """
    expected_kinds = {"norm", "linear", "attention", "mlp", "decoder_layer", "embedding", "conv", "other"}
    assert set(KIND_REQUIRED_KERNELS.keys()) == expected_kinds
    for kind, options in KIND_REQUIRED_KERNELS.items():
        assert isinstance(options, list), kind
        for opt in options:
            assert isinstance(opt, set) and opt, f"{kind}: empty option set"
            assert all(isinstance(k, str) for k in opt), kind


# ---------------------------------------------------------------------------
# _is_test_path regression tests
# ---------------------------------------------------------------------------


def test_lint_is_test_path_tests_grandparent_exempts(tmp_path):
    """Block file under <root>/foo/tests/bar/block.py — `tests` is a grandparent
    in the path; lint should exempt and return no violations."""
    target = tmp_path / "foo" / "tests" / "bar"
    target.mkdir(parents=True)
    f = target / "block.py"
    f.write_text("import torch\n" "class M:\n" "    def forward(self, x):\n" "        return x.cpu()\n")
    assert lint_block(f) == []


def test_lint_is_test_path_tests_was_here_not_exempt(tmp_path):
    """A sibling dir named `tests_was_here/` is NOT a `tests` ancestor and must
    not exempt the file."""
    target = tmp_path / "tests_was_here"
    target.mkdir()
    f = target / "block.py"
    f.write_text("import torch\n" "class M:\n" "    def forward(self, x):\n" "        return x.cpu()\n")
    violations = lint_block(f)
    assert len(violations) == 1
    assert violations[0].pattern == ".cpu()"


def test_lint_is_test_path_test_subdir_isnt_test(tmp_path):
    """A directory named `test_subdir_isnt_test/` (matches pytest tmp_path's
    `test_<funcname>0/` pattern) must NOT exempt; only files literally named
    test_* or directories literally named `tests` count."""
    target = tmp_path / "test_subdir_isnt_test"
    target.mkdir()
    f = target / "block.py"
    f.write_text("import torch\n" "class M:\n" "    def forward(self, x):\n" "        return x.cpu()\n")
    violations = lint_block(f)
    assert len(violations) == 1
    assert violations[0].pattern == ".cpu()"


# ---------------------------------------------------------------------------
# .cpu() / .numpy() strict-call-form
# ---------------------------------------------------------------------------


def test_lint_rejects_cpu_with_kwargs(tmp_path):
    """`.cpu(non_blocking=True)` is still a host-residency leak and must be flagged."""
    f = tmp_path / "block.py"
    f.write_text(
        "import torch\n" "class M:\n" "    def forward(self, x):\n" "        return x.cpu(non_blocking=True)\n"
    )
    violations = lint_block(f)
    assert len(violations) == 1
    assert violations[0].pattern == ".cpu()"


# ---------------------------------------------------------------------------
# verify_block composite wrapper
# ---------------------------------------------------------------------------


def test_verify_block_clean_returns_ok(tmp_path):
    """A clean block with all required kernels and matching reference → ok=True."""
    block = tmp_path / "block.py"
    block.write_text("import ttnn\n" "class M:\n" "    def forward(self, x):\n" "        return ttnn.rms_norm(x)\n")
    reference = tmp_path / "ref.py"
    reference.write_text("import ttnn\n")
    verdict = verify_block(block, ["ttnn.rms_norm"], "norm", reference)
    assert verdict.ok is True
    assert verdict.lint == []
    assert verdict.missing_kernels == []
    assert verdict.new_host_ops == []


def test_verify_block_failing_lint_marks_not_ok(tmp_path):
    """A block with .cpu() in forward should produce lint violations and ok=False."""
    block = tmp_path / "block.py"
    block.write_text("import torch\n" "class M:\n" "    def forward(self, x):\n" "        return x.cpu()\n")
    reference = tmp_path / "ref.py"
    reference.write_text("import torch\n")
    verdict = verify_block(block, ["ttnn.rms_norm"], "norm", reference)
    assert verdict.ok is False
    assert len(verdict.lint) >= 1
    # The .cpu() also appears as a "new_host_op" since reference doesn't have one
    assert len(verdict.new_host_ops) >= 1


def test_verify_block_missing_kernel_marks_not_ok(tmp_path):
    """A clean block but traced ops missing a required kernel → ok=False."""
    block = tmp_path / "block.py"
    block.write_text("import ttnn\n")
    reference = tmp_path / "ref.py"
    reference.write_text("import ttnn\n")
    verdict = verify_block(block, [], "attention", reference)
    assert verdict.ok is False
    assert len(verdict.missing_kernels) == 2  # both attention option-sets missing


# ---------------------------------------------------------------------------
# verify_use_case
# ---------------------------------------------------------------------------


def test_verify_use_case_clean_returns_ok(tmp_path):
    model_file = tmp_path / "uc1_model.py"
    model_file.write_text(
        "from .layernorm import LayerNorm\n"
        "from .seamless_mha import SeamlessMha\n"
        "class UC1Model:\n"
        "    def __init__(self, device):\n"
        "        self.norm = LayerNorm(device)\n"
        "        self.attn = SeamlessMha(device)\n"
    )
    use_case = {
        "name": "uc1",
        "components_used": ["LayerNorm", "SeamlessMha"],
        "hf_class": "XxxForUC1",
        "validation_metric": "bleu",
    }
    verdict = verify_use_case(model_file, use_case)
    assert verdict.ok
    assert verdict.issues == []


def test_verify_use_case_rejects_missing_component_import(tmp_path):
    model_file = tmp_path / "uc1_model.py"
    model_file.write_text("class UC1Model: pass\n")  # imports nothing
    use_case = {
        "name": "uc1",
        "components_used": ["LayerNorm", "SeamlessMha"],
        "hf_class": "XxxForUC1",
        "validation_metric": "bleu",
    }
    verdict = verify_use_case(model_file, use_case)
    assert not verdict.ok
    assert any("LayerNorm" in issue for issue in verdict.issues)
    assert any("SeamlessMha" in issue for issue in verdict.issues)


def test_verify_use_case_demo_must_invoke_hf_reference(tmp_path):
    model_file = tmp_path / "uc1_model.py"
    model_file.write_text("from .x import X\nclass UC1Model: pass\n")

    demo_file = tmp_path / "demo_uc1.py"
    demo_file.write_text("# does not run HF\n")

    use_case = {
        "name": "uc1",
        "components_used": ["X"],
        "hf_class": "XxxForUC1",
        "validation_metric": "bleu",
    }
    verdict = verify_use_case(model_file, use_case, demo_path=demo_file)
    assert not verdict.ok
    assert any("XxxForUC1" in issue for issue in verdict.issues)


def test_verify_use_case_test_must_enforce_metric(tmp_path):
    model_file = tmp_path / "uc1_model.py"
    model_file.write_text("from .x import X\nclass UC1Model: pass\n")

    test_file = tmp_path / "test_e2e_uc1.py"
    test_file.write_text("def test_uc1(): pass\n")  # doesn't use bleu

    use_case = {
        "name": "uc1",
        "components_used": ["X"],
        "hf_class": "XxxForUC1",
        "validation_metric": "bleu",
    }
    verdict = verify_use_case(model_file, use_case, test_path=test_file)
    assert not verdict.ok
    assert any("bleu" in issue for issue in verdict.issues)


def test_verify_use_case_all_three_checks_pass(tmp_path):
    model_file = tmp_path / "uc1_model.py"
    model_file.write_text("from .x import X\nclass UC1Model: pass\n")

    demo_file = tmp_path / "demo_uc1.py"
    demo_file.write_text("from transformers import XxxForUC1\n" "def main(): hf = XxxForUC1.from_pretrained('x')\n")

    test_file = tmp_path / "test_e2e_uc1.py"
    test_file.write_text("from demo.validate import bleu\ndef test_uc1(): pass\n")

    use_case = {
        "name": "uc1",
        "components_used": ["X"],
        "hf_class": "XxxForUC1",
        "validation_metric": "bleu",
    }
    verdict = verify_use_case(model_file, use_case, demo_path=demo_file, test_path=test_file)
    assert verdict.ok, verdict.issues


def test_verify_use_case_missing_model_file_fails(tmp_path):
    use_case = {
        "name": "uc1",
        "components_used": [],
        "hf_class": "XxxForUC1",
        "validation_metric": "bleu",
    }
    verdict = verify_use_case(tmp_path / "missing.py", use_case)
    assert not verdict.ok
    assert any("not found" in issue for issue in verdict.issues)


# ---------------------------------------------------------------------------
# verify_optimization_artifact
# ---------------------------------------------------------------------------


def test_verify_optimization_artifact_ok_status_requires_artifact():
    result = {"status": "ok", "tracy_artifact": "", "notes": "no improvement found"}
    verdict = verify_optimization_artifact(result)
    assert not verdict.ok
    assert any("non-empty tracy_artifact" in i for i in verdict.issues)


def test_verify_optimization_artifact_missing_file_rejected(tmp_path):
    result = {
        "status": "ok",
        "tracy_artifact": str(tmp_path / "nonexistent_traced_run.csv"),
        "notes": "traced run done",
    }
    verdict = verify_optimization_artifact(result)
    assert not verdict.ok
    assert any("does not exist" in i for i in verdict.issues)


def test_verify_optimization_artifact_empty_file_rejected(tmp_path):
    empty = tmp_path / "empty_traced.csv"
    empty.write_text("")
    result = {"status": "ok", "tracy_artifact": str(empty), "notes": "traced"}
    verdict = verify_optimization_artifact(result)
    assert not verdict.ok
    assert any("empty" in i for i in verdict.issues)


def test_verify_optimization_artifact_untraced_evidence_rejected(tmp_path):
    csv = tmp_path / "untraced_capture.csv"
    csv.write_text("OP_CODE,DEVICE_KERNEL_DURATION_NS\nMatmul,1000\n")
    result = {
        "status": "ok",
        "tracy_artifact": str(csv),
        "notes": "host-dispatch dominated; no single op > 5%",
    }
    verdict = verify_optimization_artifact(result, require_traced_path=True)
    assert not verdict.ok
    assert any("traced" in i for i in verdict.issues)


def test_verify_optimization_artifact_traced_evidence_ok(tmp_path):
    csv = tmp_path / "traced_run_cpp_device_perf_report.csv"
    csv.write_text("OP_CODE,DEVICE_KERNEL_DURATION_NS\nMatmul,1000\n")
    result = {
        "status": "ok",
        "tracy_artifact": str(csv),
        "notes": "captured under --traced; top op 8% of step",
    }
    verdict = verify_optimization_artifact(result)
    assert verdict.ok, verdict.issues


def test_verify_optimization_artifact_non_ok_status_skips_check():
    # status=fail or status=blocked has different routing; artifact not required.
    for st in ("fail", "blocked"):
        result = {"status": st, "tracy_artifact": "", "notes": "device hung"}
        verdict = verify_optimization_artifact(result)
        assert verdict.ok, f"status={st} should skip artifact check"


def test_verify_optimization_artifact_require_traced_false_accepts_either(tmp_path):
    csv = tmp_path / "any_capture.csv"
    csv.write_text("OP,US\n")
    result = {"status": "ok", "tracy_artifact": str(csv), "notes": "untraced is fine here"}
    verdict = verify_optimization_artifact(result, require_traced_path=False)
    assert verdict.ok, verdict.issues
