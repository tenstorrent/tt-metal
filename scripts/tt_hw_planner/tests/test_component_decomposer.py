"""Unit tests for `component_decomposer.decompose_component`.

Pins decomposition rules: non-trivial children are emitted with proper
submodule paths; containers / leaves / trivial subtrees are skipped.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: E402
from torch import nn  # noqa: E402

from scripts.tt_hw_planner.component_decomposer import (  # noqa: E402
    decompose_component,
    failure_class_warrants_decomposition,
    should_attempt_decomposition,
)


# ---------------------------------------------------------------------------
# decompose_component: non-trivial children only
# ---------------------------------------------------------------------------


class _InnerBlock(nn.Module):
    """Two-leaf block — leaf count is 2 (linear + linear)."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(8, 8)
        self.fc2 = nn.Linear(8, 8)

    def forward(self, x):
        return self.fc2(self.fc1(x))


class _Neck(nn.Module):
    """One-leaf wrapper — should be filtered by min_leaf_count default 2."""

    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(8)

    def forward(self, x):
        return self.norm(x)


class _VisionEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block_a = _InnerBlock()
        self.block_b = _InnerBlock()
        self.neck = _Neck()
        self.dropout = nn.Dropout(0.1)  # leaf class — skipped
        self.layers = nn.ModuleList([_InnerBlock(), _InnerBlock()])  # container — skipped

    def forward(self, x):  # pragma: no cover
        return x


def test_decompose_emits_non_trivial_children() -> None:
    enc = _VisionEncoder()
    out = decompose_component(parent_path="vision_encoder", parent_module=enc)
    names = [c.name for c in out]
    assert "block_a" in names
    assert "block_b" in names
    # Neck has only 1 leaf — filtered out by min_leaf_count=2 default.
    assert "neck" not in names


def test_decompose_skips_leaf_classes() -> None:
    enc = _VisionEncoder()
    out = decompose_component(parent_path="vision_encoder", parent_module=enc)
    assert "dropout" not in {c.name for c in out}


def test_decompose_skips_container_classes() -> None:
    enc = _VisionEncoder()
    out = decompose_component(parent_path="vision_encoder", parent_module=enc)
    assert "layers" not in {c.name for c in out}


def test_decompose_records_correct_submodule_path() -> None:
    enc = _VisionEncoder()
    out = decompose_component(parent_path="vision_encoder", parent_module=enc)
    by_name = {c.name: c for c in out}
    assert by_name["block_a"].submodule_path == "vision_encoder.block_a"
    assert by_name["block_b"].submodule_path == "vision_encoder.block_b"


def test_decompose_records_parent_path() -> None:
    enc = _VisionEncoder()
    out = decompose_component(parent_path="vision_encoder", parent_module=enc)
    assert all(c.parent_path == "vision_encoder" for c in out)


def test_decompose_records_class_name() -> None:
    enc = _VisionEncoder()
    out = decompose_component(parent_path="vision_encoder", parent_module=enc)
    by_name = {c.name: c for c in out}
    assert by_name["block_a"].class_name == "_InnerBlock"


def test_decompose_records_leaf_count() -> None:
    enc = _VisionEncoder()
    out = decompose_component(parent_path="vision_encoder", parent_module=enc)
    by_name = {c.name: c for c in out}
    # Each _InnerBlock has 2 linear leaves.
    assert by_name["block_a"].leaf_count == 2


def test_decompose_sorts_by_leaf_count_desc_then_name() -> None:
    """Ensures deterministic ordering for downstream consumers."""

    class _Heavy(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(8, 8)
            self.b = nn.Linear(8, 8)
            self.c = nn.Linear(8, 8)

    class _Light(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(8, 8)
            self.b = nn.Linear(8, 8)

    class _Holder(nn.Module):
        def __init__(self):
            super().__init__()
            self.zebra = _Light()
            self.alpha = _Heavy()
            self.beta = _Light()

    out = decompose_component(parent_path="root", parent_module=_Holder())
    names = [c.name for c in out]
    # alpha is heaviest → first; then alphabetical among ties.
    assert names[0] == "alpha"
    assert names[1:] == ["beta", "zebra"]


def test_decompose_returns_empty_for_primitive_module() -> None:
    out = decompose_component(parent_path="lin", parent_module=nn.Linear(8, 8))
    assert out == []


def test_decompose_handles_none_module_gracefully() -> None:
    assert decompose_component(parent_path="x", parent_module=None) == []


def test_decompose_handles_module_without_named_children() -> None:
    class _NotAModule:
        pass

    assert decompose_component(parent_path="x", parent_module=_NotAModule()) == []


def test_decompose_indexed_children_get_bracket_paths() -> None:
    """When iterating a ModuleList's children by direct attribute access,
    digit-named children get [N] notation in the submodule path."""

    class _ListHolder(nn.Module):
        def __init__(self):
            super().__init__()
            # Simulate digit-named children by attaching modules onto an
            # object that yields digit child names from named_children.
            self._modules["0"] = _InnerBlock()
            self._modules["1"] = _InnerBlock()

    out = decompose_component(parent_path="encoder.layers", parent_module=_ListHolder())
    paths = sorted(c.submodule_path for c in out)
    assert paths == ["encoder.layers[0]", "encoder.layers[1]"]


def test_decompose_min_leaf_count_param_is_honored() -> None:
    """Bumping min_leaf_count filters more aggressively."""
    enc = _VisionEncoder()
    out = decompose_component(parent_path="vision_encoder", parent_module=enc, min_leaf_count=3)
    # Each _InnerBlock has only 2 leaves; with threshold 3 they're filtered out.
    assert out == []


# ---------------------------------------------------------------------------
# should_attempt_decomposition: precondition gate
# ---------------------------------------------------------------------------


def test_should_attempt_for_agent_stuck() -> None:
    enc = _VisionEncoder()
    assert should_attempt_decomposition(parent_module=enc, failure_class="AGENT_STUCK") is True


def test_should_attempt_for_kernel_verified_missing() -> None:
    enc = _VisionEncoder()
    assert should_attempt_decomposition(parent_module=enc, failure_class="KERNEL_VERIFIED_MISSING") is True


def test_should_not_attempt_for_cold_intended() -> None:
    enc = _VisionEncoder()
    assert should_attempt_decomposition(parent_module=enc, failure_class="COLD_INTENDED") is False


def test_should_not_attempt_for_tool_bug() -> None:
    enc = _VisionEncoder()
    assert should_attempt_decomposition(parent_module=enc, failure_class="TOOL_BUG") is False


def test_should_not_attempt_for_hf_error() -> None:
    enc = _VisionEncoder()
    assert should_attempt_decomposition(parent_module=enc, failure_class="HF_ERROR") is False


def test_should_not_attempt_when_module_is_none() -> None:
    assert should_attempt_decomposition(parent_module=None, failure_class="AGENT_STUCK") is False


# ---------------------------------------------------------------------------
# failure_class_warrants_decomposition: class-only gate (used by auto-iterate
# to emit a CTA without paying the cost of HF model load)
# ---------------------------------------------------------------------------


def test_class_only_warrants_for_eligible_classes() -> None:
    """Class-only gate must return True for the same 4 classes that
    `should_attempt_decomposition` considers eligible, independent of
    whether the torch module is available."""
    for cls in ("AGENT_STUCK", "KERNEL_VERIFIED_MISSING", "CONSTRAINT_MISMATCH", "ITERATION_BUDGET"):
        assert failure_class_warrants_decomposition(cls) is True, cls


def test_class_only_does_not_warrant_for_ineligible() -> None:
    for cls in ("COLD_INTENDED", "TOOL_BUG", "HF_ERROR", "ITERATE_MORE", ""):
        assert failure_class_warrants_decomposition(cls) is False, cls


def test_class_only_gate_does_not_require_torch_module() -> None:
    """The class-only gate is the bug-fix enabler: it must NOT depend on
    having a torch module in hand. Auto-iterate calls this with no HF
    model loaded — the previous `object()`-placeholder path silently
    returned False because object() has no `named_children`."""
    assert failure_class_warrants_decomposition("AGENT_STUCK") is True
    # Sanity: no exception even with no module argument at all.
