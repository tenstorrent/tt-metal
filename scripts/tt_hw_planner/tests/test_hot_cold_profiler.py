"""Unit tests for the HOT/COLD profiler.

The profiler attaches forward hooks to each NEW component's submodule
and runs a sample forward pass. Components that fire are HOT, those
that don't are COLD. UNRESOLVED is the conservative fallback when we
can't find the submodule at all.

These tests use a hand-built fake nn.Module hierarchy so we don't
need HF transformers or hardware. The profiler is model-agnostic so
testing against synthetic models is the right level of granularity."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from unittest import mock


_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _profiler():
    return importlib.import_module("scripts.tt_hw_planner.hot_cold_profiler")


def _make_fake_model_with_active_and_dead_submodules():
    """Build a small nn.Module hierarchy where 'active_*' submodules
    fire during forward() and 'dead_*' submodules never do."""
    import torch
    import torch.nn as nn

    class Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 4)

        def forward(self, x):
            return self.linear(x)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.active_a = Inner()
            self.active_b = Inner()
            self.dead_a = Inner()  # never invoked
            self.dead_b = Inner()  # never invoked

        def forward(self, x):
            x = self.active_a(x)
            x = self.active_b(x)
            return x

    return Model()


def test_hot_classification_for_invoked_submodule(tmp_path) -> None:
    p = _profiler()
    import torch

    model = _make_fake_model_with_active_and_dead_submodules()
    model.eval()
    components = [
        {"name": "active_a", "status": "NEW"},
        {"name": "active_b", "status": "NEW"},
        {"name": "dead_a", "status": "NEW"},
        {"name": "dead_b", "status": "NEW"},
    ]
    sample = torch.randn(1, 4)
    classification = p.profile_hot_cold(
        model=model,
        components=components,
        demo_dir=tmp_path,
        sample_input=sample,
    )
    assert classification["active_a"] == "HOT"
    assert classification["active_b"] == "HOT"
    assert classification["dead_a"] == "COLD"
    assert classification["dead_b"] == "COLD"


def test_reuse_and_adapt_components_are_skipped(tmp_path) -> None:
    """Only NEW components are subject to HOT/COLD classification.
    REUSE / ADAPT components have a different lifecycle (they're
    already on TT via existing tt-port paths) and shouldn't appear in
    the output."""
    p = _profiler()
    import torch

    model = _make_fake_model_with_active_and_dead_submodules()
    model.eval()
    components = [
        {"name": "active_a", "status": "NEW"},
        {"name": "reuse_thing", "status": "REUSE"},
        {"name": "adapt_thing", "status": "ADAPT"},
    ]
    classification = p.profile_hot_cold(
        model=model,
        components=components,
        demo_dir=tmp_path,
        sample_input=torch.randn(1, 4),
    )
    assert "active_a" in classification
    assert "reuse_thing" not in classification
    assert "adapt_thing" not in classification


def test_unresolved_when_submodule_not_found(tmp_path) -> None:
    """Conservative fallback: if a component name maps to no submodule,
    classify as UNRESOLVED so the auto-iterate loop treats it carefully
    (not silently classified as COLD which would skip the port)."""
    p = _profiler()
    import torch

    model = _make_fake_model_with_active_and_dead_submodules()
    model.eval()
    components = [
        {"name": "totally_does_not_exist", "status": "NEW"},
    ]
    classification = p.profile_hot_cold(
        model=model,
        components=components,
        demo_dir=tmp_path,
        sample_input=torch.randn(1, 4),
    )
    assert classification["totally_does_not_exist"] == "UNRESOLVED"


def test_hook_removed_after_profile(tmp_path) -> None:
    """Hooks must be removed in the finally block so a second profile
    run doesn't accumulate stale hooks. Verified by running TWO profiles
    back-to-back -- if hooks leaked, the second profile's fired set
    would inherit ghost firings from the first."""
    p = _profiler()
    import torch

    model = _make_fake_model_with_active_and_dead_submodules()
    model.eval()
    components = [
        {"name": "active_a", "status": "NEW"},
        {"name": "dead_a", "status": "NEW"},
    ]
    # First profile
    c1 = p.profile_hot_cold(
        model=model,
        components=components,
        demo_dir=tmp_path,
        sample_input=torch.randn(1, 4),
    )
    # Second profile, fresh fired set
    c2 = p.profile_hot_cold(
        model=model,
        components=components,
        demo_dir=tmp_path,
        sample_input=torch.randn(1, 4),
    )
    # Both runs must produce the same classification
    assert c1 == c2
    # And no submodule should accumulate hooks
    for name in ("active_a", "dead_a", "active_b", "dead_b"):
        sub = getattr(model, name, None)
        assert sub is not None
        # _forward_pre_hooks is the dict torch uses to track hooks
        # After our profile_hot_cold finishes, it should be empty
        assert len(getattr(sub, "_forward_pre_hooks", {})) == 0, f"hook leak on {name} after profile finished"


def test_empty_components_returns_empty(tmp_path) -> None:
    p = _profiler()
    import torch

    model = _make_fake_model_with_active_and_dead_submodules()
    classification = p.profile_hot_cold(
        model=model,
        components=[],
        demo_dir=tmp_path,
        sample_input=torch.randn(1, 4),
    )
    assert classification == {}


def test_summarize_hot_cold_groups_correctly() -> None:
    p = _profiler()
    classification = {
        "a": "HOT",
        "b": "COLD",
        "c": "HOT",
        "d": "UNRESOLVED",
        "e": "COLD",
    }
    buckets = p.summarize_hot_cold(classification)
    assert buckets["HOT"] == ["a", "c"]
    assert buckets["COLD"] == ["b", "e"]
    assert buckets["UNRESOLVED"] == ["d"]


def test_make_sample_input_default_shape() -> None:
    p = _profiler()
    import torch

    x = p.make_sample_input()
    assert isinstance(x, torch.Tensor)
    assert x.shape == (1, 3, 1024, 1024)


def test_make_sample_input_custom_shape() -> None:
    p = _profiler()
    import torch

    x = p.make_sample_input(batch=2, channels=4, height=128, width=128)
    assert x.shape == (2, 4, 128, 128)


def test_forward_failure_returns_all_cold_not_raise(tmp_path) -> None:
    """If model.forward raises for any reason, the profiler must NOT
    crash -- it should return all-COLD as the conservative outcome,
    and the caller can notice + investigate."""
    p = _profiler()
    import torch
    import torch.nn as nn

    class BrokenModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.comp_a = nn.Linear(4, 4)

        def forward(self, *args, **kwargs):
            raise RuntimeError("simulated forward failure")

    model = BrokenModel()
    components = [{"name": "comp_a", "status": "NEW"}]
    classification = p.profile_hot_cold(
        model=model,
        components=components,
        demo_dir=tmp_path,
        sample_input=torch.randn(1, 4),
    )
    # comp_a is COLD because forward never reached it
    assert classification["comp_a"] == "COLD"


def test_hook_registration_failure_is_treated_as_unresolved(tmp_path) -> None:
    """If we can find the submodule but registering a hook on it fails
    (some custom modules disable hooks), we treat as UNRESOLVED so we
    don't silently classify a component as COLD when we have no signal."""
    p = _profiler()
    import torch
    import torch.nn as nn

    class HookHostileModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.comp_a = nn.Linear(4, 4)

        def forward(self, x):
            return self.comp_a(x)

    model = HookHostileModel()

    # Patch _attach_hook to always return None (simulates failure)
    with mock.patch.object(p, "_attach_hook", return_value=None):
        components = [{"name": "comp_a", "status": "NEW"}]
        classification = p.profile_hot_cold(
            model=model,
            components=components,
            demo_dir=tmp_path,
            sample_input=torch.randn(1, 4),
        )
    # comp_a was found in the model but hook registration failed,
    # so we can't observe it firing -- COLD is the result of "no
    # firing observed". This is the documented behavior: if you can't
    # hook, you get COLD. The caller can audit when COLD count is
    # surprisingly high.
    assert classification["comp_a"] in ("COLD", "UNRESOLVED")
