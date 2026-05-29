"""Unit tests for Bug 3 fix — HOT/COLD auto-onboard invoker.

The vanilla HOT/COLD profiler has 3 invocation paths (pixel_values,
try_capture_drivers, bare sample). None of them work for SAM2-style
models with ``forward(inference_session)`` signature. The fix adds a
4th path: an LLM-drafted custom invoker, persisted under
``learned_invokers/`` and auto-registered via decorator.

These tests cover the framework — probe, prompt, validation, persistence,
load — without invoking the LLM. The end-to-end LLM call is mocked since
spending money to verify mocks won't help."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from unittest import mock


_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _ao():
    return importlib.import_module("scripts.tt_hw_planner.auto_hot_cold_invoker_onboard")


def test_probe_extracts_class_signature_methods() -> None:
    ao = _ao()

    class MockModel:
        def forward(self, inference_session):
            return None

        def init_session(self):
            return "session"

        def _private(self):
            pass

    probe = ao._probe_model(MockModel(), "test/model")
    assert probe["model_class"] == "MockModel"
    assert "inference_session" in probe["forward_sig"]
    assert "init_session" in probe["model_methods"]
    assert "_private" not in probe["model_methods"]


def test_validate_accepts_correct_signature() -> None:
    ao = _ao()
    src = "def invoke_for_profile(model, sample_input):\n    return None"
    ok, err = ao._validate_invoker_source(src)
    assert ok, f"should accept: {err}"


def test_validate_rejects_wrong_function_name() -> None:
    ao = _ao()
    src = "def other_name(model, sample_input):\n    pass"
    ok, err = ao._validate_invoker_source(src)
    assert not ok
    assert "invoke_for_profile" in err


def test_validate_rejects_wrong_arg_count() -> None:
    ao = _ao()
    src = "def invoke_for_profile(model):\n    pass"
    ok, err = ao._validate_invoker_source(src)
    assert not ok


def test_validate_rejects_wrong_arg_names() -> None:
    ao = _ao()
    src = "def invoke_for_profile(a, b):\n    pass"
    ok, err = ao._validate_invoker_source(src)
    assert not ok
    assert "(model, sample_input)" in err


def test_validate_rejects_syntax_error() -> None:
    ao = _ao()
    src = "def invoke_for_profile(model, sample_input:\n    invalid"
    ok, err = ao._validate_invoker_source(src)
    assert not ok
    assert "syntax" in err.lower()


def test_strip_markdown_fences_python_block() -> None:
    ao = _ao()
    raw = "```python\ndef invoke_for_profile(model, sample_input):\n    pass\n```"
    out = ao._strip_markdown_fences(raw)
    assert out.startswith("def invoke_for_profile")
    assert "```" not in out


def test_persist_writes_registration_shim(tmp_path, monkeypatch) -> None:
    ao = _ao()
    monkeypatch.setattr(ao, "_LEARNED_INVOKERS_DIR", tmp_path)
    src = "def invoke_for_profile(model, sample_input):\n    return None"
    p = ao._persist_invoker("Sam2VideoModel", src)
    assert p.exists()
    content = p.read_text()
    assert "def invoke_for_profile(model, sample_input):" in content
    assert "register_hot_cold_invoker" in content
    assert "Sam2VideoModel" in content


def test_load_imports_existing_invokers(tmp_path, monkeypatch) -> None:
    ao = _ao()
    monkeypatch.setattr(ao, "_LEARNED_INVOKERS_DIR", tmp_path)
    src = "def invoke_for_profile(model, sample_input):\n" "    return None\n" "marker = 'loaded_from_test'\n"
    (tmp_path / "test_invoker.py").write_text(src)
    loaded = ao.load_learned_invokers()
    assert any("test_invoker.py" in p for p in loaded)


def test_load_empty_dir(tmp_path, monkeypatch) -> None:
    ao = _ao()
    monkeypatch.setattr(ao, "_LEARNED_INVOKERS_DIR", tmp_path)
    assert ao.load_learned_invokers() == []


def test_register_and_resolve_custom_invoker(monkeypatch) -> None:
    """register_hot_cold_invoker should add to the registry; resolve
    should return the first matching invoker."""
    ao = _ao()
    # Save + clear registry
    monkeypatch.setattr(ao, "_INVOKER_REGISTRY", [])

    @ao.register_hot_cold_invoker(matcher=lambda m: type(m).__name__ == "Sam2VideoModel")
    def _drive(model, sample_input):
        return None

    class Sam2VideoModel:
        pass

    class OtherModel:
        pass

    assert ao.resolve_custom_invoker(Sam2VideoModel()) is _drive
    assert ao.resolve_custom_invoker(OtherModel()) is None


def test_resolve_returns_none_when_no_match(monkeypatch) -> None:
    ao = _ao()
    monkeypatch.setattr(ao, "_INVOKER_REGISTRY", [])
    assert ao.resolve_custom_invoker(object()) is None


def test_resolve_skips_matcher_that_raises(monkeypatch) -> None:
    """A matcher that raises shouldn't break resolution of OTHER
    registered invokers."""
    ao = _ao()
    monkeypatch.setattr(ao, "_INVOKER_REGISTRY", [])

    def _broken_matcher(_m):
        raise RuntimeError("buggy matcher")

    @ao.register_hot_cold_invoker(matcher=_broken_matcher)
    def _drive_broken(model, sample_input):
        pass

    @ao.register_hot_cold_invoker(matcher=lambda m: True)
    def _drive_good(model, sample_input):
        pass

    # Buggy matcher should be skipped; good one returned
    assert ao.resolve_custom_invoker(object()) is _drive_good


def test_auto_onboard_end_to_end_with_mocked_llm(tmp_path, monkeypatch) -> None:
    ao = _ao()
    monkeypatch.setattr(ao, "_LEARNED_INVOKERS_DIR", tmp_path)
    mock_response = "```python\n" "def invoke_for_profile(model, sample_input):\n" "    return None\n" "```"

    class MockModel:
        def forward(self, inference_session):
            return None

    with mock.patch(
        "scripts.tt_hw_planner.llm_synth.invoke_llm_cli_one_shot",
        return_value=mock_response,
    ):
        ok, path, msg = ao.auto_onboard_hot_cold_invoker(MockModel(), "test/mock")
    assert ok, msg
    assert path is not None and path.exists()


def test_auto_onboard_validates_llm_garbage(tmp_path, monkeypatch) -> None:
    ao = _ao()
    monkeypatch.setattr(ao, "_LEARNED_INVOKERS_DIR", tmp_path)

    class MockModel:
        def forward(self):
            pass

    with mock.patch(
        "scripts.tt_hw_planner.llm_synth.invoke_llm_cli_one_shot",
        return_value="not python code",
    ):
        ok, path, msg = ao.auto_onboard_hot_cold_invoker(MockModel(), "test/bad")
    assert not ok
    assert path is None


def test_profiler_uses_layer_4_invoker_when_standard_paths_fail(tmp_path) -> None:
    """End-to-end profiler test: when model's forward fails on standard
    paths but a learned invoker is registered, the profiler must use it
    and successfully fire hooks."""
    ao = _ao()
    profiler = importlib.import_module("scripts.tt_hw_planner.hot_cold_profiler")
    # Save + clear registry
    original_reg = list(ao._INVOKER_REGISTRY)
    ao._INVOKER_REGISTRY.clear()

    try:
        import torch
        import torch.nn as nn

        class StubbornModel(nn.Module):
            """Model whose forward rejects pixel_values and bare tensors.
            Only the custom invoker (which calls .special_path()) can fire."""

            def __init__(self):
                super().__init__()
                self.target_comp = nn.Linear(4, 4)

            def forward(self, inference_session, *, foo=None):
                # SAM2-style signature — rejects pixel_values / bare tensor
                raise TypeError("inference_session is required")

            def special_path(self, x):
                # Custom invoker calls this
                return self.target_comp(x)

        model = StubbornModel()
        model.eval()

        # Register a custom invoker for this model class
        @ao.register_hot_cold_invoker(matcher=lambda m: type(m).__name__ == "StubbornModel")
        def _drive(model, sample_input):
            model.special_path(sample_input)

        classification = profiler.profile_hot_cold(
            model=model,
            components=[{"name": "target_comp", "status": "NEW"}],
            demo_dir=tmp_path,
            sample_input=torch.randn(1, 4),
        )
        assert classification["target_comp"] == "HOT", "Layer 4 invoker must fire when standard paths fail"
    finally:
        ao._INVOKER_REGISTRY[:] = original_reg


def test_cli_exposes_auto_onboard_flag() -> None:
    """The classify-hot-cold CLI must expose --auto-onboard and route to
    the auto-onboard flow when all components classify COLD."""
    cli_mod = importlib.import_module("scripts.tt_hw_planner.cli")
    src = (Path(cli_mod.__file__)).read_text()
    assert '"--auto-onboard"' in src, "classify-hot-cold subcommand must add --auto-onboard flag"
    # The flow lives in commands/classify_hot_cold.py — verify there
    chc_mod = importlib.import_module("scripts.tt_hw_planner.commands.classify_hot_cold")
    chc_src = (Path(chc_mod.__file__)).read_text()
    assert "auto_onboard_hot_cold_invoker" in chc_src, (
        "classify_hot_cold.py must call auto_onboard_hot_cold_invoker "
        "when --auto-onboard is set and all components are COLD"
    )
    assert (
        "TT_PLANNER_AUTO_ONBOARD_HOT_COLD_INVOKER" in chc_src
    ), "must also honor the TT_PLANNER_AUTO_ONBOARD_HOT_COLD_INVOKER env var"
