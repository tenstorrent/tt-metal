"""Unit tests for auto_capture_driver_onboard.

Covers the building blocks of the LLM-drafted-capture-driver pipeline:
  - _probe_model: extracts model class, signature, methods, module classes
  - _build_prompt: renders the LLM prompt with bounded size
  - _strip_markdown_fences: pulls Python source out of an LLM response
  - _validate_driver_source: AST-checks the LLM-drafted source for shape
  - _persist_driver: writes the wrapper file to learned_drivers/
  - load_learned_drivers: imports persisted .py files

The LLM-invocation path is mocked -- tests exercise the framework, not the
real claude/cursor binaries.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from unittest import mock


def _import_module():
    from scripts.tt_hw_planner import auto_capture_driver_onboard as ao

    return ao


def test_probe_model_extracts_class_name_and_signature():
    ao = _import_module()

    class MockModel:
        def forward(self, x, y=None):
            return x + (y or 0)

    probe = ao._probe_model(MockModel(), "test/model-id")
    assert probe["model_class"] == "MockModel"
    assert probe["model_id"] == "test/model-id"
    assert "x" in probe["forward_sig"]


def test_probe_model_handles_no_forward():
    ao = _import_module()

    class NoForward:
        pass

    probe = ao._probe_model(NoForward(), "x/y")
    assert probe["forward_sig"] == "(no forward method)"


def test_build_prompt_includes_all_context():
    ao = _import_module()

    probe = {
        "model_class": "Foo",
        "model_id": "owner/foo",
        "forward_sig": "(self, x)",
        "model_methods": "method_a\n    method_b",
        "module_classes": "FooConfig\n    FooProcessor",
    }
    prompt = ao._build_prompt(probe, uncaptured=["comp_a"], tried=["model(pixel_values): TypeError"])
    assert "Foo" in prompt
    assert "owner/foo" in prompt
    assert "comp_a" in prompt
    assert "TypeError" in prompt
    assert "method_a" in prompt
    assert "FooProcessor" in prompt


def test_strip_markdown_fences_handles_python_block():
    ao = _import_module()
    raw = "```python\ndef driver(model, pixel_values):\n    pass\n```"
    out = ao._strip_markdown_fences(raw)
    assert out.startswith("def driver(model, pixel_values):")
    assert "```" not in out


def test_strip_markdown_fences_handles_bare_fence():
    ao = _import_module()
    raw = "```\ndef driver(model, pixel_values):\n    return None\n```"
    out = ao._strip_markdown_fences(raw)
    assert out.startswith("def driver")


def test_strip_markdown_fences_passes_clean_source():
    ao = _import_module()
    src = "def driver(model, pixel_values):\n    return None"
    out = ao._strip_markdown_fences(src)
    assert out == src


def test_strip_markdown_fences_extracts_first_fenced_block_with_prose():
    """LLM common pattern: prose preamble + fenced code + prose epilogue.
    The old stripper only handled whole-response fences; the new one
    extracts the first embedded fenced block."""
    ao = _import_module()
    src = (
        "Here's a driver function for SAM2:\n\n"
        "```python\n"
        "def driver(model, pixel_values):\n"
        "    return None\n"
        "```\n\n"
        "This driver does X, Y, Z."
    )
    out = ao._strip_markdown_fences(src)
    assert out.startswith("def driver")
    assert "Here's a driver" not in out
    assert "does X, Y, Z" not in out


def test_strip_markdown_fences_handles_py_alias():
    """The `py` fence alias (instead of `python`) is also common."""
    ao = _import_module()
    src = "```py\ndef driver(model, pixel_values):\n    return None\n```"
    out = ao._strip_markdown_fences(src)
    assert "def driver" in out
    assert "```" not in out


def test_strip_markdown_fences_drops_prose_preamble_no_fence():
    """LLM might produce prose before the def without any fence at all."""
    ao = _import_module()
    src = "Sure, here is the driver function:\n\n" "def driver(model, pixel_values):\n" "    return None"
    out = ao._strip_markdown_fences(src)
    assert out.startswith("def driver")
    assert "Sure" not in out


def test_strip_markdown_fences_keeps_top_level_imports():
    """If the LLM (incorrectly but commonly) emits top-level imports
    before the def, we keep them so AST parse can flag them properly
    rather than silently truncating valid Python."""
    ao = _import_module()
    src = "import torch\n\ndef driver(model, pixel_values):\n    return None"
    out = ao._strip_markdown_fences(src)
    # The function should be present
    assert "def driver" in out
    # The import should NOT be silently truncated
    assert "import torch" in out


# ---------------------------------------------------------------------------
# Closed-loop iteration: _try_run_driver and _match_fired_components
# ---------------------------------------------------------------------------


def test_try_run_driver_returns_clean_on_noop_driver():
    """A no-op driver runs without raising; fired_paths is empty."""
    import torch

    ao = _import_module()

    class _M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 4)

        def forward(self, x):
            return self.linear(x)

    model = _M()
    src = "def driver(model, pixel_values):\n    return None"
    ran, err, fired = ao._try_run_driver(src, model, target_component_names=["linear"], pixel_values=torch.zeros(1, 4))
    assert ran is True
    assert err == ""
    assert fired == set()  # driver didn't invoke forward, so no fire


def test_try_run_driver_captures_runtime_exception():
    """A driver that raises is caught; returns (False, error message)."""
    import torch

    ao = _import_module()

    class _M(torch.nn.Module):
        pass

    model = _M()
    src = "def driver(model, pixel_values):\n    raise RuntimeError('intentional')"
    ran, err, fired = ao._try_run_driver(src, model, target_component_names=[], pixel_values=None)
    assert ran is False
    assert "RuntimeError" in err
    assert "intentional" in err


def test_try_run_driver_detects_target_component_fires():
    """When the driver runs forward, hooks fire on visited submodules.
    The returned fired_paths set must include those submodule paths."""
    import torch

    ao = _import_module()

    class _M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Linear(4, 4)
            self.decoder = torch.nn.Linear(4, 4)

        def forward(self, x):
            return self.decoder(self.encoder(x))

    model = _M()
    src = "def driver(model, pixel_values):\n    model(pixel_values)"
    ran, err, fired = ao._try_run_driver(
        src, model, target_component_names=["encoder", "decoder"], pixel_values=torch.zeros(1, 4)
    )
    assert ran is True
    assert "encoder" in fired
    assert "decoder" in fired


def test_match_fired_components_path_prefix_mode():
    """When components_by_path is provided, matching uses path prefix:
    target `name` with `submodule_path='X'` is matched if any fired
    path equals X, descends from X.something, or X[index]."""
    ao = _import_module()
    fired_paths = {"vision_encoder", "vision_encoder.layers.0.attn"}
    by_path = {"vision_enc_comp": "vision_encoder", "attn_comp": "vision_encoder.layers"}
    matched = ao._match_fired_components(["vision_enc_comp", "attn_comp"], fired_paths, components_by_path=by_path)
    assert "vision_enc_comp" in matched  # exact match on "vision_encoder"
    assert "attn_comp" in matched  # "vision_encoder.layers" is prefix of "vision_encoder.layers.0.attn"


def test_match_fired_components_path_indexed_descendant():
    """Bracket-indexed descendants are matched (ModuleList children)."""
    ao = _import_module()
    fired_paths = {"encoder.layers[0].mlp"}
    by_path = {"layers": "encoder.layers"}
    matched = ao._match_fired_components(["layers"], fired_paths, components_by_path=by_path)
    assert matched == {"layers"}


def test_match_fired_components_substring_fallback():
    """Without components_by_path, fallback to substring matching on
    fired paths (used when caller doesn't have resolved paths)."""
    ao = _import_module()
    fired_paths = {"encoder.layers.0.attn", "encoder.layers.0.mlp"}
    matched = ao._match_fired_components(["attn", "mlp"], fired_paths, components_by_path=None)
    assert matched == {"attn", "mlp"}


def test_match_fired_components_no_match():
    """No fired paths → no targets matched."""
    ao = _import_module()
    matched = ao._match_fired_components(["foo"], set(), components_by_path=None)
    assert matched == set()


def test_match_fired_components_path_no_partial_word():
    """Path prefix matching must respect word boundaries:
    `vision_encoder` should NOT match a fired path `vision_encoder_2`."""
    ao = _import_module()
    fired_paths = {"vision_encoder_2.neck"}
    by_path = {"ve": "vision_encoder"}
    matched = ao._match_fired_components(["ve"], fired_paths, components_by_path=by_path)
    assert matched == set()  # vision_encoder_2 is not under vision_encoder


def test_validate_driver_accepts_correct_signature():
    ao = _import_module()
    source = "def driver(model, pixel_values):\n    return None"
    ok, err = ao._validate_driver_source(source)
    assert ok, f"should accept: {err}"


def test_validate_driver_rejects_empty():
    ao = _import_module()
    ok, err = ao._validate_driver_source("")
    assert not ok
    assert "empty" in err.lower()


def test_validate_driver_rejects_syntax_error():
    ao = _import_module()
    ok, err = ao._validate_driver_source("def driver(model, pixel_values:\n    invalid")
    assert not ok
    assert "syntax" in err.lower()


def test_validate_driver_rejects_missing_driver_name():
    ao = _import_module()
    src = "def other_name(model, pixel_values):\n    pass"
    ok, err = ao._validate_driver_source(src)
    assert not ok
    assert "driver" in err.lower()


def test_validate_driver_rejects_wrong_arg_count():
    ao = _import_module()
    src = "def driver(model):\n    pass"
    ok, err = ao._validate_driver_source(src)
    assert not ok
    assert "2" in err


def test_validate_driver_rejects_wrong_arg_names():
    ao = _import_module()
    src = "def driver(a, b):\n    pass"
    ok, err = ao._validate_driver_source(src)
    assert not ok


def test_persist_driver_writes_file_and_wrapper(tmp_path, monkeypatch):
    ao = _import_module()
    monkeypatch.setattr(ao, "_LEARNED_DRIVERS_DIR", tmp_path)
    src = "def driver(model, pixel_values):\n    return None"
    path = ao._persist_driver("Sam2VideoModel", src)
    assert path.exists()
    content = path.read_text()
    assert "def driver(model, pixel_values):" in content
    assert "register_capture_driver" in content
    assert "Sam2VideoModel" in content


def test_persist_driver_safe_name():
    ao = _import_module()
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        ao._LEARNED_DRIVERS_DIR = Path(td)
        try:
            path = ao._persist_driver("My/Weird Class!", "def driver(model, pixel_values): pass")
            assert path.name.endswith(".py")
            assert "/" not in path.name
            assert " " not in path.name
            assert "!" not in path.name
        finally:
            ao._LEARNED_DRIVERS_DIR = Path(__file__).parent.parent / "learned_drivers"


def test_load_learned_drivers_imports_existing_files(tmp_path, monkeypatch):
    ao = _import_module()

    test_drv = tmp_path / "test_drv.py"
    test_drv.write_text(
        "def driver(model, pixel_values):\n" "    return None\n" "marker_value = 'loaded_from_test_drv'\n"
    )
    monkeypatch.setattr(ao, "_LEARNED_DRIVERS_DIR", tmp_path)
    loaded = ao.load_learned_drivers()
    assert len(loaded) == 1
    assert "test_drv.py" in loaded[0]


def test_load_learned_drivers_empty_dir(tmp_path, monkeypatch):
    ao = _import_module()
    monkeypatch.setattr(ao, "_LEARNED_DRIVERS_DIR", tmp_path)
    loaded = ao.load_learned_drivers()
    assert loaded == []


def test_load_learned_drivers_missing_dir(tmp_path, monkeypatch):
    ao = _import_module()
    monkeypatch.setattr(ao, "_LEARNED_DRIVERS_DIR", tmp_path / "does_not_exist")
    loaded = ao.load_learned_drivers()
    assert loaded == []


def test_auto_onboard_capture_driver_end_to_end_mocked(tmp_path, monkeypatch):
    """End-to-end with a mocked LLM that returns valid driver source."""
    ao = _import_module()
    monkeypatch.setattr(ao, "_LEARNED_DRIVERS_DIR", tmp_path)

    mock_response = (
        "```python\n"
        "def driver(model, pixel_values):\n"
        "    # synthesized driver: ignore everything\n"
        "    return None\n"
        "```"
    )

    class MockModel:
        def forward(self, x):
            return x

    with mock.patch(
        "scripts.tt_hw_planner.llm_synth.invoke_llm_cli_one_shot",
        return_value=mock_response,
    ):
        ok, path, msg = ao.auto_onboard_capture_driver(
            model=MockModel(),
            model_id="test/mock",
            uncaptured_components=["comp_a", "comp_b"],
            framework_attempts=["model(...): failed"],
        )
    assert ok, f"end-to-end should succeed: {msg}"
    assert path is not None
    assert path.exists()


def test_auto_onboard_capture_driver_invalid_response(tmp_path, monkeypatch):
    """If the LLM returns garbage, auto-onboard rejects it."""
    ao = _import_module()
    monkeypatch.setattr(ao, "_LEARNED_DRIVERS_DIR", tmp_path)

    class MockModel:
        def forward(self):
            pass

    with mock.patch(
        "scripts.tt_hw_planner.llm_synth.invoke_llm_cli_one_shot",
        return_value="this is not python code at all",
    ):
        ok, path, msg = ao.auto_onboard_capture_driver(
            model=MockModel(),
            model_id="test/bad",
            uncaptured_components=[],
            framework_attempts=[],
        )
    assert not ok
    assert path is None
    assert "validation" in msg.lower()


def test_auto_onboard_capture_driver_llm_raises(tmp_path, monkeypatch):
    ao = _import_module()
    monkeypatch.setattr(ao, "_LEARNED_DRIVERS_DIR", tmp_path)

    class MockModel:
        def forward(self):
            pass

    with mock.patch(
        "scripts.tt_hw_planner.llm_synth.invoke_llm_cli_one_shot",
        side_effect=RuntimeError("network down"),
    ):
        ok, path, msg = ao.auto_onboard_capture_driver(
            model=MockModel(),
            model_id="test/down",
            uncaptured_components=[],
            framework_attempts=[],
        )
    assert not ok
    assert "network down" in msg


def _run_all():
    fns = [
        test_probe_model_extracts_class_name_and_signature,
        test_probe_model_handles_no_forward,
        test_build_prompt_includes_all_context,
        test_strip_markdown_fences_handles_python_block,
        test_strip_markdown_fences_handles_bare_fence,
        test_strip_markdown_fences_passes_clean_source,
        test_validate_driver_accepts_correct_signature,
        test_validate_driver_rejects_empty,
        test_validate_driver_rejects_syntax_error,
        test_validate_driver_rejects_missing_driver_name,
        test_validate_driver_rejects_wrong_arg_count,
        test_validate_driver_rejects_wrong_arg_names,
        test_persist_driver_safe_name,
    ]
    import tempfile

    extra_with_tmp = [
        test_persist_driver_writes_file_and_wrapper,
        test_load_learned_drivers_imports_existing_files,
        test_load_learned_drivers_empty_dir,
        test_load_learned_drivers_missing_dir,
        test_auto_onboard_capture_driver_end_to_end_mocked,
        test_auto_onboard_capture_driver_invalid_response,
        test_auto_onboard_capture_driver_llm_raises,
    ]

    passed = 0
    failed = 0

    class _MockMonkeypatch:
        def __init__(self):
            self._orig = []

        def setattr(self, target, name, value):
            self._orig.append((target, name, getattr(target, name, None)))
            setattr(target, name, value)

        def restore(self):
            for target, name, val in reversed(self._orig):
                setattr(target, name, val)

    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
            passed += 1
        except AssertionError as exc:
            print(f"  FAIL  {fn.__name__}: {exc}")
            failed += 1
        except Exception as exc:
            print(f"  ERROR {fn.__name__}: {type(exc).__name__}: {exc}")
            failed += 1

    for fn in extra_with_tmp:
        try:
            with tempfile.TemporaryDirectory() as td:
                mp = _MockMonkeypatch()
                try:
                    fn(Path(td), mp)
                finally:
                    mp.restore()
            print(f"  PASS  {fn.__name__}")
            passed += 1
        except AssertionError as exc:
            print(f"  FAIL  {fn.__name__}: {exc}")
            failed += 1
        except Exception as exc:
            print(f"  ERROR {fn.__name__}: {type(exc).__name__}: {exc}")
            failed += 1

    print(f"\n  {passed}/{passed + failed} pass")
    return failed == 0


if __name__ == "__main__":
    import sys

    sys.exit(0 if _run_all() else 1)
