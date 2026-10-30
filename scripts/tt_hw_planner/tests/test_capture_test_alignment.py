"""Unit tests for Bug 2 fix — capture/test signature alignment.

Before the fix, the PCC test scaffold resolved torch_module via
``_CANDIDATE_SUBMODULE_PATHS`` alone. The capture-inputs step saved
``submodule_path`` in manifest.json but the test ignored it. When
capture's resolution differed from test's, captured args didn't fit
the test-resolved torch_module's signature and the test silently
failed with misleading errors like ``_make_arg_for() inputs are
shape-incompatible`` (even though captured inputs WERE loaded).

The fix injects ``_captured_submodule_path()`` into the test scaffold
via CAPTURE_LOADER_SOURCE. The test now reads manifest.json's
submodule_path and uses it as the FIRST candidate path -- falling back
to ``_CANDIDATE_SUBMODULE_PATHS`` only when the manifest path fails or
isn't present.

These tests verify the alignment logic, not the test execution itself
(which requires HF model load + hardware)."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _capture_inputs():
    return importlib.import_module("scripts.tt_hw_planner.capture_inputs")


def test_capture_loader_source_includes_submodule_path_helper() -> None:
    """The CAPTURE_LOADER_SOURCE injected into tests must define
    ``_captured_submodule_path`` -- the function that reads manifest.json
    and returns the path capture hooked."""
    ci = _capture_inputs()
    src = ci.CAPTURE_LOADER_SOURCE
    assert "def _captured_submodule_path(component_name):" in src, (
        "CAPTURE_LOADER_SOURCE must include _captured_submodule_path so "
        "the test scaffold can read the manifest's submodule_path"
    )
    # Reads manifest.json
    assert "manifest.json" in src
    assert "submodule_path" in src


def test_capture_loader_source_handles_missing_manifest_gracefully() -> None:
    """The injected function must return None (not raise) when manifest
    is missing -- so tests for components that weren't captured still
    work via _CANDIDATE_SUBMODULE_PATHS fallback."""
    ci = _capture_inputs()
    src = ci.CAPTURE_LOADER_SOURCE
    # Look for the early return when manifest doesn't exist
    assert "if not manifest_p.is_file():" in src
    assert "return None" in src
    # And exception handling around json.loads
    assert "except Exception:" in src


def test_pcc_test_template_uses_captured_submodule_path_first() -> None:
    """The PCC test template in bringup_loop.py must check
    _captured_submodule_path BEFORE iterating _CANDIDATE_SUBMODULE_PATHS.
    Source-grep the template body."""
    bl = importlib.import_module("scripts.tt_hw_planner.bringup_loop")
    template = bl._PCC_TEST_TEMPLATE
    # The build function should reference _captured_submodule_path
    assert "_captured_submodule_path(COMPONENT_NAME)" in template, (
        "_build_torch_reference must call _captured_submodule_path " "before iterating _CANDIDATE_SUBMODULE_PATHS"
    )
    # Manifest path resolved BEFORE the candidate-paths loop
    idx_captured = template.find("_captured_submodule_path(COMPONENT_NAME)")
    idx_candidate_loop = template.find("for path in _CANDIDATE_SUBMODULE_PATHS:")
    assert idx_captured != -1 and idx_candidate_loop != -1
    assert idx_captured < idx_candidate_loop, (
        "the captured-path check must occur BEFORE the candidate loop, "
        "not after — otherwise captured manifest path is just a fallback "
        "instead of a primary"
    )


def test_template_falls_back_to_candidates_when_captured_path_fails() -> None:
    """If the manifest path resolves to None (e.g., HF module structure
    changed), the test must fall through to _CANDIDATE_SUBMODULE_PATHS,
    not skip immediately."""
    bl = importlib.import_module("scripts.tt_hw_planner.bringup_loop")
    template = bl._PCC_TEST_TEMPLATE
    # Look for the conditional fallback
    assert "if torch_module is None:" in template, "must have a fallback path when manifest resolution fails"


def test_captured_submodule_path_function_logic() -> None:
    """Execute the actual function logic from CAPTURE_LOADER_SOURCE to
    verify it reads manifest.json correctly. Uses exec since the function
    is defined as a string template."""
    ci = _capture_inputs()
    # Compile and exec the source so the function is callable
    namespace: dict = {}
    exec(ci.CAPTURE_LOADER_SOURCE, namespace)
    fn = namespace["_captured_submodule_path"]

    import tempfile
    import os

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        # Simulate test layout: <demo_dir>/tests/pcc/test_X.py
        test_dir = td_path / "tests" / "pcc"
        test_dir.mkdir(parents=True)
        # Simulate captured manifest
        captured = td_path / "_captured" / "comp_a"
        captured.mkdir(parents=True)
        (captured / "manifest.json").write_text(
            json.dumps({"submodule_path": "vision_encoder.neck", "component": "comp_a"})
        )

        # Monkey-patch __file__ so parents[2] resolves to td_path
        # We can do this by setting the function's globals' __file__
        fake_test_file = test_dir / "test_comp_a.py"
        fake_test_file.write_text("")
        # The function uses __file__ via fn's enclosing scope; emulate by
        # creating the namespace's __file__:
        namespace["__file__"] = str(fake_test_file)
        # Re-exec since __file__ was just set
        exec(ci.CAPTURE_LOADER_SOURCE, namespace)
        fn = namespace["_captured_submodule_path"]
        result = fn("comp_a")
        assert result == "vision_encoder.neck", f"expected 'vision_encoder.neck', got {result!r}"


def test_captured_submodule_path_returns_none_when_no_manifest() -> None:
    ci = _capture_inputs()
    namespace: dict = {}
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        test_dir = td_path / "tests" / "pcc"
        test_dir.mkdir(parents=True)
        fake_test_file = test_dir / "test_phantom.py"
        fake_test_file.write_text("")
        namespace["__file__"] = str(fake_test_file)
        exec(ci.CAPTURE_LOADER_SOURCE, namespace)
        fn = namespace["_captured_submodule_path"]
        result = fn("phantom_component")
        assert result is None, f"missing manifest must return None, got {result!r}"


def test_captured_submodule_path_returns_none_on_malformed_manifest() -> None:
    """Malformed JSON in manifest must not crash the test scaffold."""
    ci = _capture_inputs()
    namespace: dict = {}
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        test_dir = td_path / "tests" / "pcc"
        test_dir.mkdir(parents=True)
        fake_test_file = test_dir / "test_x.py"
        fake_test_file.write_text("")
        captured = td_path / "_captured" / "x"
        captured.mkdir(parents=True)
        (captured / "manifest.json").write_text("not valid json {{{")
        namespace["__file__"] = str(fake_test_file)
        exec(ci.CAPTURE_LOADER_SOURCE, namespace)
        fn = namespace["_captured_submodule_path"]
        result = fn("x")
        assert result is None


def test_captured_submodule_path_returns_none_when_submodule_path_missing_key() -> None:
    """Manifest exists but has no submodule_path key (older schema or
    partial write) → return None."""
    ci = _capture_inputs()
    namespace: dict = {}
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        test_dir = td_path / "tests" / "pcc"
        test_dir.mkdir(parents=True)
        fake_test_file = test_dir / "test_y.py"
        fake_test_file.write_text("")
        captured = td_path / "_captured" / "y"
        captured.mkdir(parents=True)
        (captured / "manifest.json").write_text(json.dumps({"component": "y"}))
        namespace["__file__"] = str(fake_test_file)
        exec(ci.CAPTURE_LOADER_SOURCE, namespace)
        fn = namespace["_captured_submodule_path"]
        result = fn("y")
        assert result is None
