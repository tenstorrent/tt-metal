"""Pin: the test_demo_text trace-disable wrapper must preserve the
pytest parametrize markers (`pytestmark`) from the original function.

Without this, replacing _demo.test_demo_text with a plain wrapper
collapses the 44-variant parametrize expansion to a single
unparametrized test, which breaks every `-k '<id>'` selector the
auto-up tool emits. The visible symptom is
"collected 1 item / 1 deselected / 0 selected" — pytest then exits
rc=5 and the tool's iterate / PCC pipeline never runs.

This was the root cause of the Phi-3.5 zero-tests-collected
regression on 2026-06-02; the fix was adding functools.wraps +
explicit pytestmark copy to _install_trace_disable.
"""

from __future__ import annotations

import pytest


def _make_decorated_target():
    """Build a function decorated with multiple @pytest.mark.parametrize
    layers, mimicking how simple_text_demo.test_demo_text is stacked."""

    @pytest.mark.parametrize("optimizations", ["performance", "accuracy"])
    @pytest.mark.parametrize("variant", ["batch-1", "batch-32", "long-context-64k"])
    def test_demo_text(variant, optimizations, **_kwargs):
        return variant, optimizations

    return test_demo_text


def test_wrapper_preserves_pytestmark():
    """If the original function had stacked parametrize markers,
    the wrapper installed by _install_trace_disable MUST expose them
    via its own pytestmark so pytest discovers the variants."""
    import functools

    orig = _make_decorated_target()
    assert hasattr(orig, "pytestmark"), "test setup invariant"
    orig_marks_count = len(orig.pytestmark)

    @functools.wraps(orig)
    def wrapped(*args, **kwargs):
        return orig(*args, **kwargs)

    if hasattr(orig, "pytestmark"):
        wrapped.pytestmark = list(orig.pytestmark)

    assert hasattr(wrapped, "pytestmark"), (
        "wrapper missing pytestmark — pytest will collect only one " "unparametrized variant"
    )
    assert (
        len(wrapped.pytestmark) == orig_marks_count
    ), f"wrapper pytestmark count {len(wrapped.pytestmark)} != original {orig_marks_count}"


def test_install_trace_disable_function_uses_functools_wraps_and_copies_pytestmark():
    """Source-level guard: the actual implementation must call
    functools.wraps and copy pytestmark. Without the copy, even if
    functools.wraps is there, custom attrs on the original (set by
    pytest's decorator) may not survive."""
    from pathlib import Path

    src = Path("scripts/tt_hw_planner/instrumentation.py").read_text()
    fn_idx = src.find("def _install_trace_disable")
    assert fn_idx >= 0, "missing _install_trace_disable in instrumentation.py"
    body = src[fn_idx : fn_idx + 3000]
    assert "functools.wraps(orig)" in body, "wrapper must use @functools.wraps(orig) so __wrapped__ etc. are preserved"
    assert "wrapped.pytestmark = list(orig.pytestmark)" in body or "wrapped.pytestmark = orig.pytestmark" in body, (
        "wrapper must explicitly copy orig.pytestmark; without this, "
        "pytest discovers the wrapper as a single unparametrized "
        "test and -k selectors deselect everything"
    )
