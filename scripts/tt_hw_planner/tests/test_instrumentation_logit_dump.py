"""Unit tests for the step-0 logit dump install decision.

The strict end-to-end PCC gate in ``correctness/text.py`` requires the
TT-side demo to emit a ``==LOGITS PATH:`` marker. That marker is only
produced when ``instrumentation._install_logit_dump`` patches
``Generator.prefill_forward_text`` — which historically was opt-in via
the ``TT_HW_PLANNER_DUMP_LOGITS`` env var.

Audit (2026-06-02): the env var was never set by any caller, so the
patch never installed, so the strict gate never fired, so SUCCESS was
stamped on token-overlap alone. Fix: default-on capture with opt-OUT
disable semantics, gated behind a pure helper that these tests pin.

These tests cover the PURE decision (``_should_install_logit_dump``).
The impure side — actually patching Generator — depends on ``ttnn`` /
``models.tt_transformers`` being importable, which is out of scope for
fast unit tests. The pure helper is the contract; if it returns True
the install proceeds, if False it bails.
"""

from __future__ import annotations

import os
from contextlib import contextmanager

from scripts.tt_hw_planner.instrumentation import (
    LOGITS_DUMP_ENV_VAR,
    _should_install_logit_dump,
)


@contextmanager
def _env_var(name: str, value):
    """Set an env var inside the contextmanager; restore previous value
    (including unset) on exit. ``value=None`` means delete the var."""
    sentinel = object()
    prev = os.environ.get(name, sentinel)
    try:
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
        yield
    finally:
        if prev is sentinel:
            os.environ.pop(name, None)
        else:
            os.environ[name] = prev  # type: ignore[arg-type]


def test_unset_env_var_installs_by_default() -> None:
    """No env var → should install. This is the headline change vs
    the legacy opt-in semantics that prevented the strict gate from
    ever firing in production runs."""
    with _env_var(LOGITS_DUMP_ENV_VAR, None):
        assert _should_install_logit_dump() is True


def test_explicit_enable_installs() -> None:
    """Legacy opt-in value still works for backward compat."""
    with _env_var(LOGITS_DUMP_ENV_VAR, "1"):
        assert _should_install_logit_dump() is True


def test_explicit_disable_skips_install() -> None:
    """Opt-OUT path: any of the falsy strings skips the patch.
    Case-insensitive because operators set env vars in mixed casing."""
    for val in ("0", "false", "FALSE", "False", "no", "NO", "off", "OFF"):
        with _env_var(LOGITS_DUMP_ENV_VAR, val):
            assert _should_install_logit_dump() is False, f"value={val!r} should disable"


def test_whitespace_tolerated() -> None:
    """Trailing/leading whitespace in env vars is common from shell
    quoting; treat ' 0 ' the same as '0'."""
    with _env_var(LOGITS_DUMP_ENV_VAR, "  0  "):
        assert _should_install_logit_dump() is False


def test_unknown_value_installs() -> None:
    """Defensive: any unrecognized value installs (default-on).
    Avoids surprise disables from typos like 'fasle'."""
    for val in ("yes", "true", "on", "fasle", "anything", "2"):
        with _env_var(LOGITS_DUMP_ENV_VAR, val):
            assert _should_install_logit_dump() is True, f"value={val!r} should install"


def test_empty_string_installs() -> None:
    """Empty value (e.g. ``export TT_HW_PLANNER_DUMP_LOGITS=``) → install.
    Matches the unset-var behavior so operators get the same default
    regardless of whether they exported an empty var."""
    with _env_var(LOGITS_DUMP_ENV_VAR, ""):
        assert _should_install_logit_dump() is True
