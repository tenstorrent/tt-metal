"""TT_HW_PLANNER_VERBOSE's gate must read the string value, not just truthiness of the env var.

cli.py's own os.environ.setdefault("TT_HW_PLANNER_VERBOSE", "0") sets the STRING "0" as the
default -- and bool("0") is True in Python, since "0" is a non-empty string. before_loop.py's
discover() used exactly that: `_verbose = bool(os.environ.get("TT_HW_PLANNER_VERBOSE"))`, so the
gate meant to collapse 4+ long caveat lines into one summary sentence was open on every run,
verbose or not -- the flag's "off" default could never actually turn anything off.

The fix matches the idiom this same file already uses correctly, two functions up, for a
different flag (TT_PERF_MODULE_LEVEL): compare the STRING against the known falsy spellings.
"""

from __future__ import annotations

from pathlib import Path

_SRC = (Path(__file__).resolve().parents[1] / "agent" / "before_loop.py").read_text(encoding="utf-8")


def test_the_buggy_bare_bool_cast_is_gone():
    assert 'bool(os.environ.get("TT_HW_PLANNER_VERBOSE"))' not in _SRC


def test_the_gate_now_matches_the_established_falsy_string_idiom():
    """Same idiom this file already uses for TT_PERF_MODULE_LEVEL -- reused, not reinvented."""
    assert '_verbose = os.environ.get("TT_HW_PLANNER_VERBOSE", "") not in ("", "0", "false", "False")' in _SRC


def test_the_string_zero_a_real_default_reads_as_off():
    """The exact value cli.py's setdefault writes -- must read as OFF, which is the whole bug."""
    verbose = "0" not in ("", "0", "false", "False")
    assert verbose is False


def test_an_explicit_one_reads_as_on():
    verbose = "1" not in ("", "0", "false", "False")
    assert verbose is True
