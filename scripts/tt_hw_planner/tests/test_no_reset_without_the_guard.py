"""No code path may reset a board without first asking whether it needs resetting.

A reset is not a neutral act. `tt-smi -r` halts each chip and brings it back in turn, so a sequence
that errors partway leaves the chips it already halted with no firmware and no telemetry --
permanently, until a host reboot. device_recovery carries the rule written after that happened on
2026-08-17: read each chip's die temperature from sysfs (a file read, 0.3 ms, answers even while
the board is saturated) and skip the reset when every chip reports a plausible one, because a chip
with a running ARC has nothing for a reset to restore.

The rule was real and the incident repeated anyway, on 2026-09-22: a single-chip run, a four-chip
reset issued while every chip reported 61-77C, and the last two chips in the list never came back.
The guard did not fail -- it was never consulted. It lived behind a private name called only from
`recover()`, while cli and trace_gate ran `tt-smi -r` through their own subprocess calls.

So the guard is public now and every executor asks it. This test is the part that keeps it that
way: patching call sites is exactly how the gap opened, since it relies on whoever adds the next
one remembering. device_recovery's own words -- "a policy that holds at three of eight call sites
is not a policy".
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
_SEARCH = (
    _ROOT / "scripts" / "tt_hw_planner",
    _ROOT / "models" / "experimental" / "perf_automation",
)
# The guard's own home: it is what the others must call, so it cannot be made to call itself.
_GUARD_MODULE = "device_recovery.py"
_GUARD = "board_needs_reset"
# Asking a board about itself is always allowed; only the reset verb is gated.
_RESET_FLAG = "-r"


def _py_files():
    for root in _SEARCH:
        for p in root.rglob("*.py"):
            if "/tests/" in p.as_posix() or p.name.startswith("test_"):
                continue
            yield p


def _reset_invocations(path: Path):
    """(line, enclosing function) for every literal `tt-smi -r` argv this file builds."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return []
    scopes: list = []
    out: list = []

    class V(ast.NodeVisitor):
        def visit_FunctionDef(self, node):
            scopes.append(node)
            self.generic_visit(node)
            scopes.pop()

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_List(self, node):
            self._argv(node)
            self.generic_visit(node)

        def _argv(self, node):
            consts = [e.value for e in node.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)]
            if not consts:
                return
            if not any(c == "tt-smi" or c.endswith("/tt-smi") for c in consts):
                return
            if _RESET_FLAG not in consts:
                return
            out.append((node.lineno, scopes[-1] if scopes else None))

    V().visit(tree)
    return out


def test_every_reset_asks_the_guard_first() -> None:
    offenders = []
    for path in _py_files():
        if path.name == _GUARD_MODULE:
            continue
        for lineno, fn in _reset_invocations(path):
            body = ast.unparse(fn) if fn is not None else path.read_text(encoding="utf-8")
            if _GUARD not in body:
                where = f"{path.relative_to(_ROOT)}:{lineno}"
                offenders.append(f"{where} (in {fn.name if fn else '<module>'})")
    assert not offenders, (
        "these run `tt-smi -r` without consulting device_recovery."
        + _GUARD
        + "(); a reset issued at a board that is answering is how chips are lost permanently:\n  "
        + "\n  ".join(offenders)
    )


def test_the_guard_is_importable_under_its_public_name() -> None:
    """The gap existed because the rule was private. A test that only greps would pass on a typo."""
    import importlib.util as ilu

    p = _ROOT / "models" / "experimental" / "perf_automation" / "agent" / _GUARD_MODULE
    src = ast.parse(p.read_text(encoding="utf-8"))
    names = {n.name for n in src.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    assert _GUARD in names, f"{_GUARD}() is not defined in {_GUARD_MODULE}"
    assert not _GUARD.startswith("_"), "the guard must be public for other executors to call it"
    assert ilu.find_spec is not None


def test_the_detector_would_catch_a_new_bypass(tmp_path: Path) -> None:
    """Proves the scan above can fail -- a test that cannot fail protects nothing."""
    bad = tmp_path / "sneaky.py"
    bad.write_text("import subprocess\n" "def reclaim():\n" "    subprocess.run(['tt-smi', '-r', '0,1'])\n")
    found = _reset_invocations(bad)
    assert found, "the detector missed a plain tt-smi -r call"
    fn = found[0][1]
    assert fn is not None and _GUARD not in ast.unparse(fn)


def test_a_guarded_reset_is_accepted(tmp_path: Path) -> None:
    good = tmp_path / "fine.py"
    good.write_text(
        "import subprocess\n"
        "from x import board_needs_reset\n"
        "def reclaim():\n"
        "    if not board_needs_reset():\n"
        "        return True\n"
        "    subprocess.run(['tt-smi', '-r', '0,1'])\n"
    )
    fn = _reset_invocations(good)[0][1]
    assert _GUARD in ast.unparse(fn), "a correctly guarded reset must not be flagged"


def test_listing_and_telemetry_calls_are_not_gated(tmp_path: Path) -> None:
    """`tt-smi -ls` / `-s` ask the board about itself and must stay free of the guard."""
    probe = tmp_path / "probe.py"
    probe.write_text("import subprocess\n" "def look():\n" "    subprocess.run(['tt-smi', '-ls'])\n")
    assert _reset_invocations(probe) == [], "a non-reset tt-smi call was treated as a reset"
