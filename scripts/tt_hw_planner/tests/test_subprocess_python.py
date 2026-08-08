"""Pin: every subprocess invocation of `python -m scripts.tt_hw_planner ...`
must use `sys.executable`, not the bare string `"python"`.

Background: 2026-06-04 seamless-m4t promote run silently corrupted
`bringup_status.json` because `_refresh_plan` in `bringup_loop.py` invoked
the scaffold subprocess with bare `"python"`. On a system where the
shell's PATH puts a venv WITHOUT PyTorch ahead of the project venv, the
subprocess's `AutoModel.from_pretrained` fails, scaffold writes a
0-component plan, and the iter loop's apply step silently rejects every
LLM solution downstream. `sys.executable` guarantees the same Python
interpreter as the parent process — eliminating PATH dependence.

This test fails if any `subprocess.run(["python", ...])` pattern
reappears in the planner code.
"""

from __future__ import annotations

import re
from pathlib import Path


_PLANNER_ROOT = Path(__file__).resolve().parent.parent


def _planner_py_files() -> list[Path]:
    out = []
    for p in _PLANNER_ROOT.rglob("*.py"):
        if "/tests/" in str(p) or p.name.startswith("test_"):
            continue
        out.append(p)
    return out


def test_no_bare_python_subprocess_invocation() -> None:
    """No file under `scripts/tt_hw_planner/` may invoke `subprocess.run`
    (or Popen) with the bare string `"python"` as argv[0]. Always use
    `sys.executable` so the subprocess inherits the parent's interpreter
    and its installed packages (especially `torch`)."""
    # Catches: subprocess.run(["python", ...]) / subprocess.Popen(["python", ...])
    # Allows: subprocess.run([sys.executable, ...]) / [_sys.executable, ...]
    # Allows: documentation/comments that mention "python -m ..." commands
    bad_pat = re.compile(
        r"subprocess\.(?:run|Popen|check_output|check_call|call)\s*\(\s*\[\s*[\"']python[\"']",
    )
    offenders: list[tuple[Path, int, str]] = []
    for f in _planner_py_files():
        try:
            text = f.read_text(encoding="utf-8")
        except Exception:
            continue
        for m in bad_pat.finditer(text):
            line_no = text.count("\n", 0, m.start()) + 1
            line = text.splitlines()[line_no - 1] if line_no - 1 < len(text.splitlines()) else ""
            offenders.append((f, line_no, line.strip()))

    assert (
        not offenders
    ), "Found bare 'python' subprocess invocations — use sys.executable instead. " "Offenders:\n  " + "\n  ".join(
        f"{f}:{ln}: {src}" for f, ln, src in offenders
    )


def test_refresh_plan_uses_sys_executable() -> None:
    """Specifically pin `_refresh_plan` in bringup_loop.py — this is
    the original site of the bug. Even if the broad scan above missed
    a variant, this pin would catch a regression."""
    src = (_PLANNER_ROOT / "bringup_loop.py").read_text(encoding="utf-8")
    body_start = src.find("def _refresh_plan(")
    assert body_start != -1, "_refresh_plan function not found in bringup_loop.py"
    body_end_marker = src.find("\ndef ", body_start + 1)
    body = src[body_start : body_end_marker if body_end_marker != -1 else len(src)]
    assert "sys.executable" in body, (
        "_refresh_plan must invoke the scaffold subprocess via sys.executable; "
        "bare 'python' resolves via PATH and may pick a torch-less venv, "
        "causing scaffold to write 0-component plans and corrupt "
        "bringup_status.json (the 2026-06-03 seamless-m4t failure mode)."
    )
    # Negative: bare "python" must NOT appear as a literal argv0 in the cmd list
    assert (
        'cmd = ["python"' not in body and 'cmd=["python"' not in body
    ), "_refresh_plan still references bare 'python' as subprocess argv[0]"
