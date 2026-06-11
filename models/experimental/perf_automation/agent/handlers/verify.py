"""VERIFY handler (PLAN 8.5.1) — REAL, deterministic, no device, no agent.

The cheapest rung of the ladder: ast.parse each edited file, then import it in a
child process (isolated, so a crash can't take down the loop). Returns a verdict
the engine routes: ok -> GATE_PCC; parse/import error -> REPAIR_CODE (or REVERT
once the code-fix budget is spent).
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

from .. import states

# load a file by path and execute its module body in a child process
_IMPORT_SNIPPET = (
    "import importlib.util,sys;"
    "spec=importlib.util.spec_from_file_location('_verify_mod', sys.argv[1]);"
    "m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)"
)


def _verify_files(files: list) -> dict:
    for f in files:
        p = Path(f)
        try:
            ast.parse(p.read_text())
        except SyntaxError as exc:
            return {"status": "parse_error", "file": str(f), "error": str(exc)}
        except FileNotFoundError:
            return {"status": "parse_error", "file": str(f), "error": "file not found"}
    for f in files:
        r = subprocess.run(
            [sys.executable, "-c", _IMPORT_SNIPPET, str(f)],
            capture_output=True,
            text=True,
            timeout=180,
        )
        if r.returncode != 0:
            return {"status": "import_error", "file": str(f), "error": r.stderr.strip()[-800:]}
    return {"status": "ok"}


def verify(ctx) -> str:
    files = [ctx.model_root() / f for f in (ctx.state.get("last_edit") or {}).get("files", [])]
    verdict = _verify_files(files)
    ctx.state["last_verdict"] = verdict

    if verdict["status"] == "ok":
        return states.GATE_PCC
    if ctx.state.get("code_fix_attempts", 0) < states.MAX_CODE_FIX:
        return states.REPAIR_CODE
    ctx.state["last_decision"] = {"result": "discard", "reason": "edit_failed"}
    return states.REVERT
