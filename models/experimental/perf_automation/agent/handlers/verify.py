"""VERIFY handler (PLAN 8.5.1) — REAL, deterministic. SYNTAX check only.

`ast.parse` each edited file (instant, reliable). We do NOT run a standalone
import check: model files use absolute package imports (`from models.common...`),
so loading one out of its package context gives a false `import_error` on even a
perfect edit. Real import/runtime errors are caught downstream by GATE_PCC, which
runs the actual e2e test in the correct environment and routes its `crash`
verdict to REPAIR_CODE.

ok -> GATE_PCC ; parse_error -> REPAIR_CODE (or REVERT once the code-fix budget is spent).
"""

from __future__ import annotations

import ast
from pathlib import Path

from .. import states


def _resolve(ctx, raw: list):
    from .. import gitio

    model_root = ctx.model_root()
    try:
        repo = gitio.repo_root(model_root)
    except Exception:
        repo = model_root
    out = []
    for f in raw:
        p = Path(f)
        if p.is_absolute():
            out.append(p)
        elif (model_root / p).exists():
            out.append(model_root / p)
        elif (repo / p).exists():
            out.append(repo / p)
        else:
            out.append(model_root / p)
    return out


def _verify_files(files: list) -> dict:
    for f in files:
        p = Path(f)
        try:
            ast.parse(p.read_text())
        except SyntaxError as exc:
            return {"status": "parse_error", "file": str(f), "error": str(exc)}
        except FileNotFoundError:
            return {"status": "parse_error", "file": str(f), "error": "file not found"}
    return {"status": "ok"}


def verify(ctx) -> str:
    files = _resolve(ctx, (ctx.state.get("last_edit") or {}).get("files", []))
    verdict = _verify_files(files)
    ctx.state["last_verdict"] = verdict

    if verdict["status"] == "ok":
        return states.GATE_PCC
    if ctx.state.get("code_fix_attempts", 0) < states.code_fix_budget(ctx.state.get("selected_lever")):
        return states.REPAIR_CODE
    ctx.state["last_decision"] = {"result": "discard", "reason": "edit_failed", "error": verdict.get("error")}
    return states.REVERT
