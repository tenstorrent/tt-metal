"""Deterministic model skeleton (PLAN 8.x — code localization for editing).

Pure `ast` over the model's source files: classes, module functions, and every
`ttnn.*` op call with the variable it's assigned to, its line, and its enclosing
scope. NO LLM, NO hardware — reliable structure the lead reads to localize an
edit (the LLM later adds *semantics*, e.g. which linear is q vs k vs v).

`render_skeleton(map, op_substrings=...)` emits a compact, optionally op-filtered
text view (signatures + op lines, never bodies) — the context-budget control.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

_TTNN_ROOTS = {"ttnn", "tt_lib"}

# op_class -> substrings that select the relevant ttnn ops in the skeleton
OP_CLASS_SUBSTRINGS = {
    "matmul": ["linear", "matmul"],
    "attention": ["attention", "sdpa", "qkv", "concat_heads", "rotary"],
    "reduction": ["norm", "mean", "softmax", "sum"],
    "eltwise": ["add", "mul", "gelu", "silu", "activation", "unary"],
    "datamove": ["to_memory_config", "sharded", "reshard", "typecast", "tilize"],
}


def _dotted(node) -> str | None:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _call_name(call) -> str | None:
    return _dotted(call.func) if isinstance(call, ast.Call) else None


def _scope_of(node) -> str:
    parts: list[str] = []
    cur = getattr(node, "parent", None)
    while cur is not None:
        if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            parts.append(cur.name)
        cur = getattr(cur, "parent", None)
    return ".".join(reversed(parts)) if parts else "<module>"


def _assigned_target(node) -> str | None:
    p = getattr(node, "parent", None)
    if isinstance(p, ast.Assign) and p.value is node and len(p.targets) == 1:
        t = p.targets[0]
        if isinstance(t, ast.Name):
            return t.id
        if isinstance(t, ast.Tuple):
            return ",".join(e.id for e in t.elts if isinstance(e, ast.Name))
    if isinstance(p, ast.AnnAssign) and p.value is node and isinstance(p.target, ast.Name):
        return p.target.id
    return None


def build_model_map(files: list, root=None) -> dict:
    out: dict[str, Any] = {"files": {}}
    for f in files:
        p = Path(f)
        rel = str(p.relative_to(root)) if root else str(p)
        try:
            tree = ast.parse(p.read_text())
        except (SyntaxError, FileNotFoundError) as exc:
            out["files"][rel] = {"error": str(exc)}
            continue
        for node in ast.walk(tree):
            for ch in ast.iter_child_nodes(node):
                ch.parent = node
        classes, functions, ops = [], [], []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and isinstance(getattr(node, "parent", None), ast.Module):
                methods = [
                    {"name": n.name, "line": n.lineno}
                    for n in node.body
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                ]
                classes.append({"name": node.name, "line": node.lineno, "methods": methods})
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and isinstance(
                getattr(node, "parent", None), ast.Module
            ):
                functions.append({"name": node.name, "line": node.lineno, "args": [a.arg for a in node.args.args]})
            elif isinstance(node, ast.Call):
                name = _call_name(node)
                if name and name.split(".")[0] in _TTNN_ROOTS:
                    ops.append(
                        {
                            "name": name,
                            "line": node.lineno,
                            "scope": _scope_of(node),
                            "assigned_to": _assigned_target(node),
                        }
                    )
        out["files"][rel] = {"classes": classes, "functions": functions, "ops": ops}
    return out


def render_skeleton(model_map: dict, op_substrings: list | None = None) -> str:
    lines: list[str] = []
    for rel, info in model_map["files"].items():
        if "error" in info:
            lines.append(f"## {rel}  (parse error: {info['error']})")
            continue
        lines.append(f"## {rel}")
        for c in info["classes"]:
            ms = ", ".join(m["name"] for m in c["methods"])
            lines.append(f"  class {c['name']}  (L{c['line']})  methods: {ms}")
        for fn in info["functions"]:
            lines.append(f"  def {fn['name']}({', '.join(fn['args'])})  (L{fn['line']})")
        ops = info["ops"]
        if op_substrings:
            ops = [o for o in ops if any(s in o["name"] for s in op_substrings)]
        for o in ops:
            tgt = f"{o['assigned_to']} = " if o.get("assigned_to") else ""
            lines.append(f"    L{o['line']:>4}  {o['scope']}: {tgt}{o['name']}")
    return "\n".join(lines)
