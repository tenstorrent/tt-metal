"""Generate synthetic .pyi stubs for C++-bound classes.

For every class binding `(python_name=X, helper="class_", cpp=Y)` we emit a
Python class declaration named X. For every other binding whose
`cpp_qualified_name` starts with `Y::`, we emit a method or attribute on X.

Stubs are written to `dep-graph/cache/pyi_stubs/<module>.pyi` mirroring the
package structure that ttnn imports them from (mostly `ttnn._ttnn.<sub>`).

The intent is to let Jedi resolve `obj.method()` when `obj: ttnn.MemoryConfig`
— Jedi follows `ttnn.MemoryConfig` through `ttnn/__init__.py` to
`ttnn._ttnn.tensor.MemoryConfig`, and if a stub lives at the right path the
class methods become visible.

Usage:
    python gen_pyi_stubs.py --db dep-graph/out/dep-graph.sqlite \
                            --out dep-graph/cache/pyi_stubs
"""
from __future__ import annotations

import argparse
import sqlite3
from collections import defaultdict
from pathlib import Path


def _identifier_safe(name: str) -> str:
    """Replace `::`, template brackets, etc. with `_` for valid Python names."""
    if not name:
        return "_"
    return (
        name.replace("::", "_")
            .replace("<", "_")
            .replace(">", "_")
            .replace(",", "_")
            .replace(" ", "")
            .replace("(", "_")
            .replace(")", "_")
            .replace("*", "_")
            .replace("&", "_")
    )


def gather_class_bindings(con: sqlite3.Connection) -> dict[str, dict]:
    """For each helper='class_' binding, build:
        {cpp_qualified_name: {
            "python_name": str,
            "methods":   list[(method_python_name, cpp_qualified_name, helper)],
            "fields":    list[(field_python_name, cpp_qualified_name, helper)],
        }}
    """
    classes: dict[str, dict] = {}
    for r in con.execute("""
        SELECT DISTINCT python_name, cpp_qualified_name
        FROM bindings WHERE helper = 'class_'
    """):
        py_name, cpp_qn = r
        if cpp_qn.startswith("std::"):
            continue  # stdlib classes — skip stubgen
        classes[cpp_qn] = {
            "python_name": py_name,
            "methods": [],
            "fields": [],
        }
    # For each non-class binding, find which class it belongs to (longest-prefix match on cpp_qualified_name).
    for r in con.execute("""
        SELECT python_name, cpp_qualified_name, helper
        FROM bindings WHERE helper <> 'class_'
    """):
        py_name, cpp_qn, helper = r
        if not cpp_qn:
            continue
        # Find the longest class prefix.
        best_class = None
        for class_cpp in classes:
            if cpp_qn.startswith(class_cpp + "::"):
                if best_class is None or len(class_cpp) > len(best_class):
                    best_class = class_cpp
        if best_class is None:
            continue
        if helper in ("def_ro", "def_rw", "def_prop_ro", "def_prop_rw"):
            classes[best_class]["fields"].append((py_name, cpp_qn, helper))
        else:
            classes[best_class]["methods"].append((py_name, cpp_qn, helper))
    return classes


def render_stub(class_cpp: str, info: dict) -> str:
    py_name = info["python_name"]
    safe_name = _identifier_safe(py_name)
    out = []
    out.append(f"class {safe_name}:")
    out.append(f'    """Synthetic stub for C++ class {class_cpp}."""')
    seen_methods: set[str] = set()
    seen_fields: set[str] = set()
    for name, _, _ in info["fields"]:
        s = _identifier_safe(name)
        if s in seen_fields or s == safe_name:
            continue
        seen_fields.add(s)
        out.append(f"    {s}: object")
    for name, _, helper in info["methods"]:
        s = _identifier_safe(name)
        if s in seen_methods or s == safe_name:
            continue
        seen_methods.add(s)
        if helper == "def_static":
            out.append(f"    @staticmethod")
            out.append(f"    def {s}(*args, **kwargs) -> object: ...")
        else:
            out.append(f"    def {s}(self, *args, **kwargs) -> object: ...")
    if len(out) == 2:
        # No members; emit a single pass body so the class is well-formed.
        out.append("    pass")
    out.append("")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="/workspace/dep-graph/out/dep-graph.sqlite")
    ap.add_argument("--out", default="/workspace/dep-graph/cache/pyi_stubs")
    args = ap.parse_args()

    con = sqlite3.connect(args.db)
    classes = gather_class_bindings(con)
    if not classes:
        print("no class bindings found")
        return
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "_ttnn_stubs.pyi"
    rendered = []
    for class_cpp, info in sorted(classes.items()):
        rendered.append(render_stub(class_cpp, info))
    out_path.write_text("\n".join(rendered))
    print(f"wrote {len(classes)} class stubs to {out_path}")


if __name__ == "__main__":
    main()
