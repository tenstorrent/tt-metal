"""Read-only source census; generated inventory is not a runtime dtype trace.

Run from any directory. Inventories tracked Python operator references/calls,
excluding research snapshots. Manual companion reports resolve input/config
provenance and indirect calls. JSON output avoids pretending runtime-dependent
settings are statically known.
"""

import ast
import collections
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
OPS = {
    "scaled_dot_product_attention", "chunked_scaled_dot_product_attention",
    "joint_scaled_dot_product_attention", "ring_joint_scaled_dot_product_attention",
    "exp_ring_joint_scaled_dot_product_attention", "ring_distributed_scaled_dot_product_attention",
    "sparse_sdpa", "sparse_sdpa_msa", "ring_mla", "flash_mla_prefill", "chunked_flash_mla_prefill",
    "scaled_dot_product_attention_composite", "sdpa",
}


def dotted(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = dotted(node.value)
        return f"{base}.{node.attr}" if base else None
    return None


def main():
    files = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    records, failures = [], []
    for filename in files:
        if not filename.endswith(".py") or filename.startswith("experiments/"):
            continue
        path = ROOT / filename
        try:
            tree = ast.parse(path.read_text(), filename=filename)
        except (OSError, SyntaxError, UnicodeError) as exc:
            failures.append({"file": filename, "error": str(exc)})
            continue
        imports = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    imports[alias.asname or alias.name] = f"{node.module}.{alias.name}"
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imports[alias.asname or alias.name.split(".")[0]] = alias.name if alias.asname else alias.name.split(".")[0]
        parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
        for node in ast.walk(tree):
            name = dotted(node)
            if not name or not isinstance(getattr(node, "ctx", None), ast.Load):
                continue
            prefix, _, suffix = name.partition(".")
            resolved = imports.get(prefix, prefix) + ("." + suffix if suffix else "")
            if resolved.split(".")[-1] not in OPS:
                continue
            parent = parents.get(node)
            if isinstance(parent, ast.Attribute):
                continue
            call = parent if isinstance(parent, ast.Call) and parent.func is node else None
            family = ("ttnn" if resolved.startswith("ttnn.") else
                      "ttml" if resolved.startswith("ttml.") else
                      "torch_reference" if resolved.startswith(("torch.", "torchvision.")) else
                      "indirect_or_other")
            records.append({
                "file": filename, "line": node.lineno, "callee": name, "resolved": resolved,
                "family": family, "kind": "call" if call else "reference",
                "kwargs": {kw.arg or "**": ast.unparse(kw.value) for kw in call.keywords} if call else {},
            })
    records.sort(key=lambda x: (x["file"], x["line"], x["callee"]))
    result = {
        "scope": "Tracked Python source, excluding experiments; non-decode operator names only",
        "caveat": "References include function-valued dispatch; unresolved wrappers need companion manual audit. Counts are static sites, not runtime calls.",
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "family_counts": dict(collections.Counter(x["family"] for x in records)),
        "records": records, "parse_failures": failures,
    }
    output = Path(__file__).with_name("python-census.json")
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
