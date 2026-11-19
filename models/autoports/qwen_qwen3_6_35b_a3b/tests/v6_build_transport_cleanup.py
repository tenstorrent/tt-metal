# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Build isolated ab28 transport-only diagnostics; never imports TTNN or Torch."""

import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / "doc/fused_decoder/endurance"
BASE = E / "v6_corrected_fma_full_v5_triangular_compact_silu_formatted_integration.py.snapshot"
BASE_SHA = "ab28a91bb3cb5e8b88e5371da0defe3b0d731e86a131c8523483e4b6b65b0334"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def named(scope, name):
    return next(n for n in scope.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == name)


def digest(node):
    return hashlib.sha256(ast.dump(node).encode()).hexdigest()


def build(variant):
    assert variant in ("placement", "indices", "combined")
    assert sha(BASE) == BASE_SHA
    text = BASE.read_text()
    original = ast.parse(text)
    cls = named(original, "FusedDecoder")
    edits = []
    if variant in ("placement", "combined"):
        fn = named(cls, "_forward")
        # Direct body only: never descend into the preserved full-kind guard.
        before = next(
            n
            for n in fn.body[1:]
            if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "mlp_input" and isinstance(n.value, ast.IfExp)
        )
        assert ast.unparse(before) == "mlp_input = ttnn.to_memory_config(x, self.norm_memory) if decode else x"
        after = next(
            n
            for n in fn.body[1:]
            if isinstance(n, ast.If)
            and ast.unparse(n.test) == "decode"
            and ast.unparse(n.body[0]) == "mlp_input = ttnn.to_memory_config(mlp_input, DRAM)"
        )
        edits.extend([(before, "        mlp_input = x\n"), (after, "")])
    if variant in ("indices", "combined"):
        fn = named(cls, "_moe")
        node = next(
            n
            for n in ast.walk(fn)
            if isinstance(n, ast.Assign)
            and ast.unparse(n.targets[0]) == "indices"
            and ast.unparse(n.value) == "reshape(ids, (1, 8))"
        )
        assert node.lineno > fn.body[0].end_lineno
        edits.append((node, "            indices = ttnn.reshape(ids, (1, 8))\n"))
    lines = text.splitlines(keepends=True)
    changes = []
    for node, replacement in sorted(edits, key=lambda edit: edit[0].lineno, reverse=True):
        old = "".join(lines[node.lineno - 1 : node.end_lineno])
        changes.append(dict(line=node.lineno, before=old, after=replacement))
        lines[node.lineno - 1 : node.end_lineno] = [replacement]
    result = "".join(lines)
    candidate = ast.parse(result)
    restored = copy.deepcopy(candidate)
    restored_cls = named(restored, "FusedDecoder")
    candidate_cls = named(candidate, "FusedDecoder")
    changed_names = []
    for name in ("_forward", "_moe"):
        old_fn, new_fn = named(cls, name), named(restored_cls, name)
        # Full-kind branches remain byte-for-byte AST identical.
        assert ast.dump(old_fn.body[0]) == ast.dump(new_fn.body[0])
        if ast.dump(old_fn) != ast.dump(new_fn):
            changed_names.append(name)
            restored_cls.body[restored_cls.body.index(new_fn)] = copy.deepcopy(old_fn)
    assert ast.dump(restored) == ast.dump(original), "No helper, setup, arithmetic, or other method changes"
    # Prove literal reverse replacements restore every source byte, including kernels.
    restored_lines = result.splitlines(keepends=True)
    shift = 0
    inverse_spans = []
    for change in sorted(changes, key=lambda item: item["line"]):
        start = change["line"] - 1 + shift
        added = len(change["after"].splitlines(keepends=True))
        inverse_spans.append((start, added, change["before"]))
        shift += added - len(change["before"].splitlines(keepends=True))
    for start, added, old in reversed(inverse_spans):
        restored_lines[start : start + added] = old.splitlines(keepends=True)
    assert "".join(restored_lines) == text
    # Exact forward edits are independently reconstructible from original line spans.
    assert len(changes) == (3 if variant == "combined" else 2 if variant == "placement" else 1)
    for change in changes:
        assert change["before"]
    # Arithmetic and public dispatch bodies are unchanged except reviewed transport sites.
    unchanged = {n.name: digest(n) for n in cls.body if isinstance(n, ast.FunctionDef) and n.name not in changed_names}
    assert all(digest(named(candidate_cls, name)) == value for name, value in unchanged.items())
    compile(result, str(BASE) + ":" + variant, "exec")
    report = dict(
        status="source_only_diagnostic",
        acceptance_eligible=False,
        device_executed=False,
        variant=variant,
        base=str(BASE),
        base_sha256=BASE_SHA,
        changes=changes,
        changed_methods=changed_names,
        unchanged_method_ast_sha256=unchanged,
        full_kind_guards_unchanged=True,
        exact_site_reverse_restores_all_source_bytes=True,
        full_module_ast_restored_after_replacing_only_changed_methods=True,
        runtime_added_calls=["ttnn.reshape"] if variant != "placement" else [],
        arithmetic_boundaries_changed=False,
        setup_and_kernel_strings_unchanged=True,
        prefill_scope="Placement removal applies only where decode=True; indices change includes public singleton prefill. Multi-token routing unchanged.",
        snapshot_sha256=hashlib.sha256(result.encode()).hexdigest(),
    )
    return result, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=E)
    args = parser.parse_args()
    for variant in ("placement", "indices", "combined"):
        output = args.output_dir / f"v6_ab28_transport_{variant}.py.snapshot"
        report_path = args.output_dir / f"v6_ab28_transport_{variant}_build.json"
        assert not output.exists() and not report_path.exists(), "Preserve evidence"
        code, report = build(variant)
        output.write_text(code)
        report.update(snapshot=str(output), builder_sha256=sha(__file__), builder=str(Path(__file__)))
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print(variant, report["snapshot_sha256"])


if __name__ == "__main__":
    main()
