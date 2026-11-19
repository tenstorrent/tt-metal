# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Exact historical reconstruction for one unused PLE CLI-export source change.

The capacity probe imports only load_reference. Reconstruct the complete old
module, requiring its exact recorded SHA256, by removing only two stdlib imports
and the optional CLI argument/export blocks inside main. No runtime or executed
reference helper may change. This proof is specific to this importer/module and
does not authorize arbitrary function-level source exemptions.
"""

import argparse
import ast
import difflib
import hashlib
import json
from pathlib import Path

from .validate_trace_evidence import digest, require

DEPENDENCY = "tests/probe_ple_decoder.py"
IMPORTER = "tests/probe_decoder_capacity.py"
TARGET_MODULE = "models.autoports.qwen_qwen3_8_flash_next.tests.probe_ple_decoder"


def text_digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def prove_import_closure(root, provenance_path):
    record = json.loads(provenance_path.read_text())
    require(record["source_sha256"][IMPORTER] == digest(root / IMPORTER), "Capacity importer changed")
    importer = ast.parse((root / IMPORTER).read_text())
    imports = [node for node in ast.walk(importer) if isinstance(node, ast.ImportFrom) and node.module == TARGET_MODULE]
    require(
        len(imports) == 1 and [(name.name, name.asname) for name in imports[0].names] == [("load_reference", None)],
        "Capacity importer must import only unchanged load_reference",
    )
    require(
        not any(
            isinstance(node, ast.Import) and any(name.name == TARGET_MODULE for name in node.names)
            for node in ast.walk(importer)
        ),
        "Unexpected module-wide import",
    )
    source = (root / DEPENDENCY).read_text()
    tree = ast.parse(source)
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    parser_blocks, export_blocks = [], []
    for node in ast.walk(main):
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and any(isinstance(arg, ast.Constant) and arg.value == "--device-output-cache" for arg in node.value.args)
        ):
            parser_blocks.append(node)
        if isinstance(node, ast.If) and ast.unparse(node.test) == "args.device_output_cache is not None":
            export_blocks.append(node)
    require(
        len(parser_blocks) == len(export_blocks) == 1, "Expected exactly the optional CLI argument and export blocks"
    )
    imports_added = [
        node
        for node in tree.body
        if isinstance(node, ast.Import)
        and [(name.name, name.asname) for name in node.names] in [[("hashlib", None)], [("inspect", None)]]
    ]
    require(len(imports_added) == 2, "Expected only two added stdlib imports")
    removed = parser_blocks + export_blocks + imports_added
    historical = "".join(
        line
        for number, line in enumerate(source.splitlines(keepends=True), 1)
        if not any(node.lineno <= number <= node.end_lineno for node in removed)
    )
    old_hash = record["source_sha256"][DEPENDENCY]
    require(text_digest(historical) == old_hash, "Exact historical full-file hash reconstruction failed")
    old_tree = ast.parse(historical)

    def import_context(module):
        return ast.dump(
            ast.Module(
                body=[
                    node
                    for node in module.body
                    if not (isinstance(node, ast.FunctionDef) and node.name == "main")
                    and not (
                        isinstance(node, ast.Import)
                        and [(name.name, name.asname) for name in node.names]
                        in [[("hashlib", None)], [("inspect", None)]]
                    )
                ],
                type_ignores=[],
            ),
            include_attributes=False,
        )

    require(import_context(tree) == import_context(old_tree), "Import-time context or non-main definition changed")
    unchanged = {}
    old_functions = {node.name: node for node in old_tree.body if isinstance(node, ast.FunctionDef)}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name != "main":
            current_text = ast.get_source_segment(source, node)
            require(
                current_text == ast.get_source_segment(historical, old_functions[node.name]),
                f"Function text changed: {node.name}",
            )
            unchanged[node.name] = text_digest(current_text)
    require("load_reference" in unchanged, "Missing imported function proof")
    return {
        "scope": __doc__,
        "dependency": DEPENDENCY,
        "importer": IMPORTER,
        "recorded_dependency_sha256": old_hash,
        "current_dependency_sha256": text_digest(source),
        "historical_source_reconstructed_sha256": text_digest(historical),
        "importer_sha256": digest(root / IMPORTER),
        "recorded_argv": record["argv"],
        "recorded_started_utc": record["started_utc"],
        "import_context_ast_sha256": text_digest(import_context(tree)),
        "unchanged_function_text_sha256": unchanged,
        "added_imports": ["hashlib", "inspect"],
        "historical_to_current_diff": "".join(
            difflib.unified_diff(
                historical.splitlines(keepends=True),
                source.splitlines(keepends=True),
                fromfile="recorded/probe_ple_decoder.py",
                tofile="current/probe_ple_decoder.py",
            )
        ),
        "imported_execution_unchanged": True,
        "passed": True,
        "validator_sha256": digest(Path(__file__)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = prove_import_closure(args.root.resolve(), args.provenance)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps({key: report[key] for key in ("passed", "recorded_dependency_sha256", "current_dependency_sha256")})
    )


if __name__ == "__main__":
    main()
