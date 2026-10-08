# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Verify archived receipt hashes, smoke equality and published-source provenance."""

import ast
import gzip
import hashlib
import json
from pathlib import Path


def read_original(directory, name):
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe manifest path: {name}")
    path = directory / relative
    if path.is_file():
        return path.read_bytes()
    return gzip.decompress(path.with_name(path.name + ".gz").read_bytes())


def top_level_imports_and_body(source, remove_unused_reference_import=False):
    tree = ast.parse(source)
    imports = []
    body = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            imports.extend(("import", name.name, name.asname) for name in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.extend((f"from:{node.level}:{node.module}", name.name, name.asname) for name in node.names)
        else:
            if remove_unused_reference_import and isinstance(node, ast.FunctionDef) and node.name == "reference":
                first = node.body[0]
                if not isinstance(first, ast.Import) or [(n.name, n.asname) for n in first.names] != [("torch", None)]:
                    raise ValueError("Unexpected archived reference import")
                node.body.pop(0)
            body.append(node)
    tree.body = body
    return sorted(imports, key=repr), ast.dump(tree, include_attributes=False)


def smoke_outputs(report):
    return {
        (case["context"], query["active_tokens"], variant["name"]): variant["output_sha256"]
        for case in report["cases"]
        if case["context"] == 1024
        for query in case["causal_queries"]
        for variant in query["variants"]
    }


def main():
    directory = Path(__file__).resolve().parent
    receipts = directory / "receipts"
    collection = json.loads((receipts / "collection.json").read_text())
    for name, expected in collection["file_sha256"].items():
        actual = hashlib.sha256(read_original(receipts, name)).hexdigest()
        if actual != expected:
            raise ValueError(f"Collected artifact hash mismatch: {name}")
    original = json.loads((receipts / "smoke-v1.json").read_text())
    final = json.loads((receipts / "long-v2.json").read_text())
    before, after = smoke_outputs(original), smoke_outputs(final)
    if len(before) != 6 or before != after:
        raise ValueError("Six smoke outputs are not identical between compiler settings")
    variants = [v for c in final["cases"] for q in c["causal_queries"] for v in q["variants"]]
    if len(variants) != 24 or final["state"] != "completed" or not final["cleanup_completed"]:
        raise ValueError("Final experiment is incomplete")
    for variant in variants:
        metrics = variant["execution_on_quantized"]
        passed = metrics["finite"] and all(
            user["relative_rms"] <= 0.02 and user["pcc"] >= 0.999 for user in metrics["per_user"]
        )
        if not passed or not variant["kernel_gate_passed"]:
            raise ValueError("A recorded execution comparison fails the unchanged kernel gate")
        if not all(variant[name]["finite"] for name in ("quantization_only", "execution_on_quantized", "total")):
            raise ValueError("Nonfinite recorded numerical comparison")
    provenance = json.loads((directory / "publication.json").read_text())
    executed = read_original(receipts, "probe-v2.py")
    published = (directory / provenance["published_probe"]).read_bytes()
    if hashlib.sha256(executed).hexdigest() != provenance["executed_probe_sha256"]:
        raise ValueError("Executed source provenance hash mismatch")
    if hashlib.sha256(published).hexdigest() != provenance["published_probe_sha256"]:
        raise ValueError("Published source provenance hash mismatch")
    executed_imports, executed_body = top_level_imports_and_body(executed, remove_unused_reference_import=True)
    published_imports, published_body = top_level_imports_and_body(published)
    executed_imports.remove(("import", "math", None))
    if executed_imports != published_imports or executed_body != published_body:
        raise ValueError("Published source differs beyond formatting/import order and the two unused import removals")
    print(
        json.dumps(
            dict(
                original_artifact_hashes_verified=len(collection["file_sha256"]),
                identical_smoke_outputs=len(before),
                completed_finite_cases_with_kernel_gate_passed=len(variants),
                published_source_provenance_verified=True,
                experiment_rerun=False,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
