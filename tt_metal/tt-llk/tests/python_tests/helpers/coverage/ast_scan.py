# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Run the LLK AST analyzer on the real build's context and keep the result beside the ELF."""

import hashlib
import json
import os
import sys
from pathlib import Path

from .enumerate import headers

ROOT = Path(__file__).resolve().parents[4]

if os.environ.get("LLK_AST_ANALYZER"):
    sys.path.insert(0, str(Path(os.environ["LLK_AST_ANALYZER"]) / "python"))
try:
    import llk_ast
except ImportError as error:
    raise ImportError(
        "Coverage needs the llk-ast-analyzer Python module: run `make` in its checkout "
        "and set LLK_AST_ANALYZER to that directory (or add its python/ to PYTHONPATH)"
    ) from error


def scan_context(metadata: Path) -> dict:
    context = json.loads(metadata.read_text())
    inventory = llk_ast.scan_context(str(metadata))
    metadata.with_suffix(".log").write_text(inventory.diagnostics)
    document = json.loads(llk_ast.to_json(inventory))
    document.update(
        arch=context["arch"],
        trisc=metadata.name.split(".")[0],
        build_axes=context.get("build_axes", {}),
        test=metadata.parent.parent.parent.name,
        variant=metadata.parent.parent.name,
        diagnostics=inventory.diagnostics,
        files=[{"path": path, "parsed": True} for path in inventory.files]
        + [{"path": path, "parsed": False} for path in inventory.missing_files],
    )
    definitions = []
    scope = {(ROOT / path).resolve() for path in headers(ROOT, context["arch"])}
    for definition in document["definitions"]:
        path = Path(definition.pop("file")).resolve()
        if path not in scope:
            continue
        definition.update(
            arch=context["arch"],
            header=str(path),
            template_parameters=[
                parameter["declaration"] for parameter in definition["parameters"]
            ],
            trisc=document["trisc"],
            build_axes=document["build_axes"],
        )
        for parameter in definition["parameters"] + definition["runtime_parameters"]:
            for value in parameter["values"]:
                value["value"] = int(value["value"])
        definition["key"] = hashlib.sha256(
            json.dumps(definition, sort_keys=True).encode()
        ).hexdigest()
        definitions.append(definition)
    document["definitions"] = definitions
    by_id = {definition["id"]: definition for definition in definitions}
    document["instances"] = {
        symbol: {
            **instance,
            "definition": by_id[instance["definition"]]["key"],
            "function": by_id[instance["definition"]]["function"],
        }
        for symbol, instance in document["instances"].items()
        if instance["definition"] in by_id
    }
    metadata.with_suffix(".ast.json").write_text(json.dumps(document) + "\n")
    return document


def saved_scans(variant: Path) -> list[dict]:
    return [
        json.loads(path.read_text())
        for path in sorted((variant / "elf").glob("*.coverage.ast.json"))
    ]
