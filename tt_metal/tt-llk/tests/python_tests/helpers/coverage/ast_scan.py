# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Run the LLK AST analyzer once per build context over the whole library."""

import hashlib
import json
import os
import sys
import threading
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


def scope_headers(arch: str) -> list[Path]:
    return sorted((ROOT / path).resolve() for path in headers(ROOT, arch))


def scan_key(context: dict, trisc: str, variant: Path) -> tuple:
    """The library inventory depends on the compiler, its flags and the thread only."""
    context = {k: v for k, v in context.items() if k != "source"}
    # The variant's own include directory holds the generated test kernel, not library headers.
    context["flags"] = [flag for flag in context["flags"] if str(variant) not in flag]
    return json.dumps(context, sort_keys=True), trisc


def library_document(metadata: Path, context: dict, scope: list[Path]) -> dict:
    inventory = llk_ast.scan_context(str(metadata), [str(path) for path in scope])
    document = json.loads(llk_ast.to_json(inventory))
    # Headers that only parse with errors keep their definitions but are reported as unparsed.
    headers = document.pop("headers")
    document.update(
        arch=context["arch"],
        trisc=metadata.name.split(".")[0],
        build_axes=context.get("build_axes", {}),
        complete=all(header["parsed"] for header in headers),
        files=[
            {
                "path": header["path"],
                "parsed": header["parsed"] and not header["errors"],
            }
            for header in headers
        ],
    )
    definitions = []
    for definition in document["definitions"]:
        path = Path(definition.pop("file")).resolve()
        if path not in scope:
            continue
        definition.pop("calls")  # already folded into the assertions by the analyzer
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
    return document


class LibraryScans:
    """One scan per distinct build context, shared by every variant built in it."""

    def __init__(self):
        self.documents = {}
        self.lock = threading.Lock()

    def document(self, metadata: Path) -> dict:
        context = json.loads(metadata.read_text())
        trisc = metadata.name.split(".")[0]
        variant = metadata.parent.parent
        key = scan_key(context, trisc, variant)
        with self.lock:
            if key not in self.documents:
                scope = scope_headers(context["arch"])
                self.documents[key] = library_document(metadata, context, scope)
        return {
            **self.documents[key],
            "test": variant.parent.name,
            "variant": variant.name,
        }

    def variant(self, variant: Path) -> list[dict]:
        return [
            self.document(metadata)
            for metadata in sorted((variant / "elf").glob("*.coverage.json"))
        ]
