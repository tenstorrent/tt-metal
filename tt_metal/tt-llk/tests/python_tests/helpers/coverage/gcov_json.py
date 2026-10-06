# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Read gcov's per-function view, retaining raw names and branch identities."""

import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path


def observation_key(record: dict) -> tuple:
    """Keep different build axes and instrumentation layouts separate."""
    return (
        record["arch"],
        record["trisc"],
        json.dumps(record.get("build_axes", {}), sort_keys=True),
        record["source"],
        record["symbol"],
        record["graph"],
        record.get("definition") or "",
    )


def read_gcov(gcov: str, notes: Path) -> dict:
    result = subprocess.run(
        [
            gcov,
            "--json-format",
            "--stdout",
            "--branch-probabilities",
            "--branch-counts",
            str(notes.resolve()),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    document = json.loads(result.stdout)
    if str(document.get("format_version")) not in {"1", "2"}:
        raise ValueError(
            f"Unsupported gcov JSON version: {document.get('format_version')}"
        )
    return document


def function_records(document: dict, instances: dict):
    """One record per gcov function that the AST scan identified."""
    for source in document["files"]:
        source_lines = None
        source_digest = None
        source_path = (
            Path(document.get("current_working_directory", ".")) / source["file"]
        ).resolve()
        lines_by_function = {}
        for line in source["lines"]:
            lines_by_function.setdefault(line.get("function_name"), []).append(line)
        for function in source["functions"]:
            instance = instances.get(function["name"])
            if instance is None:
                continue
            if source_lines is None:
                try:
                    source_text = source_path.read_text()
                    source_lines = source_text.splitlines()
                    source_digest = hashlib.sha256(source_text.encode()).hexdigest()
                except (OSError, UnicodeError):
                    source_lines = []
            lines = [dict(line) for line in lines_by_function.get(function["name"], [])]
            for line in lines:
                number = line["line_number"]
                if 0 < number <= len(source_lines):
                    line["source_text"] = source_lines[number - 1]
            # This distinguishes incompatible instrumentation, NOT semantic equivalence.
            topology = [
                [
                    line["line_number"],
                    [
                        [
                            branch.get("source_block_id"),
                            branch.get("destination_block_id"),
                            branch.get("fallthrough"),
                            branch.get("throw"),
                        ]
                        for branch in line.get("branches", [])
                    ],
                ]
                for line in lines
            ]
            graph = hashlib.sha256(
                json.dumps([function["blocks"], topology], sort_keys=True).encode()
            ).hexdigest()
            yield {
                "symbol": function["name"],
                "signature": function.get("demangled_name", function["name"]),
                "function": instance["function"],
                "arguments": instance["arguments"],
                "definition": instance["definition"],
                "source": source["file"],
                "source_absolute": str(source_path),
                "source_digest": source_digest,
                "start_line": function["start_line"],
                "blocks": function["blocks"],
                "blocks_executed": function["blocks_executed"],
                "execution_count": function["execution_count"],
                "graph": graph,
                "lines": lines,
            }


def initial_document(gcov: str, notes: Path) -> dict:
    """Read build-only evidence without accidentally consuming leftover counters."""
    with tempfile.TemporaryDirectory(prefix="llk-gcno-") as directory:
        copied = Path(directory) / notes.name
        shutil.copyfile(notes, copied)
        return read_gcov(gcov, copied)


def collect_variant(
    gcov: str, variant: Path, arch: str, initial: bool = False
) -> list[dict]:
    records = []
    notes_files = sorted((variant / "elf").glob("*.gcno"))
    if not notes_files:
        raise ValueError(f"No gcno files in {variant / 'elf'}")
    for notes in notes_files:
        match = re.match(r"(unpack|math|pack|sfpu)(?:[.-]|$)", notes.name)
        if match is None:
            raise ValueError(f"Cannot identify TRISC for {notes}")
        metadata = notes.parent / f"{match.group(1)}.coverage.json"
        context = json.loads(metadata.read_text()) if metadata.exists() else {}
        if context.get("arch", arch) != arch:
            raise ValueError(
                f"Architecture mismatch in {metadata}: expected {arch}, found {context['arch']}"
            )
        if arch == "quasar" and not context:
            raise ValueError(
                f"Missing Quasar build axes: {metadata}; rebuild with coverage"
            )
        document = initial_document(gcov, notes) if initial else read_gcov(gcov, notes)
        ast_path = metadata.with_suffix(".ast.json")
        scan = json.loads(ast_path.read_text()) if ast_path.exists() else {}
        instances = scan.get("instances", {}) if scan.get("complete") else {}
        for record in function_records(document, instances):
            record.update(
                arch=arch,
                trisc=match.group(1),
                test=variant.parent.name,
                variant=variant.name,
                object=notes.name,
                build_axes=context.get("build_axes", {}),
                measurement=(
                    "build_only"
                    if initial or not notes.with_suffix(".gcda").exists()
                    else "runtime"
                ),
            )
            records.append(record)
    return records
