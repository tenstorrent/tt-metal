# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Join saved AST definitions with per-instantiation compiler/runtime evidence."""

from collections import defaultdict
from pathlib import Path

from .enumerate import canonical_source, headers


def evidence_status(records: list[dict]) -> str:
    if any(row["execution_count"] > 0 for row in records):
        return "ran"
    if not records:
        return "absent"
    if all(row.get("measurement") == "build_only" for row in records):
        return "unmeasured"
    return "missed"


def source_baseline(
    records: list[dict], root: Path, architectures: list[str], scans: list[dict]
) -> dict:
    definitions, parsed, incomplete = {}, defaultdict(set), []
    for scan in scans:
        if scan["arch"] not in architectures:
            continue
        if not scan["complete"]:
            incomplete.append(
                {
                    key: scan[key]
                    for key in ("arch", "trisc", "test", "variant", "diagnostics")
                }
            )
            continue
        parsed[scan["arch"]].update(
            canonical_source(file["path"], root)
            for file in scan["files"]
            if file["parsed"]
        )
        for definition in scan["definitions"]:
            definitions[definition["key"]] = {
                **definition,
                "header": canonical_source(definition["header"], root),
            }

    indexed, unmatched = defaultdict(list), {}
    for record in records:
        key = record.get("definition")
        if key in definitions:
            indexed[key].append(record)
        else:
            unmatched[(record["source"], record["signature"])] = {
                "source": record["source"],
                "signature": record["signature"],
                "reason": "No complete AST definition for this build; rebuild with coverage",
            }

    stale = set()
    for key, definition in definitions.items():
        entries = indexed[key]
        definition["status"] = evidence_status(entries)
        if any(
            row.get("source_digest") != definition["source_digest"] for row in entries
        ):
            definition["status"] = "stale"
            stale.add(definition["header"])
        definition["emitted_instantiations"] = len({row["symbol"] for row in entries})
        definition["executed_instantiations"] = len(
            {row["symbol"] for row in entries if row["execution_count"] > 0}
        )
        definition["threads"] = [definition["trisc"]]
        definition["tests"] = sorted(
            {
                row.get("run", {}).get("test") or row["test"]
                for row in entries
                if row["execution_count"] > 0
            }
        )

    unparsed = {
        arch: sorted(headers(root, arch) - parsed[arch]) for arch in architectures
    }
    return {
        "method": "Clang AST definitions, parameters and decisions, including uninstantiated templates",
        "scope": "All explicit function definitions in architecture llk_lib, common/inc, shared common and metal llk_sfpu headers",
        "condition_policy": "Active preprocessor branches per saved build context; unparsed headers and failed scans are listed separately.",
        "complete": bool(scans)
        and not incomplete
        and not any(unparsed.values())
        and not unmatched
        and not stale,
        "definitions": sorted(
            definitions.values(),
            key=lambda d: (d["arch"], d["header"], d["line"], d["key"]),
        ),
        "unparsed_headers": unparsed,
        "incomplete_scans": incomplete,
        "unmatched_observations": list(unmatched.values()),
        "stale_sources": sorted(stale),
        "counts": {
            state: sum(d["status"] == state for d in definitions.values())
            for state in ("ran", "missed", "unmeasured", "absent", "stale")
        },
    }
