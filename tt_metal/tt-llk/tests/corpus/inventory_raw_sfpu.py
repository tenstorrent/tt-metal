#!/usr/bin/env python3
"""Deterministic source inventory for raw-SFPU migration review.

This counts source spellings, not instantiated code or validated regions.
Includes experimental headers and inactive preprocessor branches. It does not
infer register masks, approve a migration, or gate compilation/execution.
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path


LEXEME = re.compile(r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'')
RAW = re.compile(r"\bTT(?:I|_OP)?_(SFP[A-Z0-9_]+)\b")
REPLAY = re.compile(r"\b(?:lltt::(?:record\w*|replay)|TT(?:I|_OP)?_REPLAY)\b")


def without_comments_and_literals(source):
    # Preserve line numbers while excluding examples in comments/strings.
    return LEXEME.sub(lambda match: "\n" * match[0].count("\n") or " ", source)


def inventory(root):
    rows = []
    for arch in ("blackhole", "wormhole_b0", "quasar"):
        directory = root / f"tt_llk_{arch}" / "common/inc/sfpu"
        for path in sorted(directory.rglob("*.h")):
            source = without_comments_and_literals(path.read_text())
            sites = [
                {"line": source.count("\n", 0, match.start()) + 1,
                 "spelling": match[0], "operation": match[1]}
                for match in RAW.finditer(source)
            ]
            if not sites:
                continue
            operations = Counter(site["operation"] for site in sites)
            rows.append({
                "path": str(path.relative_to(root)),
                "architecture": arch,
                "operations": dict(sorted(operations.items())),
                "configured_loadmacro": "SFPLOADMACRO" in operations,
                "replay_spelling_present": bool(REPLAY.search(source)),
                "lreg_api_spelling_present": bool(re.search(r"\bl_reg\s*\[", source)),
                "sites": sites,
            })
    return {
        "scope": "three architecture common/inc/sfpu trees, including experimental",
        "meaning": "source sites only; not instantiated, migrated, or verified coverage",
        "headers_by_architecture": dict(sorted(Counter(row["architecture"] for row in rows).items())),
        "headers": len(rows),
        "source_sites": sum(len(row["sites"]) for row in rows),
        "rows": rows,
    }


def self_test():
    source = '// TTI_SFPLOAD(0)\n"TT_SFPSTORE"; /* TT_OP_SFPMAD */\nTTI_SFPNOP; TT_OP_SFPLOAD(0);'
    clean = without_comments_and_literals(source)
    assert clean.count("\n") == source.count("\n")
    assert [match[1] for match in RAW.finditer(clean)] == ["SFPNOP", "SFPLOAD"]
    assert REPLAY.search("lltt::record_exec(0)")
    assert REPLAY.search("TTI_REPLAY(0)")
    assert not REPLAY.search(without_comments_and_literals('// lltt::replay(0)'))
    print("PASS: comment/literal exclusion, line preservation, raw encodings and replay spellings")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llk-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        print(json.dumps(inventory(args.llk_root), indent=2, sort_keys=True))
