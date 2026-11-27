#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Generate compute_api_order_sfpu.yaml from the unary-op source generator.

`ttnn/.../unary_op_utils.cpp` emits each SFPU op as a pair of adjacent source
strings: `<op>_tile_init();` and `<op>_tile(...);`.  Those are the only names a
compute kernel actually calls for SFPU, and the LLK sanitizer does not model
SFPU, so the rule table mines them here.  The tests re-run this generator and
fail if the committed YAML is stale.
"""

from __future__ import annotations

import collections
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC = "ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp"
OUT = "tt_metal/tt-llk/infra/compute_api_order_sfpu.yaml"

_PAIR_RE = re.compile(
    r'"([A-Za-z_][A-Za-z0-9_]*_init[^"]*)"[^;{}]{0,400}?"'
    r'([A-Za-z_][A-Za-z0-9_]*(?:_tile|_int|_uint)[A-Za-z0-9_]*)\s*(?:<|\()',
    re.S,
)


def pairs() -> dict[str, dict]:
    text = (REPO_ROOT / SRC).read_text()
    clean = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    clean = re.sub(r"//[^\n]*", "", clean)
    groups: dict[str, dict] = collections.OrderedDict()
    for m in _PAIR_RE.finditer(clean):
        init = m.group(1).split("<")[0].split("(")[0].strip()
        op = m.group(2).strip()
        if op.endswith("_init"):
            continue
        line = text[: m.start()].count("\n") + 1
        g = groups.setdefault(init, {"ops": set(), "line": line})
        g["ops"].add(op)
        g["line"] = min(g["line"], line)
    return groups


def build_yaml() -> str:
    groups = pairs()
    lines = [
        "# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.",
        "# SPDX-License-Identifier: Apache-2.0",
        "#",
        "# GENERATED FILE - do not edit by hand.",
        "# Regenerate with: python3 tt_metal/tt-llk/infra/gen_compute_api_order_sfpu.py",
        "#",
        "# SFPU *_tile_init / *_tile pairs mined from",
        f"#   {SRC}",
        "# which emits both calls as adjacent source strings. The LLK sanitizer does not",
        "# model SFPU, so these rules are not part of the sanitizer-agreement test.",
        "sfpu_groups:",
    ]
    for init, g in groups.items():
        ops = ", ".join(sorted(g["ops"]))
        lines.append(f"  - id: sfpu.{init.removesuffix('_init')}")
        lines.append(f"    init: [{init}]")
        lines.append(f"    execute: [{ops}]")
        lines.append(f"    citation: {SRC}:{g['line']}")
    return "\n".join(lines) + "\n"


def main() -> int:
    out = REPO_ROOT / OUT
    out.write_text(build_yaml())
    print(f"wrote {OUT}: {len(pairs())} groups")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
