# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Render the vision PCC log (tests/vision/vision_test_utils.pcc_record) as markdown tables.

    python models/demos/pplx_decider_v1_27b/tests/vision/vision_pcc_report.py [pcc_vision.jsonl]

Table 1: teacher-forced modules (patch embed, 27 blocks, merger) x image, min per row.
Table 2: whole tower from pixels (gate) and the free-running hidden-state PCC after selected blocks.
"""

import json
import sys
from pathlib import Path

from models.demos.pplx_decider_v1_27b.tests.vision.vision_test_utils import IMAGES, PCC_LOG

SHORT = [i.split("_")[0] for i in IMAGES]


def _latest(log: Path) -> dict:
    latest = {}
    for line in log.read_text().splitlines():
        r = json.loads(line)
        latest[(r["module"], r["block"], r["image"])] = r
    return latest


def _row(latest, module, block, label, digits=6):
    cells, vals = [], []
    for image in IMAGES:
        r = latest.get((module, block, image))
        if r is None:
            cells.append("")
            continue
        vals.append(r["pcc"])
        cells.append(f"{r['pcc']:.{digits}f}" + ("" if r["passed"] else " FAIL"))
    worst = f"{min(vals):.{digits}f}" if vals else ""
    return f"| {label} | " + " | ".join(cells) + f" | {worst} |"


def render(log: Path) -> str:
    latest = _latest(log)
    head = "| module | " + " | ".join(SHORT) + " | min |"
    sep = "|---|" + "---:|" * (len(SHORT) + 1)
    out = ["Teacher forced (each module fed the HF golden input), bar 0.995:", "", head, sep]
    out.append(_row(latest, "patch_embed", None, "patch embed (Conv3d as matmul)"))
    out.append(_row(latest, "patch_embed_pos", None, "patch embed + pos embed"))
    for b in range(27):
        out.append(_row(latest, "block", b, f"block {b}"))
    out.append(_row(latest, "merger", None, "merger"))
    out += ["", "Whole tower from pixels only (gate: features, bar 0.99) and free-running hidden-state drift:"]
    out += ["", head, sep]
    out.append(_row(latest, "tower_e2e", None, "**tower features (gate)**"))
    for b in (0, 6, 13, 20, 24, 25, 26):
        out.append(_row(latest, "tower_e2e_block", b, f"hidden after block {b}"))
    gated_modules = ("patch_embed", "patch_embed_pos", "block", "merger", "tower_e2e")
    gated = [r for r in latest.values() if r["module"] in gated_modules]
    out += ["", f"Gated cases: {len(gated)}; failures: {sum(not r['passed'] for r in gated)}."]
    extra = sorted((r for r in latest.values() if r["module"].startswith("mask_")), key=lambda r: r["module"])
    if extra:
        out += ["", "Key-mask checks: " + "; ".join(f"{r['module']} {r['pcc']:.6f}" for r in extra)]
    return "\n".join(out)


if __name__ == "__main__":
    print(render(Path(sys.argv[1]) if len(sys.argv) > 1 else PCC_LOG))
