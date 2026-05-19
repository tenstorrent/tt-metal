# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Render the optimizer status board for a run.

Each block is one of:
  GREEN     - no remaining findings (or all suppressed)
  YELLOW    - findings present but no patch applied
  RED       - patch applied but verify failed (needs revert)
  APPLIED   - patch on disk, no fresh re-collect yet
  PENDING   - block hasn't been diagnosed yet
  N/A       - block's `requires` are not GREEN
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .collect import find_run
from .optimizers.catalog import list_blocks
from .runner import PATCHES_SUBDIR, diagnose_all


@dataclass
class BlockStatus:
    name: str
    level: int
    status: str
    finding_count: int
    patch_count: int
    requires: List[str]
    notes: str = ""


def _patch_files_for_block(patch_dir: Path, block: str) -> List[Path]:
    if not patch_dir.exists():
        return []
    return sorted(patch_dir.glob(f"{block}__*.patch"))


def compute_status(run_id: str, run_dir_root: Optional[Path] = None) -> List[BlockStatus]:
    run_dir = find_run(run_id, run_dir_root)
    patch_dir = run_dir / PATCHES_SUBDIR
    findings_by_block = diagnose_all(run_id, run_dir_root=run_dir_root)

    statuses: Dict[str, BlockStatus] = {}
    for entry in list_blocks():
        findings = findings_by_block.get(entry.name, [])
        patches = _patch_files_for_block(patch_dir, entry.name)
        if not findings and not patches:
            status = "GREEN"
        elif patches and not findings:
            status = "GREEN"  # findings were resolved by the patch
        elif patches and findings:
            status = "APPLIED"  # patches exist but new findings remain
        elif findings:
            status = "YELLOW"
        else:
            status = "PENDING"
        statuses[entry.name] = BlockStatus(
            name=entry.name,
            level=entry.level,
            status=status,
            finding_count=len(findings),
            patch_count=len(patches),
            requires=list(entry.requires),
            notes="",
        )

    # Mark N/A blocks whose `requires` aren't all GREEN.
    for entry in list_blocks():
        st = statuses[entry.name]
        for req in st.requires:
            req_status = statuses.get(req)
            if req_status and req_status.status not in ("GREEN", "APPLIED"):
                st.status = "N/A"
                st.notes = f"requires {req} (currently {req_status.status})"
                break

    return [statuses[e.name] for e in list_blocks()]


def render_status_board(run_id: str, run_dir_root: Optional[Path] = None) -> str:
    rows = compute_status(run_id, run_dir_root=run_dir_root)
    out: List[str] = []
    out.append(f"=== perf status: {run_id} ===")
    out.append(f"{'NAME':<28s} {'L':>2s}  {'STATUS':<9s}  {'FINDINGS':>9s}  {'PATCHES':>8s}  NOTES")
    out.append("-" * 90)
    glyphs = {
        "GREEN": "\033[32mGREEN\033[0m",
        "YELLOW": "\033[33mYELLOW\033[0m",
        "RED": "\033[31mRED\033[0m",
        "APPLIED": "\033[36mAPPLIED\033[0m",
        "PENDING": "PENDING",
        "N/A": "N/A",
    }
    plain = {k: k for k in glyphs}
    use_glyphs = plain  # avoid ANSI in static reports; CLI can switch via term check
    import sys

    if sys.stdout.isatty():
        use_glyphs = glyphs
    for r in rows:
        out.append(
            f"{r.name:<28s}  L{r.level} {use_glyphs[r.status]:<18s}  {r.finding_count:>9d}  {r.patch_count:>8d}  {r.notes}"
        )
    return "\n".join(out)
