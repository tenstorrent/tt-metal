# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Shared CI summary plumbing for DeepSeek/GLM/Kimi prefill tests.

Every prefill summary (per-layer PCC, chunk-timing perf, ...) lands under one root, PREFILL_SUMMARIES,
in a per-kind subdir (PREFILL_SUMMARIES/pcc, PREFILL_SUMMARIES/perf), one file per parameterized run so a
sweep never clobbers itself. A CI publish step globs a subdir and concatenates the files into
$GITHUB_STEP_SUMMARY.
"""

import getpass
import os
from pathlib import Path

from loguru import logger


def summaries_root() -> Path:
    """Root for all prefill CI summaries. A shared fixed dir is owned by whoever creates it first and
    raises PermissionError for everyone else, so the default is per-user."""
    return Path(os.getenv("PREFILL_SUMMARIES", f"/tmp/prefill_summaries_{getpass.getuser()}"))


def summary_dir(kind: str) -> Path:
    """PREFILL_SUMMARIES/<kind>, created if absent."""
    d = (summaries_root() / kind).resolve()
    d.mkdir(parents=True, exist_ok=True)
    return d


def render_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    """Fixed-width ASCII grid table (borders + header + rows) as a list of lines. `headers` and each row
    are already-formatted cell strings; columns widen to the longest cell and are left-justified."""
    widths = [len(h) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))
    sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"

    def render_row(values: list[str]) -> str:
        return "| " + " | ".join(v.ljust(widths[idx]) for idx, v in enumerate(values)) + " |"

    return [sep, render_row(headers), sep, *[render_row(r) for r in rows], sep]


def _is_primary_rank() -> bool:
    """True on MPI rank 0 (or a non-MPI run). Under `mpirun --pernode` every node runs the whole test,
    so without this every rank would race-write the same file on the shared volume."""
    for var in ("OMPI_COMM_WORLD_RANK", "PMIX_RANK", "PMI_RANK"):
        rank = os.environ.get(var)
        if rank is not None:
            return rank == "0"
    return True


def emit_summary(kind: str, run_name: str, title: str, lines: list[str]) -> Path | None:
    """On rank 0 only: print the block to stdout AND persist it as PREFILL_SUMMARIES/<kind>/<run_name>.md.
    stdout (not loguru, which is stderr-bound here) so the block lands in the captured `-s` output. The
    file fences the block so it renders monospaced when a CI step concatenates these into
    $GITHUB_STEP_SUMMARY. Returns None on non-primary ranks (nothing written)."""
    if not _is_primary_rank():
        return None
    body = "\n".join(lines)
    print(f"{title}\n{body}", flush=True)
    path = summary_dir(kind) / f"{run_name}.md"
    path.write_text(f"### {title}\n\n```text\n{body}\n```\n")
    logger.info(f"{kind} summary -> {path}")
    return path
