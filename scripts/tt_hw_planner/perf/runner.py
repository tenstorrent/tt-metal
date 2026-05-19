# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Walk the optimizer-block catalog level by level.

The runner is the user-facing state machine that turns a diagnostic
report into a concrete patch. Two modes:

  apply  block_name --cluster CID [--dry-run]
      Runs the named block's diagnose -> propose; writes a patch file
      under perf-data/<run>/patches/ unless dry-run.

  revert block_name --cluster CID
      Deletes the matching patch file (re-collect to confirm).

Status board (`status_board.py`) renders the green/yellow/red/pending
table from the patch directory contents.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .ceilings import BoxSpec, load_box_spec
from .cluster import cluster_rows
from .collect import find_run, load_run_meta, require_healthy_run
from .join import JoinedRow, join_run
from .optimizers.base import ModelSource, OptimizerBlock, Patch, PatchKind, VerificationResult
from .optimizers.catalog import get_block, list_blocks
from .regions import classify_all


PATCHES_SUBDIR = "patches"


@dataclass
class ApplyResult:
    block: str
    cluster_id: str
    patch_path: Optional[Path]
    findings: int
    rationale: str
    dry_run: bool


def _patch_path(run_dir: Path, block: str, cluster_id: str) -> Path:
    safe_cid = cluster_id.replace("/", "_")[:96]
    return run_dir / PATCHES_SUBDIR / f"{block}__{safe_cid}.patch"


def _patch_to_text(p: Patch) -> str:
    """Serialize a Patch as a deterministic human-readable text blob.

    The runner never touches the model source directly; the text record
    is what the user reviews + applies (manually for v1 SOURCE_REWRITE,
    or by re-running the iteration loop for KWARG_REPLACE / TUNING_TABLE
    patches which the demo reads at start-up).
    """
    lines = [
        "# tt_hw_planner perf patch",
        f"# kind: {p.kind.value}",
        f"# target: {p.target.path}"
        + (f":{p.target.line}" if p.target.line else "")
        + (f"  func={p.target.func}" if p.target.func else "")
        + (f"  variable={p.target.variable}" if p.target.variable else ""),
        f"# rationale: {p.rationale}",
        "",
        json.dumps(
            {
                "kind": p.kind.value,
                "target": {
                    "path": p.target.path,
                    "line": p.target.line,
                    "func": p.target.func,
                    "variable": p.target.variable,
                },
                "new_kwargs": p.new_kwargs,
                "rationale": p.rationale,
                "extra": p.extra,
            },
            indent=2,
            default=str,
        ),
        "",
    ]
    if p.diff_text:
        lines.append("# inline diff (informational):")
        lines.append(p.diff_text)
    return "\n".join(lines)


def _load_run(run_id: str, run_dir_root: Optional[Path] = None) -> Tuple[Path, List[JoinedRow], BoxSpec, dict]:
    run_dir = find_run(run_id, run_dir_root)
    meta = require_healthy_run(run_dir)
    box = load_box_spec(meta["box"], tuple(meta["mesh_shape"]))  # type: ignore[arg-type]
    rows = join_run(
        run_id=run_id,
        tracy_csv=run_dir / "ops_perf_results.csv" if (run_dir / "ops_perf_results.csv").exists() else None,
        tracer_master=run_dir / "ttnn_operations_master.json"
        if (run_dir / "ttnn_operations_master.json").exists()
        else None,
        num_hidden_layers=meta.get("num_hidden_layers"),
        module_hierarchy=run_dir / "ttnn_module_hierarchy.json"
        if (run_dir / "ttnn_module_hierarchy.json").exists()
        else None,
    )
    cluster_rows(rows)
    classify_all(rows, box)
    return run_dir, rows, box, meta


def _model_source(meta: dict) -> ModelSource:
    from ..bringup import REPO_ROOT

    return ModelSource(
        repo_root=REPO_ROOT,
        demo_test_path=meta.get("test_path", "models/tt_transformers/demo/simple_text_demo.py"),
        base_model_name=meta.get("model_id", "").split("/")[-1],
        mesh_device=meta.get("mesh_device", ""),
        model_id=meta.get("model_id", ""),
    )


def apply_block(
    block_name: str,
    *,
    run_id: str,
    cluster_id: Optional[str] = None,
    dry_run: bool = False,
    run_dir_root: Optional[Path] = None,
) -> ApplyResult:
    """Run diagnose -> propose for `block_name`; write a patch file."""
    block: OptimizerBlock = get_block(block_name)
    run_dir, rows, box, meta = _load_run(run_id, run_dir_root)
    source = _model_source(meta)

    findings = block.diagnose(rows, source, box)
    if cluster_id:
        findings = [f for f in findings if f.cluster_id == cluster_id]
    patches = block.propose(findings, source)

    if dry_run or not patches:
        return ApplyResult(
            block=block_name,
            cluster_id=cluster_id or "*",
            patch_path=None,
            findings=len(findings),
            rationale=patches[0].rationale if patches else "no proposable findings",
            dry_run=True,
        )

    paths_written: List[Path] = []
    for p, f in zip(patches, findings if findings else patches):
        cid = getattr(f, "cluster_id", cluster_id or "all")
        out = _patch_path(run_dir, block_name, cid)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(_patch_to_text(p))
        paths_written.append(out)

    return ApplyResult(
        block=block_name,
        cluster_id=cluster_id or "*",
        patch_path=paths_written[0] if paths_written else None,
        findings=len(findings),
        rationale=patches[0].rationale if patches else "",
        dry_run=False,
    )


def revert_block(
    block_name: str,
    *,
    run_id: str,
    cluster_id: Optional[str] = None,
    run_dir_root: Optional[Path] = None,
) -> List[Path]:
    """Delete one or more patch files. Returns paths that were removed."""
    run_dir = find_run(run_id, run_dir_root)
    patch_dir = run_dir / PATCHES_SUBDIR
    if not patch_dir.exists():
        return []
    removed: List[Path] = []
    for p in patch_dir.glob(f"{block_name}__*.patch"):
        if cluster_id and not p.name.endswith(f"__{cluster_id.replace('/', '_')[:96]}.patch"):
            continue
        p.unlink()
        removed.append(p)
    return removed


def diagnose_all(run_id: str, *, run_dir_root: Optional[Path] = None) -> Dict[str, List[Any]]:
    """Run every block's diagnose() and return findings by block name."""
    run_dir, rows, box, meta = _load_run(run_id, run_dir_root)
    source = _model_source(meta)
    out: Dict[str, List[Any]] = {}
    for entry in list_blocks():
        try:
            block = get_block(entry.name)
            out[entry.name] = block.diagnose(rows, source, box)
        except Exception as e:  # pragma: no cover
            out[entry.name] = []
            print(f"WARN: {entry.name}.diagnose() raised: {e}")
    return out


def verify_run(
    block_name: str,
    *,
    before_run_id: str,
    after_run_id: str,
    run_dir_root: Optional[Path] = None,
) -> VerificationResult:
    """Compare two runs through the given block's verify()."""
    block: OptimizerBlock = get_block(block_name)
    _, before_rows, _, before_meta = _load_run(before_run_id, run_dir_root)
    _, after_rows, _, _ = _load_run(after_run_id, run_dir_root)
    source = _model_source(before_meta)
    return block.verify(before_rows, after_rows, source)
