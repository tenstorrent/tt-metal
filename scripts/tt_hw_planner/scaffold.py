# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
First-draft port generator for new HuggingFace models.

`prepare` answers "how do I run a model that tt-metal already knows about".
`scaffold` answers "what minimal set of edits would make tt-metal know about
this new model?".

Strictly limited scope: only handles the `READY` compat verdict (all building
blocks already exist; the model just isn't wired into tt_transformers's
config tables). For `FEASIBLE WITH WORK` / `BLOCKED` it refuses with a useful
pointer at the closest sibling / missing block.

Output is a unified diff plus a manifest of new files to copy from the
closest already-ported sibling. `--apply` writes the changes into the working
tree; otherwise nothing on disk is touched.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from .bringup import (
    MODEL_CONFIG_PATH,
    REPO_ROOT,
    TRACE_REGION_PATH,
    derive_base_model_name,
)
from .compatibility import Status, check_compatibility
from .probe import probe_model


MODEL_PARAMS_DIR = REPO_ROOT / "models" / "tt_transformers" / "model_params"


def _find_sibling_params_dir(sibling_tail: str, sibling_base: str) -> Optional[Path]:
    """Locate the sibling's model_params directory.  The repo names these
    inconsistently — sometimes the HF tail (`Qwen3-32B`), sometimes with an
    `-Instruct` / `-it` suffix (`Qwen2.5-7B-Instruct`, `gemma-3-27b-it`).
    Try the exact tail first, then a one-name-startswith match on the base."""
    if not MODEL_PARAMS_DIR.is_dir():
        return None
    candidate = MODEL_PARAMS_DIR / sibling_tail
    if candidate.is_dir():
        return candidate
    matches = [p for p in MODEL_PARAMS_DIR.iterdir() if p.is_dir() and p.name.startswith(sibling_base)]
    return matches[0] if len(matches) == 1 else None


class ScaffoldError(RuntimeError):
    """Raised when scaffold can't proceed and the caller should print a clean
    error to the user."""


@dataclass
class ScaffoldChange:
    kind: str  # "edit" | "create"
    path: str  # repo-relative
    diff: Optional[str] = None  # for edits: unified diff for display
    new_content: Optional[bytes] = None  # full post-change file bytes
    source: Optional[str] = None  # for creates: copied-from path (repo-relative)
    added_lines: int = 0


@dataclass
class ScaffoldPlan:
    new_model_id: str
    new_base_name: str
    new_tail: str
    sibling_model_id: str
    sibling_base_name: str
    sibling_tail: str
    compat_overall: str
    compat_summary: str
    changes: List[ScaffoldChange] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Table-line insertion (line-based, preserves formatting)
# ---------------------------------------------------------------------------


def _insert_after_sibling_in_table(src: str, sibling_key: str, new_key: str) -> Optional[str]:
    """Find the line that defines `"<sibling_key>": {...},` and clone it under
    `"<new_key>": {...},`.  The `{` requirement makes us match only inside the
    chunk-size / trace-region tables (whose values are dicts), not unrelated
    tables like `base_model_tokenizer_mapping` (whose values are strings)."""
    lines = src.splitlines(keepends=True)
    pattern = re.compile(rf'^(\s*)"{re.escape(sibling_key)}"\s*:\s*(\{{.*)$')
    for i, line in enumerate(lines):
        m = pattern.match(line.rstrip("\n"))
        if m:
            indent, rest = m.group(1), m.group(2)
            new_line = f'{indent}"{new_key}": {rest}\n'
            lines.insert(i + 1, new_line)
            return "".join(lines)
    return None


def _table_key_present(src: str, key: str) -> bool:
    """True if `key` is a dict-valued entry anywhere in the file (i.e. an
    entry in one of the tuning tables, not a tokenizer-mapping string entry)."""
    return bool(re.search(rf'^\s*"{re.escape(key)}"\s*:\s*\{{', src, re.MULTILINE))


def _build_table_insert(file_path: Path, sibling_key: str, new_key: str) -> Optional[ScaffoldChange]:
    src = file_path.read_text()
    if _table_key_present(src, new_key):
        return None
    new_src = _insert_after_sibling_in_table(src, sibling_key, new_key)
    if new_src is None:
        return None
    diff_iter = difflib.unified_diff(
        src.splitlines(keepends=True),
        new_src.splitlines(keepends=True),
        fromfile=f"a/{file_path.relative_to(REPO_ROOT)}",
        tofile=f"b/{file_path.relative_to(REPO_ROOT)}",
        n=3,
    )
    diff_text = "".join(diff_iter)
    added = sum(1 for ln in diff_text.splitlines() if ln.startswith("+") and not ln.startswith("+++"))
    return ScaffoldChange(
        kind="edit",
        path=str(file_path.relative_to(REPO_ROOT)),
        diff=diff_text,
        new_content=new_src.encode(),
        added_lines=added,
    )


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------


def plan_scaffold(new_model_id: str) -> ScaffoldPlan:
    probe = probe_model(new_model_id)
    if not probe.raw_config:
        raise ScaffoldError(f"could not load config.json for {new_model_id} — set HF_TOKEN for gated repos")

    compat = check_compatibility(new_model_id, probe.raw_config)

    if compat.in_external_demo and compat.primary_demo is not None:
        raise ScaffoldError(
            f"{new_model_id} is supported via an external demo at "
            f"`{compat.primary_demo.as_posix()}`, not via `tt_transformers/`. "
            "Scaffold only adds rows to `tt_transformers`'s tuning tables, so it "
            "does not apply here. "
            f"Run: python -m scripts.tt_hw_planner prepare {new_model_id} --box <BOX> --execute"
        )
    if compat.overall.startswith("ALREADY SUPPORTED"):
        raise ScaffoldError(
            f"{new_model_id} is already supported — no scaffolding needed. "
            f"Run: python -m scripts.tt_hw_planner prepare {new_model_id}"
        )
    missing = [r for r in compat.results if r.needed and r.status == Status.MISSING]
    if compat.overall == "BLOCKED" or missing:
        blockers = [r.block.name for r in missing] or ["(see compat output)"]
        raise ScaffoldError(
            f"{new_model_id} is BLOCKED — missing TT building block(s): {blockers}. "
            "Scaffolding can't help; new TTNN kernel work is required first."
        )

    # Scaffold accepts READY *and* FEASIBLE WITH WORK provided there are no
    # MISSING blocks — PARTIAL blocks emit a warning per block but don't
    # block the diff, because the table entries + JSON copies are valid
    # regardless of those known runtime limitations.
    if compat.overall not in ("READY", "FEASIBLE WITH WORK"):
        raise ScaffoldError(f"unexpected compat verdict {compat.overall!r}; refusing to scaffold")

    sibling_id = compat.similar_supported_model
    if not sibling_id:
        family = compat.architecture_family
        raise ScaffoldError(
            f"no closest already-ported sibling found for architecture family '{family}'. "
            "Scaffold needs a sibling to copy from; this model is the first of its kind "
            "in tt-metal. Add an entry for the model_type in `closest_supported_model()` "
            "or port a sibling first."
        )

    new_base = derive_base_model_name(new_model_id)
    sibling_base = derive_base_model_name(sibling_id)
    new_tail = new_model_id.split("/")[-1]
    sibling_tail = sibling_id.split("/")[-1]

    if new_base == sibling_base:
        raise ScaffoldError(
            f"new model derives the same base_model_name '{new_base}' as the "
            f"sibling — they would collide in the tuning tables."
        )

    changes: List[ScaffoldChange] = []
    skipped: List[str] = []
    warnings: List[str] = []

    edit1 = _build_table_insert(MODEL_CONFIG_PATH, sibling_base, new_base)
    if edit1:
        changes.append(edit1)
    elif _table_key_present(MODEL_CONFIG_PATH.read_text(), new_base):
        skipped.append(f"MAX_PREFILL_CHUNK_SIZES_DIV1024 already contains '{new_base}'")
    else:
        skipped.append(
            f"sibling '{sibling_base}' has no entry in MAX_PREFILL_CHUNK_SIZES_DIV1024 — "
            "demo will fall back to MAX_PREFILL_CHUNK_SIZE=4 (×1024)"
        )

    edit2 = _build_table_insert(TRACE_REGION_PATH, sibling_base, new_base)
    if edit2:
        changes.append(edit2)
    elif _table_key_present(TRACE_REGION_PATH.read_text(), new_base):
        skipped.append(f"trace_region_size_dict already contains '{new_base}'")
    else:
        skipped.append(
            f"sibling '{sibling_base}' has no entry in trace_region_size_dict — "
            "demo will use the parametrize default"
        )

    sibling_params_dir = _find_sibling_params_dir(sibling_tail, sibling_base)
    new_params_dir = MODEL_PARAMS_DIR / new_tail
    if sibling_params_dir is not None:
        if new_params_dir.exists():
            skipped.append(f"model_params/{new_tail}/ already exists — leaving untouched")
        else:
            json_files = sorted(p for p in sibling_params_dir.iterdir() if p.is_file() and p.suffix == ".json")
            if not json_files:
                skipped.append(f"sibling dir {sibling_params_dir.relative_to(REPO_ROOT)} contains no JSON files")
            for src_file in json_files:
                content = src_file.read_bytes()
                rel_target = (new_params_dir / src_file.name).relative_to(REPO_ROOT)
                changes.append(
                    ScaffoldChange(
                        kind="create",
                        path=str(rel_target),
                        new_content=content,
                        source=str(src_file.relative_to(REPO_ROOT)),
                        added_lines=content.count(b"\n"),
                    )
                )
    else:
        skipped.append(f"no model_params/ dir found for sibling '{sibling_tail}' — using built-in defaults")

    partial_blocks = [r for r in compat.results if r.needed and r.status == Status.PARTIAL]
    for r in partial_blocks:
        warnings.append(f"{r.block.name} is PARTIAL — {r.notes or r.block.notes or 'see compatibility.py'}")

    # Heuristic: warn when the size mismatch is large enough that the copied
    # chunk-size row probably wants editing.
    if probe.total_params and edit1:
        sibling_probe = None
        try:
            sibling_probe = probe_model(sibling_id)
        except Exception:
            sibling_probe = None
        if sibling_probe and sibling_probe.total_params:
            ratio = probe.total_params / sibling_probe.total_params
            if ratio < 0.5 or ratio > 2.0:
                warnings.append(
                    f"sibling is {sibling_probe.total_params / 1e9:.1f}B params, "
                    f"new model is {probe.total_params / 1e9:.1f}B — the copied "
                    "MAX_PREFILL_CHUNK_SIZES_DIV1024 row was tuned for a different "
                    "size; verify it fits your KV budget."
                )

    if not changes:
        raise ScaffoldError(
            "nothing to scaffold — sibling had no entries to copy, and no new "
            "model_params files to create. Manual port required."
        )

    return ScaffoldPlan(
        new_model_id=new_model_id,
        new_base_name=new_base,
        new_tail=new_tail,
        sibling_model_id=sibling_id,
        sibling_base_name=sibling_base,
        sibling_tail=sibling_tail,
        compat_overall=compat.overall,
        compat_summary=compat.effort_summary,
        changes=changes,
        skipped=skipped,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Apply
# ---------------------------------------------------------------------------


def apply_scaffold(plan: ScaffoldPlan) -> List[str]:
    applied: List[str] = []
    for ch in plan.changes:
        target = REPO_ROOT / ch.path
        target.parent.mkdir(parents=True, exist_ok=True)
        if ch.new_content is None:
            continue
        if ch.kind == "edit":
            target.write_bytes(ch.new_content)
            applied.append(f"M  {ch.path}  (+{ch.added_lines} line)")
        else:
            target.write_bytes(ch.new_content)
            applied.append(f"A  {ch.path}")
    return applied


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------


def render_text(plan: ScaffoldPlan, *, show_diff: bool = True) -> str:
    sep = "=" * 78
    out: List[str] = [sep]
    out.append(f"  SCAFFOLDING {plan.new_model_id}  (compat={plan.compat_overall})")
    out.append(sep)
    out.append("")
    out.append(f"  Sibling:        {plan.sibling_model_id}")
    out.append(f"  Sibling base:   {plan.sibling_base_name}  ({plan.sibling_tail})")
    out.append(f"  New base:       {plan.new_base_name}  ({plan.new_tail})")
    if plan.compat_summary:
        out.append(f"  Compat note:    {plan.compat_summary}")
    out.append("")

    out.append("  Proposed changes:")
    for ch in plan.changes:
        if ch.kind == "edit":
            out.append(f"    M  {ch.path}  (+{ch.added_lines} line)")
        else:
            out.append(f"    A  {ch.path}")
            if ch.source:
                out.append(f"          copied from {ch.source}")
    out.append("")

    if plan.warnings:
        out.append("  Warnings (review before applying):")
        for w in plan.warnings:
            out.append(f"    [warn] {w}")
        out.append("")

    if plan.skipped:
        out.append("  Skipped:")
        for s in plan.skipped:
            out.append(f"    -  {s}")
        out.append("")

    if show_diff:
        any_diff = any(ch.diff for ch in plan.changes)
        if any_diff:
            out.append("  Diff (edits only; new files listed above):")
            out.append("")
            for ch in plan.changes:
                if ch.diff:
                    for line in ch.diff.splitlines():
                        out.append("    " + line)
                    out.append("")

    out.append("  Next steps:")
    out.append(f"    python -m scripts.tt_hw_planner scaffold {plan.new_model_id} --apply")
    out.append(f"    python -m scripts.tt_hw_planner prepare  {plan.new_model_id} --execute")
    out.append("")
    out.append(sep)
    return "\n".join(out)


def render_apply(plan: ScaffoldPlan, applied: List[str]) -> str:
    sep = "=" * 78
    out: List[str] = [sep]
    out.append(f"  APPLIED scaffold for {plan.new_model_id}")
    out.append(sep)
    out.append("")
    for line in applied:
        out.append(f"    {line}")
    out.append("")
    out.append("  Now run:")
    out.append(f"    python -m scripts.tt_hw_planner prepare {plan.new_model_id} --execute")
    out.append("")
    out.append("  To undo:")
    out.append("    git restore models/tt_transformers/tt/model_config.py")
    out.append("    git restore models/tt_transformers/demo/trace_region_config.py")
    out.append(f"    rm -rf models/tt_transformers/model_params/{plan.new_tail}/")
    out.append("")
    out.append(sep)
    return "\n".join(out)


def render_json(plan: ScaffoldPlan, applied: Optional[List[str]] = None) -> str:
    import json

    payload = {
        "new_model_id": plan.new_model_id,
        "sibling_model_id": plan.sibling_model_id,
        "compat_overall": plan.compat_overall,
        "changes": [
            {
                "kind": ch.kind,
                "path": ch.path,
                "source": ch.source,
                "added_lines": ch.added_lines,
            }
            for ch in plan.changes
        ],
        "warnings": plan.warnings,
        "skipped": plan.skipped,
        "applied": applied,
    }
    return json.dumps(payload, indent=2)


def render_patch(plan: ScaffoldPlan) -> str:
    """Concatenate the edit diffs into a single `git apply`-compatible patch.
    New-file creations are listed in a manifest header (they can't be
    represented as a clean text diff without dumping the full contents)."""
    parts: List[str] = []
    creates = [ch for ch in plan.changes if ch.kind == "create"]
    if creates:
        parts.append(f"# scaffold for {plan.new_model_id} — also creates:")
        for ch in creates:
            parts.append(f"#   {ch.path}  (cp {ch.source} {ch.path})")
        parts.append("")
    for ch in plan.changes:
        if ch.diff:
            parts.append(ch.diff)
    return "\n".join(parts)
