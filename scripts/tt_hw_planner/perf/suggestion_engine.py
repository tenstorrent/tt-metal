# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Map a user-run's submodules to reference-model submodules and
propose optimizer blocks that close the perf gap.

This is the "auto-tune by example" core: instead of asking the user
to know which optimizer block applies to which submodule, we look up
each user submodule in the reference DB and recommend whatever
optimizer blocks the reference used to make that submodule fast.

The output is a list of ``Suggestion`` records ranked by expected
runtime savings; the dashboard inspector renders them as cards and the
``perf suggest`` CLI prints them as text.

Generic across any HF model whose architecture family appears in (or
aliases into) the reference DB. Models for which no references exist
get zero suggestions plus a hint to curate one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .module_graph import ModuleGraph, ModuleNode
from .reference_db import ModuleReference, find_module_reference


@dataclass
class Suggestion:
    """One actionable per-module recommendation."""

    # The user's submodule we're talking about
    attribute_path: str
    module_class: str
    layer_index: Optional[int]

    # Current state (from the user's run)
    user_runtime_ms_p50: float
    user_fpu_util_pct: float
    user_op_count: int

    # What the reference is and how this user-module compares
    reference: ModuleReference
    delta_runtime_ms: float  # positive = user is slower
    delta_fpu_util_pct: float  # positive = user has more headroom
    is_significant: bool  # True iff delta is worth showing

    # The proposed action (subset of reference.optimizer_blocks_applied
    # filtered to entries the user hasn't already applied — TODO once
    # we track applied state per-run; for now we return them all)
    proposed_blocks: List[str] = field(default_factory=list)

    # Free-form explanation rendered above the Apply button
    rationale: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "attribute_path": self.attribute_path,
            "module_class": self.module_class,
            "layer_index": self.layer_index,
            "user_runtime_ms_p50": self.user_runtime_ms_p50,
            "user_fpu_util_pct": self.user_fpu_util_pct,
            "user_op_count": self.user_op_count,
            "reference_id": self.reference.reference_id,
            "reference_model_id": self.reference.model_id,
            "reference_quality": self.reference.quality,
            "reference_source_run_id": self.reference.source_run_id,
            "reference_runtime_ms_p50": self.reference.runtime_ms_p50,
            "reference_fpu_util_pct": self.reference.fpu_util_pct,
            "delta_runtime_ms": self.delta_runtime_ms,
            "delta_fpu_util_pct": self.delta_fpu_util_pct,
            "is_significant": self.is_significant,
            "proposed_blocks": list(self.proposed_blocks),
            "rationale": self.rationale,
        }


# Tuning knobs for "is this worth surfacing"
_MIN_DELTA_MS_ABS = 0.05  # ignore <50us deltas — noise
_MIN_DELTA_RATIO = 0.10  # and <10% relative — also noise
_MIN_LAYER_REPRESENTATIVE_INDEX = 2  # use layer 2+ for "representative",
# layers 0/1 often differ due to
# warmup or special handling


# ---------------------------------------------------------------------------
# Module collapse: one suggestion per ROLE, not per LAYER
# ---------------------------------------------------------------------------


def _representative_node(graph: ModuleGraph, role_key: str, nodes: List[ModuleNode]) -> ModuleNode:
    """Pick a single representative node for a role's per-layer instances.

    Strategy:
      1. Prefer the lowest layer index >= _MIN_LAYER_REPRESENTATIVE_INDEX
         (most layers in transformers are interchangeable, so any one of
         the "middle" layers is fine).
      2. Fall back to the node with the most ops if no layered match.
    """
    layered = sorted(
        [n for n in nodes if n.layer_index is not None],
        key=lambda n: (n.layer_index, n.attribute_path),
    )
    for n in layered:
        if n.layer_index is not None and n.layer_index >= _MIN_LAYER_REPRESENTATIVE_INDEX:
            return n
    # No layered representative; fall back to most-active node
    return max(nodes, key=lambda n: n.total_device_ns)


def _role_key(attribute_path: str) -> str:
    """Collapse "model.layers.7.self_attn.q_proj" to a layer-stripped
    key "model.layers.*.self_attn.q_proj" so per-layer duplicates fold
    into one suggestion.
    """
    parts = attribute_path.split(".")
    out: List[str] = []
    for i, p in enumerate(parts):
        # Replace pure-integer segments with "*". Skips zero-length
        # segments (which can't happen here but cheap to guard).
        if p and p.isdigit():
            out.append("*")
        else:
            out.append(p)
    return ".".join(out)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def propose_optimizations(
    *,
    graph: ModuleGraph,
    arch_family: str,
    mesh_shape: Tuple[int, int],
    dtype: Optional[str],
    box: Optional[str],
) -> List[Suggestion]:
    """Walk the user's ModuleGraph and emit one Suggestion per matched
    submodule role, ranked by expected runtime savings.

    Behavior:
      - Modules with no ops directly attributed (i.e. pure-wrapper
        modules like ``model``, ``decoder``) are skipped.
      - Each (role_key) group is collapsed to one representative node
        via ``_representative_node``.
      - If the reference DB has no match for a role, no suggestion is
        emitted for that role (the inspector still shows current
        metrics, just without a comparison).
      - The result list is sorted by ``delta_runtime_ms × user_op_count``
        descending so the highest-impact suggestions come first.
    """
    by_role: Dict[str, List[ModuleNode]] = {}
    for node in graph.nodes.values():
        if node.op_count == 0:
            continue
        key = _role_key(node.attribute_path)
        by_role.setdefault(key, []).append(node)

    suggestions: List[Suggestion] = []
    for role_key, nodes in by_role.items():
        rep = _representative_node(graph, role_key, nodes)
        ref = find_module_reference(
            attribute_path=rep.attribute_path,
            module_class=rep.module_class,
            arch_family=arch_family,
            mesh_shape=mesh_shape,
            dtype=dtype,
            box=box,
        )
        if ref is None:
            continue

        # Convert ns -> ms for comparison
        user_runtime_ms = rep.median_device_ns / 1e6
        delta_ms = user_runtime_ms - ref.runtime_ms_p50
        delta_fpu = ref.fpu_util_pct - rep.mean_fpu_util_pct

        significant = (
            abs(delta_ms) >= _MIN_DELTA_MS_ABS
            and ref.runtime_ms_p50 > 0
            and abs(delta_ms) / max(ref.runtime_ms_p50, 1e-9) >= _MIN_DELTA_RATIO
        )

        # Number of layered instances of this role (e.g. 32 for a 32-layer
        # model's q_proj). Used to estimate run-wide impact.
        n_instances = len(nodes)
        per_call_savings_ms = max(delta_ms, 0.0)
        run_wide_savings_ms = per_call_savings_ms * rep.op_count * n_instances

        rationale_parts: List[str] = []
        if delta_ms > 0:
            rationale_parts.append(
                f"Your `{role_key}` runs {user_runtime_ms:.2f} ms; "
                f"the {ref.model_id} reference runs {ref.runtime_ms_p50:.2f} ms "
                f"({delta_ms:+.2f} ms, "
                f"{(delta_ms / max(ref.runtime_ms_p50, 1e-9)) * 100:+.0f}%)."
            )
        elif delta_ms < 0:
            rationale_parts.append(
                f"Your `{role_key}` is FASTER than the {ref.model_id} reference "
                f"({-delta_ms:.2f} ms ahead). The reference's config is still "
                f"informative — your kernel may be using a different optimization."
            )
        else:
            rationale_parts.append(f"Your `{role_key}` matches the {ref.model_id} reference.")

        if delta_fpu > 5:
            rationale_parts.append(
                f"Headroom: the reference reaches {ref.fpu_util_pct:.0f}% FPU util "
                f"vs your {rep.mean_fpu_util_pct:.0f}%."
            )
        if ref.mesh_shape and ref.mesh_shape != tuple(mesh_shape):
            rationale_parts.append(
                f"NOTE: reference was measured on mesh {ref.mesh_shape} (you are on "
                f"{tuple(mesh_shape)}); expect deltas from the mismatch."
            )
        if ref.dtype and dtype and ref.dtype.lower() != dtype.lower():
            rationale_parts.append(f"NOTE: reference dtype is {ref.dtype} (you are on {dtype}).")

        proposed = list(ref.optimizer_blocks_applied)
        if proposed and per_call_savings_ms > 0:
            rationale_parts.append(
                f"Estimated run-wide savings if all {n_instances} layered instances "
                f"close the gap: {run_wide_savings_ms:.1f} ms total."
            )

        suggestions.append(
            Suggestion(
                attribute_path=rep.attribute_path,
                module_class=rep.module_class,
                layer_index=rep.layer_index,
                user_runtime_ms_p50=user_runtime_ms,
                user_fpu_util_pct=rep.mean_fpu_util_pct,
                user_op_count=rep.op_count,
                reference=ref,
                delta_runtime_ms=delta_ms,
                delta_fpu_util_pct=delta_fpu,
                is_significant=significant,
                proposed_blocks=proposed,
                rationale=" ".join(rationale_parts),
            )
        )

    suggestions.sort(
        key=lambda s: (s.delta_runtime_ms * s.user_op_count, s.delta_runtime_ms),
        reverse=True,
    )
    return suggestions


# ---------------------------------------------------------------------------
# Text rendering for the CLI
# ---------------------------------------------------------------------------


def render_suggestions_text(suggestions: List[Suggestion]) -> str:
    if not suggestions:
        return (
            "No suggestions produced.\n\n"
            "  This can mean any of:\n"
            "    - No module-class reference matches your model's arch_family.\n"
            "      Curate one: see scripts/tt_hw_planner/perf/references/README.md\n"
            "    - The run had no module-hierarchy sidecar (re-collect; the\n"
            "      sidecar is produced automatically with --trace-params).\n"
            "    - Your model is already matching the reference everywhere."
        )

    lines: List[str] = []
    lines.append(f"{'#':>2}  {'STATUS':<6}  {'ROLE':<46}  {'YOU (ms)':>9}  {'REF (ms)':>9}  {'\u0394 ms':>8}  PROPOSE")
    lines.append("-" * 120)
    for i, s in enumerate(suggestions, 1):
        status = "ACT" if s.is_significant and s.delta_runtime_ms > 0 else "info"
        proposal = ", ".join(s.proposed_blocks) if s.proposed_blocks else "-"
        role = _role_key(s.attribute_path)
        if len(role) > 46:
            role = role[:43] + "..."
        lines.append(
            f"{i:>2}  {status:<6}  {role:<46}  {s.user_runtime_ms_p50:>9.3f}  "
            f"{s.reference.runtime_ms_p50:>9.3f}  {s.delta_runtime_ms:>+8.3f}  {proposal}"
        )

    lines.append("")
    quality_counts: Dict[str, int] = {}
    for s in suggestions:
        q = s.reference.quality
        quality_counts[q] = quality_counts.get(q, 0) + 1
    qparts = ", ".join(f"{k}={v}" for k, v in sorted(quality_counts.items()))
    lines.append(f"Reference quality mix: {qparts}")
    lines.append("")
    lines.append("Top 3 explanations:")
    for s in suggestions[:3]:
        lines.append(f"  - [{_role_key(s.attribute_path)}]")
        for sentence in s.rationale.split(". "):
            sentence = sentence.strip()
            if sentence:
                lines.append(f"      {sentence}.")
        if s.proposed_blocks:
            lines.append(f"      Run: tt_hw_planner perf apply {s.proposed_blocks[0]} --run <run>")
    return "\n".join(lines)


__all__ = [
    "Suggestion",
    "propose_optimizations",
    "render_suggestions_text",
]
