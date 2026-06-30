# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Rendering of analysis results as JSON or a human-readable text report."""

from __future__ import annotations

import json
from collections import Counter, defaultdict

from .model import ApiCall, KernelAnalysis, RunAnalysis


def to_json(analysis: RunAnalysis, indent: int = 2) -> str:
    return json.dumps(analysis.to_dict(), indent=indent)


def render_text(analysis: RunAnalysis) -> str:
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append("LLK API usage report")
    lines.append(f"root: {analysis.root}")
    lines.append(f"compute kernels analyzed: {len(analysis.kernels)}")
    lines.append("=" * 78)

    for kernel in analysis.kernels:
        lines.extend(_render_kernel(kernel))

    lines.append("")
    lines.append("#" * 78)
    lines.append("# Aggregated unique LLK APIs (deduplicated across all kernels)")
    lines.append("#" * 78)
    for api in analysis.aggregate():
        templates = ", ".join(f"{t.name}={t.display_value}" for t in api.template_args)
        threads = ",".join(sorted(t.value for t in api.threads))
        lines.append(f"  [{api.layer.value}] {api.name}<{templates}>")
        lines.append(f"      threads={threads}  used_by_kernels={len(api.kernels)}")
    return "\n".join(lines)


def _render_kernel(kernel: KernelAnalysis) -> list[str]:
    lines = ["", f"### kernel: {kernel.name}", f"    path: {kernel.path}"]

    if kernel.descriptors is not None:
        d = kernel.descriptors
        lines.append(
            "    config: "
            f"dst_accum_mode={d.dst_accum_mode} "
            f"dst_sync_mode={d.dst_sync_mode} "
            f"math_fidelity={d.math_fidelity} approx={d.approx_mode}"
        )
        for cb in d.unpack_inputs:
            lines.append(
                f"      in  cb{cb.index}: {cb.data_format} "
                f"tile={cb.tile_r_dim}x{cb.tile_c_dim} "
                f"size={cb.tile_size_bytes}B faces={cb.num_faces}"
            )
        for cb in d.pack_outputs:
            lines.append(
                f"      out cb{cb.index}: {cb.data_format} "
                f"tile={cb.tile_r_dim}x{cb.tile_c_dim} "
                f"size={cb.tile_size_bytes}B faces={cb.num_faces}"
            )

    if kernel.errors:
        for err in kernel.errors:
            lines.append(f"    ERROR: {err}")

    # Group every occurrence by (layer, thread, API config, call site), then emit
    # groups ordered by API layer -> thread (unpack/math/pack) -> function name.
    groups: dict[tuple, list[ApiCall]] = defaultdict(list)
    for call in kernel.api_calls:
        groups[call.group_key].append(call)

    last_section: tuple | None = None
    for key in sorted(groups, key=lambda k: groups[k][0].sort_key):
        calls = groups[key]
        rep = calls[0]
        section = (rep.layer, rep.thread)
        if section != last_section:
            lines.append(f"  -- {rep.thread.value} / {rep.layer.value} --")
            last_section = section
        lines.extend(_render_group(calls))
    return lines


def _render_group(calls: list[ApiCall]) -> list[str]:
    """Render all occurrences of one API config at one call site."""
    rep = calls[0]
    op = f"  (op: {rep.operation})" if rep.operation else ""
    lines = [f"   {rep.display_header}{op}"]

    n_calls = len(calls)
    combo_counts = Counter(c.static_arg_combo for c in calls)
    n_combos = len(combo_counts)
    site = rep.call_site or "<unknown call site>"
    lines.append(f"      @ {site}    " f"({n_calls} call{_s(n_calls)}, {n_combos} distinct arg-combo{_s(n_combos)})")

    # One representative call per combo, so we can render its argument values.
    rep_by_combo: dict[tuple, ApiCall] = {}
    for call in calls:
        rep_by_combo.setdefault(call.static_arg_combo, call)
    for combo, count in sorted(combo_counts.items()):
        static_args = [a for a in rep_by_combo[combo].runtime_args if a.is_static]
        if not static_args:
            continue  # no constant args to show for this combo
        rendered = ", ".join(a.display for a in static_args)
        lines.append(f"        (x{count})  {rendered}")

    dynamic = _ordered_unique(name for c in calls for name in c.dynamic_arg_names)
    if dynamic:
        lines.append(f"        dynamic args: {', '.join(dynamic)}")
    return lines


def _s(count: int) -> str:
    return "" if count == 1 else "s"


def _ordered_unique(items) -> list[str]:
    seen: dict[str, None] = {}
    for item in items:
        seen.setdefault(item, None)
    return list(seen)
