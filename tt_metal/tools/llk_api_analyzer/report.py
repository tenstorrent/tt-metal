# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Rendering of analysis results as JSON or a human-readable text report."""

from __future__ import annotations

import csv
import io
import json
from collections import Counter, defaultdict

from tabulate import tabulate

from .descriptors import KernelDescriptors
from .model import ApiCall, KernelAnalysis, RunAnalysis, format_function_display


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
        threads = ",".join(sorted(t.value for t in api.threads))
        lines.append(f"  [{api.layer.value}] {format_function_display(api.name, api.template_args)}")
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
    op = f"  (op: `{rep.operation}`)" if rep.operation else ""
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


# ---------------------------------------------------------------------------
# Collapsed single-table view: one row per distinct LLK call (across the run).
# ---------------------------------------------------------------------------

TABLE_COLUMNS = [
    "LLK API",
    "TTNN Ops",
    "Op Args",
    "Input Data Formats",
    "Output Data Formats",
    "Tile Dims",
    "Math Fidelity",
    "Math Approx",
    "FP32 Dest Accum",
    "Dst Sync Mode",
]


def collapse_rows(analysis: RunAnalysis) -> list[list[str]]:
    """Flatten the analysis into one row per distinct LLK call.

    A row is keyed by everything except the TTNN op; rows that are otherwise
    identical across kernels/ops are merged, with every contributing TTNN op
    listed in the ``TTNN Ops`` column. Kernel-level configuration (fidelity,
    approx, dest-accumulate, sync) comes from that call's kernel descriptors;
    data formats and tile dims are inferred per call from its CB operands.
    """
    merged: dict[tuple, set[str]] = defaultdict(set)
    for kernel in analysis.kernels:
        desc = kernel.descriptors
        cb_formats = _cb_format_map(desc)
        cb_tile_dims = _cb_tile_dim_map(desc)
        fidelity = desc.math_fidelity if desc and desc.math_fidelity else "-"
        approx = _tri_bool(desc.approx_mode if desc else None)
        accum = _tri_bool(desc.dst_accum_mode if desc else None)
        sync = desc.dst_sync_mode if desc and desc.dst_sync_mode else "-"

        for call in kernel.unique_calls():
            in_formats, out_formats = _call_io_formats(cb_formats, call, kernel)
            tiles = _call_tile_dims(cb_tile_dims, call, kernel)
            key = (
                call.display_header,
                _op_args(call),
                in_formats,
                out_formats,
                tiles,
                fidelity,
                approx,
                accum,
                sync,
            )
            ops_for_key = merged[key]  # ensure the row exists even with no op
            if call.operation:
                ops_for_key.add(call.operation)

    rows: list[list[str]] = []
    for key, ops in merged.items():
        llk, args, in_fmt, out_fmt, tiles, fidelity, approx, accum, sync = key
        ttnn_ops = ", ".join(f"`{op}`" for op in sorted(ops)) if ops else "-"
        rows.append([llk, ttnn_ops, args, in_fmt, out_fmt, tiles, fidelity, approx, accum, sync])
    rows.sort(key=lambda r: (r[0], r[2], r[3]))
    return rows


def render_table(analysis: RunAnalysis) -> str:
    """Render the collapsed analysis as a Markdown (GitHub) pipe table.

    Cells are pre-escaped (:func:`_md_cell`) because ``tabulate`` does not
    escape ``|`` inside cells, which would otherwise break the pipe table.
    """
    rows = [[_md_cell(c) for c in row] for row in collapse_rows(analysis)]
    return tabulate(rows, headers=TABLE_COLUMNS, tablefmt="github")


def render_csv(analysis: RunAnalysis) -> str:
    """Render the collapsed analysis as CSV (RFC-4180, values quoted as needed)."""
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(TABLE_COLUMNS)
    writer.writerows(collapse_rows(analysis))
    return buffer.getvalue().rstrip("\n")


# Runtime-arg names that carry an input circular-buffer index all contain
# "operand" (e.g. operand, operandA, unpA_operand, srca_new_operand).
#
# Output-CB argument names are matched by substring so the reconfig variants are
# caught alongside the plain names: "output" covers output / pack_output /
# new_output / old_output (e.g. llk_pack_reconfig_data_format), and "ocb" covers
# ocb / new_ocb / old_ocb. A few exact names (out_cb) carry an output CB index
# but match no substring. Names containing "tile" (e.g. output_tile_index) are
# tile offsets, not CBs, and are excluded.
_OUTPUT_CB_ARG_SUBSTRINGS = ("output", "ocb")
_OUTPUT_CB_ARGS = frozenset({"out_cb"})


def _is_output_cb_arg(name: str) -> bool:
    if "tile" in name:
        return False
    return name in _OUTPUT_CB_ARGS or any(sub in name for sub in _OUTPUT_CB_ARG_SUBSTRINGS)


def _cb_format_map(desc: KernelDescriptors | None) -> dict[int, str]:
    if desc is None:
        return {}
    mapping: dict[int, str] = {}
    for cb in list(desc.unpack_inputs) + list(desc.pack_outputs):
        mapping.setdefault(cb.index, cb.data_format)
    return mapping


def _cb_tile_dim_map(desc: KernelDescriptors | None) -> dict[int, str]:
    if desc is None:
        return {}
    mapping: dict[int, str] = {}
    for cb in list(desc.unpack_inputs) + list(desc.pack_outputs):
        if cb.tile_r_dim is not None and cb.tile_c_dim is not None:
            mapping.setdefault(cb.index, f"{cb.tile_r_dim}x{cb.tile_c_dim}")
    return mapping


def _classify_cbs_for_call(call: ApiCall) -> tuple[set[int], set[int]]:
    """CB indices referenced by this call's static runtime arguments."""
    inputs: set[int] = set()
    outputs: set[int] = set()
    for arg in call.runtime_args:
        if not arg.is_static:
            continue
        if "operand" in arg.name:
            inputs.update(arg.static_values)
        elif _is_output_cb_arg(arg.name):
            outputs.update(arg.static_values)
    return inputs, outputs


def _classify_cbs_kernel(kernel: KernelAnalysis) -> tuple[set[int], set[int]]:
    """Union of all CB indices referenced statically anywhere in the kernel."""
    inputs: set[int] = set()
    outputs: set[int] = set()
    for call in kernel.api_calls:
        in_cbs, out_cbs = _classify_cbs_for_call(call)
        inputs.update(in_cbs)
        outputs.update(out_cbs)
    return inputs, outputs


def _resolve_call_cbs(call: ApiCall, kernel: KernelAnalysis, configured_cbs) -> tuple[set[int], set[int]]:
    """Select the (input, output) CB indices to report for one LLK call.

    1. Use CB indices from *this call's* static ``operand*`` / ``output`` args.
    2. If the call names no CBs at all, fall back to the kernel-wide union.
    3. If that is also empty, fall back to every configured CB (both directions).

    When the call itself names CBs, a direction it does not name stays empty
    (rendered as ``-``) rather than borrowing the fallback.
    """
    in_cbs, out_cbs = _classify_cbs_for_call(call)
    if in_cbs or out_cbs:
        return in_cbs, out_cbs
    in_cbs, out_cbs = _classify_cbs_kernel(kernel)
    if in_cbs or out_cbs:
        return in_cbs, out_cbs
    all_cbs = set(configured_cbs)
    return all_cbs, all_cbs


def _call_io_formats(cb_formats: dict[int, str], call: ApiCall, kernel: KernelAnalysis) -> tuple[str, str]:
    """Resolve the input/output data-format columns for one LLK call."""
    in_cbs, out_cbs = _resolve_call_cbs(call, kernel, cb_formats)
    return _cb_column(cb_formats, in_cbs), _cb_column(cb_formats, out_cbs)


def _call_tile_dims(cb_tile_dims: dict[int, str], call: ApiCall, kernel: KernelAnalysis) -> str:
    """Resolve the tile-dimension column for one LLK call (inputs + outputs)."""
    in_cbs, out_cbs = _resolve_call_cbs(call, kernel, cb_tile_dims)
    return _cb_column(cb_tile_dims, in_cbs | out_cbs)


def _cb_column(cb_values: dict[int, str], cbs) -> str:
    """Render ``cb<i>=<value>`` for the given CBs, or ``-`` when none apply."""
    indices = sorted(i for i in cbs if i in cb_values)
    if not indices:
        return "-"
    return ", ".join(f"cb{i}={cb_values[i]}" for i in indices)


def _op_args(call: ApiCall) -> str:
    if not call.runtime_args:
        return "-"
    return ", ".join(a.display if a.is_static else f"{a.name}=?" for a in call.runtime_args)


def _tri_bool(value: bool | None) -> str:
    if value is None:
        return "-"
    return "true" if value else "false"


def _md_cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")
