#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Usage:
    dump_op_mesh

Description:
    Renders the system mesh as a 2D grid. Columns headers are `C0`, `C1`, …
    (mesh column index); the first column `R\\C` carries the mesh row index.
    Each cell starts with a chip identifier on line 1:
      - `T<tray>:N<asic_loc> (Device <id>)` on UBB (tray + on-tray ASIC
        position from `ClusterDescriptor`),
      - bare `(Device <id>)` on other clusters,
      - `(Mesh <m> Chip <c>)` for chips that live on a different host
        (multi-host meshes).
    Subsequent lines list the ops currently dispatched on that chip, one per
    `host_assigned_id`, sorted ascending. A device this process opened but
    that has no live ops shows `(idle)`; a device on this host that this
    process never opened shows `(unused)`; cross-host chips show `(remote)`.

    A leading `[!] ` on an op line marks that op as a *straggler* - its
    `host_assigned_id` is strictly less than the highest live id in the
    mesh (the leading edge of dispatch). In a hang, the leading-edge chips
    are themselves stuck - they're waiting on the stragglers to finish -
    so stragglers are the root cause and get the marker; leading-edge ops
    stay clean. On a multi-op chip, only the lagging op line is flagged;
    the leading-edge op on the same chip is unmarked. In a healthy mesh
    (every chip on the same op id) nothing is flagged.

    For physical tray/board grouping, see the `Tray/Board` column in
    `device_info`. Shape and per-cell device identity come from Inspector's
    `getSystemMesh()` (works for single-device → multi-host Galaxy).

Owner:
    onenezicTT
"""

from __future__ import annotations

from dataclasses import make_dataclass

from inspector_data import run as get_inspector_data
from metal_device_id_mapping import run as get_metal_device_id_mapping
from operation_provider import run as get_operation_provider, RunningOperationAggregation
from triage import ScriptConfig, ScriptPriority, log_check, run_script, triage_field
from ttexalens.context import Context


script_config = ScriptConfig(
    depends=["operation_provider", "metal_device_id_mapping", "inspector_data"],
    priority=ScriptPriority.HIGH,
)


def _shape_to_2d(global_shape) -> tuple[int, int]:
    """Coerce arbitrary mesh shape to (rows, cols). 1D → (1, N); 2D → (R, C); >2D flattens."""
    dims = [int(d) for d in global_shape]
    if len(dims) == 0:
        return (0, 0)
    if len(dims) == 1:
        return (1, dims[0])
    if len(dims) == 2:
        return (dims[0], dims[1])
    # Higher-rank meshes aren't standard today; flatten so we still render something useful.
    prod = 1
    for d in dims[1:]:
        prod *= d
    return (dims[0], prod)


def _build_label_to_ops(
    aggregations: dict[int, RunningOperationAggregation],
) -> dict[str, list[RunningOperationAggregation]]:
    out: dict[str, list[RunningOperationAggregation]] = {}
    for agg in aggregations.values():
        for label in agg.device_labels:
            out.setdefault(label, []).append(agg)
    return out


def _leading_edge_op_id(aggregations: dict[int, RunningOperationAggregation]) -> int | None:
    return max(aggregations) if aggregations else None


def _ubb_prefix(cd, chip_id: int) -> str:
    # `get_tray_id` returns None for non-UBB boards by contract - use that as the predicate.
    tray = cd.get_tray_id(chip_id)
    if tray is None:
        return ""
    return f"T{tray}:N{cd.get_asic_location(chip_id)}"


def _op_line(agg: RunningOperationAggregation, leading_edge: int | None) -> str:
    body = f"{agg.host_assigned_id}: {agg.operation_name or 'N/A'}"
    is_straggler = leading_edge is not None and agg.host_assigned_id < leading_edge
    return f"[!] {body}" if is_straggler else body


def _format_cell(
    mapped_device,
    metal_id_mapping,
    cd,
    label_to_ops: dict[str, list[RunningOperationAggregation]],
    leading_edge: int | None,
    use_unique_id_labels: bool,
    in_use_metal_ids: set[int],
) -> str:
    if not mapped_device.isLocal:
        return f"(Mesh {mapped_device.fabricMeshId} Chip {mapped_device.fabricChipId})\n(remote)"

    metal_id = mapped_device.localChipId
    device_id = metal_id_mapping.get_device_id(metal_id)
    if device_id is None:
        return f"(Metal Device {metal_id})\n(no exalens map)"

    prefix = _ubb_prefix(cd, device_id)
    chip_label = f"{prefix} (Device {device_id})".lstrip()

    if metal_id not in in_use_metal_ids:
        return f"{chip_label}\n(unused)"

    op_key = hex(metal_id_mapping.get_unique_id(metal_id)) if use_unique_id_labels else str(device_id)
    aggs = label_to_ops.get(op_key, [])
    if not aggs:
        return f"{chip_label}\n(idle)"

    op_lines = [_op_line(agg, leading_edge) for agg in sorted(aggs, key=lambda a: a.host_assigned_id)]
    return chip_label + "\n" + "\n".join(op_lines)


def run(args, context: Context):
    inspector_data = get_inspector_data(args, context)
    try:
        system_mesh = inspector_data.getSystemMesh().systemMesh
    except Exception as e:
        log_check(False, f"dump_op_mesh: getSystemMesh() failed: {e}")
        return None

    rows, cols = _shape_to_2d(system_mesh.globalShape)
    if rows == 0 or cols == 0:
        log_check(False, f"dump_op_mesh: empty mesh shape {list(system_mesh.globalShape)}")
        return None

    mapped = list(system_mesh.mappedDevices)
    expected_cells = rows * cols
    if len(mapped) != expected_cells:
        log_check(
            False,
            f"dump_op_mesh: mappedDevices count ({len(mapped)}) does not match globalShape product ({expected_cells}).",
        )
        return None

    cd = context.cluster_descriptor
    metal_id_mapping = get_metal_device_id_mapping(args, context)
    bundle = get_operation_provider(args, context)
    label_to_ops = _build_label_to_ops(bundle.aggregations)
    leading_edge = _leading_edge_op_id(bundle.aggregations)
    use_unique_id_labels = any(label.startswith("0x") for label in label_to_ops)
    in_use_metal_ids = set(inspector_data.getDevicesInUse().metalDeviceIds)

    fields_spec: list[tuple[str, type, object]] = [("r", str, triage_field("R\\C"))]
    fields_spec.extend((f"c{c}", str, triage_field(f"C{c}")) for c in range(cols))
    MeshRow = make_dataclass("MeshRow", fields_spec)

    out = []
    for r in range(rows):
        cells = [
            _format_cell(
                mapped[r * cols + c],
                metal_id_mapping,
                cd,
                label_to_ops,
                leading_edge,
                use_unique_id_labels,
                in_use_metal_ids,
            )
            for c in range(cols)
        ]
        out.append(MeshRow(str(r), *cells))
    return out


if __name__ == "__main__":
    run_script()
