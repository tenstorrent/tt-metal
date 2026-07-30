#!/usr/bin/env python3
"""Generate ring-pipeline Mesh Graph Descriptors for the topology-mapper sweep.

This emits one MGD per (shape, stage-count) pair. Every generated MGD is a closed ring of
identical per-stage meshes (last stage -> stage 0), used to stress the topology mapper on a
mock cluster (e.g. the SC36 mock: 36 hosts x 32 ASICs = 1152 ASICs total).

Everything is data-driven from pipeline_sweep_config.yaml -- SHAPE (per-stage mesh geometry
+ host split), PINNINGS (optional many-to-all ASIC groups), and LOOPBACK SIZE (the inter-stage
ring link width) all come from the config, so new shapes need no code change. See the config
file for the field-by-field schema.

Regenerate in place with:

    python3 tests/scripts/multihost/gen_pipeline_sweep_mgds.py

Override the config or output directory with --config / --out-dir. NOTE: the generated
sweep_*.textproto files are intentionally NOT committed to the repo -- generate them on demand.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_CONFIG = _SCRIPT_DIR / "pipeline_sweep_config.yaml"
_DEFAULT_OUT = _SCRIPT_DIR.parents[2] / "tests/tt_metal/tt_fabric/custom_mesh_descriptors/pipeline_sweep"


@dataclass(frozen=True)
class PinningGroup:
    name: str
    mesh_id_regex: str
    chip_ids: list[int]
    positions: list[tuple[int, int]]


@dataclass(frozen=True)
class Loopback:
    """Inter-stage ring link ("loopback"): the channel spec applied to every stage->stage edge."""

    channels: int
    policy: str
    assign_z: bool


@dataclass(frozen=True)
class ShapeConfig:
    name: str
    device_dims: list[int]
    device_dim_types: list[str]
    host_dims: list[int]
    mesh_channels: int
    mesh_channels_policy: str
    loopback: Loopback
    stages: list[int]
    pinnings: list[PinningGroup]

    @property
    def asics_per_stage(self) -> int:
        n = 1
        for d in self.device_dims:
            n *= int(d)
        return n


def _load_config(path: Path) -> dict[str, ShapeConfig]:
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict) or "shapes" not in raw:
        raise ValueError(f"{path}: expected top-level 'shapes' mapping")

    shapes: dict[str, ShapeConfig] = {}
    for shape_name, shape_cfg in raw["shapes"].items():
        if not isinstance(shape_cfg, dict):
            raise ValueError(f"{path}: shapes.{shape_name} must be a mapping")
        shapes[shape_name] = _parse_shape(path, str(shape_name), shape_cfg)
    if not shapes:
        raise ValueError(f"{path}: no shapes defined")
    return shapes


def _parse_shape(path: Path, name: str, cfg: dict) -> ShapeConfig:
    prefix = f"{path}: shapes.{name}"

    device = cfg.get("device_topology")
    if not isinstance(device, dict) or not device.get("dims") or not device.get("dim_types"):
        raise ValueError(f"{prefix}.device_topology must have non-empty dims and dim_types")
    device_dims = [int(d) for d in device["dims"]]
    device_dim_types = [str(t) for t in device["dim_types"]]
    if len(device_dims) != len(device_dim_types):
        raise ValueError(f"{prefix}.device_topology: dims and dim_types length mismatch")

    host = cfg.get("host_topology")
    if not isinstance(host, dict) or not host.get("dims"):
        raise ValueError(f"{prefix}.host_topology.dims is required")
    host_dims = [int(d) for d in host["dims"]]

    mesh_ch = cfg.get("mesh_channels") or {}
    mesh_channels = int(mesh_ch.get("count", 2))
    mesh_channels_policy = str(mesh_ch.get("policy", "RELAXED"))

    lb = cfg.get("loopback") or {}
    loopback = Loopback(
        channels=int(lb.get("channels", 8)),
        policy=str(lb.get("policy", "RELAXED")),
        assign_z=bool(lb.get("assign_z", False)),
    )

    stages = cfg.get("stages")
    if not stages:
        raise ValueError(f"{prefix}.stages must be a non-empty list")

    pinning_groups: list[PinningGroup] = []
    pinnings_cfg = cfg.get("pinnings")
    if pinnings_cfg is not None:
        groups = pinnings_cfg.get("groups")
        if not groups:
            raise ValueError(f"{prefix}.pinnings.groups must be non-empty")
        for i, group in enumerate(groups):
            pinning_groups.append(_parse_pinning_group(path, name, i, group))

    return ShapeConfig(
        name=name,
        device_dims=device_dims,
        device_dim_types=device_dim_types,
        host_dims=host_dims,
        mesh_channels=mesh_channels,
        mesh_channels_policy=mesh_channels_policy,
        loopback=loopback,
        stages=[int(s) for s in stages],
        pinnings=pinning_groups,
    )


def _parse_pinning_group(path: Path, shape: str, index: int, group: Any) -> PinningGroup:
    prefix = f"{path}: shapes.{shape}.pinnings.groups[{index}]"
    if not isinstance(group, dict):
        raise ValueError(f"{prefix} must be a mapping")

    name = group.get("name", f"group{index}")
    mesh_id_regex = group.get("mesh_id_regex")
    chip_ids = group.get("chip_ids")
    positions_raw = group.get("positions")
    if not mesh_id_regex:
        raise ValueError(f"{prefix}.mesh_id_regex is required")
    if not chip_ids:
        raise ValueError(f"{prefix}.chip_ids must be a non-empty list")
    if not positions_raw:
        raise ValueError(f"{prefix}.positions must be a non-empty list")

    positions: list[tuple[int, int]] = []
    for j, pos in enumerate(positions_raw):
        if not isinstance(pos, dict) or "tray_id" not in pos or "asic_location" not in pos:
            raise ValueError(f"{prefix}.positions[{j}] must be {{tray_id, asic_location}}")
        positions.append((int(pos["tray_id"]), int(pos["asic_location"])))

    return PinningGroup(
        name=str(name),
        mesh_id_regex=str(mesh_id_regex),
        chip_ids=[int(c) for c in chip_ids],
        positions=positions,
    )


def _list(values: list) -> str:
    return ", ".join(str(v) for v in values)


def _instances(stages: int) -> str:
    return "\n".join(f'  instances {{ mesh {{ mesh_descriptor: "M0" mesh_id: {i} }} }}' for i in range(stages))


def _ring_connections(stages: int, loopback: Loopback) -> str:
    blocks = []
    for i in range(stages):
        j = (i + 1) % stages  # last edge wraps back to 0 -> closed ring
        block = [
            "  connections {",
            f'    nodes {{ mesh {{ mesh_descriptor: "M0" mesh_id: {i} }} }}',
            f'    nodes {{ mesh {{ mesh_descriptor: "M0" mesh_id: {j} }} }}',
            f"    channels {{ count: {loopback.channels} policy: {loopback.policy} }}",
        ]
        if loopback.assign_z:
            block.append("    assign_z_direction: true")
        block.append("  }")
        blocks.append("\n".join(block))
    return "\n".join(blocks)


def _pinnings_section(groups: list[PinningGroup]) -> str:
    if not groups:
        return ""

    chip_summary = ", ".join(str(c) for c in groups[0].chip_ids)
    lines = [
        "# --- Pinnings ---------------------------------------------------------------",
        f"# All-to-all corner pinning per mesh: logical chips ({chip_summary}) may map to any of",
        "# the tray/asic positions in each group. The solver enforces a bijection. Groups are keyed",
        "# by mesh_id_regex.",
        "#",
        '# mesh_id_regex expands PER matched mesh (one all-to-all group each). Ranges like "0-8"',
        '# and comma lists like "0,2,4-6" are also supported.',
        "",
    ]
    for group in groups:
        lines.append(f"# {group.name} mesh ids ({group.mesh_id_regex})")
        lines.append("pinnings {")
        for chip_id in group.chip_ids:
            lines.append(f'  logical_fabric_node_id {{ mesh_id_regex: "{group.mesh_id_regex}" chip_id: {chip_id} }}')
        for tray_id, asic_location in group.positions:
            lines.append(f"  physical_asic_position {{ tray_id: {tray_id} asic_location: {asic_location} }}")
        lines.append("}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def build_mgd(shape: ShapeConfig, stages: int) -> str:
    total_asics = stages * shape.asics_per_stage
    pinnings_block = _pinnings_section(shape.pinnings)
    body = f"""# --- Meshes ---------------------------------------------------------------
# GENERATED by tests/scripts/multihost/gen_pipeline_sweep_mgds.py -- do not edit by hand.
# {shape.name} pipeline ring, {stages} stages ({stages} x {shape.asics_per_stage} = {total_asics} ASICs).
# Each stage is a {_list(shape.device_dims)} {_list(shape.device_dim_types)} mesh split across a
# {_list(shape.host_dims)} host topology; the ring closes (stage {stages - 1} -> stage 0) with a
# loopback of {shape.loopback.channels} channels ({shape.loopback.policy}).

mesh_descriptors {{
  name: "M0"
  arch: BLACKHOLE
  device_topology {{ dims: [ {_list(shape.device_dims)} ] dim_types: [ {_list(shape.device_dim_types)} ] }}
  host_topology   {{ dims: [ {_list(shape.host_dims)} ] }}
  channels {{ count: {shape.mesh_channels} policy: {shape.mesh_channels_policy} }}
}}

graph_descriptors {{
  name: "G0"
  type: "FABRIC"
{_instances(stages)}

{_ring_connections(stages, shape.loopback)}
}}

# --- Instantiation ----------------------------------------------------------
top_level_instance {{ graph {{ graph_descriptor: "G0" graph_id: 0 }} }}
"""
    if pinnings_block:
        body += "\n" + pinnings_block
    return body


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--config",
        type=Path,
        default=_DEFAULT_CONFIG,
        help=f"YAML config with shapes, stage counts, loopback sizes and pinnings (default: {_DEFAULT_CONFIG.name}).",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=_DEFAULT_OUT,
        help=f"Output directory for the generated MGDs (default: {_DEFAULT_OUT}).",
    )
    ap.add_argument(
        "--shape",
        action="append",
        dest="only_shapes",
        default=None,
        help="Only generate this shape (repeatable). Default: every shape in the config.",
    )
    args = ap.parse_args()

    shapes = _load_config(args.config.resolve())
    selected = list(shapes)
    if args.only_shapes:
        missing = [s for s in args.only_shapes if s not in shapes]
        if missing:
            raise SystemExit(f"{args.config}: unknown shape(s): {', '.join(missing)}")
        selected = list(args.only_shapes)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for shape_name in selected:
        shape = shapes[shape_name]
        for stages in shape.stages:
            path = args.out_dir / f"sweep_{shape_name}_pipeline_{stages}stage_mesh_graph_descriptor.textproto"
            path.write_text(build_mgd(shape, stages))
            written.append(path)

    print(f"Config: {args.config.resolve()}")
    print(f"Wrote {len(written)} MGDs to {args.out_dir}:")
    for path in written:
        print(f"  {path.name}")


if __name__ == "__main__":
    main()
