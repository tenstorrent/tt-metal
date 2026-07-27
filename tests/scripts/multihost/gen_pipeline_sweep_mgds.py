#!/usr/bin/env python3
"""Generate the pipeline-stage Mesh Graph Descriptors swept by the bh-pipeline-sweep group.

This emits two families of ring-pipeline MGDs used to stress the topology mapper on the SC36
mock cluster (36 hosts x 32 ASICs = 1152 ASICs total):

  * 2x4 shape  -- device_topology [4, 2] RING,LINE (8 ASICs/stage, the repo's canonical "2x4
                  pipeline"). Single-host stages, no pinnings. Stage counts come from the config.

  * 4x4 shape  -- device_topology [4, 4] RING,RING (16 ASICs/stage). Single mesh type (M0): every
                  stage is a 4x4 torus split across a "1x2" split-host galaxy (host_topology [2, 1],
                  which is how the schema expresses the 1x2 split, matching blitz_decode_quad_galaxy_4x4).
                  All-to-all corner pinnings per stage (defined in the config file). Stage counts come
                  from the config.

All rings close (last stage -> stage 0). Regenerate in place with:

    python3 tests/scripts/multihost/gen_pipeline_sweep_mgds.py

Pinnings are read from pipeline_sweep_config.yaml (override with --config).
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
class ShapeConfig:
    name: str
    stages: list[int]
    pinnings: list[PinningGroup]


def _load_config(path: Path) -> dict[str, ShapeConfig]:
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict) or "shapes" not in raw:
        raise ValueError(f"{path}: expected top-level 'shapes' mapping")

    shapes: dict[str, ShapeConfig] = {}
    for shape_name, shape_cfg in raw["shapes"].items():
        if not isinstance(shape_cfg, dict):
            raise ValueError(f"{path}: shapes.{shape_name} must be a mapping")
        stages = shape_cfg.get("stages")
        if not stages:
            raise ValueError(f"{path}: shapes.{shape_name}.stages must be a non-empty list")

        pinning_groups: list[PinningGroup] = []
        pinnings_cfg = shape_cfg.get("pinnings")
        if pinnings_cfg is not None:
            groups = pinnings_cfg.get("groups")
            if not groups:
                raise ValueError(f"{path}: shapes.{shape_name}.pinnings.groups must be non-empty")
            for i, group in enumerate(groups):
                pinning_groups.append(_parse_pinning_group(path, shape_name, i, group))

        shapes[shape_name] = ShapeConfig(name=shape_name, stages=list(stages), pinnings=pinning_groups)
    return shapes


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


def _instances(template_of_stage: list[str]) -> str:
    lines = []
    for i, tmpl in enumerate(template_of_stage):
        lines.append(f'  instances {{ mesh {{ mesh_descriptor: "{tmpl}" mesh_id: {i} }} }}')
    return "\n".join(lines)


def _ring_connections(template_of_stage: list[str], *, channels: int, assign_z: bool) -> str:
    n = len(template_of_stage)
    blocks = []
    for i in range(n):
        j = (i + 1) % n  # last edge wraps back to 0 -> closed ring
        src, dst = template_of_stage[i], template_of_stage[j]
        block = [
            "  connections {",
            f'    nodes {{ mesh {{ mesh_descriptor: "{src}" mesh_id: {i} }} }}',
            f'    nodes {{ mesh {{ mesh_descriptor: "{dst}" mesh_id: {j} }} }}',
            f"    channels {{ count: {channels} policy: RELAXED }}",
        ]
        if assign_z:
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
        "# by mesh_id_regex (parity-based for the 1x2 split-host galaxy wiring).",
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


def build_2x4(stages: int) -> str:
    template_of_stage = ["M0"] * stages
    return f"""# --- Meshes ---------------------------------------------------------------
# GENERATED by tests/scripts/multihost/gen_pipeline_sweep_mgds.py -- do not edit by hand.
# 2x4 pipeline ring, {stages} stages ({stages} x 8 = {stages * 8} ASICs). Each stage is a 4x2 RING,LINE
# single-host galaxy slice; the ring closes (stage {stages - 1} -> stage 0). No pinnings (single-host
# stages map freely on the SC36 mock).

mesh_descriptors {{
  name: "M0"
  arch: BLACKHOLE
  device_topology {{ dims: [ 4, 2 ] dim_types: [ RING, LINE ] }}
  host_topology   {{ dims: [ 1, 1 ] }}
  channels {{ count: 2 policy: RELAXED }}
}}

graph_descriptors {{
  name: "G0"
  type: "FABRIC"
{_instances(template_of_stage)}

{_ring_connections(template_of_stage, channels=8, assign_z=False)}
}}

# --- Instantiation ----------------------------------------------------------
top_level_instance {{ graph {{ graph_descriptor: "G0" graph_id: 0 }} }}
"""


def build_4x4(stages: int, pinnings: list[PinningGroup]) -> str:
    template_of_stage = ["M0"] * stages
    pinnings_block = _pinnings_section(pinnings)
    return f"""# --- Meshes ---------------------------------------------------------------
# GENERATED by tests/scripts/multihost/gen_pipeline_sweep_mgds.py -- do not edit by hand.
# 4x4 pipeline ring, {stages} stages ({stages} x 16 = {stages * 16} ASICs). Single mesh type (M0):
# every stage is a 4x4 RING,RING torus split across a 1x2 split-host galaxy (host_topology [2,1]).
# All-to-all corner pinnings from pipeline_sweep_config.yaml; position set alternates by mesh-id
# parity. The ring closes (stage {stages - 1} -> stage 0). Mirrors blitz_decode_quad_galaxy_4x4
# scaled to {stages} stages.

mesh_descriptors {{
  name: "M0"
  arch: BLACKHOLE
  device_topology {{ dims: [ 4, 4 ] dim_types: [ RING, RING ] }}
  host_topology   {{ dims: [ 2, 1 ] }}
  channels {{ count: 2 policy: STRICT }}
}}

graph_descriptors {{
  name: "G0"
  type: "FABRIC"
{_instances(template_of_stage)}

{_ring_connections(template_of_stage, channels=8, assign_z=False)}
}}

# --- Instantiation ----------------------------------------------------------
top_level_instance {{ graph {{ graph_descriptor: "G0" graph_id: 0 }} }}

{pinnings_block}"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--config",
        type=Path,
        default=_DEFAULT_CONFIG,
        help=f"YAML config with stage counts and pinnings (default: {_DEFAULT_CONFIG.name}).",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=_DEFAULT_OUT,
        help=f"Output directory for the generated MGDs (default: {_DEFAULT_OUT}).",
    )
    args = ap.parse_args()

    shapes = _load_config(args.config.resolve())
    if "2x4" not in shapes or "4x4" not in shapes:
        raise SystemExit(f"{args.config}: expected both '2x4' and '4x4' under shapes")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for stages in shapes["2x4"].stages:
        path = args.out_dir / f"sweep_2x4_pipeline_{stages}stage_mesh_graph_descriptor.textproto"
        path.write_text(build_2x4(stages))
        written.append(path)
    for stages in shapes["4x4"].stages:
        path = args.out_dir / f"sweep_4x4_pipeline_{stages}stage_mesh_graph_descriptor.textproto"
        path.write_text(build_4x4(stages, shapes["4x4"].pinnings))
        written.append(path)

    print(f"Config: {args.config.resolve()}")
    print(f"Wrote {len(written)} MGDs to {args.out_dir}:")
    for path in written:
        print(f"  {path.name}")


if __name__ == "__main__":
    main()
