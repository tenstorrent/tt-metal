# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Calibrated plan (per-layer owner rows) -> placement preset (kind ``bijective_within_tp_group``).

Preset JSON: kind, model, mesh_shape [rows, groups], num_experts, top_k, source_main, calibration provenance,
and ``layers`` keyed by global layer id, each {enabled, physical_to_logical}. Layers whose plan equals the
original layout are omitted. ``physical_to_logical`` is written one chip (experts_per_chip slots) per line.
A sidecar ``<preset>.calibration.json`` stores the exact owner rows the preset was built from.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np

from .spec import MoeSpec, inverse, validate_mapping

KIND = "bijective_within_tp_group"


def build(
    spec: MoeSpec, layers, owners_per_layer, plan_name: str, source: str = "", source_main: str = ""
) -> tuple[dict, bytes, dict]:
    calib = dict(
        plan=plan_name,
        source=source,
        layers=list(map(int, layers)),
        owners=np.asarray(owners_per_layer).astype(int).tolist(),
    )
    calib_bytes = json.dumps(calib, sort_keys=True, separators=(",", ":")).encode()
    out_layers, moved = {}, {}
    identity = spec.identity_owners()
    for layer, owners in zip(layers, owners_per_layer):
        owners = np.asarray(owners, dtype=np.int64)
        if (owners == identity).all():
            continue
        phys = spec.physical_from_owners(owners)
        if not (spec.owners_from_physical(phys) == owners).all():
            raise AssertionError(f"layer {layer}: preset does not reproduce the plan")
        out_layers[str(int(layer))] = dict(enabled=True, physical_to_logical=phys)
        moved[int(layer)] = spec.moved_experts(phys)
    preset = dict(
        kind=KIND,
        model=spec.name,
        mesh_shape=[spec.rows, spec.groups],
        num_experts=spec.num_experts,
        top_k=spec.top_k,
        source_main=source_main,
        calibration_plan_sha256=hashlib.sha256(calib_bytes).hexdigest(),
        layers=out_layers,
    )
    stats = dict(
        layers=len(out_layers),
        omitted=[int(l) for l in layers if str(int(l)) not in out_layers],
        moved_experts=sum(moved.values()),
        moved_per_layer_min_max=[min(moved.values()), max(moved.values())] if moved else [0, 0],
        calibration_plan_sha256=preset["calibration_plan_sha256"],
    )
    return preset, calib_bytes, stats


def dumps(preset: dict, experts_per_chip: int) -> str:
    text = json.dumps(preset, indent=2)

    def fmt(m):
        nums = [x.strip() for x in m.group(1).split(",")]
        rows = [", ".join(nums[i : i + experts_per_chip]) for i in range(0, len(nums), experts_per_chip)]
        return '"physical_to_logical": [\n' + ",\n".join(" " * 8 + r for r in rows) + "\n      ]"

    text = re.sub(r'"physical_to_logical": \[([\s\d,]+?)\]', fmt, text)
    if json.loads(text) != preset:
        raise AssertionError("preset formatting changed its content")
    return text + "\n"


def write(path: Path, spec: MoeSpec, preset: dict, calib_bytes: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps(preset, spec.experts_per_chip))
    path.with_suffix(".calibration.json").write_bytes(calib_bytes)


def load(path: Path, spec: MoeSpec) -> dict:
    """Read and validate a preset against a spec: kind, model topology, every layer a valid placement."""
    p = json.loads(Path(path).read_text())
    if p.get("kind") != KIND or not isinstance(p.get("layers"), dict):
        raise ValueError(f"{path}: not a {KIND} preset")
    if "num_experts" in p and p["num_experts"] != spec.num_experts:
        raise ValueError(f"{path}: preset has {p['num_experts']} experts, spec {spec.num_experts}")
    if "mesh_shape" in p and p["mesh_shape"] != [spec.rows, spec.groups]:
        raise ValueError(f"{path}: preset mesh {p['mesh_shape']} differs from spec {[spec.rows, spec.groups]}")
    for key, entry in p["layers"].items():
        phys = validate_mapping(entry["physical_to_logical"], spec)
        if "logical_to_physical" in entry and entry["logical_to_physical"] != inverse(phys):
            raise ValueError(f"{path}: layer {key} inverse mismatch")
    return p


def owners_per_layer(preset: dict, spec: MoeSpec, layers) -> list:
    """Per-layer owner rows of a preset (identity for layers it omits), e.g. to score a committed preset."""
    return [
        spec.owners_from_physical(preset["layers"][str(l)]["physical_to_logical"])
        if str(l) in preset["layers"]
        else spec.identity_owners()
        for l in layers
    ]
