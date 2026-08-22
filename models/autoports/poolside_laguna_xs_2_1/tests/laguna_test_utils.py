# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Shared topology and diagnostics helpers for Laguna bring-up harnesses.

This module deliberately has no TTNN import at module scope.  Profile parsing and
validation can therefore run on development hosts without a device-enabled TTNN
installation; hardware is touched only by :func:`open_mesh` and memory helpers.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence


TESTS_DIR = Path(__file__).resolve().parent
MODEL_DIR = TESTS_DIR.parent
DOC_DIR = MODEL_DIR / "doc"
REPO_ROOT = MODEL_DIR.parents[2]
P150_MESH_GRAPH_DESC = (
    REPO_ROOT / "tt_metal" / "fabric" / "mesh_graph_descriptors" / "p150_mesh_graph_descriptor.textproto"
)

PROFILE_ENV = "LAGUNA_PROFILE"
FABRIC_ENV = "LAGUNA_FABRIC_CONFIG"
LEGACY_MESH_ENV = "TT_LAGUNA_MESH"
VISIBLE_DEVICES_ENV = "TT_VISIBLE_DEVICES"
CCL_TOPOLOGY_ENV = "TT_LAGUNA_CCL_TOPOLOGY"
CCL_NUM_LINKS_ENV = "TT_LAGUNA_CCL_NUM_LINKS"
MESH_GRAPH_DESC_ENV = "TT_MESH_GRAPH_DESC_PATH"


@dataclass(frozen=True)
class LagunaTestProfile:
    """A qualified serving topology used by correctness and performance harnesses."""

    name: str
    mesh_shape: tuple[int, int]
    fabric_config: str
    max_context: int
    trace_region_size: int = 1_500_000_000
    mesh_graph_desc_path: str | None = None

    @property
    def num_devices(self) -> int:
        return self.mesh_shape[0] * self.mesh_shape[1]

    @property
    def ccl_topology(self) -> str | None:
        if self.num_devices == 1:
            return None
        return "ring" if self.fabric_config == "FABRIC_1D_RING" else "linear"


PROFILES: Mapping[str, LagunaTestProfile] = {
    "p150": LagunaTestProfile("p150", (1, 1), "DISABLED", 65_536),
    # Ring/two-link is the conservative D2 default. Qualification can select linear/one-link via
    # LAGUNA_FABRIC_CONFIG + TT_LAGUNA_CCL_TOPOLOGY/TT_LAGUNA_CCL_NUM_LINKS.
    "p150x2": LagunaTestProfile("p150x2", (1, 2), "FABRIC_1D_RING", 131_072),
    "p150x4": LagunaTestProfile("p150x4", (1, 4), "FABRIC_1D_RING", 131_072),
}

_PROFILE_ALIASES = {
    "1": "p150",
    "d1": "p150",
    "1x1": "p150",
    "1,1": "p150",
    "2": "p150x2",
    "d2": "p150x2",
    "1x2": "p150x2",
    "1,2": "p150x2",
    "4": "p150x4",
    "d4": "p150x4",
    "1x4": "p150x4",
    "1,4": "p150x4",
}
_FABRIC_ALIASES = {
    "disabled": "DISABLED",
    "none": "DISABLED",
    "linear": "FABRIC_1D",
    "fabric_1d": "FABRIC_1D",
    "ring": "FABRIC_1D_RING",
    "fabric_1d_ring": "FABRIC_1D_RING",
}


def _canonical_profile(value: str) -> str:
    key = value.strip().lower().replace(" ", "")
    key = _PROFILE_ALIASES.get(key, key)
    if key not in PROFILES:
        choices = ", ".join(PROFILES)
        raise ValueError(f"unknown Laguna profile {value!r}; expected one of: {choices}")
    return key


def _canonical_fabric(value: str) -> str:
    key = value.strip().lower()
    key = _FABRIC_ALIASES.get(key, value.strip().upper())
    if key not in {"DISABLED", "FABRIC_1D", "FABRIC_1D_RING"}:
        raise ValueError(
            f"unsupported Laguna fabric {value!r}; expected DISABLED, FABRIC_1D/linear, "
            "or FABRIC_1D_RING/ring"
        )
    return key


def _visible_device_count(value: str) -> int:
    devices = [token.strip() for token in value.split(",") if token.strip()]
    if not devices or len(set(devices)) != len(devices):
        raise ValueError(f"{VISIBLE_DEVICES_ENV} must contain distinct comma-separated device IDs")
    return len(devices)


def resolve_profile(
    profile: str | None = None,
    *,
    fabric_config: str | None = None,
    trace_region_size: int | None = None,
    environ: Mapping[str, str] | None = None,
    validate_visible_devices: bool = True,
) -> LagunaTestProfile:
    """Resolve CLI/env topology settings and reject mismatched device selections.

    Precedence is explicit argument, ``LAGUNA_PROFILE``, legacy
    ``TT_LAGUNA_MESH``, then the established P150x4 regression profile.
    """

    env = os.environ if environ is None else environ
    selected = profile or env.get(PROFILE_ENV)
    legacy_mesh = env.get(LEGACY_MESH_ENV)
    if selected is None and legacy_mesh:
        selected = legacy_mesh
    selected = _canonical_profile(selected or "p150x4")
    spec = PROFILES[selected]

    if legacy_mesh and _canonical_profile(legacy_mesh) != selected:
        raise ValueError(
            f"{PROFILE_ENV}={selected} conflicts with {LEGACY_MESH_ENV}={legacy_mesh}; "
            "remove the legacy override"
        )

    fabric_value = fabric_config or env.get(FABRIC_ENV)
    ccl_topology = env.get(CCL_TOPOLOGY_ENV)
    if ccl_topology and spec.num_devices > 1:
        ccl_topology = ccl_topology.strip().lower()
        if ccl_topology not in {"linear", "ring"}:
            raise ValueError(f"{CCL_TOPOLOGY_ENV} must be linear or ring")
        derived_fabric = "FABRIC_1D" if ccl_topology == "linear" else "FABRIC_1D_RING"
        if fabric_value and _canonical_fabric(fabric_value) != derived_fabric:
            raise ValueError(
                f"{FABRIC_ENV}={fabric_value} conflicts with {CCL_TOPOLOGY_ENV}={ccl_topology}"
            )
        fabric_value = derived_fabric
    if fabric_value:
        fabric = _canonical_fabric(fabric_value)
        if spec.num_devices == 1 and fabric != "DISABLED":
            raise ValueError("the p150 profile requires fabric DISABLED")
        if spec.num_devices > 1 and fabric == "DISABLED":
            raise ValueError(f"the {spec.name} profile requires a 1D fabric")
        spec = replace(spec, fabric_config=fabric)

    links = env.get(CCL_NUM_LINKS_ENV)
    if links and spec.num_devices > 1:
        try:
            num_links = int(links)
        except ValueError as exc:
            raise ValueError(f"{CCL_NUM_LINKS_ENV} must be 1 or 2") from exc
        if num_links not in {1, 2}:
            raise ValueError(f"{CCL_NUM_LINKS_ENV} must be 1 or 2")

    if trace_region_size is not None:
        if trace_region_size <= 0:
            raise ValueError("trace region size must be positive")
        spec = replace(spec, trace_region_size=trace_region_size)

    graph_desc_value = env.get(MESH_GRAPH_DESC_ENV)
    if spec.num_devices == 1:
        graph_desc = Path(graph_desc_value) if graph_desc_value else P150_MESH_GRAPH_DESC
        if not graph_desc.is_file():
            raise ValueError(
                f"{MESH_GRAPH_DESC_ENV} must name an existing P150 mesh graph descriptor; got {graph_desc}"
            )
        spec = replace(spec, mesh_graph_desc_path=str(graph_desc.resolve()))
    elif graph_desc_value:
        graph_desc = Path(graph_desc_value)
        try:
            is_singleton_desc = graph_desc.resolve() == P150_MESH_GRAPH_DESC.resolve()
        except OSError:
            is_singleton_desc = graph_desc.name == P150_MESH_GRAPH_DESC.name
        if is_singleton_desc:
            raise ValueError(
                f"unset singleton {MESH_GRAPH_DESC_ENV} when running the multi-device {spec.name} profile"
            )

    visible = env.get(VISIBLE_DEVICES_ENV)
    if validate_visible_devices and visible is not None:
        count = _visible_device_count(visible)
        if count != spec.num_devices:
            raise ValueError(
                f"{VISIBLE_DEVICES_ENV} exposes {count} device(s), but {spec.name} requires {spec.num_devices}"
            )
    return spec


def add_profile_args(parser: argparse.ArgumentParser, *, default_trace_region_size: int | None = None) -> None:
    """Add the common profile overrides to an existing command-line harness."""

    parser.add_argument(
        "--profile",
        choices=tuple(PROFILES),
        default=None,
        help=f"device profile (default: ${PROFILE_ENV}, then p150x4)",
    )
    parser.add_argument(
        "--fabric-config",
        choices=("DISABLED", "FABRIC_1D", "FABRIC_1D_RING"),
        default=None,
        help=f"qualified fabric override (default: ${FABRIC_ENV} or profile default)",
    )
    parser.add_argument(
        "--trace-region-size",
        type=int,
        default=default_trace_region_size,
        help="trace allocation in bytes",
    )


def profile_from_args(args: argparse.Namespace, *, environ: Mapping[str, str] | None = None) -> LagunaTestProfile:
    return resolve_profile(
        getattr(args, "profile", None),
        fabric_config=getattr(args, "fabric_config", None),
        trace_region_size=getattr(args, "trace_region_size", None),
        environ=environ,
    )


def fabric_enum(ttnn_module, profile: LagunaTestProfile):
    try:
        return getattr(ttnn_module.FabricConfig, profile.fabric_config)
    except AttributeError as exc:
        raise RuntimeError(f"this TTNN build does not provide FabricConfig.{profile.fabric_config}") from exc


def open_mesh(ttnn_module, profile: LagunaTestProfile):
    """Configure fabric before opening the exact mesh described by ``profile``."""

    # A singleton half-P300 is reported as ClusterType.CUSTOM. SetFabricConfig, including
    # SetFabricConfig(DISABLED), requires a custom fabric graph for that cluster and fails before
    # open. D1 has no collectives, so leave global fabric state untouched exactly as the runtime
    # plugin does. Multi-device profiles still require configuration before open.
    if profile.num_devices == 1:
        if not profile.mesh_graph_desc_path:
            raise ValueError("the p150 profile must be resolved with a singleton mesh graph descriptor")
        os.environ[MESH_GRAPH_DESC_ENV] = profile.mesh_graph_desc_path
    else:
        current_desc = os.environ.get(MESH_GRAPH_DESC_ENV)
        if current_desc and Path(current_desc).resolve() == P150_MESH_GRAPH_DESC.resolve():
            # Mixed comparison harnesses may open D1 and then D2/D4 in one process. Do not leak
            # the singleton custom-graph override into the subsequent multi-device mesh open.
            os.environ.pop(MESH_GRAPH_DESC_ENV, None)
        os.environ.setdefault(CCL_TOPOLOGY_ENV, profile.ccl_topology or "ring")
        os.environ.setdefault(CCL_NUM_LINKS_ENV, "2")
        ttnn_module.set_fabric_config(fabric_enum(ttnn_module, profile))
    try:
        mesh = ttnn_module.open_mesh_device(
            ttnn_module.MeshShape(*profile.mesh_shape), trace_region_size=profile.trace_region_size
        )
    except Exception:
        if profile.num_devices > 1:
            ttnn_module.set_fabric_config(ttnn_module.FabricConfig.DISABLED)
        raise
    actual = mesh.get_num_devices()
    if actual != profile.num_devices:
        ttnn_module.close_mesh_device(mesh)
        if profile.num_devices > 1:
            ttnn_module.set_fabric_config(ttnn_module.FabricConfig.DISABLED)
        raise RuntimeError(f"opened {actual} devices for {profile.name}; expected {profile.num_devices}")
    return mesh


def close_mesh(ttnn_module, mesh) -> None:
    num_devices = mesh.get_num_devices()
    try:
        ttnn_module.close_mesh_device(mesh)
    finally:
        if num_devices > 1:
            ttnn_module.set_fabric_config(ttnn_module.FabricConfig.DISABLED)


def mesh_mapper(ttnn_module, mesh, profile: LagunaTestProfile):
    """Replicate inputs only when a tensor actually spans multiple devices."""

    return ttnn_module.ReplicateTensorToMesh(mesh) if profile.num_devices > 1 else None


def compose_replicated(ttnn_module, tensor, mesh, profile: LagunaTestProfile):
    if profile.num_devices == 1:
        return ttnn_module.to_torch(tensor)
    return ttnn_module.to_torch(tensor, mesh_composer=ttnn_module.ConcatMeshToTensor(mesh, dim=0))[0:1]


def memory_snapshot(ttnn_module, mesh, label: str, *, synchronize: bool = True) -> dict[str, int | float | str]:
    """Return a JSON-friendly DRAM residency snapshot for one device or a mesh."""

    if synchronize:
        ttnn_module.synchronize_device(mesh)
    view = ttnn_module.get_memory_view(mesh, ttnn_module.BufferType.DRAM)
    banks = int(view.num_banks)
    total_per_bank = int(view.total_bytes_per_bank)
    allocated_per_bank = int(view.total_bytes_allocated_per_bank)
    free_per_bank = int(view.total_bytes_free_per_bank)
    largest_per_bank = int(view.largest_contiguous_bytes_free_per_bank)
    total = banks * total_per_bank
    allocated = banks * allocated_per_bank
    free = banks * free_per_bank
    return {
        "label": label,
        "num_banks": banks,
        "total_bytes": total,
        "allocated_bytes": allocated,
        "free_bytes": free,
        "free_fraction": (free / total) if total else 0.0,
        "total_bytes_per_bank": total_per_bank,
        "allocated_bytes_per_bank": allocated_per_bank,
        "free_bytes_per_bank": free_per_bank,
        "largest_contiguous_bytes_free_per_bank": largest_per_bank,
    }


def print_memory_snapshot(ttnn_module, mesh, label: str, *, synchronize: bool = True) -> dict[str, int | float | str]:
    snapshot = memory_snapshot(ttnn_module, mesh, label, synchronize=synchronize)
    print("MEMORY", json.dumps(snapshot, sort_keys=True))
    return snapshot


def assert_memory_margin(
    snapshot: Mapping[str, int | float | str],
    *,
    min_free_fraction: float = 0.10,
    min_contiguous_bytes_per_bank: int = 128 * 1024 * 1024,
) -> None:
    """Apply the P150 serving qualification margin to a captured snapshot."""

    free_fraction = float(snapshot["free_fraction"])
    largest = int(snapshot["largest_contiguous_bytes_free_per_bank"])
    if free_fraction < min_free_fraction or largest < min_contiguous_bytes_per_bank:
        raise AssertionError(
            f"DRAM margin failed at {snapshot['label']}: free={free_fraction:.2%} "
            f"(need {min_free_fraction:.2%}), largest contiguous/bank={largest / 2**20:.1f} MiB "
            f"(need {min_contiguous_bytes_per_bank / 2**20:.1f} MiB)"
        )


def parse_layers(value: str | None) -> list[int] | None:
    return [int(item) for item in value.split(",")] if value else None


def profile_summary(profile: LagunaTestProfile) -> dict[str, str | int | Sequence[int] | None]:
    return {
        "profile": profile.name,
        "mesh_shape": list(profile.mesh_shape),
        "num_devices": profile.num_devices,
        "fabric_config": profile.fabric_config,
        "ccl_topology": profile.ccl_topology,
        "max_context": profile.max_context,
        "trace_region_size": profile.trace_region_size,
        "mesh_graph_desc_path": profile.mesh_graph_desc_path,
    }
