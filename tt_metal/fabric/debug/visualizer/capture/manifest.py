# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Load a fabric debug manifest, validate it, and enumerate this host's peek targets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


SUPPORTED_MANIFEST_VERSION = 1


class ManifestError(ValueError):
    """The input is not a supported fabric debug manifest."""


class RouterTarget:
    """One fabric-router ERISC that this host should peek."""

    def __init__(
        self,
        mesh_id,
        chip_id,
        eth_chan,
        physical_chip_id,
        logical_core,
        virtual_core,
        direction,
        routing_plane,
        link_class,
        layout_id,
        instance,
    ):
        self.mesh_id = mesh_id
        self.chip_id = chip_id
        self.eth_chan = eth_chan
        self.physical_chip_id = physical_chip_id
        self.logical_core = logical_core
        self.virtual_core = virtual_core
        self.direction = direction
        self.routing_plane = routing_plane
        self.link_class = link_class
        self.layout_id = layout_id
        self.instance = instance

    def endpoint(self):
        """Return the endpoint shape shared by manifests and snapshots."""
        return {
            "mesh_id": self.mesh_id,
            "chip_id": self.chip_id,
            "eth_chan": self.eth_chan,
        }

    def sort_key(self):
        return (self.mesh_id, self.chip_id, self.eth_chan)


class FabricManifest:
    """A validated manifest and the local routers selected from it."""

    def __init__(self, path, data, router_targets, sha256):
        self.path = path
        self.data = data
        self.router_targets = router_targets
        self.sha256 = sha256
        self.run = data["run"]
        self.hal = data["hal"]
        self.heartbeat = data["heartbeat"]
        self.fabric_context = data["fabric_context"]
        self.router_template = data["router_template"]
        self.stream_assignment = data["stream_assignment"]
        self.enums = data["enums"]
        self.layouts = data["layouts"]

    def router_layout(self, mesh_id: int, chip_id: int, eth_chan: int) -> dict[str, Any]:
        """Return the corresponding layout for a local router."""
        for target in self.router_targets:
            if (target.mesh_id, target.chip_id, target.eth_chan) == (mesh_id, chip_id, eth_chan):
                return self.layouts[target.layout_id]
        raise ManifestError(
            f"no local router (mesh_id={mesh_id}, chip_id={chip_id}, eth_chan={eth_chan})"
        )

    def stream_regs_for_router(self, mesh_id: int, chip_id: int, eth_chan: int) -> tuple[int, ...]:
        """Enabled overlay stream ids for the named router's layout, sorted uniquely."""
        layout = self.router_layout(mesh_id, chip_id, eth_chan)
        stream_ids: set[int] = set()
        for region in layout["regions"]:
            if (
                region.get("backing") == "stream_reg"
                and region.get("allocated")
                and region.get("enabled")
                and "stream_id" in region
            ):
                stream_ids.add(region["stream_id"])
        return tuple(sorted(stream_ids))


def _require_object(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ManifestError(f"{location} must be a JSON object")
    return value


def _require_array(value: Any, location: str) -> list[Any]:
    if not isinstance(value, list):
        raise ManifestError(f"{location} must be a JSON array")
    return value


def _require_int(value: Any, location: str) -> int:
    # bool is an int subclass in Python, but is never a valid identifier here.
    if isinstance(value, bool) or not isinstance(value, int):
        raise ManifestError(f"{location} must be an integer")
    return value


def _require_string(value: Any, location: str) -> str:
    if not isinstance(value, str):
        raise ManifestError(f"{location} must be a string")
    return value


def _require_coord(value: Any, location: str) -> tuple[int, int]:
    coord = _require_array(value, location)
    if len(coord) != 2:
        raise ManifestError(f"{location} must contain exactly two coordinates")
    return (
        _require_int(coord[0], f"{location}[0]"),
        _require_int(coord[1], f"{location}[1]"),
    )


def _required(mapping: dict[str, Any], key: str, location: str) -> Any:
    if key not in mapping:
        raise ManifestError(f"{location}.{key} is required")
    return mapping[key]


def _validate_header(data: dict[str, Any]) -> None:
    kind = _required(data, "kind", "manifest")
    if kind != "fabric_debug_manifest":
        raise ManifestError(f"manifest.kind must be 'fabric_debug_manifest', got {kind!r}")

    version = _required(data, "manifest_version", "manifest")
    if version != SUPPORTED_MANIFEST_VERSION:
        raise ManifestError(
            f"unsupported manifest_version {version!r}; expected {SUPPORTED_MANIFEST_VERSION}"
        )

    run = _require_object(_required(data, "run", "manifest"), "manifest.run")
    _require_string(_required(run, "arch", "manifest.run"), "manifest.run.arch")
    _require_string(
        _required(run, "fabric_config", "manifest.run"),
        "manifest.run.fabric_config",
    )
    _require_int(_required(run, "host_rank", "manifest.run"), "manifest.run.host_rank")
    _require_int(_required(run, "mpi_rank", "manifest.run"), "manifest.run.mpi_rank")
    world_size = _require_int(
        _required(run, "world_size", "manifest.run"),
        "manifest.run.world_size",
    )
    if world_size < 1:
        raise ManifestError("manifest.run.world_size must be at least 1")
    _require_string(_required(run, "written_at", "manifest.run"), "manifest.run.written_at")

    for block in (
        "hal",
        "heartbeat",
        "fabric_context",
        "router_template",
        "stream_assignment",
        "enums",
        "layouts",
    ):
        _require_object(_required(data, block, "manifest"), f"manifest.{block}")

    _validate_layouts(data)


def _enumerate_local_router_targets(data: dict[str, Any]) -> tuple[RouterTarget, ...]:
    meshes = _require_array(_required(data, "meshes", "manifest"), "manifest.meshes")
    targets: list[RouterTarget] = []
    endpoints: set[tuple[int, int, int]] = set()

    for mesh_index, raw_mesh in enumerate(meshes):
        mesh_location = f"manifest.meshes[{mesh_index}]"
        mesh = _require_object(raw_mesh, mesh_location)
        mesh_id = _require_int(_required(mesh, "mesh_id", mesh_location), f"{mesh_location}.mesh_id")
        chips = _require_array(_required(mesh, "chips", mesh_location), f"{mesh_location}.chips")

        for chip_index, raw_chip in enumerate(chips):
            chip_location = f"{mesh_location}.chips[{chip_index}]"
            chip = _require_object(raw_chip, chip_location)
            chip_id = _require_int(
                _required(chip, "fabric_chip_id", chip_location),
                f"{chip_location}.fabric_chip_id",
            )
            is_local = _required(chip, "is_local", chip_location)
            if not isinstance(is_local, bool):
                raise ManifestError(f"{chip_location}.is_local must be a boolean")

            master_router_chan = _required(chip, "master_router_chan", chip_location)
            if is_local:
                _require_int(master_router_chan, f"{chip_location}.master_router_chan")
            elif master_router_chan is not None:
                raise ManifestError(f"{chip_location}.master_router_chan must be null for a non-local chip")

            # A non-local chip belongs to another host process. Its empty router
            # list describes topology, but it is not a legal ttexalens target here.
            if not is_local:
                continue

            physical_chip_id = _require_int(
                _required(chip, "physical_chip_id", chip_location),
                f"{chip_location}.physical_chip_id",
            )
            routers = _require_array(
                _required(chip, "routers", chip_location),
                f"{chip_location}.routers",
            )

            for router_index, raw_router in enumerate(routers):
                router_location = f"{chip_location}.routers[{router_index}]"
                router = _require_object(raw_router, router_location)
                eth_chan = _require_int(
                    _required(router, "eth_chan", router_location),
                    f"{router_location}.eth_chan",
                )
                endpoint = (mesh_id, chip_id, eth_chan)
                if endpoint in endpoints:
                    raise ManifestError(
                        "duplicate local router endpoint "
                        f"(mesh_id={mesh_id}, chip_id={chip_id}, eth_chan={eth_chan})"
                    )
                endpoints.add(endpoint)

                layout_id = _require_string(
                    _required(router, "layout_id", router_location),
                    f"{router_location}.layout_id",
                )
                layouts = data["layouts"]
                if layout_id not in layouts:
                    raise ManifestError(f"{router_location}.layout_id {layout_id!r} is not in manifest.layouts")
                instance = _require_object(
                    _required(router, "instance", router_location),
                    f"{router_location}.instance",
                )

                routing_plane_value = router.get("routing_plane")
                routing_plane = (
                    None
                    if routing_plane_value is None
                    else _require_int(routing_plane_value, f"{router_location}.routing_plane")
                )
                targets.append(
                    RouterTarget(
                        mesh_id=mesh_id,
                        chip_id=chip_id,
                        eth_chan=eth_chan,
                        physical_chip_id=physical_chip_id,
                        logical_core=_require_coord(
                            _required(router, "logical_core", router_location),
                            f"{router_location}.logical_core",
                        ),
                        virtual_core=_require_coord(
                            _required(router, "virtual_core", router_location),
                            f"{router_location}.virtual_core",
                        ),
                        direction=_require_string(
                            _required(router, "direction", router_location),
                            f"{router_location}.direction",
                        ),
                        routing_plane=routing_plane,
                        link_class=_require_string(
                            _required(router, "link_class", router_location),
                            f"{router_location}.link_class",
                        ),
                        layout_id=layout_id,
                        instance=instance,
                    )
                )

    return tuple(sorted(targets, key=lambda target: target.sort_key()))


def _validate_layouts(data: dict[str, Any]) -> None:
    layouts = _require_object(_required(data, "layouts", "manifest"), "manifest.layouts")
    hal = _require_object(_required(data, "hal", "manifest"), "manifest.hal")
    unreserved = _require_object(_required(hal, "unreserved", "manifest.hal"), "manifest.hal.unreserved")
    unreserved_base = _require_int(
        _required(unreserved, "base", "manifest.hal.unreserved"),
        "manifest.hal.unreserved.base",
    )
    unreserved_size = _require_int(
        _required(unreserved, "size", "manifest.hal.unreserved"),
        "manifest.hal.unreserved.size",
    )
    unreserved_end = unreserved_base + unreserved_size

    for layout_id, raw_layout in layouts.items():
        location = f"manifest.layouts[{layout_id}]"
        layout = _require_object(raw_layout, location)
        _require_int(_required(layout, "router_count", location), f"{location}.router_count")
        regions = _require_array(_required(layout, "regions", location), f"{location}.regions")
        parsed_regions: list[tuple[str, dict[str, Any]]] = []
        ids: set[str] = set()
        for region_index, raw_region in enumerate(regions):
            region_location = f"{location}.regions[{region_index}]"
            region = _require_object(raw_region, region_location)
            region_id = _require_string(_required(region, "id", region_location), f"{region_location}.id")
            if region_id in ids:
                raise ManifestError(f"{region_location}.id {region_id!r} is duplicated")
            ids.add(region_id)
            parsed_regions.append((region_location, region))

        for region_location, region in parsed_regions:
            parent = _require_string(_required(region, "parent", region_location), f"{region_location}.parent")
            if parent and parent not in ids:
                raise ManifestError(f"{region_location}.parent {parent!r} is unknown")

            backing = _require_string(_required(region, "backing", region_location), f"{region_location}.backing")
            allocated = _required(region, "allocated", region_location)
            enabled = _required(region, "enabled", region_location)
            if not isinstance(allocated, bool):
                raise ManifestError(f"{region_location}.allocated must be a boolean")
            if not isinstance(enabled, bool):
                raise ManifestError(f"{region_location}.enabled must be a boolean")

            if backing in ("unreserved_l1", "fixed_l1"):
                address = _require_int(_required(region, "address", region_location), f"{region_location}.address")
                size = _require_int(_required(region, "size", region_location), f"{region_location}.size")
                if backing == "unreserved_l1" and allocated and size != 0:
                    end = address + size
                    if address < unreserved_base or end > unreserved_end:
                        raise ManifestError(
                            f"{region_location} [{address}, {end}) lies outside UNRESERVED "
                            f"[{unreserved_base}, {unreserved_end})"
                        )
            elif backing == "stream_reg" and allocated:
                stream_id = _require_int(
                    _required(region, "stream_id", region_location),
                    f"{region_location}.stream_id",
                )
                if stream_id < 0 or stream_id >= 32:
                    raise ManifestError(f"{region_location}.stream_id must be in [0, 32)")

            if region.get("schema", "") == "packet_ring" and all(
                key in region for key in ("count", "stride", "size")
            ):
                count = _require_int(region["count"], f"{region_location}.count")
                stride = _require_int(region["stride"], f"{region_location}.stride")
                size = _require_int(region["size"], f"{region_location}.size")
                if count * stride != size:
                    raise ManifestError(f"{region_location} count*stride must equal size")


def load_manifest(path: str | Path) -> FabricManifest:
    """Load *path* and return its local fabric-router peek targets."""

    manifest_path = Path(path)
    try:
        raw_bytes = manifest_path.read_bytes()
    except OSError as error:
        raise ManifestError(f"could not read manifest {manifest_path}: {error}") from error

    sha256 = hashlib.sha256(raw_bytes).hexdigest()
    try:
        raw_data = json.loads(raw_bytes)
    except json.JSONDecodeError as error:
        raise ManifestError(f"manifest {manifest_path} is not valid JSON: {error}") from error

    data = _require_object(raw_data, "manifest")
    _validate_header(data)
    targets = _enumerate_local_router_targets(data)
    return FabricManifest(
        path=manifest_path.resolve(),
        data=data,
        router_targets=targets,
        sha256=sha256,
    )
