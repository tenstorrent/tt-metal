# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Load a fabric debug manifest, validate it, and enumerate this host's peek targets."""

from __future__ import annotations

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

    def __init__(self, path, data, router_targets):
        self.path = path
        self.data = data
        self.router_targets = router_targets
        self.run = data["run"]


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
                    )
                )

    return tuple(sorted(targets, key=lambda target: target.sort_key()))


def load_manifest(path: str | Path) -> FabricManifest:
    """Load *path* and return its local fabric-router peek targets."""

    manifest_path = Path(path)
    try:
        with manifest_path.open(encoding="utf-8") as manifest_file:
            raw_data = json.load(manifest_file)
    except OSError as error:
        raise ManifestError(f"could not read manifest {manifest_path}: {error}") from error
    except json.JSONDecodeError as error:
        raise ManifestError(f"manifest {manifest_path} is not valid JSON: {error}") from error

    data = _require_object(raw_data, "manifest")
    _validate_header(data)
    targets = _enumerate_local_router_targets(data)
    return FabricManifest(path=manifest_path.resolve(), data=data, router_targets=targets)
