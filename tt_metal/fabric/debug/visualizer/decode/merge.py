# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Merge per-rank manifests and snapshots into one topology and router set."""

from __future__ import annotations

import collections
from copy import deepcopy
from typing import Any

from .inputs import DecodeError, DecodeInput

Endpoint = tuple[int, int, int]


def endpoint_key(value: dict[str, Any]) -> Endpoint:
    try:
        return (int(value["mesh_id"]), int(value["chip_id"]), int(value["eth_chan"]))
    except (KeyError, TypeError, ValueError) as error:
        raise DecodeError(f"invalid router endpoint {value!r}") from error


def endpoint_object(key: Endpoint) -> dict[str, int]:
    return {"mesh_id": key[0], "chip_id": key[1], "eth_chan": key[2]}


def _merge_optional(current: Any, incoming: Any, location: str) -> Any:
    if current is None:
        return deepcopy(incoming)
    if incoming is None:
        return current
    if current != incoming:
        raise DecodeError(f"{location} disagrees across manifests: {current!r} != {incoming!r}")
    return current


def _manifest_topology(inputs: tuple[DecodeInput, ...]):
    meshes: dict[int, dict[str, Any]] = {}
    chips: dict[tuple[int, int], dict[str, Any]] = {}
    routers: dict[Endpoint, dict[str, Any]] = {}
    links: dict[tuple[Endpoint, Endpoint], dict[str, Any]] = {}

    for item in inputs:
        data = item.manifest.manifest.data
        for mesh in data["meshes"]:
            mesh_id = int(mesh["mesh_id"])
            description = {key: deepcopy(value) for key, value in mesh.items() if key != "chips"}
            if mesh_id in meshes and meshes[mesh_id] != description:
                raise DecodeError(f"mesh {mesh_id} disagrees across manifests")
            meshes[mesh_id] = description
            for chip in mesh["chips"]:
                chip_id = int(chip["fabric_chip_id"])
                key = (mesh_id, chip_id)
                candidate = {
                    "fabric_chip_id": chip_id,
                    "mesh_coord": deepcopy(chip.get("mesh_coord")),
                    "physical_chip_id": chip.get("physical_chip_id"),
                    "asic_id": chip.get("asic_id"),
                }
                if key in chips:
                    previous = chips[key]
                    previous["mesh_coord"] = _merge_optional(
                        previous["mesh_coord"], candidate["mesh_coord"], f"chip {key}.mesh_coord"
                    )
                    previous["physical_chip_id"] = _merge_optional(
                        previous["physical_chip_id"],
                        candidate["physical_chip_id"],
                        f"chip {key}.physical_chip_id",
                    )
                    previous["asic_id"] = _merge_optional(
                        previous["asic_id"], candidate["asic_id"], f"chip {key}.asic_id"
                    )
                else:
                    chips[key] = candidate

                for router in chip.get("routers", []):
                    endpoint = (mesh_id, chip_id, int(router["eth_chan"]))
                    candidate_router = deepcopy(router)
                    if endpoint in routers and routers[endpoint] != candidate_router:
                        raise DecodeError(f"router {endpoint} disagrees across manifests")
                    routers[endpoint] = candidate_router

        for link in data["links"]:
            src = endpoint_key(link["src"])
            dst = endpoint_key(link["dst"])
            key = (src, dst)
            if key in links and links[key] != link:
                raise DecodeError(f"link {src}->{dst} disagrees across manifests")
            links[key] = deepcopy(link)
            routers.setdefault(src, {"eth_chan": src[2]})
            routers.setdefault(dst, {"eth_chan": dst[2]})

    topology_meshes = []
    for mesh_id in sorted(meshes):
        result_mesh = deepcopy(meshes[mesh_id])
        result_chips = []
        for key in sorted(key for key in chips if key[0] == mesh_id):
            result_chip = deepcopy(chips[key])
            result_chip["routers"] = [
                endpoint_object(endpoint)
                for endpoint in sorted(routers)
                if endpoint[:2] == key
            ]
            result_chips.append(result_chip)
        result_mesh["chips"] = result_chips
        topology_meshes.append(result_mesh)
    return topology_meshes, routers, [links[key] for key in sorted(links)]


def merge_inputs(inputs: tuple[DecodeInput, ...]) -> dict[str, Any]:
    """Return merged topology, router skeletons, and coverage."""

    if not inputs:
        raise DecodeError("cannot merge an empty input set")
    meshes, router_metadata, links = _manifest_topology(inputs)
    owners: dict[Endpoint, tuple[int, dict[str, Any]]] = {}

    for input_index, item in enumerate(inputs):
        if item.snapshot is None:
            continue
        local_chips: set[tuple[int, int]] = set()
        for mesh in item.manifest.manifest.data["meshes"]:
            for chip in mesh["chips"]:
                if chip.get("is_local"):
                    local_chips.add((int(mesh["mesh_id"]), int(chip["fabric_chip_id"])))

        for sample in item.snapshot["samples"][0]["routers"]:
            endpoint = endpoint_key(sample["id"])
            if endpoint[:2] not in local_chips:
                raise DecodeError(
                    f"{item.snapshot_path}: router {endpoint} is not local in its paired manifest"
                )
            if endpoint in owners:
                raise DecodeError(f"router {endpoint} is owned by more than one snapshot")
            owners[endpoint] = (input_index, sample)
            router_metadata.setdefault(endpoint, {"eth_chan": endpoint[2]})

    routers = []
    counts: collections.Counter[str] = collections.Counter()
    identity_mismatch = 0
    manifest_sha_unverified = 0
    for endpoint in sorted(router_metadata):
        metadata = router_metadata[endpoint]
        owner = owners.get(endpoint)
        if owner is None:
            status = "not_captured"
            counts[status] += 1
            routers.append(
                {
                    "id": endpoint_object(endpoint),
                    "layout_id": metadata.get("layout_id"),
                    "instance": deepcopy(metadata.get("instance")),
                    "direction": metadata.get("direction"),
                    "routing_plane": metadata.get("routing_plane"),
                    "link_class": metadata.get("link_class"),
                    "capture": {
                        "status": status,
                        "error": "no local snapshot supplied",
                        "snapshot_index": None,
                        "torn": False,
                        "owner_alive": None,
                        "manifest_sha_verified": None,
                    },
                    "identity": {"my_mesh_id": None, "my_device_id": None, "matches": None},
                    "lifecycle": None,
                }
            )
            continue

        input_index, sample = owner
        item = inputs[input_index]
        assert item.snapshot is not None
        status = str(sample["status"])
        counts[status] += 1
        identity = sample.get("identity", {})
        matches = identity.get("matches_manifest")
        if matches is False:
            identity_mismatch += 1
        if not item.manifest_sha_verified:
            manifest_sha_unverified += 1
        routers.append(
            {
                "id": endpoint_object(endpoint),
                "layout_id": metadata.get("layout_id"),
                "instance": deepcopy(metadata.get("instance")),
                "direction": metadata.get("direction"),
                "routing_plane": metadata.get("routing_plane"),
                "link_class": metadata.get("link_class"),
                "capture": {
                    "status": status,
                    "error": sample.get("error"),
                    "snapshot_index": input_index,
                    "torn": bool(sample.get("streams", {}).get("torn", False)),
                    "owner_alive": item.snapshot.get("provenance", {}).get("owner_alive"),
                    "manifest_sha_verified": item.manifest_sha_verified,
                },
                "identity": {
                    "my_mesh_id": identity.get("my_mesh_id"),
                    "my_device_id": identity.get("my_device_id"),
                    "matches": matches,
                },
                "lifecycle": deepcopy(sample.get("lifecycle")),
            }
        )

    coverage = {
        "routers_total": len(routers),
        "captured": len(owners),
        **{
            status: counts[status]
            for status in (
                "ok",
                "unreadable",
                "reset",
                "torn",
                "unknown",
                "unsupported",
                "not_captured",
            )
        },
        "identity_mismatch": identity_mismatch,
        "manifest_sha_unverified": manifest_sha_unverified,
    }
    return {
        "topology": {"meshes": meshes, "links": links},
        "routers": routers,
        "coverage": coverage,
    }
