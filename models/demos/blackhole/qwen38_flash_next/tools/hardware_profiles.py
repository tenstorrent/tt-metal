"""Hardware profiles: which four Blackhole chips make the model's 1x4 mesh, and how the host wires them.

A profile pins the KMD device nodes, the route derivation (``physical_route``), the derived route to expect and the
system mesh shape ttnn discovers.  The public table holds the QuietBox, the 4x p150 line and the QuietBox 2
profiles; a private table (a module named by ``QWEN38_HARDWARE_PROFILE_TABLE`` exposing ``HARDWARE_PROFILES``)
extends it for other hosts.
"""
from __future__ import annotations

import importlib
import os
import re
import socket
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import ttnn

PROFILE_TABLE_VARIABLE = "QWEN38_HARDWARE_PROFILE_TABLE"
SYMMETRIC_ALLOCATOR_SOURCE = "ttnn_mesh_allocator_symmetric_per_device"


class HardwareProfileError(RuntimeError):
    pass


@dataclass(frozen=True)
class ResidentHardwareProfile:
    """One four-chip mesh.

    ``partition`` names the lane on the host (``"qb"``: the whole box; ``"a"``/``"b"``: four chips of a larger
    host, ``"0"``/``"1"`` the same on a QuietBox 2 class host).  ``ethernet_graph`` selects the route derivation:
    ``"line"`` is ``physical_route.derive_canonical_line_route``, ``"ring"`` is ``derive_fabric_line_route`` (chips in
    an ethernet ring, opened as a 1x4 LINE through a mesh graph descriptor).  ``route`` / ``route_nodes`` pin the
    derived order; ``None`` means the route is derived at start (``qwen38_chat_session.resolve_route``) and
    recorded, not pinned.  ``boards`` (serial per node), ``bdfs`` and ``numa_node`` are
    optional identity checks (empty: not checked).  ``required_locks`` are exclusive flocks the launcher must hold
    (fd -> path; empty: none).  ``lan_serving`` lets the chat server bind a non-loopback address without
    ``--allow-lan``.  ``mesh_graph_descriptor`` names the descriptor file next to this module the launcher exports as
    ``TT_MESH_GRAPH_DESC_PATH`` (``None``: ttnn's auto-discovery).
    """

    host: str
    partition: str
    visible_devices: str
    device_nodes: tuple[int, int, int, int]
    numa_node: int | None
    ethernet_graph: str
    route: tuple[int, int, int, int] | None
    route_nodes: tuple[int, int, int, int] | None
    system_mesh_local_shape: tuple[int, int]
    boards: Mapping[int, str] = field(default_factory=dict)
    bdfs: Mapping[int, str] = field(default_factory=dict)
    required_locks: Mapping[int, str] = field(default_factory=dict)
    lan_serving: bool = False
    mesh_graph_descriptor: str | None = None
    tested: bool = True

    @property
    def lane(self) -> str:
        return f"{self.host} partition-{self.partition.upper()}"

    def with_device_nodes(
        self, nodes: tuple[int, int, int, int], *, present: Sequence[int] | None = None
    ) -> "ResidentHardwareProfile":
        """The same profile on other KMD nodes (four chips of a larger host)."""

        if len(nodes) != 4 or len(set(nodes)) != 4 or any(type(node) is not int or node < 0 for node in nodes):
            raise HardwareProfileError(f"device nodes must be four distinct non-negative ints, got {nodes!r}")
        # ``TT_VISIBLE_DEVICES`` takes UMD's device indices, 0..N-1 over the nodes present under /dev/tenstorrent (the cluster
        # descriptor keeps naming the chips by their node numbers).  Four nodes that are all the host presents (a container
        # holding four nodes of a larger host) are the indices 0, 1, 2, 3 whatever their names; on a host that presents more,
        # the node numbers (the hosts these profiles run on number their nodes 0..N-1 without gaps, so the two agree there).
        present_nodes = tuple(sorted(present)) if present is not None else present_device_nodes()
        visible = tuple(range(len(nodes))) if present_nodes == tuple(sorted(nodes)) else nodes
        return replace(self, device_nodes=nodes, visible_devices=",".join(str(value) for value in visible))

    @property
    def default_lane(self) -> bool:
        """The lane a host runs when ``TT_VISIBLE_DEVICES`` names none: the whole box, partition B, instance 0."""

        return self.partition in ("qb", "b", "0")


DEVICE_ROOT = Path("/dev/tenstorrent")


def present_device_nodes(device_root: Path = DEVICE_ROOT) -> tuple[int, ...]:
    """The ``/dev/tenstorrent/<N>`` nodes this process can see, sorted; empty when the directory is absent."""

    try:
        return tuple(sorted(int(entry.name) for entry in device_root.iterdir() if entry.name.isdigit()))
    except OSError:
        return ()


# QuietBox: four p150b devices in an ethernet ring. The current fabric solver's
# validated line order is derived and recorded at startup; logical and KMD IDs
# vary between boxes and runtime versions. Never pin a public SKU to one host's IDs.
QUIETBOX = ResidentHardwareProfile(
    host="tt-quietbox",
    partition="qb",
    visible_devices="0,1,2,3",
    device_nodes=(0, 1, 2, 3),
    numa_node=None,
    ethernet_graph="ring",
    route=None,
    route_nodes=None,
    system_mesh_local_shape=(1, 4),
    lan_serving=True,
    mesh_graph_descriptor="qb_p150_x4_1x4_line_mesh_graph_descriptor.textproto",
)


# p150 line: 4x p150 in one host, KMD nodes 0-3, the chips in an ethernet line opened as the 1x4 with
# ttnn's default mesh descriptor (the same 1x4 mesh); the route is
# derived at start and recorded.  ``--device-nodes`` moves it to another four nodes.
P150_LINE = ResidentHardwareProfile(
    host="p150-line",
    partition="line",
    visible_devices="0,1,2,3",
    device_nodes=(0, 1, 2, 3),
    numa_node=None,
    ethernet_graph="line",
    route=None,
    route_nodes=None,
    system_mesh_local_shape=(4, 1),
    lan_serving=True,
)


def _quietbox_2_instance(instance: int) -> ResidentHardwareProfile:
    """UNTESTED: a p300-based box.  Each p300 card is two Blackhole dies (two KMD nodes) joined on the card; the
    QuietBox 2 (2x p300c = nodes 0-3) has one ring over the on-card links and the two Warp400 links, so it is one
    instance.  A four-card host (8 dies) is two instances of four consecutive nodes.  ttnn classifies a p300
    cluster that is not exactly 2 or 4 dies as CUSTOM and refuses to open without a mesh graph descriptor, so the
    launcher always exports the descriptor.  The route is unpinned: derived at start from the order in which the
    fabric embedded the LINE descriptor onto the ring (``physical_route.derive_fabric_line_route``; a QuietBox 2 on
    2026-09-18: (1, 0, 3, 2), not the ring walk (0, 1, 2, 3)) and recorded."""

    first = 4 * instance
    nodes = (first, first + 1, first + 2, first + 3)
    return ResidentHardwareProfile(
        host="tt-quietbox-2",
        partition=str(instance),
        visible_devices=",".join(str(node) for node in nodes),
        device_nodes=nodes,
        numa_node=None,
        ethernet_graph="ring",
        route=None,
        route_nodes=None,
        system_mesh_local_shape=(1, 4),
        lan_serving=True,
        mesh_graph_descriptor="qb2_p300_1x4_line_mesh_graph_descriptor.textproto",
        tested=False,
    )


HARDWARE_PROFILES: dict[str, ResidentHardwareProfile] = {
    "tt-quietbox": QUIETBOX,
    "p150-line": P150_LINE,
    "tt-quietbox-2": _quietbox_2_instance(0),
    "tt-quietbox-2-instance-1": _quietbox_2_instance(1),
}


def hardware_profile_table() -> dict[str, ResidentHardwareProfile]:
    """The public profiles plus the table of the module named by ``QWEN38_HARDWARE_PROFILE_TABLE`` (if set)."""

    table = dict(HARDWARE_PROFILES)
    module_name = os.environ.get(PROFILE_TABLE_VARIABLE)
    if module_name:
        table.update(importlib.import_module(module_name).HARDWARE_PROFILES)
    return table


def resolve_hardware_profile(
    name: str | None, *, table: Mapping[str, ResidentHardwareProfile] | None = None
) -> ResidentHardwareProfile:
    """By name, or (``None``) the profile of this host whose visible devices are ``TT_VISIBLE_DEVICES``.

    A host with several profiles (two lanes) takes the one ``TT_VISIBLE_DEVICES`` names, its default lane when
    unset.  A named public profile is a box type, not a host, and resolves anywhere (a QuietBox 2 shipped with the
    hostname ``tt-quietbox``).  A private table's entry is one host's lane: on a host the table knows, a name from
    another host refuses, so a launcher cannot open another host's lane by mistake; a host the table does not know
    takes any named profile.
    """

    profiles = hardware_profile_table() if table is None else table
    host = socket.gethostname().split(".", 1)[0]
    if name is not None:
        if name not in profiles:
            raise HardwareProfileError(f"unknown hardware profile {name!r}, expected one of {sorted(profiles)}")
        profile = profiles[name]
        if (
            HARDWARE_PROFILES.get(name) != profile
            and profile.host != host
            and any(other.host == host for other in profiles.values())
        ):
            raise HardwareProfileError(f"hardware profile {name!r} belongs to {profile.host}, not {host}")
        return profile
    candidates = {key: profile for key, profile in profiles.items() if profile.host == host}
    if not candidates:
        hosts = sorted({profile.host for profile in profiles.values()})
        raise HardwareProfileError(f"no hardware profile for host {host!r} (profiles exist for {hosts}); name one")
    visible = os.environ.get("TT_VISIBLE_DEVICES")
    if visible is None:
        matching = [profile for profile in candidates.values() if profile.default_lane]
    else:
        matching = [profile for profile in candidates.values() if profile.visible_devices == visible]
    if len(matching) != 1:
        raise HardwareProfileError(
            f"TT_VISIBLE_DEVICES {visible!r} selects {len(matching)} of {host}'s profiles "
            f"{sorted((key, profile.visible_devices) for key, profile in candidates.items())}, expected one"
        )
    return matching[0]


def mesh_graph_descriptor_path(profile: ResidentHardwareProfile) -> Path | None:
    return None if profile.mesh_graph_descriptor is None else Path(__file__).with_name(profile.mesh_graph_descriptor)


def verify_inherited_locks(profile: ResidentHardwareProfile) -> list[dict[str, Any]]:
    """Every lock the profile requires is held exclusively on the expected inherited fd."""

    proof = []
    for descriptor, expected in profile.required_locks.items():
        try:
            actual = os.readlink(f"/proc/self/fd/{descriptor}")
            fdinfo = Path(f"/proc/self/fdinfo/{descriptor}").read_text(encoding="ascii")
        except OSError as error:
            raise HardwareProfileError(f"cannot inspect required lock fd {descriptor}: {error}") from error
        if actual != expected or re.search(r"^lock:.*FLOCK.*WRITE(?:\s|$)", fdinfo, re.MULTILINE) is None:
            raise HardwareProfileError(
                f"fd {descriptor} is not the required exclusive lock: {actual!r} != {expected!r}"
            )
        proof.append({"fd": descriptor, "path": actual, "runner_pid": os.getpid()})
    return proof


def live_mapping(
    mesh: Any, chips_with_mmio: Mapping[int, int], profile: ResidentHardwareProfile
) -> list[dict[str, Any]]:
    """The open mesh's coordinates against the profile: physical id, KMD node, and the optional board / BDF / NUMA
    identities; the node order must be the pinned route."""

    if profile.route is None or profile.route_nodes is None:
        raise HardwareProfileError(f"profile {profile.lane} has no route yet (resolve it before the mesh opens)")
    mapping = []
    for column, physical_id in enumerate(profile.route):
        coordinate = ttnn.MeshCoordinate(0, column)
        actual_id = int(mesh.get_device_id(coordinate))
        node = chips_with_mmio.get(actual_id)
        if actual_id != physical_id or node is None:
            raise HardwareProfileError(f"mesh coordinate {(0, column)} maps to logical={actual_id} node={node}")
        sysfs = Path(f"/sys/class/tenstorrent/tenstorrent!{node}/device").resolve(strict=True)
        board = profile.boards.get(node)
        if board is not None:
            board_link = Path(f"/dev/tenstorrent/by-id/blackhole-{board}")
            if not board_link.is_symlink() or int(board_link.resolve(strict=True).name) != node:
                raise HardwareProfileError(f"board identity differs for node {node}: {board_link}")
        fabric = mesh.get_fabric_node_id(coordinate)
        record = {
            "mesh_coordinate": [0, column],
            "ttnn_physical_id": actual_id,
            "device_node": node,
            "bdf": sysfs.name.lower(),
            "numa_node": int((sysfs / "numa_node").read_text(encoding="ascii").strip()),
            "board_id": board,
            "fabric_mesh_id": int(fabric.mesh_id),
            "fabric_chip_id": int(fabric.chip_id),
        }
        expected_bdf = profile.bdfs.get(node)
        if (expected_bdf is not None and record["bdf"] != expected_bdf) or (
            profile.numa_node is not None and record["numa_node"] != profile.numa_node
        ):
            raise HardwareProfileError(
                f"{profile.lane} platform identity differs: {record} vs bdf={expected_bdf} numa_node={profile.numa_node}"
            )
        mapping.append(record)
    if tuple(item["device_node"] for item in mapping) != profile.route_nodes:
        raise HardwareProfileError(f"canonical route changed: {mapping} vs nodes {profile.route_nodes}")
    return mapping


def symmetric_mesh_dram_memory(mesh: Any, route: tuple[int, int, int, int]) -> dict[str, Any]:
    """One MeshDevice allocator observation that applies to every card.

    MeshDevice owns one virtual allocator cloned from the reference physical device; mesh buffers reserve the same
    address range on every local device, so this view is the per-device allocation state.  Tensor shards report the
    MeshDevice as their owner and cannot serve as an allocator probe.
    """

    device_ids = tuple(int(value) for value in mesh.get_device_ids())
    if len(device_ids) != 4 or set(device_ids) != set(route):
        raise HardwareProfileError(f"mesh allocator does not cover the canonical physical IDs: {device_ids}")
    for column, physical_id in enumerate(route):
        if int(mesh.get_device_id(ttnn.MeshCoordinate(0, column))) != physical_id:
            raise HardwareProfileError("mesh allocator physical order differs from the canonical route")
    view = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
    banks = int(view.num_banks)
    total = int(view.total_bytes_per_bank)
    allocated = int(view.total_bytes_allocated_per_bank)
    free = int(view.total_bytes_free_per_bank)
    largest = int(view.largest_contiguous_bytes_free_per_bank)
    if banks <= 0 or total <= 0 or min(allocated, free, largest) < 0 or allocated + free != total or largest > free:
        raise HardwareProfileError("symmetric mesh allocator DRAM view is inconsistent")
    return {
        "source": SYMMETRIC_ALLOCATOR_SOURCE,
        "applies_to_physical_ids": list(route),
        "num_banks": banks,
        "total_bytes_per_bank": total,
        "allocated_bytes_per_bank": allocated,
        "free_bytes_per_bank": free,
        "largest_contiguous_bytes_free_per_bank": largest,
        "aggregate_total_bytes": banks * total,
        "aggregate_allocated_bytes": banks * allocated,
        "aggregate_free_bytes": banks * free,
    }
