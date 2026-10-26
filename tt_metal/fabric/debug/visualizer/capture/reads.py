# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Works out what capture reads on each router core, using only the fabric manifest.

Each read is either L1 bytes, `size` bytes at `address` (an L1Read), or one 32-bit register word (a RegisterRead).
"""

from __future__ import annotations

from dataclasses import dataclass

from tt_metal.fabric.debug.visualizer.common.manifest_lookup import ManifestError, lookup
from tt_metal.fabric.debug.visualizer.common.router_paths import LeafKind, get_router_path, walk_router

# A register, core or stream, is one 32-bit word.
REGISTER_BYTES = 4


@dataclass(frozen=True)
class L1Read:
    """`size` bytes of L1 at `address` on a router core."""

    address: int
    size: int


@dataclass(frozen=True)
class RegisterRead:
    """One 32-bit register at `address` on a router core."""

    address: int


@dataclass(frozen=True)
class RouterPlan:
    """What capture reads on one router core beyond the areas and core registers, and how to find the core."""

    path: str  # the router's manifest path, such as "M0/C0/E1", as link.peer writes it
    asic_id: int  # its chip's asic_id
    eth_chan: int
    virtual_core: tuple[int, int]
    logical_core: tuple[int, int]
    regions: dict[str, L1Read]  # by path inside the router, in manifest order
    streams: dict[tuple[int, str], RegisterRead]  # by (stream_id, register), each pair once, by id and then register


@dataclass(frozen=True)
class ReadPlan:
    """What capture reads on each local chip's router cores. Every core has the same areas and core registers."""

    arch: str  # run.arch
    core_registers: dict[str, RegisterRead]  # by name, in array order
    areas: dict[str, L1Read]  # the non-null areas, by name, in manifest order
    heartbeat: RegisterRead | None  # the heartbeat area as one word, which liveness rounds sample, or None if it's null
    routers: tuple[RouterPlan, ...]  # in manifest order


def _l1_read(node: dict, where: str) -> L1Read:
    return L1Read(lookup(node, "address", "integer", where=where), lookup(node, "size", "integer", where=where))


@dataclass(frozen=True)
class _StreamRegisters:
    where: str  # archs.<arch>.stream_registers
    base: int
    stride: int
    count: int
    # Each register's byte offset in a stream's block, by name (ie. {"buf_space_available": 256, "remote_src": 0})
    offsets: dict[str, int]

    def read(self, stream_id: int, register: str, where: str) -> RegisterRead:
        """The register a stream reference at `where` names."""
        # ttexalens's safe mode reads a past-the-end stream id's address without complaint.
        if not 0 <= stream_id < self.count:
            raise ManifestError(f"{where}.stream_id is {stream_id}; {self.where}.count is {self.count}")
        if register not in self.offsets:
            raise ManifestError(
                f"{where}.register is {register!r}; {self.where}.registers names {sorted(self.offsets)}"
            )
        return RegisterRead(self.base + stream_id * self.stride + self.offsets[register])


def _get_stream_registers(manifest: dict, where: str) -> _StreamRegisters:
    registers = lookup(manifest, f"{where}.registers", "object", of="object")
    return _StreamRegisters(
        where=where,
        base=lookup(manifest, f"{where}.base", "integer"),
        stride=lookup(manifest, f"{where}.stride", "integer"),
        count=lookup(manifest, f"{where}.count", "integer"),
        offsets={
            name: lookup(register, "offset", "integer", where=f"{where}.registers.{name}")
            for name, register in registers.items()
        },
    )


def _get_core(router: dict, key: str, where: str) -> tuple[int, int]:
    core = lookup(router, f"identity.{key}", "array", of="integer", where=where)
    if len(core) != 2:
        raise ManifestError(f"{where}.identity.{key} is {core}; capture needs two coordinates")
    return (core[0], core[1])


def _get_asic_id(chip: dict, where: str) -> int:
    asic_id = lookup(chip, "asic_id", "string", where=where)
    try:
        return int(asic_id, 16)
    except ValueError:
        raise ManifestError(f"{where}.asic_id is {asic_id!r}; capture needs a hex number") from None


def _get_router_plan(
    router: dict, path: str, where: str, asic_id: int, stream_registers: _StreamRegisters
) -> RouterPlan:
    regions: dict[str, L1Read] = {}
    streams: dict[tuple[int, str], RegisterRead] = {}
    for leaf in walk_router(router, where):
        if leaf.kind is LeafKind.REGION:
            regions[leaf.path] = _l1_read(leaf.node, leaf.where)
        elif leaf.kind is LeafKind.STREAM:
            stream_id = lookup(leaf.node, "stream_id", "integer", where=leaf.where)
            register = lookup(leaf.node, "register", "string", where=leaf.where)
            streams[stream_id, register] = stream_registers.read(stream_id, register, leaf.where)
        # An element reference points into a region, which is read whole.
    return RouterPlan(
        path=path,
        asic_id=asic_id,
        eth_chan=lookup(router, "identity.eth_chan", "integer", where=where),
        virtual_core=_get_core(router, "virtual_core", where),
        logical_core=_get_core(router, "logical_core", where),
        regions=regions,
        streams={key: streams[key] for key in sorted(streams)},
    )


def get_read_plan(manifest: dict) -> ReadPlan:
    """What capture reads on the router cores of a parsed fabric manifest's local chips."""
    kind = lookup(manifest, "kind", "string")
    if kind != "fabric_manifest":
        raise ManifestError(f"kind is {kind!r}; capture needs a fabric_manifest")
    arch = lookup(manifest, "run.arch", "string")
    arch_where = f"archs.{arch}"
    lookup(manifest, arch_where, "object")

    # Collect all the core registers the manifest names
    core_registers: dict[str, RegisterRead] = {}
    for index, register in enumerate(lookup(manifest, f"{arch_where}.core_registers", "array", of="object")):
        register_where = f"{arch_where}.core_registers[{index}]"
        name = lookup(register, "name", "string", where=register_where)
        if name in core_registers:
            raise ManifestError(f"{register_where}.name is {name!r}, which an earlier core register has")
        address = lookup(register, "address", "integer", where=register_where)
        size = lookup(register, "size", "integer", where=register_where)
        if size != REGISTER_BYTES:
            raise ManifestError(
                f"{register_where}.size is {size}; capture reads a register as one {REGISTER_BYTES}-byte word"
            )
        core_registers[name] = RegisterRead(address)

    # Collect all the L1 areas the manifest names
    areas = {
        name: _l1_read(area, f"{arch_where}.areas.{name}")
        for name, area in lookup(manifest, f"{arch_where}.areas", "object", of=("object", "null")).items()
        if area is not None
    }
    heartbeat = areas.get("heartbeat")
    if heartbeat is not None and heartbeat.size != REGISTER_BYTES:
        raise ManifestError(
            f"{arch_where}.areas.heartbeat.size is {heartbeat.size}; "
            f"capture samples it as one {REGISTER_BYTES}-byte word"
        )

    # Collect all the stream registers the manifest names
    stream_registers = _get_stream_registers(manifest, f"{arch_where}.stream_registers")
    routers = []
    for mesh_key in lookup(manifest, "meshes", "object", of="object"):
        chips_where = f"meshes.{mesh_key}.chips"
        for chip_key, chip in lookup(manifest, chips_where, "object", of="object").items():
            chip_where = f"{chips_where}.{chip_key}"
            if not lookup(chip, "is_local", "boolean", where=chip_where):
                continue
            asic_id = _get_asic_id(chip, chip_where)
            for router_key, router in lookup(chip, "routers", "object", of="object", where=chip_where).items():
                routers.append(
                    _get_router_plan(
                        router,
                        get_router_path(mesh_key, chip_key, router_key),
                        f"{chip_where}.routers.{router_key}",
                        asic_id,
                        stream_registers,
                    )
                )

    return ReadPlan(
        arch=arch,
        core_registers=core_registers,
        areas=areas,
        heartbeat=None if heartbeat is None else RegisterRead(heartbeat.address),
        routers=tuple(routers),
    )
