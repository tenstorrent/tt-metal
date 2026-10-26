# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Decodes one router's part of a snapshot: every area, region, stream reference and element reference the manifest
locates for it, each keyed the way the manifest names it. The values come from the type parser, so nothing here knows
what a value means.
"""

from __future__ import annotations

from dataclasses import dataclass

from tt_metal.fabric.debug.visualizer.common.manifest_lookup import ManifestError, lookup
from tt_metal.fabric.debug.visualizer.common.router_paths import LeafKind, get_router, walk_router
from tt_metal.fabric.debug.visualizer.decode.inputs import Input, InputError
from tt_metal.fabric.debug.visualizer.decode.type_parser import Types, decode_value

# A stream register is one 32-bit word.
REGISTER_BYTES = 4


@dataclass(frozen=True)
class StreamValue:
    """
    A stream reference's register, read before and after the router's L1. Each value is the word's low `bits` bits,
    decoded by the reference's schema. The whole words are kept too, since the bits above the value can hold other
    state. A word capture didn't read is None, and so is its value.
    """

    stream_id: int
    register: str
    bits: int  # from the manifest's stream_registers
    before: object
    after: object
    before_word: int | None
    after_word: int | None


@dataclass(frozen=True)
class ElementValue:
    """An {array, index} reference: its element of the decoded array, or None when capture didn't read the array."""

    array: str  # the array region's path in the router
    index: int
    value: object


@dataclass(frozen=True)
class DecodedRouter:
    """One router's part of a snapshot, decoded. Areas and regions hold the type parser's values."""

    path: str  # such as M0/C0/E1
    error: dict | None  # the snapshot's error, if capture stopped part way through the router
    areas: dict[str, object]  # by name
    regions: dict[str, object]  # by path in the router
    streams: dict[str, StreamValue]  # by the reference's path in the router
    elements: dict[str, ElementValue]  # by the reference's path in the router


def _decode_l1(node, item: dict, snapshot: Input, types: Types, where: str):
    """An area's or region's value, once its snapshot entry is shown to copy its manifest node's address and size."""
    copied = (item.get("address"), item.get("size"))
    expected = (node.get("address"), node.get("size")) if isinstance(node, dict) else None
    if copied != expected or not isinstance(item.get("offset"), int):
        raise InputError(
            f"snapshot {snapshot.json_path} has {where} as {item}, but the manifest has {node}, so the snapshot wasn't "
            "taken with this manifest's layout"
        )
    return decode_value(node, snapshot.bin, item["offset"], types, where)


def _decode_word(schema: str, word: int | None, bits: int, types: Types, where: str):
    """The word's low `bits` bits, decoded by `schema`."""
    if word is None:
        return None
    data = (word & ((1 << bits) - 1)).to_bytes(REGISTER_BYTES, "little")
    return decode_value({"schema": schema, "size": REGISTER_BYTES}, data, 0, types, where)


def _get_register_bits(manifest: dict, where: str) -> dict[str, int]:
    """Each stream register's value width, by name, from the manifest's stream_registers at `where`."""
    widths = {}
    for name, register in lookup(manifest, f"{where}.registers", "object", of="object").items():
        bits = lookup(register, "bits", "integer", where=f"{where}.registers.{name}")
        if not 0 < bits <= REGISTER_BYTES * 8:
            raise ManifestError(f"{where}.registers.{name}.bits is {bits}; a register is {REGISTER_BYTES * 8} bits")
        widths[name] = bits
    return widths


def decode_router(snapshot: Input, types: Types, path: str) -> DecodedRouter:
    """
    The snapshot's router at `path`, such as M0/C0/E1, decoded. If capture stopped part way through the router, what it
    read before then is decoded, and a value it never read is None. Raises InputError when the snapshot and the
    manifest disagree about the router, and ManifestError when the manifest describes something decode can't use.
    """
    entry = snapshot.json["routers"].get(path)
    if entry is None:
        raise InputError(f"snapshot {snapshot.json_path} has no router {path}")
    router, router_where = get_router(snapshot.manifest, path)
    leaves = {leaf.path: leaf for leaf in walk_router(router, router_where)}
    arch_where = f"archs.{lookup(snapshot.manifest, 'run.arch', 'string')}"
    areas_where = f"{arch_where}.areas"
    areas = lookup(snapshot.manifest, areas_where, "object")
    registers_where = f"{arch_where}.stream_registers"
    register_bits = _get_register_bits(snapshot.manifest, registers_where)

    decoded_areas = {
        name: _decode_l1(areas.get(name), item, snapshot, types, f"{areas_where}.{name}")
        for name, item in entry["areas"].items()
    }
    regions = {}
    for region_path, item in entry["regions"].items():
        leaf = leaves.get(region_path)
        if leaf is None or leaf.kind is not LeafKind.REGION:
            raise InputError(
                f"snapshot {snapshot.json_path} has a region {region_path} in {path}, which the manifest's router "
                "doesn't have"
            )
        regions[region_path] = _decode_l1(leaf.node, item, snapshot, types, leaf.where)

    streams, elements = {}, {}
    for leaf in leaves.values():
        if leaf.kind is LeafKind.STREAM:
            stream_id = lookup(leaf.node, "stream_id", "integer", where=leaf.where)
            register = lookup(leaf.node, "register", "string", where=leaf.where)
            schema = lookup(leaf.node, "schema", "string", where=leaf.where)
            bits = register_bits.get(register)
            if bits is None:
                raise ManifestError(
                    f"{leaf.where}.register is {register!r}; {registers_where}.registers names {sorted(register_bits)}"
                )
            read = entry["streams"].get(str(stream_id), {}).get(register, {})
            streams[leaf.path] = StreamValue(
                stream_id,
                register,
                bits,
                before=_decode_word(schema, read.get("before"), bits, types, leaf.where),
                after=_decode_word(schema, read.get("after"), bits, types, leaf.where),
                before_word=read.get("before"),
                after_word=read.get("after"),
            )
        elif leaf.kind is LeafKind.ELEMENT:
            array = lookup(leaf.node, "array", "string", where=leaf.where)
            index = lookup(leaf.node, "index", "integer", where=leaf.where)
            target = leaves.get(array)
            if target is None or target.kind is not LeafKind.REGION:
                raise ManifestError(f"{leaf.where}.array is {array!r}, which names no region of the router")
            count = lookup(target.node, "num_elements", "integer", where=target.where)
            if not 0 <= index < count:
                raise ManifestError(f"{leaf.where}.index is {index}; {array} has {count} elements")
            values = regions.get(array)
            elements[leaf.path] = ElementValue(array, index, None if values is None else values[index])

    return DecodedRouter(path, entry.get("error"), decoded_areas, regions, streams, elements)
