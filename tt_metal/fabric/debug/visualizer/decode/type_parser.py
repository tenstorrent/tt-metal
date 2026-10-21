# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Parses the types the fabric manifest describes, such as its structs, unions and enums, and decodes bytes into values
with them, using only what the manifest says. There isn't any fabric-specific logic here.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional

from tt_metal.fabric.debug.visualizer.common.manifest_lookup import ManifestError, lookup
from tt_metal.fabric.debug.visualizer.decode.routes import read_packed

# An integer schema: u (unsigned) or i (signed), then its width in bits.
_INTEGER = re.compile(r"([ui])(8|16|32|64)")
# A schema that names a type: its kind, then the type's name, which for a class template includes its arguments.
_NAMED = re.compile(r"(struct|union|enum):(.+)")


@dataclass(frozen=True)
class Shape:
    """
    How a manifest node's bytes are laid out: its schema and the keys that size it. The same keys describe an L1
    region, an area and a struct member.
    """

    schema: str
    size: int
    count: int = 0  # num_elements, for an array or a packed table
    element: Shape | None = None  # one element of an array
    bits_per_entry: int = 0  # for a packed table
    tag: str | None = None  # for a union member, the sibling member whose value picks the union's member


@dataclass(frozen=True)
class Member:
    name: str
    offset: int
    shape: Shape


@dataclass(frozen=True)
class StructType:
    size: int
    members: tuple[Member, ...]  # in manifest order, pad members included


@dataclass(frozen=True)
class UnionType:
    size: int
    tag_schema: str  # the enum: schema of the member that tags it
    members: dict[str, Member]  # by the tag value, an enum value name, that picks each
    default_member: Member | None  # the member every other tag value picks, if the union has one


@dataclass(frozen=True)
class EnumType:
    names: dict[int, str]  # each value's name, by value


@dataclass(frozen=True)
class Types:
    """The manifest's struct and union layouts for the run's arch, and its enums, checked."""

    where: str  # archs.<arch>.types
    structs: dict[str, StructType]
    unions: dict[str, UnionType]
    enums: dict[str, EnumType]

    def size_of(self, schema: str) -> int | None:
        """The size of the struct or union that a schema names, or None if it names none."""
        kind, _, name = schema.partition(":")
        found = {"struct": self.structs, "union": self.unions}.get(kind, {}).get(name)
        return None if found is None else found.size


# The size of the struct or union a schema names, or None: Types.size_of, or a table of every type's size while the
# types are still being read.
SizeOf = Callable[[str], Optional[int]]


def _get_count(node: dict, key: str, where: str) -> int:
    value = lookup(node, key, "integer", where=where)
    if value < 1:
        raise ManifestError(f"{where}.{key} is {value}; decode needs at least 1")
    return value


def _check_schema(schema: str, where: str, size_of: SizeOf, enums: dict[str, EnumType]):
    if schema in ("bytes", "pad") or _INTEGER.fullmatch(schema) or size_of(schema) is not None:
        return
    named = _NAMED.fullmatch(schema)
    if named is None:
        raise ManifestError(f"{where}.schema is {schema!r}, which isn't a schema decode knows")
    if named[1] == "enum":
        if named[2] not in enums:
            raise ManifestError(f"{where}.schema is {schema!r}; enums has no {named[2]}")
        return
    raise ManifestError(f"{where}.schema is {schema!r}; types has no {named[1]} {named[2]}")


def _check_element_size(schema: str, size: int, where: str, size_of: SizeOf):
    """An integer, struct or union element is exactly its type's size. An enum, bytes or pad can be any size."""
    integer = _INTEGER.fullmatch(schema)
    expected = int(integer[2]) // 8 if integer else size_of(schema)
    if expected is not None and size != expected:
        raise ManifestError(f"{where}: a {schema} element is {expected} bytes, not {size}")


def _get_shape(node: dict, where: str, size_of: SizeOf, enums: dict[str, EnumType]) -> Shape:
    schema = lookup(node, "schema", "string", where=where)
    _check_schema(schema, where, size_of, enums)
    size = _get_count(node, "size", where)
    is_union = schema.startswith("union:")
    tag = lookup(node, "tag", "string", where=where) if "tag" in node else None
    if is_union and tag is None:
        raise ManifestError(f"{where} is a {schema} with no tag, the member whose value picks the union's member")
    if tag is not None and not is_union:
        raise ManifestError(f"{where} has a tag, but only a union: member has one")

    if "num_elements" not in node:
        if "size_per_element" in node or "bits_per_entry" in node:
            raise ManifestError(f"{where} has size_per_element or bits_per_entry but no num_elements")
        _check_element_size(schema, size, where, size_of)
        return Shape(schema, size, tag=tag)

    count = _get_count(node, "num_elements", where)
    if is_union:
        raise ManifestError(f"{where} is an array of {schema}, which decode can't tag")
    if "bits_per_entry" in node:
        if "size_per_element" in node:
            raise ManifestError(f"{where} has both size_per_element and bits_per_entry")
        if not schema.startswith("enum:"):
            raise ManifestError(f"{where} is a packed table of {schema}; a packed table holds an enum:")
        bits_per_entry = _get_count(node, "bits_per_entry", where)
        if count * bits_per_entry > 8 * size:
            raise ManifestError(f"{where}: {count} entries of {bits_per_entry} bits don't fit in {size} bytes")
        return Shape(schema, size, count=count, bits_per_entry=bits_per_entry)

    element_size = _get_count(node, "size_per_element", where)
    if count * element_size != size:
        raise ManifestError(f"{where}: {count} elements of {element_size} bytes aren't its size, {size}")
    _check_element_size(schema, element_size, where, size_of)
    return Shape(schema, size, count=count, element=Shape(schema, element_size))


def _get_member(node: dict, where: str, size_of: SizeOf, enums: dict[str, EnumType]) -> Member:
    name = lookup(node, "name", "string", where=where)
    offset = lookup(node, "offset", "integer", where=where)
    return Member(name, offset, _get_shape(node, where, size_of, enums))


def _get_struct(described: dict, size: int, where: str, size_of: SizeOf, enums: dict[str, EnumType]) -> StructType:
    members = []
    names = set()
    for index, node in enumerate(lookup(described, "members", "array", of="object", where=where)):
        member_where = f"{where}.members[{index}]"
        member = _get_member(node, member_where, size_of, enums)
        if member.offset < 0 or member.offset + member.shape.size > size:
            raise ManifestError(
                f"{member_where} is bytes {member.offset} to {member.offset + member.shape.size}, outside the "
                f"struct's {size}"
            )
        # Pad members can share a name: the HAL's layouts call every gap "pad".
        if member.shape.schema != "pad":
            if member.name in names:
                raise ManifestError(f"{member_where}.name is {member.name!r}, which an earlier member has")
            names.add(member.name)
        members.append(member)
    return StructType(size, tuple(members))


def _get_union(described: dict, size: int, where: str, size_of: SizeOf, enums: dict[str, EnumType]) -> UnionType:
    tag_schema = lookup(described, "tag_schema", "string", where=where)
    tag_enum = _NAMED.fullmatch(tag_schema)
    if tag_enum is None or tag_enum[1] != "enum" or tag_enum[2] not in enums:
        raise ManifestError(f"{where}.tag_schema is {tag_schema!r}, which names no enum in enums")
    tag_names = set(enums[tag_enum[2]].names.values())

    by_name: dict[str, Member] = {}
    by_tag_value: dict[str, Member] = {}
    for index, node in enumerate(lookup(described, "members", "array", of="object", where=where)):
        member_where = f"{where}.members[{index}]"
        member = _get_member(node, member_where, size_of, enums)
        if member.offset != 0 or member.shape.size > size:
            raise ManifestError(
                f"{member_where} is bytes {member.offset} to {member.offset + member.shape.size}; a union member "
                f"starts at 0 and fits in the union's {size}"
            )
        if member.shape.tag is not None:
            raise ManifestError(f"{member_where} is a tagged union, which decode can't nest in a union")
        if member.name in by_name:
            raise ManifestError(f"{member_where}.name is {member.name!r}, which an earlier member has")
        by_name[member.name] = member
        if "tag_value" in node:
            tag_value = lookup(node, "tag_value", "string", where=member_where)
            if tag_value not in tag_names:
                raise ManifestError(f"{member_where}.tag_value is {tag_value!r}, which {tag_schema} doesn't name")
            if tag_value in by_tag_value:
                raise ManifestError(f"{member_where}.tag_value is {tag_value!r}, which an earlier member has")
            by_tag_value[tag_value] = member

    default_member = None
    if "default_member" in described:
        default_name = lookup(described, "default_member", "string", where=where)
        if default_name not in by_name:
            raise ManifestError(f"{where}.default_member is {default_name!r}, which names no member")
        default_member = by_name[default_name]
    return UnionType(size, tag_schema, by_tag_value, default_member)


def _check_tags(struct: StructType, where: str, unions: dict[str, UnionType]):
    """Each tagged union member's tag is a sibling member holding one value of the union's tag enum."""
    siblings = {member.name: member for member in struct.members if member.shape.schema != "pad"}
    for index, member in enumerate(struct.members):
        if member.shape.tag is None:
            continue
        union = unions[member.shape.schema[len("union:") :]]
        tag = siblings.get(member.shape.tag)
        if tag is None or tag.shape.schema != union.tag_schema or tag.shape.count:
            raise ManifestError(
                f"{where}.members[{index}].tag is {member.shape.tag!r}; it must name a member holding one "
                f"{union.tag_schema}"
            )


def get_types_from_manifest(manifest: dict) -> Types:
    """
    Reads the struct and union layouts for the manifest's run.arch, and its enums, and checks every one before any
    bytes are read. Raises ManifestError naming the first value decode can't use.
    """
    arch = lookup(manifest, "run.arch", "string")
    where = f"archs.{arch}.types"
    described = lookup(manifest, where, "object", of="object")

    enums: dict[str, EnumType] = {}
    for enum in lookup(manifest, "enums", "object", of="object"):
        names: dict[int, str] = {}
        for name, value in lookup(manifest, f"enums.{enum}.values", "object", of="integer").items():
            if value in names:
                raise ManifestError(f"enums.{enum}.values gives {value} to both {names[value]!r} and {name!r}")
            names[value] = name
        enums[enum] = EnumType(names)

    # Every type's size first, by its schema, because a member can name a type that comes later.
    declared_sizes = {}
    for name, type_ in described.items():
        kind = lookup(type_, "kind", "string", where=f"{where}.{name}") if "kind" in type_ else "struct"
        if kind not in ("struct", "union"):
            raise ManifestError(f"{where}.{name}.kind is {kind!r}; decode knows a struct or a union")
        declared_sizes[f"{kind}:{name}"] = _get_count(type_, "size", f"{where}.{name}")

    structs, unions = {}, {}
    for name, type_ in described.items():
        if f"union:{name}" in declared_sizes:
            size = declared_sizes[f"union:{name}"]
            unions[name] = _get_union(type_, size, f"{where}.{name}", declared_sizes.get, enums)
        else:
            size = declared_sizes[f"struct:{name}"]
            structs[name] = _get_struct(type_, size, f"{where}.{name}", declared_sizes.get, enums)
    for name, struct in structs.items():
        _check_tags(struct, f"{where}.{name}", unions)
    return Types(where, structs, unions, enums)


def get_shape(node: dict, types: Types, where: str) -> Shape:
    """The checked shape of a manifest node with an L1 region's keys, such as a region or an area at `where`."""
    return _get_shape(node, where, types.size_of, types.enums)


def _check_in_bounds(raw: bytes, offset: int, size: int, where: str):
    if offset < 0 or offset + size > len(raw):
        raise ValueError(f"{where} is bytes {offset} to {offset + size}, past the {len(raw)} bytes given")


def _decode_union(member: Member, tag_name: str | None, raw: bytes, offset: int, types: Types):
    union = types.unions[member.shape.schema[len("union:") :]]
    picked = union.members.get(tag_name, union.default_member)
    if picked is None:
        return {"member": None, "bytes": {"offset": offset, "size": union.size}}
    return {"member": picked.name, "value": _decode(picked.shape, raw, offset, types)}


def _decode_struct(struct: StructType, raw: bytes, offset: int, types: Types) -> dict:
    # Every key first, in member order. Tagged unions are filled in last, since a tag can come after its union.
    values = {member.name: None for member in struct.members if member.shape.schema != "pad"}
    for member in struct.members:
        if member.shape.schema != "pad" and not member.shape.tag:
            values[member.name] = _decode(member.shape, raw, offset + member.offset, types)
    for member in struct.members:
        if member.shape.tag:
            tag_name = values[member.shape.tag]["name"]
            values[member.name] = _decode_union(member, tag_name, raw, offset + member.offset, types)
    return values


def _decode(shape: Shape, raw: bytes, offset: int, types: Types):
    if shape.bits_per_entry:
        names = types.enums[shape.schema[len("enum:") :]].names
        entries = read_packed(raw[offset : offset + shape.size], shape.bits_per_entry, shape.count)
        return [{"raw": entry, "name": names.get(entry)} for entry in entries]
    if shape.element is not None:
        return [_decode(shape.element, raw, offset + index * shape.element.size, types) for index in range(shape.count)]
    if shape.schema == "bytes":
        return {"bytes": {"offset": offset, "size": shape.size}}
    if shape.schema == "pad":
        return None
    data = raw[offset : offset + shape.size]
    integer = _INTEGER.fullmatch(shape.schema)
    if integer:
        return int.from_bytes(data, "little", signed=integer[1] == "i")
    kind, name = _NAMED.fullmatch(shape.schema).groups()
    if kind == "enum":
        value = int.from_bytes(data, "little")
        return {"raw": value, "name": types.enums[name].names.get(value)}
    return _decode_struct(types.structs[name], raw, offset, types)


def decode_value(node: dict, raw: bytes, offset: int, types: Types, where: str):
    """
    The value the manifest node at `where` describes, decoded from raw[offset:]. `raw` is usually a snapshot's whole
    .bin, so a bytes value's offset is an offset into the .bin. Raises ManifestError when the node can't be decoded,
    and ValueError when its bytes go past the end of `raw`.
    """
    shape = get_shape(node, types, where)
    if shape.tag is not None:
        raise ManifestError(f"{where} is a tagged union, which decodes only inside its struct")
    _check_in_bounds(raw, offset, shape.size, where)
    return _decode(shape, raw, offset, types)


def decode_struct(name: str, raw: bytes, offset: int, types: Types) -> dict:
    """Struct `name` decoded from raw[offset:], as {member: value} in member order, without its pad members."""
    struct = types.structs.get(name)
    if struct is None:
        raise ManifestError(f"{types.where} has no struct {name!r}")
    _check_in_bounds(raw, offset, struct.size, f"{types.where}.{name}")
    return _decode_struct(struct, raw, offset, types)
