# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Generic helpers for navigating a parsed DWARF tree.

These utilities are deliberately independent of any LLK-specific knowledge:
they resolve DIE names across ``abstract_origin``/``specification`` links,
build a value->name table for every enumeration, and render type DIEs as
readable strings. The LLK-specific logic lives in :mod:`extractor`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from elftools.dwarf.die import DIE
from elftools.dwarf.dwarfinfo import DWARFInfo

# Reference forms whose value is an offset relative to the containing CU.
_CU_RELATIVE_REF_FORMS = {
    "DW_FORM_ref1",
    "DW_FORM_ref2",
    "DW_FORM_ref4",
    "DW_FORM_ref8",
    "DW_FORM_ref_udata",
}
_NAME_LINK_ATTRS = ("DW_AT_abstract_origin", "DW_AT_specification")
_TRANSPARENT_TYPE_TAGS = ("DW_TAG_const_type", "DW_TAG_volatile_type", "DW_TAG_typedef")


def attr_str(die: DIE, name: str) -> str | None:
    """Return a string-valued attribute, decoded, or ``None`` if absent."""
    attr = die.attributes.get(name)
    if attr is None:
        return None
    value = attr.value
    return value.decode("utf-8", "replace") if isinstance(value, bytes) else str(value)


def follow_ref(die: DIE, attr_name: str) -> DIE | None:
    """Resolve a reference attribute to its target DIE (handles cross-CU refs)."""
    if attr_name not in die.attributes:
        return None
    try:
        return die.get_DIE_from_attribute(attr_name)
    except Exception:  # noqa: BLE001 - malformed/unsupported refs must not abort the walk
        return None


class EnumTable:
    """Maps every enumeration type to its ``{value: enumerator_name}``.

    Indexed both by DIE offset and by type name. The name index is needed
    because an enum can appear as an empty *declaration* in the CU that uses it
    while its enumerators are *defined* in another CU; resolving by name unifies
    the two.
    """

    def __init__(self) -> None:
        self._by_offset: dict[int, dict[int, str]] = {}
        self._by_name: dict[str, dict[int, str]] = {}

    def add(self, enum_die: DIE) -> None:
        members: dict[int, str] = {}
        for child in enum_die.iter_children():
            if child.tag != "DW_TAG_enumerator":
                continue
            const = child.attributes.get("DW_AT_const_value")
            name = attr_str(child, "DW_AT_name")
            if const is not None and name is not None:
                members[const.value] = name
        self._by_offset[enum_die.offset] = members
        if members:
            type_name = attr_str(enum_die, "DW_AT_name")
            if type_name:
                # Prefer the populated definition over an empty declaration.
                self._by_name.setdefault(type_name, {}).update(members)

    def name_for(self, enum_offset: int, value: int, type_name: str | None = None) -> str | None:
        resolved = self._by_offset.get(enum_offset, {}).get(value)
        if resolved is None and type_name is not None:
            resolved = self._by_name.get(type_name, {}).get(value)
        return resolved


def collect_enum_value_names(dwarf: DWARFInfo, type_name: str) -> dict[int, str]:
    """Collect ``{value: enumerator_name}`` for a named enum across all CUs.

    Used to decode integer codes (e.g. ``tt::DataFormat`` / tensix ``DataFormat``)
    directly from the ELF's own debug info instead of a hand-maintained table, so
    the mapping stays correct per architecture. First definition wins per value.
    """
    result: dict[int, str] = {}
    for cu in dwarf.iter_CUs():
        for die in cu.iter_DIEs():
            if die.tag != "DW_TAG_enumeration_type" or attr_str(die, "DW_AT_name") != type_name:
                continue
            for child in die.iter_children():
                if child.tag != "DW_TAG_enumerator":
                    continue
                const = child.attributes.get("DW_AT_const_value")
                name = attr_str(child, "DW_AT_name")
                if const is not None and isinstance(const.value, int) and name is not None:
                    result.setdefault(const.value, name)
    return result


def resolve_name(die: DIE, _depth: int = 0) -> str | None:
    """Return a DIE's name, following ``abstract_origin``/``specification``.

    Inlined subroutines and many optimized DIEs carry no ``DW_AT_name`` of their
    own and instead point at an abstract definition that does.
    """
    if _depth > 8:
        return None
    name = attr_str(die, "DW_AT_name")
    if name is not None:
        return name
    for link in _NAME_LINK_ATTRS:
        target = follow_ref(die, link)
        if target is not None:
            resolved = resolve_name(target, _depth + 1)
            if resolved is not None:
                return resolved
    return None


def definition_die(die: DIE) -> DIE:
    """Return the abstract definition for an inlined/concrete DIE, or itself."""
    for link in _NAME_LINK_ATTRS:
        target = follow_ref(die, link)
        if target is not None:
            return target
    return die


def strip_type_qualifiers(type_die: DIE | None) -> DIE | None:
    """Peel ``const``/``volatile``/``typedef`` wrappers off a type DIE."""
    seen: set[int] = set()
    while type_die is not None and type_die.tag in _TRANSPARENT_TYPE_TAGS:
        if type_die.offset in seen:
            break
        seen.add(type_die.offset)
        type_die = follow_ref(type_die, "DW_AT_type")
    return type_die


def type_name(type_die: DIE | None, _depth: int = 0) -> str:
    """Render a (possibly nested) type DIE as a short human-readable string."""
    if type_die is None:
        return "void"
    if _depth > 8:
        return "..."
    tag = type_die.tag
    named = attr_str(type_die, "DW_AT_name")
    if named is not None and tag not in ("DW_TAG_pointer_type", "DW_TAG_reference_type"):
        return named
    inner = follow_ref(type_die, "DW_AT_type")
    if tag == "DW_TAG_pointer_type":
        return f"{type_name(inner, _depth + 1)}*"
    if tag == "DW_TAG_reference_type":
        return f"{type_name(inner, _depth + 1)}&"
    if tag == "DW_TAG_rvalue_reference_type":
        return f"{type_name(inner, _depth + 1)}&&"
    if tag in _TRANSPARENT_TYPE_TAGS:
        prefix = "const " if tag == "DW_TAG_const_type" else "volatile " if tag == "DW_TAG_volatile_type" else ""
        return f"{prefix}{type_name(inner, _depth + 1)}"
    return named or tag


@dataclass
class FileTable:
    """Resolves ``DW_AT_decl_file`` indices to source paths for one CU.

    Assumes DWARF 5, whose file and directory indices are both 0-based into the
    line program's tables (the toolchain that builds these kernels emits DWARF
    5). ``DW_AT_decl_file`` therefore indexes ``files`` directly.
    """

    files: list[str] = field(default_factory=list)

    def path_for_index(self, index: int) -> str | None:
        if 0 <= index < len(self.files):
            return self.files[index]
        return None

    @classmethod
    def build(cls, dwarf: DWARFInfo, cu) -> "FileTable":
        line_program = dwarf.line_program_for_CU(cu)
        if line_program is None:
            return cls()
        header = line_program.header
        include_dirs = [d.decode("utf-8", "replace") for d in header["include_directory"]]
        paths: list[str] = []
        for entry in header["file_entry"]:
            name = entry.name.decode("utf-8", "replace")
            dir_index = entry["dir_index"]
            directory = include_dirs[dir_index] if 0 <= dir_index < len(include_dirs) else ""
            paths.append(f"{directory}/{name}" if directory else name)
        return cls(paths)


class SourceResolver:
    """Resolves the source path of any DIE, across CUs and definition links.

    A ``DW_AT_decl_file`` index is only meaningful against the line program of
    the CU that *contains* the DIE bearing it. Inlined subroutines, however,
    often reach their ``decl_file`` only through an ``abstract_origin`` ->
    ``specification`` chain that lands in a different CU, so resolution must use
    the bearing DIE's own CU. File tables are cached per CU.
    """

    def __init__(self, dwarf: DWARFInfo) -> None:
        self._dwarf = dwarf
        self._tables: dict[int, FileTable] = {}

    def path_for(self, die: DIE) -> str | None:
        bearer = self._decl_file_bearer(die)
        if bearer is None:
            return None
        index = bearer.attributes["DW_AT_decl_file"].value
        return self._table_for(bearer.cu).path_for_index(index)

    def call_site(self, die: DIE) -> tuple[str | None, int | None, int | None]:
        """Return ``(file, line, column)`` of an inlined subroutine's call site.

        ``DW_AT_call_file`` is an index into the line program of the CU that
        *contains* the inlined DIE (not the abstract definition's CU), so it is
        resolved against ``die.cu`` directly.
        """
        attrs = die.attributes
        file_attr = attrs.get("DW_AT_call_file")
        line_attr = attrs.get("DW_AT_call_line")
        column_attr = attrs.get("DW_AT_call_column")
        path = None
        if file_attr is not None:
            path = self._table_for(die.cu).path_for_index(file_attr.value)
        return (
            path,
            line_attr.value if line_attr is not None else None,
            column_attr.value if column_attr is not None else None,
        )

    def _table_for(self, cu) -> FileTable:
        table = self._tables.get(cu.cu_offset)
        if table is None:
            table = FileTable.build(self._dwarf, cu)
            self._tables[cu.cu_offset] = table
        return table

    @staticmethod
    def _decl_file_bearer(die: DIE, _depth: int = 0) -> DIE | None:
        if _depth > 8:
            return None
        if "DW_AT_decl_file" in die.attributes:
            return die
        for link in _NAME_LINK_ATTRS:
            target = follow_ref(die, link)
            if target is not None:
                bearer = SourceResolver._decl_file_bearer(target, _depth + 1)
                if bearer is not None:
                    return bearer
        return None
