#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

import contextlib
import re
import sys
from dataclasses import dataclass
from typing import List, Dict, TextIO


@dataclass
class Field:
    name: str
    type: str
    struct_idx: int | None
    array_size_idx: int | None


@dataclass
class Struct:
    name: str
    fields: List[Field]
    array_sizes: List[str]
    struct_names: List[str]


class LabelAllocator:
    def __init__(self):
        self.labels: List[str] = []
        self.idx_map: Dict[str, int] = {}

    def lookup(self, label: str) -> int:
        idx = self.idx_map.get(label)
        if idx is None:
            idx = len(self.labels)
            self.labels.append(label)
            self.idx_map[label] = idx
        return idx


class CodeGen:
    SCALAR_TYPES = frozenset(
        [
            "char",
            "int8_t",
            "int16_t",
            "int32_t",
            "int64_t",
            "byte",
            "uint8_t",
            "uint16_t",
            "uint32_t",
            "uint64_t",
            "bool",
        ]
    )

    def __init__(self, input_file: TextIO, driver_ns: str, driver_include_path: str, interface_ns: str):
        self.input_file = input_file
        self.lineno = 0
        self.structs: List[Struct] = []
        self.struct_ids: Dict[str, int] = {}
        self.includes: List[str] = []
        self.enums: List[str] = []
        self.constants: List[str] = []
        self.scalar_types: set[str] = set(self.SCALAR_TYPES)
        self.driver_ns = driver_ns
        self.driver_include_path = driver_include_path
        self.interface_ns = interface_ns

    def is_scalar_type(self, type_name: str):
        if match := re.fullmatch(r"std::atomic<(\w+)>", type_name):
            type_name = match.group(1)
        return type_name in self.scalar_types

    def getline(self):
        skipping = 0
        while line := self.input_file.readline():
            self.lineno += 1
            if skipping == 0:
                if line.find("// CODEGEN:skip") != -1:
                    continue
                if (idx := line.find("//")) != -1:
                    line = line[:idx]
                line = line.strip()
                if line.startswith("#if"):
                    skipping = 1
                    continue
                if line.startswith("#include"):
                    self.includes.append(line)
                    continue
                if line == "" or line.startswith("#"):
                    # ignore all other directives
                    continue
                return line
            else:
                if line.startswith("#if"):
                    skipping += 1
                if skipping == 1 and line.startswith("#else"):
                    skipping = 0
                if line.startswith("#endif"):
                    skipping -= 1
        return None

    def parse_error(self, line: str, reason: str):
        print(f"error:{self.input_file.name}:{self.lineno}:")
        print(f"{self.lineno:4d} |", line)
        print("Failed to parse the above line: ", reason)
        sys.exit(1)

    def parse(self):
        while line := self.getline():
            if line.startswith("static_assert"):
                continue
            if re.match(r"(?:static )?constexpr ", line):
                self.constants.append(line)
                continue
            match = re.match(r"enum(?: class)? (\w+) (?:: \w+ )?{", line)
            if match:
                enum_name = match.group(1)
                self.scalar_types.add(enum_name)
                self.enums.append(line)
                # sigh, clang-format forces some enum definitions to be on a single line
                if not line.endswith(";"):
                    self.parse_enum()
                continue
            match = re.fullmatch(r"(?:struct|union) (\w+) {", line)
            if not match:
                self.parse_error(line, "expect struct definition")
            struct_name = match.group(1)
            struct_def = self.parse_struct(struct_name)
            struct_id = len(self.structs)
            self.structs.append(struct_def)
            self.struct_ids[struct_name] = struct_id

    def parse_enum(self):
        while line := self.getline():
            self.enums.append(line)
            if line == "};":
                return
        self.parse_error("<end of file>", "incomplete enum definition")

    def parse_struct(self, name: str):
        fields: List[Field] = []
        arrays = LabelAllocator()
        structs = LabelAllocator()
        level = 1
        while level > 0:
            line = self.getline()
            if not line:
                self.parse_error("<end of file>", "incomplete struct definition")
            if re.fullmatch(r"}(?: *__attribute__\(\(\w+\)\))*;", line):
                level -= 1
                continue
            # anonymous structs/unions are allowed
            if re.fullmatch(r"(?:struct|union) {", line):
                level += 1
                continue
            match = re.fullmatch(
                r"(?:volatile )?(?:struct | union )?([\w<>: ]+) +(\w+)(?:\[([\w:]+)\])? *(?:=.*)?;", line
            )
            if not match:
                self.parse_error(line, "expect struct field definition")
            type_name = match.group(1)
            field_name = match.group(2)
            array_size = match.group(3)
            if array_size is not None:
                array_idx = arrays.lookup(array_size)
            else:
                array_idx = None
            if type_name in self.struct_ids:
                struct_idx = structs.lookup(type_name)
            else:
                if not self.is_scalar_type(type_name):
                    self.parse_error(line, f"unresolved type {type_name}")
                struct_idx = None
            fields.append(Field(field_name, type_name, struct_idx, array_idx))
        return Struct(name, fields, arrays.labels, structs.labels)

    def generate_interface_header(self):
        print("#pragma once\n")
        my_includes = set(
            ["#include <array>", "#include <cstddef>", "#include <cstdint>", f'#include "{self.driver_include_path}"']
        )
        for include in self.includes:
            my_includes.discard(include)
            print(include)
        print(*my_includes, sep="\n")
        if self.interface_ns:
            print(f"namespace {self.interface_ns} {{")
        for constant in self.constants:
            print(constant)
        self.emit_types()
        self.emit_factory()
        print("using namespace types;\n")
        if self.interface_ns:
            print(f"}}  // namespace {self.interface_ns}")

    def emit_types(self):
        print("namespace types {")
        for enum in self.enums:
            print(enum)
        for struct in self.structs:
            print(
                f"struct {struct.name} : {self.driver_ns}::StructBuffer<{struct.name}> {{\n" "enum class Field {",
                end=" ",
            )
            print(*[field.name for field in struct.fields], sep=", ")
            print(" };")
            print(
                f"static constexpr size_t fields_count = {len(struct.fields)};\n"
                f"template <bool Const, Field F> struct FieldTraits;"
            )
            self.emit_view_template(struct)
            print(f"using View = BasicView<false>;\n" f"using ConstView = BasicView<true>;")
            print("};\n")
        print(
            "// Magic numbers are generated by code generator.\n"
            "// They will match the positions in the offsets array in the\n"
            "// (also generated) implementation file."
        )
        for struct in self.structs:
            self.emit_field_traits(struct)
        print("}  // namespace types\n")

    def emit_view_template(self, struct: Struct):
        print(
            f"template <bool Const>\n"
            f"struct BasicView : public {self.driver_ns}::BaseStructView<Const, {struct.name}> {{\n"
        )
        print(f"using {self.driver_ns}::BaseStructView<Const, {struct.name}>::BaseStructView;")
        for field in struct.fields:
            print(
                f"decltype(auto) {field.name}() const {{ return this->template get<{struct.name}::Field::{field.name}>(); }}"
            )
        print("};")

    def emit_field_traits(self, struct: Struct):
        for field_id, field in enumerate(struct.fields):
            args: List[str | int] = [field.type, field_id]
            if field.struct_idx is None:
                scalar_or_struct = "Scalar"
            else:
                scalar_or_struct = "Struct"
                struct_info_idx = len(struct.fields) + len(struct.array_sizes) + field.struct_idx
                args.append(struct_info_idx)
            if field.array_size_idx is None:
                field_or_array = "Field"
            else:
                field_or_array = "Array"
                array_size_idx = len(struct.fields) + field.array_size_idx
                args.append(array_size_idx)
            print(
                "template <bool Const>\n"
                f"struct {struct.name}::FieldTraits<Const, {struct.name}::Field::{field.name}> : "
                f"{self.driver_ns}::BaseStructView<Const, {struct.name}>::template {scalar_or_struct}{field_or_array}<",
                end="",
            )
            print(*args, sep=", ", end="> {};\n")

    def emit_factory(self):
        print(
            "class Factory {\n"
            "public:\n"
            f"using Impl = std::array<{self.driver_ns}::StructInfo, {len(self.structs)}>;\n"
        )
        print(
            "private:\n"
            "template <typename Struct> struct StructTraits;\n"
            "template <typename Struct> const auto& info() const { return (*impl_)[StructTraits<Struct>::index]; }\n"
            "public:\n"
            "Factory(const Impl& impl) : impl_(&impl) {}\n"
            "// Create a memory-owning object\n"
            "template <typename Struct> Struct create() const { return {info<Struct>()}; };\n"
            "// Create a view from arbitrary buffers - caller must guarantee the buffer has appropriate size\n"
            "template <typename Struct> Struct::View create_view(std::byte* base) const { return {info<Struct>(), base}; }\n"
            "// Create a const view from arbitrary buffers - caller must guarantee the buffer has appropriate size\n"
            "template <typename Struct> Struct::ConstView create_view(const std::byte* base) const { return {info<Struct>(), base}; }\n"
            "// Query the size of a struct\n"
            "template <typename Struct> size_t size_of() const { return info<Struct>().get_size(); }\n"
            "// Query the offset of a struct field\n"
            "template <typename Struct> size_t offset_of(Struct::Field f) const { return info<Struct>().offset_of(static_cast<size_t>(f)); }\n"
            "private:\n"
            "const Impl* impl_;\n"
            "};"
        )
        for i, struct in enumerate(self.structs):
            print(
                f"template<>"
                f"struct Factory::StructTraits<types::{struct.name}> {{"
                f"static constexpr size_t index = {i};"
                "};"
            )

    def generate_impl_fragment(self, raw_structs_ns: str):
        print("namespace offsets {\n")
        if raw_structs_ns:
            print(
                f"using namespace {raw_structs_ns};  // in case the generated code needs to refer to the constants defined there..."
            )
        for struct in self.structs:
            print(
                f"static const uintptr_t {struct.name}[] = {{\n"
                f"// {len(struct.fields)} fields\n"
                f"sizeof({raw_structs_ns}::{struct.name}),"
            )
            for field in struct.fields[1:]:
                print(f"offsetof({raw_structs_ns}::{struct.name}, {field.name}),")
            print(f"// {len(struct.array_sizes)} array sizes")
            for array_size in struct.array_sizes:
                print(array_size, end=",\n")
            print(f"// {len(struct.struct_names)} referenced structs")
            for struct_name in struct.struct_names:
                print(f"reinterpret_cast<uintptr_t>(offsets::{struct_name}),")
            offsets_len = len(struct.fields) + len(struct.array_sizes) + len(struct.struct_names)
            print(
                "};\n"
                f"// Total = {len(struct.fields)} + {len(struct.array_sizes)} + {len(struct.struct_names)} = {offsets_len}\n"
                f"static_assert(sizeof(offsets::{struct.name}) == {offsets_len} * sizeof(uintptr_t));\n"
            )
        print("}  // namespace offsets")
        print(
            f"{self.interface_ns}::Factory create_factory() {{\n"
            f"static const {self.interface_ns}::Factory::Impl impl_ {{"
        )
        for struct in self.structs:
            print(f"offsets::{struct.name},")
        print("};\n" "return {impl_};\n" "}")


import argparse

parser = argparse.ArgumentParser()

parser.add_argument("src_file")
parser.add_argument("out_intf_file")
parser.add_argument("out_impl_file")
parser.add_argument("-a", "--append_mode", action=argparse.BooleanOptionalAction)
parser.add_argument("-d", "--driver_ns", default="")
parser.add_argument("-D", "--driver_include_path", default="struct_view_driver.h")
parser.add_argument("-i", "--interface_ns", default="")
parser.add_argument("-r", "--raw_structs_ns", default="")
args = parser.parse_args()

with open(args.src_file, "r") as f:
    cg = CodeGen(f, args.driver_ns, args.driver_include_path, args.interface_ns)
    cg.parse()
    print(f"parsed {len(cg.structs)} structs.")

mode = "a" if args.append_mode else "w"
with open(args.out_intf_file, mode) as g:
    with contextlib.redirect_stdout(g):
        cg.generate_interface_header()
with open(args.out_impl_file, mode) as g:
    with contextlib.redirect_stdout(g):
        cg.generate_impl_fragment(args.raw_structs_ns)
