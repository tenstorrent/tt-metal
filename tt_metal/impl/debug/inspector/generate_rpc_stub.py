#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Script to generate inspector_capnp.pyi file from Cap'n Proto schema.
It helps with PyLance and mypy and it is not required to run the code.
"""

import sys
import os
from typing import Any

try:
    import capnp
except:
    print(
        "Cannot generate stub file for Inspector. If you need python autocomplete/static analysis, install with: pip install pycapnp"
    )
    sys.exit(0)

CAPNP_TYPE_TO_PYTHON = {
    "void": "None",
    "bool": "bool",
    "int8": "int",
    "int16": "int",
    "int32": "int",
    "int64": "int",
    "uint8": "int",
    "uint16": "int",
    "uint32": "int",
    "uint64": "int",
    "float32": "float",
    "float64": "float",
    "text": "str",
    "data": "bytes",
}


def get_display_name(schema) -> str:
    return schema.node.displayName[schema.node.displayNamePrefixLength :]


def quote(string: str) -> str:
    return f'"{string}"'


def print_indented(f, indent: int, text: str):
    f.write("    " * indent + text + "\n")


def get_type_hint(type, type_ids: dict[int, str]) -> str:
    slot_type = type.which()
    if slot_type == "list":
        return f"Sequence[{get_type_hint(type.list.elementType, type_ids)}]"
    elif slot_type in CAPNP_TYPE_TO_PYTHON:
        return CAPNP_TYPE_TO_PYTHON[slot_type]
    elif slot_type == "enum":
        return type_ids[type.enum.typeId]
    elif slot_type == "struct":
        return type_ids[type.struct.typeId]
    else:
        assert slot_type == "anyPointer"
        raise NotImplementedError("AnyPointer type is not supported in stub generation")


def print_type(f, node_schema, type_ids: dict[int, str], indent: int = 0, name: str | None = None):
    if name is None:
        name = get_display_name(node_schema)
    node_type = node_schema.node.which()
    f.write("\n")
    if node_type == "enum":
        print_indented(
            f,
            indent,
            f"{name}: TypeAlias = Literal[{', '.join(quote(v.name) for v in node_schema.node.enum.enumerants)}]",
        )
    elif node_type == "struct":
        print_indented(f, indent, f"class {name}:")
        indent += 1

        # Nested types
        nested_types: list[tuple[Any, str | None]] = [
            (node_schema.get_nested(n.name), None) for n in node_schema.node.nestedNodes
        ]

        # Print fields
        for field, raw_field in zip(node_schema.node.struct.fields, node_schema.as_struct().fields_list):
            field_type = field.which()
            if field_type == "slot":
                print_indented(f, indent, f"{field.name}: {get_type_hint(field.slot.type, type_ids)}")
            else:
                assert field_type == "group"
                if raw_field.schema.node.struct.discriminantCount:
                    group_name = f"_Union_{field.name}"
                else:
                    group_name = f"_Group_{field.name}"
                nested_types.append((raw_field.schema, group_name))
                full_type_name = f"{type_ids[node_schema.node.id]}.{group_name}"
                type_ids[raw_field.schema.node.id] = full_type_name
                print_indented(f, indent, f"{field.name}: {full_type_name}")

        # Print unnamed union members
        if node_schema.node.struct.discriminantCount:
            field_names = [
                f'"{field.name}"' for field in node_schema.node.struct.fields if field.discriminantValue != 65535
            ]
            print_indented(f, indent, f"def which(self) -> Literal[{', '.join(field_names)}]: ...")

        # Print nested types
        for nested_type in nested_types:
            print_type(f, nested_type[0], type_ids, indent, nested_type[1])
    elif node_type == "interface":
        print_indented(f, indent, f"class {name}:")
        indent += 1
        for method_name, method in node_schema.as_interface().methods.items():
            method_name_cap = method_name[0].upper() + method_name[1:]
            method_result = f"{method_name_cap}Results"
            param_fields = method.param_type.fields
            params = ["self"] + [
                f"{param_name}: {get_type_hint(param.proto.slot.type, type_ids)}"
                for param_name, param in param_fields.items()
            ]
            print_indented(f, indent, f"def {method_name}({', '.join(params)}) -> {name}.{method_result}: ...")
        for method_name, method in node_schema.as_interface().methods.items():
            method_name_cap = method_name[0].upper() + method_name[1:]
            method_result = f"{method_name_cap}Results"
            print_type(f, method.result_type, type_ids, indent, method_result)
    else:
        print(f" warning: {node_type} type is not supported in stub generation")


def fill_type_ids(schema, type_ids: dict[int, str], prefix: str = ""):
    for node in schema.node.nestedNodes:
        node_schema = schema.get_nested(node.name)
        name = get_display_name(node_schema)
        if prefix:
            name = f"{prefix}.{name}"
        type_ids[node.id] = name
        fill_type_ids(node_schema, type_ids, name)


def main():
    if len(sys.argv) != 3:
        print("Usage: generate_rpc_stub.py <capnp_file> <output_stub_file>")
        sys.exit(1)

    capnp_file = sys.argv[1]
    output_stub_file = sys.argv[2]

    try:
        # Import Cap'n Proto schema
        capnp.remove_import_hook()
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parser = capnp.SchemaParser()
        module = parser.load(capnp_file, imports=[os.path.dirname(p) for p in capnp.__path__])

        with open(output_stub_file, "w") as f:
            f.write(
                f"""# Auto-generated RPC stub from {capnp_file}

from __future__ import annotations

from typing import Literal, Sequence, TypeAlias
"""
            )
            type_ids: dict[int, str] = {}
            fill_type_ids(module.schema, type_ids)

            for node in module.schema.node.nestedNodes:
                node_schema = module.schema.get_nested(node.name)
                print_type(f, node_schema, type_ids)

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
