# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import re
import json
import sys
import os

from elftools.elf.elffile import ELFFile
from elftools.dwarf.descriptions import (
    describe_DWARF_expr,
)
from elftools.dwarf.locationlists import LocationEntry, LocationParser, LocationExpr
from elftools.dwarf.dwarf_expr import DWARFExprParser
from elftools.dwarf.datatype_cpp import parse_cpp_datatype


def show_loclist(loclist, dwarf_info, indent, cu_offset):
    """Display a location list nicely, decoding the DWARF expressions
    contained within.
    """
    d = []
    for loc_entity in loclist:
        if isinstance(loc_entity, LocationEntry):
            d.append(
                "%s <<%s>>" % (loc_entity, describe_DWARF_expr(loc_entity.loc_expr, dwarf_info.structs, cu_offset))
            )
        else:
            d.append(str(loc_entity))
    return "\n".join(indent + s for s in d)


def parse_variable_location(parent, dwarf_info, CU, DIE, loc_parser):
    loc = loc_parser.parse_from_attribute(DIE.attributes["DW_AT_location"], CU["version"])

    loc_expr = None

    # Only handle L1 variables for time being, no constants
    if isinstance(loc, LocationExpr):
        loc_expr = describe_DWARF_expr(loc.loc_expr, dwarf_info.structs, CU.cu_offset)
        loc_expr = int(re.sub("[()]", "", loc_expr).split(": ")[-1])

    return loc_expr


def get_variable_offset_info(fpath):
    with open(fpath, "rb") as f:
        elf_file = ELFFile(f)
        dwarf_info = elf_file.get_dwarf_info()

        location_lists = dwarf_info.location_lists()
        loc_parser = LocationParser(location_lists)

        variables = {}

        for CU in dwarf_info.iter_CUs():
            for DIE in CU.iter_DIEs():
                if "DW_AT_name" not in DIE.attributes.keys() or "DW_AT_location" not in DIE.attributes.keys():
                    continue

                parent = DIE.get_parent()
                if (
                    parent is not None
                    and "DW_AT_name" in parent.attributes.keys()
                    and parent.attributes["DW_AT_name"].value == b"kernel_main"
                ):
                    # Returns the variable location
                    name = DIE.attributes["DW_AT_name"].value.decode("utf-8")
                    variable_location = parse_variable_location(parent, dwarf_info, CU, DIE, loc_parser)

                    if variable_location is not None:
                        variables[name] = variable_location

    return {"variable_offset_info": variables}


def get_variable_type_info(fpath):
    with open(fpath, "rb") as f:
        elf_file = ELFFile(f)
        dwarf_info = elf_file.get_dwarf_info()

        location_lists = dwarf_info.location_lists()
        loc_parser = LocationParser(location_lists)

        variables = {}

        for CU in dwarf_info.iter_CUs():
            for DIE in CU.iter_DIEs():
                if "DW_AT_name" not in DIE.attributes.keys() or "DW_AT_location" not in DIE.attributes.keys():
                    continue

                parent = DIE.get_parent()
                if (
                    parent is not None
                    and "DW_AT_name" in parent.attributes.keys()
                    and parent.attributes["DW_AT_name"].value == b"kernel_main"
                ):
                    # Returns the variable type
                    name = DIE.attributes["DW_AT_name"].value.decode("utf-8")

                    # Bug in pyelftools: it doesn't check for qualifier volatile, so this is
                    # a workaround
                    if parse_cpp_datatype(DIE).name == "volatile ":
                        datatype = parse_cpp_datatype(DIE.get_DIE_from_attribute("DW_AT_type")).name
                        datatype = f"volatile {datatype}"
                    else:
                        datatype = parse_cpp_datatype(DIE).name

                    dietype = CU.get_DIE_from_refaddr(DIE.get_DIE_from_attribute("DW_AT_type").offset)
                    while "DW_AT_type" in dietype.attributes.keys():
                        dietype = CU.get_DIE_from_refaddr(dietype.get_DIE_from_attribute("DW_AT_type").offset)

                    # If got to the base type, there should be a byte size attribute
                    if "DW_AT_byte_size" in dietype.attributes.keys():
                        dtype_num_bytes = dietype.attributes["DW_AT_byte_size"].value
                    else:
                        dtype_num_bytes = None

                    variables[name] = [datatype, dtype_num_bytes]

    return {"variable_type_info": variables}


def get_frame_base_expression(fpath):
    def get_location_list_offset():
        for CU in dwarf_info.iter_CUs():
            for DIE in CU.iter_DIEs():
                if "DW_AT_name" not in DIE.attributes.keys() or "DW_AT_frame_base" not in DIE.attributes.keys():
                    continue

                if DIE.attributes["DW_AT_name"].value != b"kernel_main":
                    continue

                return DIE.attributes["DW_AT_frame_base"].value

    with open(fpath, "rb") as f:
        elf_file = ELFFile(f)
        dwarf_info = elf_file.get_dwarf_info()

        location_list_offset = get_location_list_offset()
        assert location_list_offset is not None, "Could not find the frame base location list offset information"

        frame_base_expression = [
            describe_DWARF_expr(loc_entry.loc_expr, dwarf_info.structs)
            for loc_entry in dwarf_info.location_lists().get_location_list_at_offset(location_list_offset)
        ]
    return frame_base_expression


def get_function_dwarf_offset(fpath, function_name):
    offset = None
    with open(fpath, "rb") as f:
        elf_file = ELFFile(f)
        dwarf_info = elf_file.get_dwarf_info()
        for CU in dwarf_info.iter_CUs():
            for DIE in CU.iter_DIEs():
                if "DW_AT_name" not in DIE.attributes.keys():
                    continue

                if DIE.attributes["DW_AT_name"].value != function_name:
                    continue
                return DIE.offset


def get_inlined_function_call_file_and_location(fpath, function_name):
    call_file_number = None
    call_path = None
    call_lines = []
    with open(fpath, "rb") as f:
        elf_file = ELFFile(f)
        dwarf_info = elf_file.get_dwarf_info()
        for CU in dwarf_info.iter_CUs():
            for DIE in CU.iter_DIEs():
                if DIE.tag != "DW_TAG_inlined_subroutine":
                    continue

                DIE_from_attr = DIE.get_DIE_from_attribute("DW_AT_abstract_origin")

                if DIE_from_attr.tag != "DW_TAG_subprogram":
                    continue

                if "DW_AT_name" not in DIE_from_attr.attributes.keys():
                    continue

                if DIE_from_attr.attributes["DW_AT_name"].value != function_name:
                    continue

                if call_path is None:
                    if call_file_number is None:
                        call_file_number = DIE.attributes["DW_AT_call_file"].value
                    else:
                        assert (
                            call_file_number == DIE.attributes["DW_AT_call_file"].value
                        ), "'call_file_number' changed within the same file"
                    line_program = dwarf_info.line_program_for_CU(CU)
                    call_file_info = line_program.header["file_entry"][call_file_number - 1]
                    call_file = call_file_info["name"]
                    call_file_dir_index = call_file_info["dir_index"]
                    call_file_dir_path = line_program.header["include_directory"][call_file_dir_index - 1]
                    call_path = f"{call_file_dir_path.decode('utf-8')}/{call_file.decode('utf-8')}"

                call_line = DIE.attributes["DW_AT_call_line"].value
                call_lines.append(call_line)

    return {"breakpoint_path_and_call_lines": [call_path, call_lines]}


def compute_offset_from_frame_pointer(frame_base_expression):
    offset = 0
    for expr in frame_base_expression:
        offset += int(re.sub("[()]", "", expr).split(": ")[-1])

    return {"offset_from_frame_pointer": offset}


def create_json_from_debug_information(*args):
    json_dict = {}

    for arg in args:
        json_dict.update(arg)

    return json.dumps(json_dict, indent=4)


if __name__ == "__main__":
    assert "TT_METAL_HOME" in os.environ, "'TT_METAL_HOME' must be set"

    assert len(sys.argv) == 3, "Must supply exactly two arguments, a thread type and an op consecutively"
    thread_type = sys.argv[1]
    op = sys.argv[2]
    op_dir = f"{os.environ['TT_METAL_HOME']}/built/kernels/{op}"
    assert os.path.isdir(op_dir), f"'{op_dir}' does not exist"
    assert os.path.isdir(f"{op_dir}/{thread_type}"), f"{op_dir}/{thread_type} does not exist"

    op_elf = f"{op_dir}/{thread_type}/{thread_type}.elf"
    assert os.path.isfile(op_elf), f"'{op_elf}' does not exist"

    # Extract info from elf
    frame_base_expression = get_frame_base_expression(op_elf)
    debug_info = [
        compute_offset_from_frame_pointer(frame_base_expression),
        get_variable_offset_info(op_elf),
        get_variable_type_info(op_elf),
        get_inlined_function_call_file_and_location(op_elf, b"breakpoint"),
    ]

    # Create a dictionary which merges all of this above data
    json_obj = create_json_from_debug_information(*debug_info)

    # breakpoint()
    with open(f"{op_dir}/{thread_type}/{thread_type}_debug_dwarf_info.json", "w") as j:
        j.write(json_obj)
