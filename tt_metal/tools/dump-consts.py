# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Usage:
    dump-consts.py <elf_file_path> [--function=<function_name>]

Options:
    -h --help                   Show this screen
    --function=<function_name>  Limits the search to a specific function

Description:
    This script prints compiler-assigned constant values from the ELF file. It will
    print constants found in all functions if no function name is provided.

Example:
    python dump-consts.py built/0/kernels/command_queue_producer/266580784283234019/brisc/brisc.elf --function=kernel_main
"""
from docopt import docopt
from elftools.elf.elffile import ELFFile


def build_abstract_origin_index(dwarfinfo):
    """Build an index of all DW_AT_abstract_origin DIEs and their const_values"""
    index = {}
    for CU in dwarfinfo.iter_CUs():
        for DIE in CU.iter_DIEs():
            if "DW_AT_abstract_origin" in DIE.attributes and "DW_AT_const_value" in DIE.attributes:
                origin_offset = DIE.attributes["DW_AT_abstract_origin"].value
                const_value = DIE.attributes["DW_AT_const_value"].value
                index[origin_offset] = const_value
    return index


def find_constexpr_values(dwarfinfo, function_name, die_index):
    """Find all DW_AT_const_expr DIEs and print their values"""
    constexprs_map = {}  # Map function name to a list of constexprs
    for CU in dwarfinfo.iter_CUs():
        for DIE in CU.iter_DIEs():
            if DIE.tag == "DW_TAG_subprogram" and "DW_AT_name" in DIE.attributes:
                die_function_name = DIE.attributes["DW_AT_name"].value.decode("utf-8")
                constexprs_map[die_function_name] = []
                if die_function_name == function_name or not function_name:
                    for child in DIE.iter_children():
                        if child.tag == "DW_TAG_variable":
                            name = (
                                child.attributes["DW_AT_name"].value.decode("utf-8")
                                if "DW_AT_name" in child.attributes
                                else "<unnamed>"
                            )
                            if child.offset in die_index:
                                # print (f"Constexpr: {name}, DW_AT_const_expr offset: {child.offset}, const_value = {hex(die_index[child.offset])}")
                                value = die_index[child.offset]
                                constexprs_map[die_function_name].append((name, value))

    for die_function_name, constexprs in constexprs_map.items():
        if constexprs:
            if not function_name:
                print(f"Function: {die_function_name}")
            print("\n".join(f"  {name} = {value} ({hex(value)})" for name, value in constexprs))


def main():
    arguments = docopt(__doc__)
    elf_file_path = arguments["<elf_file_path>"]
    function_name = arguments["--function"]

    with open(elf_file_path, "rb") as f:
        elffile = ELFFile(f)
        if not elffile.has_dwarf_info():
            print(
                "ELF file has no DWARF info. To instruct the compiler to generate DWARF info, add TT_METAL_RISCV_DEBUG_INFO=1 to your environment."
            )
            return

        dwarfinfo = elffile.get_dwarf_info()
        die_index = build_abstract_origin_index(dwarfinfo)
        find_constexpr_values(dwarfinfo, function_name, die_index)


if __name__ == "__main__":
    main()
