#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Script Name: dump_callstacks.py

Usage:
    dump_callstacks [--full_callstack] [--gdb_callstack]

Options:
    --full_callstack   Dump full callstack with all frames. Defaults to dumping only the top frame.
    --gdb_callstack    Dump callstack using GDB client instead of built-in methods.
"""

from triage import ScriptConfig
from devices_to_check import run as get_devices_to_check
from dispatcher_data import run as get_dispatcher_data, DispatcherData, DispatcherCoreData
from tabulate import tabulate
from ttexalens.coordinate import OnChipCoordinate
from ttexalens.context import Context
from ttexalens.hardware.risc_debug import CallstackEntry, ParsedElfFile
from ttexalens.tt_exalens_lib import top_callstack, callstack, parse_elf
from utils import BLUE, GREEN, ORANGE, RST, DEFAULT_TABLE_FORMAT

script_config = ScriptConfig(
    depends=["dispatcher_data"],
)


def get_gdb_callstack(
    location: OnChipCoordinate, risc_name: str, dispatcher_core_data: DispatcherCoreData, context: Context
) -> list[CallstackEntry]:
    raise NotImplementedError("Using GDB callstack is not implemented yet. Please use the built-in callstack methods.")


def get_callstack(
    location: OnChipCoordinate,
    risc_name: str,
    dispatcher_core_data: DispatcherCoreData,
    elfs_cache: dict[str, ParsedElfFile],
    full_callstack: bool,
) -> list[CallstackEntry]:
    context = location._device._context
    if dispatcher_core_data.firmware_path not in elfs_cache:
        elfs_cache[dispatcher_core_data.firmware_path] = parse_elf(dispatcher_core_data.firmware_path, context)
    elfs: list[ParsedElfFile] = [elfs_cache[dispatcher_core_data.firmware_path]]
    offsets: list[int | None] = [None]
    if dispatcher_core_data.kernel_path is not None:
        if dispatcher_core_data.kernel_path not in elfs_cache:
            elfs_cache[dispatcher_core_data.kernel_path] = parse_elf(dispatcher_core_data.kernel_path, context)
        elfs.append(elfs_cache[dispatcher_core_data.kernel_path])
        offsets.append(dispatcher_core_data.kernel_text_offset)
    try:
        if not full_callstack:
            pc = location._device.get_block(location).get_risc_debug(risc_name).get_pc()
            return top_callstack(pc, elfs, offsets, context)
        else:
            device_id = location._device._id
            return callstack(location, elfs, offsets, risc_name, device_id=device_id, context=context)
    except:
        return []


def format_callstack(callstack: list[CallstackEntry]) -> list[str]:
    """Return string representation of the callstack."""
    frame_number_width = len(str(len(callstack) - 1))
    result = []
    for i, frame in enumerate(callstack):
        line = f"  #{i:<{frame_number_width}} "
        if frame.pc is not None:
            line += f"{BLUE}0x{frame.pc:08X}{RST} in "
        if frame.function_name is not None:
            line += f"{ORANGE}{frame.function_name}{RST} () "
        if frame.file is not None:
            line += f"at {GREEN}{frame.file} {frame.line}:{frame.column}{RST}"
        result.append(line)
    return result


def dump_callstacks(
    devices: list[int], dispatcher_data: DispatcherData, context: Context, full_callstack: bool, gdb_callstack: bool
):
    blocks_to_test = ["functional_workers", "eth"]
    elfs_cache: dict[str, ParsedElfFile] = {}
    table = [
        ["Dev", "Loc", "Proc", "RD PTR", "Base", "Offset", "Kernel ID:Name", "PC", "Kernel Callstack", "Kernel Path"]
    ]
    for device_id in devices:
        device = context.devices[device_id]
        for block_to_test in blocks_to_test:
            for location in device.get_block_locations(block_to_test):
                noc_block = device.get_block(location)

                # We support only idle eth blocks for now
                if noc_block.block_type == "eth" and noc_block not in device.idle_eth_blocks:
                    continue

                for risc_name in noc_block.risc_names:
                    dispatcher_core_data = dispatcher_data.get_core_data(location, risc_name)
                    if gdb_callstack:
                        callstack = get_gdb_callstack(location, risc_name, dispatcher_core_data, context)
                    else:
                        callstack = get_callstack(location, risc_name, dispatcher_core_data, elfs_cache, full_callstack)
                    table.append(
                        [
                            str(device_id),
                            location.to_str("logical"),
                            risc_name.upper(),
                            str(dispatcher_core_data.launch_msg_rd_ptr),
                            f"0x{dispatcher_core_data.kernel_config_base:x}",
                            f"0x{dispatcher_core_data.kernel_text_offset:x}",
                            f"{dispatcher_core_data.watcher_kernel_id}:{dispatcher_core_data.kernel_name}",
                            f"0x{callstack[0].pc:x}" if len(callstack) > 0 and callstack[0].pc is not None else "N/A",
                            "" if len(callstack) > 0 else "N/A",
                            dispatcher_core_data.kernel_path if dispatcher_core_data.kernel_path else "N/A",
                        ]
                    )
                    for line in format_callstack(callstack):
                        table.append(["", "", "", "", "", "", "", "", line])
    print(tabulate(table, headers="firstrow", tablefmt=DEFAULT_TABLE_FORMAT))


def run(args, context: Context):
    full_callstack = args["--full_callstack"]
    gdb_callstack = args["--gdb_callstack"]
    devices = get_devices_to_check(args, context)
    dispatcher_data = get_dispatcher_data(args, context)
    dump_callstacks(devices, dispatcher_data, context, full_callstack, gdb_callstack)


if __name__ == "__main__":
    from triage import run_script

    run_script()
