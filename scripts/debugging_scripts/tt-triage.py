#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Script Name: tt-triage.py

Usage:
    tt-triage [--halt-on-error] [--inspector-log-path=<inspector_log_path>] [-v | --verbose] [-V | --vverbose] [--dev=<device_id>]...

Options:
    -h --help                                  Show this screen.
    --dev=<device_id>                          Specify the device id. 'all' is also an option  [default: in_use]
    -v --verbose                               Print verbose output.                           [default: False]
    -V --vverbose                              Print more verbose output.                      [default: False]
    --inspector-log-path=<inspector_log_path>  Path to the inspector log directory.
    --halt-on-error                            Halt on first error.                            [default: False]

Description:
    Diagnoses Tenstorrent AI hardware by performing comprehensive health checks on ARC processors, NOC connectivity, L1 memory, and RISC-V cores.
    Identifies running kernels and provides callstack information to troubleshoot failed operations. Use with --verbose for detailed diagnostics
    or --halt-on-error to stop on first failure.
    Example use with tt-metal:
        export TT_METAL_HOME=~/work/tt-metal
        ./build_metal.sh --build-programming-examples
        build/programming_examples/matmul_multi_core
        tt-triage
"""

from collections import namedtuple
import time
import os
import sys
from parse_inspector_logs import get_data as get_inspector_data, InspectorData

RST = "\033[0m"
BLUE = "\033[34m"  # For good values
RED = "\033[31m"  # For bad values
GREEN = "\033[32m"  # For instructions
ORANGE = "\033[33m"  # For warnings
VERBOSE_CLR = "\033[94m"  # For verbose output

# Global variables for verbosity settings
VERBOSE = False
VVERBOSE = False
context = None

try:
    from tabulate import tabulate, TableFormat, Line, DataRow
    from docopt import docopt
    import yaml
    from ttexalens.tt_exalens_init import init_ttexalens
    from ttexalens.tt_exalens_lib import (
        read_words_from_device,
        read_word_from_device,
        top_callstack,
        parse_elf,
        read_arc_telemetry_entry,
    )
    from ttexalens.device import Device
    from ttexalens.hardware.noc_block import NocBlock
    from ttexalens.hw.tensix.wormhole.wormhole import WormholeDevice
    from ttexalens.hw.tensix.blackhole.blackhole import BlackholeDevice
    from ttexalens.firmware import ELF
    from ttexalens.parse_elf import mem_access
except ImportError as e:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    install_script = os.path.join(os.path.dirname(script_dir), "install_debugger.sh")
    print(f"Module '{e}' not found. Please install tt-exalens by running: {GREEN}")
    print(f"  {install_script}{RST}")
    exit(1)


DEFAULT_TABLE_FORMAT = TableFormat(
    lineabove=Line("╭", "─", "┬", "╮"),
    linebelowheader=Line("├", "─", "┼", "┤"),
    linebetweenrows=None,
    linebelow=Line("╰", "─", "┴", "╯"),
    headerrow=DataRow("│", "│", "│"),
    datarow=DataRow("│", "│", "│"),
    padding=1,
    with_header_hide=None,
)


class TTTriageError(Exception):
    """Base class for TT Triage errors."""

    pass


def raiseTTTriageError(msg):
    """Raise a TT Triage error."""
    if HALT_ON_ERROR:
        raise TTTriageError(msg)
    else:
        print(f"{RED}ERROR: {msg}{RST}")


def verbose(msg):
    """Print message if verbose mode is enabled (-v)"""
    if VERBOSE or VVERBOSE:
        print(f"{VERBOSE_CLR}{msg}{RST}")


def vverbose(msg):
    """Print message if verbose mode is enabled (-vv)."""
    if VVERBOSE:
        print(f"{VERBOSE_CLR}{msg}{RST}")


def title(msg):
    """Print a title."""
    print(f"{GREEN}= {msg}{RST}")


def check_ARC(dev):
    """Checking that ARC heartbeat is running. Estimating ARC uptime (-v)."""
    title(check_ARC.__doc__)

    # Postcode must be correct (C0DE)
    # postcode = dev.ARC.ARC_RESET.SCRATCH[0].read()
    arc = dev.arc_block
    postcode = arc.get_register_store().read_register("ARC_RESET_SCRATCH0")
    if postcode & 0xFFFF0000 != 0xC0DE0000:
        print(f"ARC postcode: {RED}0x{postcode:08x}{RST}. Expected {BLUE}0xc0de____{RST}")
        raiseTTTriageError(check_ARC.__doc__)

    if type(dev) == WormholeDevice:
        # Heartbeat must be increasing
        # heartbeat_0 = dev.ARC.reset_unit.DEBUG.heartbeat.read()
        heartbeat_0 = read_arc_telemetry_entry(dev.id(), "TAG_ARC0_HEALTH")
        delay_seconds = 0.1
        time.sleep(delay_seconds)
        # heartbeat_1 = dev.ARC.ARC_CSM.DEBUG.heartbeat.read()
        heartbeat_1 = read_arc_telemetry_entry(dev.id(), "TAG_ARC0_HEALTH")
        if heartbeat_1 <= heartbeat_0:
            print(f"ARC heartbeat not increasing: {RED}{heartbeat_1}{RST}.")
            raiseTTTriageError(check_ARC.__doc__)

        # Compute uptime
        # arcclk_mhz = dev.ARC.ARC_CSM.AICLK_PPM.curr_arcclk.read()
        arcclk_mhz = read_arc_telemetry_entry(dev.id(), "TAG_ARCCLK")
        heartbeats_per_second = (heartbeat_1 - heartbeat_0) / delay_seconds
        uptime_seconds = heartbeat_1 / heartbeats_per_second

        # Heartbeat must be between 500 and 20000 hb/s
        if heartbeats_per_second < 500:
            print(
                f"ARC heartbeat is too low: {RED}{heartbeats_per_second}{RST}hb/s. Expected at least {BLUE}500{RST}hb/s"
            )
            raiseTTTriageError(check_ARC.__doc__)
        if heartbeats_per_second > 20000:
            print(
                f"ARC heartbeat is too high: {RED}{heartbeats_per_second}{RST}hb/s. Expected at most {BLUE}20000{RST}hb/s"
            )
            raiseTTTriageError(check_ARC.__doc__)

        # Print heartbeat and uptime
        verbose(f"ARC heartbeat: {heartbeat_1} - {heartbeat_0} = {heartbeats_per_second}hb/s, ARCCLK: {arcclk_mhz} MHz")
        days = int(uptime_seconds // (24 * 3600))
        hours = int((uptime_seconds % (24 * 3600)) // 3600)
        minutes = int((uptime_seconds % 3600) // 60)
        seconds = int(uptime_seconds % 60)
        verbose(f"Approximate ARC uptime: {GREEN}{days}d {hours:02}:{minutes:02}:{seconds:02}s{RST}")
    elif type(dev) == BlackholeDevice:
        # Heartbeat must be increasing
        # heartbeat_0 = dev.ARC.reset_unit.DEBUG.heartbeat.read()
        heartbeat_0 = read_arc_telemetry_entry(dev.id(), "TAG_TIMER_HEARTBEAT")
        delay_seconds = 0.2
        time.sleep(delay_seconds)
        # heartbeat_1 = dev.ARC.ARC_CSM.DEBUG.heartbeat.read()
        heartbeat_1 = read_arc_telemetry_entry(dev.id(), "TAG_TIMER_HEARTBEAT")
        if heartbeat_1 <= heartbeat_0:
            print(f"ARC heartbeat not increasing: {RED}{heartbeat_1}{RST}.")
            raiseTTTriageError(check_ARC.__doc__)

        # Compute uptime
        # arcclk_mhz = dev.ARC.ARC_CSM.AICLK_PPM.curr_arcclk.read()
        arcclk_mhz = read_arc_telemetry_entry(dev.id(), "TAG_ARCCLK")
        heartbeats_per_second = (heartbeat_1 - heartbeat_0) / delay_seconds
        uptime_seconds = heartbeat_1 / heartbeats_per_second

        # Heartbeat must be between 10 and 50
        if heartbeats_per_second < 10:
            print(
                f"ARC heartbeat is too low: {RED}{heartbeats_per_second}{RST}hb/s. Expected at least {BLUE}10{RST}hb/s"
            )
            raiseTTTriageError(check_ARC.__doc__)
        if heartbeats_per_second > 50:
            print(
                f"ARC heartbeat is too high: {RED}{heartbeats_per_second}{RST}hb/s. Expected at most {BLUE}50{RST}hb/s"
            )
            raiseTTTriageError(check_ARC.__doc__)

        # Print heartbeat and uptime
        verbose(f"ARC heartbeat: {heartbeat_1} - {heartbeat_0} = {heartbeats_per_second}hb/s, ARCCLK: {arcclk_mhz} MHz")
        days = int(uptime_seconds // (24 * 3600))
        hours = int((uptime_seconds % (24 * 3600)) // 3600)
        minutes = int((uptime_seconds % 3600) // 60)
        seconds = int(uptime_seconds % 60)
        verbose(f"Approximate ARC uptime: {GREEN}{days}d {hours:02}:{minutes:02}:{seconds:02}s{RST}")
    else:
        print(f"{ORANGE}ARC uptime check is not available on this device.{RST}")


def check_L1(dev):
    """Checking location 0x0 of each core's L1 for the presence of FW"""
    title(check_L1.__doc__)

    # Firmware loaded L1[0] == 0x3800306f
    addr = 0x0
    expected = 0x7800306F if type(dev) == WormholeDevice else 0x7A00306F
    for loc in dev.get_block_locations(block_type="functional_workers"):
        data = read_words_from_device(loc, addr, device_id=dev.id(), word_count=1, context=context)
        if data[0] != expected:
            print(f"L1 @{loc}, addr 0x{addr:08x}: {RED}0x{data[0]:08x}{RST}. Expected {BLUE}{expected:08x}{RST}")
            raiseTTTriageError(check_L1.__doc__)


def noc_ping(dev: Device, loc, use_noc1: False):
    noc_id = 0 if not use_noc1 else 1
    noc_str = "noc0" if not use_noc1 else "noc1"
    noc_node_id_address = dev.get_block(loc).get_register_store(noc_id).get_register_noc_address("NOC_NODE_ID")
    assert (
        noc_node_id_address is not None
    ), f"NOC node ID address not found for {noc_str} at location {loc.to_str('logical')}"
    data = read_words_from_device(loc, noc_node_id_address, device_id=dev.id(), word_count=1)
    n_x = data[0] & 0x3F
    n_y = (data[0] >> 6) & 0x3F
    loc_to_noc = loc.to(noc_str)
    if loc_to_noc != (n_x, n_y):
        print(f"loc {RED}{loc_to_noc}{RST} Expected {BLUE}({n_x}, {n_y}){RST} but got {RED}{loc_to_noc}{RST}")


def check_NOC(dev):
    """Checking that we can reach all NOC endpoints through NOC0 (TODO: NOC1)"""
    title(check_NOC.__doc__)

    # Ping all locations
    for block_type in ["functional_workers", "eth"]:
        verbose(f"Checking {block_type} locations")
        all_locs = dev.get_block_locations(block_type)
        for loc in all_locs:
            noc_ping(dev, loc, use_noc1=False)
            noc_ping(dev, loc, use_noc1=True)


def collect_pcs_from_riscv(
    dev: Device,
    blocks: list[NocBlock],
):
    """Collect PC from RISC-V cores through the debug bus."""

    pcs = dict()  # location -> {pc_name -> pc}
    # Get the PC from the RISC-V cores
    for block in blocks:
        store = block.debug_bus
        assert store is not None, f"Debug bus not found for location {block.location.to_str('logical')}"
        pc_dict = dict()
        for risc_name in block.risc_names:
            pc_dict[risc_name + "_pc"] = block.get_risc_debug(risc_name).get_pc()
        pcs[block.location] = pc_dict
    return pcs


def print_pcs_from_riscv(dev, pcs):
    table = []
    header_row = None

    # Dump PC through debug bus
    if VERBOSE or VVERBOSE:
        for loc, pc_dict in pcs.items():
            # Set header row in first iteration
            if header_row is None:
                header_row = ["Loc", *[risc_name + "_pc" for risc_name in dev.get_block(loc).risc_names]]
                table.append(header_row)

            loc_row = [f"{loc.to_str('logical')}"]
            for sig in header_row[1:]:
                pc = pc_dict[sig]
                loc_row.append(f"0x{pc:x}")
            table.append(loc_row)

    verbose(tabulate(table, headers="firstrow", tablefmt=DEFAULT_TABLE_FORMAT))


def check_riscV(dev: Device):
    """Checking that the RISC-V cores are running. Dumping PC through debug bus (-v)."""
    title(check_riscV.__doc__)

    if type(dev) == WormholeDevice:
        # RISC-V soft resets are released
        # Reference table for RISC-V core states
        # after tt-smi reset  - after metal run
        # - 0: 0x7fdff7fd     - 0x3fcff3fc  # These vary by device
        # - 1: 0xdff7fdff     - 0xcff3fcff
        # - 2: 0xffffff7f     - 0x0000ff3f
        # - 3: 0xffffffff     - 0x00000000  # These do not vary by device
        # - 4: 0xffffffff     - 0x00000000
        # - 5: 0xffffffff     - 0x00000000
        # - 6: 0xffffffff     - 0x00000000
        # - 7: 0xffffffff     - 0x00000000

        expected_after_metal_run = {
            0: 0x3FCFF3FC,
            1: 0xCFF3FCFF,
            2: 0x0000FF3F,
            3: 0x00000000,
            4: 0x00000000,
            5: 0x00000000,
            6: 0x00000000,
            7: 0x00000000,
        }

        # TODO: Check this code
        # for i in range(len(dev.ARC.ARC_RESET.RISCV_RESET)):
        #     read_value = dev.ARC.ARC_RESET.RISCV_RESET[i].read()
        for i in range(3, 8):
            read_value = read_word_from_device(dev.arc_block.location, 0x880030040 + i * 4)

            verbose(f"{i}: 0x{read_value:08x}")
            if read_value != expected_after_metal_run[i]:
                print(
                    f"Mismatch in RiscV reset register {i}: Expected {BLUE}0x{expected_after_metal_run[i]:08x}{RST}, but got {RED}0x{read_value:08x}{RST}"
                )
                raiseTTTriageError("RISC-V core state does not match expected 'after metal run' values.")
    else:
        print(f"{ORANGE}RISC-V soft resets check is not available on this device.{RST}")

    # Collect PC from RISC-V cores through debug bus
    pcs_tensix = collect_pcs_from_riscv(dev, blocks=dev.get_blocks(block_type="functional_workers"))
    pcs_idle_eth = collect_pcs_from_riscv(dev, blocks=dev.idle_eth_blocks)

    # Printing
    print_pcs_from_riscv(dev, pcs_tensix)
    print_pcs_from_riscv(dev, pcs_idle_eth)


def format_callstack(cs):
    """Return string representation of the callstack."""
    frame_number_width = len(str(len(cs) - 1))
    result = []
    for i, frame in enumerate(cs):
        line = f"  #{i:<{frame_number_width}} "
        if frame.pc is not None:
            line += f"{BLUE}0x{frame.pc:08X}{RST} in "
        if frame.function_name is not None:
            line += f"{ORANGE}{frame.function_name}{RST} () "
        if frame.file is not None:
            line += f"at {GREEN}{frame.file} {frame.line}:{frame.column}{RST}"
        result.append(line)
    return result


def get_firmware_elf_path(a_kernel_path: str, risc_name: str) -> str:
    firmware_elf_path = a_kernel_path + f"../../../firmware/{risc_name.lower()}/{risc_name.lower()}.elf"
    return os.path.realpath(firmware_elf_path)


def init_running_ops_table(enum_values) -> list[list[str]] | None:
    if VVERBOSE:
        return [["Loc", "Proc", "RD PTR", "Base", "Offset", "Kernel ID:Name", "PC", "Kernel Callstack", "Kernel Path"]]
    elif VERBOSE:
        return [["Loc", *enum_values["ProcessorTypes"].keys(), "Kernel ID:Name"]]
    else:
        return None


def get_info_from_firmware_elf(fw_elf, loc_mem_reader, programmable_core_type, proc_type, proc_class):
    launch_msg_rd_ptr = mem_access(fw_elf, "mailboxes->launch_msg_rd_ptr", loc_mem_reader)[0][0]

    # Refer to tt_metal/api/tt-metalium/dev_msgs.h for struct kernel_config_msg_t
    kernel_config_base = mem_access(
        fw_elf,
        f"mailboxes->launch[{launch_msg_rd_ptr}].kernel_config.kernel_config_base[{programmable_core_type}]",
        loc_mem_reader,
    )[0][
        0
    ]  # Indexed with enum ProgrammableCoreType - tt_metal/hw/inc/*/core_config.h
    kernel_text_offset = mem_access(
        fw_elf,
        f"mailboxes->launch[{launch_msg_rd_ptr}].kernel_config.kernel_text_offset[{proc_type}]",
        loc_mem_reader,
    )[0][
        0
    ]  # Size 5 (NUM_PROCESSORS_PER_CORE_TYPE) - seems to be DM0,DM1,MATH0,MATH1,MATH2
    watcher_kernel_id = (
        mem_access(
            fw_elf,
            f"mailboxes->launch[{launch_msg_rd_ptr}].kernel_config.watcher_kernel_ids[{proc_class}]",
            loc_mem_reader,
        )[0][0]
        & 0xFFFF
    )  # enum dispatch_core_processor_classes

    return launch_msg_rd_ptr, kernel_config_base, kernel_text_offset, watcher_kernel_id


def get_running_ops_table(dev, blocks, enum_values, inspector_data, programmable_core_type, fw_elf, pcs, a_kernel_path):
    printout_table = init_running_ops_table(enum_values)

    if printout_table is None:
        return printout_table

    elf_cache = dict()

    # Get the kernel_config_base for each core
    for block in blocks:
        loc = block.location
        if VERBOSE:
            row = [loc.to_str("logical")]

        for risc_name in block.risc_names:
            proc_name = risc_name.upper()
            proc_type = enum_values["ProcessorTypes"][proc_name]
            proc_class = enum_values["dispatch_core_processor_classes"][proc_name]

            # Create a local wrapper for mem_reader that captures loc and dev
            loc_mem_reader = ELF.get_mem_reader(context, dev.id(), loc)

            try:
                (
                    launch_msg_rd_ptr,
                    kernel_config_base,
                    kernel_text_offset,
                    watcher_kernel_id,
                ) = get_info_from_firmware_elf(fw_elf, loc_mem_reader, programmable_core_type, proc_type, proc_class)
            except Exception as e:
                print()
                print(f"Loc: {loc}, Process: {proc_name}")
                print(e)
                print("Could not obtain information from elf, skipping process...")
                continue

            kernel = inspector_data.kernels.get(watcher_kernel_id)
            kernel_name = kernel.name if kernel else ""

            cs = []
            if proc_name.lower() == "erisc" or proc_name.lower() == "erisc0":
                fw_elf_path = a_kernel_path + "../../../firmware/idle_erisc/idle_erisc.elf"
            elif proc_name.lower() == "erisc1":
                fw_elf_path = a_kernel_path + "../../../firmware/subordinate_idle_erisc/subordinate_idle_erisc.elf"
            else:
                fw_elf_path = a_kernel_path + f"../../../firmware/{proc_name.lower()}/{proc_name.lower()}.elf"
                fw_elf_path = os.path.realpath(fw_elf_path)
            kernel_path = ""

            if kernel_name:
                assert kernel is not None, f"Kernel with watcher_kernel_id {watcher_kernel_id} not found."
                if proc_name.lower() == "erisc" or proc_name.lower() == "erisc0":
                    kernel_path = kernel.path + "/idle_erisc/idle_erisc.elf"
                elif proc_name.lower() == "erisc1":
                    kernel_path = kernel.path + "/subordinate_idle_erisc/subordinate_idle_erisc.elf"
                else:
                    kernel_path = kernel.path + f"/{proc_name.lower()}/{proc_name.lower()}.elf"
                kernel_path = os.path.realpath(kernel_path)
                if not os.path.exists(kernel_path):
                    raiseTTTriageError(f"Kernel ELF file {kernel_path} does not exist.")

                pc = pcs[loc][proc_name.lower() + "_pc"]
                if VVERBOSE:
                    print(f".", end="", flush=True)

                    if fw_elf_path not in elf_cache:
                        elf_cache[fw_elf_path] = parse_elf(fw_elf_path, context)
                    if kernel_path not in elf_cache:
                        elf_cache[kernel_path] = parse_elf(kernel_path, context)
                    if proc_name == "NCRISC" and type(dev) == WormholeDevice:
                        kernel_offset = 0xFFC00000
                    else:
                        kernel_offset = kernel_config_base + kernel_text_offset

                    cs = top_callstack(
                        pc, [elf_cache[fw_elf_path], elf_cache[kernel_path]], [None, kernel_offset], context=context
                    )
            else:
                pc = pcs[loc][proc_name.lower() + "_pc"]
                if VVERBOSE:
                    print(f".", end="", flush=True)

                    if fw_elf_path not in elf_cache:
                        elf_cache[fw_elf_path] = parse_elf(fw_elf_path, context)

                    cs = top_callstack(pc, elf_cache[fw_elf_path], context=context)

            if VVERBOSE:
                pc = pcs[loc][proc_name.lower() + "_pc"]
                row = [
                    loc.to_str("logical"),
                    proc_name,
                    str(launch_msg_rd_ptr),
                    f"0x{kernel_config_base:x}",
                    f"0x{kernel_text_offset:x}",
                    f"{watcher_kernel_id}:{kernel_name}",
                    f"0x{pc:x}",
                    "",
                    f"{kernel_path}",
                ]

            elif VERBOSE:
                row.append(f"{watcher_kernel_id}:{kernel_name}")

            if kernel_name or kernel_config_base or VVERBOSE:
                printout_table.append(row)
                if cs:
                    for line in format_callstack(cs):
                        printout_table.append(["", "", "", "", "", "", "", line])

    return printout_table


def dump_running_ops(dev: Device, inspector_data: InspectorData | None):
    """Print the running operations on the device."""
    title(dump_running_ops.__doc__)

    if inspector_data is None:
        print(f"  {ORANGE}We don't have inspector data. We will skip running ops dump.{RST}")
        return

    a_kernel_path = next(iter(inspector_data.kernels.values())).path

    # Get firmware elfs which we will use to get to the important offsets.
    brisc_elf_path = get_firmware_elf_path(a_kernel_path, "brisc")
    idle_erisc_elf_path = get_firmware_elf_path(a_kernel_path, "idle_erisc")

    # Check if firmware elf paths exist
    if not os.path.exists(brisc_elf_path):
        raiseTTTriageError(f"BRISC ELF file {brisc_elf_path} does not exist.")

    if not os.path.exists(idle_erisc_elf_path):
        raiseTTTriageError(f"IDLE ERISC ELF file {idle_erisc_elf_path} does not exist.")

    # Parse firmware elfs
    brisc_elf = parse_elf(brisc_elf_path, context)
    idle_erisc_elf = parse_elf(idle_erisc_elf_path, context)

    # Check if debug info is obtained correctly
    if not brisc_elf:
        raiseTTTriageError(
            f"Failed to extract DWARF info from ELF file {brisc_elf_path}.\nRun workload with TT_METAL_RISCV_DEBUG_INFO=1 to enable debug info."
        )
        return

    if not idle_erisc_elf:
        raiseTTTriageError(
            f"Failed to extract DWARF info from ELF file {idle_erisc_elf_path}.\nRun workload with TT_METAL_RISCV_DEBUG_INFO=1 to enable debug info."
        )
        return

    # Acces the value of enumerator for supported blocks
    ProgrammableCoreTypes_TENSIX = brisc_elf.enumerators["ProgrammableCoreType::TENSIX"].value
    ProgrammableCoreTypes_IDLE_ETH = brisc_elf.enumerators["ProgrammableCoreType::IDLE_ETH"].value

    # Collect pcs
    pcs_tensix = collect_pcs_from_riscv(dev, blocks=dev.get_blocks(block_type="functional_workers"))
    pcs_idle_eth = collect_pcs_from_riscv(dev, blocks=dev.idle_eth_blocks)

    # Enumerators for tensix block
    enum_values_tenisx = {
        "ProcessorTypes": {
            "BRISC": brisc_elf.enumerators["TensixProcessorTypes::DM0"].value,
            "NCRISC": brisc_elf.enumerators["TensixProcessorTypes::DM1"].value,
            "TRISC0": brisc_elf.enumerators["TensixProcessorTypes::MATH0"].value,
            "TRISC1": brisc_elf.enumerators["TensixProcessorTypes::MATH1"].value,
            "TRISC2": brisc_elf.enumerators["TensixProcessorTypes::MATH2"].value,
        },
        "dispatch_core_processor_classes": {
            "BRISC": brisc_elf.enumerators["dispatch_core_processor_classes::DISPATCH_CLASS_TENSIX_DM0"].value,
            "NCRISC": brisc_elf.enumerators["dispatch_core_processor_classes::DISPATCH_CLASS_TENSIX_DM1"].value,
            "TRISC0": brisc_elf.enumerators["dispatch_core_processor_classes::DISPATCH_CLASS_TENSIX_COMPUTE"].value,
            "TRISC1": brisc_elf.enumerators["dispatch_core_processor_classes::DISPATCH_CLASS_TENSIX_COMPUTE"].value,
            "TRISC2": brisc_elf.enumerators["dispatch_core_processor_classes::DISPATCH_CLASS_TENSIX_COMPUTE"].value,
        },
    }

    # Enumerators for eth block
    enum_values_eth = {
        "ProcessorTypes": {
            "ERISC": idle_erisc_elf.enumerators["EthProcessorTypes::DM0"].value,
            "ERISC0": idle_erisc_elf.enumerators["EthProcessorTypes::DM0"].value,
        },
        "dispatch_core_processor_classes": {
            "ERISC": idle_erisc_elf.enumerators["dispatch_core_processor_classes::DISPATCH_CLASS_ETH_DM0"].value,
            "ERISC0": idle_erisc_elf.enumerators["dispatch_core_processor_classes::DISPATCH_CLASS_ETH_DM0"].value,
            "ERISC1": idle_erisc_elf.enumerators["dispatch_core_processor_classes::DISPATCH_CLASS_ETH_DM1"].value,
        },
    }

    if type(dev) == BlackholeDevice:
        enum_values_eth["ProcessorTypes"]["ERISC1"] = idle_erisc_elf.enumerators["EthProcessorTypes::DM1"].value

    # Getting running ops tables
    running_ops_table_tensix = get_running_ops_table(
        dev,
        dev.get_blocks(block_type="functional_workers"),
        enum_values_tenisx,
        inspector_data,
        ProgrammableCoreTypes_TENSIX,
        brisc_elf,
        pcs_tensix,
        a_kernel_path,
    )
    runinng_ops_table_idle_eth = get_running_ops_table(
        dev,
        dev.idle_eth_blocks,
        enum_values_eth,
        inspector_data,
        ProgrammableCoreTypes_IDLE_ETH,
        idle_erisc_elf,
        pcs_idle_eth,
        a_kernel_path,
    )

    # Printing tables if verbose is True
    if VERBOSE or VVERBOSE:
        print()  # Newline after the last PC
        if running_ops_table_tensix is not None:
            print(tabulate(running_ops_table_tensix, headers="firstrow", tablefmt=DEFAULT_TABLE_FORMAT))
        if runinng_ops_table_idle_eth is not None:
            print(tabulate(runinng_ops_table_idle_eth, headers="firstrow", tablefmt=DEFAULT_TABLE_FORMAT))

    # WIP:
    # # Print callstack for this location using tt_exalens_lib
    # fw_elf_path = a_kernel_path + "../../../firmware/trisc1/trisc1.elf"
    # fw_elf_path = os.path.realpath(fw_elf_path)
    # if not os.path.exists(fw_elf_path):
    #     raiseTTTriageError(f"FW ELF file {fw_elf_path} does not exist.")

    # if kernel_name:
    #     kernel_path = runtime_data['kernels'][watcher_kernel_id]['out'] + "trisc1/trisc1.elf"
    #     if not os.path.exists(kernel_path):
    #         raiseTTTriageError(f"Kernel ELF file {kernel_path} does not exist.")
    #     try:
    #         from ttexalens.tt_exalens_lib import callstack
    #         cs = callstack(loc, [fw_elf_path, kernel_path], [0, kernel_config_base + kernel_text_offset], 2, 100, True, False, dev.id(), context)
    #         if cs:
    #             print_callstack(cs)
    #         else:
    #             print(f"{ORANGE}No callstack available for location {loc.to_str('logical')}{RST}")
    #     except Exception as e:
    #         print(f"{RED}Error getting callstack: {e}{RST}")


def main(argv=None):
    """Main function that runs the triage script."""
    global VERBOSE, VVERBOSE, context, HALT_ON_ERROR

    args = docopt(__doc__, argv=argv)
    VERBOSE = args["--verbose"]
    VVERBOSE = args["--vverbose"]
    HALT_ON_ERROR = args["--halt-on-error"]

    context = init_ttexalens(use_noc1=False)
    device_ids = list(context.devices.keys())

    # Fetch inspector data
    inspector_log_directory = args["--inspector-log-path"]
    try:
        inspector_data = get_inspector_data(inspector_log_directory)
    except:
        inspector_data = None
        print(f"  {ORANGE}Inspector directory does not exist. Running tests that don't include it.{RST}")

    # Populate integer array with device ids
    if len(args["--dev"]) == 1 and args["--dev"][0].lower() == "in_use":
        if inspector_data is not None:
            device_ids = inspector_data.devices_in_use
            if len(device_ids) == 0:
                print(
                    f"{ORANGE}No devices in use found in inspector data. Switching to use all available devices. If you are using ttnn check if you have enabled program cache.{RST}"
                )
                device_ids = [int(id) for id in context.devices.keys()]
        else:
            print(f"{ORANGE}Inspector data not found. Using all available devices.{RST}")
            device_ids = [int(id) for id in context.devices.keys()]
    elif len(args["--dev"]) == 1 and args["--dev"][0].lower() == "all":
        device_ids = [int(id) for id in context.devices.keys()]
    else:
        device_ids = [int(id) for id in args["--dev"]]

    verbose(f"Device IDs: {device_ids}")

    try:
        for device_id in device_ids:
            title(f"Checking device {device_id}")
            dev = context.devices[device_id]
            # Check if dev is Wormhole or Blackhole
            if type(dev) == WormholeDevice or type(dev) == BlackholeDevice:
                check_ARC(dev)
                check_NOC(dev)
                check_L1(dev)
                check_riscV(dev)
                dump_running_ops(dev, inspector_data)
            else:
                raiseTTTriageError(f"{dev._arch} devices are not supported yet.")
    except TTTriageError as e:
        print(f"{RED}ERROR: {e}{RST}")
        if args["--halt-on-error"]:
            return 1

    print(f"{GREEN}DONE: OK{RST}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
