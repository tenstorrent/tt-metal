#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Usage:
    dump_callstacks [--full-callstack] [--gdb-callstack] [--active-eth] [--all-cores]

Options:
    --full-callstack   Dump full callstack with all frames. Defaults to dumping only the top frame.
    --gdb-callstack    Dump callstack using GDB client instead of built-in methods.
    --active-eth       Override default behaviour of not dumping callstack for active eth cores if full callstack or gdb callstack is used.
    --all-cores        Show all cores including ones with Go Message = DONE. By default, DONE cores are filtered out.

Description:
    Dumps callstacks for all devices in the system and for every supported risc processor.
    By default, filters out cores with DONE status and shows essential fields.
    Use --all-cores to see all cores, and -v/-vv to show more columns.

    Color output is automatically enabled when stdout is a TTY (terminal) and can be overridden
    with TT_TRIAGE_COLOR environment variable (0=disable, 1=enable).
"""

from dataclasses import dataclass

from triage import ScriptConfig, TTTriageError, log_check, recurse_field, triage_field, hex_serializer, run_script
from ttexalens import util
from ttexalens.hw.tensix.wormhole.wormhole import WormholeDevice
from dispatcher_data import run as get_dispatcher_data, DispatcherData, DispatcherCoreData
from elfs_cache import run as get_elfs_cache, ElfsCache
from run_checks import run as get_run_checks
from ttexalens.coordinate import OnChipCoordinate
from ttexalens.context import Context
from ttexalens.gdb.gdb_server import GdbServer, ServerSocket
from ttexalens.hardware.risc_debug import CallstackEntry, ParsedElfFile
from ttexalens.tt_exalens_lib import top_callstack, callstack
from utils import BLUE, GREEN, ORANGE, RED, RST

import re
import socket
import subprocess
import threading
from contextlib import closing
from pathlib import Path

script_config = ScriptConfig(
    depends=["run_checks", "dispatcher_data", "elfs_cache"],
)


def get_process_ids(gdb_server: GdbServer):
    process_ids: dict[OnChipCoordinate, dict[str, int]] = {}
    for pid, process in gdb_server.available_processes.items():
        location = process.risc_debug.risc_location.location
        risc_name = process.risc_debug.risc_location.risc_name

        if location in process_ids:
            process_ids[location][risc_name] = pid
        else:
            process_ids[location] = {risc_name: pid}

    return process_ids


def make_add_symbol_file_command(paths: list[str], offsets: list[int | None]) -> str:
    add_symbol_file_cmd = ""
    for path, offset in zip(paths, offsets):
        add_symbol_file_cmd += (
            f"\tadd-symbol-file {path} {offset}\n" if offset is not None else f"\tadd-symbol-file {path}\n"
        )

    # Removing first tab symbol and last new line symbol for prettier look of gdb script
    return add_symbol_file_cmd[1:-1]


def make_gdb_script(
    pid: int,
    elf_paths: list[str],
    offsets: list[int | None],
    port: int,
    start_callstack_label: str,
    end_callstack_label: str,
) -> str:
    return f"""\
        target extended-remote localhost:{port}
        set prompt
        attach {pid}
        {make_add_symbol_file_command(elf_paths, offsets)}
        printf "{start_callstack_label}\\n"
        backtrace
        printf "{end_callstack_label}\\n"
        detach
        quit
    """


def get_callstack_entry(line: str) -> CallstackEntry:
    pattern = re.compile(
        r"#\d+\s+"  # Skip the frame number
        r"(?:(?P<pc>0x[0-9a-fA-F]+)\s+in\s+)?"  # Capture pc if available
        r"(?P<function_name>\w+)"  # Capture function name
        r"(?:\s*\(.*?\))?"  # Ignore parentheses
        r"\s+at\s+"  # Skip at
        r"(?P<file_path>.*?):(?P<line>\d+)"  # Capture file path and line
    )

    entry = CallstackEntry()
    match = pattern.match(line)
    if match:
        entry.pc = int(match.groupdict()["pc"], 16) if match.groupdict()["pc"] is not None else None
        entry.function_name = match.groupdict()["function_name"]
        entry.file = match.groupdict()["file_path"]
        entry.line = int(match.groupdict()["line"]) if match.groupdict()["line"] is not None else None

    return entry


def extract_callstack_from_gdb_output(
    gdb_output: str, start_callstack_label: str, end_callstack_label
) -> list[CallstackEntry]:
    cs: list[CallstackEntry] = []
    is_cs = False

    current_line = ""
    for line in gdb_output.splitlines():
        if start_callstack_label in line:
            is_cs = True
        elif end_callstack_label in line:
            is_cs = False
        else:
            if is_cs:
                if "#" in line:
                    if current_line != "":
                        cs.append(get_callstack_entry(current_line))
                        current_line = ""

                    current_line += line.strip()
                # Ensure every callstack entry is in one line
                else:
                    current_line += " "
                    current_line += line.strip()

    # We implicitly skip last line since it is message not callstack entry
    return cs


# Class for storing callstack and message that should be displayed together
@dataclass
class KernelCallstackWithMessage:
    callstack: list[CallstackEntry]
    message: str | None


def get_gdb_callstack(
    location: OnChipCoordinate, risc_name: str, dispatcher_core_data: DispatcherCoreData, port: int, process_ids: dict
) -> KernelCallstackWithMessage:
    # Start GDB client
    gdb_client = subprocess.Popen(
        ["tt-exalens", "--gdb"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Labels for determining where backtrace output is
    start_callstack_label = "START CALLSTACK"
    end_callstack_label = "END CALLSTACK"

    elf_paths: list[str] = [dispatcher_core_data.firmware_path]
    offsets: list[int | None] = [None]
    if dispatcher_core_data.kernel_path is not None:
        elf_paths.append(dispatcher_core_data.kernel_path)
        offsets.append(dispatcher_core_data.kernel_offset)

    gdb_script = make_gdb_script(
        process_ids[location][risc_name], elf_paths, offsets, port, start_callstack_label, end_callstack_label
    )

    # Run gdb script
    if gdb_client.stdin is not None:
        gdb_client.stdin.write(gdb_script)
        gdb_client.stdin.flush()

    callstack = extract_callstack_from_gdb_output(
        gdb_client.communicate()[0], start_callstack_label, end_callstack_label
    )
    return KernelCallstackWithMessage(callstack=callstack, message=None)


def get_callstack(
    location: OnChipCoordinate,
    risc_name: str,
    dispatcher_core_data: DispatcherCoreData,
    elfs_cache: ElfsCache,
    full_callstack: bool,
) -> KernelCallstackWithMessage:
    context = location._device._context
    elfs: list[ParsedElfFile] = [elfs_cache[dispatcher_core_data.firmware_path]]
    offsets: list[int | None] = [None]
    if dispatcher_core_data.kernel_path is not None:
        elfs.append(elfs_cache[dispatcher_core_data.kernel_path])
        offsets.append(dispatcher_core_data.kernel_offset)
    try:
        if not full_callstack:
            pc = location._device.get_block(location).get_risc_debug(risc_name).get_pc()
            try:
                cs = top_callstack(pc, elfs, offsets, context)
                error_message = None
                if len(cs) == 0:
                    error_message = "PC was not in range of any provided ELF files."
                    if location in location._device.active_eth_block_locations:
                        error_message += " Probably context switch occurred and PC is contained in base ERISC firmware."
                return KernelCallstackWithMessage(callstack=cs, message=error_message)
            except Exception as e:
                return KernelCallstackWithMessage(callstack=[], message=str(e))
        else:
            try:
                cs = callstack(location, elfs, offsets, risc_name)
                error_message = None
                if len(cs) == 0:
                    error_message = "PC was not in range of any provided ELF files."
                    if location in location._device.active_eth_block_locations:
                        error_message += " Probably context switch occurred and PC is contained in base ERISC firmware."
                return KernelCallstackWithMessage(callstack=cs, message=error_message)
            except Exception as e:
                try:
                    # If full callstack failed, we default to top callstack
                    pc = location._device.get_block(location).get_risc_debug(risc_name).get_pc()
                    error_message = str(e) + " - defaulting to top callstack"
                    cs = top_callstack(pc, elfs, offsets, context)
                    if len(cs) == 0:
                        additional_message = "PC was not in range of any provided ELF files."
                        if location in location._device.active_eth_block_locations:
                            additional_message += (
                                " Probably context switch occurred and PC is contained in base ERISC firmware."
                            )
                        error_message = "\n".join([error_message, additional_message])
                    return KernelCallstackWithMessage(callstack=cs, message=error_message)
                except Exception as e:
                    # If top callstack failed too, print both error messages
                    return KernelCallstackWithMessage(callstack=[], message="\n".join([error_message, str(e)]))
    except Exception as e:
        return KernelCallstackWithMessage(callstack=[], message=str(e))


def _format_callstack(callstack: list[CallstackEntry]) -> list[str]:
    """Return string representation of the callstack."""
    frame_number_width = len(str(len(callstack) - 1))
    result = []
    cwd = Path.cwd()

    for i, frame in enumerate(callstack):
        line = f"  #{i:<{frame_number_width}} "
        if frame.pc is not None:
            line += f"{BLUE}0x{frame.pc:08X}{RST} in "
        if frame.function_name is not None:
            line += f"{ORANGE}{frame.function_name}{RST} () "
        if frame.file is not None:
            # Convert absolute path to relative path with ./ prefix
            file_path = Path(frame.file)
            try:
                if file_path.is_absolute():
                    rel_path = file_path.relative_to(cwd)
                    display_path = f"./{rel_path}"
                else:
                    display_path = frame.file
            except ValueError:
                # Path is not relative to cwd, keep as is
                display_path = frame.file

            line += f"at {GREEN}{display_path}{RST}"
            if frame.line is not None:
                line += f" {GREEN}{frame.line}{RST}"
                if frame.column is not None:
                    line += f"{GREEN}:{frame.column}{RST}"
        result.append(line)
    return result


def format_callstack_with_message(callstack_with_message: KernelCallstackWithMessage) -> str:
    """Return string representation of the callstack with optional error message. Adding empty line at the beginning for prettier look."""
    empty_line = ""  # For prettier look

    if callstack_with_message.message is not None:
        return "\n".join(
            [f"{RED}{callstack_with_message.message}{RST}"] + _format_callstack(callstack_with_message.callstack)
        )
    else:
        return "\n".join([empty_line] + _format_callstack(callstack_with_message.callstack))


@dataclass
class DumpCallstacksData:
    """Callstack data with verbosity levels.

    Level 0: Essential fields (Kernel ID:Name, Go Message, Subdevice, Preload, Waypoint, PC, Callstack)
    Level 1+: Includes all dispatcher core data fields
    """

    # Recurse into dispatcher core data (fields have their own verbose levels)
    dispatcher_core_data: DispatcherCoreData = recurse_field()

    # Always show PC and callstack
    pc: int | None = triage_field("PC", hex_serializer)
    kernel_callstack_with_message: KernelCallstackWithMessage = triage_field(
        "Kernel Callstack", format_callstack_with_message
    )


def dump_callstacks(
    location: OnChipCoordinate,
    risc_name: str,
    dispatcher_data: DispatcherData,
    elfs_cache: ElfsCache,
    full_callstack: bool,
    gdb_callstack: bool,
    gdb_server: GdbServer | None,
    process_ids: dict[OnChipCoordinate, dict[str, int]] | None,
    show_all_cores: bool = False,
    force_active_eth: bool = False,
) -> DumpCallstacksData | None:
    result: DumpCallstacksData | None = None

    try:
        dispatcher_core_data = dispatcher_data.get_core_data(location, risc_name)

        # Skip DONE cores unless --all-cores is specified
        if not show_all_cores and dispatcher_core_data.go_message == "DONE":
            return result

        risc_debug = location._device.get_block(location).get_risc_debug(risc_name)
        if risc_debug.is_in_reset():
            return DumpCallstacksData(
                dispatcher_core_data=dispatcher_core_data,
                pc=None,
                kernel_callstack_with_message=KernelCallstackWithMessage(callstack=[], message="Core is in reset"),
            )
        if location in location._device.active_eth_block_locations and not force_active_eth:
            callstack_with_message = get_callstack(
                location, risc_name, dispatcher_core_data, elfs_cache, full_callstack=False
            )
        else:
            if gdb_callstack:
                if risc_name == "ncrisc":
                    # Cannot attach to NCRISC process due to lack of debug hardware so we are defaulting to top callstack
                    error_message = (
                        "Cannot attach to NCRISC process due to lack of debug hardware - defaulting to top callstack"
                    )
                    # Default to top callstack
                    callstack_with_message = get_callstack(
                        location, risc_name, dispatcher_core_data, elfs_cache, full_callstack=False
                    )
                    # If top callstack failed too, print both error messages
                    callstack_with_message.message = (
                        error_message
                        if callstack_with_message.message is None
                        else "\n".join([error_message, callstack_with_message.message])
                    )
                else:
                    assert gdb_server is not None and process_ids is not None
                    callstack_with_message = get_gdb_callstack(
                        location, risc_name, dispatcher_core_data, gdb_server.server.port, process_ids
                    )
                    # If GDB failed to get callstack, we default to top callstack
                    if len(callstack_with_message.callstack) == 0:
                        error_message = "Failed to get callstack from GDB. Look for error message above the table."
                        callstack_with_message = get_callstack(
                            location, risc_name, dispatcher_core_data, elfs_cache, full_callstack=False
                        )
                        # If top callstack failed too, print both error messages
                        callstack_with_message.message = (
                            error_message
                            if callstack_with_message.message is None
                            else "\n".join([error_message, callstack_with_message.message])
                        )

                # If GDB has not recoreded PC we do that ourselves, this also provides PC for NCRISC case
                if len(callstack_with_message.callstack) > 0 and callstack_with_message.callstack[0].pc is None:
                    try:
                        callstack_with_message.callstack[0].pc = (
                            location._device.get_block(location).get_risc_debug(risc_name).get_pc()
                        )
                    except:
                        pass
            else:
                callstack_with_message = get_callstack(
                    location, risc_name, dispatcher_core_data, elfs_cache, full_callstack
                )

        # Create result with dispatcher core data (verbose levels handled in serialization)
        result = DumpCallstacksData(
            dispatcher_core_data=dispatcher_core_data,
            pc=callstack_with_message.callstack[0].pc
            if len(callstack_with_message.callstack) > 0
            else location._device.get_block(location).get_risc_debug(risc_name).get_pc(),
            kernel_callstack_with_message=callstack_with_message,
        )

    except Exception as e:
        log_check(
            False,
            f"{ORANGE}Failed to dump callstacks for {risc_name} at {location} on device {location._device._id}: {e}{RST}",
        )
        return result

    return result


# Global lock for thread-safe port finding
_port_lock = threading.Lock()


def find_available_port() -> int:
    """
    Find an available port for gdb_server in a thread-safe manner.
    Returns:
        An available port number
    """
    try:
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.bind(("", 0))  # 0 → OS picks a free port
            s.listen()
            return s.getsockname()[1]
    except (socket.error, OSError) as e:
        # If we get here, no port was found
        raise TTTriageError(f"No available port found: {e}")


def start_gdb_server(port: int, context: Context) -> GdbServer:
    """Start GDB server and return it."""
    try:
        server = ServerSocket(port)
        server.start()
        gdb_server = GdbServer(context, server)
        gdb_server.start()
    except Exception as e:
        raise TTTriageError(f"Failed to start GDB server on port {port}. Error: {e}")

    return gdb_server


def run(args, context: Context):
    from triage import set_verbose_level

    full_callstack: bool = args["--full-callstack"]
    gdb_callstack: bool = args["--gdb-callstack"]
    show_all_cores: bool = args["--all-cores"]
    active_eth: bool = args["--active-eth"]

    # Set verbose level from -v count (controls which columns are displayed)
    verbose_level = args["-v"]
    set_verbose_level(verbose_level)

    BLOCK_TYPES_TO_CHECK = ["tensix", "idle_eth", "active_eth"]
    # We are skipping active eth cores if full callstack or gdb callstack is used by default, this can be overridden with --active-eth
    force_active_eth = (full_callstack or gdb_callstack) and active_eth
    if force_active_eth:
        util.WARN(
            "Getting full or gdb callstack may break active eth core. Use tt-smi reset to fix. See issue #661 in tt-exalens for more details."
        )

    elfs_cache = get_elfs_cache(args, context)
    run_checks = get_run_checks(args, context)
    dispatcher_data = get_dispatcher_data(args, context)

    gdb_server: GdbServer | None = None
    process_ids: dict[OnChipCoordinate, dict[str, int]] | None = None
    if gdb_callstack:
        # Locking thread until we start gdb server on available port
        with _port_lock:
            port = find_available_port()
            gdb_server = start_gdb_server(port, context)
        process_ids = get_process_ids(gdb_server)

    callstacks_data = run_checks.run_per_core_check(
        lambda location, risc_name: dump_callstacks(
            location,
            risc_name,
            dispatcher_data,
            elfs_cache,
            full_callstack,
            gdb_callstack,
            gdb_server,
            process_ids,
            show_all_cores,
            force_active_eth,
        ),
        block_filter=BLOCK_TYPES_TO_CHECK,
    )

    # After all callstacks are dumped, stop GDB server if it was started
    if gdb_server is not None:
        gdb_server.stop()

    return callstacks_data


if __name__ == "__main__":
    run_script()
