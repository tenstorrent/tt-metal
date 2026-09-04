#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Usage:
    debug_risc [--risc=<risc_name>] [--neo=<neo_id>] [--gdb-port=<port>] [--gdb-command=<command>]...

Options:
    --risc=<risc_name>       RISC core to attach to (brisc, trisc0, trisc1, trisc2, erisc, erisc0,
                             erisc1, drisc). Required - if it is omitted, the names available on the
                             selected core are listed.
    --neo=<neo_id>           NEO id of the RISC core, for blocks that have more than one NEO.
    --gdb-port=<port>        Port the GDB server listens on. Default: a free port picked by the OS.
    --gdb-command=<command>  GDB command to execute after attaching. Repeatable, executed in order.

Description:
    Starts a GDB server for a single RISC core, attaches a GDB client to it and hands the terminal
    over to that client. Symbols for the firmware and for the kernel currently loaded on the core are
    added automatically, taken from the same dispatcher data that dump_callstacks uses.

    The core is selected with the run_checks options (--dev / --loc) plus --risc. --loc takes a
    single location - only one core can be debugged at a time:

        tools/triage/debug_risc.py --loc=1,1 --risc=brisc
        tools/triage/debug_risc.py --dev=0 --loc=e0,0 --risc=erisc0 --gdb-command=backtrace

    Attaching halts the core and detaching from it (GDB 'detach' or 'quit') resumes it. A core left
    halted keeps the program running on it stopped, so this script warns if GDB leaves it that way.

    When the GDB client exits, so does this script - the session was the output, so there is no
    triage result to report.

    This script is disabled for regular tt-triage runs - it is interactive and would block them
    forever. Because tt-triage does not register the options of a disabled script, it also has to be
    started directly (as above) rather than through 'tt-triage.py --run=debug_risc'.

Owner:
    tt-vjovanovic
"""

import os
import subprocess
import sys
import traceback

import utils
from dispatcher_data import run as get_dispatcher_data, DispatcherData
from run_checks import run as get_run_checks, RunChecks, BLOCK_TYPES
from triage import ScriptConfig, TTTriageError, run_script
from ttexalens.context import Context
from ttexalens.coordinate import OnChipCoordinate
from ttexalens.device import Device
from ttexalens.exceptions import TTException
from ttexalens.gdb.gdb_client import get_gdb_client_path
from ttexalens.gdb.gdb_data import GdbProcess
from ttexalens.gdb.gdb_server import GdbServer, ServerSocket
from ttexalens.hardware.risc_debug import RiscDebug, RiscLocation

script_config = ScriptConfig(
    depends=["run_checks", "dispatcher_data"],
    # Interactive script - it hands the terminal over to a GDB client and waits for a human, so it
    # must never take part in a regular tt-triage run. It is started directly instead.
    disabled=True,
)


def find_device(run_checks: RunChecks) -> Device:
    """Return the single selected device, or explain how to narrow the selection down to one."""
    devices = run_checks.devices
    if len(devices) == 0:
        raise TTTriageError("No device selected. Use --dev=<device_id> to select the device to debug.")
    if len(devices) > 1:
        device_ids = ", ".join(str(device.id) for device in devices)
        raise TTTriageError(
            f"{len(devices)} devices selected ({device_ids}), but only one core can be debugged at a time. "
            f"Use --dev=<device_id> to select one."
        )
    return devices[0]


def find_location(run_checks: RunChecks, device: Device, requested_locations: list[str]) -> OnChipCoordinate:
    """Return the single requested location, checking that it is a block of the selected device."""
    if len(requested_locations) != 1:
        raise TTTriageError(
            "Use --loc=<location> exactly once to select the core to debug - only one core can be debugged at "
            "a time. Logical coordinates only: R,C (tensix), eX,Y (eth), dX,Y / CHn (dram)."
        )

    # run_checks has already parsed the location and narrowed its block lists down to it, so the
    # first location it still reports is the requested one. It can be reported under more than one
    # block type (eth is also active/idle eth), which makes no difference here.
    for block_type in BLOCK_TYPES:
        locations = run_checks.block_locations[device].get(block_type, [])
        if len(locations) > 0:
            return locations[0]
    raise TTTriageError(f"{requested_locations[0]} is not a block of device {device.id}.")


def find_risc_debug(location: OnChipCoordinate, risc_name: str | None, neo_id: int | None) -> RiscDebug:
    """Return the RiscDebug of the requested core, checking that it can actually be debugged."""
    noc_block = location.device.get_block(location)
    available = ", ".join(noc_block.risc_names)
    if risc_name is None:
        raise TTTriageError(
            f"Use --risc=<risc_name> to select the core to debug. Available on this block: {available}."
        )
    if risc_name not in noc_block.risc_names:
        raise TTTriageError(f"{location.to_user_str()} has no {risc_name}. Available on this block: {available}.")

    risc_debug = noc_block.get_risc_debug(risc_name, neo_id)
    if not risc_debug.can_debug():
        raise TTTriageError(
            f"{risc_name} on {location.to_user_str()} has no debug hardware, so GDB cannot attach to it. "
            f"Use dump_callstacks to get its top callstack instead."
        )
    if risc_debug.is_in_reset():
        raise TTTriageError(f"{risc_name} on {location.to_user_str()} is in reset.")
    return risc_debug


def get_elfs(dispatcher_data: DispatcherData, location: OnChipCoordinate, risc_name: str):
    """Return (firmware path, kernel path, kernel offset) of the program loaded on the core."""
    try:
        core_data = dispatcher_data.get_cached_core_data(location, risc_name)
    except Exception as e:
        raise TTTriageError(
            f"Could not read dispatcher data for {risc_name} on {location.to_user_str()}: {e}. "
            f"The firmware ELF is needed to attach, since the GDB server only serves cores it knows an ELF for."
        )
    return core_data.firmware_path, core_data.kernel_path, core_data.kernel_offset


def start_gdb_server(context: Context, port: int | None) -> GdbServer:
    """Start the GDB server, on the given port or on a free port picked by the OS."""
    try:
        # ServerSocket picks (and binds) a free port under a lock when no port is given, so nothing
        # else can take it in between.
        server = ServerSocket(port)
        server.start()
        # debug_only_with_elfs limits the served processes to the cores we registered an ELF for,
        # which is exactly the one core we are debugging. It also keeps the process list cheap - the
        # alternative walks and probes every debuggable core of every device.
        gdb_server = GdbServer(context, server, debug_only_with_elfs=True)
        gdb_server.start()
    except Exception as e:
        where = "a free port" if port is None else f"port {port}"
        raise TTTriageError(f"Failed to start GDB server on {where}. Error: {e}")
    return gdb_server


def find_process_id(gdb_server: GdbServer, risc_location: RiscLocation) -> int:
    """Return the id of the GDB server process that represents the given core."""
    available_processes: dict[int, GdbProcess] = gdb_server.available_processes
    for process_id, process in available_processes.items():
        if process.risc_debug.risc_location == risc_location:
            return process_id
    raise TTTriageError(f"GDB server does not serve a process for {risc_location}.")


def make_gdb_client_command(
    port: int,
    process_id: int,
    elf_paths: list[str],
    offsets: list[int | None],
    extra_commands: list[str],
) -> list[str]:
    """Build the command line of a GDB client that is attached to the core and left to the user."""
    commands = [
        # Symbols are added below for files GDB already knows about, and quitting detaches from a
        # process we attached to; neither needs to be confirmed by the user.
        "set confirm off",
        f"target extended-remote localhost:{port}",
        f"attach {process_id}",
    ]
    for path, offset in zip(elf_paths, offsets):
        commands.append(f"add-symbol-file {path}" if offset is None else f"add-symbol-file {path} {offset}")
    commands.append("set confirm on")
    commands.extend(extra_commands)

    argv = [get_gdb_client_path(), "-q"]
    for command in commands:
        argv += ["-ex", command]
    return argv


def exit_process(exit_code: int):
    """Leave the process without interpreter teardown, which keeps the nanobind leak check quiet."""
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)


def run_gdb_client(argv: list[str]) -> int:
    """Run the GDB client on our terminal and return its exit code."""
    gdb_client = subprocess.Popen(argv)
    while True:
        try:
            exit_code = gdb_client.wait()
            # wait() reports a signal as -signum, which is not a usable exit code.
            return exit_code if exit_code >= 0 else 128 - exit_code
        except KeyboardInterrupt:
            # Ctrl-C is meant for the GDB client, which uses it to interrupt the core. We share the
            # terminal with it, so we just keep waiting.
            continue


def run(args, context: Context):
    risc_name: str | None = args["--risc"]
    gdb_commands: list[str] = args["--gdb-command"] or []
    try:
        neo_id = int(args["--neo"]) if args["--neo"] is not None else None
        gdb_port = int(args["--gdb-port"]) if args["--gdb-port"] is not None else None
    except ValueError as e:
        raise TTTriageError(f"Invalid argument: {e}")

    run_checks = get_run_checks(args, context)
    dispatcher_data = get_dispatcher_data(args, context)

    device = find_device(run_checks)
    location = find_location(run_checks, device, args["--loc"] or [])
    risc_debug = find_risc_debug(location, risc_name, neo_id)
    assert risc_name is not None  # find_risc_debug raises if it is not given

    firmware_path, kernel_path, kernel_offset = get_elfs(dispatcher_data, location, risc_name)
    elf_paths: list[str] = [firmware_path]
    offsets: list[int | None] = [None]
    if kernel_path is not None:
        elf_paths.append(kernel_path)
        offsets.append(kernel_offset)
    else:
        utils.WARN("  No kernel is loaded on this core, only firmware symbols will be available.")

    # The GDB server reports a core as a debuggable process only if it knows which ELF runs on it,
    # and it serves that path to GDB as the executable of the process.
    context.elf_loaded(risc_debug.risc_location, firmware_path)

    gdb_server = start_gdb_server(context, gdb_port)
    try:
        port = gdb_server.server.port
        assert port is not None
        process_id = find_process_id(gdb_server, risc_debug.risc_location)
        argv = make_gdb_client_command(port, process_id, elf_paths, offsets, gdb_commands)

        utils.INFO(f"  GDB server listening on localhost:{port}")
        utils.INFO(
            f"  Attaching to process {process_id}: {risc_name} on {location.to_user_str()} of device {device.id}"
        )
        utils.INFO("  The core is halted while GDB is attached. Detach or quit GDB to resume it.")
        exit_code = run_gdb_client(argv)
    finally:
        gdb_server.stop()

    # The session was the output, so there is nothing for the triage framework to report. Leave with
    # the exit code of the GDB client, before run_script gets to print its result table.
    exit_process(exit_code)


if __name__ == "__main__":
    try:
        run_script()
    except (TTTriageError, TTException) as e:
        # This script always runs standalone, so nothing above us turns a failure into a triage
        # report. Both of these carry a message that says what to do about it (a bad argument, a
        # core that cannot be debugged, no workload to look at), and ttexalens installs a
        # sys.excepthook that would bury that message under a tabulated traceback. Hardware errors
        # are not caught here - they do not derive from Exception, and their traceback is the point.
        utils.ERROR(f"{e}")
        if utils.Verbosity.supports(utils.Verbosity.DEBUG):
            traceback.print_exc()
        exit_process(1)
