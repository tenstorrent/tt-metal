#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Usage:
    callstack_provider [--full-callstack] [--gdb-callstack] [--active-eth]

Options:
    --full-callstack   Dump full callstack with all frames. Defaults to dumping only the top frame.
    --gdb-callstack    Dump callstack using GDB client instead of built-in methods.
    --active-eth       Override default behaviour of not dumping callstack for active eth cores if full callstack or gdb callstack is used.

Description:
    Provides callstack extraction functionality for RISC cores on devices.

Owner:
    adjordjevic-TT
"""

from dataclasses import dataclass
import io

from triage import (
    ScriptConfig,
    TTTriageError,
    log_warning,
    recurse_field,
    triage_field,
    hex_serializer,
    run_script,
    triage_singleton,
)
from dispatcher_data import run as get_dispatcher_data, DispatcherData, DispatcherCoreData
from elfs_cache import run as get_elfs_cache, ElfsCache
from ttexalens.coordinate import OnChipCoordinate
from ttexalens.context import Context
from ttexalens.gdb.gdb_server import GdbServer, ServerSocket
from ttexalens.gdb.gdb_client import get_gdb_callstack
from ttexalens.elf import ElfFile
from ttexalens.elf import CallstackEntry
from ttexalens.tt_exalens_lib import top_callstack, callstack
from ttexalens.umd_device import TimeoutDeviceRegisterError
from utils import WARN

import socket
import threading
from contextlib import closing
from pathlib import Path

script_config = ScriptConfig(
    data_provider=True,
    depends=["dispatcher_data", "elfs_cache"],
)


# Class for storing callstack and message that should be displayed together
@dataclass
class KernelCallstackWithMessage:
    callstack: list[CallstackEntry]
    message: str | None


@dataclass
class KernelArgsDump:
    """Raw per-core kernel args for one (core, risc): compile-time (Inspector) + runtime (live device
    L1). Values only — a consumer-side script maps names/indices to semantics."""

    named_cta: list[tuple[str, int]]
    positional_cta: list[int]
    unique_rta: list[int]
    common_rta: list[int]
    watcher_count_word: bool


def format_kernel_args(kernel_args: "KernelArgsDump | None") -> str:
    """Render kernel args as raw hex, grouped by kind; strips the watcher count-word from RTA."""
    if kernel_args is None:
        return "N/A"
    # No brackets around value lists: the triage renderer parses [..] as rich markup and would eat them.
    lines: list[str] = []
    if kernel_args.named_cta:
        lines.append("CTA named: " + ", ".join(f"{n}={v:#x}" for n, v in kernel_args.named_cta))
    if kernel_args.positional_cta:
        lines.append("CTA pos: " + ", ".join(f"{v:#x}" for v in kernel_args.positional_cta))
    strip = kernel_args.watcher_count_word
    unique = kernel_args.unique_rta[1:] if (strip and kernel_args.unique_rta) else kernel_args.unique_rta
    common = kernel_args.common_rta[1:] if (strip and kernel_args.common_rta) else kernel_args.common_rta
    if unique:
        lines.append("RTA core: " + ", ".join(f"{v:#x}" for v in unique))
    if common:
        lines.append("RTA common: " + ", ".join(f"{v:#x}" for v in common))
    return "\n".join(lines) if lines else "N/A"


def _build_kernel_args(
    dispatcher_data: DispatcherData,
    location: OnChipCoordinate,
    risc_name: str,
    dispatcher_core_data: DispatcherCoreData,
) -> KernelArgsDump | None:
    """Collect a core's compile-time args (Inspector) + runtime args (live device L1). None if empty."""
    try:
        kernel = dispatcher_data.find_kernel(dispatcher_core_data.watcher_kernel_id)
    except Exception:
        kernel = None
    named = [(a.name, int(a.value)) for a in (getattr(kernel, "namedCompileTimeArgs", None) or [])] if kernel else []
    positional = [int(v) for v in (getattr(kernel, "compileTimeArgs", None) or [])] if kernel else []

    rta = dispatcher_data.read_runtime_args(location, risc_name)
    unique = rta.unique if rta else []
    common = rta.common if rta else []
    watcher = rta.watcher_count_word if rta else False

    if not (named or positional or unique or common):
        return None
    return KernelArgsDump(named, positional, unique, common, watcher)


def _pc_not_in_range_message(dispatcher_core_data: DispatcherCoreData) -> str:
    msg = "PC was not in range of any provided ELF files."
    if dispatcher_core_data.block_type == "active_eth":
        msg += " Probably context switch occurred and PC is contained in base ERISC firmware."
    if dispatcher_core_data.kernel_lookup_warning:
        msg += "\n" + dispatcher_core_data.kernel_lookup_warning
    return msg


def get_callstack(
    location: OnChipCoordinate,
    risc_name: str,
    dispatcher_core_data: DispatcherCoreData,
    elfs_cache: ElfsCache,
    full_callstack: bool,
    rewind_pc_for_ebreak: bool,
) -> KernelCallstackWithMessage:
    context = location.device._context
    elfs: list[ElfFile] = [elfs_cache[dispatcher_core_data.firmware_path]]
    offsets: list[int | None] = [None]
    if dispatcher_core_data.kernel_path is not None:
        elfs.append(elfs_cache[dispatcher_core_data.kernel_path])
        offsets.append(dispatcher_core_data.kernel_offset)
    try:
        if not full_callstack:
            pc = location.device.get_block(location).get_risc_debug(risc_name).get_pc()
            if rewind_pc_for_ebreak:
                pc = pc - 4
            try:
                cs = top_callstack(pc, elfs, offsets, context)
                error_message = None
                if len(cs) == 0:
                    error_message = _pc_not_in_range_message(dispatcher_core_data)
                return KernelCallstackWithMessage(callstack=cs, message=error_message)
            except TimeoutDeviceRegisterError:
                raise
            except Exception as e:
                return KernelCallstackWithMessage(callstack=[], message=str(e))
        else:
            try:
                cs = callstack(location, elfs, offsets, risc_name)
                error_message = None
                if len(cs) == 0:
                    error_message = _pc_not_in_range_message(dispatcher_core_data)
                return KernelCallstackWithMessage(callstack=cs, message=error_message)
            except TimeoutDeviceRegisterError:
                raise
            except Exception as e:
                error_message = str(e) + " - defaulting to top callstack"
                try:
                    # If full callstack failed, we default to top callstack
                    pc = location.device.get_block(location).get_risc_debug(risc_name).get_pc()
                    if rewind_pc_for_ebreak:
                        pc = pc - 4
                    cs = top_callstack(pc, elfs, offsets, context)
                    if len(cs) == 0:
                        error_message = "\n".join([error_message, _pc_not_in_range_message(dispatcher_core_data)])
                    return KernelCallstackWithMessage(callstack=cs, message=error_message)
                except TimeoutDeviceRegisterError:
                    raise
                except Exception as e:
                    # If top callstack failed too, print both error messages
                    return KernelCallstackWithMessage(callstack=[], message="\n".join([error_message, str(e)]))
    except TimeoutDeviceRegisterError:
        raise
    except Exception as e:
        return KernelCallstackWithMessage(callstack=[], message=str(e))


def _format_callstack(callstack: list[CallstackEntry]) -> list[str]:
    """Return string representation of the callstack."""
    frame_number_width = len(str(len(callstack) - 1))
    result = []
    cwd = Path.cwd()

    for i, frame in enumerate(callstack):
        line = f"#{i:<{frame_number_width}} "
        if frame.pc is not None:
            line += f"[blue]0x{frame.pc:08X}[/] in "
        if frame.function_name is not None:
            line += f"[yellow]{frame.function_name}[/] () "
        if frame.file_info is not None:
            fi = frame.file_info
            # Convert absolute path to relative path with ./ prefix
            file_path = Path(fi.file)
            try:
                if file_path.is_absolute():
                    rel_path = file_path.relative_to(cwd)
                    display_path = f"./{rel_path}"
                else:
                    display_path = fi.file
            except ValueError:
                # Path is not relative to cwd, keep as is
                display_path = fi.file

            line += f"at [green]{display_path}[/]"
            line += f" [green]{fi.line}[/]"
            line += f"[green]:{fi.column}[/]"
        result.append(line)
    return result


def format_callstack_with_message(callstack_with_message: KernelCallstackWithMessage) -> str:
    """Return string representation of the callstack with optional error message. Adding empty line at the beginning for prettier look."""
    empty_line = ""  # For prettier look

    if callstack_with_message.message is not None:
        return "\n".join(
            [f"[error]{callstack_with_message.message}[/]"] + _format_callstack(callstack_with_message.callstack)
        )
    else:
        return "\n".join([empty_line] + _format_callstack(callstack_with_message.callstack))


@dataclass
class CallstacksData:
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
    kernel_args: KernelArgsDump | None = triage_field("Kernel Args", format_kernel_args)


class CallstackProvider:
    def __init__(
        self,
        dispatcher_data: DispatcherData,
        elfs_cache: ElfsCache,
        full_callstack: bool,
        gdb_callstack: bool,
        gdb_server: GdbServer | None,
        force_active_eth: bool = False,
    ):
        self.dispatcher_data = dispatcher_data
        self.elfs_cache = elfs_cache
        self.full_callstack = full_callstack
        self.gdb_callstack = gdb_callstack
        self.gdb_server = gdb_server
        self.force_active_eth = force_active_eth
        self._callstack_cache: dict[tuple, CallstacksData] = {}
        self.lock = threading.Lock()  # For thread-safe cache access

    def __del__(self):
        # After all callstacks are dumped, stop GDB server if it was started
        if self.gdb_server is not None:
            self.gdb_server.stop()

    def get_cached_callstacks(
        self,
        location: OnChipCoordinate,
        risc_name: str,
        rewind_pc_for_ebreak: bool = False,
        use_full_callstack: bool | None = None,
        use_gdb_callstack: bool | None = None,
    ) -> CallstacksData:
        full = use_full_callstack if use_full_callstack is not None else self.full_callstack
        gdb = use_gdb_callstack if use_gdb_callstack is not None else self.gdb_callstack

        cache_key = (
            location.device.id,
            location.to_str("noc0"),
            risc_name,
            full,
            gdb,
            rewind_pc_for_ebreak,
        )

        with self.lock:
            if cache_key in self._callstack_cache:
                return self._callstack_cache[cache_key]

        callstacks = self.get_callstacks(
            location,
            risc_name,
            rewind_pc_for_ebreak=rewind_pc_for_ebreak,
            use_full_callstack=use_full_callstack,
            use_gdb_callstack=use_gdb_callstack,
        )

        with self.lock:
            self._callstack_cache[cache_key] = callstacks

        return callstacks

    def get_callstacks(
        self,
        location: OnChipCoordinate,
        risc_name: str,
        rewind_pc_for_ebreak: bool = False,
        use_full_callstack: bool | None = None,
        use_gdb_callstack: bool | None = None,
    ) -> CallstacksData:
        dispatcher_core_data = self.dispatcher_data.get_cached_core_data(location, risc_name)
        risc_debug = location.noc_block.get_risc_debug(risc_name)

        if risc_debug.is_in_reset():
            return CallstacksData(
                dispatcher_core_data=dispatcher_core_data,
                pc=None,
                kernel_callstack_with_message=KernelCallstackWithMessage(callstack=[], message="Core is in reset"),
                kernel_args=None,
            )

        if dispatcher_core_data.block_type == "active_eth" and not self.force_active_eth:
            callstack_with_message = get_callstack(
                location,
                risc_name,
                dispatcher_core_data,
                self.elfs_cache,
                full_callstack=False,
                rewind_pc_for_ebreak=rewind_pc_for_ebreak,
            )
        else:
            if use_gdb_callstack or (use_gdb_callstack is None and self.gdb_callstack):
                if risc_name == "ncrisc":
                    # Cannot attach to NCRISC process due to lack of debug hardware so we are defaulting to top callstack
                    error_message = (
                        "Cannot attach to NCRISC process due to lack of debug hardware - defaulting to top callstack"
                    )
                    # Default to top callstack
                    callstack_with_message = get_callstack(
                        location,
                        risc_name,
                        dispatcher_core_data,
                        self.elfs_cache,
                        full_callstack=False,
                        rewind_pc_for_ebreak=rewind_pc_for_ebreak,
                    )
                    # If top callstack failed too, print both error messages
                    callstack_with_message.message = (
                        error_message
                        if callstack_with_message.message is None
                        else "\n".join([error_message, callstack_with_message.message])
                    )
                else:
                    assert self.gdb_server is not None
                    elf_paths: list[str] = [dispatcher_core_data.firmware_path]
                    offsets: list[int | None] = [None]
                    if dispatcher_core_data.kernel_path is not None:
                        elf_paths.append(dispatcher_core_data.kernel_path)
                        offsets.append(dispatcher_core_data.kernel_offset)
                    gdb_callstack = get_gdb_callstack(location, risc_name, elf_paths, offsets, self.gdb_server)
                    callstack_with_message = KernelCallstackWithMessage(callstack=gdb_callstack, message=None)
                    # If GDB failed to get callstack, surface errors and default to top callstack
                    if len(gdb_callstack) == 0:
                        error_message = ""
                        if isinstance(self.gdb_server.error_stream, io.StringIO):
                            error_message = f"\n  {self.gdb_server.error_stream.getvalue().strip()}"
                            # Clear after read so we don't repeat the same errors next time
                            self.gdb_server.error_stream.seek(0)
                            self.gdb_server.error_stream.truncate(0)
                        # Default to top callstack
                        callstack_with_message = get_callstack(
                            location,
                            risc_name,
                            dispatcher_core_data,
                            self.elfs_cache,
                            full_callstack=False,
                            rewind_pc_for_ebreak=rewind_pc_for_ebreak,
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
                            location.device.get_block(location).get_risc_debug(risc_name).get_pc()
                        )
                    except TimeoutDeviceRegisterError:
                        raise
                    except:
                        pass
            else:
                callstack_with_message = get_callstack(
                    location,
                    risc_name,
                    dispatcher_core_data,
                    self.elfs_cache,
                    use_full_callstack or (use_full_callstack is None and self.full_callstack),
                    rewind_pc_for_ebreak=rewind_pc_for_ebreak,
                )

        # Create result with dispatcher core data (verbose levels handled in serialization)
        return CallstacksData(
            dispatcher_core_data=dispatcher_core_data,
            pc=callstack_with_message.callstack[0].pc
            if len(callstack_with_message.callstack) > 0
            else risc_debug.get_pc(),
            kernel_callstack_with_message=callstack_with_message,
            kernel_args=_build_kernel_args(self.dispatcher_data, location, risc_name, dispatcher_core_data),
        )


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
            return int(s.getsockname()[1])
    except (socket.error, OSError) as e:
        # If we get here, no port was found
        raise TTTriageError(f"No available port found: {e}")


def start_gdb_server(port: int, context: Context) -> GdbServer:
    """Start GDB server and return it along with an error stream buffer."""
    try:
        server = ServerSocket(port)
        server.start()
        gdb_server = GdbServer(context, server, error_stream=io.StringIO())
        gdb_server.start()
    except TimeoutDeviceRegisterError:
        raise
    except Exception as e:
        raise TTTriageError(f"Failed to start GDB server on port {port}. Error: {e}")

    return gdb_server


@triage_singleton
def run(args, context: Context):
    full_callstack: bool = args["--full-callstack"]
    gdb_callstack: bool = args["--gdb-callstack"]
    active_eth: bool = args["--active-eth"]
    force_active_eth = (full_callstack or gdb_callstack) and active_eth

    if context.devices[0].is_blackhole():
        if full_callstack:
            log_warning(
                "Full callstack is currently disabled for blackhole devices due to https://github.com/tenstorrent/tt-exalens/issues/902"
            )
        if gdb_callstack:
            log_warning(
                "GDB callstack is currently disabled for blackhole devices due to https://github.com/tenstorrent/tt-exalens/issues/902"
            )

    if force_active_eth:
        WARN(
            "Getting full or gdb callstack may break active eth core. Use tt-smi reset to fix. See issue #661 in tt-exalens for more details."
        )

    elfs_cache = get_elfs_cache(args, context)
    dispatcher_data = get_dispatcher_data(args, context)

    gdb_server: GdbServer | None = None
    if gdb_callstack:
        # Locking thread until we start gdb server on available port
        with _port_lock:
            port = find_available_port()
            gdb_server = start_gdb_server(port, context)

    return CallstackProvider(
        dispatcher_data,
        elfs_cache,
        full_callstack,
        gdb_callstack,
        gdb_server,
        force_active_eth,
    )


if __name__ == "__main__":
    run_script()
