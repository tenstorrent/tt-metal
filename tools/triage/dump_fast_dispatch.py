#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Usage:
    dump_fast_dispatch

Options:

Description:
    Read important variables from fast dispatch kernels.
"""

from dataclasses import dataclass
from triage import ScriptConfig, triage_field, run_script
from run_checks import run as get_run_checks
from elfs_cache import ParsedElfFile, run as get_elfs_cache, ElfsCache
from dispatcher_data import run as get_dispatcher_data, DispatcherData
from ttexalens.coordinate import OnChipCoordinate
from ttexalens.context import Context
from ttexalens.tt_exalens_lib import read_word_from_device
from ttexalens.firmware import ELF


script_config = ScriptConfig(
    depends=["run_checks", "dispatcher_data", "elfs_cache"],
)


@dataclass
class DumpWaitGlobalsData:
    location: OnChipCoordinate = triage_field("Loc")
    risc_name: str = triage_field("Proc")
    kernel_name: str = triage_field("Kernel Name")
    last_wait_count: int | None = triage_field("last_wait_count")
    last_wait_stream: int | None = triage_field("last_wait_stream")
    wait_stream_value: int | None = triage_field("wait_stream_value")
    cb_fence: int | None = triage_field("cb_fence")
    cmd_ptr: int | None = triage_field("cmd_ptr")
    last_event: int | None = triage_field("last_event")
    is_d_variant: int | None = triage_field("is_d_variant")
    is_h_variant: int | None = triage_field("is_h_variant")


def _read_symbol_value(elf_obj: ParsedElfFile, symbol: str, mem_reader) -> int | None:
    """Resolve and read an integer symbol value from the kernel ELF using the provided mem_reader.

    Returns None if the symbol cannot be read.
    """
    try:
        return int(elf_obj.get_global(symbol, mem_reader).get_value())
    except Exception:
        return None


def _get_mem_reader(location: OnChipCoordinate, risc_name: str):
    """Get a mem_reader for the given RISC at the location, falling back to default if needed."""
    try:
        return ELF.get_mem_reader(location, risc_name)
    except Exception:
        return ELF.get_mem_reader(location)


def read_wait_globals(
    location: OnChipCoordinate,
    risc_name: str,
    dispatcher_data: DispatcherData,
    elf_cache: ElfsCache,
) -> DumpWaitGlobalsData | None:
    """Read wait globals and related constants from the current kernel at this core.

    Returns a populated DumpWaitGlobalsData if any relevant values were found; otherwise None.
    """
    # If no kernel loaded, nothing to read
    dispatcher_core_data = dispatcher_data.get_core_data(location, risc_name)
    if dispatcher_core_data.kernel_path is None:
        return None
    assert dispatcher_core_data.kernel_name is not None

    kernel_elf = elf_cache[dispatcher_core_data.kernel_path]
    mem_reader = _get_mem_reader(location, risc_name)
    # Inline: read wait-related globals directly from ELF
    last_wait_count = _read_symbol_value(kernel_elf, "last_wait_count", mem_reader)
    last_wait_stream = _read_symbol_value(kernel_elf, "last_wait_stream", mem_reader)
    last_event = _read_symbol_value(kernel_elf, "last_event", mem_reader)
    circular_buffer_fence = _read_symbol_value(kernel_elf, "cb_fence", mem_reader)
    command_pointer = _read_symbol_value(kernel_elf, "cmd_ptr", mem_reader)

    def get_const_value(name: str):
        try:
            return kernel_elf.get_constant(name)
        except Exception:
            return None

    stream_addr0 = None
    stream_addr1 = None
    stream_width = None

    is_d_variant = get_const_value("is_d_variant")
    is_h_variant = get_const_value("is_h_variant")
    stream_addr0 = get_const_value("stream_addr0")
    stream_addr1 = get_const_value("stream_addr1")
    stream_width = get_const_value("stream_width")

    wait_stream_value = None
    if stream_addr0 is not None and stream_addr1 is not None and last_wait_stream is not None:
        stream_stride_bytes = stream_addr1 - stream_addr0
        wait_stream_value = read_word_from_device(
            location,
            stream_addr0 + stream_stride_bytes * last_wait_stream,
        )

    if last_wait_count is not None and stream_width is not None:
        # Wrap the global wait count to the stream width, to match the stream wrap behavior
        last_wait_count = last_wait_count & ((1 << stream_width) - 1)

    has_values = (
        last_wait_count is not None
        or last_wait_stream is not None
        or is_d_variant is not None
        or is_h_variant is not None
    )

    if not has_values:
        return None

    return DumpWaitGlobalsData(
        location=location,
        risc_name=risc_name,
        kernel_name=dispatcher_core_data.kernel_name,
        last_wait_count=last_wait_count,
        last_wait_stream=last_wait_stream,
        wait_stream_value=wait_stream_value,
        cb_fence=circular_buffer_fence,
        cmd_ptr=command_pointer,
        last_event=last_event,
        is_d_variant=is_d_variant,
        is_h_variant=is_h_variant,
    )


def run(args, context: Context):
    """Entry point for triage framework."""
    run_checks = get_run_checks(args, context)
    dispatcher_data = get_dispatcher_data(args, context)
    elfs_cache = get_elfs_cache(args, context)
    BLOCK_TYPES_TO_CHECK = ["tensix", "idle_eth"]
    return run_checks.run_per_core_check(
        lambda location, risc_name: read_wait_globals(location, risc_name, dispatcher_data, elfs_cache),
        block_filter=BLOCK_TYPES_TO_CHECK,
    )


if __name__ == "__main__":
    run_script()
