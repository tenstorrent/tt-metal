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
from ttexalens.elf import MemoryAccess


script_config = ScriptConfig(
    depends=["run_checks", "dispatcher_data", "elfs_cache", "inspector_data"],
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
    x: int | None = triage_field("x")
    y: int | None = triage_field("y")
    worker_type: str | None = triage_field("worker_type")
    cq_id: int | None = triage_field("cq_id")
    servicing_device_id: int | None = triage_field("servicing_device_id")
    last_event_issued_to_cq: int | None = triage_field("last_event_issued_to_cq")


def _read_symbol_value(elf_obj: ParsedElfFile, symbol: str, mem_access: MemoryAccess) -> int | None:
    """Resolve and read an integer symbol value from the kernel ELF using the provided mem_access.

    Returns None if the symbol cannot be read.
    """
    try:
        return int(elf_obj.get_global(symbol, mem_access).read_value())
    except Exception:
        return None


def read_wait_globals(
    location: OnChipCoordinate,
    risc_name: str,
    dispatcher_data: DispatcherData,
    elf_cache: ElfsCache,
    inspector_data: InspectorData,
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
    loc_mem_access = MemoryAccess.get(location.noc_block.get_risc_debug(risc_name))
    # Inline: read wait-related globals directly from ELF
    last_wait_count = _read_symbol_value(kernel_elf, "last_wait_count", loc_mem_access)
    last_wait_stream = _read_symbol_value(kernel_elf, "last_wait_stream", loc_mem_access)
    last_event = _read_symbol_value(kernel_elf, "last_event", loc_mem_access)
    circular_buffer_fence = _read_symbol_value(kernel_elf, "cb_fence", loc_mem_access)
    command_pointer = _read_symbol_value(kernel_elf, "cmd_ptr", loc_mem_access)

    def get_const_value(name: str) -> int | None:
        try:
            value = kernel_elf.get_constant(name)
            assert isinstance(value, int)
            return value
        except Exception:
            return None

    stream_addr0 = None
    stream_addr1 = None
    stream_width = None

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

    has_values = last_wait_count is not None or last_wait_stream is not None

    # Get virtual coordinate for this specific core
    virtual_coord = location.to("translated")
    chip_id = location._device._id
    x, y = virtual_coord

    # Create virtual core object
    vc = VirtualCore()
    vc.chip = chip_id
    vc.x = x
    vc.y = y

    # Try to get dispatch core info for this specific location
    dispatch_core_info = get_core_info(inspector_data, vc, "getDispatchCoreInfo")
    # Try to get dispatch_s core info for this specific location
    dispatch_s_core_info = get_core_info(inspector_data, vc, "getDispatchSCoreInfo")
    # Try to get prefetch core info for this specific location
    prefetch_core_info = get_core_info(inspector_data, vc, "getPrefetchCoreInfo")

    # Override dispatch_core_info based on kernel
    # If its a dispatch_subordinate kernel, use the dispatch_s_core_info
    # instead of dispatch_core_info
    if dispatcher_core_data.kernel_name == "cq_dispatch_subordinate":
        dispatch_core_info = dispatch_s_core_info

    # If there are no values and no dispatch/prefetch core info, return None
    # All three should be None if the core is not a dispatch/dispatch_s/prefetch core
    if not has_values and not any([dispatch_core_info, dispatch_s_core_info, prefetch_core_info]):
        return None

    # Get the core info for the core
    core_info = dispatch_core_info or prefetch_core_info

    # BRISC = cq_prefetch + cq_dispatch
    # NCRISC = cq_dispatch_subordinate
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
        x=x,
        y=y,
        worker_type=getattr(core_info, "workType", None),
        cq_id=getattr(core_info, "cqId", None),
        servicing_device_id=getattr(core_info, "servicingDeviceId", None),
        last_event_issued_to_cq=getattr(core_info, "eventID", None),
    )


def run(args, context: Context):
    """Entry point for triage framework."""
    run_checks = get_run_checks(args, context)
    dispatcher_data = get_dispatcher_data(args, context)
    elfs_cache = get_elfs_cache(args, context)
    inspector_data = get_inspector_data(args, context)

    BLOCK_TYPES_TO_CHECK = ["tensix", "idle_eth"]
    return run_checks.run_per_core_check(
        lambda location, risc_name: read_wait_globals(location, risc_name, dispatcher_data, elfs_cache, inspector_data),
        block_filter=BLOCK_TYPES_TO_CHECK,
    )


if __name__ == "__main__":
    run_script()
