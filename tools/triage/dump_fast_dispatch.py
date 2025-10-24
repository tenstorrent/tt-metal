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


# This class is used to lookup core info for a given kernel name
# It contains dispatch_info, dispatch_s_info, and prefetch_info
# Returns the appropriate core info for a given kernel name
# If no info is found, returns None
@dataclass
class MultiCategoryCoreLookup:
    """A core might have info in multiple categories."""

    dispatch_info: Optional[Any] = None
    dispatch_s_info: Optional[Any] = None
    prefetch_info: Optional[Any] = None

    def get_info_for_kernel(self, kernel_name: str) -> Optional[Any]:
        """
        Get the appropriate core info based on kernel name.

        dispatch_subordinate kernel should use dispatch_s info when kernel name is "cq_dispatch_subordinate"
        dispatch and dispatch_subordinate kernels can have same virtual coordinates if they are on the same device.

        For all other cases, use the info that matches the category:
        - cq_dispatch kernels use dispatch info
        - cq_dispatch_subordinate kernels use dispatch_s info
        - cq_prefetch kernels use prefetch info
        BRISC : cq_prefetch / cq_dispatch
        NCRISC: cq_dispatch_subordinate
        """
        # For dispatch kernels on BRISC, use dispatch info
        if kernel_name == "cq_dispatch":
            # Use dispatch info
            return self.dispatch_info
        # For dispatch_subordinate kernels on NCRISC, use dispatch_s info
        elif kernel_name == "cq_dispatch_subordinate":
            return self.dispatch_s_info
        # For prefetch kernels on BRISC, use prefetch info
        elif kernel_name == "cq_prefetch":
            return self.prefetch_info

    def has_any_info(self) -> bool:
        """Check if this core has any info."""
        return any([self.dispatch_info, self.dispatch_s_info, self.prefetch_info])


# This function builds a lookup map for core info for a given kernel name
def build_core_lookup_map(inspector_data: InspectorData) -> dict[tuple[int, int, int], MultiCategoryCoreLookup]:
    """
    Build lookup map for core info for a given kernel name

    Returns a dictionary mapping (chip, x, y) to a MultiCategoryCoreLookup object
    MultiCategoryCoreLookup object contains dispatch_info, dispatch_s_info, and prefetch_info
    """
    # Get all core info from inspector data
    all_cores = inspector_data.getAllDispatchCoreInfos()

    # Convert to dictionary for faster lookup
    # key is (chip, x, y) and value is a MultiCategoryCoreLookup object
    # MultiCategoryCoreLookup object contains dispatch_info, dispatch_s_info, and prefetch_info
    lookup: dict[tuple[int, int, int], MultiCategoryCoreLookup] = {}

    for category_group in all_cores.coresByCategory:
        category = category_group.category  # "dispatch", "dispatchS", or "prefetch"

        for core_entry in category_group.entries:
            key = (core_entry.key.chip, core_entry.key.x, core_entry.key.y)

            # Get or create entry for this core
            if key not in lookup:
                lookup[key] = MultiCategoryCoreLookup()

            # Store in appropriate field based on category
            if category == "dispatch":
                lookup[key].dispatch_info = core_entry.info
            elif category == "dispatchS":
                lookup[key].dispatch_s_info = core_entry.info
            elif category == "prefetch":
                lookup[key].prefetch_info = core_entry.info

    return lookup


def read_wait_globals(
    location: OnChipCoordinate,
    risc_name: str,
    dispatcher_data: DispatcherData,
    elf_cache: ElfsCache,
    core_lookup: dict[tuple[int, int, int], MultiCategoryCoreLookup],
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

    # Lookup core info for the given kernel name based on virtual coordinates
    multi_info = core_lookup.get((chip_id, x, y))

    # Get the appropriate core info for the given kernel name
    core_info = None
    if multi_info is not None and multi_info.has_any_info():
        core_info = multi_info.get_info_for_kernel(dispatcher_core_data.kernel_name)

    # If no values and no core info, return None
    if not has_values and core_info is None:
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

    # Get inspector data
    inspector_data = get_inspector_data(args, context)
    # Build lookup map for core info for a given kernel name
    core_lookup = build_core_lookup_map(inspector_data)

    BLOCK_TYPES_TO_CHECK = ["tensix", "idle_eth"]
    return run_checks.run_per_core_check(
        lambda location, risc_name: read_wait_globals(location, risc_name, dispatcher_data, elfs_cache, core_lookup),
        block_filter=BLOCK_TYPES_TO_CHECK,
    )


if __name__ == "__main__":
    run_script()
