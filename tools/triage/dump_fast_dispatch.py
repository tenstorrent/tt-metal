#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Usage:
    dump_fast_dispatch

Options:

Description:
    Read important variables from fast dispatch kernels. These can help identify the state the dispatcher is in, to help
    determine the cause of any hangs.

Owner:
    jbaumanTT
"""

from dataclasses import dataclass
from triage import ScriptConfig, triage_field, run_script, log_check
from ttexalens.memory_access import MemoryAccess, RiscDebugMemoryAccess
from run_checks import run as get_run_checks
from elfs_cache import ParsedElfFile, run as get_elfs_cache, ElfsCache
from dispatcher_data import run as get_dispatcher_data, DispatcherData
from ttexalens.coordinate import OnChipCoordinate
from ttexalens.context import Context
from ttexalens.tt_exalens_lib import read_word_from_device
from inspector_data import run as get_inspector_data, InspectorData
from metal_device_id_mapping import run as get_metal_device_id_mapping, MetalDeviceIdMapping
from typing import Optional, Any

# Dumping dispatch debug information for triage purposes
# Shows dispatcher core info and purpose to help with issue diagnosis
script_config = ScriptConfig(
    depends=["run_checks", "dispatcher_data", "elfs_cache", "inspector_data", "metal_device_id_mapping"],
)


@dataclass
class DumpWaitGlobalsData:
    location: OnChipCoordinate = triage_field("Loc")
    risc_name: str = triage_field("Proc")
    kernel_name: str = triage_field("Kernel Name")
    worker_type: str | None = triage_field("worker_type")
    cq_id: int | None = triage_field("cq_id")
    servicing_device_id: int | None = triage_field("servicing_device_id")
    # Verbose fields for detailed debugging
    last_wait_count: int | None = triage_field("last_wait_count", verbose=2)
    last_wait_stream: int | None = triage_field("last_wait_stream", verbose=2)
    wait_stream_value: int | None = triage_field("wait_stream_value", verbose=2)
    cb_fence: int | None = triage_field("cb_fence", verbose=2)
    cmd_ptr: int | None = triage_field("cmd_ptr", verbose=2)
    last_event: int | None = triage_field("last_event", verbose=2)
    x: int | None = triage_field("x", verbose=2)
    y: int | None = triage_field("y", verbose=2)
    last_event_issued_to_cq: int | None = triage_field("last_event_issued_to_cq", verbose=2)
    # Number of extra pages available in the circular buffer that are not included in cb_fence_
    sem_minus_local: int | None = triage_field("cb_extra_pages", verbose=1)


def _read_symbol_value(
    elf_obj: ParsedElfFile, symbol: str, mem_access: MemoryAccess, check_value: bool = True
) -> int | None:
    """Resolve and read an integer symbol value from the kernel ELF using the provided mem_access.

    Returns None if the symbol cannot be read.
    """
    try:
        return int(elf_obj.get_global(symbol, mem_access).read_value())
    except Exception as e:
        if check_value:
            log_check(False, f"Failed to read symbol {symbol} from kernel {elf_obj.elf_file_path} with error {str(e)}")
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
        return None

    def has_any_info(self) -> bool:
        """Check if this core has any info."""
        return any([self.dispatch_info, self.dispatch_s_info, self.prefetch_info])


# This function builds a lookup map for core info for a given kernel name
def build_core_lookup_map(
    inspector_data: InspectorData, metal_device_id_mapping: MetalDeviceIdMapping
) -> dict[tuple[int, int, int], MultiCategoryCoreLookup]:
    """
    Build lookup map for core info for a given kernel name using unique_id as chip key.

    Returns a dictionary mapping (unique_id, x, y) to a MultiCategoryCoreLookup object
    MultiCategoryCoreLookup object contains dispatch_info, dispatch_s_info, and prefetch_info
    """
    # Get all core info from inspector data
    all_cores = inspector_data.getAllDispatchCoreInfos()

    # Convert to dictionary for faster lookup
    # key is (unique_id, x, y) and value is a MultiCategoryCoreLookup object
    # MultiCategoryCoreLookup object contains dispatch_info, dispatch_s_info, and prefetch_info
    lookup: dict[tuple[int, int, int], MultiCategoryCoreLookup] = {}

    for category_group in all_cores.coresByCategory:
        category = category_group.category  # "dispatch", "dispatchS", or "prefetch"

        for core_entry in category_group.entries:
            # core_entry.key.chip is metal device_id
            metal_device_id = core_entry.key.chip
            unique_id = metal_device_id_mapping.get_unique_id(metal_device_id)

            # Use unique_id as the key
            key = (unique_id, core_entry.key.x, core_entry.key.y)

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

    # Skipping because we cannot read NCRISC private memory on wormhole
    if risc_name == "ncrisc" and location.device.is_wormhole():
        return None

    # If no kernel loaded, nothing to read
    dispatcher_core_data = dispatcher_data.get_cached_core_data(location, risc_name)
    if dispatcher_core_data.kernel_path is None:
        return None
    assert dispatcher_core_data.kernel_name is not None

    kernel_elf = elf_cache[dispatcher_core_data.kernel_path]
    loc_mem_access = RiscDebugMemoryAccess(location.noc_block.get_risc_debug(risc_name), ensure_halted_access=False)
    is_dispatcher_kernel = (
        dispatcher_core_data.kernel_name == "cq_dispatch"
        or dispatcher_core_data.kernel_name == "cq_dispatch_subordinate"
    )
    # Inline: read wait-related globals directly from ELF
    last_wait_count = _read_symbol_value(
        kernel_elf, "last_wait_count", loc_mem_access, check_value=is_dispatcher_kernel
    )
    last_wait_stream = _read_symbol_value(
        kernel_elf, "last_wait_stream", loc_mem_access, check_value=is_dispatcher_kernel
    )
    last_event = _read_symbol_value(
        kernel_elf, "last_event", loc_mem_access, check_value=dispatcher_core_data.kernel_name == "cq_dispatch"
    )
    try:
        circular_buffer_fence = kernel_elf.get_global("dispatch_cb_reader", loc_mem_access).cb_fence_
    except Exception:
        if dispatcher_core_data.kernel_name == "cq_dispatch":
            log_check(False, f"Failed to read circular_buffer_fence for kernel {dispatcher_core_data.kernel_name}")
        circular_buffer_fence = None
    command_pointer = _read_symbol_value(kernel_elf, "cmd_ptr", loc_mem_access, check_value=is_dispatcher_kernel)

    def get_const_value(name: str, check_value: bool = True) -> int | None:
        try:
            value = kernel_elf.get_constant(name)
            assert isinstance(value, int)
            return value
        except Exception:
            if check_value:
                log_check(False, f"Failed to read constant {name} for kernel {dispatcher_core_data.kernel_name}")
            return None

    stream_addr0 = get_const_value("stream_addr0", check_value=is_dispatcher_kernel)
    stream_addr1 = get_const_value("stream_addr1", check_value=is_dispatcher_kernel)
    stream_width = get_const_value("stream_width", check_value=is_dispatcher_kernel)

    wait_stream_value = None
    if stream_addr0 is not None and stream_addr1 is not None and last_wait_stream is not None:
        stream_stride_bytes = stream_addr1 - stream_addr0
        wait_stream_value = read_word_from_device(
            location,
            stream_addr0 + stream_stride_bytes * last_wait_stream,
        )

        if is_dispatcher_kernel:
            log_check(
                wait_stream_value is not None,
                f"Failed to read wait_stream_value for kernel {dispatcher_core_data.kernel_name}. There may be a problem with the dispatcher kernel.",
            )

    if last_wait_count is not None and stream_width is not None:
        # Wrap the global wait count to the stream width, to match the stream wrap behavior
        last_wait_count = last_wait_count & ((1 << stream_width) - 1)

    # Compute sem_minus_local for dispatcher kernels by reading the live semaphore and subtracting local_count_
    sem_minus_local: int | None = None
    try:
        if dispatcher_core_data.kernel_name == "cq_dispatch":
            my_dispatch_cb_sem_id = int(kernel_elf.get_constant("my_dispatch_cb_sem_id"))
            fd_core_type_idx = int(kernel_elf.get_constant("fd_core_type_idx"))

            # sem_l1_base is a firmware global array of L1 pointers; index by core type
            sem_base_ptr = kernel_elf.get_global("sem_l1_base", loc_mem_access)[fd_core_type_idx]
            sem_value = sem_base_ptr[my_dispatch_cb_sem_id * 16 // 4]
            local_count = kernel_elf.get_global("dispatch_cb_reader", loc_mem_access).local_count_

            # Two's-complement 32-bit wrapping difference
            delta = (int(sem_value) - int(local_count)) & 0xFFFFFFFF
            sem_minus_local = delta - 0x100000000 if (delta & 0x80000000) else delta
    except Exception:
        log_check(
            False,
            f"Failed to read sem_minus_local for kernel {dispatcher_core_data.kernel_name}. There may be a problem with the dispatcher kernel.",
        )
        # Leave as None if any lookups fail
        sem_minus_local = None

    # Get virtual coordinate for this specific core
    virtual_coord = location.to("translated")
    # Use unique_id instead of device.id to avoid mapping issues with TT_METAL_VISIBLE_DEVICES
    chip_id = location._device.unique_id
    x, y = virtual_coord

    # Lookup core info for the given kernel name based on virtual coordinates
    multi_info = core_lookup.get((chip_id, x, y))

    # Get the appropriate core info for the given kernel name
    # Note: multi_info should exist since we pre-filtered, but check for chip ID mapping issues
    core_info = multi_info.get_info_for_kernel(dispatcher_core_data.kernel_name) if multi_info else None

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
        servicing_device_id=getattr(core_info, "servicingMetalDeviceId", None),
        last_event_issued_to_cq=getattr(core_info, "eventID", None),
        sem_minus_local=sem_minus_local,
    )


def run(args, context: Context):
    """Entry point for triage framework."""
    from triage import set_verbose_level

    # Set verbose level from -v count (controls which columns are displayed)
    verbose_level = args["-v"]
    set_verbose_level(verbose_level)
    run_checks = get_run_checks(args, context)
    dispatcher_data = get_dispatcher_data(args, context)
    elfs_cache = get_elfs_cache(args, context)

    # Get inspector data and device ID mapping
    inspector_data = get_inspector_data(args, context)
    metal_device_id_mapping = get_metal_device_id_mapping(args, context)
    # Build lookup map for core info for a given kernel name
    core_lookup = build_core_lookup_map(inspector_data, metal_device_id_mapping)

    # Build dispatch_core_pairs by finding all RISC cores with dispatcher kernels
    dispatch_core_pairs = []
    # Relevant dispatcher kernel names
    dispatcher_kernel_names = {"cq_dispatch", "cq_dispatch_subordinate", "cq_prefetch"}

    # Go through all cores in the core_lookup (now keyed by unique_id)
    # And check if they have dispatcher kernels loaded
    for (chip_unique_id, x, y), info in core_lookup.items():
        if not info.has_any_info():
            continue

        device = run_checks.get_device_by_unique_id(chip_unique_id)
        # Create OnChipCoordinate for this dispatcher core location
        location = OnChipCoordinate(x, y, "translated", device)

        # Check all RISC cores at this location for dispatcher kernels
        noc_block = location._device.get_block(location)
        for risc_name in noc_block.risc_names:
            dispatcher_core_data = dispatcher_data.get_cached_core_data(location, risc_name)
            if (
                dispatcher_core_data.kernel_name is not None
                and dispatcher_core_data.kernel_name in dispatcher_kernel_names
            ):
                dispatch_core_pairs.append((location, risc_name))

    # Convert to set for fast lookup
    dispatch_cores_set = set(dispatch_core_pairs)

    # Define a wrapper function that filters to only dispatcher cores
    # Aim of this is to avoid checking non-dispatcher cores and fasten the process
    def filtered_read_wait_globals(location: OnChipCoordinate, risc_name: str) -> DumpWaitGlobalsData | None:
        """Wrapper that only processes dispatcher cores using fast set lookup."""
        if (location, risc_name) not in dispatch_cores_set:
            return None
        return read_wait_globals(location, risc_name, dispatcher_data, elfs_cache, core_lookup)

    BLOCK_TYPES_TO_CHECK = ["tensix", "idle_eth"]
    return run_checks.run_per_core_check(
        filtered_read_wait_globals,
        block_filter=BLOCK_TYPES_TO_CHECK,
    )


if __name__ == "__main__":
    run_script()
