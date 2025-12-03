# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import inspect
import os
import time
from enum import Enum, IntEnum
from pathlib import Path

from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.hardware_controller import HardwareController
from ttexalens.coordinate import OnChipCoordinate
from ttexalens.debug_tensix import TensixDebug
from ttexalens.hardware.risc_debug import CallstackEntry, RiscDebug, RiscLocation
from ttexalens.tt_exalens_lib import (
    callstack,
    check_context,
    convert_coordinate,
    load_elf,
    read_from_device,
    read_word_from_device,
    validate_device_id,
    write_to_device,
    write_words_to_device,
)

from .format_config import DataFormat, FormatConfig
from .llk_params import DestAccumulation, Mailbox
from .pack import (
    pack_bfp8_b,
    pack_bfp16,
    pack_fp16,
    pack_fp32,
    pack_int8,
    pack_int32,
    pack_uint8,
    pack_uint16,
    pack_uint32,
)
from .target_config import TestTargetConfig
from .unpack import (
    unpack_bfp8_b,
    unpack_bfp16,
    unpack_fp16,
    unpack_fp32,
    unpack_int8,
    unpack_int32,
    unpack_res_tiles,
    unpack_uint8,
    unpack_uint16,
    unpack_uint32,
)

# Constant - indicates the TRISC kernel run status
KERNEL_COMPLETE = 1  # Kernel completed its run


class BootMode(Enum):
    BRISC = "brisc"
    TRISC = "trisc"
    EXALENS = "exalens"
    DEFAULT = "default"


CHIP_DEFAULT_BOOT_MODES = {
    ChipArchitecture.WORMHOLE: BootMode.BRISC,
    ChipArchitecture.BLACKHOLE: BootMode.BRISC,
    ChipArchitecture.QUASAR: BootMode.TRISC,
}


# Constant - indicates that the RISC core doesn't exist on the chip
INVALID_CORE = -1


class RiscCore(IntEnum):
    BRISC = INVALID_CORE if get_chip_architecture() == ChipArchitecture.QUASAR else 11
    TRISC0 = 11 if get_chip_architecture() == ChipArchitecture.QUASAR else 12
    TRISC1 = 12 if get_chip_architecture() == ChipArchitecture.QUASAR else 13
    TRISC2 = 13 if get_chip_architecture() == ChipArchitecture.QUASAR else 14
    TRISC3 = 14 if get_chip_architecture() == ChipArchitecture.QUASAR else INVALID_CORE

    def __str__(self):
        return self.name.lower()


# Constant - list of all valid cores on the chip
ALL_CORES = [core for core in RiscCore if core != INVALID_CORE]


def resolve_default_boot_mode(boot_mode: BootMode) -> BootMode:
    if boot_mode == BootMode.DEFAULT:
        CHIP_ARCH = get_chip_architecture()
        boot_mode = CHIP_DEFAULT_BOOT_MODES[CHIP_ARCH]
    return boot_mode


def get_register_store(location="0,0", device_id=0, neo_id=0):
    CHIP_ARCH = get_chip_architecture()
    context = check_context()
    device = context.devices[device_id]
    chip_coordinate = OnChipCoordinate.create(location, device=device)
    noc_block = device.get_block(chip_coordinate)
    if CHIP_ARCH == ChipArchitecture.QUASAR:
        match neo_id:
            case 0:
                register_store = noc_block.neo0.register_store
            case 1:
                register_store = noc_block.neo1.register_store
            case 2:
                register_store = noc_block.neo2.register_store
            case 3:
                register_store = noc_block.neo3.register_store
            case _:
                raise ValueError(f"Invalid neo_id {neo_id} for Quasar architecture")
    else:
        if neo_id != 0:
            raise ValueError(f"Invalid non zero neo_id for non Quasar architecture")
        register_store = noc_block.get_register_store()
    return register_store


def get_soft_reset_mask(cores: list[RiscCore]):
    if INVALID_CORE in cores:
        raise ValueError("Attempting to reset a core that doesn't exist on this chip")
    return sum(1 << core.value for core in cores)


def set_tensix_soft_reset(
    value, cores: list[RiscCore] = ALL_CORES, location="0,0", device_id=0
):
    soft_reset = get_register_store(location, device_id).read_register(
        "RISCV_DEBUG_REG_SOFT_RESET_0"
    )
    if value:
        soft_reset |= get_soft_reset_mask(cores)
    else:
        soft_reset &= ~get_soft_reset_mask(cores)
    get_register_store(location, device_id).write_register(
        "RISCV_DEBUG_REG_SOFT_RESET_0", soft_reset
    )


def collect_results(
    formats: FormatConfig,
    tile_count: int,
    address: int = 0x1C000,
    location: str = "0,0",
    sfpu: bool = False,
    tile_dimensions=[32, 32],
    num_faces: int = 4,
    face_r_dim: int = 16,  # Default to 16 for backward compatibility
):
    # Always read full tiles - hardware still outputs full tile data
    # but with variable face dimensions, only part of it is valid
    tile_elements = tile_dimensions[0] * tile_dimensions[1]
    read_bytes_cnt = (
        formats.output_format.num_bytes_per_tile(tile_elements) * tile_count
    )

    read_data = read_from_device(location, address, num_bytes=read_bytes_cnt)
    res_from_L1 = unpack_res_tiles(
        read_data,
        formats,
        tile_count=tile_count,
        sfpu=sfpu,
        num_faces=num_faces,
        face_r_dim=face_r_dim,
    )
    return res_from_L1


def exalens_device_setup(chip_arch, device_id=0, location="0,0"):
    context = check_context()
    device = context.devices[device_id]
    chip_coordinate = OnChipCoordinate.create(location, device=device)
    debug_tensix = TensixDebug(chip_coordinate, device_id, context)
    ops = debug_tensix.device.instructions

    if chip_arch == ChipArchitecture.BLACKHOLE:
        get_register_store(location, device_id).write_register(
            "RISCV_DEBUG_REG_DEST_CG_CTRL", 0
        )
        debug_tensix.inject_instruction(ops.TT_OP_ZEROACC(3, 0, 0, 1, 0), 0)
    else:
        debug_tensix.inject_instruction(ops.TT_OP_ZEROACC(3, 0, 0), 0)

    debug_tensix.inject_instruction(ops.TT_OP_SFPENCC(3, 0, 0, 10), 0)
    debug_tensix.inject_instruction(ops.TT_OP_NOP(), 0)

    debug_tensix.inject_instruction(ops.TT_OP_SFPCONFIG(0, 11, 1), 0)

    debug_tensix.inject_instruction(ops.TT_OP_SEMINIT(1, 0, 2), 0)
    debug_tensix.inject_instruction(ops.TT_OP_SEMINIT(1, 0, 7), 0)
    debug_tensix.inject_instruction(ops.TT_OP_SEMINIT(1, 0, 4), 0)


def run_elf_files(testname, boot_mode, device_id=0, location="0,0"):
    CHIP_ARCH = get_chip_architecture()
    LLK_HOME = os.environ.get("LLK_HOME")
    BUILD_DIR = Path(LLK_HOME) / "tests" / "build" / CHIP_ARCH.value
    TEST_DIR = BUILD_DIR / "tests" / testname

    boot_mode = resolve_default_boot_mode(boot_mode)

    if CHIP_ARCH == ChipArchitecture.QUASAR and boot_mode != BootMode.TRISC:
        raise ValueError("Quasar only supports TRISC boot mode")

    # Perform soft reset
    set_tensix_soft_reset(1, location=location, device_id=device_id)

    # Load TRISC ELF files
    trisc_names = ["unpack", "math", "pack"]
    trisc_start_addresses = [0x16DFF0, 0x16DFF4, 0x16DFF8]
    is_wormhole = get_chip_architecture() == ChipArchitecture.WORMHOLE

    elfs = [
        str((TEST_DIR / "elf" / f"{trisc_name}.elf").absolute())
        for trisc_name in trisc_names
    ]

    for i, elf in enumerate(elfs):
        if is_wormhole:
            start_address = load_elf(
                elf_file=elf,
                location=location,
                risc_name=f"trisc{i}",
                neo_id=0 if CHIP_ARCH == ChipArchitecture.QUASAR else None,
                return_start_address=True,
            )
            write_words_to_device(location, trisc_start_addresses[i], [start_address])
        else:
            load_elf(
                elf_file=elf,
                location=location,
                risc_name=f"trisc{i}",
                neo_id=0 if CHIP_ARCH == ChipArchitecture.QUASAR else None,
            )

    # Reset the profiler barrier
    TRISC_PROFILER_BARRIER_ADDRESS = 0x16AFF4
    write_words_to_device(location, TRISC_PROFILER_BARRIER_ADDRESS, [0, 0, 0])

    match boot_mode:
        case BootMode.BRISC:
            brisc_elf_path = BUILD_DIR / "shared" / "elf" / "brisc.elf"
            load_elf(
                elf_file=str(brisc_elf_path.absolute()),
                location=location,
                risc_name="brisc",
            )
            set_tensix_soft_reset(
                0, [RiscCore.BRISC], location=location, device_id=device_id
            )
        case BootMode.TRISC:
            set_tensix_soft_reset(
                0, [RiscCore.TRISC0], location=location, device_id=device_id
            )
        case BootMode.EXALENS:
            exalens_device_setup(CHIP_ARCH, device_id, location)
            set_tensix_soft_reset(
                0,
                [RiscCore.TRISC0, RiscCore.TRISC1, RiscCore.TRISC2],
                location=location,
                device_id=device_id,
            )

    return elfs


def write_stimuli_to_l1(
    test_config,
    buffer_A,
    buffer_B,
    stimuli_A_format: DataFormat,
    stimuli_B_format: DataFormat,
    tile_count_A: int = 1,
    tile_count_B: int = None,
    location="0,0",
    num_faces=4,
    buffer_C=None,
    stimuli_C_format: DataFormat = None,
    tile_count_C: int = None,
):
    """
    Write stimuli to L1 with support for 2 or 3 input tensors.

    Args:
        test_config: Used to store addresses of A, B, (C) and Result
        buffer_A: Flattened tensor data for matrix A
        buffer_B: Flattened tensor data for matrix B
        stimuli_A_format: DataFormat for matrix A
        stimuli_B_format: DataFormat for matrix B
        tile_count_A: Number of tiles in matrix A
        tile_count_B: Number of tiles in matrix B
        location: Core location string
        num_faces: Number of faces for packing
        buffer_C: Optional flattened tensor data for matrix C (for 3-input operations)
        stimuli_C_format: Optional DataFormat for matrix C
        tile_count_C: Optional number of tiles in matrix C

    Returns:
        int: Address where result will be stored
    """

    TILE_ELEMENTS = 1024

    # Calculate L1 addresses
    tile_size_A_bytes = stimuli_A_format.num_bytes_per_tile(TILE_ELEMENTS)
    tile_size_B_bytes = stimuli_B_format.num_bytes_per_tile(TILE_ELEMENTS)

    buffer_A_address = 0x1A000
    buffer_B_address = buffer_A_address + tile_size_A_bytes * tile_count_A

    # Handle optional third buffer
    if buffer_C is not None:
        if stimuli_C_format is None or tile_count_C is None:
            raise ValueError(
                "If buffer_C is provided, stimuli_C_format and tile_count_C must also be provided"
            )

        tile_size_C_bytes = stimuli_C_format.num_bytes_per_tile(TILE_ELEMENTS)
        buffer_C_address = buffer_B_address + tile_size_B_bytes * tile_count_B
        result_buffer_address = buffer_C_address + tile_size_C_bytes * tile_count_C
    else:
        buffer_C_address = None
        result_buffer_address = buffer_B_address + tile_size_B_bytes * tile_count_B

    # Helper function to get packer
    def get_packer(data_format):
        packers = {
            DataFormat.Float16: pack_fp16,
            DataFormat.Float16_b: pack_bfp16,
            DataFormat.Float32: pack_fp32,
            DataFormat.Bfp8_b: pack_bfp8_b,
            DataFormat.Int32: pack_int32,
            DataFormat.UInt32: pack_uint32,
            DataFormat.UInt16: pack_uint16,
            DataFormat.Int8: pack_int8,
            DataFormat.UInt8: pack_uint8,
        }
        return packers.get(data_format)

    pack_function_A = get_packer(stimuli_A_format)
    pack_function_B = get_packer(stimuli_B_format)

    # Validate pack functions for A and B
    if not pack_function_A or not pack_function_B:
        raise ValueError(
            f"Unsupported data formats: {stimuli_A_format.name}, {stimuli_B_format.name}"
        )

    # Handle optional third buffer pack function
    pack_function_C = None
    if buffer_C is not None:
        pack_function_C = get_packer(stimuli_C_format)
        if not pack_function_C:
            raise ValueError(
                f"Unsupported data format for buffer_C: {stimuli_C_format.name}"
            )

    def write_matrix(
        buffer, tile_count, pack_function, base_address, tile_size, num_faces
    ):
        addresses = []
        packed_data_list = []

        pack_function_lambda = lambda buffer_tile: (
            pack_function(buffer_tile, num_faces=num_faces)
            if pack_function == pack_bfp8_b
            else pack_function(buffer_tile)
        )

        for i in range(tile_count):
            start_idx = TILE_ELEMENTS * i
            tile_data = buffer[start_idx : start_idx + TILE_ELEMENTS]
            packed_data = pack_function_lambda(tile_data)

            addresses.append(base_address + i * tile_size)
            packed_data_list.append(packed_data)

        for addr, data in zip(addresses, packed_data_list):
            write_to_device(location, addr, data)

    write_matrix(
        buffer_A,
        tile_count_A,
        pack_function_A,
        buffer_A_address,
        tile_size_A_bytes,
        num_faces,
    )
    write_matrix(
        buffer_B,
        tile_count_B,
        pack_function_B,
        buffer_B_address,
        tile_size_B_bytes,
        num_faces,
    )

    # Write optional third buffer
    if buffer_C is not None:
        write_matrix(
            buffer_C,
            tile_count_C,
            pack_function_C,
            buffer_C_address,
            tile_size_C_bytes,
            num_faces,
        )

    # Set buffer addresses in device to be defined in build header
    test_config["buffer_A_address"] = buffer_A_address
    test_config["buffer_B_address"] = buffer_B_address
    if buffer_C_address is not None:
        test_config["buffer_C_address"] = buffer_C_address
    test_config["result_buffer_address"] = result_buffer_address

    return result_buffer_address


def get_result_from_device(
    formats: FormatConfig,
    read_data_bytes: bytes,
    location: str = "0,0",
    sfpu: bool = False,
):
    # Dictionary of format to unpacking function mappings
    unpackers = {
        DataFormat.Float16: unpack_fp16,
        DataFormat.Float16_b: unpack_bfp16,
        DataFormat.Float32: unpack_fp32,
        DataFormat.Int32: unpack_int32,
        DataFormat.UInt32: unpack_uint32,
        DataFormat.UInt16: unpack_uint16,
        DataFormat.Int8: unpack_int8,
        DataFormat.UInt8: unpack_uint8,
    }

    # Handling "Bfp8_b" format separately with sfpu condition
    if formats.output_format == DataFormat.Bfp8_b:
        unpack_func = unpack_bfp16 if sfpu else unpack_bfp8_b
    else:
        unpack_func = unpackers.get(formats.output_format)

    if unpack_func:
        num_args = len(inspect.signature(unpack_func).parameters)
        if num_args > 1:
            return unpack_func(
                read_data_bytes, formats.input_format, formats.output_format
            )
        else:
            return unpack_func(read_data_bytes)
    else:
        raise ValueError(f"Unsupported format: {formats.output_format}")


def read_dest_register(dest_acc: DestAccumulation, num_tiles: int = 1):
    """
    Reads values in the destination register from the device.
        - Only supported on BH . Due to hardware bug, TRISCs exit the halted state after a single read and must be rehalted for each read. On wormhole they cannot be halted again. This breaks multi-read loops (e.g., 1024 reads).
        - On blackhole, debug_risc.read_memory() re-halts the TRISC, so multi-read loops work. Until the debug team provides a workaround, memory reads are limited to blackhole only.
        - We read with TRISC 0 (Risc ID 1) because this is the only core that can be rehalted.

    Args:
        num_tiles: Number of tiles to read from the destination register.
        dest_acc: Whether destination accumulation is enabled or not.

    Prerequisite: Disable flag that clears dest register after packing (in llk_pack_common.h) otherwise you will read garbage values.
        - For BH in pack_dest_section_done_, comment out this line : TT_ZEROACC(p_zeroacc::CLR_HALF, is_fp32_dest_acc_en, 0, ADDR_MOD_1, (dest_offset_id) % 2);

    Note:
        - The destination register is read from the address 0xFFBD8000.
        - Number of tiles that can fit in dest register depends on size of datum. If dest register is in 32 bit mode (dest accumulation is enabled), num_tiles must be ≤ 8. Otherwise, ≤ 16.
    """

    risc_id = 1  # we want to use TRISC 0 for reading the destination register
    noc_id = 0  # NOC ID for the device
    device_id = 0  # Device ID for the device
    location = "0,0"  # Core location in the format "tile_id,risc_id"
    base_address = 0xFFBD8000

    context = check_context()
    validate_device_id(device_id, context)
    coordinate = convert_coordinate(location, device_id, context)

    if risc_id != 1:
        raise ValueError(
            "Risc id is not 1. Only TRISC 0 can be halted and read from memory."
        )

    location = RiscLocation(loc=coordinate, noc_id=noc_id, risc_id=risc_id)
    debug_risc = RiscDebug(location=location, context=context, verbose=False)

    assert num_tiles <= (8 if dest_acc == DestAccumulation.Yes else 16)

    word_size = 4  # bytes per 32-bit integer
    num_words = num_tiles * 1024
    addresses = [base_address + i * word_size for i in range(num_words)]

    with debug_risc.ensure_halted():
        dest_reg = [debug_risc.read_memory(addr) for addr in addresses]

    return dest_reg


def is_assert_hit(risc_name, core_loc="0,0", device_id=0):
    # check if the core is stuck on an EBREAK instruction

    context = check_context()
    device = context.devices[device_id]
    coordinate = convert_coordinate(core_loc, device_id, context)
    block = device.get_block(coordinate)
    risc_debug = block.get_risc_debug(risc_name)

    return risc_debug.is_ebreak_hit()


def _print_callstack(risc_name: str, callstack: list[CallstackEntry]):
    print(f"====== ASSERT HIT ON RISC CORE {risc_name.upper()} =======")

    LLK_HOME = Path(os.environ.get("LLK_HOME"))
    TESTS_DIR = LLK_HOME / "tests"

    for idx, entry in enumerate(callstack):
        # Format PC hex like Rust does

        pc = f"0x{entry.pc:016x}" if entry.pc is not None else "0x????????????????"
        file_path = (TESTS_DIR / Path(entry.file)).resolve()

        # first line: idx, pc, function
        print(f"{idx:>4}: {pc} - {entry.function_name}")

        # second line: file, line, column
        print(f"{' '*25}| at {file_path}:{entry.line}:{entry.column}")


def handle_if_assert_hit(elfs: list[str], core_loc="0,0", device_id=0):
    trisc_cores = [RiscCore.TRISC0, RiscCore.TRISC1, RiscCore.TRISC2]
    assertion_hits = []

    for core in trisc_cores:
        risc_name = str(core)
        if is_assert_hit(risc_name, core_loc=core_loc, device_id=device_id):
            _print_callstack(
                risc_name,
                callstack(core_loc, elfs, risc_name=risc_name, device_id=device_id),
            )
            assertion_hits.append(risc_name)

    HardwareController().reset_card()

    if assertion_hits:
        raise AssertionError(
            f"Assert was hit on device on cores: {', '.join(assertion_hits)}"
        )


def wait_for_tensix_operations_finished(elfs, core_loc="0,0", timeout=5, max_backoff=5):
    """
    Polls a value from the device with an exponential backoff timer and fails if it doesn't read 1 within the timeout.

    Args:
        location: The location of the core to poll.
        mailbox_addr: The mailbox address to read from.
        timeout: Maximum time to wait (in seconds) before timing out. Default is 30 seconds. If running on a simulator it is 600 seconds.
        max_backoff: Maximum backoff time (in seconds) between polls. Default is 5 seconds.
    """

    mailboxes = {Mailbox.Unpacker, Mailbox.Math, Mailbox.Packer}

    test_target = TestTargetConfig()
    timeout = 600 if test_target.run_simulator else timeout

    start_time = time.time()
    backoff = 0.1  # Initial backoff time in seconds

    completed = set()
    end_time = start_time + timeout
    while time.time() < end_time:
        for mailbox in mailboxes - completed:
            if read_word_from_device(core_loc, mailbox.value) == KERNEL_COMPLETE:
                completed.add(mailbox)

        if completed == mailboxes:
            return

        # Disable any waiting if running on simulator
        # this makes simulator tests run ever so slightly faster
        if not test_target.run_simulator:
            time.sleep(backoff)
            backoff = min(backoff * 2, max_backoff)  # Exponential backoff with a cap

    handle_if_assert_hit(
        elfs,
        core_loc=core_loc,
    )

    trisc_hangs = [mailbox.name for mailbox in (mailboxes - completed)]
    raise TimeoutError(
        f"Timeout reached: waited {timeout} seconds for {', '.join(trisc_hangs)}"
    )


def reset_mailboxes():
    """Reset all core mailboxes before each test."""
    location = "0, 0"
    reset_value = 0  # Constant - indicates the TRISC kernel run status
    mailboxes = [Mailbox.Packer, Mailbox.Math, Mailbox.Unpacker]
    for mailbox in mailboxes:
        write_words_to_device(location=location, addr=mailbox.value, data=reset_value)
