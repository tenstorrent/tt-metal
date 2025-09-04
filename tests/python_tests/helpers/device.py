# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import inspect
import os
import time
from enum import Enum, IntEnum
from pathlib import Path

from ttexalens.coordinate import OnChipCoordinate
from ttexalens.debug_tensix import TensixDebug
from ttexalens.tt_exalens_lib import (
    check_context,
    load_elf,
    read_from_device,
    read_word_from_device,
    write_to_device,
    write_words_to_device,
)

from helpers.chip_architecture import ChipArchitecture, get_chip_architecture

from .format_arg_mapping import (
    DestAccumulation,
    Mailbox,
)
from .format_config import DataFormat, FormatConfig
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

MAX_READ_BYTE_SIZE_16BIT = 2048

# Constants for soft reset operation
TRISC_SOFT_RESET_MASK = 0x7800  # Reset mask for TRISCs (unpack, math, pack) and BRISC

# Constant - indicates the TRISC kernel run status
KERNEL_COMPLETE = 1  # Kernel completed its run


class BootMode(Enum):
    BRISC = "brisc"
    TRISC = "trisc"
    EXALENS = "exalens"


class RiscCore(IntEnum):
    BRISC = 11
    TRISC0 = 12
    TRISC1 = 13
    TRISC2 = 14


def collect_results(
    formats: FormatConfig,
    tile_count: int,
    address: int = 0x1C000,
    location: str = "0,0",
    sfpu: bool = False,
    tile_dimensions=[32, 32],
    num_faces: int = 4,
):
    # Calculate tile elements based on tile dimensions instead of hardcoding 1024
    tile_elements = tile_dimensions[0] * tile_dimensions[1]

    # Calculate bytes needed based on format and actual tile size
    read_bytes_cnt = (
        formats.output_format.num_bytes_per_tile(tile_elements) * tile_count
    )

    read_data = read_from_device(location, address, num_bytes=read_bytes_cnt)
    res_from_L1 = unpack_res_tiles(
        read_data, formats, tile_count=tile_count, sfpu=sfpu, num_faces=num_faces
    )
    return res_from_L1


def perform_tensix_soft_reset(location="0,0"):
    context = check_context()
    device = context.devices[0]
    chip_coordinate = OnChipCoordinate.create(location, device=device)
    noc_block = device.get_block(chip_coordinate)
    register_store = noc_block.get_register_store()

    # Read current soft reset register, set TRISC reset bits, and write back
    soft_reset = register_store.read_register("RISCV_DEBUG_REG_SOFT_RESET_0")
    soft_reset |= TRISC_SOFT_RESET_MASK
    register_store.write_register("RISCV_DEBUG_REG_SOFT_RESET_0", soft_reset)


def run_cores(cores: list[RiscCore], device_id=0, location="0,0"):
    context = check_context()
    device = context.devices[device_id]
    chip_coordinate = OnChipCoordinate.create(location, device=device)
    noc_block = device.get_block(chip_coordinate)
    register_store = noc_block.get_register_store()

    core_mask = 0
    for core in cores:
        core_mask |= 1 << core.value

    soft_reset = register_store.read_register("RISCV_DEBUG_REG_SOFT_RESET_0")
    soft_reset &= ~core_mask
    register_store.write_register("RISCV_DEBUG_REG_SOFT_RESET_0", soft_reset)


def exalens_device_setup(chip_arch, device_id=0, location="0,0"):
    context = check_context()
    device = context.devices[device_id]
    chip_coordinate = OnChipCoordinate.create(location, device=device)
    debug_tensix = TensixDebug(chip_coordinate, device_id, context)
    ops = debug_tensix.device.instructions

    if chip_arch == ChipArchitecture.BLACKHOLE:
        register_store = device.get_block(chip_coordinate).get_register_store()
        register_store.write_register("RISCV_DEBUG_REG_DEST_CG_CTRL", 0)
        debug_tensix.inject_instruction(ops.TT_OP_ZEROACC(3, 0, 0, 1, 0), 0)
    else:
        debug_tensix.inject_instruction(ops.TT_OP_ZEROACC(3, 0, 0), 0)

    debug_tensix.inject_instruction(ops.TT_OP_SFPENCC(3, 0, 0, 10), 0)
    debug_tensix.inject_instruction(ops.TT_OP_NOP(), 0)

    debug_tensix.inject_instruction(ops.TT_OP_SFPCONFIG(0, 11, 1), 0)

    debug_tensix.inject_instruction(ops.TT_OP_SEMINIT(1, 0, 2), 0)
    debug_tensix.inject_instruction(ops.TT_OP_SEMINIT(1, 0, 7), 0)
    debug_tensix.inject_instruction(ops.TT_OP_SEMINIT(1, 0, 4), 0)


def run_elf_files(testname, device_id=0, location="0,0", boot_mode=BootMode.BRISC):
    CHIP_ARCH = get_chip_architecture()
    LLK_HOME = os.environ.get("LLK_HOME")
    BUILD_DIR = Path(LLK_HOME) / "tests" / "build" / CHIP_ARCH.value

    # Perform soft reset
    perform_tensix_soft_reset(location)

    # Load TRISC ELF files
    trisc_names = ["unpack", "math", "pack"]
    for i, trisc_name in enumerate(trisc_names):
        elf_path = BUILD_DIR / "tests" / testname / "elf" / f"{trisc_name}.elf"
        load_elf(
            elf_file=str(elf_path.absolute()),
            location=location,
            risc_name=f"trisc{i}",
        )

    # Reset the profiler barrier
    TRISC_PROFILER_BARRIE_ADDRESS = 0x16AFF4
    write_words_to_device(location, TRISC_PROFILER_BARRIE_ADDRESS, [0, 0, 0])

    match boot_mode:
        case BootMode.BRISC:
            brisc_elf_path = BUILD_DIR / "shared" / "elf" / "brisc.elf"
            load_elf(
                elf_file=str(brisc_elf_path.absolute()),
                location=location,
                risc_name="brisc",
            )
            run_cores([RiscCore.BRISC], device_id, location)
        case BootMode.TRISC:
            run_cores([RiscCore.TRISC0], device_id, location)
        case BootMode.EXALENS:
            exalens_device_setup(CHIP_ARCH, device_id, location)
            run_cores(
                [RiscCore.TRISC0, RiscCore.TRISC1, RiscCore.TRISC2], device_id, location
            )


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
):
    """
    Write matmul stimuli to L1 with different matrix sizes.

    Args:
        test_config: Used to store addresses of A B and Result
        buffer_A: Flattened tensor data for matrix A
        buffer_B: Flattened tensor data for matrix B
        stimuli_A_format: DataFormat for matrix A
        stimuli_B_format: DataFormat for matrix B
        tile_count_A: Number of tiles in matrix A
        tile_count_B: Number of tiles in matrix B
        location: Core location string

    Returns:
        int: Address where result will be stored
    """

    TILE_ELEMENTS = 1024

    # Calculate L1 addresses
    tile_size_A_bytes = stimuli_A_format.num_bytes_per_tile(TILE_ELEMENTS)
    tile_size_B_bytes = stimuli_B_format.num_bytes_per_tile(TILE_ELEMENTS)
    buffer_A_address = 0x1A000
    buffer_B_address = buffer_A_address + tile_size_A_bytes * tile_count_A
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

    if not pack_function_A or not pack_function_B:
        raise ValueError(
            f"Unsupported data formats: {stimuli_A_format.name}, {stimuli_B_format.name}"
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

    # Set buffer addresses in device to be defined in build header
    test_config["buffer_A_address"] = buffer_A_address
    test_config["buffer_B_address"] = buffer_B_address
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

    from ttexalens.debug_risc import RiscDebug, RiscLoc
    from ttexalens.tt_exalens_lib import (
        check_context,
        convert_coordinate,
        validate_device_id,
    )

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

    location = RiscLoc(loc=coordinate, noc_id=noc_id, risc_id=risc_id)
    debug_risc = RiscDebug(location=location, context=context, verbose=False)

    assert num_tiles <= (8 if dest_acc == DestAccumulation.Yes else 16)

    word_size = 4  # bytes per 32-bit integer
    num_words = num_tiles * 1024
    addresses = [base_address + i * word_size for i in range(num_words)]

    with debug_risc.ensure_halted():
        dest_reg = [debug_risc.read_memory(addr) for addr in addresses]

    return dest_reg


def wait_until_tensix_complete(location, mailbox_addr, timeout=30, max_backoff=5):
    """
    Polls a value from the device with an exponential backoff timer and fails if it doesn't read 1 within the timeout.

    Args:
        location: The location of the core to poll.
        mailbox_addr: The mailbox address to read from.
        timeout: Maximum time to wait (in seconds) before timing out. Default is 30 seconds. If running on a simulator it is 600 seconds.
        max_backoff: Maximum backoff time (in seconds) between polls. Default is 5 seconds.
    """
    test_target = TestTargetConfig()
    timeout = 600 if test_target.run_simulator else timeout

    start_time = time.time()
    backoff = 0.1  # Initial backoff time in seconds

    while time.time() - start_time < timeout:
        if read_word_from_device(location, mailbox_addr.value) == KERNEL_COMPLETE:
            return

        time.sleep(backoff)
        # Disable exponential backoff if running on simulator
        # The simulator sits idle due to no polling - If it is idle for too long, it gets stuck
        if not test_target.run_simulator:
            backoff = min(backoff * 2, max_backoff)  # Exponential backoff with a cap

    raise TimeoutError(
        f"Timeout reached: waited {timeout} seconds for {mailbox_addr.name}"
    )


def wait_for_tensix_operations_finished(location: str = "0,0"):
    wait_until_tensix_complete(location, Mailbox.Packer)
    wait_until_tensix_complete(location, Mailbox.Math)
    wait_until_tensix_complete(location, Mailbox.Unpacker)


def reset_mailboxes():
    """Reset all core mailboxes before each test."""
    location = "0, 0"
    reset_value = 0  # Constant - indicates the TRISC kernel run status
    mailboxes = [Mailbox.Packer, Mailbox.Math, Mailbox.Unpacker]
    for mailbox in mailboxes:
        write_words_to_device(location=location, addr=mailbox.value, data=reset_value)
