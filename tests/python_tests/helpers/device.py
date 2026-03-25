# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import datetime
import os
import time
from enum import Enum, IntEnum
from pathlib import Path
from typing import List

import torch
from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from ttexalens.context import Context
from ttexalens.coordinate import OnChipCoordinate
from ttexalens.debug_tensix import TensixDebug
from ttexalens.hardware.risc_debug import CallstackEntry
from ttexalens.tt_exalens_lib import (
    ParsedElfFile,
    TTException,
    arc_msg,
    callstack,
    check_context,
    convert_coordinate,
    parse_elf,
    read_from_device,
    read_word_from_device,
    write_to_device,
    write_words_to_device,
)

from .fused_operation import FusedOperation
from .llk_params import (
    BriscCmd,
    DataFormat,
    MailboxesPerf,
    MailboxesPerfQuasar,
    format_dict,
)
from .logger import logger
from .pack import (
    pack_bfp8_b,
    pack_bfp16,
    pack_fp8_e4m3,
    pack_fp16,
    pack_fp32,
    pack_int8,
    pack_int16,
    pack_int32,
    pack_uint8,
    pack_uint16,
    pack_uint32,
)
from .tilize_untilize import untilize_block
from .unpack import unpack_res_tiles

Mailbox = (
    MailboxesPerf
    if get_chip_architecture() != ChipArchitecture.QUASAR
    else MailboxesPerfQuasar
)


class LLKAssertException(Exception):
    pass


KERNEL_COMPLETE = 0xFF


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
    # These are now just internal identifiers, not the hardware IDs
    BRISC = 0
    TRISC0 = 1
    TRISC1 = 2
    TRISC2 = 3
    TRISC3 = 4

    @property
    def value(self):
        """Overrides the standard .value to be chip-dependent and lazy."""
        arch = get_chip_architecture()
        is_quasar = arch == ChipArchitecture.QUASAR

        mapping = {
            RiscCore.BRISC: -1 if is_quasar else 11,
            RiscCore.TRISC0: 11 if is_quasar else 12,
            RiscCore.TRISC1: 12 if is_quasar else 13,
            RiscCore.TRISC2: 13 if is_quasar else 14,
            RiscCore.TRISC3: 14 if is_quasar else -1,
        }
        return mapping[self]

    def __repr__(self):
        # This forces the print output to use your lazy .value property
        return f"<{self.__class__.__name__}.{self.name}: {self.value}>"


def get_all_cores(trisc_only: bool = False):
    arch = get_chip_architecture()
    if arch == ChipArchitecture.QUASAR:
        return [RiscCore.TRISC0, RiscCore.TRISC1, RiscCore.TRISC2, RiscCore.TRISC3]

    if trisc_only:
        return [RiscCore.TRISC0, RiscCore.TRISC1, RiscCore.TRISC2]

    return [RiscCore.BRISC, RiscCore.TRISC0, RiscCore.TRISC1, RiscCore.TRISC2]


# Constant - list of all valid cores on the chip
ALL_CORES = get_all_cores()
TRISC_CORES = get_all_cores(trisc_only=True)


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


common_counter = 0


def commit_brisc_command(
    location="0,0", command: BriscCmd = BriscCmd.IDLE_STATE, timeout=0.1
):
    global common_counter

    if common_counter & 1:
        write_words_to_device(location, Mailbox.BriscCommand1.value, [command.value])
    else:
        write_words_to_device(location, Mailbox.BriscCommand0.value, [command.value])

    common_counter += 1
    end_time = time.time() + timeout
    while time.time() < end_time:
        temp_value = read_word_from_device(location, Mailbox.BriscCounter.value, 0)
        if temp_value == common_counter:
            return

    logger.error(f"{command.name} -> {hex(Mailbox.BriscCommand0.value)}")

    raise TimeoutError("Polling brisc command timed out")


def assert_if_all_in_reset(location: str = "0,0", place: str = ""):
    soft_reset = get_register_store(location, 0).read_register(
        "RISCV_DEBUG_REG_SOFT_RESET_0"
    )

    if (soft_reset & get_soft_reset_mask(ALL_CORES)) != get_soft_reset_mask(ALL_CORES):
        raise Exception(f"Not all cores are in reset! {place}")


def exalens_device_setup(chip_arch, location="0,0", device_id=0):
    context = check_context()
    device = context.devices[device_id]
    chip_coordinate = OnChipCoordinate.create(location, device=device)
    debug_tensix = TensixDebug(chip_coordinate)
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


def is_assert_hit(risc_name, core_loc="0,0", device_id=0):
    # check if the core is stuck on an EBREAK instruction

    CHIP_ARCH = get_chip_architecture()
    context = check_context()
    device = context.devices[device_id]
    coordinate = convert_coordinate(core_loc, device_id, context)
    block = device.get_block(coordinate)
    risc_debug = block.get_risc_debug(
        risc_name, neo_id=0 if CHIP_ARCH == ChipArchitecture.QUASAR else None
    )

    is_it = True

    try:
        is_it = risc_debug.is_ebreak_hit()
    except:
        raise Exception("WTF handler")

    return is_it


def _print_callstack(risc_name: str, callstack: list[CallstackEntry]) -> str:
    temp_str = f"\n====== {risc_name.upper()} STACK TRACE =======\n"

    LLK_HOME = Path(os.environ.get("LLK_HOME"))
    TESTS_DIR = LLK_HOME / "tests"

    for idx, entry in enumerate(callstack):
        # Format PC hex like Rust does
        pc = f"0x{entry.pc:016x}" if entry.pc is not None else "0x????????????????"
        file_path = (TESTS_DIR / Path(entry.file)).resolve()
        # first line: idx, pc, function
        temp_str += f"{idx:>4}: {pc} - {entry.function_name}\n"
        # second line: file, line, column
        temp_str += f"{' '*25}| at {file_path}:{entry.line}:{entry.column}\n"

    return temp_str


def handle_if_assert_hit(elfs: list[str], core_loc="0,0", device_id=0):
    assertion_hits = []
    temp_stack_traces = ""
    for core in TRISC_CORES:
        risc_name = str(core.name)
        if is_assert_hit(risc_name, core_loc=core_loc, device_id=device_id):
            temp_stack_traces += _print_callstack(
                risc_name,
                callstack(core_loc, elfs, risc_name=risc_name, device_id=device_id),
            )
            assertion_hits.append(risc_name)

    if assertion_hits:
        raise LLKAssertException(temp_stack_traces)


def reset_mailboxes(location: str = "0,0"):
    """Reset all core mailboxes (Unpacker, Math, Packer, Sfpu) before each test."""

    write_words_to_device(
        location=location, addr=Mailbox.Unpacker.value, data=[0xA3, 0xA3, 0xA3]
    )


def pull_coverage_stream_from_tensix(
    location: str | OnChipCoordinate,
    elf: str | ParsedElfFile,
    stream_path: str,
    device_id: int = 0,
    context: Context | None = None,
) -> None:

    coordinate = convert_coordinate(location, device_id, context)
    context = coordinate.context
    if isinstance(elf, str):
        elf = parse_elf(elf, context)

    COVERAGE_REGION_START_SYM = "__coverage_start"

    coverage_start = elf.symbols[COVERAGE_REGION_START_SYM].value
    if not coverage_start:
        raise TTException(f"{COVERAGE_REGION_START_SYM} not found")

    length = read_word_from_device(location, addr=coverage_start)

    data = read_from_device(location, coverage_start + 4, num_bytes=length - 4)
    with open(stream_path, "wb") as f:
        f.write(data)


def _send_arc_message(message_type: str, device_id: int):
    """Helper to send ARC messages with better abstraction."""
    ARC_COMMON_PREFIX = 0xAA00
    message_codes = {"GO_BUSY": 0x52, "GO_IDLE": 0x54}

    arc_msg(
        device_id=device_id,
        msg_code=ARC_COMMON_PREFIX | message_codes[message_type],
        wait_for_done=True,
        args=[0, 0],
        timeout=datetime.timedelta(seconds=10),
    )


def write_pipeline_operands_to_l1(
    pipeline: List[FusedOperation],
    location: str = "0,0",
):
    TILE_ELEMENTS = 1024
    current_address = 0x21000

    def calculate_size(data_format: DataFormat, tile_count: int) -> int:
        return data_format.num_bytes_per_tile(TILE_ELEMENTS) * tile_count

    def write_operand_data(operand, location):
        packers = {
            DataFormat.Float16: pack_fp16,
            DataFormat.Float16_b: pack_bfp16,
            DataFormat.Float32: pack_fp32,
            DataFormat.Bfp8_b: pack_bfp8_b,
            DataFormat.Int32: pack_int32,
            DataFormat.UInt32: pack_uint32,
            DataFormat.Int16: pack_int16,
            DataFormat.UInt16: pack_uint16,
            DataFormat.Fp8_e4m3: pack_fp8_e4m3,
            DataFormat.Int8: pack_int8,
            DataFormat.UInt8: pack_uint8,
        }

        pack_function = packers.get(operand.data_format)
        if not pack_function:
            raise ValueError(f"Unsupported data format: {operand.data_format.name}")

        tile_size = calculate_size(operand.data_format, 1)
        buffer = operand.data.flatten()

        for i in range(operand.tile_count):
            start_idx = TILE_ELEMENTS * i
            tile_data = buffer[start_idx : start_idx + TILE_ELEMENTS]

            if pack_function == pack_bfp8_b:
                packed_data = pack_function(tile_data, num_faces=4)
            else:
                packed_data = pack_function(tile_data)

            addr = operand.l1_address + i * tile_size
            write_to_device(location, addr, packed_data)

    for operation in pipeline:
        src_a = operation.src_a
        src_b = operation.src_b
        output = operation.output
        if src_a.is_input() and src_a.l1_address is None:
            src_a.l1_address = current_address
            current_address += calculate_size(src_a.data_format, src_a.tile_count)
            write_operand_data(src_a, location)

        if src_b.is_input() and src_b.l1_address is None:
            src_b.l1_address = current_address
            current_address += calculate_size(src_b.data_format, src_b.tile_count)
            write_operand_data(src_b, location)

        if output.l1_address is None:
            output_tile_count = output.tile_count
            output.l1_address = current_address
            current_address += calculate_size(output.data_format, output_tile_count)


def collect_pipeline_results(
    pipeline: List[FusedOperation],
    location: str = "0,0",
):
    TILE_ELEMENTS = 1024

    for operation in pipeline:
        output_operand = operation.output
        output_name = output_operand.name

        if output_operand.l1_address is None:
            raise ValueError(
                f"Output operand '{output_name}' does not have an L1 address. "
            )

        output_dimensions = operation.output.dimensions
        output_format = output_operand.data_format
        tile_cnt = operation.output.tile_count

        read_bytes_cnt = (
            output_operand.data_format.num_bytes_per_tile(TILE_ELEMENTS) * tile_cnt
        )
        read_data = read_from_device(
            location, output_operand.l1_address, num_bytes=read_bytes_cnt
        )

        res_from_L1 = unpack_res_tiles(
            read_data,
            output_format,
            tile_count=tile_cnt,
            sfpu=False,
            num_faces=operation.num_faces,
            face_r_dim=operation.face_r_dim,
        )

        torch_format = format_dict[output_format]
        tilized_tensor = torch.tensor(res_from_L1, dtype=torch_format)

        if output_format != DataFormat.Bfp8_b and output_dimensions is not None:
            raw_tensor = untilize_block(
                tilized_tensor,
                stimuli_format=output_format,
                dimensions=output_dimensions,
            )
        else:
            raw_tensor = tilized_tensor

        output_operand._data = tilized_tensor
        output_operand._raw_data = raw_tensor
