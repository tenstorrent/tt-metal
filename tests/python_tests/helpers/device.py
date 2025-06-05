# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import inspect
import time

from ttexalens.tt_exalens_lib import (
    check_context,
    load_elf,
    read_from_device,
    read_word_from_device,
    run_elf,
    write_to_device,
    write_words_to_device,
)

from .format_arg_mapping import Mailbox
from .format_config import DataFormat, FormatConfig
from .pack import pack_bfp8_b, pack_bfp16, pack_fp16, pack_fp32, pack_int32
from .unpack import unpack_bfp8_b, unpack_bfp16, unpack_fp16, unpack_fp32, unpack_int32
from .utils import calculate_read_byte_count

MAX_READ_BYTE_SIZE_16BIT = 2048


def collect_results(
    formats: FormatConfig,
    tensor_size: int,
    address: int = 0x1C000,
    core_loc: str = "0,0",
    sfpu: bool = False,
):

    read_bytes_cnt = calculate_read_byte_count(formats, tensor_size, sfpu)
    read_data = read_from_device(core_loc, address, num_bytes=read_bytes_cnt)
    res_from_L1 = get_result_from_device(formats, read_data, sfpu)
    return res_from_L1


def run_elf_files(testname, core_loc="0,0", run_brisc=True):
    ELF_LOCATION = "../build/elf/"

    if run_brisc:
        run_elf(f"{ELF_LOCATION}brisc.elf", core_loc, risc_id=0)

    context = check_context()
    device = context.devices[0]
    RISC_DBG_SOFT_RESET0 = device.get_tensix_register_address(
        "RISCV_DEBUG_REG_SOFT_RESET_0"
    )

    # Perform soft reset
    soft_reset = read_word_from_device(core_loc, RISC_DBG_SOFT_RESET0)
    soft_reset |= 0x7800
    write_words_to_device(core_loc, RISC_DBG_SOFT_RESET0, soft_reset)

    # Load ELF files
    for i in range(3):
        load_elf(f"{ELF_LOCATION}{testname}_trisc{i}.elf", core_loc, risc_id=i + 1)

    # Clear soft reset
    soft_reset &= ~0x7800
    write_words_to_device(core_loc, RISC_DBG_SOFT_RESET0, soft_reset)


def write_stimuli_to_l1(
    buffer_A,
    buffer_B,
    stimuli_A_format,
    stimuli_B_format,
    core_loc="0,0",
    tile_cnt=1,
):

    BUFFER_SIZE = 4096
    TILE_SIZE = 1024

    buffer_A_address = 0x1A000
    buffer_B_address = 0x1A000 + BUFFER_SIZE * tile_cnt

    for i in range(tile_cnt):

        start_index = TILE_SIZE * i
        end_index = start_index + TILE_SIZE

        # if end_index > len(buffer_A) or end_index > len(buffer_B):
        #     raise IndexError("Buffer access out of bounds")

        buffer_A_tile = buffer_A[start_index:end_index]
        buffer_B_tile = buffer_B[start_index:end_index]

        packers = {
            DataFormat.Bfp8_b: pack_bfp8_b,
            DataFormat.Float16: pack_fp16,
            DataFormat.Float16_b: pack_bfp16,
            DataFormat.Float32: pack_fp32,
            DataFormat.Int32: pack_int32,
        }

        pack_function_A = packers.get(stimuli_A_format)
        pack_function_B = packers.get(stimuli_B_format)

        write_to_device(core_loc, buffer_A_address, pack_function_A(buffer_A_tile))
        write_to_device(core_loc, buffer_B_address, pack_function_B(buffer_B_tile))

        buffer_A_address += BUFFER_SIZE
        buffer_B_address += BUFFER_SIZE


def get_result_from_device(
    formats: FormatConfig,
    read_data_bytes: bytes,
    core_loc: str = "0,0",
    sfpu: bool = False,
):
    # Dictionary of format to unpacking function mappings
    unpackers = {
        DataFormat.Float16: unpack_fp16,
        DataFormat.Float16_b: unpack_bfp16,
        DataFormat.Float32: unpack_fp32,
        DataFormat.Int32: unpack_int32,
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
            )  # Bug patchup in (unpack.py): in case unpack_src is Bfp8_b != pack_dst, L1 must be read in different order to extract correct results
        else:
            return unpack_func(read_data_bytes)
    else:
        raise ValueError(f"Unsupported format: {formats.output_format}")


def wait_until_tensix_complete(core_loc, mailbox_addr, timeout=30, max_backoff=5):
    """
    Polls a value from the device with an exponential backoff timer and fails if it doesn't read 1 within the timeout.

    Args:
        core_loc: The location of the core to poll.
        mailbox_addr: The mailbox address to read from.
        timeout: Maximum time to wait (in seconds) before timing out. Default is 30 seconds.
        max_backoff: Maximum backoff time (in seconds) between polls. Default is 5 seconds.
    """
    start_time = time.time()
    backoff = 0.1  # Initial backoff time in seconds

    while time.time() - start_time < timeout:
        if read_word_from_device(core_loc, mailbox_addr.value) == 1:
            return

        time.sleep(backoff)
        backoff = min(backoff * 2, max_backoff)  # Exponential backoff with a cap

    assert False, f"Timeout reached: waited {timeout} seconds for {mailbox_addr.name}"


def wait_for_tensix_operations_finished(core_loc: str = "0,0"):

    wait_until_tensix_complete(core_loc, Mailbox.Packer)
    wait_until_tensix_complete(core_loc, Mailbox.Math)
    wait_until_tensix_complete(core_loc, Mailbox.Unpacker)
