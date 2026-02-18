# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from conftest import skip_for_wormhole
from helpers.device import (
    read_from_device,
    wait_for_tensix_operations_finished,
    write_to_device,
)
from helpers.format_config import DataFormat
from helpers.golden_generators import (
    BroadcastGolden,
    EltwiseBinaryGolden,
    MatmulGolden,
    get_golden_generator,
)
from helpers.llk_params import (
    BroadcastType,
    DestAccumulation,
    MathFidelity,
    MathOperation,
    format_dict,
)
from helpers.pack import pack_bfp16
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig, TestMode
from helpers.test_variant_parameters import (
    BROADCAST_TYPE,
    DEST_SYNC,
    INPUT_DIMENSIONS,
    MATH_FIDELITY,
    MATH_OP,
    NUM_FACES,
    TILE_COUNT,
)
from helpers.tilize_untilize import tilize_block, untilize_block
from helpers.unpack import unpack_res_tiles
from helpers.utils import passed_test


@skip_for_wormhole
@parametrize(
    formats=input_output_formats([DataFormat.Float16_b]),
    dest_acc=[DestAccumulation.No],
    math_fidelity=[MathFidelity.LoFi],
    input_dimensions=[[32, 32]],
)
def test_sdpa_reinits(
    formats,
    dest_acc,
    math_fidelity,
    input_dimensions,
    workers_tensix_coordinates,
):
    """
    Test for SDPA reinits operations using sources/sdpa_reinits_test.cpp

    This test validates a sequence of 4 operations with reinitializations:
    1. Operation 0: Matmul(A, B) -> output1 at 0x1b000
    2. Operation 1: ReduceBlockMax(A) -> output2 at 0x1b800
    3. Operation 2: Elwsub(A, B) with column broadcast -> output3 at 0x1c000
    4. Operation 3: Matmul(A, B) -> output4 at 0x1c800

    The test exercises reinitializations between different operation types
    to ensure proper hardware reconfiguration.
    """

    # Generate input stimuli
    # Note: src_b_const_value: 1.0 in YAML means src_B is constant 1.0, not random
    src_A, tile_cnt_A, _, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )

    # src_B is constant 1.0 as per YAML config (src_b_const_value: 1.0)
    src_B = torch.ones(
        input_dimensions[0],
        input_dimensions[1],
        dtype=format_dict[formats.input_format],
    )
    tile_cnt_B = (input_dimensions[0] // 32) * (input_dimensions[1] // 32)

    # Tilize inputs for hardware
    tilized_A = tilize_block(
        src_A, dimensions=input_dimensions, stimuli_format=formats.input_format
    )
    tilized_B = tilize_block(
        src_B, dimensions=input_dimensions, stimuli_format=formats.input_format
    )

    # GOLDEN GENERATION - All 4 operations as defined in sdpa_reinits.yaml
    # ========================================================================

    # Operation 0: Matmul(A, B) -> output1
    matmul_golden = get_golden_generator(MatmulGolden)
    golden_output1 = matmul_golden(
        src_A,
        src_B,
        formats.output_format,
        math_fidelity,
        input_A_dimensions=input_dimensions,
        input_B_dimensions=input_dimensions,
        tilize=True,
    )

    # Operation 1: ReduceBlockMax(A) -> output2
    # For ReduceBlockMax, we need to compute max per row in tilized space
    # The hardware reads tilized data and outputs max in column 0 of each row within the tile
    src_A_untilized = untilize_block(
        tilized_A.flatten(), formats.input_format, input_dimensions
    )

    # Compute reduce: for each row, find max across the row (first ct_dim*32 elements)
    golden_output2_untilized = torch.zeros_like(src_A_untilized)
    for row in range(input_dimensions[0]):
        golden_output2_untilized[row, 0] = torch.max(src_A_untilized[row, :])

    # Tilize the result to match hardware output format
    golden_output2 = tilize_block(
        golden_output2_untilized,
        dimensions=input_dimensions,
        stimuli_format=formats.output_format,
    ).flatten()

    # Operation 2: Elwsub(A, B) with column broadcast -> output3
    # First broadcast src_B for column broadcast, then apply eltwise sub
    broadcast_golden = get_golden_generator(BroadcastGolden)
    src_B_broadcasted = broadcast_golden(
        BroadcastType.Column,
        tilized_B.flatten(),
        formats.input_format,
        num_faces=4,
        tile_cnt=tile_cnt_A,
        face_r_dim=16,
    )
    binary_golden = get_golden_generator(EltwiseBinaryGolden)
    golden_output3 = binary_golden(
        MathOperation.Elwsub,
        tilized_A.flatten(),
        src_B_broadcasted,
        formats.output_format,
        math_fidelity,
    )

    # Operation 3: Matmul(A, B) -> output4
    golden_output4 = matmul_golden(
        src_A,
        src_B,
        formats.output_format,
        math_fidelity,
        input_A_dimensions=input_dimensions,
        input_B_dimensions=input_dimensions,
        tilize=True,
    )

    # Calculate tile count
    tile_rows, tile_cols = input_dimensions
    output_tile_cnt = (tile_rows // 32) * (tile_cols // 32)

    # Build the test
    configuration = TestConfig(
        "sources/sdpa_reinits_test.cpp",
        formats,
        templates=[
            MATH_FIDELITY(math_fidelity),
            INPUT_DIMENSIONS(input_dimensions, input_dimensions),
            BROADCAST_TYPE(BroadcastType.Column),
            MATH_OP(mathop=MathOperation.Elwsub),
            DEST_SYNC(),
        ],
        runtimes=[
            NUM_FACES(),
            TILE_COUNT(output_tile_cnt),
        ],
        variant_stimuli=None,  # We'll write stimuli manually
        dest_acc=dest_acc,
        unpack_to_dest=False,
    )

    # Build ELFs
    configuration.generate_variant_hash()
    if TestConfig.MODE in [TestMode.PRODUCE, TestMode.DEFAULT]:
        configuration.build_elfs()

    if TestConfig.MODE == TestMode.PRODUCE:
        pytest.skip(TestConfig.SKIP_JUST_FOR_COMPILE_MARKER)

    # Manually write stimuli to the addresses the CPP file expects
    BUFFER_A_ADDR = 0x1A000
    BUFFER_B_ADDR = 0x1A800
    BUFFER_RES0_ADDR = 0x1B000  # Operation 0 output (Matmul)
    BUFFER_RES1_ADDR = 0x1B800  # Operation 1 output (ReduceBlockMax)
    BUFFER_RES2_ADDR = 0x1C000  # Operation 2 output (Elwsub)
    BUFFER_RES3_ADDR = 0x1C800  # Operation 3 output (Matmul)

    # Write input data to device
    write_to_device(
        workers_tensix_coordinates, BUFFER_A_ADDR, pack_bfp16(tilized_A.flatten())
    )
    write_to_device(
        workers_tensix_coordinates, BUFFER_B_ADDR, pack_bfp16(tilized_B.flatten())
    )

    # Run the test - all 4 operations execute in sequence with reinits
    elfs = configuration.run_elf_files(workers_tensix_coordinates)
    wait_for_tensix_operations_finished(elfs, workers_tensix_coordinates)

    # Read and validate all 4 outputs
    tile_size = 2048  # Float16_b tile size
    read_bytes_cnt = tile_size * output_tile_cnt
    torch_format = format_dict[formats.output_format]

    # Validate Operation 0: Matmul
    read_data0 = read_from_device(
        workers_tensix_coordinates, BUFFER_RES0_ADDR, num_bytes=read_bytes_cnt
    )
    res_from_L1_0 = unpack_res_tiles(
        read_data0, formats.output_format, output_tile_cnt, False, 4, 16
    )
    res_tensor_0 = torch.tensor(res_from_L1_0, dtype=torch_format)
    if not passed_test(golden_output1, res_tensor_0, formats.output_format):
        assert False, "Operation 0 (Matmul) failed"

    # Validate Operation 1: ReduceBlockMax
    read_data1 = read_from_device(
        workers_tensix_coordinates, BUFFER_RES1_ADDR, num_bytes=read_bytes_cnt
    )
    res_from_L1_1 = unpack_res_tiles(
        read_data1, formats.output_format, output_tile_cnt, False, 4, 16
    )
    res_tensor_1 = torch.tensor(res_from_L1_1, dtype=torch_format)
    if not passed_test(golden_output2, res_tensor_1, formats.output_format):
        assert False, "Operation 1 (ReduceBlockMax) failed"

    # Validate Operation 2: Elwsub with column broadcast
    read_data2 = read_from_device(
        workers_tensix_coordinates, BUFFER_RES2_ADDR, num_bytes=read_bytes_cnt
    )
    res_from_L1_2 = unpack_res_tiles(
        read_data2, formats.output_format, output_tile_cnt, False, 4, 16
    )
    res_tensor_2 = torch.tensor(res_from_L1_2, dtype=torch_format)
    if not passed_test(golden_output3, res_tensor_2, formats.output_format):
        assert False, "Operation 2 (Elwsub) failed"

    # Validate Operation 3: Matmul (final operation after all reinits)
    read_data3 = read_from_device(
        workers_tensix_coordinates, BUFFER_RES3_ADDR, num_bytes=read_bytes_cnt
    )
    res_from_L1_3 = unpack_res_tiles(
        read_data3, formats.output_format, output_tile_cnt, False, 4, 16
    )
    res_tensor_3 = torch.tensor(res_from_L1_3, dtype=torch_format)
    if not passed_test(golden_output4, res_tensor_3, formats.output_format):
        assert False, "Operation 3 (Matmul) failed"
