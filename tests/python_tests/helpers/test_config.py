# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
from enum import Enum
from pathlib import Path

from ttexalens.tt_exalens_lib import (
    read_word_from_device,
)

from .device import run_elf_files, wait_for_tensix_operations_finished
from .format_arg_mapping import (
    FPU_BINARY_OPERATIONS,
    REDUCE_OPERATIONS,
    SFPU_BINARY_OPERATIONS,
    SFPU_UNARY_OPERATIONS,
    ApproximationMode,
    DestAccumulation,
    L1BufferLocations,
    MathFidelity,
    MathOperation,
    Transpose,
    format_tile_sizes,
)
from .format_config import FormatConfig, InputOutputFormat
from .utils import run_shell_command


class ProfilerBuild(Enum):
    Yes = "true"
    No = "false"


def _generate_operation_constants(mathop: MathOperation) -> list[str]:
    """Generate the appropriate operation constants based on the math operation type."""
    constants = []

    if mathop in SFPU_UNARY_OPERATIONS:
        constants.append(
            f"constexpr auto SFPU_UNARY_OPERATION = SfpuType::{mathop.cpp_enum_value};"
        )
    elif mathop in SFPU_BINARY_OPERATIONS:
        constants.append(
            f"constexpr auto SFPU_BINARY_OPERATION = ckernel::BinaryOp::{mathop.cpp_enum_value};"
        )
    elif mathop in FPU_BINARY_OPERATIONS:
        constants.append(
            f"constexpr auto ELTWISE_BINARY_OP = ckernel::EltwiseBinaryType::{mathop.cpp_enum_value};"
        )

    return constants


def generate_build_header(
    test_config, profiler_build: ProfilerBuild = ProfilerBuild.No
):
    """
    Generate the contents of a C++ header file (build.h) with all configuration defines.

    This function creates a list of preprocessor #define statements based on the provided
    test configuration and profiler build option. The generated header is used to control
    build-time options for tests, such as data formats, math fidelity, accumulation modes,
    and other test-specific parameters.

    The resulting header content includes:
      - Basic configuration constants
      - Profiler and accumulation settings
      - Data format and math operation defines
      - Special configuration for multi-tile tests

    Args:
        test_config (dict): Dictionary containing test configuration parameters.
        profiler_build (ProfilerBuild, optional): Whether to enable profiler defines.

    Returns:
        str: The complete contents of the build.h header file as a string.

    File location: <repository>/tests/helpers/include/build.h
    """
    header_content = [
        "// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC",
        "//",
        "// SPDX-License-Identifier: Apache-2.0",
        "// AUTO-GENERATED CONFIGURATION HEADER. DO NOT EDIT MANUALLY!",
        "",
        "#pragma once",
        "",
        "#include <type_traits>",
        "",
        '#include "llk_defs.h"',
        '#include "llk_sfpu_types.h"',
        '#include "perf.h"',
        '#include "tensix_types.h"',
        "",
        "",
        "// Basic configuration",
        "constexpr std::uint32_t TILE_SIZE_CNT = 0x1000;",
    ]

    # Profiler configuration
    if profiler_build == ProfilerBuild.Yes:
        header_content.append("#define LLK_PROFILER")

    # Dest accumulation
    dest_acc = test_config.get("dest_acc", DestAccumulation.No)
    header_content.append(f"constexpr bool dest_acc_en_input = {dest_acc.value};")

    # Unpack to dest
    unpack_to_dest = str(test_config.get("unpack_to_dest", False)).lower()
    header_content.append(f"constexpr bool UNPACKING_TO_DEST = {unpack_to_dest};")

    # Unpack transpose faces
    unpack_transpose_faces = test_config.get(
        "unpack_transpose_faces", Transpose.No.value
    )
    header_content.append(
        f"constexpr bool UNPACK_TRANSPOSE_FACES = {unpack_transpose_faces};"
    )

    # Unpack transpose within face
    unpack_transpose_within_face = str(
        test_config.get("unpack_transpose_within_face", Transpose.No.value)
    ).lower()
    header_content.append(
        f"constexpr bool UNPACK_TRANSPOSE_WITHIN_FACE = {unpack_transpose_within_face};"
    )

    # Throttle level
    throttle = test_config.get("throttle", 0)
    header_content.append(f"constexpr int THROTTLE_LEVEL = {throttle};")

    # Math transpose faces
    math_transpose_faces = str(test_config.get("math_transpose_faces", False)).lower()
    header_content.append(
        f"constexpr bool MATH_TRANSPOSE_FACES = {math_transpose_faces};"
    )

    # Fused Test L1 to L1 : Input of first run is used as input for the second run ...
    # Not fusing: single L1-to-L1 iteration, so we retrieve one format configuration
    # L1_to_L1_iterations is the number of times we perform llk operations from L1 input tensor to L1 output tensor
    # If L1_to_L1_ITERATIONS is 1, we take input tensor from L1 -> unpack -> math -> pack -> L1
    # If L1_to_L1_ITERATIONS is greater than 1, we perform multiple iterations of unpack -> math -> pack, by taking results tensor in L1 to be input tensor of next iteration
    fused_L1_to_L1 = test_config.get("L1_to_L1_iterations", 1)
    header_content.append(
        f"constexpr std::uint32_t L1_to_L1_ITERATIONS = {fused_L1_to_L1};"
    )

    # Math fidelity & Approximation mode
    header_content.append(
        f"constexpr std::uint32_t MATH_FIDELITY = {test_config.get('math_fidelity', MathFidelity.LoFi).value};"
    )
    header_content.append(
        f"constexpr bool APPROX_MODE = {test_config.get('approx_mode', ApproximationMode.No).value};"
    )

    # Data format configuration
    header_content.extend(["", "// Data format configuration"])
    formats = test_config.get("formats", None)
    if isinstance(formats, InputOutputFormat):
        header_content.extend(
            [
                f"// Activating Data Format Inference Model\n",
                f"#define DATA_FORMAT_INFERENCE_MODEL true",
                f"constexpr auto UNPACK_A_IN = static_cast<std::underlying_type_t<DataFormat>>(DataFormat::{formats.input_format.name});",
                f"constexpr auto PACK_OUT = static_cast<std::underlying_type_t<DataFormat>>(DataFormat::{formats.output_format.name});",
            ]
        )
    elif isinstance(formats, FormatConfig):
        header_content.append(f"#define DATA_FORMAT_INFERENCE_MODEL false")
        header_content.extend(
            [
                f"constexpr auto UNPACK_A_IN = static_cast<std::underlying_type_t<DataFormat>>(DataFormat::{formats.unpack_A_src.name});",
                f"constexpr auto UNPACK_A_OUT = static_cast<std::underlying_type_t<DataFormat>>(DataFormat::{formats.unpack_A_dst.name});",
                f"constexpr auto UNPACK_B_IN = static_cast<std::underlying_type_t<DataFormat>>(DataFormat::{formats.unpack_B_src.name});",
                f"constexpr auto UNPACK_B_OUT = static_cast<std::underlying_type_t<DataFormat>>(DataFormat::{formats.unpack_B_dst.name});",
                f"constexpr auto PACK_IN = static_cast<std::underlying_type_t<DataFormat>>(DataFormat::{formats.pack_src.name});",
                f"constexpr auto PACK_OUT = static_cast<std::underlying_type_t<DataFormat>>(DataFormat::{formats.pack_dst.name});",
                f"constexpr auto MATH_FORMAT = static_cast<std::underlying_type_t<DataFormat>>(DataFormat::{formats.math.name});",
            ]
        )

    # Math operation configuration
    mathop = test_config.get("mathop", "no_mathop")
    if mathop != "no_mathop":
        header_content.extend(["", "// Math operation configuration"])
        header_content.extend(_generate_operation_constants(mathop))

        # Handle reduce operations
        if mathop in REDUCE_OPERATIONS:
            header_content.append(
                f"constexpr auto REDUCE_DIM = ckernel::ReduceDim::{mathop.cpp_enum_value};"
            )
            pool_type = test_config.get("pool_type", None)
            if pool_type is not None:
                header_content.append(
                    f"constexpr auto POOL_TYPE = ckernel::PoolType::{pool_type.value};"
                )

    tile_cnt = test_config.get("tile_cnt", 1)

    header_content.append("")
    # Multi-tile test configuration
    header_content.append("// Multi-tile test configuration")
    header_content.append(f"constexpr int TILE_CNT = {tile_cnt};")

    # Unpack an result buffer addresses arrays generations
    buffer_A_address = read_word_from_device("0,0", L1BufferLocations.srcA.value)
    buffer_B_address = read_word_from_device("0,0", L1BufferLocations.srcB.value)
    result_buffer_address = read_word_from_device("0,0", L1BufferLocations.Result.value)

    buffer_A_array = []
    buffer_B_array = []
    buffer_res_array = []

    if formats is not None:
        for i in range(tile_cnt):
            buffer_A_array.append(
                buffer_A_address + i * format_tile_sizes[formats.input_format]
            )
            buffer_B_array.append(
                buffer_B_address + i * format_tile_sizes[formats.input_format]
            )
            buffer_res_array.append(
                result_buffer_address + i * format_tile_sizes[formats.output_format]
            )

    buffer_A_str = ", ".join(
        f"reinterpret_cast<volatile uint32_t*>({hex(addr)})" for addr in buffer_A_array
    )
    buffer_B_str = ", ".join(
        f"reinterpret_cast<volatile uint32_t*>({hex(addr)})" for addr in buffer_B_array
    )
    buffer_res_str = ", ".join(
        f"reinterpret_cast<volatile uint32_t*>({hex(addr)})"
        for addr in buffer_res_array
    )
    header_content.append(
        "#if defined(LLK_TRISC_UNPACK) && defined(TEST_KERNEL)\n"
        "volatile uint32_t* buffer_A[TILE_CNT] = {" + buffer_A_str + "}; \n"
        "volatile uint32_t* buffer_B[TILE_CNT] = {" + buffer_B_str + "}; \n"
        "#endif\n"
        "#if defined(LLK_TRISC_PACK) && defined(TEST_KERNEL)\n"
        "volatile uint32_t* buffer_Res[TILE_CNT] = {" + buffer_res_str + "}; \n"
        "#endif\n"
    )

    input_dimensions = test_config.get("input_dimensions", [32, 32])
    block_ct_dim = input_dimensions[1] // 32
    block_rt_dim = input_dimensions[0] // 32

    header_content.extend(
        [
            "#if defined(TEST_KERNEL)",
            f"constexpr uint32_t BLOCK_CT_DIM = {block_ct_dim};",
            f"constexpr uint32_t BLOCK_RT_DIM = {block_rt_dim};",
            "#endif",
        ]
    )

    if perf_run_type := test_config.get("perf_run_type"):
        header_content.append("")
        header_content.append(
            f"constexpr auto PERF_RUN_TYPE = PerfRunType::{perf_run_type.name};"
        )

    header_content.append("")
    return "\n".join(header_content)


def write_build_header(
    test_config,
    profiler_build: ProfilerBuild = ProfilerBuild.No,
):
    header_content = generate_build_header(test_config, profiler_build)
    with open("../helpers/include/build.h", "w") as f:
        f.write(header_content)


def generate_make_command(
    test_config,
    profiler_build: ProfilerBuild = ProfilerBuild.No,
):
    """Generate make command"""
    # Simplified make command - only basic build parameters
    make_cmd = f"make -j 6 --silent testname={test_config.get('testname')} all "

    if profiler_build == ProfilerBuild.Yes:
        make_cmd += "profiler "

    return make_cmd


def build_test(
    test_config,
    profiler_build: ProfilerBuild = ProfilerBuild.No,
):
    """Only builds the files required to run a test"""

    root = os.environ.get("LLK_HOME")
    if not root:
        raise AssertionError("Environment variable LLK_HOME is not set")

    TESTS_DIR = str((Path(root) / "tests").absolute())

    write_build_header(test_config, profiler_build=profiler_build)
    make_cmd = generate_make_command(test_config, profiler_build=profiler_build)
    run_shell_command(make_cmd, cwd=TESTS_DIR)


def run_test(
    test_config,
    profiler_build: ProfilerBuild = ProfilerBuild.No,
):
    """Run the test with the given configuration"""

    build_test(test_config, profiler_build=profiler_build)

    # run test
    run_elf_files(test_config["testname"])
    wait_for_tensix_operations_finished()
