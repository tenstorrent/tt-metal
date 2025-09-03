# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
from enum import Enum
from pathlib import Path

from .device import BootMode, run_elf_files, wait_for_tensix_operations_finished
from .dimensions import validate_tile_dimensions
from .format_arg_mapping import (
    FPU_BINARY_OPERATIONS,
    REDUCE_OPERATIONS,
    SFPU_BINARY_OPERATIONS,
    SFPU_UNARY_OPERATIONS,
    ApproximationMode,
    DestAccumulation,
    DestSync,
    MathFidelity,
    MathOperation,
    StochasticRounding,
    Transpose,
    format_tile_sizes,
)
from .format_config import DataFormat, FormatConfig, InputOutputFormat
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
    test_config,
    profiler_build: ProfilerBuild = ProfilerBuild.No,
    boot_mode: BootMode = BootMode.BRISC,
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
        boot_mode (BootMode, optional): Which core / host performs initial device setup.

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

    loop_factor = test_config.get("loop_factor", 1)

    if profiler_build == ProfilerBuild.No and loop_factor != 1:
        raise ValueError(
            "test_config['loop_factor'] should only be used when profiler is enabled"
        )

    header_content.append(f"constexpr int LOOP_FACTOR = {loop_factor};")

    if boot_mode == BootMode.BRISC:
        header_content.append("#define LLK_BOOT_MODE_BRISC")
    elif boot_mode == BootMode.TRISC:
        header_content.append("#define LLK_BOOT_MODE_TRISC")

    # Dest accumulation
    dest_acc = test_config.get("dest_acc", DestAccumulation.No)
    header_content.append(f"constexpr bool dest_acc_en_input = {dest_acc.value};")

    # Unpack to dest
    unpack_to_dest = str(test_config.get("unpack_to_dest", False)).lower()
    header_content.append(f"constexpr bool UNPACKING_TO_DEST = {unpack_to_dest};")

    # Unpack transpose faces
    unpack_transpose_faces = test_config.get(
        "unpack_transpose_faces", Transpose.No
    ).value
    header_content.append(
        f"constexpr bool UNPACK_TRANSPOSE_FACES = {unpack_transpose_faces};"
    )

    # Unpack transpose within face
    unpack_transpose_within_face = test_config.get(
        "unpack_transpose_within_face", Transpose.No
    ).value
    header_content.append(
        f"constexpr bool UNPACK_TRANSPOSE_WITHIN_FACE = {unpack_transpose_within_face};"
    )

    # Throttle level
    throttle = test_config.get("throttle", 0)
    header_content.append(f"constexpr int THROTTLE_LEVEL = {throttle};")

    # Math transpose faces
    math_transpose_faces = test_config.get("math_transpose_faces", Transpose.No).value
    header_content.append(
        f"constexpr bool MATH_TRANSPOSE_FACES = {math_transpose_faces};"
    )
    # Stochastic Rounding
    stochastic_rnd = test_config.get("stochastic_rnd", StochasticRounding.No)
    header_content.append(
        f"constexpr auto STOCHASTIC_RND = ckernel::{stochastic_rnd.value};"
    )

    formats = test_config.get("formats")
    if formats:
        # Tile size mapping
        TILE_SIZES = {
            DataFormat.Bfp8_b: 68,
            DataFormat.Float32: 256,
        }

        pack_size = TILE_SIZES.get(formats.output_format, 128)
        unpack_size = TILE_SIZES.get(formats.input_format, 128)

        header_content.append(f"constexpr std::uint32_t TILE_SIZE_PACK = {pack_size};")
        header_content.append(
            f"constexpr std::uint32_t TILE_SIZE_UNPACK = {unpack_size};"
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

    # Number of faces
    num_faces = test_config.get("num_faces", 4)
    header_content.append(f"constexpr int num_faces = {num_faces};")

    # Dest synchronisation mode
    dest_sync = test_config.get("dest_sync", DestSync.Half)
    header_content.append(
        f"constexpr auto dest_sync = ckernel::DstSync::Sync{dest_sync.name};"
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

    # Optional extra unary operation (used when both a binary and unary op
    # need to be present in the same kernel, e.g. binary-eltwise followed by
    # SFPU unary).  If 'unary_op' exists, append its constant.
    unary_extra = test_config.get("unary_op", None)
    if unary_extra is not None:
        # Only add if we haven't already added a unary operation from the main mathop
        if mathop == "no_mathop" or mathop not in SFPU_UNARY_OPERATIONS:
            header_content.extend(["", "// Additional SFPU unary operation"])
            header_content.append(
                f"constexpr auto SFPU_UNARY_OPERATION = SfpuType::{unary_extra.cpp_enum_value};"
            )

    # Destination sync mode configuration
    dst_sync = test_config.get("dst_sync", None)
    if dst_sync is not None:
        header_content.extend(["", "// Destination sync configuration"])
        header_content.append(
            f"constexpr auto DST_SYNC = ckernel::DstSync::{dst_sync.value};"
        )

    tile_cnt = test_config.get("tile_cnt", 1)

    header_content.append("")
    # Multi-tile test configuration
    header_content.append("// Multi-tile test configuration")
    header_content.append(f"constexpr int TILE_CNT = {tile_cnt};")

    # Unpack + result buffer addresses arrays generations
    buffer_A_address = test_config.get("buffer_A_address", 0x1A000)
    buffer_B_address = test_config.get("buffer_B_address", 0x1B000)
    result_buffer_address = test_config.get("result_buffer_address", 0x1C000)

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

    input_A_dimensions = test_config.get("input_A_dimensions", [32, 32])
    input_B_dimensions = test_config.get("input_B_dimensions", [32, 32])

    num_rows = 32
    num_cols = 32
    validate_tile_dimensions(input_A_dimensions[0], num_cols)
    validate_tile_dimensions(input_B_dimensions[1], num_rows)
    block_rt_dim = input_A_dimensions[0] // num_cols
    block_ct_dim = input_B_dimensions[1] // num_rows

    header_content.extend(
        [
            "#if defined(TEST_KERNEL)",
            f"constexpr uint32_t BLOCK_CT_DIM = {block_ct_dim};",
            f"constexpr uint32_t BLOCK_RT_DIM = {block_rt_dim};",
            "#endif",
        ]
    )

    # Add matrix multiplication tile dimensions if they exist
    if "rt_dim" in test_config:
        header_content.append(f"constexpr uint32_t RT_DIM = {test_config['rt_dim']};")
    if "ct_dim" in test_config:
        header_content.append(f"constexpr uint32_t CT_DIM = {test_config['ct_dim']};")
    if "kt_dim" in test_config:
        header_content.append(f"constexpr uint32_t KT_DIM = {test_config['kt_dim']};")

    header_content.append("")

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
    boot_mode: BootMode = BootMode.BRISC,
):
    header_content = generate_build_header(
        test_config, profiler_build, boot_mode=boot_mode
    )
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
    boot_mode: BootMode = BootMode.BRISC,
):
    """Only builds the files required to run a test"""

    root = os.environ.get("LLK_HOME")
    if not root:
        raise AssertionError("Environment variable LLK_HOME is not set")

    TESTS_DIR = str((Path(root) / "tests").absolute())

    write_build_header(test_config, profiler_build=profiler_build, boot_mode=boot_mode)
    make_cmd = generate_make_command(test_config, profiler_build=profiler_build)
    run_shell_command(make_cmd, cwd=TESTS_DIR)


def run_test(
    test_config,
    profiler_build: ProfilerBuild = ProfilerBuild.No,
    boot_mode: BootMode = BootMode.BRISC,  # change default boot mode here
):
    """Run the test with the given configuration"""

    build_test(test_config, profiler_build=profiler_build, boot_mode=boot_mode)

    # run test
    run_elf_files(test_config["testname"], boot_mode=boot_mode)
    wait_for_tensix_operations_finished()
