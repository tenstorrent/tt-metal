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
    ApproximationMode,
    DestAccumulation,
    L1BufferLocations,
    MathFidelity,
    MathOperation,
    ReduceDimension,
    ReducePool,
    format_tile_sizes,
)
from .format_config import FormatConfig, InputOutputFormat
from .utils import run_shell_command


class ProfilerBuild(Enum):
    Yes = "true"
    No = "false"


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
        "#include <type_traits>",
        "",
        '#include "perf.h"',
        '#include "tensix_types.h"',
        "",
        "#pragma once",
        "",
        "// Basic configuration",
        "#define TILE_SIZE_CNT 0x1000",
    ]

    # Profiler configuration
    if profiler_build == ProfilerBuild.Yes:
        header_content.append("#define LLK_PROFILER")

    # Dest accumulation
    dest_acc = test_config.get("dest_acc", DestAccumulation.No)
    if dest_acc == DestAccumulation.Yes or dest_acc == "DEST_ACC":
        header_content.append("#define DEST_ACC")

    # Unpack to dest
    unpack_to_dest = str(test_config.get("unpack_to_dest", False)).lower()
    header_content.append(f"#define UNPACKING_TO_DEST {unpack_to_dest}")

    # Math fidelity & Approximation mode
    header_content.append(
        f"#define MATH_FIDELITY {test_config.get('math_fidelity', MathFidelity.LoFi).value}"
    )
    header_content.append(
        f"#define APPROX_MODE {test_config.get('approx_mode', ApproximationMode.No).value}"
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
        header_content.extend(
            ["", "// Math operation configuration", f"#define {mathop.value}"]
        )
        if mathop in [
            MathOperation.ReduceColumn,
            MathOperation.ReduceRow,
            MathOperation.ReduceScalar,
        ]:
            header_content.append(
                f"#define REDUCE_DIM {test_config.get('reduce_dim', ReduceDimension.No).value}"
            )
            header_content.append(
                f"#define POOL_TYPE {test_config.get('pool_type', ReducePool.No).value}"
            )

    tile_cnt = test_config.get("tile_cnt", 1)

    header_content.append("")
    # Multi-tile test configuration
    header_content.append("// Multi-tile test configuration")
    header_content.append(f"#define TILE_CNT {tile_cnt}")

    # Unpack an result buffer addresses arrrays generations
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

    header_content.append(
        "#if defined(TEST_KERNEL)\n"
        f"constexpr uint32_t BLOCK_CT_DIM = {block_ct_dim}; \n"
        f"constexpr uint32_t BLOCK_RT_DIM = {block_rt_dim}; \n"
        "#endif\n"
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
