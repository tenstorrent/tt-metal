# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from enum import Enum

from .format_arg_mapping import (
    ApproximationMode,
    DestAccumulation,
    MathFidelity,
    MathOperation,
    ReduceDimension,
    ReducePool,
    math_dict,
    pack_dst_dict,
    pack_src_dict,
    unpack_A_dst_dict,
    unpack_A_src_dict,
    unpack_B_dst_dict,
    unpack_B_src_dict,
)
from .format_config import InputOutputFormat


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
    formats = test_config.get("formats")
    if isinstance(formats, InputOutputFormat):
        header_content.append(f"#define {unpack_A_src_dict[formats.input_format]}")
        header_content.append(f"#define {pack_dst_dict[formats.output_format]}")
    else:
        header_content.extend(
            [
                f"#define {unpack_A_src_dict[formats.unpack_A_src]}",
                f"#define {unpack_A_dst_dict[formats.unpack_A_dst]}",
                f"#define {unpack_B_src_dict[formats.unpack_B_src]}",
                f"#define {unpack_B_dst_dict[formats.unpack_B_dst]}",
                f"#define {math_dict[formats.math]}",
                f"#define {pack_src_dict[formats.pack_src]}",
                f"#define {pack_dst_dict[formats.pack_dst]}",
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

    # Multiple tiles test specific configuration
    if test_config.get("testname") == "multiple_tiles_eltwise_test":
        header_content.extend(
            [
                "",
                "// Multiple tiles test configuration",
                "#define MULTIPLE_OPS",
                f"#define KERN_CNT {test_config.get('kern_cnt', 1)}",
            ]
        )
        pack_addr_cnt = test_config.get("pack_addr_cnt")
        pack_addrs = test_config.get("pack_addrs")
        if pack_addr_cnt is not None:
            header_content.append(f"#define PACK_ADDR_CNT {pack_addr_cnt}")
        if pack_addrs is not None:
            header_content.append(f"#define PACK_ADDRS {pack_addrs}")

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
    generate_header: bool = True,
):
    """Generate make command. Optionally also generate build.h header file."""

    if generate_header:
        write_build_header(test_config, profiler_build=profiler_build)

    # Simplified make command - only basic build parameters
    make_cmd = f"make -j 6 --silent testname={test_config.get('testname')} "

    return make_cmd
