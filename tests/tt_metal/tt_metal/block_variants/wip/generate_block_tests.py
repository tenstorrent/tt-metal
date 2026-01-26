#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Block Variants Test Generator

Generates comprehensive tests for block variant Compute API functions.
Each test validates that block operations produce identical results to
tile-by-tile operations.

Usage:
    python generate_block_tests.py --operation add_block --output tests/tt_metal/tt_metal/block_variants/
    python generate_block_tests.py --all  # Generate all tests
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# ===========================================================================
# Test Configuration
# ===========================================================================

OPERATIONS = {
    "eltwise_binary": {
        "ops": ["add", "sub", "mul"],
        "header": "compute_kernel_api/eltwise_binary.h",
        "init_pattern": "binary_op_init_common(cb_in0, cb_in1, cb_out);\n    {op}_tiles_init(cb_in0, cb_in1);",
        "tile_call": "{op}_tiles(cb_in0, cb_in1, tile_idx, tile_idx, tile_idx)",
        "block_call": "{op}_block<Ht, Wt>(cb_in0, cb_in1, 0, 0, 0)",
        "num_inputs": 2,
        "priority": 1,
    },
    "broadcast": {
        "ops": ["add_tiles_bcast", "sub_tiles_bcast", "mul_tiles_bcast"],
        "header": "compute_kernel_api/bcast.h",
        "init_pattern": "init_bcast<ELW{OP}, BroadcastType::COL>(cb_in0, cb_in1, cb_out);",
        "tile_call": "{op}<BroadcastType::COL>(cb_in0, cb_in1, tile_idx, tile_idx, tile_idx)",
        "block_call": "{op}_block<BroadcastType::COL, Ht, Wt>(cb_in0, cb_in1, 0, 0, 0)",
        "num_inputs": 2,
        "priority": 2,
        "broadcast_types": ["ROW", "COL", "SCALAR"],
    },
    "transpose": {
        "ops": ["transpose_wh"],
        "header": "compute_kernel_api/transpose_wh.h",
        "init_pattern": "transpose_wh_init(cb_in, cb_out);",
        "tile_call": "transpose_wh_tile(cb_in, tile_idx, tile_idx)",
        "block_call": "transpose_wh_block<Ht, Wt>(cb_in, 0, 0)",
        "num_inputs": 1,
        "priority": 3,
    },
    "reduce": {
        "ops": ["reduce"],
        "header": "compute_kernel_api/reduce_custom.h",
        "init_pattern": "reduce_init<REDUCE_OP, REDUCE_COL>(cb_in, cb_scaler, cb_out);",
        "tile_call": "reduce_tile<REDUCE_OP, REDUCE_COL>(cb_in, cb_scaler, tile_idx, tile_idx, tile_idx)",
        "block_call": "reduce_block<REDUCE_OP, REDUCE_COL, Ht, Wt>(cb_in, cb_scaler, 0, 0, 0)",
        "num_inputs": 2,  # Input + scaler
        "priority": 2,
        "reduce_types": ["SUM", "AVG", "MAX"],
        "reduce_dims": ["REDUCE_ROW", "REDUCE_COL", "REDUCE_SCALAR"],
    },
    "pack": {
        "ops": ["pack"],
        "header": "compute_kernel_api/pack.h",
        "init_pattern": "pack_tile_init();",
        "tile_call": "pack_tile(tile_idx, cb_out)",
        "block_call": "pack_block<Ht, Wt>(0, cb_out)",
        "num_inputs": 0,  # Pack from DEST
        "priority": 3,
    },
}

# Block sizes to test (Ht × Wt ≤ 16)
BLOCK_SIZES = [
    (1, 1),
    (1, 2),
    (1, 4),
    (1, 8),
    (1, 16),
    (2, 1),
    (2, 2),
    (2, 4),
    (2, 8),
    (4, 1),
    (4, 2),
    (4, 4),
    (8, 1),
    (8, 2),
    (16, 1),
]

# ===========================================================================
# Template Generators
# ===========================================================================


def generate_compute_kernel_tile(operation: str, op_name: str, config: Dict) -> str:
    """Generate reference compute kernel using tile-by-tile operations."""

    kernel = f"""// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

/**
 * Reference Kernel: {op_name} Tile-by-Tile
 *
 * Processes blocks tile-by-tile as a reference implementation.
 * Used to validate block variant correctness.
 */

#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "{config['header']}"

namespace NAMESPACE {{
void MAIN {{
    constexpr uint32_t Ht = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr uint32_t num_blocks = get_compile_time_arg_val(2);

    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_in1 = tt::CBIndex::c_1;
    constexpr auto cb_out = tt::CBIndex::c_2;

    // Initialize operation
    {config['init_pattern'].format(op=op_name.upper(), OP=op_name.upper())}

    // Process blocks tile-by-tile
    for (uint32_t block = 0; block < num_blocks; block++) {{
        // Wait for input tiles
"""

    if config["num_inputs"] >= 1:
        kernel += "        cb_wait_front(cb_in0, Ht * Wt);\n"
    if config["num_inputs"] >= 2:
        kernel += "        cb_wait_front(cb_in1, Ht * Wt);\n"

    kernel += (
        """
        // Acquire DEST
        tile_regs_acquire();

        // Process tile-by-tile
        for (uint32_t h = 0; h < Ht; h++) {
            for (uint32_t w = 0; w < Wt; w++) {
                uint32_t tile_idx = h * Wt + w;
                """
        + config["tile_call"].format(op=op_name)
        + """;
            }
        }

        tile_regs_commit();

        // Pack results
        cb_reserve_back(cb_out, Ht * Wt);
        tile_regs_wait();

        for (uint32_t i = 0; i < Ht * Wt; i++) {
            pack_tile(i, cb_out);
        }

        tile_regs_release();

        // Push and pop
        cb_push_back(cb_out, Ht * Wt);
"""
    )

    if config["num_inputs"] >= 1:
        kernel += "        cb_pop_front(cb_in0, Ht * Wt);\n"
    if config["num_inputs"] >= 2:
        kernel += "        cb_pop_front(cb_in1, Ht * Wt);\n"

    kernel += """    }
}
}  // namespace NAMESPACE
"""
    return kernel


def generate_compute_kernel_block(operation: str, op_name: str, config: Dict) -> str:
    """Generate test compute kernel using block operations."""

    kernel = f"""// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

/**
 * Test Kernel: {op_name} Block Operation
 *
 * Uses block variant to process multiple tiles at once.
 * Must produce identical results to tile-by-tile version.
 */

#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "{config['header']}"

namespace NAMESPACE {{
void MAIN {{
    constexpr uint32_t Ht = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr uint32_t num_blocks = get_compile_time_arg_val(2);

    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_in1 = tt::CBIndex::c_1;
    constexpr auto cb_out = tt::CBIndex::c_2;

    // Initialize operation (same as tile version)
    {config['init_pattern'].format(op=op_name.upper(), OP=op_name.upper())}

    // Process blocks using block operation
    for (uint32_t block = 0; block < num_blocks; block++) {{
        // Wait for input tiles
"""

    if config["num_inputs"] >= 1:
        kernel += "        cb_wait_front(cb_in0, Ht * Wt);\n"
    if config["num_inputs"] >= 2:
        kernel += "        cb_wait_front(cb_in1, Ht * Wt);\n"

    kernel += (
        """
        // Acquire DEST
        tile_regs_acquire();

        // USE BLOCK OPERATION - This is what we're testing!
        """
        + config["block_call"].format(op=op_name)
        + """;

        tile_regs_commit();

        // Pack results using pack_block
        cb_reserve_back(cb_out, Ht * Wt);
        tile_regs_wait();

        pack_block<Ht, Wt>(0, cb_out);

        tile_regs_release();

        // Push and pop
        cb_push_back(cb_out, Ht * Wt);
"""
    )

    if config["num_inputs"] >= 1:
        kernel += "        cb_pop_front(cb_in0, Ht * Wt);\n"
    if config["num_inputs"] >= 2:
        kernel += "        cb_pop_front(cb_in1, Ht * Wt);\n"

    kernel += """    }
}
}  // namespace NAMESPACE
"""
    return kernel


def generate_test_harness(operation: str, config: Dict) -> str:
    """Generate Gtest harness for operation."""

    test = f"""// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

/**
 * Block Variant Tests: {operation}
 *
 * Validates that block operations produce identical results to tile-by-tile processing.
 * Each test runs both a reference (tile-by-tile) and test (block) kernel with identical
 * input data and compares outputs using PCC (Pearson Correlation Coefficient).
 */

#include <gtest/gtest.h>
#include "common/command_queue_fixture.hpp"
#include "test_gold_impls.hpp"
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>

using namespace tt;
using namespace tt::tt_metal;

namespace {{

/**
 * Run {operation} block test
 *
 * @param device Device to run on
 * @param Ht Block height in tiles
 * @param Wt Block width in tiles
 * @param num_blocks Number of blocks to process
 * @param data_format Data format for tiles
 */
void run_{operation}_block_test(
    Device* device,
    uint32_t Ht,
    uint32_t Wt,
    uint32_t num_blocks = 10,
    tt::DataFormat data_format = tt::DataFormat::Float16_b) {{

    // Validate block size
    ASSERT_LE(Ht * Wt, 16) << "Block size exceeds DEST capacity (max 16 tiles)";
    ASSERT_GT(Ht, 0) << "Block height must be > 0";
    ASSERT_GT(Wt, 0) << "Block width must be > 0";

    log_info(LogTest, "Testing {operation} with Ht={{}}, Wt={{}}, blocks={{}}", Ht, Wt, num_blocks);

    // TODO: Implement full test harness
    // This is a template - agents should fill in:
    // 1. Create programs (reference and test)
    // 2. Create buffers and CBs
    // 3. Generate input data
    // 4. Run both programs
    // 5. Compare results with PCC >= 0.9999

    GTEST_SKIP() << "Test implementation pending";
}}

}}  // namespace

// =============================================================================
// Test Cases
// =============================================================================

class BlockVariantsFixture : public tt::tt_metal::CommandQueueFixture {{}};

"""

    # Generate test cases for each block size
    for ht, wt in BLOCK_SIZES:
        test_name = f"{operation.title().replace('_', '')}Block_{ht}x{wt}"
        test += f"""
TEST_F(BlockVariantsFixture, {test_name}) {{
    run_{operation}_block_test(this->device_.get(), {ht}, {wt});
}}
"""

    test += (
        """
// =============================================================================
// Stress Tests
// =============================================================================

TEST_F(BlockVariantsFixture, """
        + operation.title().replace("_", "")
        + """Block_Stress_ManyBlocks) {
    // Process many blocks to test stability
    run_"""
        + operation
        + """_block_test(this->device_.get(), 4, 4, 1000);
}

TEST_F(BlockVariantsFixture, """
        + operation.title().replace("_", "")
        + """Block_Stress_MaxCapacity) {
    // Use maximum DEST capacity
    run_"""
        + operation
        + """_block_test(this->device_.get(), 16, 1, 100);
}

"""
    )

    return test


# ===========================================================================
# Main Script
# ===========================================================================


def generate_tests_for_operation(operation: str, output_dir: Path):
    """Generate all test files for a given operation."""

    if operation not in OPERATIONS:
        print(f"❌ Unknown operation: {operation}")
        print(f"   Available: {', '.join(OPERATIONS.keys())}")
        return False

    config = OPERATIONS[operation]
    print(f"\n📦 Generating tests for: {operation}")
    print(f"   Priority: {config['priority']}")
    print(f"   Operations: {', '.join(config['ops'])}")

    # Create output directories
    kernel_dir = output_dir / "kernels"
    kernel_dir.mkdir(parents=True, exist_ok=True)

    generated_files = []

    # Generate kernels for each operation variant
    for op in config["ops"]:
        # Tile-by-tile reference kernel
        tile_kernel = generate_compute_kernel_tile(operation, op, config)
        tile_file = kernel_dir / f"compute_{op}_tiles.cpp"
        tile_file.write_text(tile_kernel)
        generated_files.append(tile_file)
        print(f"   ✅ {tile_file.name}")

        # Block operation test kernel
        block_kernel = generate_compute_kernel_block(operation, op, config)
        block_file = kernel_dir / f"compute_{op}_block.cpp"
        block_file.write_text(block_kernel)
        generated_files.append(block_file)
        print(f"   ✅ {block_file.name}")

    # Generate test harness
    test_harness = generate_test_harness(operation, config)
    test_file = output_dir / f"test_{operation}_block.cpp"
    test_file.write_text(test_harness)
    generated_files.append(test_file)
    print(f"   ✅ {test_file.name}")

    print(f"\n📊 Generated {len(generated_files)} files for {operation}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Generate block variant tests for Compute API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate tests for specific operation
  python generate_block_tests.py --operation eltwise_binary

  # Generate all tests
  python generate_block_tests.py --all

  # Custom output directory
  python generate_block_tests.py --operation reduce --output /path/to/tests
        """,
    )

    parser.add_argument("--operation", choices=list(OPERATIONS.keys()), help="Operation to generate tests for")

    parser.add_argument("--all", action="store_true", help="Generate tests for all operations")

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tt-metal/tests/tt_metal/tt_metal/block_variants"),
        help="Output directory for generated tests",
    )

    parser.add_argument("--list", action="store_true", help="List available operations and exit")

    args = parser.parse_args()

    # List operations
    if args.list:
        print("\n📋 Available Operations:")
        for op, config in sorted(OPERATIONS.items(), key=lambda x: x[1]["priority"]):
            print(f"\n  {op} (Priority {config['priority']})")
            print(f"    Functions: {', '.join(config['ops'])}")
            print(f"    Header: {config['header']}")
        return 0

    # Validate arguments
    if not args.all and not args.operation:
        parser.error("Must specify --operation or --all")

    print("╔════════════════════════════════════════════════════════════╗")
    print("║  Block Variants Test Generator                        ║")
    print("║  tt-metal Compute API - Issue #35739                   ║")
    print("╚════════════════════════════════════════════════════════════╝")

    # Generate tests
    success = True
    if args.all:
        print(f"\n🎯 Generating tests for ALL operations")
        print(f"   Output: {args.output}")

        for operation in sorted(OPERATIONS.keys(), key=lambda k: OPERATIONS[k]["priority"]):
            if not generate_tests_for_operation(operation, args.output):
                success = False
    else:
        print(f"\n🎯 Generating tests for: {args.operation}")
        print(f"   Output: {args.output}")

        success = generate_tests_for_operation(args.operation, args.output)

    if success:
        print("\n✅ Test generation complete!")
        print(f"\n📁 Files written to: {args.output}")
        print("\n🚀 Next Steps:")
        print("   1. Review generated files")
        print("   2. Complete TODO sections in test harnesses")
        print("   3. Add tests to CMakeLists.txt")
        print("   4. Build: ./build_metal.sh --build-tests")
        print("   5. Run: ./build/test/tt_metal/test_*_block")
        return 0
    else:
        print("\n❌ Test generation failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
