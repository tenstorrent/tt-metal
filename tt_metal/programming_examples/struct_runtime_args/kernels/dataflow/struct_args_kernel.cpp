// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "debug/dprint.h"  // required in all kernels using DPRINT
#include "dataflow_api.h"
#include <array>
#include <utility>
#include <tuple>

// Define the same complex struct as on the host side
enum class OperationType : uint32_t { ADD = 0, MULTIPLY = 1, SUBTRACT = 2, DIVIDE = 3 };

struct NestedData {
    uint32_t id;
    uint32_t value;
    uint32_t multiplier;
};

struct CommonRuntimeArgs {
    uint32_t global_offset;
    uint32_t global_scale;
    std::array<uint32_t, 3> constants;
    NestedData shared_data;
    OperationType operation;
};

struct RuntimeArgs {
    uint32_t core_id;
    std::array<uint32_t, 4> vector_data;
    std::pair<uint32_t, uint32_t> range;
    std::tuple<uint32_t, uint32_t, uint32_t> triple;
    NestedData nested;
    OperationType op_mode;
};

void kernel_main() {
    // Get runtime arguments using the new struct-based API
    auto& rt_args = get_runtime_arguments<RuntimeArgs>();
    auto& common_args = get_common_runtime_arguments<CommonRuntimeArgs>();

    // Print core coordinates
    DPRINT << "=== Core (" << (uint32_t)get_absolute_logical_x() << "," << (uint32_t)get_absolute_logical_y()
           << ") ===" << ENDL();

    // Print runtime args (unique per core)
    DPRINT << "Runtime Args:" << ENDL();
    DPRINT << "  core_id: " << rt_args.core_id << ENDL();

    DPRINT << "  vector_data: [";
    for (uint32_t i = 0; i < 4; i++) {
        DPRINT << rt_args.vector_data[i];
        if (i < 3) {
            DPRINT << ", ";
        }
    }
    DPRINT << "]" << ENDL();

    DPRINT << "  range: (" << rt_args.range.first << ", " << rt_args.range.second << ")" << ENDL();

    DPRINT << "  triple: (" << std::get<0>(rt_args.triple) << ", " << std::get<1>(rt_args.triple) << ", "
           << std::get<2>(rt_args.triple) << ")" << ENDL();

    DPRINT << "  nested.id: " << rt_args.nested.id << ENDL();
    DPRINT << "  nested.value: " << rt_args.nested.value << ENDL();
    DPRINT << "  nested.multiplier: " << rt_args.nested.multiplier << ENDL();

    DPRINT << "  op_mode: " << static_cast<uint32_t>(rt_args.op_mode) << ENDL();

    // Print common args (same for all cores)
    DPRINT << "Common Runtime Args:" << ENDL();
    DPRINT << "  global_offset: " << common_args.global_offset << ENDL();
    DPRINT << "  global_scale: " << common_args.global_scale << ENDL();

    DPRINT << "  constants: [";
    for (uint32_t i = 0; i < 3; i++) {
        DPRINT << common_args.constants[i];
        if (i < 2) {
            DPRINT << ", ";
        }
    }
    DPRINT << "]" << ENDL();

    DPRINT << "  shared_data.id: " << common_args.shared_data.id << ENDL();
    DPRINT << "  shared_data.value: " << common_args.shared_data.value << ENDL();
    DPRINT << "  shared_data.multiplier: " << common_args.shared_data.multiplier << ENDL();

    DPRINT << "  operation: " << static_cast<uint32_t>(common_args.operation) << ENDL();

    // Perform a simple computation using the arguments
    uint32_t result = (rt_args.core_id * common_args.global_scale) + common_args.global_offset;
    result += rt_args.vector_data[0] * rt_args.nested.multiplier;

    // Use the enum to modify the result
    switch (rt_args.op_mode) {
        case OperationType::ADD: result += 100; break;
        case OperationType::MULTIPLY: result *= 2; break;
        case OperationType::SUBTRACT: result -= 50; break;
        case OperationType::DIVIDE: result /= 2; break;
    }

    DPRINT << "Computed result (after op_mode): " << result << ENDL();
    DPRINT << ENDL();
}
