// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <utility>
#include <tuple>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

using namespace tt;
using namespace tt::tt_metal;

// Define complex POD structs for runtime arguments
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

int main() {
    // Check for DPRINT environment variable
    char* env_var = std::getenv("TT_METAL_DPRINT_CORES");
    if (env_var == nullptr) {
        fmt::print(
            "WARNING: Please set the environment variable TT_METAL_DPRINT_CORES to (0,0),(1,0) to see the output of "
            "the Data Movement kernels.\n");
        fmt::print("WARNING: For example, export TT_METAL_DPRINT_CORES=(0,0),(1,0)\n");
    }

    // Initialize mesh device (1x1) and command queue
    std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(0);
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());

    // Create a program
    Program program = CreateProgram();

    // Define two cores to demonstrate different runtime args per core
    CoreCoord core0 = {0, 0};
    CoreCoord core1 = {1, 0};
    CoreRangeSet cores({CoreRange(core0, core1)});

    // Create a single kernel that will run on both cores
    KernelHandle kernel_id = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "struct_runtime_args/kernels/dataflow/struct_args_kernel.cpp",
        cores,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    // Set common runtime arguments (shared by all cores) using struct API
    CommonRuntimeArgs common_args = {
        .global_offset = 1000,
        .global_scale = 10,
        .constants = {42, 314, 271},
        .shared_data =
            {
                .id = 999,
                .value = 5000,
                .multiplier = 3,
            },
        .operation = OperationType::MULTIPLY,
    };

    SetCommonRuntimeArgs(program, kernel_id, common_args);

    // Set unique runtime arguments for core0
    RuntimeArgs args_core0 = {
        .core_id = 0,
        .vector_data = {10, 20, 30, 40},
        .range = {0, 100},
        .triple = std::make_tuple(111, 222, 333),
        .nested =
            {
                .id = 1,
                .value = 100,
                .multiplier = 2,
            },
        .op_mode = OperationType::ADD,
    };

    // Set unique runtime arguments for core1
    RuntimeArgs args_core1 = {
        .core_id = 1,
        .vector_data = {50, 60, 70, 80},
        .range = {100, 200},
        .triple = std::make_tuple(444, 555, 666),
        .nested =
            {
                .id = 2,
                .value = 200,
                .multiplier = 4,
            },
        .op_mode = OperationType::MULTIPLY,
    };

    // Use the struct-based API to set runtime args
    SetRuntimeArgs(program, kernel_id, core0, args_core0);
    SetRuntimeArgs(program, kernel_id, core1, args_core1);

    // Alternative: Set runtime args for multiple cores using vector API
    // std::vector<CoreCoord> cores = {core0, core1};
    // std::vector<RuntimeArgs> args_vec = {args_core0, args_core1};
    // SetRuntimeArgs(program, kernel_id, cores, args_vec);

    fmt::print("Launching program with struct-based runtime arguments on cores (0,0) and (1,0)...\n");
    fmt::print("Common args: global_offset={}, global_scale={}\n", common_args.global_offset, common_args.global_scale);
    fmt::print(
        "Core (0,0): core_id={}, vector[0]={}, range=({},{}), op_mode=ADD\n",
        args_core0.core_id,
        args_core0.vector_data[0],
        args_core0.range.first,
        args_core0.range.second);
    fmt::print(
        "Core (1,0): core_id={}, vector[0]={}, range=({},{}), op_mode=MULTIPLY\n",
        args_core1.core_id,
        args_core1.vector_data[0],
        args_core1.range.first,
        args_core1.range.second);

    // Execute the program
    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);

    fmt::print("\nProgram execution completed. Check the output above for kernel DPRINT messages.\n");

    mesh_device->close();
    return 0;
}
