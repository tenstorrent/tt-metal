// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/ez/ez.hpp>

using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental::ez;

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

int main() {
    // This example demonstrates that though the RISC-V cores (Both the data movement and compute cores) are
    // RV32IM and the Tensix relies on the SFPU and FPU attached to the compute cores to perform the bulk of the
    // floating point operations, it is still possible to operate on floating point data types directly on the
    // RISC-V cores as they are fully programmable and the compiler can generate the necessary software floating point
    // operations.

    char* env_var = std::getenv("TT_METAL_DPRINT_CORES");
    if (env_var == nullptr) {
        fmt::print(
            "WARNING: Please set the environment variable TT_METAL_DPRINT_CORES to 0,0 to see the output of the Data "
            "Movement kernels.\n");
        fmt::print("WARNING: For example, export TT_METAL_DPRINT_CORES=0,0\n");
    }

    DeviceContext ctx(0);

    // Create a small DRAM buffer to hold a single float value.
    constexpr uint32_t buffer_size = 2 * 1024;
    auto dram_buffer = ctx.dram_buffer(buffer_size, buffer_size);

    // Upload a float value to the device.
    std::vector<float> init_data = {1.23};
    ctx.write(dram_buffer, init_data);

    // Build the program: a data movement kernel with a CB to transfer data from DRAM to L1.
    // We use .kernel() with explicit DataMovementConfig to keep the kernel on RISCV_0 (Data Movement
    // processor 0 / "BR"), matching the original example. (.reader() would assign RISCV_1 instead.)
    auto program =
        ProgramBuilder(CoreCoord{0, 0})
            .cb(tt::CBIndex::c_0, tt::DataFormat::Float16_b, /*num_tiles=*/1, /*page_size=*/buffer_size)
            .kernel(
                OVERRIDE_KERNEL_PREFIX "hello_world_datatypes_kernel/kernels/dataflow/float_dataflow_kernel.cpp",
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default})
            .runtime_args({dram_buffer->address()})
            .build();

    fmt::print("Hello, Core (0, 0) on Device 0, please handle the data.\n");
    ctx.run(std::move(program));
    fmt::print("Thank you, Core (0, 0) on Device 0, for handling the data.\n");

    return 0;
}
