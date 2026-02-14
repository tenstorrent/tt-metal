// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/ez/ez.hpp>

#include <cstdint>
#include <vector>

using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental::ez;

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

int main() {
    // Ensure printing from kernel is enabled (so we can see the output of the Data Movement kernels).
    char* env_var = std::getenv("TT_METAL_DPRINT_CORES");
    if (env_var == nullptr) {
        fmt::print(
            "WARNING: Please set the environment variable TT_METAL_DPRINT_CORES to 0,0 to see the output of the Data "
            "Movement kernels.\n");
        fmt::print("WARNING: For example, export TT_METAL_DPRINT_CORES=0,0\n");
    }

    // DeviceContext opens a single Tenstorrent device and provides helpers for buffer allocation,
    // data transfers, and program execution.
    DeviceContext ctx(0);

    // Adding 2 integers in RISC-V: a buffer size of 4 bytes.
    constexpr uint32_t buffer_size = sizeof(uint32_t);

    // The Tensix core does not have direct access to DRAM, so an extra buffer in L1 (SRAM) is required
    // to stage data for reads/writes. We allocate both DRAM and L1 buffers here:
    //   - DRAM buffers hold the persistent input/output data
    //   - L1 buffers serve as on-core scratch space for the Data Movement kernel
    auto src0_dram = ctx.dram_buffer(buffer_size, buffer_size);
    auto src1_dram = ctx.dram_buffer(buffer_size, buffer_size);
    auto dst_dram = ctx.dram_buffer(buffer_size, buffer_size);
    auto src0_l1 = ctx.l1_buffer(buffer_size, buffer_size);
    auto src1_l1 = ctx.l1_buffer(buffer_size, buffer_size);
    auto dst_l1 = ctx.l1_buffer(buffer_size, buffer_size);

    // Upload source data to DRAM.
    std::vector<uint32_t> src0_vec = {14};
    std::vector<uint32_t> src1_vec = {7};
    ctx.write(src0_dram, src0_vec);
    ctx.write(src1_dram, src1_vec);

    // The Data Movement cores are the only cores that can read/write data from/to DRAM. In practice you
    // would use the compute kernel for addition (which has access to much more powerful vector and matrix
    // engines), but the Data Movement cores are still fully capable of simple arithmetic — just slower.
    // Here we use a single Data Movement kernel that reads two integers from DRAM via L1, adds them,
    // and writes the result back.
    // The original kernel runs on RISCV_0 (Data Movement processor 0 / "BR"). We use .kernel() with
    // explicit DataMovementConfig to preserve this assignment, since .reader() would use RISCV_1.
    auto program =
        ProgramBuilder(CoreCoord{0, 0})
            .kernel(
                OVERRIDE_KERNEL_PREFIX "add_2_integers_in_riscv/kernels/reader_writer_add_in_riscv.cpp",
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default})
            .runtime_args({
                src0_dram->address(),
                src1_dram->address(),
                dst_dram->address(),
                src0_l1->address(),
                src1_l1->address(),
                dst_l1->address(),
            })
            .build();

    ctx.run(std::move(program));

    // Read the result back from DRAM. Everything on the command queue executes in FIFO order, so a
    // blocking read will not start until the prior kernel finishes.
    auto result_vec = ctx.read<uint32_t>(dst_dram);

    if (result_vec.size() != 1) {
        fmt::print(stderr, "Error: Expected result vector size of 1, got {}\n", result_vec.size());
        return -1;
    }
    if (result_vec[0] != 21) {
        fmt::print(stderr, "Error: Expected result of 21, got {}\n", result_vec[0]);
        return -1;
    }

    fmt::print("Success: Result is {}\n", result_vec[0]);
}
