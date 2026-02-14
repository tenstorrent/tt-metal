// SPDX-FileCopyrightText: © 2023-2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// This example demonstrates a simple data copy from DRAM into L1(SRAM) and to another place in DRAM.
// A single data movement kernel reads tiles from input DRAM, stages through L1, and writes to output DRAM.
//
// The general flow is as follows:
// 1. Open a device (DeviceContext handles initialization and teardown)
// 2. Allocate DRAM and L1 buffers for input, output, and staging
// 3. Fill the input DRAM buffer with random data
// 4. Create a data movement kernel on core {0, 0} to copy DRAM→L1→DRAM
// 5. Execute the program
// 6. Read back the output and validate against the input
// 7. DeviceContext automatically cleans up the device on scope exit

#include <random>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/experimental/ez/ez.hpp>

using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental::ez;

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

int main() {
    bool pass = true;

    try {
        DeviceContext ctx(0);
        constexpr CoreCoord core = {0, 0};

        constexpr uint32_t num_tiles = 50;
        constexpr uint32_t elements_per_tile = tt::constants::TILE_WIDTH * tt::constants::TILE_HEIGHT;
        constexpr uint32_t tile_size_bytes = sizeof(bfloat16) * elements_per_tile;

        // Allocate DRAM buffers for input/output and an L1 scratch buffer for staging.
        auto input_dram_buffer = ctx.dram_tile_buffer(num_tiles);
        auto output_dram_buffer = ctx.dram_tile_buffer(num_tiles);
        auto l1_buffer = ctx.l1_buffer(tile_size_bytes, tile_size_bytes);

        // Initialize with random data.
        std::vector<bfloat16> input_vec(elements_per_tile * num_tiles);
        std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<float> distribution(0.0f, 100.0f);
        for (auto& val : input_vec) {
            val = bfloat16(distribution(rng));
        }

        // Upload the data from host to device DRAM. ctx.write() is non-blocking by default: it
        // enqueues the transfer and returns immediately. ctx.run() calls finish() to wait for all
        // enqueued work (including writes) before reading results.
        ctx.write(input_dram_buffer, input_vec);

        // Build the program with a single data movement kernel that copies DRAM→L1→DRAM.
        // This kernel acts as both reader and writer. .reader() assigns RISCV_1 with the
        // architecture-preferred read NOC, and auto-generates TensorAccessorArgs for both buffers.
        auto program =
            ProgramBuilder(core)
                .reader(
                    OVERRIDE_KERNEL_PREFIX "loopback/kernels/loopback_dram_copy.cpp",
                    {input_dram_buffer, output_dram_buffer})
                .runtime_args({l1_buffer->address(), input_dram_buffer->address(),
                               output_dram_buffer->address(), num_tiles})
                .done()
                .build();

        ctx.run(std::move(program));
        auto result_vec = ctx.read<bfloat16>(output_dram_buffer);

        TT_FATAL(
            result_vec.size() == input_vec.size(),
            "Result vector size {} does not match input vector size {}",
            result_vec.size(),
            input_vec.size());
        for (size_t i = 0; i < input_vec.size(); i++) {
            if (input_vec[i] != result_vec[i]) {
                pass = false;
                break;
            }
        }

    } catch (const std::exception& e) {
        fmt::print(stderr, "Test failed with exception! what: {}\n", e.what());
        throw;
    }

    if (pass) {
        fmt::print("Test Passed\n");
    } else {
        TT_THROW("Test Failed");
    }

    return 0;
}
