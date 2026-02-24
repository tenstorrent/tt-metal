// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/experimental/ez/ez.hpp>

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental::ez;

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

int main(int /*argc*/, char** /*argv*/) {
    bool pass = true;

    // clang-format off
    try {
        // Create a 1x1 mesh on device 0. The same API scales to multi-device meshes.
        DeviceContext ctx(0);

        // This example program will only use 1 Tensix core. So we set the core to {0, 0}.
        constexpr CoreCoord core = {0, 0};

        // Define some constants that will be used throughout the program.
        // * Processing 64 tiles
        // * Each tile is 32x32 elements
        // * Each element is a bfloat16 (2 bytes)
        constexpr uint32_t n_tiles = 64;
        constexpr uint32_t elements_per_tile = tt::constants::TILE_WIDTH * tt::constants::TILE_HEIGHT;

        // Create 3 DRAM-backed tile buffers: two inputs (src0, src1) and one output (dst).
        // dram_tile_buffer() computes page_size and total_size from tile count and data format.
        // The page size is set to the tile size, which is required for a large portion of the NoC
        // transfer APIs to work correctly (unlike raw L1 buffers where page size can differ).
        auto src0_dram_buffer = ctx.dram_tile_buffer(n_tiles);
        auto src1_dram_buffer = ctx.dram_tile_buffer(n_tiles);
        auto dst_dram_buffer = ctx.dram_tile_buffer(n_tiles);

        // Initialize the input buffers with random data. For this example, src0 is a random vector of bfloat16 values
        std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
        std::vector<bfloat16> a_data(elements_per_tile * n_tiles);
        for(auto& val : a_data) {
            val = bfloat16(distribution(rng));
        }

        // ... and src1 is a vector of bfloat16 values initialized to -1.0f.
        constexpr float val_to_add = -1.0f;
        std::vector<bfloat16> b_data(elements_per_tile * n_tiles, bfloat16(val_to_add));

        // Upload host vectors into the mesh buffers.
        ctx.write(src0_dram_buffer, a_data);
        ctx.write(src1_dram_buffer, b_data);

        // Build the program using ProgramBuilder. This creates:
        // * 3 circular buffers (2 input, 1 output) — think of them as pipes moving data between kernels.
        //   Each CB holds 2 tiles, allowing double-buffering for overlapped reads/computes/writes.
        //   More tiles per CB can further hide latency, but there is a trade-off: circular buffers are
        //   backed by L1 (SRAM) memory, which is a precious and limited resource on each Tensix core.
        //   The hardware supports up to 32 circular buffers per core.
        // * Reader kernel: Reads tiles from DRAM and pushes them into CBs c_0 and c_1.
        //   Buffer references are automatically converted to TensorAccessorArgs compile-time args.
        // * Compute kernel: Pops tiles from c_0 and c_1, adds them, pushes results into c_16.
        //   HiFi4 is the most accurate math fidelity mode (alternatives: HiFi3, HiFi2, LoFi).
        // * Writer kernel: Pops tiles from c_16 and writes them back to DRAM.
        auto program =
            ProgramBuilder(core)
                .cb(tt::CBIndex::c_0)
                .cb(tt::CBIndex::c_1)
                .cb(tt::CBIndex::c_16)
                .reader(
                    OVERRIDE_KERNEL_PREFIX "eltwise_binary/kernels/dataflow/read_tiles.cpp",
                    {src0_dram_buffer, src1_dram_buffer})
                .runtime_args({src0_dram_buffer->address(), src1_dram_buffer->address(), n_tiles})
                .compute(OVERRIDE_KERNEL_PREFIX "eltwise_binary/kernels/compute/tiles_add.cpp")
                .runtime_args({n_tiles})
                .writer(
                    OVERRIDE_KERNEL_PREFIX "eltwise_binary/kernels/dataflow/write_tile.cpp",
                    {dst_dram_buffer})
                .runtime_args({dst_dram_buffer->address(), n_tiles})
                .build();

        // Execute the program synchronously (enqueue + wait for completion).
        ctx.run(std::move(program));

        // Read the output buffer and validate.
        auto result_vec = ctx.read<bfloat16>(dst_dram_buffer);

        constexpr float eps = 1e-2f; // loose tolerance because of the nature of bfloat16
        TT_FATAL(result_vec.size() == a_data.size(), "Result vector size mismatch");
        for (size_t i = 0; i < result_vec.size(); ++i) {
            const float expected = static_cast<float>(a_data[i]) + val_to_add;
            const float actual = static_cast<float>(result_vec[i]);

            if (std::abs(expected - actual) > eps) {
                pass = false;
                fmt::print(stderr, "Result mismatch at index {}: expected {}, got {}\n", i, expected, actual);
            }
        }

        // DeviceContext closes the device automatically when it goes out of scope.
    } catch (const std::exception& e) {
        fmt::print(stderr, "Test failed with exception!\n");
        fmt::print(stderr, "{}\n", e.what());

        throw;
    }
    // clang-format on

    if (pass) {
        fmt::print("Test Passed\n");
    } else {
        TT_THROW("Test Failed");
    }

    return 0;
}
