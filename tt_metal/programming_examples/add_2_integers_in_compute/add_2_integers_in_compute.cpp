// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/experimental/ez/ez.hpp>

#include <random>
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

    // Most data on Tensix is stored in tiles. A tile is a 2D array of 32x32 values. With BFloat16 as the
    // data format, each tile occupies 32x32x2 = 2048 bytes.
    // dram_tile_buffer() allocates DRAM-backed storage sized for the given number of tiles.
    auto src0 = ctx.dram_tile_buffer(1);
    auto src1 = ctx.dram_tile_buffer(1);
    auto dst = ctx.dram_tile_buffer(1);

    // Create random input data: src0 in [0, 14) and src1 in [0, 8).
    constexpr uint32_t n_elements_per_tile = tt::constants::TILE_WIDTH * tt::constants::TILE_WIDTH;
    std::vector<bfloat16> src0_vec(n_elements_per_tile);
    std::vector<bfloat16> src1_vec(n_elements_per_tile);
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist1(0.0f, 14.0f);
    std::uniform_real_distribution<float> dist2(0.0f, 8.0f);
    for (size_t i = 0; i < n_elements_per_tile; ++i) {
        src0_vec[i] = bfloat16(dist1(rng));
        src1_vec[i] = bfloat16(dist2(rng));
    }

    // Upload input data from host to device DRAM.
    ctx.write(src0, src0_vec);
    ctx.write(src1, src1_vec);

    // A program is a collection of kernels. Unlike OpenCL/CUDA where every core must run the same kernel
    // at a given time, Metalium allows different kernels on different cores simultaneously.
    // We use a single Tensix core {0, 0} for this example.
    constexpr CoreCoord core = {0, 0};

    // Build the program with 3 kernels forming a pipeline:
    //   Reader  → reads data from DRAM and pushes into circular buffers
    //   Compute → pops data from input CBs, adds two tiles, pushes result to output CB
    //   Writer  → pops data from output CB and writes back to DRAM
    //
    // Circular buffers (CBs) act as pipes between kernels on the same core. They are backed by L1 (SRAM)
    // memory. Each CB here holds 1 tile. The hardware supports up to 32 circular buffers.
    // cb_0 and cb_1 carry input data; cb_16 carries the output.
    auto program =
        ProgramBuilder(core)
            .cb(tt::CBIndex::c_0, /*num_tiles=*/1)
            .cb(tt::CBIndex::c_1, /*num_tiles=*/1)
            .cb(tt::CBIndex::c_16, /*num_tiles=*/1)
            .reader(OVERRIDE_KERNEL_PREFIX "add_2_integers_in_compute/kernels/dataflow/reader_binary_1_tile.cpp")
            .runtime_args({src0->address(), src1->address()})
            .done()
            .compute(OVERRIDE_KERNEL_PREFIX "add_2_integers_in_compute/kernels/compute/add_2_tiles.cpp")
            .done()
            .writer(OVERRIDE_KERNEL_PREFIX "add_2_integers_in_compute/kernels/dataflow/writer_1_tile.cpp")
            .runtime_args({dst->address()})
            .done()
            .build();

    // Execute the program and read the result back to the host.
    ctx.run(std::move(program));
    auto result_vec = ctx.read<bfloat16>(dst);

    // Compare results.
    bool success = true;
    for (size_t i = 0; i < n_elements_per_tile; ++i) {
        float expected = static_cast<float>(src0_vec[i]) + static_cast<float>(src1_vec[i]);
        if (std::abs(expected - static_cast<float>(result_vec[i])) > 3e-1f) {
            fmt::print(
                stderr, "Mismatch at index {}: expected {}, got {}\n", i, expected, static_cast<float>(result_vec[i]));
            success = false;
        }
    }
    if (!success) {
        fmt::print("Error: Result does not match expected value!\n");
    } else {
        fmt::print("Success: Result matches expected value!\n");
    }
}
