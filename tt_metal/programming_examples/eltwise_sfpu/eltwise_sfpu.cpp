// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cmath>
#include <random>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/experimental/ez/ez.hpp>

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental::ez;

/*
 * 1. Host creates one vector of data.
 * 2. Device eltwise performs a unary SFPU operation (exp) on the data.
 * 3. Read result back and compare to golden.
 * */
#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

int main() {
    bool pass = true;

    try {
        DeviceContext ctx(0);
        constexpr CoreCoord core = {0, 0};

        constexpr uint32_t n_tiles = 64;
        constexpr uint32_t elements_per_tile = tt::constants::TILE_WIDTH * tt::constants::TILE_HEIGHT;

        // Allocate DRAM tile buffers for the input and output.
        auto src0_dram_buffer = ctx.dram_tile_buffer(n_tiles);
        auto dst_dram_buffer = ctx.dram_tile_buffer(n_tiles);

        // Initialize the input data with random values.
        std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<float> dist(0.f, 1.0f);
        std::vector<bfloat16> src0_vec(n_tiles * elements_per_tile);
        for (bfloat16& v : src0_vec) {
            v = bfloat16(dist(rng));
        }

        ctx.write(src0_dram_buffer, src0_vec);

        // Build the program: reader → compute (exp) → writer.
        auto program =
            ProgramBuilder(core)
                .cb(tt::CBIndex::c_0)
                .cb(tt::CBIndex::c_16)
                .reader(
                    OVERRIDE_KERNEL_PREFIX "eltwise_sfpu/kernels/dataflow/read_tile.cpp",
                    {src0_dram_buffer})
                .runtime_args({src0_dram_buffer->address(), n_tiles})
                .done()
                .compute(OVERRIDE_KERNEL_PREFIX "eltwise_sfpu/kernels/compute/eltwise_sfpu.cpp")
                .runtime_args({n_tiles})
                .done()
                .writer(
                    OVERRIDE_KERNEL_PREFIX "eltwise_sfpu/kernels/dataflow/write_tile.cpp",
                    {dst_dram_buffer})
                .runtime_args({dst_dram_buffer->address(), n_tiles})
                .done()
                .build();

        ctx.run(std::move(program));
        auto result_vec = ctx.read<bfloat16>(dst_dram_buffer);

        // Compare to golden (CPU exp).
        constexpr float eps = 5e-2f;
        for (uint32_t i = 0; i < result_vec.size(); ++i) {
            float expected = static_cast<float>(bfloat16(std::exp(static_cast<float>(src0_vec[i]))));
            float result = static_cast<float>(result_vec[i]);
            if (std::abs(expected - result) > eps) {
                pass = false;
                fmt::print(stderr, "Result mismatch at index {}: {} != {}\n", i, expected, result);
            }
        }

    } catch (const std::exception& e) {
        fmt::print(stderr, "Test failed with exception!\n");
        fmt::print(stderr, "{}\n", e.what());
        throw;
    }

    if (pass) {
        fmt::print("Test Passed\n");
    } else {
        TT_THROW("Test Failed");
    }

    return 0;
}
