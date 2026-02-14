// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/experimental/ez/ez.hpp>

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental::ez;

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

int main() {
    bool pass = true;

    try {
        // Processing 64 tiles, each 32x32 BFloat16 elements.
        constexpr uint32_t n_tiles = 64;
        constexpr uint32_t elements = n_tiles * tt::constants::TILE_WIDTH * tt::constants::TILE_HEIGHT;

        // DeviceContext opens a single Tenstorrent device and provides helpers for buffer allocation,
        // data transfers, and program execution.
        DeviceContext ctx(0);

        // Allocate DRAM-backed tile buffers: one input and one output.
        auto src0 = ctx.dram_tile_buffer(n_tiles);
        auto dst = ctx.dram_tile_buffer(n_tiles);

        // Generate random input data in [0, 1).
        std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        std::vector<bfloat16> a_data(elements);
        for (auto& val : a_data) {
            val = bfloat16(dist(rng));
        }

        // Upload input data from host to device DRAM.
        ctx.write(src0, a_data);

        // Build the program with 3 kernels forming a reader → compute → writer pipeline:
        //   Reader  → reads tiles from DRAM and pushes into circular buffer cb_0
        //   Compute → pops tiles from cb_0, applies smoothstep via custom SFPI, pushes to cb_16
        //   Writer  → pops tiles from cb_16 and writes back to DRAM
        //
        // Circular buffers act as pipes between kernels on the same core. They are backed by L1 (SRAM)
        // memory. Each CB here uses the default of 2 tiles for double-buffering: while one tile is being
        // consumed by the receiving kernel, the sending kernel can prepare the next tile.
        //
        // Passing buffers (e.g. {src0}) to .reader() automatically generates TensorAccessorArgs as
        // compile-time arguments, telling the kernel how to access the buffer's layout.
        constexpr CoreCoord core = {0, 0};
        auto program =
            ProgramBuilder(core)
                .cb(tt::CBIndex::c_0)
                .cb(tt::CBIndex::c_16)
                .reader(
                    OVERRIDE_KERNEL_PREFIX "custom_sfpi_smoothstep/kernels/dataflow/read_tiles.cpp",
                    {src0})
                .runtime_args({src0->address(), n_tiles})
                .done()
                .compute(OVERRIDE_KERNEL_PREFIX "custom_sfpi_smoothstep/kernels/compute/tiles_smoothstep.cpp")
                .runtime_args({n_tiles})
                .done()
                .writer(
                    OVERRIDE_KERNEL_PREFIX "custom_sfpi_smoothstep/kernels/dataflow/write_tile.cpp",
                    {dst})
                .runtime_args({dst->address(), n_tiles})
                .done()
                .build();

        // Execute the program and read the result back to the host.
        ctx.run(std::move(program));
        auto result = ctx.read<bfloat16>(dst);

        // Compare with CPU-side smoothstep. Loose tolerance due to BFloat16 precision.
        constexpr float eps = 1e-2f;
        TT_FATAL(result.size() == a_data.size(), "Result vector size mismatch");
        for (size_t i = 0; i < result.size(); ++i) {
            auto smoothstep = [](float edge0, float edge1, float x) {
                // Scale, bias and saturate x to 0..1 range
                x = (x - edge0) / (edge1 - edge0);
                x = std::clamp(x, 0.0f, 1.0f);
                // Evaluate polynomial
                return x * x * (3 - 2 * x);
            };
            const float expected = smoothstep(0.0f, 1.0f, static_cast<float>(a_data[i]));
            const float actual = static_cast<float>(result[i]);
            if (std::abs(expected - actual) > eps) {
                pass = false;
                fmt::print(stderr, "Result mismatch at index {}: expected {}, got {}\n", i, expected, actual);
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
