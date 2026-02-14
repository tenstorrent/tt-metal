// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/experimental/ez/ez.hpp>

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

        // Allocate DRAM-backed tile buffers: two inputs and one output.
        auto src0 = ctx.dram_tile_buffer(n_tiles);
        auto src1 = ctx.dram_tile_buffer(n_tiles);
        auto dst = ctx.dram_tile_buffer(n_tiles);

        // Initialize the input data: src0 is random values in [0,1), src1 is all -1.0.
        std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        std::vector<bfloat16> a_data(elements);
        for (auto& val : a_data) {
            val = bfloat16(dist(rng));
        }
        constexpr float val_to_add = -1.0f;
        std::vector<bfloat16> b_data(elements, bfloat16(val_to_add));

        // Upload input data from host to device DRAM. ctx.write() is non-blocking by default: it
        // enqueues the transfer and returns immediately. ctx.run() calls finish() to wait for all
        // enqueued work (including writes) before reading results.
        ctx.write(src0, a_data);
        ctx.write(src1, b_data);

        // Build the program with 3 kernels forming a reader → compute → writer pipeline:
        //   Reader  → reads tiles from DRAM and pushes into circular buffers (cb_0, cb_1)
        //   Compute → pops tiles from input CBs, adds them using custom SFPI, pushes to output CB (cb_16)
        //   Writer  → pops tiles from output CB and writes back to DRAM
        //
        // Circular buffers act as pipes between kernels on the same core. They are backed by L1 (SRAM)
        // memory. Each CB here uses the default of 2 tiles for double-buffering: while one tile is being
        // consumed by the receiving kernel, the sending kernel can prepare the next tile. The hardware
        // supports up to 32 circular buffers.
        //
        // Passing buffers (e.g. {src0, src1}) to .reader() automatically generates TensorAccessorArgs
        // as compile-time arguments, telling the kernel how to access each buffer's layout.
        constexpr CoreCoord core = {0, 0};
        auto program =
            ProgramBuilder(core)
                .cb(tt::CBIndex::c_0)
                .cb(tt::CBIndex::c_1)
                .cb(tt::CBIndex::c_16)
                .reader(
                    OVERRIDE_KERNEL_PREFIX "custom_sfpi_add/kernels/dataflow/read_tiles.cpp",
                    {src0, src1})
                .runtime_args({src0->address(), src1->address(), n_tiles})
                .done()
                .compute(
                    OVERRIDE_KERNEL_PREFIX "custom_sfpi_add/kernels/compute/tiles_add.cpp",
                    // HiFi4 is the most accurate math fidelity mode. The ComputeConfig also supports
                    // fp32_dest_acc_en (FP32 destination accumulator), which defaults to false here since
                    // our inputs and outputs are BFloat16 and don't need FP32 accumulation precision.
                    MathFidelity::HiFi4)
                .runtime_args({n_tiles})
                .done()
                .writer(
                    OVERRIDE_KERNEL_PREFIX "custom_sfpi_add/kernels/dataflow/write_tile.cpp",
                    {dst})
                .runtime_args({dst->address(), n_tiles})
                .done()
                .build();

        // Execute the program and read the result back to the host.
        ctx.run(std::move(program));
        auto result = ctx.read<bfloat16>(dst);

        // Compare results with a loose tolerance due to BFloat16 precision.
        constexpr float eps = 1e-2f;
        TT_FATAL(result.size() == a_data.size(), "Result vector size mismatch");
        for (size_t i = 0; i < result.size(); ++i) {
            const float expected = static_cast<float>(a_data[i]) + val_to_add;
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
