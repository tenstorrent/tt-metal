// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Programming example: AXPY (y = a*x + y) chained from two FPU binary ops.
//
// Motivation
// ----------
// The eltwise_binary example fires a single FPU op (add_tiles) over a stream
// of tiles. Because the op never changes, its compute kernel can call
// add_tiles_init() exactly once before the loop and use a single
// tile_regs_acquire/release pair per iteration. That gives the (correct, but
// misleading) impression that "*_init is called once at the top of the
// kernel".
//
// This example exercises the rule that *_init must be called whenever the op
// or its input CBs change, and tile_regs_acquire/release must bracket each
// pack_tile(). See kernels/compute/axpy.cpp for the full discussion.

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/distributed.hpp>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <random>
#include <vector>

using namespace tt::tt_metal;

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

static uint32_t float_to_bfloat16_bits_u32(float v) {
    uint32_t tmp;
    std::memcpy(&tmp, &v, sizeof(tmp));
    return tmp >> 16;  // truncation rounding, sufficient for an example
}

int main(int /*argc*/, char** /*argv*/) {
    bool pass = true;

    try {
        constexpr int device_id = 0;
        std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);

        distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
        distributed::MeshWorkload workload;
        auto device_range = distributed::MeshCoordinateRange(mesh_device->shape());
        Program program = CreateProgram();

        constexpr CoreCoord core = {0, 0};

        // Working set: 64 tiles, each 32x32 bfloat16.
        constexpr uint32_t n_tiles = 64;
        constexpr uint32_t elements_per_tile = tt::constants::TILE_WIDTH * tt::constants::TILE_HEIGHT;
        constexpr uint32_t tile_size_bytes = sizeof(bfloat16) * elements_per_tile;
        const float a_value = 2.5f;

        // Random x in [0,1), random y in [0,1). Result: y_out = a*x + y.
        std::mt19937 rng(42);  // fixed seed: deterministic check, easy to debug
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        std::vector<bfloat16> x_data(elements_per_tile * n_tiles);
        std::vector<bfloat16> y_data(elements_per_tile * n_tiles);
        for (auto& v : x_data) {
            v = bfloat16(dist(rng));
        }
        for (auto& v : y_data) {
            v = bfloat16(dist(rng));
        }

        distributed::DeviceLocalBufferConfig dram_config{.page_size = tile_size_bytes, .buffer_type = BufferType::DRAM};
        distributed::ReplicatedBufferConfig buffer_config{.size = n_tiles * tile_size_bytes};

        auto x_dram = distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());
        auto y_dram = distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());
        auto out_dram = distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());

        distributed::EnqueueWriteMeshBuffer(cq, x_dram, x_data, false);
        distributed::EnqueueWriteMeshBuffer(cq, y_dram, y_data, false);

        // CB layout. Float16_b across the board, so the packer can be configured
        // once by binary_op_init_common() and reused for both pack destinations.
        constexpr uint32_t tiles_per_cb = 2;
        const auto make_cb = [&](tt::CBIndex idx, uint32_t n) {
            CreateCircularBuffer(
                program,
                core,
                CircularBufferConfig(n * tile_size_bytes, {{idx, tt::DataFormat::Float16_b}})
                    .set_page_size(idx, tile_size_bytes));
        };
        make_cb(tt::CBIndex::c_0, tiles_per_cb);   // x
        make_cb(tt::CBIndex::c_1, tiles_per_cb);   // y
        make_cb(tt::CBIndex::c_2, 1);              // a (single broadcast tile, reused)
        make_cb(tt::CBIndex::c_24, tiles_per_cb);  // ax (intermediate)
        make_cb(tt::CBIndex::c_16, tiles_per_cb);  // output

        // Reader: stream x[i], y[i] from DRAM, and materialize the 'a' tile.
        std::vector<uint32_t> reader_ct_args;
        TensorAccessorArgs(*x_dram).append_to(reader_ct_args);
        TensorAccessorArgs(*y_dram).append_to(reader_ct_args);
        KernelHandle reader = CreateKernel(
            program,
            OVERRIDE_KERNEL_PREFIX "eltwise_fpu_chain/kernels/dataflow/reader.cpp",
            core,
            ReaderDataMovementConfig{reader_ct_args});

        // Writer: drain cb_out to DRAM.
        std::vector<uint32_t> writer_ct_args;
        TensorAccessorArgs(*out_dram).append_to(writer_ct_args);
        KernelHandle writer = CreateKernel(
            program,
            OVERRIDE_KERNEL_PREFIX "eltwise_fpu_chain/kernels/dataflow/writer.cpp",
            core,
            WriterDataMovementConfig{writer_ct_args});

        // Compute: chained mul_tiles -> add_tiles per tile.
        KernelHandle compute = CreateKernel(
            program,
            OVERRIDE_KERNEL_PREFIX "eltwise_fpu_chain/kernels/compute/axpy.cpp",
            core,
            ComputeConfig{.math_fidelity = MathFidelity::HiFi4});

        SetRuntimeArgs(
            program,
            reader,
            core,
            {x_dram->address(), y_dram->address(), float_to_bfloat16_bits_u32(a_value), n_tiles});
        SetRuntimeArgs(program, writer, core, {out_dram->address(), n_tiles});
        SetRuntimeArgs(program, compute, core, {n_tiles});

        workload.add_program(device_range, std::move(program));
        distributed::EnqueueMeshWorkload(cq, workload, false);
        distributed::Finish(cq);

        std::vector<bfloat16> result_vec;
        distributed::EnqueueReadMeshBuffer(cq, result_vec, out_dram, true);

        // bfloat16 + HiFi4 has ~3 decimal digits; loose tolerance.
        constexpr float eps = 2e-2f;
        TT_FATAL(result_vec.size() == x_data.size(), "Result vector size mismatch");
        std::size_t mismatches = 0;
        for (std::size_t i = 0; i < result_vec.size(); ++i) {
            const float expected = a_value * static_cast<float>(x_data[i]) + static_cast<float>(y_data[i]);
            const float actual = static_cast<float>(result_vec[i]);
            if (std::abs(expected - actual) > eps) {
                if (mismatches < 8) {
                    fmt::print(stderr, "Mismatch at {}: expected {:.4f}, got {:.4f}\n", i, expected, actual);
                }
                ++mismatches;
            }
        }
        if (mismatches > 0) {
            pass = false;
            fmt::print(stderr, "Total mismatches: {} / {}\n", mismatches, result_vec.size());
        }

        pass &= mesh_device->close();
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
