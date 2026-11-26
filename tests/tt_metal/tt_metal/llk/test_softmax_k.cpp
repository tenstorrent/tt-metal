// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include <tt-metalium/data_formats/bfloat16.hpp>
#include <tt-metalium/execution/distributed.hpp>
#include <tt-metalium/execution/slow_dispatch.hpp>
#include <tt-metalium/workload/legacy/host_api.hpp>

#include "llk_device_fixture.hpp"

namespace tt::tt_metal::unit_tests::compute::softmax_k {
namespace {

constexpr std::uint32_t tile_dim = 32;
constexpr std::uint32_t face_dim = 16;
constexpr std::uint32_t tile_elements = tile_dim * tile_dim;

std::uint32_t tile_index(std::uint32_t r, std::uint32_t c) {
    const std::uint32_t face = (r / face_dim) * 2 + (c / face_dim);
    return face * (face_dim * face_dim) + (r % face_dim) * face_dim + (c % face_dim);
}

std::vector<float> run_softmax_k(
    distributed::MeshDevice& mesh,
    std::uint32_t num_rows,
    std::uint32_t k,
    bool fp32_dest,
    const std::vector<float>& input_row_major) {
    const CoreCoord core{0, 0};
    const auto data_fmt = fp32_dest ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    const std::uint32_t tile_bytes = tile_elements * (fp32_dest ? sizeof(float) : sizeof(bfloat16));

    std::vector<float> tilized(tile_elements);
    for (std::uint32_t r = 0; r < tile_dim; ++r) {
        for (std::uint32_t c = 0; c < tile_dim; ++c) {
            tilized[tile_index(r, c)] = input_row_major[r * tile_dim + c];
        }
    }
    std::vector<std::uint32_t> encoded;
    if (fp32_dest) {
        encoded.resize(tile_elements);
        for (std::size_t i = 0; i < tile_elements; ++i) {
            encoded[i] = std::bit_cast<std::uint32_t>(tilized[i]);
        }
    } else {
        std::vector<bfloat16> bf16(tile_elements);
        for (std::size_t i = 0; i < tile_elements; ++i) {
            bf16[i] = bfloat16(tilized[i]);
        }
        encoded = pack_bfloat16_vec_into_uint32_vec(bf16);
    }

    Program program = CreateProgram(mesh);
    auto& cq = mesh.mesh_command_queue();
    auto make_buffer = [&](std::uint32_t bytes) {
        return distributed::MeshBuffer::create(
            distributed::ReplicatedBufferConfig{.size = bytes},
            {.page_size = bytes, .buffer_type = BufferType::DRAM},
            &mesh);
    };

    CreateCircularBuffer(
        program,
        core,
        CircularBufferConfig(tile_bytes, {{tt::CBIndex::c_0, data_fmt}}).set_page_size(tt::CBIndex::c_0, tile_bytes));
    CreateCircularBuffer(
        program,
        core,
        CircularBufferConfig(tile_bytes, {{tt::CBIndex::c_16, data_fmt}}).set_page_size(tt::CBIndex::c_16, tile_bytes));

    auto input_buf = make_buffer(tile_bytes);
    auto output_buf = make_buffer(tile_bytes);
    distributed::EnqueueWriteMeshBuffer(cq, input_buf, encoded, true);

    const auto reader = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_unary.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
    SetRuntimeArgs(program, reader, core, {input_buf->address(), 0, 1});

    const auto writer = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    SetRuntimeArgs(program, writer, core, {output_buf->address(), 0, 1});

    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    if (fp32_dest) {
        unpack_to_dest_mode[tt::CBIndex::c_0] = UnpackToDestMode::UnpackToDestFp32;
    }

    CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/softmax_k.cpp",
        core,
        ComputeConfig{
            .fp32_dest_acc_en = fp32_dest,
            .unpack_to_dest_mode = unpack_to_dest_mode,
            .math_approx_mode = false,
            .compile_args = {num_rows, k}});

    distributed::MeshWorkload workload;
    const distributed::MeshCoordinate zero(0, 0);
    workload.add_program(distributed::MeshCoordinateRange(zero, zero), std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);

    std::vector<std::uint32_t> packed;
    distributed::EnqueueReadMeshBuffer(cq, packed, output_buf, true);
    std::vector<float> out_tilized;
    out_tilized.reserve(tile_elements);
    for (const auto word : packed) {
        if (fp32_dest) {
            out_tilized.push_back(std::bit_cast<float>(word));
        } else {
            out_tilized.push_back(static_cast<float>(std::bit_cast<bfloat16>(static_cast<std::uint16_t>(word))));
            out_tilized.push_back(static_cast<float>(std::bit_cast<bfloat16>(static_cast<std::uint16_t>(word >> 16))));
        }
    }
    std::vector<float> row_major(tile_elements);
    for (std::uint32_t r = 0; r < tile_dim; ++r) {
        for (std::uint32_t c = 0; c < tile_dim; ++c) {
            row_major[r * tile_dim + c] = out_tilized[tile_index(r, c)];
        }
    }
    return row_major;
}

}  // namespace

TEST_F(LLKBlackholeSingleCardFixture, SoftmaxKTileComputeApi) {
    auto& mesh = *devices_.at(0);
    for (const bool fp32_dest : {false, true}) {
        for (const std::uint32_t num_rows : {1u, 4u, 8u}) {
            for (const std::uint32_t k : {1u, 7u, 16u}) {
                SCOPED_TRACE(
                    ::testing::Message() << "fp32_dest=" << fp32_dest << ", num_rows=" << num_rows << ", k=" << k);
                std::mt19937 rng(100 + num_rows * 31 + k);
                std::normal_distribution<float> dist(0.0f, 2.0f);
                std::vector<float> input(tile_elements, 0.0f);
                for (std::uint32_t r = 0; r < num_rows; ++r) {
                    for (std::uint32_t c = 0; c < k; ++c) {
                        float val = (r == 0 && c == 0 && k >= 2) ? 0.0f : dist(rng);
                        if (!fp32_dest) {
                            val = static_cast<float>(bfloat16(val));
                        }
                        input[r * tile_dim + c] = val;
                    }
                    for (std::uint32_t c = k; c < face_dim; ++c) {
                        input[r * tile_dim + c] = 100.0f;
                    }
                }

                const auto output = run_softmax_k(mesh, num_rows, k, fp32_dest, input);
                const double tol = fp32_dest ? 1e-4 : 1e-2;
                for (std::uint32_t r = 0; r < num_rows; ++r) {
                    double max_v = -std::numeric_limits<double>::infinity();
                    for (std::uint32_t c = 0; c < k; ++c) {
                        max_v = std::max(max_v, static_cast<double>(input[r * tile_dim + c]));
                    }
                    double sum = 0.0;
                    std::vector<double> exps(k);
                    for (std::uint32_t c = 0; c < k; ++c) {
                        exps[c] = std::exp(static_cast<double>(input[r * tile_dim + c]) - max_v);
                        sum += exps[c];
                    }
                    for (std::uint32_t c = 0; c < k; ++c) {
                        EXPECT_NEAR(output[r * tile_dim + c], exps[c] / sum, tol);
                    }
                    for (std::uint32_t c = k; c < face_dim; ++c) {
                        EXPECT_EQ(output[r * tile_dim + c], 0.0f);
                    }
                }
            }
        }
    }
}

TEST_F(LLKBlackholeSingleCardFixture, SoftmaxKTileFp32MinFiniteSingleton) {
    auto& mesh = *devices_.at(0);
    std::vector<float> input(tile_elements, 0.0f);
    input[0] = -std::numeric_limits<float>::max();
    for (std::uint32_t c = 1; c < face_dim; ++c) {
        input[c] = 100.0f;
    }
    const auto output = run_softmax_k(mesh, 1u, 1u, true, input);
    EXPECT_NEAR(output[0], 1.0f, 1e-4f);
    for (std::uint32_t c = 1; c < face_dim; ++c) {
        EXPECT_EQ(output[c], 0.0f);
    }
}

}  // namespace tt::tt_metal::unit_tests::compute::softmax_k
