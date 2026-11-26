// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <bit>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include <tt-metalium/data_formats/bfloat16.hpp>
#include <tt-metalium/execution/distributed.hpp>
#include <tt-metalium/execution/slow_dispatch.hpp>
#include <tt-metalium/workload/legacy/host_api.hpp>

#include "llk_device_fixture.hpp"

namespace tt::tt_metal::unit_tests::compute::row_norm {
namespace {

constexpr std::uint32_t tile_dim = 32;
constexpr std::uint32_t face_dim = 16;
constexpr std::uint32_t tile_elements = tile_dim * tile_dim;
constexpr float kEpsilon = 1e-5f;

std::uint32_t tile_index(std::uint32_t r, std::uint32_t c) {
    const std::uint32_t face = (r / face_dim) * 2 + (c / face_dim);
    return face * (face_dim * face_dim) + (r % face_dim) * face_dim + (c % face_dim);
}

std::vector<std::uint32_t> encode_tiles(const std::vector<float>& row_major, std::uint32_t num_tiles, bool fp32) {
    std::vector<float> tilized(num_tiles * tile_elements);
    for (std::uint32_t t = 0; t < num_tiles; ++t) {
        for (std::uint32_t r = 0; r < tile_dim; ++r) {
            for (std::uint32_t c = 0; c < tile_dim; ++c) {
                tilized[t * tile_elements + tile_index(r, c)] = row_major[(t * tile_dim + r) * tile_dim + c];
            }
        }
    }
    if (fp32) {
        std::vector<std::uint32_t> words(tilized.size());
        for (std::size_t i = 0; i < tilized.size(); ++i) {
            words[i] = std::bit_cast<std::uint32_t>(tilized[i]);
        }
        return words;
    }
    std::vector<bfloat16> bf16(tilized.size());
    for (std::size_t i = 0; i < tilized.size(); ++i) {
        bf16[i] = bfloat16(tilized[i]);
    }
    return pack_bfloat16_vec_into_uint32_vec(bf16);
}

std::vector<float> decode_tiles(const std::vector<std::uint32_t>& words, std::uint32_t num_tiles, bool fp32) {
    std::vector<float> tilized;
    tilized.reserve(num_tiles * tile_elements);
    for (const auto word : words) {
        if (fp32) {
            tilized.push_back(std::bit_cast<float>(word));
        } else {
            tilized.push_back(static_cast<float>(std::bit_cast<bfloat16>(static_cast<std::uint16_t>(word))));
            tilized.push_back(static_cast<float>(std::bit_cast<bfloat16>(static_cast<std::uint16_t>(word >> 16))));
        }
    }
    std::vector<float> row_major(num_tiles * tile_elements);
    for (std::uint32_t t = 0; t < num_tiles; ++t) {
        for (std::uint32_t r = 0; r < tile_dim; ++r) {
            for (std::uint32_t c = 0; c < tile_dim; ++c) {
                row_major[(t * tile_dim + r) * tile_dim + c] = tilized[t * tile_elements + tile_index(r, c)];
            }
        }
    }
    return row_major;
}

std::vector<float> run_row_norm(
    distributed::MeshDevice& mesh,
    std::uint32_t mode,
    std::uint32_t num_rows,
    std::uint32_t tile_h,
    std::uint32_t num_tiles,
    bool fp32_dest,
    bool stats_is_fp32,
    const std::vector<float>& input_row_major) {
    const CoreCoord core{0, 0};
    const auto data_fmt = fp32_dest ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    const auto stats_fmt = stats_is_fp32 ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    const std::uint32_t tile_bytes = tile_elements * (fp32_dest ? sizeof(float) : sizeof(bfloat16));
    const std::uint32_t stats_tile_bytes = tile_elements * (stats_is_fp32 ? sizeof(float) : sizeof(bfloat16));
    const std::uint32_t out_tiles = (mode == 1) ? 1u : num_tiles;

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
        CircularBufferConfig(num_tiles * tile_bytes, {{tt::CBIndex::c_0, data_fmt}})
            .set_page_size(tt::CBIndex::c_0, tile_bytes));
    CreateCircularBuffer(
        program,
        core,
        CircularBufferConfig(stats_tile_bytes, {{tt::CBIndex::c_1, stats_fmt}})
            .set_page_size(tt::CBIndex::c_1, stats_tile_bytes));
    CreateCircularBuffer(
        program,
        core,
        CircularBufferConfig(out_tiles * tile_bytes, {{tt::CBIndex::c_16, data_fmt}})
            .set_page_size(tt::CBIndex::c_16, tile_bytes));

    auto input_buf = make_buffer(num_tiles * tile_bytes);
    auto output_buf = make_buffer(out_tiles * tile_bytes);
    distributed::EnqueueWriteMeshBuffer(cq, input_buf, encode_tiles(input_row_major, num_tiles, fp32_dest), true);

    const auto reader = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_unary.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
    SetRuntimeArgs(program, reader, core, {input_buf->address(), 0, num_tiles});

    const auto writer = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    SetRuntimeArgs(program, writer, core, {output_buf->address(), 0, out_tiles});

    const float inv_width = 1.0f / static_cast<float>(num_tiles * (tile_h / num_rows) * tile_dim);
    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    if (fp32_dest) {
        unpack_to_dest_mode[tt::CBIndex::c_0] = UnpackToDestMode::UnpackToDestFp32;
    }
    if (stats_is_fp32) {
        unpack_to_dest_mode[tt::CBIndex::c_1] = UnpackToDestMode::UnpackToDestFp32;
    }

    CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/row_norm.cpp",
        core,
        ComputeConfig{
            .fp32_dest_acc_en = fp32_dest,
            .unpack_to_dest_mode = unpack_to_dest_mode,
            .math_approx_mode = false,
            .compile_args = {
                mode,
                num_rows,
                tile_h,
                num_tiles,
                static_cast<std::uint32_t>(stats_is_fp32),
                std::bit_cast<std::uint32_t>(inv_width),
                std::bit_cast<std::uint32_t>(kEpsilon)}});

    distributed::MeshWorkload workload;
    const distributed::MeshCoordinate zero(0, 0);
    workload.add_program(distributed::MeshCoordinateRange(zero, zero), std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);

    std::vector<std::uint32_t> packed;
    distributed::EnqueueReadMeshBuffer(cq, packed, output_buf, true);
    return decode_tiles(packed, out_tiles, fp32_dest);
}

std::vector<std::vector<double>> extract_logical_rows(
    const std::vector<float>& tiles, std::uint32_t num_tiles, std::uint32_t num_rows, std::uint32_t tile_h) {
    const std::uint32_t slices_per_tile = tile_h / num_rows;
    std::vector<std::vector<double>> rows(num_rows);
    for (std::uint32_t t = 0; t < num_tiles; ++t) {
        for (std::uint32_t s = 0; s < slices_per_tile; ++s) {
            for (std::uint32_t r = 0; r < num_rows; ++r) {
                const std::uint32_t tile_row = s * num_rows + r;
                for (std::uint32_t c = 0; c < tile_dim; ++c) {
                    rows[r].push_back(tiles[(t * tile_dim + tile_row) * tile_dim + c]);
                }
            }
        }
    }
    return rows;
}

}  // namespace

TEST_F(LLKBlackholeSingleCardFixture, RowNormComputeApiSinglePass) {
    auto& mesh = *devices_.at(0);
    constexpr std::uint32_t num_rows = 4;
    constexpr std::uint32_t tile_h = 16;
    constexpr std::uint32_t num_tiles = 2;

    std::mt19937 rng(2026);
    std::uniform_real_distribution<float> dist(-2.0f, 3.0f);
    std::vector<float> input(num_tiles * tile_elements);
    for (auto& v : input) {
        v = dist(rng);
    }

    for (const std::uint32_t mode : {0u, 1u, 2u}) {
        SCOPED_TRACE(::testing::Message() << "mode=" << mode);
        std::vector<float> mode_input = input;
        if (mode == 1) {
            for (auto& v : mode_input) {
                v = v * v;
            }
        }
        const auto output = run_row_norm(mesh, mode, num_rows, tile_h, num_tiles, true, true, mode_input);
        const auto x_rows = extract_logical_rows(mode_input, num_tiles, num_rows, tile_h);

        if (mode == 1) {
            for (std::uint32_t r = 0; r < num_rows; ++r) {
                double sum = 0.0;
                for (double v : x_rows[r]) {
                    sum += v;
                }
                const double want = 1.0 / std::sqrt(sum / static_cast<double>(x_rows[r].size()) + kEpsilon);
                for (std::uint32_t row_idx = r; row_idx < tile_h; row_idx += num_rows) {
                    EXPECT_NEAR(output[row_idx * tile_dim], want, 1e-4 + 1e-4 * std::abs(want));
                }
            }
        } else {
            const auto got_rows = extract_logical_rows(output, num_tiles, num_rows, tile_h);
            for (std::uint32_t r = 0; r < num_rows; ++r) {
                double mean = 0.0;
                if (mode == 2) {
                    for (double v : x_rows[r]) {
                        mean += v;
                    }
                    mean /= static_cast<double>(x_rows[r].size());
                }
                double sum_sq = 0.0;
                for (double v : x_rows[r]) {
                    const double d = v - mean;
                    sum_sq += d * d;
                }
                const double inv_std = 1.0 / std::sqrt(sum_sq / static_cast<double>(x_rows[r].size()) + kEpsilon);
                for (std::size_t i = 0; i < x_rows[r].size(); ++i) {
                    const double want = (x_rows[r][i] - mean) * inv_std;
                    EXPECT_NEAR(got_rows[r][i], want, 1e-4 + 1e-4 * std::abs(want));
                }
            }
        }
    }
}

TEST_F(LLKBlackholeSingleCardFixture, RowNormWideRowMultiChunkAndBf16StatsInFp32Dest) {
    auto& mesh = *devices_.at(0);
    constexpr std::uint32_t num_tiles = 4;  // 2 chunks x 2 tiles

    struct Case {
        std::uint32_t num_rows;
        std::uint32_t tile_h;
        bool fp32_dest;
        bool stats_is_fp32;
        double tol;
    };
    const Case cases[] = {
        {4, 16, true, false, 1e-3},   // FP32 DEST + BF16 stats CB pack/reload (hi+lo double-BF16)
        {8, 32, true, false, 1e-3},   // M=8 FP32 DEST + BF16 stats CB pack/reload
        {4, 16, true, true, 1e-4},    // FP32 DEST + FP32 stats CB
        {4, 16, false, false, 2e-2},  // BF16 DEST + BF16 stats CB
    };

    for (const auto& tc : cases) {
        SCOPED_TRACE(
            ::testing::Message() << "num_rows=" << tc.num_rows << ", tile_h=" << tc.tile_h
                                 << ", fp32_dest=" << tc.fp32_dest << ", stats_is_fp32=" << tc.stats_is_fp32);
        std::mt19937 rng(77 + tc.num_rows * 17 + tc.tile_h);
        std::normal_distribution<float> dist0(1.5f, 1.0f);
        std::normal_distribution<float> dist1(-2.0f, 2.5f);
        std::vector<float> input(num_tiles * tile_elements);
        for (std::uint32_t t = 0; t < num_tiles; ++t) {
            for (std::uint32_t i = 0; i < tile_elements; ++i) {
                float val = (t < 2) ? dist0(rng) : dist1(rng);
                if (!tc.fp32_dest) {
                    val = static_cast<float>(bfloat16(val));
                }
                input[t * tile_elements + i] = val;
            }
        }

        const auto output =
            run_row_norm(mesh, 3u, tc.num_rows, tc.tile_h, num_tiles, tc.fp32_dest, tc.stats_is_fp32, input);
        const auto x_rows = extract_logical_rows(input, num_tiles, tc.num_rows, tc.tile_h);
        const auto got_rows = extract_logical_rows(output, num_tiles, tc.num_rows, tc.tile_h);

        for (std::uint32_t r = 0; r < tc.num_rows; ++r) {
            double mean = 0.0;
            for (double v : x_rows[r]) {
                mean += v;
            }
            mean /= static_cast<double>(x_rows[r].size());
            double var = 0.0;
            for (double v : x_rows[r]) {
                const double d = v - mean;
                var += d * d;
            }
            var /= static_cast<double>(x_rows[r].size());
            const double inv_std = 1.0 / std::sqrt(var + kEpsilon);
            for (std::size_t i = 0; i < x_rows[r].size(); ++i) {
                const double want = (x_rows[r][i] - mean) * inv_std;
                EXPECT_NEAR(got_rows[r][i], want, tc.tol + tc.tol * std::abs(want)) << "row=" << r << ", col=" << i;
            }
        }
    }
}

}  // namespace tt::tt_metal::unit_tests::compute::row_norm
