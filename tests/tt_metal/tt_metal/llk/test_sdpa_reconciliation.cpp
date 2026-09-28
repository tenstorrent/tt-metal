// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "llk_device_fixture.hpp"
#include "impl/program/program_impl.hpp"

namespace tt::tt_metal {
namespace {

void run_sdpa_recip(distributed::MeshDevice& mesh, std::uint32_t fidelity, std::uint32_t granularity) {
    auto& cq = mesh.mesh_command_queue();
    const CoreCoord core{0, 0};
    constexpr std::uint32_t tiles = 3;
    constexpr std::uint32_t tile_bytes = 2048;
    Program program = CreateProgram();
    auto make_buffer = [&]() {
        return distributed::MeshBuffer::create(
            distributed::ReplicatedBufferConfig{.size = tiles * tile_bytes},
            {.page_size = tiles * tile_bytes, .buffer_type = BufferType::DRAM},
            &mesh);
    };
    auto input = make_buffer();
    auto output = make_buffer();
    for (const auto cb : {tt::CBIndex::c_0, tt::CBIndex::c_16}) {
        CreateCircularBuffer(
            program,
            core,
            CircularBufferConfig(tile_bytes, {{cb, tt::DataFormat::Float16_b}}).set_page_size(cb, tile_bytes));
    }
    // The normalizer's logical tile is two 8x16 faces. Data is already in
    // DEST; this CB supplies the compute API's operand format/shape metadata.
    CreateCircularBuffer(
        program,
        core,
        CircularBufferConfig(512, {{tt::CBIndex::c_1, tt::DataFormat::Float16_b}})
            .set_page_size(tt::CBIndex::c_1, 512)
            .set_unpack_face_geometry(tt::CBIndex::c_1, 8, 2));
    auto reader = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_unary.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
    auto writer = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/sdpa_recip_reconciliation.cpp",
        core,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = false,
            .math_approx_mode = false,
            .compile_args = {tiles, fidelity, granularity}});

    std::vector<bfloat16> source(tiles * 1024);
    std::vector<bfloat16> golden(source.size());
    for (std::uint32_t i = 0; i < source.size(); ++i) {
        source[i] = bfloat16(static_cast<float>(static_cast<int>(i * 17 % 63) - 31) / 8.0f);
        // Two 8x32 SDPA tiles occupy the first 512 elements of each full tile.
        // The other half is a guard for writes beyond the normalized region.
        const float divisor = i % 1024 < 512 ? 2.0f : 1.0f;
        golden[i] = bfloat16(static_cast<float>(source[i]) / divisor);
    }
    distributed::EnqueueWriteMeshBuffer(cq, input, pack_bfloat16_vec_into_uint32_vec(source), true);
    SetRuntimeArgs(program, reader, core, {input->address(), 0, tiles});
    SetRuntimeArgs(program, writer, core, {output->address(), 0, tiles});
    distributed::MeshWorkload workload;
    const distributed::MeshCoordinate zero(0, 0);
    workload.add_program(distributed::MeshCoordinateRange(zero, zero), std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);
    std::vector<std::uint32_t> result;
    distributed::EnqueueReadMeshBuffer(cq, result, output, true);
    EXPECT_EQ(result, pack_bfloat16_vec_into_uint32_vec(golden));
}

constexpr std::uint32_t sdpa_tail_rows = 8;
constexpr std::uint32_t sdpa_tail_cols = 32;
constexpr std::uint32_t sdpa_tail_elements = sdpa_tail_rows * sdpa_tail_cols;

std::uint32_t sdpa_tail_face_index(std::uint32_t row, std::uint32_t col) {
    return (col / 16) * sdpa_tail_rows * 16 + row * 16 + col % 16;
}

// Exercise the real tail producer: row-wise max/sum reduction, SRCB reuse
// broadcast multiplication, and either sparse block packing or dense untilize.
// copy_tile alone cannot establish the short-face DEST layout this path uses.
void run_sdpa_tail(
    distributed::MeshDevice& mesh,
    bool normalize,
    bool untilize,
    bool fp32_dest_acc,
    bool full_sync,
    std::uint32_t num_blocks) {
    auto& cq = mesh.mesh_command_queue();
    const CoreCoord core{0, 0};
    constexpr std::uint32_t rounds = 3;
    constexpr std::uint32_t block_size = 2;
    constexpr std::uint32_t tile_bytes = sdpa_tail_elements * sizeof(bfloat16);
    const std::uint32_t tiles = block_size * num_blocks;
    Program program = CreateProgram();
    auto make_buffer = [&](std::uint32_t count) {
        return distributed::MeshBuffer::create(
            distributed::ReplicatedBufferConfig{.size = count * tile_bytes},
            {.page_size = count * tile_bytes, .buffer_type = BufferType::DRAM},
            &mesh);
    };
    std::array<std::shared_ptr<distributed::MeshBuffer>, 4> inputs{
        make_buffer(rounds), make_buffer(rounds), make_buffer(rounds * tiles), make_buffer(rounds * tiles)};
    auto output = make_buffer(rounds * tiles);
    auto output_stats = make_buffer(rounds);
    for (const auto cb :
         {tt::CBIndex::c_0,
          tt::CBIndex::c_1,
          tt::CBIndex::c_2,
          tt::CBIndex::c_3,
          tt::CBIndex::c_16,
          tt::CBIndex::c_17}) {
        const std::uint32_t pages =
            cb == tt::CBIndex::c_0 || cb == tt::CBIndex::c_1 || cb == tt::CBIndex::c_17 ? 2 : 2 * tiles;
        CreateCircularBuffer(
            program,
            core,
            CircularBufferConfig(pages * tile_bytes, {{cb, tt::DataFormat::Float16_b}})
                .set_page_size(cb, tile_bytes)
                .set_tile_dims(cb, Tile({sdpa_tail_rows, sdpa_tail_cols})));
    }
    const auto reader = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_sdpa_tail_reconciliation.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = {rounds, tiles}});
    const auto writer = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_sdpa_tail_reconciliation.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = {rounds, tiles, normalize}});
    CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/sdpa_tail_reconciliation.cpp",
        core,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = fp32_dest_acc,
            .dst_full_sync_en = full_sync,
            .math_approx_mode = false,
            .compile_args = {rounds, block_size, num_blocks, normalize, untilize}});

    std::array<std::vector<bfloat16>, 4> source{
        std::vector<bfloat16>(rounds * sdpa_tail_elements, bfloat16(0.0f)),
        std::vector<bfloat16>(rounds * sdpa_tail_elements, bfloat16(0.0f)),
        std::vector<bfloat16>(rounds * tiles * sdpa_tail_elements),
        std::vector<bfloat16>(rounds * tiles * sdpa_tail_elements)};
    std::vector<float> golden(rounds * tiles * sdpa_tail_elements);
    std::vector<float> rounding_bound(golden.size());
    std::vector<float> golden_stats(rounds * sdpa_tail_elements, 0.0f);
    for (std::uint32_t round = 0; round < rounds; ++round) {
        for (std::uint32_t row = 0; row < sdpa_tail_rows; ++row) {
            // Unequal, row-dependent maxima exercise both branches of max and
            // both exponential weights. Vary every round to catch stale state.
            const float worker_max = static_cast<float>(row % 3) * 0.5f - 0.5f;
            const float previous_max = static_cast<float>((row + round) % 4) * 0.25f - 0.25f;
            const float worker_sum = 1.0f + static_cast<float>((row + round) % 3) * 0.5f;
            const float previous_sum = 1.5f + static_cast<float>((row + 2 * round) % 4) * 0.25f;
            const auto stats_index = round * sdpa_tail_elements + sdpa_tail_face_index(row, 0);
            source[0][stats_index] = bfloat16(worker_max);
            source[0][stats_index + 1] = bfloat16(worker_sum);
            source[1][stats_index] = bfloat16(previous_max);
            source[1][stats_index + 1] = bfloat16(previous_sum);
            const float maximum = std::max(worker_max, previous_max);
            const float worker_weight = std::exp((worker_max - maximum) * 0.5f);
            const float previous_weight = std::exp((previous_max - maximum) * 0.5f);
            const float denominator = worker_sum * worker_weight + previous_sum * previous_weight;
            golden_stats[stats_index] = maximum;
            golden_stats[stats_index + 1] = denominator;
            for (std::uint32_t tile = 0; tile < tiles; ++tile) {
                for (std::uint32_t col = 0; col < sdpa_tail_cols; ++col) {
                    const auto source_index =
                        (round * tiles + tile) * sdpa_tail_elements + sdpa_tail_face_index(row, col);
                    const auto worker_value = bfloat16(
                        static_cast<float>(static_cast<int>((tile * 23 + row * 11 + col + round * 7) % 61) - 30) /
                        16.0f);
                    const auto previous_value = bfloat16(
                        static_cast<float>(static_cast<int>((tile * 13 + row * 3 + col * 7 + round * 17) % 53) - 26) /
                        16.0f);
                    source[2][source_index] = worker_value;
                    source[3][source_index] = previous_value;
                    const float worker_term = static_cast<float>(worker_value) * worker_weight;
                    const float previous_term = static_cast<float>(previous_value) * previous_weight;
                    const float value = worker_term + previous_term;
                    const auto output_index = untilize ? round * tiles * sdpa_tail_elements +
                                                             row * tiles * sdpa_tail_cols + tile * sdpa_tail_cols + col
                                                       : source_index;
                    golden[output_index] = normalize ? value / denominator : value;
                    // The two BF16-weighted terms can nearly cancel. Bound
                    // coefficient/product/accumulator rounding by the terms'
                    // magnitudes, not by their potentially tiny final sum.
                    // Four BF16 unit roundoffs cover these rounding stages;
                    // a separate relative-L2 check below rejects broad drift.
                    constexpr float bf16_unit_roundoff = 1.0f / 256.0f;
                    rounding_bound[output_index] = 4.0f * bf16_unit_roundoff *
                                                   (std::abs(worker_term) + std::abs(previous_term)) /
                                                   (normalize ? denominator : 1.0f);
                }
            }
        }
    }
    for (std::uint32_t index = 0; index < inputs.size(); ++index) {
        distributed::EnqueueWriteMeshBuffer(cq, inputs[index], pack_bfloat16_vec_into_uint32_vec(source[index]), true);
    }
    SetRuntimeArgs(
        program,
        reader,
        core,
        {inputs[0]->address(), inputs[1]->address(), inputs[2]->address(), inputs[3]->address()});
    SetRuntimeArgs(program, writer, core, {output->address(), output_stats->address()});
    distributed::MeshWorkload workload;
    const distributed::MeshCoordinate zero(0, 0);
    workload.add_program(distributed::MeshCoordinateRange(zero, zero), std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);
    std::vector<std::uint32_t> result;
    distributed::EnqueueReadMeshBuffer(cq, result, output, true);
    const auto actual = unpack_uint32_vec_into_bfloat16_vec(result);
    ASSERT_EQ(actual.size(), golden.size());
    double squared_error = 0.0;
    double squared_golden = 0.0;
    for (std::uint32_t index = 0; index < actual.size(); ++index) {
        const float value = static_cast<float>(actual[index]);
        ASSERT_TRUE(std::isfinite(value)) << "output index=" << index;
        ASSERT_NEAR(value, golden[index], rounding_bound[index]) << "output index=" << index;
        squared_error += std::pow(static_cast<double>(value) - golden[index], 2);
        squared_golden += std::pow(static_cast<double>(golden[index]), 2);
    }
    ASSERT_LT(std::sqrt(squared_error / squared_golden), 0.01);
    if (!normalize) {
        distributed::EnqueueReadMeshBuffer(cq, result, output_stats, true);
        const auto actual_stats = unpack_uint32_vec_into_bfloat16_vec(result);
        ASSERT_EQ(actual_stats.size(), golden_stats.size());
        for (std::uint32_t round = 0; round < rounds; ++round) {
            for (std::uint32_t row = 0; row < sdpa_tail_rows; ++row) {
                const auto index = round * sdpa_tail_elements + sdpa_tail_face_index(row, 0);
                ASSERT_EQ(static_cast<float>(actual_stats[index]), golden_stats[index]);
                ASSERT_NEAR(
                    static_cast<float>(actual_stats[index + 1]),
                    golden_stats[index + 1],
                    0.025f * golden_stats[index + 1]);
            }
        }
    }
}

}  // namespace

TEST_F(LLKBlackholeSingleCardFixture, SdpaRecipFidelityAndSignalling) {
    for (const auto fidelity : {0u, 2u, 3u, 4u, 255u}) {
        for (const auto granularity : {1u, 2u}) {
            SCOPED_TRACE(::testing::Message() << "fidelity=" << fidelity << ", granularity=" << granularity);
            run_sdpa_recip(*devices_.at(0), fidelity, granularity);
        }
    }
}

TEST_F(LLKBlackholeSingleCardFixture, SdpaTailShortFaceProducerAndUntilize) {
    // Three invocations reuse both DEST banks; three blocks also exercise
    // untilize's full-width row stride and nonzero block-column offsets.
    for (const bool normalize : {false, true}) {
        for (const bool untilize : {false, true}) {
            for (const bool fp32_dest_acc : {false, true}) {
                for (const bool full_sync : {false, true}) {
                    for (const auto num_blocks : {1u, 3u}) {
                        SCOPED_TRACE(
                            ::testing::Message() << "normalize=" << normalize << ", untilize=" << untilize
                                                 << ", fp32_dest_acc=" << fp32_dest_acc << ", full_sync=" << full_sync
                                                 << ", blocks=" << num_blocks);
                        run_sdpa_tail(*devices_.at(0), normalize, untilize, fp32_dest_acc, full_sync, num_blocks);
                    }
                }
            }
        }
    }
}

TEST_F(LLKBlackholeSingleCardFixture, SdpaChunkSemaphoreCompileLimits) {
    // This is a compile-time contract test; these programs are never launched.
    // 14 tiles with unit signaling and 16 tiles with grouped signaling fit the
    // 4-bit semaphore. At 16 tiles, either unit-signaling path must be rejected.
    for (const auto& args : {std::vector<std::uint32_t>{14, 1, 1}, {16, 2, 2}, {16, 1, 2}, {16, 2, 1}}) {
        SCOPED_TRACE(::testing::Message() << "chunk=" << args[0] << ", qk=" << args[1] << ", exp=" << args[2]);
        const bool fits = args[0] / args[1] + 1 <= 15 && args[0] / args[2] + 1 <= 15;
        Program program = CreateProgram();
        const CoreCoord core{0, 0};
        for (const auto cb :
             {tt::CBIndex::c_0, tt::CBIndex::c_1, tt::CBIndex::c_2, tt::CBIndex::c_3, tt::CBIndex::c_16}) {
            auto config = CircularBufferConfig(2048, {{cb, tt::DataFormat::Float16_b}}).set_page_size(cb, 2048);
            if (cb == tt::CBIndex::c_0 || cb == tt::CBIndex::c_16) {
                config.set_unpack_face_geometry(cb, 8, 2);
            }
            CreateCircularBuffer(program, core, config);
        }
        CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/compute/sdpa_chunk_compile_limits.cpp",
            core,
            ComputeConfig{.math_fidelity = MathFidelity::LoFi, .fp32_dest_acc_en = false, .compile_args = args});
        auto* device = devices_.at(0)->get_devices()[0];
        if (fits) {
            EXPECT_NO_THROW(program.impl().compile(device));
        } else {
            EXPECT_THROW(program.impl().compile(device), std::runtime_error);
        }
    }
}

}  // namespace tt::tt_metal
