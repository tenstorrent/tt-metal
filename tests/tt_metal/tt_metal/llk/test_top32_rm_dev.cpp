// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include <fmt/format.h>
#include <gtest/gtest.h>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-logger/tt-logger.hpp>
#include <umd/device/types/arch.hpp>

#include "device_fixture.hpp"
#include <tt-metalium/distributed.hpp>

#include "tt_metal/test_utils/packing.hpp"

namespace tt::tt_metal {

using namespace tt::test_utils;

namespace unit_tests::compute::top32_rm_dev {

constexpr uint32_t kOutputTiles = 1;

uint32_t num_input_tiles_for_row(uint32_t row_elements) { return (row_elements + 1023) / 1024; }

constexpr uint32_t kTopK = 32;

// Build row-major DRAM: scores are a shuffled decreasing sequence; indices[i] is the original index
// (0..L-1) of the score at row-major position i after the same shuffle.
struct ShuffledInputs {
    std::vector<uint32_t> packed_scores;
    std::vector<uint32_t> indices_u32;
};

ShuffledInputs make_shuffled_inputs_row_major(uint32_t row_elements, uint32_t seed) {
    const uint32_t nt = num_input_tiles_for_row(row_elements);
    const uint32_t total_bf16 = nt * 1024;
    const uint32_t n_el = nt * 1024;

    struct Pair {
        bfloat16 score;
        uint32_t orig_idx;
    };
    std::vector<Pair> pairs(row_elements);
    for (uint32_t i = 0; i < row_elements; i++) {
        pairs[i] = {bfloat16(static_cast<float>(row_elements - 1 - i)), i};
    }
    std::mt19937 rng(seed);
    std::shuffle(pairs.begin(), pairs.end(), rng);

    std::vector<bfloat16> scores(total_bf16, bfloat16(0.0f));
    std::vector<uint32_t> indices(n_el, 0u);
    for (uint32_t i = 0; i < row_elements; i++) {
        scores[i] = pairs[i].score;
        indices[i] = pairs[i].orig_idx;
    }
    return {pack_vector<uint32_t, bfloat16>(scores), std::move(indices)};
}

// Expect output slot k to match unshuffled rank k: score (L-1-k), index k (strict descending order in DRAM).
bool verify_top32_outputs(
    const std::vector<uint32_t>& packed_scores_out, const std::vector<uint32_t>& indices_out, uint32_t row_elements) {
    auto out_scores = unpack_vector<bfloat16, uint32_t>(packed_scores_out);
    if (out_scores.size() < kTopK || indices_out.size() < kTopK) {
        log_error(
            LogTest,
            "Top32 verify: buffer too small (scores={}, indices={}, need {})",
            out_scores.size(),
            indices_out.size(),
            kTopK);
        return false;
    }

    std::string table;
    table += fmt::format("Top32 L={} — expected: slot k has score (L-1-k) and index k\n", row_elements);
    table += fmt::format("{:-^86}\n", "");
    table += fmt::format(
        "{:>4} | {:>12} | {:>12} | {:^5} | {:>8} | {:>8} | {:^5}\n",
        "k",
        "exp_score",
        "got_score",
        "s_ok",
        "exp_idx",
        "got_idx",
        "i_ok");
    table += fmt::format("{:-^86}\n", "");

    bool all_ok = true;
    for (uint32_t k = 0; k < kTopK; k++) {
        const bfloat16 want_score = bfloat16(static_cast<float>(row_elements - 1 - k));
        const uint32_t want_idx = k;
        const bfloat16 got_score = out_scores[k];
        const uint32_t got_idx = indices_out[k];
        const bool s_ok = (want_score == got_score);
        const bool i_ok = (want_idx == got_idx);
        all_ok &= s_ok;  // can ignore i_ok because sorting is not stable
        table += fmt::format(
            "{:4} | {:12.6f} | {:12.6f} | {:^5} | {:8} | {:8} | {:^5}\n",
            k,
            static_cast<float>(want_score),
            static_cast<float>(got_score),
            s_ok ? "ok" : "BAD",
            want_idx,
            got_idx,
            i_ok ? "ok" : "BAD");
    }
    table += fmt::format("{:-^86}\n", "");
    table += fmt::format("overall: {}\n", all_ok ? "PASS" : "FAIL");

    log_info(LogTest, "{}", table);

    return all_ok;
}

bool run_top32_rm_dev(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device, uint32_t row_elements, uint32_t seed) {
    auto* device = mesh_device->get_devices()[0];
    auto& cq = mesh_device->mesh_command_queue();
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    distributed::MeshWorkload workload;
    Program program = CreateProgram();
    workload.add_program(device_range, std::move(program));
    auto& program_ = workload.get_programs().at(device_range);

    const uint32_t num_in_tiles = num_input_tiles_for_row(row_elements);
    const uint32_t tile_bf16 = tt::tile_size(DataFormat::Float16_b);
    const uint32_t tile_u32 = tt::tile_size(DataFormat::UInt32);

    const uint32_t in0_bytes = num_in_tiles * tile_bf16;
    const uint32_t in1_bytes = num_in_tiles * tile_u32;
    const uint32_t out0_bytes = kOutputTiles * tile_bf16;
    const uint32_t out1_bytes = kOutputTiles * tile_u32;

    InterleavedBufferConfig dram_in0{
        .device = device, .size = in0_bytes, .page_size = in0_bytes, .buffer_type = BufferType::DRAM};
    InterleavedBufferConfig dram_in1{
        .device = device, .size = in1_bytes, .page_size = in1_bytes, .buffer_type = BufferType::DRAM};
    InterleavedBufferConfig dram_out0{
        .device = device, .size = out0_bytes, .page_size = out0_bytes, .buffer_type = BufferType::DRAM};
    InterleavedBufferConfig dram_out1{
        .device = device, .size = out1_bytes, .page_size = out1_bytes, .buffer_type = BufferType::DRAM};

    auto buf_in0 = CreateBuffer(dram_in0);
    auto buf_in1 = CreateBuffer(dram_in1);
    auto buf_out0 = CreateBuffer(dram_out0);
    auto buf_out1 = CreateBuffer(dram_out1);

    CoreCoord core{0, 0};
    CoreRangeSet crs({CoreRange(core)});

    const uint32_t cb_depth = std::max(2u, num_in_tiles);
    CreateCircularBuffer(
        program_,
        core,
        CircularBufferConfig(cb_depth * tile_bf16, {{tt::CBIndex::c_0, DataFormat::Float16_b}})
            .set_page_size(tt::CBIndex::c_0, tile_bf16));
    CreateCircularBuffer(
        program_,
        core,
        CircularBufferConfig(cb_depth * tile_u32, {{tt::CBIndex::c_1, DataFormat::UInt32}})
            .set_page_size(tt::CBIndex::c_1, tile_u32));
    CreateCircularBuffer(
        program_,
        core,
        CircularBufferConfig(kOutputTiles * tile_bf16, {{tt::CBIndex::c_16, DataFormat::Float16_b}})
            .set_page_size(tt::CBIndex::c_16, tile_bf16));
    CreateCircularBuffer(
        program_,
        core,
        CircularBufferConfig(kOutputTiles * tile_u32, {{tt::CBIndex::c_17, DataFormat::UInt32}})
            .set_page_size(tt::CBIndex::c_17, tile_u32));

    auto reader = CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_binary.cpp",
        crs,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

    auto writer = CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_dual_unary.cpp",
        crs,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    std::vector<uint32_t> compute_compile_args = {row_elements, num_in_tiles, kOutputTiles};
    CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/compute/top32_rm_dev_compute.cpp",
        crs,
        ComputeConfig{.fp32_dest_acc_en = true, .compile_args = compute_compile_args});

    SetRuntimeArgs(program_, reader, core, {buf_in0->address(), 0u, buf_in1->address(), 0u, num_in_tiles});
    SetRuntimeArgs(program_, writer, core, {buf_out0->address(), 0u, buf_out1->address(), 0u, kOutputTiles});

    ShuffledInputs in = make_shuffled_inputs_row_major(row_elements, seed);

    detail::WriteToBuffer(buf_in0, in.packed_scores);
    detail::WriteToBuffer(buf_in1, in.indices_u32);

    EnqueueMeshWorkload(cq, workload, false);
    Finish(cq);

    std::vector<uint32_t> out0;
    std::vector<uint32_t> out1;
    detail::ReadFromBuffer(buf_out0, out0);
    detail::ReadFromBuffer(buf_out1, out1);
    return verify_top32_outputs(out0, out1, row_elements);
}

}  // namespace unit_tests::compute::top32_rm_dev

TEST_F(MeshDeviceFixture, Top32RmDevPipelineCompletes) {
    for (uint32_t row : {160u, 3232u}) {
        log_info(LogTest, "Top32 RM dev row_elements={}", row);
        EXPECT_TRUE(unit_tests::compute::top32_rm_dev::run_top32_rm_dev(this->devices_.at(0), row, 12345u));
    }
}

}  // namespace tt::tt_metal
