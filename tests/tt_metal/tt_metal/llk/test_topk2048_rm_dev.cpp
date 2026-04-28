// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <bit>
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
#include <tt-metalium/circular_buffer_constants.h>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-logger/tt-logger.hpp>

#include "device_fixture.hpp"
#include <tt-metalium/distributed.hpp>

namespace tt::tt_metal {

namespace unit_tests::compute::topk2048_rm_dev {

// Top-K output (two Float32 tiles = 2048 elements).
constexpr uint32_t kTopK = 2048;
constexpr uint32_t kOutputTiles = 2;

// Total elements to sort (valid keys). Default one full pass of two tiles; optional 4 tiles:
constexpr uint32_t N = 2048;
// constexpr uint32_t N = 4096;

enum class InputMode { Random, NiceRounds };
constexpr InputMode kInputMode = InputMode::Random;

uint32_t num_input_tiles_for_n(uint32_t n) { return (n + tt::constants::TILE_HW - 1) / tt::constants::TILE_HW; }

uint32_t pack_bf16_index(bfloat16 bf, uint16_t idx) {
    return (static_cast<uint32_t>(std::bit_cast<uint16_t>(bf)) << 16) | static_cast<uint32_t>(idx);
}

static void decode_u32_key(uint32_t w, float* out_f, uint16_t* out_idx) {
    *out_idx = static_cast<uint16_t>(w & 0xFFFFu);
    *out_f = std::bit_cast<float>(w);
}

std::vector<uint32_t> make_input_keys(uint32_t n, uint32_t num_el_padded, uint32_t seed) {
    std::vector<uint32_t> keys(num_el_padded, 0u);
    if (kInputMode == InputMode::NiceRounds) {
        for (uint32_t i = 0; i < n; i++) {
            float v = static_cast<float>(static_cast<int32_t>(i % 256) - 128);
            keys[i] = pack_bf16_index(bfloat16(v), static_cast<uint16_t>(i & 0xFFFFu));
        }
    } else {
        std::mt19937 gen(seed);
        std::uniform_real_distribution<float> dis(-32.0f, 32.0f);
        for (uint32_t i = 0; i < n; i++) {
            keys[i] = pack_bf16_index(bfloat16(dis(gen)), static_cast<uint16_t>(i & 0xFFFFu));
            // make every pair of elements the same, this way I can compare odd/even columns to verify sorting
            if (i % 2 == 1) {
                keys[i] = keys[i - 1];
            }
        }
    }
    // Pad so the last tile is full; keep keys that sort to the bottom.
    bfloat16 pad_bf = bfloat16(-std::numeric_limits<float>::infinity());
    for (uint32_t i = n; i < num_el_padded; i++) {
        keys[i] = pack_bf16_index(pad_bf, 0xFFFFu);
    }
    return keys;
}

std::vector<uint32_t> golden_topk_u32(const std::vector<uint32_t>& keys_in, uint32_t n, uint32_t k) {
    std::vector<uint32_t> a(keys_in.begin(), keys_in.begin() + n);
    std::sort(a.begin(), a.end(), [](uint32_t x, uint32_t y) {
        float fx = std::bit_cast<float>(x);
        float fy = std::bit_cast<float>(y);
        if (fx != fy) {
            return fx > fy;
        }
        return (x & 0xFFFFu) < (y & 0xFFFFu);
    });
    a.resize(k);
    return a;
}

bool verify_topk_outputs(const std::vector<uint32_t>& out_u32, const std::vector<uint32_t>& exp_u32) {
    if (out_u32.size() < kTopK || exp_u32.size() < kTopK) {
        log_error(
            LogTest,
            "TopK2048 verify: output too small (out={}, exp={}, need {})",
            out_u32.size(),
            exp_u32.size(),
            kTopK);
        return false;
    }

    std::string table;
    table += fmt::format("TopK2048 (k={}) — exact uint32 match vs sorted golden\n", kTopK);
    table += fmt::format("{:-^100}\n", "");
    table += fmt::format(
        "{:>4} | {:>10} | {:>10} | {:>5} | {:>12} | {:>12} | {:>5} | {:>5} | {:>5}\n",
        "k",
        "exp_u32",
        "got_u32",
        "u32==",
        "exp_f",
        "got_f",
        "e_idx",
        "g_idx",
        "ok");
    table += fmt::format("{:-^100}\n", "");

    bool all_ok = true;
    for (uint32_t k = 0; k < kTopK; k++) {
        const uint32_t w_exp = exp_u32[k];
        const uint32_t w_got = out_u32[k];
        const bool m = (w_exp == w_got);
        all_ok &= m;
        float fe = 0.f, fg = 0.f;
        uint16_t ie = 0, ig = 0;
        decode_u32_key(w_exp, &fe, &ie);
        decode_u32_key(w_got, &fg, &ig);
        table += fmt::format(
            "{:4} | 0x{:08X} | 0x{:08X} | {:>5} | {:12.6f} | {:12.6f} | {:5} | {:5} | {:>5}\n",
            k,
            w_exp,
            w_got,
            m ? "yes" : "no",
            fe,
            fg,
            ie,
            ig,
            m ? "ok" : "BAD");
    }
    table += fmt::format("{:-^100}\n", "");
    table += fmt::format("overall: {}\n", all_ok ? "PASS" : "FAIL");
    // log_info(LogTest, "{}", table);
    return all_ok;
}

bool run_topk2048_rm_dev(const std::shared_ptr<distributed::MeshDevice>& mesh_device, uint32_t seed) {
    auto* device = mesh_device->get_devices()[0];
    auto& cq = mesh_device->mesh_command_queue();
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    distributed::MeshWorkload workload;
    Program program = CreateProgram();
    workload.add_program(device_range, std::move(program));
    auto& program_ = workload.get_programs().at(device_range);

    const uint32_t num_in_tiles = num_input_tiles_for_n(N);
    const uint32_t tile_fp32 = tt::tile_size(::tt::DataFormat::Float32);
    const uint32_t el_per_tile = tt::constants::TILE_HW;
    const uint32_t num_el_padded = num_in_tiles * el_per_tile;

    const uint32_t in_bytes = num_in_tiles * tile_fp32;
    const uint32_t out_bytes = kOutputTiles * tile_fp32;

    InterleavedBufferConfig dram_in{
        .device = device, .size = in_bytes, .page_size = in_bytes, .buffer_type = BufferType::DRAM};
    InterleavedBufferConfig dram_out{
        .device = device, .size = out_bytes, .page_size = out_bytes, .buffer_type = BufferType::DRAM};

    auto buf_in = CreateBuffer(dram_in);
    auto buf_out = CreateBuffer(dram_out);

    CoreCoord core{0, 0};
    CoreRangeSet crs({CoreRange(core)});

    const uint32_t cb_depth = std::max(2u, num_in_tiles);
    CreateCircularBuffer(
        program_,
        core,
        CircularBufferConfig(cb_depth * tile_fp32, {{tt::CBIndex::c_0, ::tt::DataFormat::Float32}})
            .set_page_size(tt::CBIndex::c_0, tile_fp32));
    CreateCircularBuffer(
        program_,
        core,
        CircularBufferConfig(kOutputTiles * tile_fp32, {{tt::CBIndex::c_16, ::tt::DataFormat::Float32}})
            .set_page_size(tt::CBIndex::c_16, tile_fp32));

    auto reader = CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_unary_push_n.cpp",
        crs,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

    auto writer = CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary_pop_n.cpp",
        crs,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    std::vector<uint32_t> compute_compile_args = {num_in_tiles, kOutputTiles};

    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    unpack_to_dest_mode[tt::CBIndex::c_0] = UnpackToDestMode::UnpackToDestFp32;

    CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/compute/topk2048_rm_dev_compute.cpp",
        crs,
        ComputeConfig{
            .fp32_dest_acc_en = true,
            .unpack_to_dest_mode = std::move(unpack_to_dest_mode),
            .compile_args = std::move(compute_compile_args)});

    SetRuntimeArgs(program_, reader, core, {buf_in->address(), 0u, num_in_tiles, tt::CBIndex::c_0, 1u, 0u});
    SetRuntimeArgs(program_, writer, core, {buf_out->address(), 0u, kOutputTiles, tt::CBIndex::c_16, 1u, 0u});

    std::vector<uint32_t> in_keys = make_input_keys(N, num_el_padded, seed);
    std::vector<uint32_t> exp = golden_topk_u32(in_keys, N, kTopK);

    detail::WriteToBuffer(buf_in, in_keys);

    EnqueueMeshWorkload(cq, workload, false);
    Finish(cq);

    std::vector<uint32_t> out_u32;
    detail::ReadFromBuffer(buf_out, out_u32);

    return verify_topk_outputs(out_u32, exp);
}

}  // namespace unit_tests::compute::topk2048_rm_dev

TEST_F(BlackholeSingleCardFixture, TopK2048RmDevPipeline) {
    EXPECT_TRUE(unit_tests::compute::topk2048_rm_dev::run_topk2048_rm_dev(this->devices_.at(0), 12345u));
}

}  // namespace tt::tt_metal
