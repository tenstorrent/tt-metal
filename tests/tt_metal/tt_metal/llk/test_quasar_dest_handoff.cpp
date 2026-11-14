// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Quasar tile_regs_handoff(): a section that mixes an unpack-to-dest copy with a src-path run longer than the SrcA /
// SrcB banks (the fp32 partials reload of the unified matmul, #62102). Two kernel variants: every section copies then
// hands off; or (HANDOFF_EVERY_SECTION) sections alternate copy / no copy and every one hands off, so the hand-off
// also runs in a section with nothing to-dest. Both orderings must post the section handshake exactly once, checked by
// bit-exact output over four sections in SyncHalf and SyncFull (no hang, no bank skew).

#include <gtest/gtest.h>

#include <bit>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-logger/tt-logger.hpp>
#include "impl/program/program_impl.hpp"

#include "llk_device_fixture.hpp"
#include "test_golden_impls.hpp"

namespace tt::tt_metal {

namespace unit_tests::compute::quasar_dest_handoff {

constexpr uint32_t kTileHW = 32 * 32;
constexpr uint32_t kSections = 4;  // visits both SyncHalf banks twice
constexpr uint32_t kSrcTiles = 6;  // more than the two SrcA/SrcB banks: hangs without the hand-off

struct Variant {
    bool handoff_every_section;  // odd sections have no copy but still hand off
    bool dst_full_sync_en;
};

// Stimulus, small integers so every value is exact in Float16_b and every sum exact in Float32:
//   a(s,k)[r][c] = r + 1 + 4k + 32s      b(s,k)[r][c] = c + 1 + 2k + 16s      fp32(s)[r][c] = 1000(s+1) + 32r + c + 0.5
// Expected per section s: out tile 0 = fp32(s) in a copy section, a(s,0)+b(s,0) in a no-copy section;
//                         out tile 1 = a(s,5)+b(s,5): add_tiles overwrites DST[1], so the last pair wins.
float a_value(uint32_t s, uint32_t k, uint32_t r, uint32_t /*c*/) { return static_cast<float>(r + 1 + 4 * k + 32 * s); }
float b_value(uint32_t s, uint32_t k, uint32_t /*r*/, uint32_t c) { return static_cast<float>(c + 1 + 2 * k + 16 * s); }
float fp32_value(uint32_t s, uint32_t r, uint32_t c) { return 1000.0f * (s + 1) + 32.0f * r + c + 0.5f; }

std::vector<uint32_t> reinterpret_fp32_as_u32(const std::vector<float>& in) {
    std::vector<uint32_t> out(in.size());
    static_assert(sizeof(float) == sizeof(uint32_t));
    std::memcpy(out.data(), in.data(), in.size() * sizeof(float));
    return out;
}

std::vector<uint32_t> pack_bf16_as_u32(const std::vector<float>& in) {
    std::vector<uint32_t> out(in.size() / 2);
    for (size_t i = 0; i < out.size(); ++i) {
        out[i] = pack_two_bfloat16_into_uint32({bfloat16(in[2 * i]), bfloat16(in[2 * i + 1])});
    }
    return out;
}

// One 32x32 tile from a row-major generator, tilized into the device layout.
template <typename F>
std::vector<uint32_t> make_tile(F value, bool bf16) {
    std::vector<float> rm(kTileHW);
    for (uint32_t r = 0; r < 32; ++r) {
        for (uint32_t c = 0; c < 32; ++c) {
            rm[r * 32 + c] = value(r, c);
        }
    }
    ::unit_tests::compute::GoldenConfig cfg{.num_tiles_r_dim = 1, .num_tiles_c_dim = 1, .datum_bytes = bf16 ? 2u : 4u};
    return ::unit_tests::compute::gold_standard_tilize(bf16 ? pack_bf16_as_u32(rm) : reinterpret_fp32_as_u32(rm), cfg);
}

void append(std::vector<uint32_t>& to, const std::vector<uint32_t>& tile) {
    to.insert(to.end(), tile.begin(), tile.end());
}

bool run(distributed::MeshDevice& mesh_device, const Variant& v) {
    const experimental::NodeCoord node{0, 0};
    const experimental::DFBSpecName IN_A{"in_a_dfb"};
    const experimental::DFBSpecName IN_B{"in_b_dfb"};
    const experimental::DFBSpecName IN_FP32{"in_fp32_dfb"};
    const experimental::DFBSpecName OUT{"out_dfb"};
    const experimental::KernelSpecName WRITER{"writer"};
    const experimental::KernelSpecName COMPUTE{"compute"};

    const uint32_t fp32_tile_bytes = tt::tile_size(tt::DataFormat::Float32);
    auto has_copy = [&](uint32_t s) { return !v.handoff_every_section || (s % 2) == 0; };

    // Stream order = consumption order: per section kSrcTiles a tiles, kSrcTiles b tiles, one fp32 tile in copy
    // sections; output = DST[0], DST[1] per section.
    std::vector<uint32_t> a_all, b_all, fp32_all, golden;
    uint32_t num_fp32_tiles = 0;
    for (uint32_t s = 0; s < kSections; ++s) {
        for (uint32_t k = 0; k < kSrcTiles; ++k) {
            append(a_all, make_tile([&](uint32_t r, uint32_t c) { return a_value(s, k, r, c); }, true));
            append(b_all, make_tile([&](uint32_t r, uint32_t c) { return b_value(s, k, r, c); }, true));
        }
        if (has_copy(s)) {
            auto fp32_tile = make_tile([&](uint32_t r, uint32_t c) { return fp32_value(s, r, c); }, false);
            append(fp32_all, fp32_tile);
            append(golden, fp32_tile);
            ++num_fp32_tiles;
        } else {
            append(
                golden,
                make_tile([&](uint32_t r, uint32_t c) { return a_value(s, 0, r, c) + b_value(s, 0, r, c); }, false));
        }
        const uint32_t last = kSrcTiles - 1;
        append(
            golden,
            make_tile([&](uint32_t r, uint32_t c) { return a_value(s, last, r, c) + b_value(s, last, r, c); }, false));
    }

    // Whole buffer in one DRAM bank (the direct reader/writer address bank 0), tiles advanced by the native tile size.
    auto make_dram = [&](uint32_t size) {
        return distributed::MeshBuffer::create(
            distributed::ReplicatedBufferConfig{.size = size},
            {.page_size = size, .buffer_type = BufferType::DRAM},
            &mesh_device);
    };
    auto a_buffer = make_dram(a_all.size() * sizeof(uint32_t));
    auto b_buffer = make_dram(b_all.size() * sizeof(uint32_t));
    auto fp32_buffer = make_dram(fp32_all.size() * sizeof(uint32_t));
    auto out_buffer = make_dram(golden.size() * sizeof(uint32_t));

    experimental::ProgramSpec spec{.name = "quasar_dest_handoff"};
    experimental::WorkUnitSpec wu{.name = "main", .target_nodes = node};

    struct InputSpec {
        experimental::DFBSpecName dfb;
        experimental::KernelSpecName reader;
        const char* accessor;
        tt::DataFormat format;
        uint32_t num_entries;
        uint32_t num_tiles;
        std::shared_ptr<distributed::MeshBuffer> buffer;
        const std::vector<uint32_t>* data;
    };
    InputSpec inputs[] = {
        // non-const: EnqueueWriteMeshBuffer takes the buffer by non-const reference
        {IN_A,
         experimental::KernelSpecName{"reader_a"},
         "in_a",
         tt::DataFormat::Float16_b,
         2 * kSrcTiles,
         kSections * kSrcTiles,
         a_buffer,
         &a_all},
        {IN_B,
         experimental::KernelSpecName{"reader_b"},
         "in_b",
         tt::DataFormat::Float16_b,
         2 * kSrcTiles,
         kSections * kSrcTiles,
         b_buffer,
         &b_all},
        {IN_FP32,
         experimental::KernelSpecName{"reader_fp32"},
         "in_fp32",
         tt::DataFormat::Float32,
         2,
         num_fp32_tiles,
         fp32_buffer,
         &fp32_all},
    };

    experimental::KernelSpec::CompilerOptions::Defines compute_defines;
    if (v.handoff_every_section) {
        compute_defines.emplace("HANDOFF_EVERY_SECTION", "1");
    }
    experimental::KernelSpec compute_spec{
        .unique_id = COMPUTE,
        .source = "tests/tt_metal/tt_metal/test_kernels/compute/unpack_to_dest_handoff_mixed_section.cpp",
        .num_threads = 1,
        .compiler_options = {.defines = compute_defines},
        .compile_time_args = {{"num_sections", kSections}, {"num_src_tiles", kSrcTiles}},
        .hw_config =
            experimental::ComputeHardwareConfig{
                .enable_32_bit_dest = true,
                .double_buffer_dest = !v.dst_full_sync_en,
                .unpack_modes = {{IN_FP32, UnpackMode::UnpackToDest}},
            },
    };
    for (const InputSpec& in : inputs) {
        spec.dataflow_buffers.push_back({
            .unique_id = in.dfb,
            .entry_size = tt::tile_size(in.format),
            .num_entries = in.num_entries,
            .data_format_metadata = in.format,
        });
        spec.kernels.push_back({
            .unique_id = in.reader,
            .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/dram/direct_reader_unary_2_0.cpp",
            .num_threads = 1,
            .dfb_bindings = {experimental::ProducerOf(in.dfb, "out")},
            .runtime_arg_schema = {.runtime_arg_names = {"src_addr", "src_bank_id", "num_tiles", "dram_page_stride"}},
            .hw_config = experimental::DataMovementHardwareConfig{},
        });
        compute_spec.dfb_bindings.push_back(experimental::ConsumerOf(in.dfb, in.accessor));
        wu.kernels.push_back(in.reader);
    }
    spec.dataflow_buffers.push_back({
        .unique_id = OUT,
        .entry_size = fp32_tile_bytes,
        .num_entries = 4,
        .data_format_metadata = tt::DataFormat::Float32,
    });
    spec.kernels.push_back({
        .unique_id = WRITER,
        .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/dram/direct_writer_unary_2_0.cpp",
        .num_threads = 1,
        .dfb_bindings = {experimental::ConsumerOf(OUT, "in")},
        .runtime_arg_schema = {.runtime_arg_names = {"dst_addr", "dst_bank_id", "num_tiles", "dram_page_stride"}},
        .hw_config = experimental::DataMovementHardwareConfig{},
    });
    compute_spec.dfb_bindings.push_back(experimental::ProducerOf(OUT, "out"));
    spec.kernels.push_back(std::move(compute_spec));
    wu.kernels.push_back(WRITER);
    wu.kernels.push_back(COMPUTE);
    spec.work_units = {wu};

    Program program = experimental::MakeProgramFromSpec(mesh_device, spec);

    auto& cq = mesh_device.mesh_command_queue();
    experimental::ProgramRunArgs params;
    for (InputSpec& in : inputs) {
        distributed::EnqueueWriteMeshBuffer(cq, in.buffer, *in.data, /*blocking=*/true);
        params.kernel_run_args.push_back({
            .kernel = in.reader,
            .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
                node,
                {{"src_addr", in.buffer->address()},
                 {"src_bank_id", 0u},
                 {"num_tiles", in.num_tiles},
                 {"dram_page_stride", tt::tile_size(in.format)}}),
        });
    }
    params.kernel_run_args.push_back({
        .kernel = WRITER,
        .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
            node,
            {{"dst_addr", out_buffer->address()},
             {"dst_bank_id", 0u},
             {"num_tiles", 2 * kSections},
             {"dram_page_stride", fp32_tile_bytes}}),
    });
    params.kernel_run_args.push_back({.kernel = COMPUTE});
    experimental::SetProgramRunArgs(program, params);

    LaunchProgram(mesh_device, std::move(program));

    std::vector<uint32_t> result;
    distributed::EnqueueReadMeshBuffer(cq, result, out_buffer, /*blocking=*/true);

    if (result.size() != golden.size()) {
        log_error(tt::LogTest, "Size mismatch: device={} golden={}", result.size(), golden.size());
        return false;
    }
    size_t num_mismatches = 0;
    constexpr size_t max_report = 8;
    for (size_t i = 0; i < result.size(); ++i) {
        if (result[i] != golden[i]) {
            if (num_mismatches < max_report) {
                const size_t tile = i / kTileHW;
                log_error(
                    tt::LogTest,
                    "Mismatch: section {} DST[{}] tiled idx {}: golden={} device={}",
                    tile / 2,
                    tile % 2,
                    i % kTileHW,
                    std::bit_cast<float>(golden[i]),
                    std::bit_cast<float>(result[i]));
            }
            ++num_mismatches;
        }
    }
    if (num_mismatches != 0) {
        log_error(tt::LogTest, "Total mismatches: {}/{}", num_mismatches, result.size());
        return false;
    }
    return true;
}

}  // namespace unit_tests::compute::quasar_dest_handoff

TEST_F(LLKQuasarMeshDeviceSingleCardFixture, QuasarDestHandoffMixedSection) {
    for (const bool dst_full_sync_en : {false, true}) {
        SCOPED_TRACE(dst_full_sync_en ? "dst_full_sync_en=true (SyncFull)" : "dst_full_sync_en=false (SyncHalf)");
        EXPECT_TRUE(unit_tests::compute::quasar_dest_handoff::run(
            *this->devices_.at(0), {.handoff_every_section = false, .dst_full_sync_en = dst_full_sync_en}));
    }
}

TEST_F(LLKQuasarMeshDeviceSingleCardFixture, QuasarDestHandoffWithoutCopy) {
    for (const bool dst_full_sync_en : {false, true}) {
        SCOPED_TRACE(dst_full_sync_en ? "dst_full_sync_en=true (SyncFull)" : "dst_full_sync_en=false (SyncHalf)");
        EXPECT_TRUE(unit_tests::compute::quasar_dest_handoff::run(
            *this->devices_.at(0), {.handoff_every_section = true, .dst_full_sync_en = dst_full_sync_en}));
    }
}

}  // namespace tt::tt_metal
