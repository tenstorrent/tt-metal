// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Packer state across a mid-kernel dest-acc override.
//
// TensixPackReconfigInsideFp32DestAccOverride: the kernel switches Dest to the mode opposite to the one
// it was built with (set_fp32_dest_acc<!DST_ACCUM_MODE>()), copies a tile into Dest, reconfigures the
// packer from the Float16_b output configured at startup to the Bfp8_b output, and packs. The reconfig
// must leave the packer's Dest read width at what the override programmed; re-deriving it from the
// kernel-wide flag makes the packer read 32-bit rows out of a 16-bit Dest (fp32 build) or 16-bit halves
// of 32-bit rows (16-bit build) and the tile comes out scrambled. Both builds are run. Stimuli are whole
// numbers in [0, 127], which Float16_b and Bfp8_b both represent exactly, so the decoded output is
// compared with the input datum for datum.
//
// TensixPackFloat32AfterFp32DestAccDisable: an fp32 build with a Float32 output copies a tile into the
// 32-bit Dest, disables dest-acc and packs. The datums are still 32-bit, so the packer must keep the 32-bit
// Dest read for the Float32 source, as configure_pack programs it; clearing it with the mode makes the
// packer read 16-bit halves and every datum comes out wrong.
//
// TensixPackCfgAfterFp32DestAccRestore: the kernel reconfigures the packer to a Bfp8 (5-bit exponent)
// output while Dest is in 16-bit mode, then restores the 32-bit Dest with restore_fp32_dest_acc<false>().
// An fp32 -> Bfp8 pack needs the packer's exponent threshold, which zeroes datums below the 5-bit
// exponent range, so the restore must bring it back for the configured output. The pack thread reads
// the mode-dependent packer fields back after the restore and the test checks them. The fields are
// checked rather than packed data because packed data cannot tell the threshold apart on Blackhole: a
// plain fp32_dest_acc_en Float32 -> Bfp8 datacopy with no toggle at all returns an all-zero tile there.

#include <gtest/gtest.h>
#include <bit>
#include <cstdint>
#include <random>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/bfloat8.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt_stl/span.hpp>
#include <tt-logger/tt-logger.hpp>
#include "impl/program/program_impl.hpp"
#include "llk_device_fixture.hpp"

namespace tt::tt_metal {

using std::vector;

namespace unit_tests::llk::pack_reconfig_fp32_dest_acc_toggle {

constexpr std::uint32_t kNumTiles = 1;
constexpr std::uint32_t kDatumsPerTile = tt::constants::TILE_HW;
// Bfp8_b and Bfp8 keep whole numbers up to this limit exact whatever the block's shared exponent is.
constexpr std::uint32_t kWholeNumberLimit = 128;

static constexpr const char* kReconfigKernel =
    "tests/tt_metal/tt_metal/test_kernels/compute/pack_reconfig_fp32_dest_acc_toggle.cpp";
static constexpr const char* kFloat32AfterDisableKernel =
    "tests/tt_metal/tt_metal/test_kernels/compute/pack_float32_after_fp32_dest_acc_disable.cpp";
static constexpr const char* kCfgDumpKernel =
    "tests/tt_metal/tt_metal/test_kernels/compute/pack_cfg_dump_fp32_dest_acc_restore.cpp";
// Exponent floor the packer applies to an 8-bit-exponent datum packed to a 5-bit-exponent format: the
// smallest EXP_B a 5-bit exponent can hold (EXP_MIN_A = -14 -> 127 - 14).
constexpr std::uint32_t kBfpAExpThreshold = 113;

// c_0 (in_fmt) -> compute (built with fp32_dest_acc_en) -> c_16 (out_fmt). c_17 (startup_fmt) only
// configures the packer at startup.
static vector<std::uint32_t> run_fp32_dest_acc_kernel(
    distributed::MeshDevice& mesh_device,
    const char* compute_kernel,
    bool fp32_dest_acc_en,
    tt::DataFormat in_fmt,
    tt::DataFormat startup_fmt,
    tt::DataFormat out_fmt,
    const vector<std::uint32_t>& src_vec) {
    Program program = CreateProgram();
    CoreCoord core = {0, 0};

    const std::uint32_t in_tile_size = tt::tile_size(in_fmt);
    const std::uint32_t startup_tile_size = tt::tile_size(startup_fmt);
    const std::uint32_t out_tile_size = tt::tile_size(out_fmt);

    auto src_buffer = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = kNumTiles * in_tile_size},
        {.page_size = kNumTiles * in_tile_size, .buffer_type = BufferType::DRAM},
        &mesh_device);
    auto dst_buffer = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = kNumTiles * out_tile_size},
        {.page_size = kNumTiles * out_tile_size, .buffer_type = BufferType::DRAM},
        &mesh_device);

    CreateCircularBuffer(
        program,
        core,
        CircularBufferConfig(in_tile_size, {{tt::CBIndex::c_0, in_fmt}}).set_page_size(tt::CBIndex::c_0, in_tile_size));
    CreateCircularBuffer(
        program,
        core,
        CircularBufferConfig(out_tile_size, {{tt::CBIndex::c_16, out_fmt}})
            .set_page_size(tt::CBIndex::c_16, out_tile_size));
    CreateCircularBuffer(
        program,
        core,
        CircularBufferConfig(startup_tile_size, {{tt::CBIndex::c_17, startup_fmt}})
            .set_page_size(tt::CBIndex::c_17, startup_tile_size));

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
        compute_kernel,
        core,
        ComputeConfig{.fp32_dest_acc_en = fp32_dest_acc_en, .compile_args = {kNumTiles}});

    auto& cq = mesh_device.mesh_command_queue();
    distributed::EnqueueWriteMeshBuffer(cq, src_buffer, src_vec, /*blocking=*/true);
    SetRuntimeArgs(program, reader, core, {src_buffer->address(), 0, kNumTiles});
    SetRuntimeArgs(program, writer, core, {dst_buffer->address(), 0, kNumTiles});

    LaunchProgram(mesh_device, std::move(program));

    vector<std::uint32_t> result_vec;
    distributed::EnqueueReadMeshBuffer(cq, result_vec, dst_buffer, /*blocking=*/true);
    return result_vec;
}

}  // namespace unit_tests::llk::pack_reconfig_fp32_dest_acc_toggle

TEST_F(LLKMeshDeviceFixture, TensixPackReconfigInsideFp32DestAccOverride) {
    using namespace unit_tests::llk::pack_reconfig_fp32_dest_acc_toggle;
    if (this->arch_ == ARCH::QUASAR) {
        GTEST_SKIP() << "set_fp32_dest_acc is not supported on Quasar";
    }

    std::mt19937 rng(0x5EED);
    vector<bfloat16> input(kNumTiles * kDatumsPerTile);
    for (auto& datum : input) {
        datum = bfloat16(static_cast<float>(rng() % kWholeNumberLimit));
    }
    const vector<std::uint32_t> src_vec = pack_bfloat16_vec_into_uint32_vec(input);

    // fp32 build with a 16-bit section (the <false> reconfig arm), then a 16-bit build with a 32-bit
    // section (the <true> arm).
    for (bool fp32_dest_acc_en : {true, false}) {
        for (auto& device : this->devices_) {
            const vector<std::uint32_t> result = run_fp32_dest_acc_kernel(
                *device,
                kReconfigKernel,
                fp32_dest_acc_en,
                tt::DataFormat::Float16_b,
                tt::DataFormat::Float16_b,
                tt::DataFormat::Bfp8_b,
                src_vec);
            const vector<float> output = unpack_bfp8_tiles_into_float_vec(
                ttsl::make_const_span(result), /*row_major_output=*/false, /*is_exp_a=*/false);
            ASSERT_EQ(output.size(), input.size());

            std::uint32_t mismatches = 0;
            for (std::size_t i = 0; i < input.size(); ++i) {
                const float expected = static_cast<float>(input[i]);
                if (output[i] != expected) {
                    if (mismatches < 8) {
                        log_error(tt::LogTest, "datum {}: expected {} got {}", i, expected, output[i]);
                    }
                    ++mismatches;
                }
            }
            EXPECT_EQ(mismatches, 0u) << "fp32_dest_acc_en=" << fp32_dest_acc_en
                                      << ": Bfp8_b pack after pack_reconfig_data_format inside "
                                         "set_fp32_dest_acc<!DST_ACCUM_MODE>() differs from the input in "
                                      << mismatches << " of " << input.size() << " datums";
        }
    }
}

TEST_F(LLKMeshDeviceFixture, TensixPackCfgAfterFp32DestAccRestore) {
    using namespace unit_tests::llk::pack_reconfig_fp32_dest_acc_toggle;
    if (this->arch_ == ARCH::QUASAR) {
        GTEST_SKIP() << "set_fp32_dest_acc is not supported on Quasar";
    }

    // One Float32 input tile the kernel never consumes; the output page receives the dumped fields.
    const vector<std::uint32_t> src_vec(kNumTiles * kDatumsPerTile, 0u);
    for (auto& device : this->devices_) {
        const vector<std::uint32_t> words = run_fp32_dest_acc_kernel(
            *device,
            kCfgDumpKernel,
            /*fp32_dest_acc_en=*/true,
            tt::DataFormat::Float32,
            tt::DataFormat::Float16,
            tt::DataFormat::Bfp8,
            src_vec);
        ASSERT_GE(words.size(), 4u);
        log_info(
            tt::LogTest,
            "after restore_fp32_dest_acc<false>(): Exp_threshold_en={} Exp_threshold={} Read_32b_data={} "
            "Round_10b_mant={}",
            words[0],
            words[1],
            words[2],
            words[3]);
        EXPECT_EQ(words[0], 1u)
            << "exponent threshold not re-enabled for the Bfp8 output after the 32-bit Dest was restored";
        EXPECT_EQ(words[1], kBfpAExpThreshold) << "exponent threshold value after the 32-bit Dest was restored";
        EXPECT_EQ(words[2], 1u) << "Read_32b_data after the 32-bit Dest was restored";
        EXPECT_EQ(words[3], 0u) << "Round_10b_mant for a Bfp8 pack source";
    }
}

TEST_F(LLKMeshDeviceFixture, TensixPackFloat32AfterFp32DestAccDisable) {
    using namespace unit_tests::llk::pack_reconfig_fp32_dest_acc_toggle;
    if (this->arch_ == ARCH::QUASAR) {
        GTEST_SKIP() << "set_fp32_dest_acc is not supported on Quasar";
    }

    // bf16 values with fractional parts, exact in Float32: the packed words must equal them bit for bit.
    std::mt19937 rng(0xF32);
    std::uniform_real_distribution<float> dist(-100.0f, 100.0f);
    vector<bfloat16> input(kNumTiles * kDatumsPerTile);
    for (auto& datum : input) {
        datum = bfloat16(dist(rng));
    }
    const vector<std::uint32_t> src_vec = pack_bfloat16_vec_into_uint32_vec(input);

    for (auto& device : this->devices_) {
        const vector<std::uint32_t> result = run_fp32_dest_acc_kernel(
            *device,
            kFloat32AfterDisableKernel,
            /*fp32_dest_acc_en=*/true,
            tt::DataFormat::Float16_b,
            tt::DataFormat::Float32,
            tt::DataFormat::Float32,
            src_vec);
        ASSERT_GE(result.size(), input.size());

        std::uint32_t mismatches = 0;
        for (std::size_t i = 0; i < input.size(); ++i) {
            const float expected = static_cast<float>(input[i]);
            const float got = std::bit_cast<float>(result[i]);
            if (got != expected) {
                if (mismatches < 8) {
                    log_error(tt::LogTest, "datum {}: expected {} got {}", i, expected, got);
                }
                ++mismatches;
            }
        }
        EXPECT_EQ(mismatches, 0u)
            << "Float32 pack after set_fp32_dest_acc<false>() of a tile written in the 32-bit Dest "
               "differs from the input in "
            << mismatches << " of " << input.size() << " datums";
    }
}

}  // namespace tt::tt_metal
