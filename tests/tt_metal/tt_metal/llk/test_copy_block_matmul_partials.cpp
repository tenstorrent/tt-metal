// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/base.h>
#include <gtest/gtest.h>
#include <cstdint>
#include <sys/types.h>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <variant>
#include <vector>

#include <bit>
#include <chrono>
#include <random>
#include <stdexcept>
#include <thread>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/bfloat8.hpp>
#include <tt-metalium/tile.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include "tt_metal/test_utils/comparison.hpp"
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/kernel_types.hpp>
#include "impl/context/metal_context.hpp"
#include "impl/debug/watcher_server.hpp"
#include "llk_device_fixture.hpp"
#include "mesh_dispatch_fixture.hpp"
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/program.hpp>
#include <tt_stl/span.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "tt_metal/test_utils/stimulus.hpp"
#include <tt-umd/types/arch.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include "single_core_compute_runners.hpp"

namespace tt::tt_metal {

using std::vector;
using namespace tt;
using namespace tt::test_utils;

namespace unit_tests::compute::matmul_partials {

struct CopyBlockMatmulPartialsConfig {
    std::uint32_t single_tile_size = 2 * 32 * 32;
    std::uint32_t num_tiles = 1;
    // *_ublock defines no. of tiles finished with single LLK API call:
    std::uint32_t reader_ublock = 1;
    std::uint32_t writer_ublock = 1;
    std::uint32_t compute_ublock = 1;
    std::uint32_t src0_cb_index = 0;
    std::uint32_t ouput_cb_index = 16;
    // Whether or not we want the result to be stored in DST in FP32:
    bool fp32_dest_acc_en = false;
    // Whether or not to sync full/half DST between MATH and PACK:
    bool dst_full_sync_en = false;
    // Compute kernel exercised by the test. Defaults to the kernel that loops copy_block per tile with
    // a single pack_block; the single-copy_block variant points this at eltwise_copy_pack_block.cpp.
    std::string compute_kernel = "tests/tt_metal/tt_metal/test_kernels/compute/eltwise_copy_block_matmul_partials.cpp";
};

static std::vector<std::uint32_t> generate_copy_block_stimulus(
    std::uint32_t dram_buffer_size, const CopyBlockMatmulPartialsConfig& test_config) {
    std::vector<std::uint32_t> src_vec = create_random_vector_of_bfloat16(dram_buffer_size, 100, 0);

    if (test_config.fp32_dest_acc_en) {
        auto src_vec_float = generate_uniform_random_vector<float>(-100, 100, dram_buffer_size / sizeof(float), 0);
        for (auto i = 0; i < src_vec.size(); i++) {
            std::memcpy(&src_vec[i], &src_vec_float[i], sizeof(float));
            src_vec[i] &= 0xFFFFE000;
        }
    }
    return src_vec;
}

void run_single_core_copy_block_matmul_partials(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device, const CopyBlockMatmulPartialsConfig& test_config) {
    auto& cq = mesh_device->mesh_command_queue();
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    const experimental::NodeCoord node{0, 0};

    std::uint32_t single_tile_size = test_config.single_tile_size;
    std::uint32_t num_tiles = test_config.num_tiles;
    std::uint32_t dram_buffer_size = single_tile_size * num_tiles;

    tt::DataFormat data_format = test_config.fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;

    distributed::DeviceLocalBufferConfig dram_local_config{
        .page_size = dram_buffer_size, .buffer_type = tt_metal::BufferType::DRAM, .bottom_up = false};
    distributed::ReplicatedBufferConfig dram_buffer_config{.size = dram_buffer_size};
    auto src_dram_buffer = distributed::MeshBuffer::create(dram_buffer_config, dram_local_config, mesh_device.get());
    auto dst_dram_buffer = distributed::MeshBuffer::create(dram_buffer_config, dram_local_config, mesh_device.get());

    std::uint32_t num_input_tiles = test_config.reader_ublock;
    std::uint32_t num_output_tiles = test_config.writer_ublock;

    const experimental::DFBSpecName SRC0_DFB{"src0_dfb"};
    const experimental::DFBSpecName DST_DFB{"dst_dfb"};
    const experimental::KernelSpecName READER{"reader"};
    const experimental::KernelSpecName WRITER{"writer"};
    const experimental::KernelSpecName COMPUTE{"compute"};

    experimental::DataflowBufferSpec src0_dfb_spec{
        .unique_id = SRC0_DFB,
        .entry_size = single_tile_size,
        .num_entries = num_input_tiles,
        .data_format_metadata = data_format,
    };
    experimental::DataflowBufferSpec dst_dfb_spec{
        .unique_id = DST_DFB,
        .entry_size = single_tile_size,
        .num_entries = num_output_tiles,
        .data_format_metadata = data_format,
    };

    experimental::DataMovementHardwareConfig reader_hw_config;
    if (mesh_device->arch() == tt::ARCH::QUASAR) {
        reader_hw_config = experimental::DataMovementHardwareConfig{
            .config_2xx =
                experimental::DataMovementHardwareConfig::DataMovement2XXConfig{
                    .disable_dfb_implicit_sync_for_all = true,
                },
        };
    } else {
        reader_hw_config = experimental::DataMovementHardwareConfig{
            .config_1xx =
                experimental::DataMovementHardwareConfig::DataMovement1XXConfig{
                    .processor = tt_metal::DataMovementProcessor::RISCV_1,
                    .noc = tt_metal::NOC::RISCV_1_default,
                },
        };
    }
    experimental::KernelSpec reader_spec{
        .unique_id = READER,
        .source =

            "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_unary_push_n_2_0.cpp",
        .num_threads = 1,
        .dfb_bindings = {experimental::ProducerOf(SRC0_DFB, "out")},
        .runtime_arg_schema =
            {.runtime_arg_names = {"src_addr", "src_dram_bank_id", "num_tiles", "ublock_size_tiles", "reader_only"}},
        .hw_config = reader_hw_config,
    };

    experimental::DataMovementHardwareConfig writer_hw_config;
    if (mesh_device->arch() == tt::ARCH::QUASAR) {
        writer_hw_config = experimental::DataMovementHardwareConfig{
            .config_2xx =
                experimental::DataMovementHardwareConfig::DataMovement2XXConfig{
                    .disable_dfb_implicit_sync_for_all = true,
                },
        };
    } else {
        writer_hw_config = experimental::DataMovementHardwareConfig{
            .config_1xx =
                experimental::DataMovementHardwareConfig::DataMovement1XXConfig{
                    .processor = tt_metal::DataMovementProcessor::RISCV_0,
                    .noc = tt_metal::NOC::RISCV_0_default,
                },
        };
    }
    experimental::KernelSpec writer_spec{
        .unique_id = WRITER,
        .source =

            "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary_pop_n.cpp",
        .num_threads = 1,
        .dfb_bindings = {experimental::ConsumerOf(DST_DFB, "in")},
        .runtime_arg_schema =
            {.runtime_arg_names = {"dst_addr", "dst_dram_bank_id", "num_tiles", "ublock_size_tiles", "writer_only"}},
        .hw_config = writer_hw_config,
    };

    experimental::KernelSpec::CompilerOptions::Defines compute_defines;

    experimental::ComputeHardwareConfig compute_hw_config;
    {
        // When fp32_dest_acc_en is true the src DFB is Float32 and the compute kernel
        // consumes it, so the Metal 2.0 host API requires an explicit unpack_modes entry.
        // UnpackToSrc is unpack via SrcA/B, ~19-bit precision.
        experimental::ComputeHardwareConfig::ComputeUnpackModes unpack_modes{};
        if (test_config.fp32_dest_acc_en) {
            unpack_modes = {{SRC0_DFB, tt::tt_metal::UnpackMode::UnpackToSrc}};
        }
        compute_hw_config = experimental::ComputeHardwareConfig{
            .enable_32_bit_dest = test_config.fp32_dest_acc_en,
            .double_buffer_dest = !test_config.dst_full_sync_en,
            .unpack_modes = unpack_modes,
        };
    }
    experimental::KernelSpec compute_spec{
        .unique_id = COMPUTE,
        .source = test_config.compute_kernel,
        .num_threads = 1,
        .compiler_options = {.defines = compute_defines},
        .dfb_bindings =
            {{
                 .dfb_spec_name = SRC0_DFB,
                 .accessor_name = "in",
                 .endpoint_type = experimental::DFBEndpointType::CONSUMER,
                 .access_pattern = experimental::DFBAccessPattern::STRIDED,
             },
             {
                 .dfb_spec_name = DST_DFB,
                 .accessor_name = "out",
                 .endpoint_type = experimental::DFBEndpointType::PRODUCER,
                 .access_pattern = experimental::DFBAccessPattern::STRIDED,
             }},
        .compile_time_args = {{"num_tiles", num_tiles}, {"num_single_transfer", test_config.compute_ublock}},
        .hw_config = compute_hw_config,
    };

    experimental::WorkUnitSpec wu{
        .name = "main",
        .kernels = {READER, WRITER, COMPUTE},
        .target_nodes = node,
    };

    experimental::ProgramSpec spec{
        .name = "single_core_copy_block_matmul_partials",
        .kernels = {reader_spec, writer_spec, compute_spec},
        .dataflow_buffers = {src0_dfb_spec, dst_dfb_spec},
        .work_units = {wu},
    };

    Program program = experimental::MakeProgramFromSpec(*mesh_device, spec);

    distributed::MeshWorkload workload;
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    workload.add_program(device_range, std::move(program));
    auto& program_run = workload.get_programs().at(device_range);

    std::vector<std::uint32_t> src_vec = generate_copy_block_stimulus(dram_buffer_size, test_config);
    distributed::WriteShard(cq, src_dram_buffer, src_vec, zero_coord);

    experimental::ProgramRunArgs params;
    params.kernel_run_args = {
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = READER,
            .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
                node,
                {{"src_addr", src_dram_buffer->address()},
                 {"src_dram_bank_id", 0u},
                 {"num_tiles", num_tiles},
                 {"ublock_size_tiles", test_config.reader_ublock},
                 {"reader_only", 0u}}),
        },
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = WRITER,
            .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
                node,
                {{"dst_addr", dst_dram_buffer->address()},
                 {"dst_dram_bank_id", 0u},
                 {"num_tiles", num_tiles},
                 {"ublock_size_tiles", test_config.writer_ublock},
                 {"writer_only", 0u}}),
        },
        experimental::ProgramRunArgs::KernelRunArgs{.kernel = COMPUTE},
    };
    experimental::SetProgramRunArgs(program_run, params);

    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);

    std::vector<std::uint32_t> result_vec;
    distributed::ReadShard(cq, result_vec, dst_dram_buffer, zero_coord);

    EXPECT_EQ(src_vec.size(), result_vec.size());
    EXPECT_EQ(src_vec, result_vec);
}

// Compute kernel exercising the uniform copy_block + pack_block surface (shared by the
// TensixComputeCopyPackBlock* cases below); the default kernel is the struct member initializer.
constexpr const char* kBlockKernel = "tests/tt_metal/tt_metal/test_kernels/compute/eltwise_copy_pack_block.cpp";

}  // namespace unit_tests::compute::matmul_partials

////////////////////////////////////////////////////////////////////////////
//                             Test Description
// ------------------------------------------------------------------------
// These tests aim to cover usage of these API calls:
// - copy_block
// - pack_block
////////////////////////////////////////////////////////////////////////////

TEST_F(LLKMeshDeviceFixture, DISABLED_TensixComputeCopyBlockSingle) {
    for (bool fp32_dest_acc_en : {true, false}) {
        for (bool dst_full_sync_en : {true, false}) {
            log_info(LogTest, "FP32DestAcc = {}, DstSyncFull = {}", fp32_dest_acc_en, dst_full_sync_en);
            unit_tests::compute::matmul_partials::CopyBlockMatmulPartialsConfig test_config = {
                .num_tiles = 8, .fp32_dest_acc_en = fp32_dest_acc_en, .dst_full_sync_en = dst_full_sync_en};
            unit_tests::compute::matmul_partials::run_single_core_copy_block_matmul_partials(
                this->devices_.at(0), test_config);
        }
    }
}
TEST_F(LLKMeshDeviceFixture, TensixComputeCopyBlockMultiple) {
    for (bool fp32_dest_acc_en : {true, false}) {
        for (bool dst_full_sync_en : {true, false}) {
            log_info(LogTest, "FP32DestAcc = {}, DstSyncFull = {}", fp32_dest_acc_en, dst_full_sync_en);
            unit_tests::compute::matmul_partials::CopyBlockMatmulPartialsConfig test_config = {
                .num_tiles = 8,
                .reader_ublock = 8,
                .writer_ublock = 8,
                .compute_ublock = 4,  // compute_ublock must be <= get_dest_max_tiles (4 for SyncHalf+FP32)
                .fp32_dest_acc_en = fp32_dest_acc_en,
                .dst_full_sync_en = dst_full_sync_en};
            unit_tests::compute::matmul_partials::run_single_core_copy_block_matmul_partials(
                this->devices_.at(0), test_config);
            if (MetalContext::instance().get_cluster().arch() == ARCH::QUASAR) {
                return;
            }
        }
    }
}

TEST_F(LLKMeshDeviceFixture, TensixComputeCopyBlockComputeBottleneck) {
    for (bool fp32_dest_acc_en : {true, false}) {
        for (bool dst_full_sync_en : {true, false}) {
            log_info(LogTest, "FP32DestAcc = {}, DstSyncFull = {}", fp32_dest_acc_en, dst_full_sync_en);
            unit_tests::compute::matmul_partials::CopyBlockMatmulPartialsConfig test_config = {
                .num_tiles = 8,
                .reader_ublock = 8,
                .writer_ublock = 8,
                .compute_ublock = 1,
                .fp32_dest_acc_en = fp32_dest_acc_en,
                .dst_full_sync_en = dst_full_sync_en};
            unit_tests::compute::matmul_partials::run_single_core_copy_block_matmul_partials(
                this->devices_.at(0), test_config);
            if (MetalContext::instance().get_cluster().arch() == ARCH::QUASAR) {
                return;
            }
        }
    }
}

// Same coverage as TensixComputeCopyBlockMultiple, but issues one copy_block call per ublock instead
// of a loop of single-tile copy_block calls (pack_block is one call per ublock in both). The golden is an identity
// copy, so results must match bit-for-bit.
TEST_F(LLKMeshDeviceFixture, TensixComputeCopyPackBlockMultiple) {
    for (bool fp32_dest_acc_en : {true, false}) {
        for (bool dst_full_sync_en : {true, false}) {
            log_info(LogTest, "FP32DestAcc = {}, DstSyncFull = {}", fp32_dest_acc_en, dst_full_sync_en);
            unit_tests::compute::matmul_partials::CopyBlockMatmulPartialsConfig test_config = {
                .num_tiles = 8,
                .reader_ublock = 8,
                .writer_ublock = 8,
                .compute_ublock = 4,  // compute_ublock must be <= get_dest_max_tiles (4 for SyncHalf+FP32)
                .fp32_dest_acc_en = fp32_dest_acc_en,
                .dst_full_sync_en = dst_full_sync_en,
                .compute_kernel = unit_tests::compute::matmul_partials::kBlockKernel};
            unit_tests::compute::matmul_partials::run_single_core_copy_block_matmul_partials(
                this->devices_.at(0), test_config);
            if (MetalContext::instance().get_cluster().arch() == ARCH::QUASAR) {
                return;
            }
        }
    }
}

// copy_block + pack_block with a single tile per block (compute bottleneck shape).
TEST_F(LLKMeshDeviceFixture, TensixComputeCopyPackBlockComputeBottleneck) {
    for (bool fp32_dest_acc_en : {true, false}) {
        for (bool dst_full_sync_en : {true, false}) {
            log_info(LogTest, "FP32DestAcc = {}, DstSyncFull = {}", fp32_dest_acc_en, dst_full_sync_en);
            unit_tests::compute::matmul_partials::CopyBlockMatmulPartialsConfig test_config = {
                .num_tiles = 8,
                .reader_ublock = 8,
                .writer_ublock = 8,
                .compute_ublock = 1,
                .fp32_dest_acc_en = fp32_dest_acc_en,
                .dst_full_sync_en = dst_full_sync_en,
                .compute_kernel = unit_tests::compute::matmul_partials::kBlockKernel};
            unit_tests::compute::matmul_partials::run_single_core_copy_block_matmul_partials(
                this->devices_.at(0), test_config);
            if (MetalContext::instance().get_cluster().arch() == ARCH::QUASAR) {
                return;
            }
        }
    }
}

// ============================================================================
// Id-free (2.0) copy_block / copy_init / pack_block: each is a pure Float16_b -> Float16_b
// datacopy, so the host golden is the identity (device output == input, bit-for-bit). Validated against the
// input, not a legacy kernel. Runs on Blackhole (the id-free API is BH-only).
// ============================================================================
TEST_F(LLKBlackholeSingleCardFixture, TensixCopyBlockIdFreeIdentity) {
    constexpr std::uint32_t num_tiles = 64;  // multiple of the 4-tile block
    auto src_vec = create_random_vector_of_bfloat16(
        tt::tile_size(tt::DataFormat::Float16_b) * num_tiles, /*rand_max_float=*/20, /*seed=*/42, /*offset=*/-10.0f);
    auto result = unit_tests::llk::single_core::run_unary(
        *this->devices_.at(0),
        tt::DataFormat::Float16_b,
        tt::DataFormat::Float16_b,
        src_vec,
        num_tiles,
        /*fp32_dest_acc_en=*/false,
        "tests/tt_metal/tt_metal/test_kernels/compute/copy_block_2_0.cpp",
        /*cb_depth_tiles=*/num_tiles);
    EXPECT_EQ(src_vec, result);
}

TEST_F(LLKBlackholeSingleCardFixture, TensixPackBlockIdFreeIdentity) {
    constexpr std::uint32_t num_tiles = 64;  // multiple of the 4-tile block
    auto src_vec = create_random_vector_of_bfloat16(
        tt::tile_size(tt::DataFormat::Float16_b) * num_tiles, /*rand_max_float=*/20, /*seed=*/42, /*offset=*/-10.0f);
    auto result = unit_tests::llk::single_core::run_unary(
        *this->devices_.at(0),
        tt::DataFormat::Float16_b,
        tt::DataFormat::Float16_b,
        src_vec,
        num_tiles,
        /*fp32_dest_acc_en=*/false,
        "tests/tt_metal/tt_metal/test_kernels/compute/pack_block_2_0.cpp",
        /*cb_depth_tiles=*/num_tiles);
    EXPECT_EQ(src_vec, result);
}

// Id-free (2.0) copy_block over a MULTI-TILE window of block-float (Bfp8_b) tiles. This guards the block-float
// branch of tile_stride_words (internal/llk_descriptor.h): copy_block reads the 4-tile input window at
// in.l1_address + c * tile_stride_words(Bfp8_b, 32x32) and packs each DST slot back out. A Bfp8_b tile carries a
// shared-exponent section that a plain datum-count size omits, so if the per-tile stride is wrong, tiles c=1..3
// are read from the wrong L1 offset and the decoded output diverges from the input. Bfp8_b in AND out => identity
// copy (golden = the input, is_close within Bfp8 requant). NOTE: only FULL 32x32 block-float tiles are exercised
// here -- the BH compute datapath does not support partial (sub-32-row) block-float tiles, so the partial-BFP
// stride (correct in tile_stride_words, matching tt_metal Tile::get_tile_size) cannot be validated on device.
TEST_F(LLKBlackholeSingleCardFixture, TensixCopyBlockBfp8IdFree) {
    constexpr std::uint32_t num_tiles = 4;  // exactly one copy_block window
    const tt::tt_metal::Tile tile({tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH});
    const std::uint32_t datums_per_tile = tt::constants::TILE_HEIGHT * tt::constants::TILE_WIDTH;

    std::vector<float> src_floats(num_tiles * datums_per_tile);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    for (float& v : src_floats) {
        v = dist(gen);
    }
    auto src_bfp8 =
        pack_as_bfp8_tiles(ttsl::make_const_span(src_floats), /*row_major_input=*/true, /*is_exp_a=*/false, tile);

    auto result = unit_tests::llk::single_core::run_unary_tiled(
        *this->devices_.at(0),
        tt::DataFormat::Bfp8_b,
        tt::DataFormat::Bfp8_b,
        tile,
        src_bfp8,
        num_tiles,
        /*fp32_dest_acc_en=*/false,
        "tests/tt_metal/tt_metal/test_kernels/compute/copy_block_2_0.cpp",
        /*cb_depth_tiles=*/num_tiles);

    auto result_floats = unpack_bfp8_tiles_into_float_vec(
        ttsl::make_const_span(result), /*row_major_output=*/true, /*is_exp_a=*/false, tile);

    ASSERT_EQ(result_floats.size(), src_floats.size());
    EXPECT_TRUE(tt::test_utils::is_close_vectors<float>(src_floats, result_floats, [](float a, float b) {
        return tt::test_utils::is_close(a, b, /*rtol=*/0.3f, /*atol=*/0.3f);
    }));
    EXPECT_TRUE(tt::test_utils::check_pcc(src_floats, result_floats, /*min_pcc=*/0.9999));
}

namespace {

// Packer DEST capacity for 32x32 tiles under fp32 dest-acc + SyncHalf with a Float32 output: half of DEST at a
// 32-bit datum (the packer W-stride is 4096 B; a 16-bit output would keep a 2048 B stride and a bound of 8).
constexpr std::uint32_t kFp32HalfSyncDestTiles = 4;
// Tile count packed from that window to overrun it: twice the capacity, so the first out-of-window tile index
// (kFp32HalfSyncDestTiles) is reached.
constexpr std::uint32_t kOverrunTiles = 2 * kFp32HalfSyncDestTiles;

// One DEST window: `copy_tiles` Float16_b tiles copied into DST and `pack_tiles` packed back out as Float32
// through the id-free pack_block (id_free) or the legacy CB-id pack_tile loop, under fp32 dest-acc + SyncHalf.
// Returns the packed output; the caller decides whether the program may complete (see run_pack_block_dest_overrun).
std::vector<std::uint32_t> run_pack_block_dest_capacity(
    distributed::MeshDevice& mesh,
    const std::vector<std::uint32_t>& src,
    std::uint32_t copy_tiles,
    std::uint32_t pack_tiles,
    bool id_free,
    bool& completed) {
    const auto in_format = tt::DataFormat::Float16_b;
    const auto out_format = tt::DataFormat::Float32;
    const std::uint32_t in_tile_bytes = tt::tile_size(in_format);
    const std::uint32_t out_tile_bytes = tt::tile_size(out_format);
    const CoreCoord core{0, 0};
    Program program = CreateProgram();
    auto buffer = [&](std::uint32_t bytes) {
        return distributed::MeshBuffer::create(
            distributed::ReplicatedBufferConfig{.size = bytes},
            {.page_size = bytes, .buffer_type = BufferType::DRAM},
            &mesh);
    };
    auto src_buffer = buffer(copy_tiles * in_tile_bytes);
    auto dst_buffer = buffer(pack_tiles * out_tile_bytes);
    auto make_cb = [&](tt::CBIndex index, tt::DataFormat format, std::uint32_t tile_bytes, std::uint32_t depth) {
        CreateCircularBuffer(
            program,
            core,
            CircularBufferConfig(depth * tile_bytes, {{index, format}}).set_page_size(index, tile_bytes));
    };
    make_cb(tt::CBIndex::c_0, in_format, in_tile_bytes, copy_tiles);
    make_cb(tt::CBIndex::c_16, out_format, out_tile_bytes, pack_tiles);
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
        "tests/tt_metal/tt_metal/test_kernels/compute/pack_block_dest_capacity_2_0.cpp",
        core,
        ComputeConfig{
            .fp32_dest_acc_en = true,
            .dst_full_sync_en = false,
            .compile_args = {copy_tiles, pack_tiles, static_cast<std::uint32_t>(id_free)}});

    auto& cq = mesh.mesh_command_queue();
    distributed::EnqueueWriteMeshBuffer(cq, src_buffer, src, true /*blocking*/);
    SetRuntimeArgs(program, reader, core, {src_buffer->address(), 0 /*bank_id*/, copy_tiles});
    SetRuntimeArgs(program, writer, core, {dst_buffer->address(), 0 /*bank_id*/, pack_tiles});
    distributed::MeshWorkload workload;
    const distributed::MeshCoordinate zero(0, 0);
    workload.add_program(distributed::MeshCoordinateRange(zero, zero), std::move(program));
    std::vector<std::uint32_t> result;
    // The wait aborts once the watcher reports an assert; a program that completes (no guard) is read back.
    try {
        distributed::EnqueueMeshWorkload(cq, workload, false /*blocking*/);
        distributed::Finish(cq);
        completed = true;
        distributed::EnqueueReadMeshBuffer(cq, result, dst_buffer, true /*blocking*/);
    } catch (const std::runtime_error& e) {
        completed = false;
        EXPECT_NE(std::string(e.what()).find("Aborting wait due to watcher error"), std::string::npos) << e.what();
    }
    return result;
}

std::string wait_for_pack_watcher_exception() {
    constexpr auto timeout = std::chrono::milliseconds(5000);
    constexpr auto poll_interval = std::chrono::milliseconds(100);
    const auto start = std::chrono::steady_clock::now();
    std::string exception;
    while (std::chrono::steady_clock::now() - start < timeout) {
        exception = MetalContext::instance().watcher_server()->exception_message();
        if (!exception.empty()) {
            break;
        }
        std::this_thread::sleep_for(poll_interval);
    }
    return exception;
}

// A pack of exactly the capacity must complete and round-trip the input (bf16 -> fp32 DEST -> fp32 is exact); a
// pack of twice the capacity must trip the packer's DST-capacity LLK assert on TRISC2 (the legacy CB-id pack has
// it; the id-free pack must keep it).
void run_pack_block_dest_overrun(distributed::MeshDevice& mesh, bool id_free) {
    const std::string path = id_free ? "experimental::pack_block" : "pack_tile";
    auto src = create_random_vector_of_bfloat16(
        tt::tile_size(tt::DataFormat::Float16_b) * kFp32HalfSyncDestTiles,
        20 /*rand_max_float*/,
        42 /*seed*/,
        -10.0f /*offset*/);
    std::vector<std::uint32_t> golden;
    for (const auto& value : unpack_uint32_vec_into_bfloat16_vec(src)) {
        golden.push_back(std::bit_cast<std::uint32_t>(static_cast<float>(value)));
    }

    bool completed = false;
    auto result =
        run_pack_block_dest_capacity(mesh, src, kFp32HalfSyncDestTiles, kFp32HalfSyncDestTiles, id_free, completed);
    ASSERT_TRUE(completed) << path << ": packing " << kFp32HalfSyncDestTiles << " tiles did not complete";
    EXPECT_TRUE(MetalContext::instance().watcher_server()->exception_message().empty())
        << path << ": packing " << kFp32HalfSyncDestTiles
        << " tiles tripped the watcher: " << MetalContext::instance().watcher_server()->exception_message();
    EXPECT_EQ(result, golden) << path << ": in-capacity pack does not reproduce the input";

    run_pack_block_dest_capacity(mesh, src, kFp32HalfSyncDestTiles, kOverrunTiles, id_free, completed);
    EXPECT_FALSE(completed) << path << ": the overrun program completed instead of aborting on the watcher error";
    const std::string exception = wait_for_pack_watcher_exception();
    ASSERT_FALSE(exception.empty()) << path << ": packed " << kOverrunTiles << " tiles from a "
                                    << kFp32HalfSyncDestTiles
                                    << "-tile DEST window without tripping the DST-capacity assert";
    EXPECT_NE(exception.find("tripped an assert"), std::string::npos) << exception;
    EXPECT_NE(exception.find("pack_block_dest_capacity_2_0"), std::string::npos) << exception;
    EXPECT_NE(exception.find("TRISC2"), std::string::npos) << exception;
}

}  // namespace

// Opens one Blackhole device per test with the watcher and LLK asserts enabled, so a device-side LLK_ASSERT
// surfaces as a watcher exception instead of a silent hang. Mirrors MeshWatcherFixture
// (tests/tt_metal/tt_metal/debug_tools/debug_tools_fixture.hpp); the options must be set before the device
// opens, and a tripped assert parks a TRISC, so the device is reopened for every test.
class LLKBlackholePackAssertFixture : public MeshDispatchFixture {
public:
    static constexpr int kWatcherIntervalMs = 250;

    // No suite-shared device: the options below change the firmware build, so the device must open after they
    // are set, once per test.
    static void SetUpTestSuite() {}
    static void TearDownTestSuite() {}

protected:
    void SetUp() override {
        // Capture before any early exit: TearDown still runs after a skip or a failed assertion here, and must
        // restore the real previous values, not the zero-initialised members.
        auto& rtoptions = MetalContext::instance().rtoptions();
        watcher_previous_enabled_ = rtoptions.get_watcher_enabled();
        watcher_previous_interval_ = rtoptions.get_watcher_interval();
        test_mode_previous_ = rtoptions.get_test_mode_enabled();
        llk_asserts_previous_ = rtoptions.get_llk_asserts();

        if (detail::detect_arch() != tt::ARCH::BLACKHOLE) {
            GTEST_SKIP() << "This test can only be run on Blackhole cards";
        }
        rtoptions.set_watcher_enabled(true);
        rtoptions.set_watcher_interval(kWatcherIntervalMs);
        // In test mode a watcher error stops the server and fails the wait instead of ending the process.
        rtoptions.set_test_mode_enabled(true);
        rtoptions.set_llk_asserts(true);

        const ChipId mmio_device_id = *MetalContext::instance().get_cluster().mmio_chip_ids().begin();
        auto id_to_device = distributed::MeshDevice::create_unit_meshes(
            {mmio_device_id},
            l1_small_size_,
            trace_region_size_,
            1 /*num_command_queues*/,
            MetalContext::instance().resolve_dispatch_core_config());
        devices_ = {id_to_device.at(mmio_device_id)};
        arch_ = tt::ARCH::BLACKHOLE;
        MetalContext::instance().watcher_server()->clear_log();
    }

    void TearDown() override {
        for (auto& device : devices_) {
            device->close();
        }
        devices_.clear();
        auto& rtoptions = MetalContext::instance().rtoptions();
        rtoptions.set_llk_asserts(llk_asserts_previous_);
        rtoptions.set_test_mode_enabled(test_mode_previous_);
        rtoptions.set_watcher_interval(watcher_previous_interval_);
        rtoptions.set_watcher_enabled(watcher_previous_enabled_);
    }

private:
    bool watcher_previous_enabled_{};
    int watcher_previous_interval_{};
    bool test_mode_previous_{};
    bool llk_asserts_previous_{};
};

// Reference: the legacy CB-id pack_tile keeps the DST-capacity assert.
TEST_F(LLKBlackholePackAssertFixture, PackTileRejectsDestOverrun) {
    run_pack_block_dest_overrun(*devices_.at(0), false /*id_free*/);
}

// The id-free pack_block must trip the same assert instead of silently packing stale DEST rows.
TEST_F(LLKBlackholePackAssertFixture, PackBlockIdFreeRejectsDestOverrun) {
    run_pack_block_dest_overrun(*devices_.at(0), true /*id_free*/);
}

}  // namespace tt::tt_metal
