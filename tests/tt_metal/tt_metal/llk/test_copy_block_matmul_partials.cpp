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

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/kernel_types.hpp>
#include "llk_device_fixture.hpp"
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/program.hpp>
#include <tt_stl/span.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "tt_metal/test_utils/stimulus.hpp"
#include <umd/device/types/arch.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>

namespace tt::tt_metal {
class IDevice;
}  // namespace tt::tt_metal

namespace tt::tt_metal {

using std::vector;
using namespace tt;
using namespace tt::test_utils;

namespace unit_tests::compute::matmul_partials {

struct CopyBlockMatmulPartialsConfig {
    uint32_t single_tile_size = 2 * 32 * 32;
    uint32_t num_tiles = 1;
    // *_ublock defines no. of tiles finished with single LLK API call:
    uint32_t reader_ublock = 1;
    uint32_t writer_ublock = 1;
    uint32_t compute_ublock = 1;
    uint32_t src0_cb_index = 0;
    uint32_t ouput_cb_index = 16;
    // Whether or not we want the result to be stored in DST in FP32:
    bool fp32_dest_acc_en = false;
    // Whether or not to sync full/half DST between MATH and PACK:
    bool dst_full_sync_en = false;
};

static std::vector<uint32_t> generate_copy_block_stimulus(
    uint32_t dram_buffer_size, const CopyBlockMatmulPartialsConfig& test_config) {
    std::vector<uint32_t> src_vec = create_random_vector_of_bfloat16(dram_buffer_size, 100, 0);

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

    uint32_t single_tile_size = test_config.single_tile_size;
    uint32_t num_tiles = test_config.num_tiles;
    uint32_t dram_buffer_size = single_tile_size * num_tiles;

    tt::DataFormat data_format = test_config.fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;

    distributed::DeviceLocalBufferConfig dram_local_config{
        .page_size = dram_buffer_size, .buffer_type = tt_metal::BufferType::DRAM, .bottom_up = false};
    distributed::ReplicatedBufferConfig dram_buffer_config{.size = dram_buffer_size};
    auto src_dram_buffer = distributed::MeshBuffer::create(dram_buffer_config, dram_local_config, mesh_device.get());
    auto dst_dram_buffer = distributed::MeshBuffer::create(dram_buffer_config, dram_local_config, mesh_device.get());

    uint32_t num_input_tiles = test_config.reader_ublock;
    uint32_t num_output_tiles = test_config.writer_ublock;

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
        reader_hw_config = experimental::DataMovementGen2Config{.disable_dfb_implicit_sync_for_all = true};
    } else {
        reader_hw_config = experimental::DataMovementGen1Config{
            .processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default};
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
        writer_hw_config = experimental::DataMovementGen2Config{.disable_dfb_implicit_sync_for_all = true};
    } else {
        writer_hw_config = experimental::DataMovementGen1Config{
            .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default};
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
    if (test_config.fp32_dest_acc_en) {
        compute_defines.emplace("DST_ACCUM_MODE", "1");
    }

    experimental::ComputeHardwareConfig compute_hw_config;
    {
        // When fp32_dest_acc_en is true the src DFB is Float32 and the compute kernel
        // consumes it, so the Metal 2.0 host API requires an explicit unpack_to_dest_mode entry.
        // Default is unpack via SrcA/B, ~19-bit precision.
        experimental::ComputeUnpackToDestModes unpack_modes{};
        if (test_config.fp32_dest_acc_en) {
            unpack_modes = {{SRC0_DFB, tt::tt_metal::UnpackToDestMode::Default}};
        }
        if (mesh_device->arch() == tt::ARCH::QUASAR) {
            compute_hw_config = experimental::ComputeGen2Config{
                .fp32_dest_acc_en = test_config.fp32_dest_acc_en,
                .dst_full_sync_en = test_config.dst_full_sync_en,
                .unpack_to_dest_mode = unpack_modes,
            };
        } else {
            compute_hw_config = experimental::ComputeGen1Config{
                .fp32_dest_acc_en = test_config.fp32_dest_acc_en,
                .dst_full_sync_en = test_config.dst_full_sync_en,
                .unpack_to_dest_mode = unpack_modes,
            };
        }
    }
    experimental::KernelSpec compute_spec{
        .unique_id = COMPUTE,
        .source =

            "tests/tt_metal/tt_metal/test_kernels/compute/eltwise_copy_block_matmul_partials.cpp",
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

    std::vector<uint32_t> src_vec = generate_copy_block_stimulus(dram_buffer_size, test_config);
    distributed::WriteShard(cq, src_dram_buffer, src_vec, zero_coord);

    experimental::ProgramRunArgs params;
    params.kernel_run_args = {
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = READER,
            .runtime_arg_values =
                {{node,
                  {{"src_addr", src_dram_buffer->address()},
                   {"src_dram_bank_id", 0u},
                   {"num_tiles", num_tiles},
                   {"ublock_size_tiles", test_config.reader_ublock},
                   {"reader_only", 0u}}}},
        },
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = WRITER,
            .runtime_arg_values =
                {{node,
                  {{"dst_addr", dst_dram_buffer->address()},
                   {"dst_dram_bank_id", 0u},
                   {"num_tiles", num_tiles},
                   {"ublock_size_tiles", test_config.writer_ublock},
                   {"writer_only", 0u}}}},
        },
        experimental::ProgramRunArgs::KernelRunArgs{.kernel = COMPUTE},
    };
    experimental::SetProgramRunArgs(program_run, params);

    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);

    std::vector<uint32_t> result_vec;
    distributed::ReadShard(cq, result_vec, dst_dram_buffer, zero_coord);

    EXPECT_EQ(src_vec.size(), result_vec.size());
    EXPECT_EQ(src_vec, result_vec);
}

}  // namespace unit_tests::compute::matmul_partials

////////////////////////////////////////////////////////////////////////////
//                             Test Description
// ------------------------------------------------------------------------
// These tests aim to cover usage of these API calls:
// - copy_block_matmul_partials
// - pack_tile_block
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

}  // namespace tt::tt_metal
