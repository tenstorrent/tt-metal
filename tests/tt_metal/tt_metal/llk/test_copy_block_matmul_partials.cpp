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

void run_single_core_copy_block_matmul_partials_quasar(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device, const CopyBlockMatmulPartialsConfig& test_config) {
    auto& cq = mesh_device->mesh_command_queue();
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    const experimental::metal2_host_api::NodeCoord node{0, 0};

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

    constexpr const char* SRC0_DFB = "src0_dfb";
    constexpr const char* DST_DFB = "dst_dfb";
    constexpr const char* READER = "reader";
    constexpr const char* WRITER = "writer";
    constexpr const char* COMPUTE = "compute";

    experimental::metal2_host_api::DataflowBufferSpec src0_dfb_spec{
        .unique_id = SRC0_DFB,
        .entry_size = single_tile_size,
        .num_entries = num_input_tiles,
        .data_format_metadata = data_format,
        // Match pre-migration behavior: legacy DataflowBufferConfig set enable_implicit_sync=false.
        .disable_implicit_sync = true,
    };
    experimental::metal2_host_api::DataflowBufferSpec dst_dfb_spec{
        .unique_id = DST_DFB,
        .entry_size = single_tile_size,
        .num_entries = num_output_tiles,
        .data_format_metadata = data_format,
        // Match pre-migration behavior: legacy DataflowBufferConfig set enable_implicit_sync=false.
        .disable_implicit_sync = true,
    };

    experimental::metal2_host_api::KernelSpec reader_spec{
        .unique_id = READER,
        .source =
            experimental::metal2_host_api::KernelSpec::SourceFilePath{
                "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_unary_push_n.cpp"},
        .num_threads = 1,
        .dfb_bindings = {{
            .dfb_spec_name = SRC0_DFB,
            .local_accessor_name = "out",
            .endpoint_type = experimental::metal2_host_api::KernelSpec::DFBEndpointType::PRODUCER,
            .access_pattern = experimental::metal2_host_api::DFBAccessPattern::STRIDED,
        }},
        .runtime_arguments_schema =
            {.named_runtime_args = {"src_addr", "src_dram_bank_id", "num_tiles", "ublock_size_tiles", "reader_only"}},
        .config_spec =
            experimental::metal2_host_api::DataMovementConfiguration{
                .gen2_data_movement_config =
                    experimental::metal2_host_api::DataMovementConfiguration::Gen2DataMovementConfig{}},
    };

    experimental::metal2_host_api::KernelSpec writer_spec{
        .unique_id = WRITER,
        .source =
            experimental::metal2_host_api::KernelSpec::SourceFilePath{
                "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary_pop_n.cpp"},
        .num_threads = 1,
        .dfb_bindings = {{
            .dfb_spec_name = DST_DFB,
            .local_accessor_name = "in",
            .endpoint_type = experimental::metal2_host_api::KernelSpec::DFBEndpointType::CONSUMER,
            .access_pattern = experimental::metal2_host_api::DFBAccessPattern::STRIDED,
        }},
        .runtime_arguments_schema =
            {.named_runtime_args = {"dst_addr", "dst_dram_bank_id", "num_tiles", "ublock_size_tiles", "writer_only"}},
        .config_spec =
            experimental::metal2_host_api::DataMovementConfiguration{
                .gen2_data_movement_config =
                    experimental::metal2_host_api::DataMovementConfiguration::Gen2DataMovementConfig{}},
    };

    experimental::metal2_host_api::KernelSpec::CompilerOptions::Defines compute_defines;
    if (test_config.fp32_dest_acc_en) {
        compute_defines.emplace_back("DST_ACCUM_MODE", "1");
    }

    experimental::metal2_host_api::KernelSpec compute_spec{
        .unique_id = COMPUTE,
        .source =
            experimental::metal2_host_api::KernelSpec::SourceFilePath{
                "tests/tt_metal/tt_metal/test_kernels/compute/eltwise_copy_block_matmul_partials.cpp"},
        .num_threads = 1,
        .compiler_options = {.defines = compute_defines},
        .dfb_bindings =
            {{
                 .dfb_spec_name = SRC0_DFB,
                 .local_accessor_name = "in",
                 .endpoint_type = experimental::metal2_host_api::KernelSpec::DFBEndpointType::CONSUMER,
                 .access_pattern = experimental::metal2_host_api::DFBAccessPattern::STRIDED,
             },
             {
                 .dfb_spec_name = DST_DFB,
                 .local_accessor_name = "out",
                 .endpoint_type = experimental::metal2_host_api::KernelSpec::DFBEndpointType::PRODUCER,
                 .access_pattern = experimental::metal2_host_api::DFBAccessPattern::STRIDED,
             }},
        .compile_time_arg_bindings = {{"num_tiles", num_tiles}, {"num_single_transfer", test_config.compute_ublock}},
        .config_spec =
            experimental::metal2_host_api::ComputeConfiguration{
                .fp32_dest_acc_en = test_config.fp32_dest_acc_en,
                .dst_full_sync_en = test_config.dst_full_sync_en,
            },
    };

    experimental::metal2_host_api::WorkUnitSpec wu{
        .unique_id = "main",
        .kernels = {READER, WRITER, COMPUTE},
        .target_nodes = node,
    };

    experimental::metal2_host_api::ProgramSpec spec{
        .program_id = "single_core_copy_block_matmul_partials",
        .kernels = {reader_spec, writer_spec, compute_spec},
        .dataflow_buffers = {src0_dfb_spec, dst_dfb_spec},
        .work_units = {wu},
    };

    Program program = experimental::metal2_host_api::MakeProgramFromSpec(*mesh_device, spec);

    std::vector<uint32_t> src_vec = generate_copy_block_stimulus(dram_buffer_size, test_config);
    distributed::WriteShard(cq, src_dram_buffer, src_vec, zero_coord);

    experimental::metal2_host_api::ProgramRunParams params;
    params.kernel_run_params = {
        experimental::metal2_host_api::ProgramRunParams::KernelRunParams{
            .kernel_spec_name = READER,
            .named_runtime_args =
                {{.node = node,
                  .args =
                      {{"src_addr", src_dram_buffer->address()},
                       {"src_dram_bank_id", 0u},
                       {"num_tiles", num_tiles},
                       {"ublock_size_tiles", test_config.reader_ublock},
                       {"reader_only", 0u}}}},
        },
        experimental::metal2_host_api::ProgramRunParams::KernelRunParams{
            .kernel_spec_name = WRITER,
            .named_runtime_args =
                {{.node = node,
                  .args =
                      {{"dst_addr", dst_dram_buffer->address()},
                       {"dst_dram_bank_id", 0u},
                       {"num_tiles", num_tiles},
                       {"ublock_size_tiles", test_config.writer_ublock},
                       {"writer_only", 0u}}}},
        },
        experimental::metal2_host_api::ProgramRunParams::KernelRunParams{
            .kernel_spec_name = COMPUTE,
        },
    };
    experimental::metal2_host_api::SetProgramRunParameters(program, params);

    auto* dev = mesh_device->get_devices()[0];
    tt_metal::detail::LaunchProgram(dev, program, /*wait_until_cores_done=*/true);

    std::vector<uint32_t> result_vec;
    distributed::ReadShard(cq, result_vec, dst_dram_buffer, zero_coord);

    EXPECT_EQ(src_vec.size(), result_vec.size());
    EXPECT_EQ(src_vec, result_vec);
}

void run_single_core_copy_block_matmul_partials_legacy(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device, const CopyBlockMatmulPartialsConfig& test_config) {
    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    auto& cq = mesh_device->mesh_command_queue();
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    distributed::MeshWorkload workload;
    tt_metal::Program program = tt_metal::CreateProgram();
    workload.add_program(device_range, std::move(program));
    auto& program_ = workload.get_programs().at(device_range);
    auto* device = mesh_device->get_devices()[0];

    CoreCoord core = {0, 0};
    uint32_t single_tile_size = test_config.single_tile_size;
    uint32_t num_tiles = test_config.num_tiles;
    uint32_t dram_buffer_size = single_tile_size * num_tiles;

    tt_metal::InterleavedBufferConfig dram_config{
        .device = device,
        .size = dram_buffer_size,
        .page_size = dram_buffer_size,
        .buffer_type = tt_metal::BufferType::DRAM};

    auto src_dram_buffer_bf16 = CreateBuffer(dram_config);
    uint32_t dram_buffer_src_addr = src_dram_buffer_bf16->address();
    auto dst_dram_buffer = CreateBuffer(dram_config);
    uint32_t dram_buffer_dst_addr = dst_dram_buffer->address();

    uint32_t num_input_tiles = test_config.reader_ublock;
    uint32_t num_output_tiles = test_config.writer_ublock;

    uint32_t src0_cb_index = test_config.src0_cb_index;
    uint32_t input_id = src0_cb_index;
    tt_metal::CircularBufferConfig cb_src0_config =
        tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(src0_cb_index, single_tile_size);

    if (test_config.fp32_dest_acc_en) {
        cb_src0_config = tt_metal::CircularBufferConfig(
                             num_input_tiles * single_tile_size, {{src0_cb_index, tt::DataFormat::Float32}})
                             .set_page_size(src0_cb_index, single_tile_size);
    }
    tt_metal::CreateCircularBuffer(program_, core, cb_src0_config);

    uint32_t ouput_cb_index = test_config.ouput_cb_index;
    uint32_t output_id = ouput_cb_index;
    tt_metal::CircularBufferConfig cb_output_config =
        tt_metal::CircularBufferConfig(
            num_output_tiles * single_tile_size, {{ouput_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(ouput_cb_index, single_tile_size);
    if (test_config.fp32_dest_acc_en) {
        cb_output_config = tt_metal::CircularBufferConfig(
                               num_output_tiles * single_tile_size, {{ouput_cb_index, tt::DataFormat::Float32}})
                               .set_page_size(ouput_cb_index, single_tile_size);
    }
    tt_metal::CreateCircularBuffer(program_, core, cb_output_config);

    auto unary_reader_kernel = tt_metal::CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_unary_push_n.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default});

    auto unary_writer_kernel = tt_metal::CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary_pop_n.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

    vector<uint32_t> compute_kernel_args = {
        uint(num_tiles), uint(test_config.compute_ublock), uint(input_id), uint(output_id)};

    std::map<std::string, std::string> defines;
    if (test_config.fp32_dest_acc_en) {
        defines["DST_ACCUM_MODE"] = "1";
    }
    tt_metal::CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/compute/eltwise_copy_block_matmul_partials.cpp",
        core,
        tt_metal::ComputeConfig{
            .fp32_dest_acc_en = test_config.fp32_dest_acc_en,
            .dst_full_sync_en = test_config.dst_full_sync_en,
            .compile_args = compute_kernel_args,
            .defines = defines});
    ////////////////////////////////////////////////////////////////////////////
    //                      Execute Application
    ////////////////////////////////////////////////////////////////////////////
    std::vector<uint32_t> src_vec = generate_copy_block_stimulus(dram_buffer_size, test_config);

    tt_metal::detail::WriteToBuffer(src_dram_buffer_bf16, src_vec);

    tt_metal::SetRuntimeArgs(
        program_,
        unary_reader_kernel,
        core,
        {dram_buffer_src_addr,
         (uint32_t)0,  // dram bank id
         num_tiles,
         input_id,
         test_config.reader_ublock,
         false});

    tt_metal::SetRuntimeArgs(
        program_,
        unary_writer_kernel,
        core,
        {dram_buffer_dst_addr,
         (uint32_t)0,  // dram bank id
         num_tiles,
         output_id,
         test_config.writer_ublock,
         false});

    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);
    distributed::Finish(cq);

    std::vector<uint32_t> result_vec_bf16;
    tt_metal::detail::ReadFromBuffer(dst_dram_buffer, result_vec_bf16);

    ////////////////////////////////////////////////////////////////////////////
    //                      Validation & Teardown
    ////////////////////////////////////////////////////////////////////////////
    EXPECT_EQ(src_vec.size(), result_vec_bf16.size());
    EXPECT_EQ(src_vec, result_vec_bf16);
}

void run_single_core_copy_block_matmul_partials(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device, const CopyBlockMatmulPartialsConfig& test_config) {
    if (MetalContext::instance().get_cluster().arch() == ARCH::QUASAR) {
        run_single_core_copy_block_matmul_partials_quasar(mesh_device, test_config);
    } else {
        run_single_core_copy_block_matmul_partials_legacy(mesh_device, test_config);
    }
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
