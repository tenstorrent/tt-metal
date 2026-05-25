// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <fmt/base.h>
#include <gtest/gtest.h>
#include <cstddef>
#include <cstdint>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <algorithm>
#include <cmath>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <variant>
#include <vector>

#include <tt_stl/assert.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/device.hpp>
#include "llk_device_fixture.hpp"
#include <tt-metalium/distributed.hpp>
#include "hostdevcommon/kernel_structs.h"
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/program.hpp>
#include <tt_stl/span.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include "tt_metal/test_utils/comparison.hpp"
#include "tt_metal/test_utils/df/float32.hpp"
#include "tt_metal/test_utils/packing.hpp"
#include "tt_metal/test_utils/stimulus.hpp"
#include <umd/device/types/arch.hpp>
#include <tt-metalium/experimental/host_api.hpp>
#include <tt-metalium/experimental/dataflow_buffer/dataflow_buffer.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>

namespace tt::tt_metal {

using std::map;
using std::vector;
using namespace tt;
using namespace tt::test_utils;
using namespace tt::test_utils::df;

namespace unit_tests::sfpu_util {

const map<std::string, std::map<std::string, std::string>> sfpu_op_to_op_name = {
    // FIXME: #1157
    {"relu", {{"SFPU_OP_CHAIN_0", "relu_tile_init(); relu_tile(0);"}}},
    {"exponential", {{"SFPU_OP_CHAIN_0", "exp_tile_init(); exp_tile(0);"}}},
    {"reciprocal", {{"SFPU_OP_CHAIN_0", "recip_tile_init(); recip_tile(0);"}}},
    {"gelu", {{"SFPU_OP_CHAIN_0", "gelu_tile_init(); gelu_tile(0);"}}},
    {"sqrt", {{"SFPU_OP_CHAIN_0", "sqrt_tile_init(); sqrt_tile(0);"}}},
    {"sigmoid", {{"SFPU_OP_CHAIN_0", "sigmoid_tile_init(); sigmoid_tile(0);"}}},
    {"silu", {{"SFPU_OP_CHAIN_0", "silu_tile_init(); silu_tile(0);"}}},
    {"log", {{"SFPU_OP_CHAIN_0", "log_tile_init(); log_tile(0);"}}},
    {"tanh", {{"SFPU_OP_CHAIN_0", "tanh_tile_init(); tanh_tile(0);"}}},
    {"sign", {{"SFPU_OP_CHAIN_0", "sign_tile_init(); sign_tile(0);"}}},
};

bfloat16 sfpu_function(const std::string& op_name, const bfloat16& input) {
    if (op_name == "relu") {
        return bfloat16(fmaxf(static_cast<float>(input), 0.0f));
    }
    if (op_name == "exponential") {
        return bfloat16(std::exp(static_cast<float>(input)));
    }
    if (op_name == "reciprocal") {
        return bfloat16(1 / static_cast<float>(input));
    }
    if (op_name == "gelu") {
        static constexpr float alpha = M_2_SQRTPI * M_SQRT1_2;
        auto x = static_cast<float>(input);
        auto x3 = x * x * x;
        float result = x * 0.5 * (1.0 + tanhf(alpha * (x + 0.044715 * x3)));
        return bfloat16(result);
    }
    if (op_name == "sqrt") {
        return bfloat16(sqrtf(static_cast<float>(input)));
    }
    if (op_name == "sigmoid") {
        auto x = static_cast<float>(input);
        float result = 1 / (1 + std::exp(-x));
        return bfloat16(result);
    }
    if (op_name == "silu") {
        auto x = static_cast<float>(input);
        float result = x / (1 + std::exp(-x));
        return bfloat16(result);
    }
    if (op_name == "log") {
        return bfloat16(logf(static_cast<float>(input)));
    }
    if (op_name == "tanh") {
        return bfloat16(std::tanh(static_cast<float>(input)));
    }
    if (op_name == "sign") {
        float val = static_cast<float>(input);
        float result = static_cast<float>((val > 0.0f) - (val < 0.0f));
        return bfloat16(result);
    }
    TT_THROW("Unsupported op_name in test");
}
vector<uint32_t> generate_packed_sfpu_input(const unsigned int numel, const std::string& op_name, const int seed) {
    if ((op_name == "sqrt") or (op_name == "log")) {
        return generate_packed_uniform_random_vector<uint32_t, bfloat16>(0.0001f, 4.0f, numel, seed);
    }
    if ((op_name == "exponential") or (op_name == "gelu") or (op_name == "reciprocal")) {
        auto possible_values = vector<bfloat16>({-1.0f, -0.5f, 0.5f, 1.0f});
        return generate_packed_random_vector_from_vector<uint32_t, bfloat16>(possible_values, numel, seed);
    }
    return generate_packed_uniform_random_vector<uint32_t, bfloat16>(-1.0f, 1.0f, numel, seed);
}

bool is_close_packed_sfpu_output(
    const std::vector<uint32_t>& vec_a, const std::vector<uint32_t>& vec_b, const std::string& op_name) {
    if (op_name == "tanh") {
        return is_close_packed_vectors<bfloat16, uint32_t>(
            vec_a, vec_b, [&](const bfloat16& a, const bfloat16& b) { return is_close(a, b, 0.175f, 0.1f); });
    }
    if ((op_name == "gelu") or (op_name == "relu")) {
        return is_close_packed_vectors<bfloat16, uint32_t>(
            vec_a, vec_b, [&](const bfloat16& a, const bfloat16& b) { return is_close(a, b, 0.15f); });
    }
    if ((op_name == "exponential")) {
        return is_close_packed_vectors<bfloat16, uint32_t>(
            vec_a, vec_b, [&](const bfloat16& a, const bfloat16& b) { return is_close(a, b, 0.1f, 0.1f); });
    }
    if ((op_name == "log")) {
        return is_close_packed_vectors<bfloat16, uint32_t>(
            vec_a, vec_b, [&](const bfloat16& a, const bfloat16& b) { return is_close(a, b, 0.03f, 0.02f); });
    }
    return is_close_packed_vectors<bfloat16, uint32_t>(
        vec_a, vec_b, [&](const bfloat16& a, const bfloat16& b) { return is_close(a, b, 0.06f, 0.006f); });
}

}  // namespace unit_tests::sfpu_util

namespace unit_tests::compute::sfpu {

struct SfpuConfig {
    size_t num_tiles = 0;
    size_t tile_byte_size = 0;
    tt::DataFormat l1_input_data_format = tt::DataFormat::Invalid;
    tt::DataFormat l1_output_data_format = tt::DataFormat::Invalid;
    CoreRangeSet cores;
    std::string sfpu_op;
    bool approx_mode = true;
};

/// @brief Does Dram --> Reader --> CB --> Sfpu Compute --> CB --> Writer --> Dram. So far, enqueue APIs only added to
/// grayskull
/// @param device
/// @param test_config - Configuration of the test -- see struct
/// @return
bool run_sfpu_all_same_buffer(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device, const SfpuConfig& test_config) {
    const size_t byte_size = test_config.num_tiles * test_config.tile_byte_size;
    auto* device = mesh_device->get_devices()[0];

    tt::tt_metal::InterleavedBufferConfig dram_config{
        .device = device, .size = byte_size, .page_size = byte_size, .buffer_type = tt::tt_metal::BufferType::DRAM};

    auto input_dram_buffer = CreateBuffer(dram_config);
    auto output_dram_buffer = CreateBuffer(dram_config);

    // Input
    std::vector<uint32_t> packed_input = sfpu_util::generate_packed_sfpu_input(
        byte_size / sizeof(bfloat16), test_config.sfpu_op, std::chrono::system_clock::now().time_since_epoch().count());

    // Golden output
    auto input = unpack_vector<bfloat16, uint32_t>(packed_input);
    std::vector<bfloat16> golden(input.size());
    std::transform(input.begin(), input.end(), golden.begin(), [&](const bfloat16& val) {
        return sfpu_util::sfpu_function(test_config.sfpu_op, val);
    });
    std::vector<uint32_t> packed_golden = pack_vector<uint32_t, bfloat16>(golden);

    std::map<std::string, std::string> sfpu_defines = sfpu_util::sfpu_op_to_op_name.at(test_config.sfpu_op);
    sfpu_defines["SFPU_OP_EXP_INCLUDE"] = "1";
    sfpu_defines["SFPU_OP_GELU_INCLUDE"] = "1";
    sfpu_defines["SFPU_OP_RECIP_INCLUDE"] = "1";
    sfpu_defines["SFPU_OP_SQRT_INCLUDE"] = "1";
    sfpu_defines["SFPU_OP_ERF_ERFC_INCLUDE"] = "1";
    sfpu_defines["SFPU_OP_ELU_INCLUDE"] = "1";
    sfpu_defines["SFPU_OP_NEG_INCLUDE"] = "1";
    sfpu_defines["SFPU_OP_RELU_FAMILY_INCLUDE"] = "1";
    sfpu_defines["SFPU_OP_COMPUTE_KERNEL_API_INCLUDE"] = "1";

    if (device->arch() == ARCH::QUASAR) {
        // The Metal 2.0 path supports a single-core work unit, which matches every
        // existing parametrization of this test (single CoreRange of {0, 0}).
        TT_FATAL(
            test_config.cores.ranges().size() == 1,
            "Metal 2.0 sfpu path expects a single CoreRange (got {})",
            test_config.cores.size());
        const CoreRange& core_range = *test_config.cores.ranges().begin();
        TT_FATAL(core_range.start_coord == core_range.end_coord, "Metal 2.0 sfpu path expects a single-core CoreRange");
        const CoreCoord core = core_range.start_coord;
        const experimental::metal2_host_api::NodeCoord node{core.x, core.y};

        constexpr const char* IN_DFB = "in_dfb";
        constexpr const char* OUT_DFB = "out_dfb";
        constexpr const char* READER = "reader";
        constexpr const char* WRITER = "writer";
        constexpr const char* COMPUTE = "compute";

        // Legacy DataflowBufferConfig set enable_implicit_sync = false on both DFBs;
        // mirror that with disable_implicit_sync = true.
        experimental::metal2_host_api::DataflowBufferSpec in_dfb_spec{
            .unique_id = IN_DFB,
            .entry_size = static_cast<uint32_t>(test_config.tile_byte_size),
            .num_entries = static_cast<uint32_t>(test_config.num_tiles),
            .data_format_metadata = test_config.l1_input_data_format,
            .disable_implicit_sync = true,
        };
        experimental::metal2_host_api::DataflowBufferSpec out_dfb_spec{
            .unique_id = OUT_DFB,
            .entry_size = static_cast<uint32_t>(test_config.tile_byte_size),
            .num_entries = static_cast<uint32_t>(test_config.num_tiles),
            .data_format_metadata = test_config.l1_output_data_format,
            .disable_implicit_sync = true,
        };

        experimental::metal2_host_api::KernelSpec reader_spec{
            .unique_id = READER,
            .source =
                experimental::metal2_host_api::KernelSpec::SourceFilePath{"tt_metal/kernels/dataflow/reader_unary.cpp"},
            .num_threads = 1,
            .dfb_bindings = {{
                .dfb_spec_name = IN_DFB,
                .local_accessor_name = "out",
                .endpoint_type = experimental::metal2_host_api::KernelSpec::DFBEndpointType::PRODUCER,
                .access_pattern = experimental::metal2_host_api::DFBAccessPattern::STRIDED,
            }},
            .runtime_arguments_schema = {.named_runtime_args = {"src_addr", "bank_id", "num_tiles"}},
            .config_spec =
                experimental::metal2_host_api::DataMovementConfiguration{
                    .gen2_data_movement_config =
                        experimental::metal2_host_api::DataMovementConfiguration::Gen2DataMovementConfig{}},
        };

        experimental::metal2_host_api::KernelSpec writer_spec{
            .unique_id = WRITER,
            .source =
                experimental::metal2_host_api::KernelSpec::SourceFilePath{"tt_metal/kernels/dataflow/writer_unary.cpp"},
            .num_threads = 1,
            .dfb_bindings = {{
                .dfb_spec_name = OUT_DFB,
                .local_accessor_name = "in",
                .endpoint_type = experimental::metal2_host_api::KernelSpec::DFBEndpointType::CONSUMER,
                .access_pattern = experimental::metal2_host_api::DFBAccessPattern::STRIDED,
            }},
            .runtime_arguments_schema = {.named_runtime_args = {"dst_addr", "bank_id", "num_tiles"}},
            .config_spec =
                experimental::metal2_host_api::DataMovementConfiguration{
                    .gen2_data_movement_config =
                        experimental::metal2_host_api::DataMovementConfiguration::Gen2DataMovementConfig{}},
        };

        experimental::metal2_host_api::KernelSpec::CompilerOptions::Defines compute_defines;
        for (const auto& [k, v] : sfpu_defines) {
            compute_defines.emplace_back(k, v);
        }

        experimental::metal2_host_api::KernelSpec compute_spec{
            .unique_id = COMPUTE,
            .source =
                experimental::metal2_host_api::KernelSpec::SourceFilePath{"tt_metal/kernels/compute/eltwise_sfpu.cpp"},
            .num_threads = 1,
            .compiler_options = {.defines = std::move(compute_defines)},
            .dfb_bindings =
                {{
                     .dfb_spec_name = IN_DFB,
                     .local_accessor_name = "in",
                     .endpoint_type = experimental::metal2_host_api::KernelSpec::DFBEndpointType::CONSUMER,
                     .access_pattern = experimental::metal2_host_api::DFBAccessPattern::STRIDED,
                 },
                 {
                     .dfb_spec_name = OUT_DFB,
                     .local_accessor_name = "out",
                     .endpoint_type = experimental::metal2_host_api::KernelSpec::DFBEndpointType::PRODUCER,
                     .access_pattern = experimental::metal2_host_api::DFBAccessPattern::STRIDED,
                 }},
            .compile_time_arg_bindings =
                {{"per_core_block_cnt", static_cast<uint32_t>(test_config.num_tiles)}, {"per_core_block_dim", 1u}},
            .config_spec =
                experimental::metal2_host_api::ComputeConfiguration{
                    .math_approx_mode = test_config.approx_mode,
                },
        };

        experimental::metal2_host_api::WorkUnitSpec wu{
            .unique_id = "main",
            .kernels = {READER, WRITER, COMPUTE},
            .target_nodes = node,
        };

        experimental::metal2_host_api::ProgramSpec spec{
            .program_id = "sfpu_compute",
            .kernels = {reader_spec, writer_spec, compute_spec},
            .dataflow_buffers = {in_dfb_spec, out_dfb_spec},
            .work_units = {wu},
        };

        Program program = experimental::metal2_host_api::MakeProgramFromSpec(*mesh_device, spec);

        experimental::metal2_host_api::ProgramRunParams params;
        params.kernel_run_params = {
            experimental::metal2_host_api::ProgramRunParams::KernelRunParams{
                .kernel_spec_name = READER,
                .named_runtime_args =
                    {{.node = node,
                      .args =
                          {{"src_addr", input_dram_buffer->address()},
                           {"bank_id", 0u},
                           {"num_tiles", static_cast<uint32_t>(test_config.num_tiles)}}}},
            },
            experimental::metal2_host_api::ProgramRunParams::KernelRunParams{
                .kernel_spec_name = WRITER,
                .named_runtime_args =
                    {{.node = node,
                      .args =
                          {{"dst_addr", output_dram_buffer->address()},
                           {"bank_id", 0u},
                           {"num_tiles", static_cast<uint32_t>(test_config.num_tiles)}}}},
            },
            experimental::metal2_host_api::ProgramRunParams::KernelRunParams{
                .kernel_spec_name = COMPUTE,
            },
        };
        experimental::metal2_host_api::SetProgramRunParameters(program, params);

        tt_metal::detail::WriteToBuffer(input_dram_buffer, packed_input);
        tt_metal::detail::LaunchProgram(device, program, /*wait_until_cores_done=*/true);
    } else {
        auto& cq = mesh_device->mesh_command_queue();
        auto zero_coord = distributed::MeshCoordinate(0, 0);
        auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
        distributed::MeshWorkload workload;
        tt_metal::Program program = tt_metal::CreateProgram();
        workload.add_program(device_range, std::move(program));
        auto& program_ = workload.get_programs().at(device_range);

        // Same runtime args for every core
        vector<uint32_t> reader_rt_args = {
            (uint32_t)input_dram_buffer->address(),
            (uint32_t)0,
            (uint32_t)test_config.num_tiles,
        };

        vector<uint32_t> writer_rt_args = {
            (uint32_t)output_dram_buffer->address(),
            (uint32_t)0,
            (uint32_t)test_config.num_tiles,
        };

        for (const CoreRange& core_range : test_config.cores.ranges()) {
            tt_metal::CircularBufferConfig l1_input_cb_config =
                tt_metal::CircularBufferConfig(byte_size, {{tt::CBIndex::c_0, test_config.l1_input_data_format}})
                    .set_page_size(tt::CBIndex::c_0, test_config.tile_byte_size);
            tt_metal::CreateCircularBuffer(program_, core_range, l1_input_cb_config);

            tt_metal::CircularBufferConfig l1_output_cb_config =
                tt_metal::CircularBufferConfig(byte_size, {{tt::CBIndex::c_16, test_config.l1_output_data_format}})
                    .set_page_size(tt::CBIndex::c_16, test_config.tile_byte_size);
            tt_metal::CreateCircularBuffer(program_, core_range, l1_output_cb_config);

            auto reader_kernel = tt_metal::CreateKernel(
                program_,
                "tt_metal/kernels/dataflow/reader_unary.cpp",
                test_config.cores,
                tt_metal::DataMovementConfig{
                    .processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default});

            auto writer_kernel = tt_metal::CreateKernel(
                program_,
                "tt_metal/kernels/dataflow/writer_unary.cpp",
                test_config.cores,
                tt_metal::DataMovementConfig{
                    .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

            vector<uint32_t> compute_kernel_args = {
                uint32_t(test_config.num_tiles),  // per_core_block_cnt
                1                                 // per_core_block_dim
            };

            tt_metal::CreateKernel(
                program_,
                "tt_metal/kernels/compute/eltwise_sfpu.cpp",
                test_config.cores,
                tt_metal::ComputeConfig{
                    .math_approx_mode = test_config.approx_mode,
                    .compile_args = compute_kernel_args,
                    .defines = sfpu_defines});

            for (const CoreCoord& core_coord : core_range) {
                SetRuntimeArgs(program_, writer_kernel, core_coord, writer_rt_args);
                SetRuntimeArgs(program_, reader_kernel, core_coord, reader_rt_args);
            }
        }

        tt_metal::detail::WriteToBuffer(input_dram_buffer, packed_input);
        distributed::EnqueueMeshWorkload(cq, workload, false);
        distributed::Finish(cq);
    }

    std::vector<uint32_t> dest_buffer_data;
    tt_metal::detail::ReadFromBuffer(output_dram_buffer, dest_buffer_data);

    return sfpu_util::is_close_packed_sfpu_output(dest_buffer_data, packed_golden, test_config.sfpu_op);
}

}  // namespace unit_tests::compute::sfpu
class SingleCoreSingleMeshDeviceSfpuParameterizedFixture
    : public LLKMeshDeviceFixture,
      public testing::WithParamInterface<std::tuple<size_t, std::string>> {};
TEST_P(SingleCoreSingleMeshDeviceSfpuParameterizedFixture, TensixSfpuCompute) {
    size_t num_tiles = std::get<0>(GetParam());
    std::string sfpu_op = std::get<1>(GetParam());

    CoreRange core_range({0, 0}, {0, 0});
    CoreRangeSet core_range_set({core_range});
    unit_tests::compute::sfpu::SfpuConfig test_config = {
        .num_tiles = num_tiles,
        .tile_byte_size = 2 * 32 * 32,
        .l1_input_data_format = tt::DataFormat::Float16_b,
        .l1_output_data_format = tt::DataFormat::Float16_b,
        .cores = core_range_set,
        .sfpu_op = sfpu_op,
        .approx_mode = false};
    log_info(tt::LogTest, "Testing SFPU_OP={} num_tiles={}", sfpu_op, num_tiles);
    for (unsigned int id = 0; id < num_devices_; id++) {
        EXPECT_TRUE(run_sfpu_all_same_buffer(devices_.at(id), test_config));
    }
}

INSTANTIATE_TEST_SUITE_P(
    SingleCoreSfpuCompute,
    SingleCoreSingleMeshDeviceSfpuParameterizedFixture,
    ::testing::Values(
        std::make_tuple(1, "relu"),
        std::make_tuple(1, "exponential"),
        std::make_tuple(1, "reciprocal"),
        std::make_tuple(1, "gelu"),
        std::make_tuple(1, "sqrt"),
        std::make_tuple(1, "sigmoid"),
        std::make_tuple(1, "silu"),
        std::make_tuple(1, "log"),
        std::make_tuple(1, "tanh"),
        std::make_tuple(1, "sign"),
        std::make_tuple(4, "relu"),
        std::make_tuple(4, "exponential"),
        std::make_tuple(4, "reciprocal"),
        std::make_tuple(4, "gelu"),
        std::make_tuple(4, "sqrt"),
        std::make_tuple(4, "sigmoid"),
        std::make_tuple(4, "silu"),
        std::make_tuple(4, "log"),
        std::make_tuple(4, "tanh"),
        std::make_tuple(4, "sign")));

class SingleCoreSingleMeshDeviceSfpuParameterizedApproxFixture
    : public LLKMeshDeviceFixture,
      public testing::WithParamInterface<std::tuple<size_t, std::string>> {};

TEST_P(SingleCoreSingleMeshDeviceSfpuParameterizedApproxFixture, TensixSfpuCompute) {
    size_t num_tiles = std::get<0>(GetParam());
    std::string sfpu_op = std::get<1>(GetParam());

    if (((arch_ == tt::ARCH::WORMHOLE_B0) and (sfpu_op == "relu")) or
        ((arch_ == tt::ARCH::WORMHOLE_B0) and (sfpu_op == "exponential")) or
        ((arch_ == tt::ARCH::WORMHOLE_B0) and (sfpu_op == "log"))) {
        GTEST_SKIP();
    }
    CoreRange core_range({0, 0}, {0, 0});
    CoreRangeSet core_range_set({core_range});
    unit_tests::compute::sfpu::SfpuConfig test_config = {
        .num_tiles = num_tiles,
        .tile_byte_size = 2 * 32 * 32,
        .l1_input_data_format = tt::DataFormat::Float16_b,
        .l1_output_data_format = tt::DataFormat::Float16_b,
        .cores = core_range_set,
        .sfpu_op = sfpu_op,
        .approx_mode = true};
    log_info(tt::LogTest, "Testing SFPU_OP={} num_tiles={}", sfpu_op, num_tiles);
    for (unsigned int id = 0; id < num_devices_; id++) {
        EXPECT_TRUE(run_sfpu_all_same_buffer(devices_.at(id), test_config));
    }
}
INSTANTIATE_TEST_SUITE_P(
    SingleCoreSfpuCompute,
    SingleCoreSingleMeshDeviceSfpuParameterizedApproxFixture,
    ::testing::Values(
        std::make_tuple(1, "relu"),
        std::make_tuple(1, "exponential"),
        std::make_tuple(1, "reciprocal"),
        std::make_tuple(1, "gelu"),
        std::make_tuple(1, "sqrt"),
        std::make_tuple(1, "sigmoid"),
        std::make_tuple(1, "silu"),
        std::make_tuple(1, "log"),
        std::make_tuple(1, "tanh"),
        std::make_tuple(1, "sign"),
        std::make_tuple(4, "relu"),
        std::make_tuple(4, "exponential"),
        std::make_tuple(4, "reciprocal"),
        std::make_tuple(4, "gelu"),
        std::make_tuple(4, "sqrt"),
        std::make_tuple(4, "sigmoid"),
        std::make_tuple(4, "silu"),
        std::make_tuple(4, "log"),
        std::make_tuple(4, "tanh"),
        std::make_tuple(4, "sign")));

}  // namespace tt::tt_metal
