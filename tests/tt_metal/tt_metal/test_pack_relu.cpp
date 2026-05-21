// SPDX-FileCopyrightText: © 2026 AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "common/device_fixture.hpp"

#include <cmath>
#include <cstdint>
#include <functional>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include "impl/data_format/bfloat16_utils.hpp"

using namespace tt;
using namespace tt::tt_metal;

// ReluType encoding (matches ckernel::ReluType)
enum class PackReluMode : uint32_t {
    NO_RELU = 0,
    ZERO_RELU = 1,
    MIN_THRESHOLD_RELU = 2,
    MAX_THRESHOLD_RELU = 3,
};

// Pack relu config: mode in bits [1:0], bfloat16 threshold in bits [31:16]
uint32_t make_relu_config(PackReluMode mode, float threshold = 0.0f) {
    uint16_t thresh_bf16 = std::bit_cast<uint16_t>(bfloat16(threshold));
    return (static_cast<uint32_t>(thresh_bf16) << 16) | static_cast<uint32_t>(mode);
}

// Run a pack relu test using the same infrastructure as test_direct.cpp's quasar path:
// direct_reader_unary.cpp → eltwise_copy.cpp (with PACK_RELU) → direct_writer_unary.cpp
static void run_pack_relu_test(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    uint32_t relu_config,
    const std::function<float(float)>& golden_fn) {
    IDevice* dev = mesh_device->get_devices()[0];
    const experimental::metal2_host_api::NodeCoord node{0, 0};

    uint32_t single_tile_size = 2 * 1024;
    uint32_t num_tiles = 1;
    uint32_t dram_buffer_size = single_tile_size * num_tiles;

    InterleavedBufferConfig src_config{
        .device = dev, .size = dram_buffer_size, .page_size = single_tile_size, .buffer_type = BufferType::DRAM};
    auto src_dram_buffer = CreateBuffer(src_config);
    uint32_t dram_buffer_src_addr = src_dram_buffer->address();

    InterleavedBufferConfig dst_config{
        .device = dev, .size = dram_buffer_size, .page_size = single_tile_size, .buffer_type = BufferType::DRAM};
    auto dst_dram_buffer = CreateBuffer(dst_config);
    uint32_t dram_buffer_dst_addr = dst_dram_buffer->address();

    constexpr const char* INPUT_DFB = "input_dfb";
    constexpr const char* OUTPUT_DFB = "output_dfb";
    constexpr const char* READER = "reader";
    constexpr const char* WRITER = "writer";
    constexpr const char* COMPUTE = "compute";

    // Legacy DataflowBufferConfig used enable_implicit_sync = true on both DFBs;
    // keep DataflowBufferSpec::disable_implicit_sync at its default (false) and
    // set the program-level reservation flag below.
    experimental::metal2_host_api::DataflowBufferSpec input_dfb_spec{
        .unique_id = INPUT_DFB,
        .entry_size = single_tile_size,
        .num_entries = 2,
        .data_format_metadata = tt::DataFormat::Float16_b,
    };
    experimental::metal2_host_api::DataflowBufferSpec output_dfb_spec{
        .unique_id = OUTPUT_DFB,
        .entry_size = single_tile_size,
        .num_entries = 2,
        .data_format_metadata = tt::DataFormat::Float16_b,
    };

    experimental::metal2_host_api::KernelSpec reader_spec{
        .unique_id = READER,
        .source =
            experimental::metal2_host_api::KernelSpec::SourceFilePath{
                "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/dram/direct_reader_unary.cpp"},
        .num_threads = 1,
        .dfb_bindings = {{
            .dfb_spec_name = INPUT_DFB,
            .local_accessor_name = "out",
            .endpoint_type = experimental::metal2_host_api::KernelSpec::DFBEndpointType::PRODUCER,
            .access_pattern = experimental::metal2_host_api::DFBAccessPattern::STRIDED,
        }},
        .compile_time_arg_bindings = {{"use_dfbs", 1u}},
        .runtime_arguments_schema =
            {.named_runtime_args = {"src_addr", "src_bank_id", "num_tiles", "dram_page_stride"}},
        .config_spec =
            experimental::metal2_host_api::DataMovementConfiguration{
                .gen2_data_movement_config =
                    experimental::metal2_host_api::DataMovementConfiguration::Gen2DataMovementConfig{}},
    };

    experimental::metal2_host_api::KernelSpec writer_spec{
        .unique_id = WRITER,
        .source =
            experimental::metal2_host_api::KernelSpec::SourceFilePath{
                "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/dram/direct_writer_unary.cpp"},
        .num_threads = 1,
        .dfb_bindings = {{
            .dfb_spec_name = OUTPUT_DFB,
            .local_accessor_name = "in",
            .endpoint_type = experimental::metal2_host_api::KernelSpec::DFBEndpointType::CONSUMER,
            .access_pattern = experimental::metal2_host_api::DFBAccessPattern::STRIDED,
        }},
        .compile_time_arg_bindings = {{"use_dfbs", 1u}},
        .runtime_arguments_schema =
            {.named_runtime_args = {"dst_addr", "dst_bank_id", "num_tiles", "dram_page_stride"}},
        .config_spec =
            experimental::metal2_host_api::DataMovementConfiguration{
                .gen2_data_movement_config =
                    experimental::metal2_host_api::DataMovementConfiguration::Gen2DataMovementConfig{}},
    };

    experimental::metal2_host_api::KernelSpec compute_spec{
        .unique_id = COMPUTE,
        .source =
            experimental::metal2_host_api::KernelSpec::SourceFilePath{
                "tests/tt_metal/tt_metal/test_kernels/compute/eltwise_copy.cpp"},
        .num_threads = 1,
        .compiler_options = {.defines = {{"PACK_RELU", "1"}}},
        .dfb_bindings =
            {{
                 .dfb_spec_name = INPUT_DFB,
                 .local_accessor_name = "in",
                 .endpoint_type = experimental::metal2_host_api::KernelSpec::DFBEndpointType::CONSUMER,
                 .access_pattern = experimental::metal2_host_api::DFBAccessPattern::STRIDED,
             },
             {
                 .dfb_spec_name = OUTPUT_DFB,
                 .local_accessor_name = "out",
                 .endpoint_type = experimental::metal2_host_api::KernelSpec::DFBEndpointType::PRODUCER,
                 .access_pattern = experimental::metal2_host_api::DFBAccessPattern::STRIDED,
             }},
        .compile_time_arg_bindings = {{"per_core_tile_cnt", num_tiles}, {"use_dfbs", 1u}},
        .runtime_arguments_schema = {.named_runtime_args = {"relu_config"}},
        .config_spec = experimental::metal2_host_api::ComputeConfiguration{},
    };

    experimental::metal2_host_api::WorkUnitSpec wu{
        .unique_id = "main",
        .kernels = {READER, WRITER, COMPUTE},
        .target_nodes = node,
    };

    experimental::metal2_host_api::ProgramSpec spec{
        .program_id = "pack_relu",
        .kernels = {reader_spec, writer_spec, compute_spec},
        .dataflow_buffers = {input_dfb_spec, output_dfb_spec},
        .work_units = {wu},
    };

    Program program = experimental::metal2_host_api::MakeProgramFromSpec(*mesh_device, spec);

    // Stimulus: random bfloat16 in [-1, 1]
    std::vector<uint32_t> src_vec = create_random_vector_of_bfloat16(dram_buffer_size, 1.0f, 0xCAFE);
    detail::WriteToBuffer(src_dram_buffer, src_vec);

    const uint32_t src_aligned_page_size = static_cast<uint32_t>(src_dram_buffer->aligned_page_size());
    const uint32_t dst_aligned_page_size = static_cast<uint32_t>(dst_dram_buffer->aligned_page_size());

    experimental::metal2_host_api::ProgramRunParams params;
    params.kernel_run_params = {
        experimental::metal2_host_api::ProgramRunParams::KernelRunParams{
            .kernel_spec_name = READER,
            .named_runtime_args =
                {{.node = node,
                  .args =
                      {{"src_addr", dram_buffer_src_addr},
                       {"src_bank_id", 0u},
                       {"num_tiles", num_tiles},
                       {"dram_page_stride", src_aligned_page_size}}}},
        },
        experimental::metal2_host_api::ProgramRunParams::KernelRunParams{
            .kernel_spec_name = WRITER,
            .named_runtime_args =
                {{.node = node,
                  .args =
                      {{"dst_addr", dram_buffer_dst_addr},
                       {"dst_bank_id", 0u},
                       {"num_tiles", num_tiles},
                       {"dram_page_stride", dst_aligned_page_size}}}},
        },
        experimental::metal2_host_api::ProgramRunParams::KernelRunParams{
            .kernel_spec_name = COMPUTE,
            .named_runtime_args = {{.node = node, .args = {{"relu_config", relu_config}}}},
        },
    };
    experimental::metal2_host_api::SetProgramRunParameters(program, params);

    detail::LaunchProgram(dev, program, true);

    std::vector<uint32_t> result_vec;
    detail::ReadFromBuffer(dst_dram_buffer, result_vec);

    // Build golden
    std::vector<uint32_t> golden(src_vec.size());
    for (size_t i = 0; i < src_vec.size(); ++i) {
        auto [lo_bf, hi_bf] = unpack_two_bfloat16_from_uint32(src_vec[i]);
        float lo_f = golden_fn(static_cast<float>(lo_bf));
        float hi_f = golden_fn(static_cast<float>(hi_bf));
        golden[i] = pack_two_bfloat16_into_uint32({bfloat16(lo_f), bfloat16(hi_f)});
    }

    auto comparison_function = [](float a, float b) {
        const float atol = 0.02f;
        const float rtol = 0.05f;
        float maxabs = fmaxf(fabsf(a), fabsf(b));
        float absdiff = fabsf(a - b);
        return (absdiff <= atol) || absdiff < rtol * maxabs;
    };

    int argfail = -1;
    bool pass = packed_uint32_t_vector_comparison(result_vec, golden, comparison_function, &argfail);
    EXPECT_TRUE(pass) << "Failure position=" << argfail;
}

// ZERO_RELU: max(0, x)
TEST_F(QuasarMeshDeviceSingleCardFixture, PackReluZero) {
    run_pack_relu_test(
        this->devices_.at(0), make_relu_config(PackReluMode::ZERO_RELU), [](float x) { return std::max(0.0f, x); });
}

// MIN_THRESHOLD_RELU: x <= threshold ? 0 : x (threshold = 0.25)
TEST_F(QuasarMeshDeviceSingleCardFixture, PackReluMinThreshold) {
    const float threshold = 0.25f;
    run_pack_relu_test(
        this->devices_.at(0), make_relu_config(PackReluMode::MIN_THRESHOLD_RELU, threshold), [threshold](float x) {
            return x <= threshold ? 0.0f : x;
        });
}

// MAX_THRESHOLD_RELU: clamp to [0, threshold] (threshold = 0.5)
TEST_F(QuasarMeshDeviceSingleCardFixture, PackReluMaxThreshold) {
    const float threshold = 0.5f;
    run_pack_relu_test(
        this->devices_.at(0), make_relu_config(PackReluMode::MAX_THRESHOLD_RELU, threshold), [threshold](float x) {
            return x < 0.0f ? 0.0f : std::min(x, threshold);
        });
}
