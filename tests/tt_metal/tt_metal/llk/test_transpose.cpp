// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <fmt/base.h>
#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>
#include <sys/types.h>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tilize_utils.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <variant>
#include <vector>

#include <tt_stl/assert.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/kernel_types.hpp>
#include "llk_device_fixture.hpp"
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/host_api.hpp>
#include "hostdevcommon/kernel_structs.h"
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/program.hpp>
#include <tt_stl/span.hpp>
#include "test_golden_impls.hpp"
#include <tt-metalium/tt_backend_api_types.hpp>
#include "tt_metal/test_utils/df/float32.hpp"
#include <tt-metalium/tensor_accessor_args.hpp>
#include "impl/data_format/bfloat16_utils.hpp"
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/tensor/mesh_tensor.hpp>

namespace tt::tt_metal {
class IDevice;
}  // namespace tt::tt_metal

namespace tt::tt_metal {

using std::vector;
using namespace tt;
using namespace tt::test_utils;
using namespace tt::test_utils::df;

namespace unit_tests::compute::transpose {

enum TransposeType : uint8_t { WH = 0 };

struct TransposeConfig {
    bool short_init;
    bool transpose_dest;
    uint32_t single_tile_size;
    std::vector<uint32_t> shape;
    TransposeType transpose_type;
};

// Tiled dimensions derived from a 4-D NCHW tensor shape, with shared validation.
struct TransposeDims {
    uint32_t W;
    uint32_t H;
    uint32_t NC;
    uint32_t Wt;
    uint32_t Ht;
    uint32_t num_tensor_tiles;
};

static TransposeDims compute_and_validate_transpose_dims(const std::vector<uint32_t>& shape) {
    TT_FATAL(shape.size() == 4, "Error");
    const uint32_t W = shape[3];
    const uint32_t H = shape[2];
    const uint32_t NC = shape[1] * shape[0];
    TT_FATAL(W % 32 == 0 && H % 32 == 0, "Error");
    TT_FATAL(H > 0 && W > 0 && NC > 0, "Error");
    const uint32_t Wt = W / 32;
    // size of DST register, with unary r/w this currently only works if the entire Wt fits into DST for reduce
    TT_FATAL(Wt <= 16, "Error");
    const uint32_t Ht = H / 32;
    return TransposeDims{
        .W = W,
        .H = H,
        .NC = NC,
        .Wt = Wt,
        .Ht = Ht,
        .num_tensor_tiles = NC * H * W / (32 * 32),
    };
}

void validate_transpose_wh(
    const std::vector<uint32_t>& src_vec, const std::vector<uint32_t>& shape, const std::vector<uint32_t>& result_vec) {
    int argfail = -1;
    auto comparison_function = [](float a, float b) {
        const float rtol = 0.02f;
        const float atol = 1e-3f;
        float maxabs = fmaxf(fabsf(a), fabsf(b));
        float absdiff = fabsf(a - b);
        auto result = (absdiff <= atol) || absdiff < rtol * maxabs;
        if (!result) {
            absdiff *= 1.0f;  // breakpoint spot
        }
        return result;
    };

    // recover a linear view of input vector for consumption by gold_ function
    auto u16_src0_vec = u16_from_u32_vector(src_vec);
    vector<uint16_t> src_linear =
        convert_layout<uint16_t>(u16_src0_vec, shape, TensorLayoutType::TILED_NFACES, TensorLayoutType::LIN_ROW_MAJOR);
    vector<uint16_t> gold_reduced =
        ::unit_tests::compute::gold_transpose_wh(src_linear, shape);  // result is uint16_t untilized

    // Tilize from row major and convert to pairs (uint32_t)
    TT_FATAL(shape.size() == 4, "Error");
    vector<uint32_t> shapeR{shape[0], shape[1], shape[3], shape[2]};
    auto gold_4f_u32 = u32_from_u16_vector(convert_layout<uint16_t>(
        gold_reduced, shapeR, TensorLayoutType::LIN_ROW_MAJOR, TensorLayoutType::TILED_NFACES));

    bool pass = packed_uint32_t_vector_comparison(result_vec, gold_4f_u32, comparison_function, &argfail);
    if (not pass) {
        log_error(LogTest, "Failure position={}", argfail);
    }
    EXPECT_TRUE(pass);
}

// Reads the destination buffer, checks the expected size, and validates the transpose result.
// Used by the legacy (non-Quasar) launch path.
static void read_and_validate_transpose_result(
    const std::shared_ptr<tt_metal::Buffer>& dst_dram_buffer,
    const std::vector<uint32_t>& src_vec,
    const std::vector<uint32_t>& shape,
    const TransposeDims& dims) {
    std::vector<uint32_t> result_vec;
    tt_metal::detail::ReadFromBuffer(dst_dram_buffer, result_vec);
    // Expecting one tile in H, and half the elements since the vector packs 2 uint16_ts.
    EXPECT_EQ(result_vec.size(), dims.NC * dims.H * dims.W / 2);
    validate_transpose_wh(src_vec, shape, result_vec);
}

// Quasar (Metal 2.0) variant: reads the destination tensor's underlying reference buffer
// instead of a raw shared_ptr<Buffer>. Shape/size check and golden comparison are shared
// with the legacy path via validate_transpose_wh().
static void read_and_validate_transpose_result_quasar(
    const MeshTensor& dst_tensor,
    const std::vector<uint32_t>& src_vec,
    const std::vector<uint32_t>& shape,
    const TransposeDims& dims) {
    std::vector<uint32_t> result_vec;
    tt_metal::detail::ReadFromBuffer(*dst_tensor.mesh_buffer().get_reference_buffer(), result_vec);
    EXPECT_EQ(result_vec.size(), dims.NC * dims.H * dims.W / 2);
    validate_transpose_wh(src_vec, shape, result_vec);
}

// Build a TensorSpec describing a flat DRAM-interleaved buffer of `total_entries`
// pages, each `entry_size` bytes. Used to bind src/dst tensors as TensorParameters
// to the reader/writer kernels via the Metal 2.0 named TensorAccessor ctor.
static inline tt::tt_metal::TensorSpec make_flat_dram_tensor_spec(uint32_t entry_size, uint32_t total_entries) {
    const uint32_t entry_size_words = entry_size / sizeof(uint32_t);
    auto page_config = tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR);
    auto memory_config =
        tt::tt_metal::MemoryConfig{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM};
    auto tensor_layout = tt::tt_metal::TensorLayout(tt::tt_metal::DataType::UINT32, page_config, memory_config);
    return tt::tt_metal::TensorSpec(tt::tt_metal::Shape{total_entries, entry_size_words}, tensor_layout);
}

void run_single_core_transpose_quasar(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device, const TransposeConfig& test_config) {
    auto* device = mesh_device->get_devices()[0];
    const experimental::metal2_host_api::NodeCoord node{0, 0};

    const TransposeDims dims = compute_and_validate_transpose_dims(test_config.shape);
    const uint32_t NC = dims.NC, Wt = dims.Wt, Ht = dims.Ht, num_tensor_tiles = dims.num_tensor_tiles;

    uint32_t dram_buffer_size = test_config.single_tile_size * num_tensor_tiles;

    auto in_tensor = MeshTensor::allocate_on_device(
        *mesh_device, make_flat_dram_tensor_spec(test_config.single_tile_size, num_tensor_tiles), TensorTopology{});
    auto out_tensor = MeshTensor::allocate_on_device(
        *mesh_device, make_flat_dram_tensor_spec(test_config.single_tile_size, num_tensor_tiles), TensorTopology{});

    constexpr uint32_t num_buffer_tiles = 32;
    constexpr uint32_t num_output_buffer_tiles = 32;

    constexpr const char* INPUT_DFB = "input_dfb";
    constexpr const char* OUTPUT_DFB = "output_dfb";
    constexpr const char* READER = "reader";
    constexpr const char* WRITER = "writer";
    constexpr const char* COMPUTE = "compute";
    constexpr const char* IN_TENSOR = "in_tensor";
    constexpr const char* OUT_TENSOR = "out_tensor";

    experimental::metal2_host_api::DataflowBufferSpec input_dfb_spec{
        .unique_id = INPUT_DFB,
        .entry_size = test_config.single_tile_size,
        .num_entries = num_buffer_tiles,
        .data_format_metadata = tt::DataFormat::Float16_b,
        .disable_implicit_sync = true,
    };
    experimental::metal2_host_api::DataflowBufferSpec output_dfb_spec{
        .unique_id = OUTPUT_DFB,
        .entry_size = test_config.single_tile_size,
        .num_entries = num_output_buffer_tiles,
        .data_format_metadata = tt::DataFormat::Float16_b,
        .disable_implicit_sync = true,
    };

    experimental::metal2_host_api::KernelSpec reader_spec{
        .unique_id = READER,
        .source =
            experimental::metal2_host_api::KernelSpec::SourceFilePath{
                "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_unary_transpose_wh_8bank.cpp"},
        .num_threads = 1,
        .dfb_bindings = {{
            .dfb_spec_name = INPUT_DFB,
            .local_accessor_name = "out",
            .endpoint_type = experimental::metal2_host_api::KernelSpec::DFBEndpointType::PRODUCER,
            .access_pattern = experimental::metal2_host_api::DFBAccessPattern::STRIDED,
        }},
        .tensor_bindings = {{.tensor_parameter_name = IN_TENSOR, .accessor_name = "src_tensor"}},
        .runtime_arguments_schema = {.named_runtime_args = {"N", "Ht", "Wt", "HtWt"}},
        .config_spec =
            experimental::metal2_host_api::DataMovementConfiguration{
                .gen2_data_movement_config =
                    experimental::metal2_host_api::DataMovementConfiguration::Gen2DataMovementConfig{}},
    };

    experimental::metal2_host_api::KernelSpec writer_spec{
        .unique_id = WRITER,
        .source =
            experimental::metal2_host_api::KernelSpec::SourceFilePath{
                "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary_8bank.cpp"},
        .num_threads = 1,
        .dfb_bindings = {{
            .dfb_spec_name = OUTPUT_DFB,
            .local_accessor_name = "in",
            .endpoint_type = experimental::metal2_host_api::KernelSpec::DFBEndpointType::CONSUMER,
            .access_pattern = experimental::metal2_host_api::DFBAccessPattern::STRIDED,
        }},
        .tensor_bindings = {{.tensor_parameter_name = OUT_TENSOR, .accessor_name = "dst_tensor"}},
        .runtime_arguments_schema = {.named_runtime_args = {"num_tiles"}},
        .config_spec =
            experimental::metal2_host_api::DataMovementConfiguration{
                .gen2_data_movement_config =
                    experimental::metal2_host_api::DataMovementConfiguration::Gen2DataMovementConfig{}},
    };

    experimental::metal2_host_api::KernelSpec::CompilerOptions::Defines compute_defines;
    if (test_config.short_init) {
        compute_defines.emplace_back("SHORT_INIT", "1");
    }

    experimental::metal2_host_api::KernelSpec compute_spec{
        .unique_id = COMPUTE,
        .source =
            experimental::metal2_host_api::KernelSpec::SourceFilePath{
                "tests/tt_metal/tt_metal/test_kernels/compute/transpose_wh.cpp"},
        .num_threads = 1,
        .compiler_options = {.defines = compute_defines},
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
        .compile_time_arg_bindings = {{"NHtWt", Ht * Wt * NC}},
        .config_spec = experimental::metal2_host_api::ComputeConfiguration{},
    };

    experimental::metal2_host_api::WorkUnitSpec wu{
        .unique_id = "main",
        .kernels = {READER, WRITER, COMPUTE},
        .target_nodes = node,
    };

    experimental::metal2_host_api::ProgramSpec spec{
        .program_id = "transpose_wh",
        .kernels = {reader_spec, writer_spec, compute_spec},
        .dataflow_buffers = {input_dfb_spec, output_dfb_spec},
        .tensor_parameters =
            {
                {.unique_id = IN_TENSOR, .spec = in_tensor.tensor_spec()},
                {.unique_id = OUT_TENSOR, .spec = out_tensor.tensor_spec()},
            },
        .work_units = {wu},
    };

    Program program = experimental::metal2_host_api::MakeProgramFromSpec(*mesh_device, spec);

    experimental::metal2_host_api::ProgramRunParams params;
    params.kernel_run_params = {
        experimental::metal2_host_api::ProgramRunParams::KernelRunParams{
            .kernel_spec_name = READER,
            .named_runtime_args = {{.node = node, .args = {{"N", NC}, {"Ht", Ht}, {"Wt", Wt}, {"HtWt", Ht * Wt}}}},
        },
        experimental::metal2_host_api::ProgramRunParams::KernelRunParams{
            .kernel_spec_name = WRITER,
            .named_runtime_args = {{.node = node, .args = {{"num_tiles", num_tensor_tiles}}}},
        },
        experimental::metal2_host_api::ProgramRunParams::KernelRunParams{
            .kernel_spec_name = COMPUTE,
        },
    };
    params.tensor_args = {
        {.tensor_parameter_name = IN_TENSOR, .tensor = in_tensor},
        {.tensor_parameter_name = OUT_TENSOR, .tensor = out_tensor},
    };
    experimental::metal2_host_api::SetProgramRunParameters(program, params);

    vector<uint32_t> src_vec = create_random_vector_of_bfloat16(dram_buffer_size, 100.0f, 0x1234);
    tt_metal::detail::WriteToBuffer(*in_tensor.mesh_buffer().get_reference_buffer(), src_vec);

    tt_metal::detail::LaunchProgram(device, program, /*wait_until_cores_done=*/true);

    read_and_validate_transpose_result_quasar(out_tensor, src_vec, test_config.shape, dims);
}

void run_single_core_transpose_legacy(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device, const TransposeConfig& test_config) {
    auto& cq = mesh_device->mesh_command_queue();
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    distributed::MeshWorkload workload;
    Program program = tt_metal::CreateProgram();
    workload.add_program(device_range, std::move(program));
    auto& program_ = workload.get_programs().at(device_range);
    auto* device = mesh_device->get_devices()[0];

    CoreCoord core = {0, 0};

    const TransposeDims dims = compute_and_validate_transpose_dims(test_config.shape);
    const uint32_t NC = dims.NC, Wt = dims.Wt, Ht = dims.Ht, num_tensor_tiles = dims.num_tensor_tiles;

    uint32_t dram_buffer_size = test_config.single_tile_size * num_tensor_tiles;

    tt_metal::InterleavedBufferConfig dram_config{
        .device = device,
        .size = dram_buffer_size,
        .page_size = test_config.single_tile_size,
        .buffer_type = tt_metal::BufferType::DRAM};

    std::shared_ptr<tt_metal::Buffer> src_dram_buffer = CreateBuffer(dram_config);
    uint32_t dram_buffer_src_addr = src_dram_buffer->address();

    std::shared_ptr<tt_metal::Buffer> dst_dram_buffer = CreateBuffer(dram_config);
    uint32_t dram_buffer_dst_addr = dst_dram_buffer->address();

    uint32_t num_buffer_tiles = 32;
    uint32_t num_output_buffer_tiles = 32;

    uint32_t src0_cb_index = 0;
    tt_metal::CircularBufferConfig cb_src0_config =
        tt_metal::CircularBufferConfig(
            num_buffer_tiles * test_config.single_tile_size, {{src0_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(src0_cb_index, test_config.single_tile_size);
    tt_metal::CreateCircularBuffer(program_, core, cb_src0_config);

    uint32_t ouput_cb_index = tt::CBIndex::c_16;
    tt_metal::CircularBufferConfig cb_output_config =
        tt_metal::CircularBufferConfig(
            num_output_buffer_tiles * test_config.single_tile_size, {{ouput_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(ouput_cb_index, test_config.single_tile_size);
    tt_metal::CreateCircularBuffer(program_, core, cb_output_config);

    std::vector<uint32_t> reader_cta;
    reader_cta.push_back(static_cast<uint32_t>(tt::CBIndex::c_0));
    tt::tt_metal::TensorAccessorArgs(src_dram_buffer).append_to(reader_cta);
    std::vector<uint32_t> writer_cta;
    writer_cta.push_back(static_cast<uint32_t>(tt::CBIndex::c_16));
    tt::tt_metal::TensorAccessorArgs(dst_dram_buffer).append_to(writer_cta);

    vector<uint32_t> compute_kernel_args = {uint(Ht * Wt * NC)};

    std::map<std::string, std::string> defines = {};

    if (test_config.short_init) {
        defines["SHORT_INIT"] = "1";
    }

    auto unary_reader_kernel = tt_metal::CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_unary_transpose_wh_8bank.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt_metal::NOC::RISCV_1_default,
            .compile_args = reader_cta});

    auto unary_writer_kernel = tt_metal::CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary_8bank.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .compile_args = writer_cta});

    tt_metal::CreateKernel(
        program_,
        test_config.transpose_dest ? "tests/tt_metal/tt_metal/test_kernels/compute/transpose_wh_dest.cpp"
                                   : "tests/tt_metal/tt_metal/test_kernels/compute/transpose_wh.cpp",
        core,
        tt_metal::ComputeConfig{.compile_args = compute_kernel_args, .defines = defines});

    tt_metal::SetRuntimeArgs(
        program_,
        unary_reader_kernel,
        core,
        {
            dram_buffer_src_addr,
            (uint32_t)0,  // unused to maintain compat
            (uint32_t)0,  // unused to maintain compat
            num_tensor_tiles,
            NC,
            Ht,
            Wt,
            Ht * Wt,
            (uint32_t)0  // unused scaler slot kept for compat
        });

    tt_metal::SetRuntimeArgs(
        program_,
        unary_writer_kernel,
        core,
        {dram_buffer_dst_addr,
         (uint32_t)0,  // unused to maintain compat
         num_tensor_tiles});

    vector<uint32_t> src_vec = create_random_vector_of_bfloat16(dram_buffer_size, 100.0f, 0x1234);
    tt_metal::detail::WriteToBuffer(src_dram_buffer, src_vec);

    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);

    read_and_validate_transpose_result(dst_dram_buffer, src_vec, test_config.shape, dims);
}

void run_single_core_transpose(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device, const TransposeConfig& test_config) {
    if (MetalContext::instance().get_cluster().arch() == ARCH::QUASAR) {
        run_single_core_transpose_quasar(mesh_device, test_config);
    } else {
        run_single_core_transpose_legacy(mesh_device, test_config);
    }
}

}  // namespace unit_tests::compute::transpose

TEST_F(LLKMeshDeviceFixture, TensixComputeTransposeWH) {
    unit_tests::compute::transpose::TransposeConfig test_config = {
        .short_init = false,
        .transpose_dest = false,
        .single_tile_size = 2 * 1024,
        .shape = {1, 3, 3 * 32 * 1, 4 * 32 * 1},
        .transpose_type = unit_tests::compute::transpose::TransposeType::WH};
    unit_tests::compute::transpose::run_single_core_transpose(this->devices_.at(0), test_config);
}

TEST_F(LLKMeshDeviceFixture, TensixComputeTransposeWHShortInit) {
    unit_tests::compute::transpose::TransposeConfig test_config = {
        .short_init = true,
        .transpose_dest = false,
        .single_tile_size = 2 * 1024,
        .shape = {1, 3, 3 * 32 * 1, 4 * 32 * 1},
        .transpose_type = unit_tests::compute::transpose::TransposeType::WH};
    unit_tests::compute::transpose::run_single_core_transpose(this->devices_.at(0), test_config);
}

TEST_F(LLKMeshDeviceFixture, TensixComputeTransposeWHDest) {
    if (MetalContext::instance().get_cluster().arch() == ARCH::QUASAR) {
        GTEST_SKIP() << "TensixComputeTransposeWHDest not implemented for Quasar yet";
    }
    unit_tests::compute::transpose::TransposeConfig test_config = {
        .short_init = false,
        .transpose_dest = true,
        .single_tile_size = 2 * 1024,
        .shape = {1, 3, 3 * 32 * 1, 4 * 32 * 1},
        .transpose_type = unit_tests::compute::transpose::TransposeType::WH};
    unit_tests::compute::transpose::run_single_core_transpose(this->devices_.at(0), test_config);
}

}  // namespace tt::tt_metal
