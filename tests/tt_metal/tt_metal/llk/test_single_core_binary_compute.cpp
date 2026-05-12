// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt_stl/reflection.hpp>
#include <chrono>
#include <fmt/base.h>
#include <gtest/gtest.h>
#include <cstddef>
#include <cstdint>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <algorithm>
#include <bit>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <variant>
#include <vector>

#include <tt_stl/assert.hpp>
#include <tt-metalium/base_types.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/kernel_types.hpp>
#include "llk_device_fixture.hpp"
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/program.hpp>
#include <tt_stl/span.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include "tt_metal/test_utils/comparison.hpp"
#include "tt_metal/test_utils/df/float32.hpp"
#include "tt_metal/test_utils/env_vars.hpp"
#include "tt_metal/test_utils/packing.hpp"
#include "tt_metal/test_utils/stimulus.hpp"
#include <umd/device/types/arch.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/distributed.hpp>

namespace tt::tt_metal {
class IDevice;
}  // namespace tt::tt_metal

namespace tt::tt_metal {

using std::map;
using std::vector;
using namespace tt;
using namespace tt::test_utils;
using namespace tt::test_utils::df;

namespace unit_tests::compute::binary {
const map<std::string, std::string> binary_op_name_to_op_type = {
    {"add", "EltwiseBinaryType::ELWADD"},
    {"sub", "EltwiseBinaryType::ELWSUB"},
    {"mul", "EltwiseBinaryType::ELWMUL"},
    {"add_with_dest_reuse", "EltwiseBinaryType::ELWADD"},
    {"sub_with_dest_reuse", "EltwiseBinaryType::ELWSUB"},
    {"mul_with_dest_reuse", "EltwiseBinaryType::ELWMUL"},
};
const map<std::string, std::string> binary_op_name_to_op_kernel = {
    {"add", "add_tiles"},
    {"sub", "sub_tiles"},
    {"mul", "mul_tiles"},
};

struct SingleCoreBinaryConfig {
    size_t num_tiles = 0;
    size_t tile_byte_size = 0;
    size_t input_dram_byte_address = 0;
    tt::DataFormat l1_input_data_format = tt::DataFormat::Invalid;
    tt::DataFormat l1_output_data_format = tt::DataFormat::Invalid;
    CoreCoord core;
    std::string binary_op;
    bool acc_to_dest = false;
    bool full_init = true;
    MathFidelity math_fidelity = MathFidelity::HiFi4;
    tt::tt_metal::Tile tile = tt::tt_metal::Tile({32, 32});
};

void set_math_fid_masks(
    uint16_t& srca_fid_mask, uint16_t& srcb_fid_mask, MathFidelity math_fidelity = MathFidelity::HiFi4) {
    switch (math_fidelity) {
        case MathFidelity::HiFi4:
        case MathFidelity::HiFi3: {
            break;
        }
        case MathFidelity::HiFi2: {
            srcb_fid_mask = 0xFFFE;
            ;
            break;
        }
        case MathFidelity::LoFi: {
            srca_fid_mask = 0xFFF8;
            srcb_fid_mask = 0xFFFE;
            break;
        }
        default: {
            TT_THROW("Unsupported MathFidelity={}", math_fidelity);
            break;
        }
    }
}

// --- Stimulus and golden helpers (shared between QUASAR and legacy paths) ---

struct BinaryStimulus {
    std::vector<uint32_t> packed_input0;
    std::vector<uint32_t> packed_input1;
    std::vector<uint32_t> packed_input2;
    std::vector<uint32_t> packed_golden;
};

static BinaryStimulus generate_binary_stimulus(const SingleCoreBinaryConfig& test_config, bool is_quasar) {
    const size_t byte_size = test_config.num_tiles * test_config.tile_byte_size;
    BinaryStimulus s;
    s.packed_input0 = generate_packed_uniform_random_vector<uint32_t, bfloat16>(
        -1.0f, 1.0f, byte_size / sizeof(bfloat16), std::chrono::system_clock::now().time_since_epoch().count());
    s.packed_input1 = generate_packed_uniform_random_vector<uint32_t, bfloat16>(
        -1.0f, 1.0f, byte_size / sizeof(bfloat16), std::chrono::system_clock::now().time_since_epoch().count());
    s.packed_input2 = generate_packed_uniform_random_vector<uint32_t, bfloat16>(
        -1.0f, 1.0f, byte_size / sizeof(bfloat16), std::chrono::system_clock::now().time_since_epoch().count());

    auto input0 = unpack_vector<bfloat16, uint32_t>(s.packed_input0);
    auto input1 = unpack_vector<bfloat16, uint32_t>(s.packed_input1);
    auto input2 = unpack_vector<bfloat16, uint32_t>(s.packed_input2);

    std::vector<float> temp_golden(input0.size());
    uint16_t srca_fid_mask = 0xFFFF;
    uint16_t srcb_fid_mask = 0xFFFF;
    if (!is_quasar) {
        set_math_fid_masks(srca_fid_mask, srcb_fid_mask, test_config.math_fidelity);
    }

    std::transform(
        input0.begin(),
        input0.end(),
        input1.begin(),
        temp_golden.begin(),
        [&](const bfloat16& lhs, const bfloat16& rhs) {
            if (test_config.binary_op == "add") {
                return (static_cast<float>(lhs) + static_cast<float>(rhs));
            }
            if (test_config.binary_op == "sub") {
                return (static_cast<float>(lhs) - static_cast<float>(rhs));
            }
            if (test_config.binary_op == "mul") {
                return (
                    static_cast<float>(
                        std::bit_cast<bfloat16>(static_cast<uint16_t>(std::bit_cast<uint16_t>(lhs) & srca_fid_mask))) *
                    static_cast<float>(
                        std::bit_cast<bfloat16>(static_cast<uint16_t>(std::bit_cast<uint16_t>(rhs) & srcb_fid_mask))));
            }
            if (test_config.binary_op.find("with_dest_reuse") != std::string::npos) {
                return static_cast<float>(lhs);
            }
            TT_THROW("Unsupported binary_op={}", test_config.binary_op);
        });

    std::vector<bfloat16> golden(input0.size());
    std::transform(
        input2.begin(), input2.end(), temp_golden.begin(), golden.begin(), [&](const bfloat16& lhs, const float& rhs) {
            if (test_config.acc_to_dest || test_config.binary_op == "add_with_dest_reuse") {
                return (static_cast<float>(lhs) + rhs);
            }
            if (test_config.binary_op == "sub_with_dest_reuse") {
                return (static_cast<float>(lhs) - rhs);
            }
            if (test_config.binary_op == "mul_with_dest_reuse") {
                return (
                    static_cast<float>(
                        std::bit_cast<bfloat16>(static_cast<uint16_t>(std::bit_cast<uint16_t>(lhs) & srca_fid_mask))) *
                    static_cast<float>(std::bit_cast<bfloat16>(
                        static_cast<uint16_t>(std::bit_cast<uint16_t>(bfloat16(rhs)) & srcb_fid_mask))));
            }
            return rhs;
        });
    s.packed_golden = pack_vector<uint32_t, bfloat16>(golden);
    return s;
}

// Four DRAM buffers (3 inputs + 1 output) shared between QUASAR and legacy paths.
struct BinaryBuffers {
    std::shared_ptr<distributed::MeshBuffer> input0;
    std::shared_ptr<distributed::MeshBuffer> input1;
    std::shared_ptr<distributed::MeshBuffer> input2;
    std::shared_ptr<distributed::MeshBuffer> output;
};

static BinaryBuffers create_and_populate_binary_buffers(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    distributed::MeshCommandQueue& cq,
    const distributed::MeshCoordinate& zero_coord,
    size_t byte_size,
    BinaryStimulus& stimulus) {
    distributed::DeviceLocalBufferConfig dram_config{
        .page_size = byte_size, .buffer_type = tt::tt_metal::BufferType::DRAM, .bottom_up = false};
    distributed::ReplicatedBufferConfig buffer_config{.size = byte_size};

    BinaryBuffers buffers;
    buffers.input0 = distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());
    buffers.input1 = distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());
    buffers.input2 = distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());
    buffers.output = distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());

    distributed::WriteShard(cq, buffers.input0, stimulus.packed_input0, zero_coord, false);
    distributed::WriteShard(cq, buffers.input1, stimulus.packed_input1, zero_coord, false);
    distributed::WriteShard(cq, buffers.input2, stimulus.packed_input2, zero_coord, false);

    return buffers;
}

static bool read_and_validate_binary_result(
    distributed::MeshCommandQueue& cq,
    const std::shared_ptr<distributed::MeshBuffer>& output_dram_buffer,
    const distributed::MeshCoordinate& zero_coord,
    const BinaryStimulus& stimulus) {
    std::vector<uint32_t> dest_buffer_data;
    distributed::ReadShard(cq, dest_buffer_data, output_dram_buffer, zero_coord, false);

    return is_close_packed_vectors<bfloat16, uint32_t>(
        dest_buffer_data, stimulus.packed_golden, [&](const bfloat16& a, const bfloat16& b) {
            return is_close(a, b, 0.0155f);
        });
}

static std::map<std::string, std::string> build_binary_defines(const SingleCoreBinaryConfig& test_config) {
    std::map<std::string, std::string> defines = {
        {"ELTWISE_OP_TYPE", binary_op_name_to_op_type.at(test_config.binary_op)}};
    if (test_config.binary_op.find("_with_dest_reuse") != std::string::npos) {
        defines["ELTWISE_DEST_REUSE_TYPE"] = "EltwiseBinaryReuseDestType::DEST_TO_SRCA";
    } else {
        defines["ELTWISE_OP"] = binary_op_name_to_op_kernel.at(test_config.binary_op);
        if (test_config.full_init) {
            defines["FULL_INIT"] = "1";
        }
        if (test_config.acc_to_dest) {
            defines["LOAD_BUF2_DATA"] = "1";
            defines["ACC_TO_DEST"] = "1";
        }
        defines["ELTWISE_OP_INIT"] = defines["ELTWISE_OP"] + "_init";
        if (test_config.binary_op == "mul") {
            defines["MUL_TILES_WITH_DST_ACCUM"] = "1";
        }
    }
    return defines;
}

static bool single_core_binary_quasar(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device, const SingleCoreBinaryConfig& test_config) {
    const size_t byte_size = test_config.num_tiles * test_config.tile_byte_size;
    auto& cq = mesh_device->mesh_command_queue();
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    const experimental::metal2_host_api::NodeCoord node{
        static_cast<uint32_t>(test_config.core.x), static_cast<uint32_t>(test_config.core.y)};

    auto stimulus = generate_binary_stimulus(test_config, /*is_quasar=*/true);
    auto buffers = create_and_populate_binary_buffers(mesh_device, cq, zero_coord, byte_size, stimulus);
    auto& input0_dram_buffer = buffers.input0;
    auto& input1_dram_buffer = buffers.input1;
    auto& input2_dram_buffer = buffers.input2;
    auto& output_dram_buffer = buffers.output;

    auto defines_map = build_binary_defines(test_config);
    experimental::metal2_host_api::KernelSpec::CompilerOptions::Defines defines;
    defines.reserve(defines_map.size());
    for (auto& kv : defines_map) {
        defines.emplace_back(kv.first, kv.second);
    }

    constexpr const char* INP0_DFB = "inp0_dfb";
    constexpr const char* INP1_DFB = "inp1_dfb";
    constexpr const char* INP2_DFB = "inp2_dfb";
    constexpr const char* OUT_DFB = "out_dfb";
    constexpr const char* READER = "reader";
    constexpr const char* WRITER = "writer";
    constexpr const char* COMPUTE = "compute";

    auto make_input_dfb = [&](const std::string& name) {
        return experimental::metal2_host_api::DataflowBufferSpec{
            .unique_id = name,
            .entry_size = static_cast<uint32_t>(test_config.tile_byte_size),
            .num_entries = static_cast<uint32_t>(test_config.num_tiles),
            .data_format_metadata = test_config.l1_input_data_format,
            .tile_format_metadata = test_config.tile,
            // Match pre-migration behavior: legacy DataflowBufferConfig set enable_implicit_sync=false.
            .disable_implicit_sync = true,
        };
    };

    experimental::metal2_host_api::DataflowBufferSpec inp0_dfb_spec = make_input_dfb(INP0_DFB);
    experimental::metal2_host_api::DataflowBufferSpec inp1_dfb_spec = make_input_dfb(INP1_DFB);
    experimental::metal2_host_api::DataflowBufferSpec inp2_dfb_spec = make_input_dfb(INP2_DFB);
    experimental::metal2_host_api::DataflowBufferSpec out_dfb_spec{
        .unique_id = OUT_DFB,
        .entry_size = static_cast<uint32_t>(test_config.tile_byte_size),
        .num_entries = static_cast<uint32_t>(test_config.num_tiles),
        .data_format_metadata = test_config.l1_output_data_format,
        .tile_format_metadata = test_config.tile,
        // Match pre-migration behavior: legacy DataflowBufferConfig set enable_implicit_sync=false.
        .disable_implicit_sync = true,
    };

    experimental::metal2_host_api::KernelSpec reader_spec{
        .unique_id = READER,
        .source =
            experimental::metal2_host_api::KernelSpec::SourceFilePath{
                "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_binary.cpp"},
        .num_threads = 1,
        .compiler_options = {.defines = defines},
        .dfb_bindings =
            {{
                 .dfb_spec_name = INP0_DFB,
                 .local_accessor_name = "in0",
                 .endpoint_type = experimental::metal2_host_api::KernelSpec::DFBEndpointType::PRODUCER,
                 .access_pattern = experimental::metal2_host_api::DFBAccessPattern::STRIDED,
             },
             {
                 .dfb_spec_name = INP1_DFB,
                 .local_accessor_name = "in1",
                 .endpoint_type = experimental::metal2_host_api::KernelSpec::DFBEndpointType::PRODUCER,
                 .access_pattern = experimental::metal2_host_api::DFBAccessPattern::STRIDED,
             },
             {
                 .dfb_spec_name = INP2_DFB,
                 .local_accessor_name = "in2",
                 .endpoint_type = experimental::metal2_host_api::KernelSpec::DFBEndpointType::PRODUCER,
                 .access_pattern = experimental::metal2_host_api::DFBAccessPattern::STRIDED,
             }},
        .runtime_arguments_schema =
            {.named_runtime_args =
                 {"src0_addr", "src0_bank_id", "src1_addr", "src1_bank_id", "num_tiles", "src2_addr", "src2_bank_id"}},
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

    experimental::metal2_host_api::KernelSpec compute_spec{
        .unique_id = COMPUTE,
        .source =
            experimental::metal2_host_api::KernelSpec::SourceFilePath{"tt_metal/kernels/compute/eltwise_binary.cpp"},
        .num_threads = 1,
        .compiler_options = {.defines = defines},
        .dfb_bindings =
            {{
                 .dfb_spec_name = INP0_DFB,
                 .local_accessor_name = "in0",
                 .endpoint_type = experimental::metal2_host_api::KernelSpec::DFBEndpointType::CONSUMER,
                 .access_pattern = experimental::metal2_host_api::DFBAccessPattern::STRIDED,
             },
             {
                 .dfb_spec_name = INP1_DFB,
                 .local_accessor_name = "in1",
                 .endpoint_type = experimental::metal2_host_api::KernelSpec::DFBEndpointType::CONSUMER,
                 .access_pattern = experimental::metal2_host_api::DFBAccessPattern::STRIDED,
             },
             {
                 .dfb_spec_name = INP2_DFB,
                 .local_accessor_name = "in2",
                 .endpoint_type = experimental::metal2_host_api::KernelSpec::DFBEndpointType::CONSUMER,
                 .access_pattern = experimental::metal2_host_api::DFBAccessPattern::STRIDED,
             },
             {
                 .dfb_spec_name = OUT_DFB,
                 .local_accessor_name = "out",
                 .endpoint_type = experimental::metal2_host_api::KernelSpec::DFBEndpointType::PRODUCER,
                 .access_pattern = experimental::metal2_host_api::DFBAccessPattern::STRIDED,
             }},
        .runtime_arguments_schema = {.named_runtime_args = {"per_core_block_cnt", "per_core_block_size", "acc_to_dst"}},
        .config_spec =
            experimental::metal2_host_api::ComputeConfiguration{
                .math_fidelity = test_config.math_fidelity,
            },
    };

    experimental::metal2_host_api::WorkUnitSpec wu{
        .unique_id = "main",
        .kernels = {READER, WRITER, COMPUTE},
        .target_nodes = node,
    };

    experimental::metal2_host_api::ProgramSpec spec{
        .program_id = "single_core_binary",
        .kernels = {reader_spec, writer_spec, compute_spec},
        .dataflow_buffers = {inp0_dfb_spec, inp1_dfb_spec, inp2_dfb_spec, out_dfb_spec},
        .work_units = {wu},
    };

    Program program = experimental::metal2_host_api::MakeProgramFromSpec(*mesh_device, spec);

    const uint32_t num_tiles_u = static_cast<uint32_t>(test_config.num_tiles);
    experimental::metal2_host_api::ProgramRunParams params;
    params.kernel_run_params = {
        experimental::metal2_host_api::ProgramRunParams::KernelRunParams{
            .kernel_spec_name = READER,
            .named_runtime_args =
                {{.node = node,
                  .args =
                      {{"src0_addr", input0_dram_buffer->address()},
                       {"src0_bank_id", 0u},
                       {"src1_addr", input1_dram_buffer->address()},
                       {"src1_bank_id", 0u},
                       {"num_tiles", num_tiles_u},
                       {"src2_addr", input2_dram_buffer->address()},
                       {"src2_bank_id", 0u}}}},
        },
        experimental::metal2_host_api::ProgramRunParams::KernelRunParams{
            .kernel_spec_name = WRITER,
            .named_runtime_args =
                {{.node = node,
                  .args = {{"dst_addr", output_dram_buffer->address()}, {"bank_id", 0u}, {"num_tiles", num_tiles_u}}}},
        },
        experimental::metal2_host_api::ProgramRunParams::KernelRunParams{
            .kernel_spec_name = COMPUTE,
            .named_runtime_args =
                {{.node = node,
                  .args = {{"per_core_block_cnt", num_tiles_u}, {"per_core_block_size", 1u}, {"acc_to_dst", 0u}}}},
        },
    };
    experimental::metal2_host_api::SetProgramRunParameters(program, params);

    auto* dev = mesh_device->get_devices()[0];
    tt_metal::detail::LaunchProgram(dev, program, /*wait_until_cores_done=*/true);

    return read_and_validate_binary_result(cq, output_dram_buffer, zero_coord, stimulus);
}

static bool single_core_binary_legacy(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device, const SingleCoreBinaryConfig& test_config) {
    const size_t byte_size = test_config.num_tiles * test_config.tile_byte_size;

    auto& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    Program program = tt::tt_metal::CreateProgram();
    workload.add_program(device_range, std::move(program));
    auto& program_ = workload.get_programs().at(device_range);

    auto stimulus = generate_binary_stimulus(test_config, /*is_quasar=*/false);
    auto buffers = create_and_populate_binary_buffers(mesh_device, cq, zero_coord, byte_size, stimulus);
    auto& input0_dram_buffer = buffers.input0;
    auto& input1_dram_buffer = buffers.input1;
    auto& input2_dram_buffer = buffers.input2;
    auto& output_dram_buffer = buffers.output;

    tt_metal::CircularBufferConfig l1_input0_cb_config =
        tt_metal::CircularBufferConfig(byte_size, {{0, test_config.l1_input_data_format}})
            .set_page_size(0, test_config.tile_byte_size);
    tt_metal::CreateCircularBuffer(program_, test_config.core, l1_input0_cb_config);

    tt_metal::CircularBufferConfig l1_input1_cb_config =
        tt_metal::CircularBufferConfig(byte_size, {{1, test_config.l1_input_data_format}})
            .set_page_size(1, test_config.tile_byte_size);
    tt_metal::CreateCircularBuffer(program_, test_config.core, l1_input1_cb_config);

    tt_metal::CircularBufferConfig l1_input2_cb_config =
        tt_metal::CircularBufferConfig(byte_size, {{2, test_config.l1_input_data_format}})
            .set_page_size(2, test_config.tile_byte_size);
    tt_metal::CreateCircularBuffer(program_, test_config.core, l1_input2_cb_config);

    tt_metal::CircularBufferConfig l1_output_cb_config =
        tt_metal::CircularBufferConfig(byte_size, {{16, test_config.l1_output_data_format}})
            .set_page_size(16, test_config.tile_byte_size);
    tt_metal::CreateCircularBuffer(program_, test_config.core, l1_output_cb_config);

    auto defines = build_binary_defines(test_config);
    std::vector<uint32_t> compute_cta;

    auto reader_kernel = tt_metal::CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_binary.cpp",
        test_config.core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt_metal::NOC::RISCV_1_default,
            .defines = defines});

    auto writer_kernel = tt_metal::CreateKernel(
        program_,
        "tt_metal/kernels/dataflow/writer_unary.cpp",
        test_config.core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

    auto binary_kernel = tt_metal::CreateKernel(
        program_,
        "tt_metal/kernels/compute/eltwise_binary.cpp",
        test_config.core,
        tt_metal::ComputeConfig{
            .math_fidelity = test_config.math_fidelity, .compile_args = compute_cta, .defines = defines});

    SetRuntimeArgs(program_, binary_kernel, test_config.core, {uint32_t(test_config.num_tiles), 1});

    tt_metal::SetRuntimeArgs(
        program_,
        reader_kernel,
        test_config.core,
        {
            (uint32_t)input0_dram_buffer->address(),
            (uint32_t)0,
            (uint32_t)input1_dram_buffer->address(),
            (uint32_t)0,
            (uint32_t)test_config.num_tiles,
            (uint32_t)input2_dram_buffer->address(),
            (uint32_t)0,
        });
    tt_metal::SetRuntimeArgs(
        program_,
        writer_kernel,
        test_config.core,
        {
            (uint32_t)output_dram_buffer->address(),
            (uint32_t)0,
            (uint32_t)test_config.num_tiles,
        });

    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);

    return read_and_validate_binary_result(cq, output_dram_buffer, zero_coord, stimulus);
}

/// @brief Does Dramx2 --> Reader --> CB --> Binary Compute --> CB --> Writer --> Dram
/// @param device
/// @param test_config - Configuration of the test -- see struct
/// @return
bool single_core_binary(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device, const SingleCoreBinaryConfig& test_config) {
    if (MetalContext::instance().get_cluster().arch() == ARCH::QUASAR) {
        return single_core_binary_quasar(mesh_device, test_config);
    }
    return single_core_binary_legacy(mesh_device, test_config);
}
}  // namespace unit_tests::compute::binary

TEST_F(LLKMeshDeviceFixtureSlowDispatchOnly, TensixBinaryComputeSingleCoreSingleTileAdd) {
    for (uint8_t i = uint8_t(MathFidelity::LoFi); i <= uint8_t(MathFidelity::HiFi4); i++) {
        if (i == 1) {
            continue;
        }
        unit_tests::compute::binary::SingleCoreBinaryConfig test_config = {
            .tile_byte_size = 2 * 32 * 32,
            .l1_input_data_format = tt::DataFormat::Float16_b,
            .l1_output_data_format = tt::DataFormat::Float16_b,
            .core = CoreCoord(0, 0),
            .binary_op = "add",
            .math_fidelity = MathFidelity(i)};
        test_config.num_tiles = 1;
        log_info(tt::LogTest, "Math Fidelity = {}", i);
        for (unsigned int id = 0; id < num_devices_; id++) {
            ASSERT_TRUE(unit_tests::compute::binary::single_core_binary(devices_.at(id), test_config));
            // TODO: Remove early return once back-to-back tests are passing on Quasar
            if (this->arch_ == ARCH::QUASAR) {
                return;
            }
        }
    }
}

TEST_F(LLKMeshDeviceFixtureSlowDispatchOnly, TensixBinaryComputeSingleCoreSingleTileSub) {
    for (uint8_t i = uint8_t(MathFidelity::LoFi); i <= uint8_t(MathFidelity::HiFi4); i++) {
        if (i == 1) {
            continue;
        }
        unit_tests::compute::binary::SingleCoreBinaryConfig test_config = {
            .tile_byte_size = 2 * 32 * 32,
            .l1_input_data_format = tt::DataFormat::Float16_b,
            .l1_output_data_format = tt::DataFormat::Float16_b,
            .core = CoreCoord(0, 0),
            .binary_op = "sub",
            .math_fidelity = MathFidelity(i)};
        test_config.num_tiles = 1;
        log_info(tt::LogTest, "Math Fidelity = {}", i);
        for (unsigned int id = 0; id < num_devices_; id++) {
            ASSERT_TRUE(unit_tests::compute::binary::single_core_binary(devices_.at(id), test_config));
            // TODO: Remove early return once back-to-back tests are passing on Quasar
            if (this->arch_ == ARCH::QUASAR) {
                return;
            }
        }
    }
}

TEST_F(LLKMeshDeviceFixtureSlowDispatchOnly, TensixBinaryComputeSingleCoreSingleTileMul) {
    for (uint8_t i = uint8_t(MathFidelity::LoFi); i <= uint8_t(MathFidelity::HiFi4); i++) {
        if (i == 1) {
            continue;
        }
        unit_tests::compute::binary::SingleCoreBinaryConfig test_config = {
            .tile_byte_size = 2 * 32 * 32,
            .l1_input_data_format = tt::DataFormat::Float16_b,
            .l1_output_data_format = tt::DataFormat::Float16_b,
            .core = CoreCoord(0, 0),
            .binary_op = "mul",
            .math_fidelity = MathFidelity(i)};
        test_config.num_tiles = 1;
        log_info(tt::LogTest, "Math Fidelity = {}", i);
        for (unsigned int id = 0; id < num_devices_; id++) {
            ASSERT_TRUE(unit_tests::compute::binary::single_core_binary(devices_.at(id), test_config));
            // TODO: Remove early return once back-to-back tests are passing on Quasar
            if (this->arch_ == ARCH::QUASAR) {
                return;
            }
        }
    }
}

TEST_F(LLKMeshDeviceFixtureSlowDispatchOnly, TensixBinaryComputeSingleCoreSingleTileAddFullInit) {
    for (uint8_t i = uint8_t(MathFidelity::LoFi); i <= uint8_t(MathFidelity::HiFi4); i++) {
        if (i == 1) {
            continue;
        }
        unit_tests::compute::binary::SingleCoreBinaryConfig test_config = {
            .tile_byte_size = 2 * 32 * 32,
            .l1_input_data_format = tt::DataFormat::Float16_b,
            .l1_output_data_format = tt::DataFormat::Float16_b,
            .core = CoreCoord(0, 0),
            .binary_op = "add",
            .full_init = true,
            .math_fidelity = MathFidelity(i)};
        test_config.num_tiles = 1;
        log_info(tt::LogTest, "Math Fidelity = {}", i);
        for (unsigned int id = 0; id < num_devices_; id++) {
            ASSERT_TRUE(unit_tests::compute::binary::single_core_binary(devices_.at(id), test_config));
            // TODO: Remove early return once back-to-back tests are passing on Quasar
            if (this->arch_ == ARCH::QUASAR) {
                return;
            }
        }
    }
}

TEST_F(LLKMeshDeviceFixtureSlowDispatchOnly, TensixBinaryComputeSingleCoreSingleTileSubFullInit) {
    for (uint8_t i = uint8_t(MathFidelity::LoFi); i <= uint8_t(MathFidelity::HiFi4); i++) {
        if (i == 1) {
            continue;
        }
        unit_tests::compute::binary::SingleCoreBinaryConfig test_config = {
            .tile_byte_size = 2 * 32 * 32,
            .l1_input_data_format = tt::DataFormat::Float16_b,
            .l1_output_data_format = tt::DataFormat::Float16_b,
            .core = CoreCoord(0, 0),
            .binary_op = "sub",
            .full_init = true,
            .math_fidelity = MathFidelity(i)};
        test_config.num_tiles = 1;
        log_info(tt::LogTest, "Math Fidelity = {}", i);
        for (unsigned int id = 0; id < num_devices_; id++) {
            ASSERT_TRUE(unit_tests::compute::binary::single_core_binary(devices_.at(id), test_config));
            // TODO: Remove early return once back-to-back tests are passing on Quasar
            if (this->arch_ == ARCH::QUASAR) {
                return;
            }
        }
    }
}

TEST_F(LLKMeshDeviceFixtureSlowDispatchOnly, TensixBinaryComputeSingleCoreSingleTileMulFullInit) {
    for (uint8_t i = uint8_t(MathFidelity::LoFi); i <= uint8_t(MathFidelity::HiFi4); i++) {
        if (i == 1) {
            continue;
        }
        unit_tests::compute::binary::SingleCoreBinaryConfig test_config = {
            .tile_byte_size = 2 * 32 * 32,
            .l1_input_data_format = tt::DataFormat::Float16_b,
            .l1_output_data_format = tt::DataFormat::Float16_b,
            .core = CoreCoord(0, 0),
            .binary_op = "mul",
            .full_init = true,
            .math_fidelity = MathFidelity(i)};
        test_config.num_tiles = 1;
        log_info(tt::LogTest, "Math Fidelity = {}", i);
        for (unsigned int id = 0; id < num_devices_; id++) {
            ASSERT_TRUE(unit_tests::compute::binary::single_core_binary(devices_.at(id), test_config));
            // TODO: Remove early return once back-to-back tests are passing on Quasar
            if (this->arch_ == ARCH::QUASAR) {
                return;
            }
        }
    }
}

TEST_F(LLKMeshDeviceFixtureSlowDispatchOnly, TensixBinaryComputeSingleCoreMultiTileAddWithDestReuse) {
    if (this->arch_ == tt::ARCH::QUASAR) {
        GTEST_SKIP() << "DestReuse test support will be added with DestReuse bring-up";
    }
    for (uint8_t i = uint8_t(MathFidelity::LoFi); i <= uint8_t(MathFidelity::HiFi4); i++) {
        if (i == 1) {
            continue;
        }
        unit_tests::compute::binary::SingleCoreBinaryConfig test_config = {
            .tile_byte_size = 2 * 32 * 32,
            .l1_input_data_format = tt::DataFormat::Float16_b,
            .l1_output_data_format = tt::DataFormat::Float16_b,
            .core = CoreCoord(0, 0),
            .binary_op = "add_with_dest_reuse",
            .math_fidelity = MathFidelity(i)};
        test_config.num_tiles = 4;
        log_info(tt::LogTest, "Math Fidelity = {}", i);
        for (unsigned int id = 0; id < num_devices_; id++) {
            ASSERT_TRUE(unit_tests::compute::binary::single_core_binary(devices_.at(id), test_config));
            // TODO: Remove early return once back-to-back tests are passing on Quasar
            if (this->arch_ == ARCH::QUASAR) {
                return;
            }
        }
    }
}

TEST_F(LLKMeshDeviceFixtureSlowDispatchOnly, TensixBinaryComputeSingleCoreMultiTileSubWithDestReuse) {
    if (this->arch_ == tt::ARCH::QUASAR) {
        GTEST_SKIP() << "DestReuse test support will be added with DestReuse bring-up";
    }
    for (uint8_t i = uint8_t(MathFidelity::LoFi); i <= uint8_t(MathFidelity::HiFi4); i++) {
        if (i == 1) {
            continue;
        }
        unit_tests::compute::binary::SingleCoreBinaryConfig test_config = {
            .tile_byte_size = 2 * 32 * 32,
            .l1_input_data_format = tt::DataFormat::Float16_b,
            .l1_output_data_format = tt::DataFormat::Float16_b,
            .core = CoreCoord(0, 0),
            .binary_op = "sub_with_dest_reuse",
            .math_fidelity = MathFidelity(i)};
        test_config.num_tiles = 4;
        log_info(tt::LogTest, "Math Fidelity = {}", i);
        for (unsigned int id = 0; id < num_devices_; id++) {
            ASSERT_TRUE(unit_tests::compute::binary::single_core_binary(devices_.at(id), test_config));
            // TODO: Remove early return once back-to-back tests are passing on Quasar
            if (this->arch_ == ARCH::QUASAR) {
                return;
            }
        }
    }
}

TEST_F(LLKMeshDeviceFixtureSlowDispatchOnly, TensixBinaryComputeSingleCoreMultiTileMulWithDestReuse) {
    if (this->arch_ == tt::ARCH::QUASAR) {
        GTEST_SKIP() << "DestReuse test support will be added with DestReuse bring-up";
    }
    for (uint8_t i = uint8_t(MathFidelity::LoFi); i <= uint8_t(MathFidelity::HiFi4); i++) {
        if (i == 1) {
            continue;
        }
        unit_tests::compute::binary::SingleCoreBinaryConfig test_config = {
            .tile_byte_size = 2 * 32 * 32,
            .l1_input_data_format = tt::DataFormat::Float16_b,
            .l1_output_data_format = tt::DataFormat::Float16_b,
            .core = CoreCoord(0, 0),
            .binary_op = "mul_with_dest_reuse",
            .math_fidelity = MathFidelity(i)};
        test_config.num_tiles = 4;
        log_info(tt::LogTest, "Math Fidelity = {}", i);
        for (unsigned int id = 0; id < num_devices_; id++) {
            ASSERT_TRUE(unit_tests::compute::binary::single_core_binary(devices_.at(id), test_config));
            // TODO: Remove early return once back-to-back tests are passing on Quasar
            if (this->arch_ == ARCH::QUASAR) {
                return;
            }
        }
    }
}

TEST_F(LLKMeshDeviceFixtureSlowDispatchOnly, TensixBinaryComputeSingleCoreMultiTileAdd) {
    for (uint8_t i = uint8_t(MathFidelity::LoFi); i <= uint8_t(MathFidelity::HiFi4); i++) {
        if (i == 1) {
            continue;
        }
        unit_tests::compute::binary::SingleCoreBinaryConfig test_config = {
            .tile_byte_size = 2 * 32 * 32,
            .l1_input_data_format = tt::DataFormat::Float16_b,
            .l1_output_data_format = tt::DataFormat::Float16_b,
            .core = CoreCoord(0, 0),
            .binary_op = "add",
            .math_fidelity = MathFidelity(i)};
        test_config.num_tiles = 4;
        log_info(tt::LogTest, "Math Fidelity = {}", i);
        for (unsigned int id = 0; id < num_devices_; id++) {
            ASSERT_TRUE(unit_tests::compute::binary::single_core_binary(devices_.at(id), test_config));
            // TODO: Remove early return once back-to-back tests are passing on Quasar
            if (this->arch_ == ARCH::QUASAR) {
                return;
            }
        }
    }
}

TEST_F(LLKMeshDeviceFixtureSlowDispatchOnly, TensixBinaryComputeSingleCoreMultiTileSub) {
    for (uint8_t i = uint8_t(MathFidelity::LoFi); i <= uint8_t(MathFidelity::HiFi4); i++) {
        if (i == 1) {
            continue;
        }
        unit_tests::compute::binary::SingleCoreBinaryConfig test_config = {
            .tile_byte_size = 2 * 32 * 32,
            .l1_input_data_format = tt::DataFormat::Float16_b,
            .l1_output_data_format = tt::DataFormat::Float16_b,
            .core = CoreCoord(0, 0),
            .binary_op = "sub",
            .math_fidelity = MathFidelity(i)};
        test_config.num_tiles = 4;
        log_info(tt::LogTest, "Math Fidelity = {}", i);
        for (unsigned int id = 0; id < num_devices_; id++) {
            ASSERT_TRUE(unit_tests::compute::binary::single_core_binary(devices_.at(id), test_config));
            // TODO: Remove early return once back-to-back tests are passing on Quasar
            if (this->arch_ == ARCH::QUASAR) {
                return;
            }
        }
    }
}

TEST_F(LLKMeshDeviceFixtureSlowDispatchOnly, TensixBinaryComputeSingleCoreMultiTileMul) {
    for (uint8_t i = uint8_t(MathFidelity::LoFi); i <= uint8_t(MathFidelity::HiFi4); i++) {
        if (i == 1) {
            continue;
        }
        unit_tests::compute::binary::SingleCoreBinaryConfig test_config = {
            .tile_byte_size = 2 * 32 * 32,
            .l1_input_data_format = tt::DataFormat::Float16_b,
            .l1_output_data_format = tt::DataFormat::Float16_b,
            .core = CoreCoord(0, 0),
            .binary_op = "mul",
            .math_fidelity = MathFidelity(i)};
        test_config.num_tiles = 4;
        log_info(tt::LogTest, "Math Fidelity = {}", i);
        for (unsigned int id = 0; id < num_devices_; id++) {
            ASSERT_TRUE(unit_tests::compute::binary::single_core_binary(devices_.at(id), test_config));
            // TODO: Remove early return once back-to-back tests are passing on Quasar
            if (this->arch_ == ARCH::QUASAR) {
                return;
            }
        }
    }
}

TEST_F(LLKMeshDeviceFixtureSlowDispatchOnly, TensixBinaryComputeSingleCoreMultiTileAddDestAcc) {
    if (this->arch_ == tt::ARCH::QUASAR) {
        GTEST_SKIP() << "DestAcc test support will be added with DestReuse bring-up";
    }
    for (uint8_t i = uint8_t(MathFidelity::LoFi); i <= uint8_t(MathFidelity::HiFi4); i++) {
        if (i == 1) {
            continue;
        }
        unit_tests::compute::binary::SingleCoreBinaryConfig test_config = {
            .num_tiles = 4,
            .tile_byte_size = 2 * 32 * 32,
            .l1_input_data_format = tt::DataFormat::Float16_b,
            .l1_output_data_format = tt::DataFormat::Float16_b,
            .core = CoreCoord(0, 0),
            .binary_op = "add",
            .acc_to_dest = true,
            .math_fidelity = MathFidelity(i),
        };
        log_info(tt::LogTest, "Math Fidelity = {}", i);
        for (unsigned int id = 0; id < num_devices_; id++) {
            ASSERT_TRUE(unit_tests::compute::binary::single_core_binary(devices_.at(id), test_config));
            // TODO: Remove early return once back-to-back tests are passing on Quasar
            if (this->arch_ == ARCH::QUASAR) {
                return;
            }
        }
    }
}

TEST_F(LLKMeshDeviceFixtureSlowDispatchOnly, TensixBinaryComputeSingleCoreMultiTileSubDestAcc) {
    if (this->arch_ == tt::ARCH::QUASAR) {
        GTEST_SKIP() << "DestAcc test support will be added with DestReuse bring-up";
    }
    for (uint8_t i = uint8_t(MathFidelity::LoFi); i <= uint8_t(MathFidelity::HiFi4); i++) {
        if (i == 1) {
            continue;
        }
        unit_tests::compute::binary::SingleCoreBinaryConfig test_config = {
            .num_tiles = 4,
            .tile_byte_size = 2 * 32 * 32,
            .l1_input_data_format = tt::DataFormat::Float16_b,
            .l1_output_data_format = tt::DataFormat::Float16_b,
            .core = CoreCoord(0, 0),
            .binary_op = "sub",
            .acc_to_dest = true,
            .math_fidelity = MathFidelity(i),
        };
        log_info(tt::LogTest, "Math Fidelity = {}", i);
        for (unsigned int id = 0; id < num_devices_; id++) {
            ASSERT_TRUE(unit_tests::compute::binary::single_core_binary(devices_.at(id), test_config));
            // TODO: Remove early return once back-to-back tests are passing on Quasar
            if (this->arch_ == ARCH::QUASAR) {
                return;
            }
        }
    }
}

TEST_F(LLKMeshDeviceFixtureSlowDispatchOnly, TensixBinaryComputeSingleCoreMultiTileMulDestAcc) {
    if (this->arch_ == tt::ARCH::QUASAR) {
        GTEST_SKIP() << "DestAcc test support will be added with DestReuse bring-up";
    }
    for (uint8_t i = uint8_t(MathFidelity::LoFi); i <= uint8_t(MathFidelity::HiFi4); i++) {
        if (i == 1) {
            continue;
        }
        unit_tests::compute::binary::SingleCoreBinaryConfig test_config = {
            .num_tiles = 4,
            .tile_byte_size = 2 * 32 * 32,
            .l1_input_data_format = tt::DataFormat::Float16_b,
            .l1_output_data_format = tt::DataFormat::Float16_b,
            .core = CoreCoord(0, 0),
            .binary_op = "mul",
            .acc_to_dest = true,
            .math_fidelity = MathFidelity(i),
        };
        log_info(tt::LogTest, "Math Fidelity = {}", i);
        for (unsigned int id = 0; id < num_devices_; id++) {
            ASSERT_TRUE(unit_tests::compute::binary::single_core_binary(devices_.at(id), test_config));
            // TODO: Remove early return once back-to-back tests are passing on Quasar
            if (this->arch_ == ARCH::QUASAR) {
                return;
            }
        }
    }
}

}  // namespace tt::tt_metal
