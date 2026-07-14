// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <tt-metalium/bfloat8.hpp>
#include <bit>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <variant>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/constants.hpp>
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
#include "tt_metal/test_utils/comparison.hpp"
#include "tt_metal/test_utils/packing.hpp"
#include "tt_metal/test_utils/print_helpers.hpp"
#include <umd/device/types/arch.hpp>
#include "tt_metal/test_utils/bfloat_utils.hpp"
#include <tt-metalium/experimental/metal2_host_api/program.hpp>

namespace tt::tt_metal {
class IDevice;
}  // namespace tt::tt_metal

namespace tt::tt_metal {

using std::vector;
using namespace tt;
using namespace tt::test_utils;

namespace unit_tests::compute::reconfig {

struct ReconfigConfig {
    size_t num_tiles = 0;
    // Number of tiles finished with single LLK API call:
    size_t ublock_size_tiles = 0;
    // Reconfig LLK API calls can either explicitly or implicitly take previous
    // CB indices; which version of the call is used is defined by this flag:
    bool explicit_reconfig = false;
    // Some reconfig calls are joined for SrcA/B; whether split or joined calls
    // are used is defined with this flag:
    bool split_src_reconfig = false;
    // This flag defines whether regular packing to L1 is used, or the one
    // where the result is accumulated with the previous value:
    bool l1_acc = false;
    // Whether or not we want the result to be stored in DST in FP32 and/or
    // accumulated with previous DST value is controlled with this flag:
    bool fp32_dest_acc_en = false;
    // Whether to test with copy_tile or copy_block_matmul_partials is contro-
    // lled with this flag:
    bool block_copy = true;
    // Whether or not to sync full/half DST between MATH and PACK:
    bool dst_full_sync_en = false;
};

using VariantVectorType = std::variant<std::vector<float>, std::vector<bfloat16>>;

/// @brief Does Dramx3 --> Reader --> CB --> Add with acc --> CB --> Writer --> Dram
/// @param device
/// @param test_config - Configuration of the test -- see struct
/// @return
bool single_core_reconfig(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device, const ReconfigConfig& test_config) {
    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    bool pass = true;
    uint32_t in0_id = 0;
    uint32_t in1_id = 1;
    uint32_t in2_id = 2;
    uint32_t out0_id = 16;
    uint32_t out1_id = 17;
    static float out0_result_old = 0;
    // Since golden is not perfect, some corner cases for these values will
    // make the tests fail. However, this is a representative example since
    // it utilizes the full BFP16 presicion and range:
    float in0_val = 1.0;
    float in1_val = 127.0;
    float in2_val = 0.0078125;
    uint32_t single_tile_size_fp32 = 4 * tt::constants::TILE_HW;
    uint32_t single_tile_size_bfp16b = 2 * tt::constants::TILE_HW;
    uint32_t single_tile_size_bfp8b = tt::constants::BFLOAT8_B_TILE_HW;
    uint32_t single_tile_size_out0 = test_config.fp32_dest_acc_en ? single_tile_size_fp32 : single_tile_size_bfp16b;
    const size_t dram_buffer_size_bfp16b = test_config.num_tiles * single_tile_size_bfp16b;
    const size_t dram_buffer_size_bfp8b = test_config.num_tiles * single_tile_size_bfp8b;
    const size_t dram_buffer_size_out0 = test_config.num_tiles * single_tile_size_out0;

    CoreCoord core = {0, 0};

    auto& cq = mesh_device->mesh_command_queue();
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    distributed::MeshWorkload workload;
    tt_metal::Program program = tt_metal::CreateProgram();
    workload.add_program(device_range, std::move(program));
    auto& program_ = workload.get_programs().at(device_range);
    auto* device = mesh_device->get_devices()[0];

    tt::tt_metal::InterleavedBufferConfig dram_config_bfp16b{
        .device = device,
        .size = dram_buffer_size_bfp16b,
        .page_size = dram_buffer_size_bfp16b,
        .buffer_type = tt::tt_metal::BufferType::DRAM};

    tt::tt_metal::InterleavedBufferConfig dram_config_bfp8b{
        .device = device,
        .size = dram_buffer_size_bfp8b,
        .page_size = dram_buffer_size_bfp8b,
        .buffer_type = tt::tt_metal::BufferType::DRAM};

    tt::tt_metal::InterleavedBufferConfig dram_config_out0{
        .device = device,
        .size = dram_buffer_size_out0,
        .page_size = dram_buffer_size_out0,
        .buffer_type = tt::tt_metal::BufferType::DRAM};

    // This will be srcB in Bfp8_b
    auto input0_dram_buffer = CreateBuffer(dram_config_bfp8b);
    uint32_t input0_dram_byte_address = input0_dram_buffer->address();

    // This will be srcA in Float16_b
    auto input1_dram_buffer = CreateBuffer(dram_config_bfp16b);
    uint32_t input1_dram_byte_address = input1_dram_buffer->address();

    // This will be DEST in Float16_b
    auto input2_dram_buffer = CreateBuffer(dram_config_bfp16b);
    uint32_t input2_dram_byte_address = input2_dram_buffer->address();

    // This will be Output0 in Float32 or Float16_b depending on fp32_dest_acc_en
    auto output0_dram_buffer = CreateBuffer(dram_config_out0);
    uint32_t output0_dram_byte_address = output0_dram_buffer->address();

    // This will be Output1 in Bfp8_b
    auto output1_dram_buffer = CreateBuffer(dram_config_bfp8b);
    uint32_t output1_dram_byte_address = output1_dram_buffer->address();

    tt_metal::CircularBufferConfig l1_input0_cb_config =
        tt_metal::CircularBufferConfig(dram_buffer_size_bfp8b, {{in0_id, tt::DataFormat::Bfp8_b}})
            .set_page_size(in0_id, single_tile_size_bfp8b);
    tt_metal::CreateCircularBuffer(program_, core, l1_input0_cb_config);

    tt_metal::CircularBufferConfig l1_input1_cb_config =
        tt_metal::CircularBufferConfig(dram_buffer_size_bfp16b, {{in1_id, tt::DataFormat::Float16_b}})
            .set_page_size(in1_id, single_tile_size_bfp16b);
    tt_metal::CreateCircularBuffer(program_, core, l1_input1_cb_config);

    tt_metal::CircularBufferConfig l1_input2_cb_config =
        tt_metal::CircularBufferConfig(dram_buffer_size_bfp16b, {{in2_id, tt::DataFormat::Float16_b}})
            .set_page_size(in2_id, single_tile_size_bfp16b);
    tt_metal::CreateCircularBuffer(program_, core, l1_input2_cb_config);

    tt_metal::CircularBufferConfig l1_output0_cb_config =
        tt_metal::CircularBufferConfig(
            dram_buffer_size_out0,
            {{out0_id, (test_config.fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b)}})
            .set_page_size(out0_id, single_tile_size_out0);
    tt_metal::CreateCircularBuffer(program_, core, l1_output0_cb_config);

    tt_metal::CircularBufferConfig l1_output1_cb_config =
        tt_metal::CircularBufferConfig(dram_buffer_size_bfp8b, {{out1_id, tt::DataFormat::Bfp8_b}})
            .set_page_size(out1_id, single_tile_size_bfp8b);
    tt_metal::CreateCircularBuffer(program_, core, l1_output1_cb_config);

    vector<uint32_t> compute_kernel_args = {};
    std::map<std::string, std::string> defines;

    defines["LOAD_BUF2_DATA"] = "1";  // Needed always in order for reader kernel to load data from CB2
    defines["EXPLICIT_RECONFIG"] = test_config.explicit_reconfig ? "1" : "0";
    defines["SPLIT_SRC_RECONFIG"] = test_config.split_src_reconfig ? "1" : "0";
    defines["BLOCK_COPY"] = test_config.block_copy ? "1" : "0";
    defines["L1_ACC"] = test_config.l1_acc ? "1" : "0";

    auto reader_kernel = tt_metal::CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_binary.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt_metal::NOC::RISCV_1_default,
            .defines = defines});

    auto writer_kernel = tt_metal::CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_binary.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

    auto compute_kernel = tt_metal::CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/compute/reconfig.cpp",
        core,
        tt_metal::ComputeConfig{
            .fp32_dest_acc_en = test_config.fp32_dest_acc_en,
            .dst_full_sync_en = test_config.dst_full_sync_en,
            .compile_args = compute_kernel_args,
            .defines = defines});

    SetRuntimeArgs(
        program_,
        compute_kernel,
        core,
        {
            uint32_t(test_config.num_tiles),
            uint32_t(test_config.ublock_size_tiles),
        });

    ////////////////////////////////////////////////////////////////////////////
    //                      Stimulus Generation
    ////////////////////////////////////////////////////////////////////////////
    // Since we're testing compute threads' reconfiguration, it's not necessary
    // for input tensors to be filled with random values, only that one src reg
    // is in different format than the other. If thread reconfiguration is done
    // incorrectly or underlying API/LLK is broken, this will be shown in either
    // difference in output sizes or values.
    std::vector<uint32_t> src0_vec = create_constant_vector_of_bfp8(dram_buffer_size_bfp8b, in0_val, false);
    std::vector<uint32_t> src1_vec = create_constant_vector_of_bfloat16(dram_buffer_size_bfp16b, in1_val);
    std::vector<uint32_t> src2_vec = create_constant_vector_of_bfloat16(dram_buffer_size_bfp16b, in2_val);

    ////////////////////////////////////////////////////////////////////////////
    //                      Golden Generation
    ////////////////////////////////////////////////////////////////////////////
    auto input0 = unpack_bfp8_tiles_into_float_vec(src0_vec, true, false);
    auto input1 = unpack_uint32_vec_into_bfloat16_vec(src1_vec);
    auto input2 = unpack_uint32_vec_into_bfloat16_vec(src2_vec);

    // Intermediate result stored in temp_golden should be represented in
    // 19 bits since that's the width of srcA/B/FPU. This is why it's
    // float32 in golden. As for golden1, it should be Bfp8_b in the end,
    // but since there's no available conversion from Float16_b to Bfp8_b,
    // it remains in float and is then converted to Bfp8_b.
    std::vector<float> temp_golden(input1.size());

    // It's tricky to make a variable-type vector, so create two for each case
    // of fp32_dest_acc_en, fp32 when true, fp16 when false
    std::vector<float> golden0_fp32(input1.size());
    std::vector<bfloat16> golden0_bfp16(input1.size());
    // This vector will hold unpacked Bfp8 result:
    std::vector<float> golden1(input1.size());
    // This vector will hold packed fp16_b/fp32 result:
    std::vector<uint32_t> packed_golden0(input1.size());
    for (auto i = 0; i < temp_golden.size(); i++) {
        // Do temp = SrcA + SrcB:
        temp_golden[i] = static_cast<float>(input1[i]) + static_cast<float>(bfloat16(input0[i]));
        // Do temp + DST, store in out0 vector depending on fp32_dest_acc_en:
        if (test_config.fp32_dest_acc_en) {
            golden0_fp32[i] = temp_golden[i] + static_cast<float>(input2[i]);
        } else {
            golden0_bfp16[i] = bfloat16(temp_golden[i] + static_cast<float>(input2[i]));
        }
        // Do out1 = temp + DST:
        golden1[i] = static_cast<float>(bfloat16(temp_golden[i] + static_cast<float>(input2[i])));
        // Do out0[bfp16] = temp + L1, this makes sense only if not fp32_dest_acc_en:
        if (test_config.l1_acc && !test_config.fp32_dest_acc_en) {
            golden0_bfp16[i] = bfloat16(static_cast<float>(golden0_bfp16[i]) + out0_result_old);
        } else {
            out0_result_old = static_cast<float>(golden0_bfp16[i]);
        }
        // Cast float32 to "packed "uint32 out0 vector if fp32_dest_acc_en:
        if (test_config.fp32_dest_acc_en) {
            packed_golden0[i] = std::bit_cast<uint32_t>(golden0_fp32[i]);
        }
    }
    // Pack out0 vector if not fp32_dest_acc_en:
    if (!test_config.fp32_dest_acc_en) {
        packed_golden0 = pack_vector<uint32_t, bfloat16>(golden0_bfp16);
    }
    // Pack out1 vector:
    std::vector<uint32_t> packed_golden1 = pack_as_bfp8_tiles(ttsl::make_const_span(golden1), true, false);

    // ////////////////////////////////////////////////////////////////////////////
    // //                      Compile and Execute Application
    // ////////////////////////////////////////////////////////////////////////////
    tt_metal::detail::WriteToBuffer(input0_dram_buffer, src0_vec);
    tt_metal::detail::WriteToBuffer(input1_dram_buffer, src1_vec);
    tt_metal::detail::WriteToBuffer(input2_dram_buffer, src2_vec);

    static constexpr uint32_t k_input0_dram_bank_id = 0;
    static constexpr uint32_t k_input1_dram_bank_id = 0;
    static constexpr uint32_t k_input2_dram_bank_id = 0;
    static constexpr uint32_t k_output0_dram_bank_id = 0;
    static constexpr uint32_t k_output1_dram_bank_id = 0;

    tt_metal::SetRuntimeArgs(
        program_,
        reader_kernel,
        core,
        {
            (uint32_t)input0_dram_byte_address,
            k_input0_dram_bank_id,  // dram bank id
            (uint32_t)input1_dram_byte_address,
            k_input1_dram_bank_id,
            (uint32_t)test_config.num_tiles,
            (uint32_t)input2_dram_byte_address,
            k_input2_dram_bank_id,
        });
    tt_metal::SetRuntimeArgs(
        program_,
        writer_kernel,
        core,
        {
            (uint32_t)output0_dram_byte_address,
            k_output0_dram_bank_id,
            (uint32_t)out0_id,
            (uint32_t)output1_dram_byte_address,
            k_output1_dram_bank_id,
            (uint32_t)out1_id,
            (uint32_t)test_config.num_tiles,
            (uint32_t)test_config.ublock_size_tiles,
        });

    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);

    // ////////////////////////////////////////////////////////////////////////////
    // //                      Comparison Checking
    // ////////////////////////////////////////////////////////////////////////////
    std::vector<uint32_t> dest0_buffer_data(src1_vec.size());
    std::vector<uint32_t> dest1_buffer_data(src0_vec.size());
    tt_metal::detail::ReadFromBuffer(output0_dram_buffer, dest0_buffer_data);
    tt_metal::detail::ReadFromBuffer(output1_dram_buffer, dest1_buffer_data);

    pass &= is_close_packed_vectors<bfloat16, uint32_t>(
        dest0_buffer_data, packed_golden0, [&](const bfloat16& a, const bfloat16& b) {
            return is_close(a, b, 0.0155f);
        });
    pass &= is_close_packed_vectors<bfloat16, uint32_t>(
        dest1_buffer_data, packed_golden1, [&](const bfloat16& a, const bfloat16& b) {
            return is_close(a, b, 0.0155);
        });

    return pass;
}

bool single_core_unpack_reconfig_quasar(const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    // Three matmul_tiles ops with unpack reconfig between pairs; see reconfig_unpack_quasar.cpp.
    constexpr uint32_t kNumOps = 3;
    const uint32_t f16_tile_size = tt::tile_size(tt::DataFormat::Float16_b);
    const uint32_t f32_tile_size = tt::tile_size(tt::DataFormat::Float32);
    const uint32_t out_bytes = kNumOps * f16_tile_size;

    const CoreCoord core = {0, 0};
    auto& cq = mesh_device->mesh_command_queue();
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    const experimental::NodeCoord node{static_cast<uint32_t>(core.x), static_cast<uint32_t>(core.y)};

    distributed::DeviceLocalBufferConfig f16_dram_cfg{
        .page_size = f16_tile_size, .buffer_type = tt::tt_metal::BufferType::DRAM, .bottom_up = false};
    distributed::DeviceLocalBufferConfig f32_dram_cfg{
        .page_size = f32_tile_size, .buffer_type = tt::tt_metal::BufferType::DRAM, .bottom_up = false};
    distributed::ReplicatedBufferConfig f16_buf_cfg{.size = f16_tile_size};
    distributed::ReplicatedBufferConfig f32_buf_cfg{.size = f32_tile_size};
    distributed::ReplicatedBufferConfig out_buf_cfg{.size = out_bytes};

    auto inp0_dram = distributed::MeshBuffer::create(f16_buf_cfg, f16_dram_cfg, mesh_device.get());
    auto inp1_dram = distributed::MeshBuffer::create(f16_buf_cfg, f16_dram_cfg, mesh_device.get());
    auto inp2_dram = distributed::MeshBuffer::create(f32_buf_cfg, f32_dram_cfg, mesh_device.get());
    auto inp3_dram = distributed::MeshBuffer::create(f32_buf_cfg, f32_dram_cfg, mesh_device.get());
    auto inp4_dram = distributed::MeshBuffer::create(f16_buf_cfg, f16_dram_cfg, mesh_device.get());
    auto inp5_dram = distributed::MeshBuffer::create(f16_buf_cfg, f16_dram_cfg, mesh_device.get());
    auto out_dram = distributed::MeshBuffer::create(out_buf_cfg, f16_dram_cfg, mesh_device.get());

    const experimental::DFBSpecName INP0_DFB{"in0"};
    const experimental::DFBSpecName INP1_DFB{"in1"};
    const experimental::DFBSpecName INP2_DFB{"in2"};
    const experimental::DFBSpecName INP3_DFB{"in3"};
    const experimental::DFBSpecName INP4_DFB{"in4"};
    const experimental::DFBSpecName INP5_DFB{"in5"};
    const experimental::DFBSpecName OUT_DFB{"out"};
    const experimental::KernelSpecName READER{"reader"};
    const experimental::KernelSpecName WRITER{"writer"};
    const experimental::KernelSpecName COMPUTE{"compute"};

    auto make_f16_input_dfb = [&](const experimental::DFBSpecName& name) {
        return experimental::DataflowBufferSpec{
            .unique_id = name,
            .entry_size = f16_tile_size,
            .num_entries = 1,
            .data_format_metadata = tt::DataFormat::Float16_b,
        };
    };
    auto make_f32_input_dfb = [&](const experimental::DFBSpecName& name) {
        return experimental::DataflowBufferSpec{
            .unique_id = name,
            .entry_size = f32_tile_size,
            .num_entries = 1,
            .data_format_metadata = tt::DataFormat::Float32,
        };
    };
    experimental::DataflowBufferSpec inp0_dfb_spec = make_f16_input_dfb(INP0_DFB);
    experimental::DataflowBufferSpec inp1_dfb_spec = make_f16_input_dfb(INP1_DFB);
    experimental::DataflowBufferSpec inp2_dfb_spec = make_f32_input_dfb(INP2_DFB);
    experimental::DataflowBufferSpec inp3_dfb_spec = make_f32_input_dfb(INP3_DFB);
    experimental::DataflowBufferSpec inp4_dfb_spec = make_f16_input_dfb(INP4_DFB);
    experimental::DataflowBufferSpec inp5_dfb_spec = make_f16_input_dfb(INP5_DFB);
    experimental::DataflowBufferSpec out_dfb_spec{
        .unique_id = OUT_DFB,
        .entry_size = f16_tile_size,
        .num_entries = kNumOps,
        .data_format_metadata = tt::DataFormat::Float16_b,
    };

    using DFBEndpoint = experimental::DFBEndpointType;
    using DFBAccess = experimental::DFBAccessPattern;
    auto dfb_binding = [](const experimental::DFBSpecName& name, DFBEndpoint endpoint) {
        return experimental::DFBBinding{
            .dfb_spec_name = name,
            .accessor_name = name.get(),
            .endpoint_type = endpoint,
            .access_pattern = DFBAccess::STRIDED,
        };
    };

    experimental::KernelSpec reader_spec{
        .unique_id = READER,
        .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_six_input.cpp",
        .num_threads = 1,
        .dfb_bindings =
            {dfb_binding(INP0_DFB, DFBEndpoint::PRODUCER),
             dfb_binding(INP1_DFB, DFBEndpoint::PRODUCER),
             dfb_binding(INP2_DFB, DFBEndpoint::PRODUCER),
             dfb_binding(INP3_DFB, DFBEndpoint::PRODUCER),
             dfb_binding(INP4_DFB, DFBEndpoint::PRODUCER),
             dfb_binding(INP5_DFB, DFBEndpoint::PRODUCER)},
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"src0_addr",
                  "src0_bank_id",
                  "src1_addr",
                  "src1_bank_id",
                  "src2_addr",
                  "src2_bank_id",
                  "src3_addr",
                  "src3_bank_id",
                  "src4_addr",
                  "src4_bank_id",
                  "src5_addr",
                  "src5_bank_id",
                  "num_tiles"}},
        .hw_config = experimental::DataMovementGen2Config{.disable_dfb_implicit_sync_for_all = true},
    };

    experimental::KernelSpec writer_spec{
        .unique_id = WRITER,
        .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary.cpp",
        .num_threads = 1,
        .dfb_bindings = {{
            .dfb_spec_name = OUT_DFB,
            .accessor_name = "in",
            .endpoint_type = DFBEndpoint::CONSUMER,
            .access_pattern = DFBAccess::STRIDED,
        }},
        .runtime_arg_schema = {.runtime_arg_names = {"dst_addr", "bank_id", "num_tiles"}},
        .hw_config = experimental::DataMovementGen2Config{.disable_dfb_implicit_sync_for_all = true},
    };

    experimental::KernelSpec compute_spec{
        .unique_id = COMPUTE,
        .source = "tests/tt_metal/tt_metal/test_kernels/compute/reconfig_unpack_quasar.cpp",
        .num_threads = 1,
        .dfb_bindings =
            {dfb_binding(INP0_DFB, DFBEndpoint::CONSUMER),
             dfb_binding(INP1_DFB, DFBEndpoint::CONSUMER),
             dfb_binding(INP2_DFB, DFBEndpoint::CONSUMER),
             dfb_binding(INP3_DFB, DFBEndpoint::CONSUMER),
             dfb_binding(INP4_DFB, DFBEndpoint::CONSUMER),
             dfb_binding(INP5_DFB, DFBEndpoint::CONSUMER),
             dfb_binding(OUT_DFB, DFBEndpoint::PRODUCER)},
        .hw_config =
            experimental::ComputeGen2Config{
                .math_fidelity = MathFidelity::HiFi4,
                .fp32_dest_acc_en = true,
                .unpack_to_dest_mode =
                    {{INP2_DFB, tt::tt_metal::UnpackToDestMode::Default},
                     {INP3_DFB, tt::tt_metal::UnpackToDestMode::Default}},
            },
    };

    experimental::WorkUnitSpec wu{
        .name = "main",
        .kernels = {READER, WRITER, COMPUTE},
        .target_nodes = node,
    };

    experimental::ProgramSpec spec{
        .name = "reconfig_quasar",
        .kernels = {reader_spec, writer_spec, compute_spec},
        .dataflow_buffers =
            {inp0_dfb_spec, inp1_dfb_spec, inp2_dfb_spec, inp3_dfb_spec, inp4_dfb_spec, inp5_dfb_spec, out_dfb_spec},
        .work_units = {wu},
    };

    Program program = experimental::MakeProgramFromSpec(*mesh_device, spec);

    // Random stimulus: U(0, 1) per element (keep magnitudes small so matmul output
    // stays in a sensible bfloat16 range; each output element is sum of 32 products).
    // d0/d1, d4/d5 are bfloat16 (Float16_b tile); d2/d3 are float32 (Float32 tile).
    constexpr int kRandMax = 1;
    constexpr uint32_t elems_per_tile = tt::constants::TILE_HW;
    auto src0 = create_random_vector_of_bfloat16(f16_tile_size, kRandMax, /*seed=*/0x1001);
    auto src1 = create_random_vector_of_bfloat16(f16_tile_size, kRandMax, /*seed=*/0x1002);
    auto gen_random_f32 = [&](uint32_t seed) {
        std::vector<uint32_t> packed(elems_per_tile);
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> dist(0.0f, static_cast<float>(kRandMax));
        for (uint32_t i = 0; i < elems_per_tile; ++i) {
            packed[i] = std::bit_cast<uint32_t>(dist(rng));
        }
        return packed;
    };
    auto src2 = gen_random_f32(/*seed=*/0x1003);
    auto src3 = gen_random_f32(/*seed=*/0x1004);
    auto src4 = create_random_vector_of_bfloat16(f16_tile_size, kRandMax, /*seed=*/0x1005);
    auto src5 = create_random_vector_of_bfloat16(f16_tile_size, kRandMax, /*seed=*/0x1006);

    distributed::WriteShard(cq, inp0_dram, src0, zero_coord, false);
    distributed::WriteShard(cq, inp1_dram, src1, zero_coord, false);
    distributed::WriteShard(cq, inp2_dram, src2, zero_coord, false);
    distributed::WriteShard(cq, inp3_dram, src3, zero_coord, false);
    distributed::WriteShard(cq, inp4_dram, src4, zero_coord, false);
    distributed::WriteShard(cq, inp5_dram, src5, zero_coord, false);

    auto in0 = unpack_uint32_vec_into_bfloat16_vec(src0);
    auto in1 = unpack_uint32_vec_into_bfloat16_vec(src1);
    auto unpack_f32 = [](const std::vector<uint32_t>& packed) {
        std::vector<float> out(packed.size());
        for (size_t i = 0; i < packed.size(); ++i) {
            out[i] = std::bit_cast<float>(packed[i]);
        }
        return out;
    };
    auto in2 = unpack_f32(src2);
    auto in3 = unpack_f32(src3);
    auto in4 = unpack_uint32_vec_into_bfloat16_vec(src4);
    auto in5 = unpack_uint32_vec_into_bfloat16_vec(src5);

    // Face-aware index for a 32x32 tile laid out as 4 faces of 16x16 (face-row-major,
    // then row-major within face). Matches how the device sees a tile in DRAM/L1.
    auto face_idx = [](uint32_t row, uint32_t col) -> uint32_t {
        const uint32_t face = (row / tt::constants::FACE_HEIGHT) * 2 + (col / tt::constants::FACE_WIDTH);
        const uint32_t r = row % tt::constants::FACE_HEIGHT;
        const uint32_t c = col % tt::constants::FACE_WIDTH;
        return face * tt::constants::FACE_HW + r * tt::constants::FACE_WIDTH + c;
    };

    // Matmul of two face-layout tiles. Inputs are converted to float for the
    // sum; output is bfloat16-truncated (pack format is Float16_b).
    auto matmul_face = [&](auto& A, auto& B) -> std::vector<bfloat16> {
        std::vector<bfloat16> C(elems_per_tile);
        for (uint32_t i = 0; i < tt::constants::TILE_HEIGHT; ++i) {
            for (uint32_t j = 0; j < tt::constants::TILE_WIDTH; ++j) {
                float sum = 0.0f;
                for (uint32_t k = 0; k < tt::constants::TILE_WIDTH; ++k) {
                    sum += static_cast<float>(A[face_idx(i, k)]) * static_cast<float>(B[face_idx(k, j)]);
                }
                C[face_idx(i, j)] = bfloat16(sum);
            }
        }
        return C;
    };

    auto golden_op0 = matmul_face(in0, in1);
    auto golden_op1 = matmul_face(in2, in3);
    auto golden_op2 = matmul_face(in4, in5);
    std::vector<bfloat16> golden(kNumOps * elems_per_tile);
    for (uint32_t e = 0; e < elems_per_tile; ++e) {
        golden[0 * elems_per_tile + e] = golden_op0[e];
        golden[1 * elems_per_tile + e] = golden_op1[e];
        golden[2 * elems_per_tile + e] = golden_op2[e];
    }
    auto packed_golden = pack_vector<uint32_t, bfloat16>(golden);

    experimental::ProgramRunArgs params;
    params.kernel_run_args = {
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = READER,
            .runtime_arg_values =
                {{node,
                  {{"src0_addr", static_cast<uint32_t>(inp0_dram->address())},
                   {"src0_bank_id", 0u},
                   {"src1_addr", static_cast<uint32_t>(inp1_dram->address())},
                   {"src1_bank_id", 0u},
                   {"src2_addr", static_cast<uint32_t>(inp2_dram->address())},
                   {"src2_bank_id", 0u},
                   {"src3_addr", static_cast<uint32_t>(inp3_dram->address())},
                   {"src3_bank_id", 0u},
                   {"src4_addr", static_cast<uint32_t>(inp4_dram->address())},
                   {"src4_bank_id", 0u},
                   {"src5_addr", static_cast<uint32_t>(inp5_dram->address())},
                   {"src5_bank_id", 0u},
                   {"num_tiles", 1u}}}},
        },
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = WRITER,
            .runtime_arg_values =
                {{node,
                  {{"dst_addr", static_cast<uint32_t>(out_dram->address())}, {"bank_id", 0u}, {"num_tiles", kNumOps}}}},
        },
        experimental::ProgramRunArgs::KernelRunArgs{.kernel = COMPUTE},
    };
    experimental::SetProgramRunArgs(program, params);

    auto* dev = mesh_device->get_devices()[0];
    tt_metal::detail::LaunchProgram(dev, program, /*wait_until_cores_done=*/true);

    std::vector<uint32_t> dest_buffer_data;
    distributed::ReadShard(cq, dest_buffer_data, out_dram, zero_coord, false);

    auto device_unpacked = unpack_vector<bfloat16, uint32_t>(dest_buffer_data);
    auto golden_unpacked = unpack_vector<bfloat16, uint32_t>(packed_golden);

    bool pass = true;
    for (uint32_t t = 0; t < kNumOps; ++t) {
        uint32_t mismatches = 0;
        int first_mismatch_local = -1;
        float worst_absdiff = 0.0f;
        for (uint32_t e = 0; e < elems_per_tile; ++e) {
            const bfloat16 a = device_unpacked[t * elems_per_tile + e];
            const bfloat16 b = golden_unpacked[t * elems_per_tile + e];
            if (!is_close(a, b, 0.0155f, 0.001f)) {
                if (first_mismatch_local < 0) {
                    first_mismatch_local = static_cast<int>(e);
                }
                ++mismatches;
                const float absdiff = std::fabs(static_cast<float>(a) - static_cast<float>(b));
                worst_absdiff = std::fmax(worst_absdiff, absdiff);
            }
        }
        if (mismatches == 0) {
            log_info(tt::LogTest, "OP[{}]: PASS", t);
        } else {
            pass = false;
            log_error(
                tt::LogTest,
                "OP[{}]: FAIL ({}/{} mismatches; first idx {} (global {}); worst absdiff={:.4f})",
                t,
                mismatches,
                elems_per_tile,
                first_mismatch_local,
                t * elems_per_tile + first_mismatch_local,
                worst_absdiff);
        }
    }

    return pass;
}

bool single_core_pack_reconfig_quasar(const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    // Same 3×2 matmul flow as unpack reconfig; pack reconfig+init per output at pack time; see
    // reconfig_pack_quasar.cpp.
    const uint32_t f16_tile_size = tt::tile_size(tt::DataFormat::Float16_b);
    const uint32_t f32_tile_size = tt::tile_size(tt::DataFormat::Float32);

    const CoreCoord core = {0, 0};
    auto& cq = mesh_device->mesh_command_queue();
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    const experimental::NodeCoord node{static_cast<uint32_t>(core.x), static_cast<uint32_t>(core.y)};

    distributed::DeviceLocalBufferConfig f16_dram_cfg{
        .page_size = f16_tile_size, .buffer_type = tt::tt_metal::BufferType::DRAM, .bottom_up = false};
    distributed::DeviceLocalBufferConfig f32_dram_cfg{
        .page_size = f32_tile_size, .buffer_type = tt::tt_metal::BufferType::DRAM, .bottom_up = false};
    distributed::ReplicatedBufferConfig f16_buf_cfg{.size = f16_tile_size};
    distributed::ReplicatedBufferConfig f32_buf_cfg{.size = f32_tile_size};

    auto inp0_dram = distributed::MeshBuffer::create(f16_buf_cfg, f16_dram_cfg, mesh_device.get());
    auto inp1_dram = distributed::MeshBuffer::create(f16_buf_cfg, f16_dram_cfg, mesh_device.get());
    auto inp2_dram = distributed::MeshBuffer::create(f32_buf_cfg, f32_dram_cfg, mesh_device.get());
    auto inp3_dram = distributed::MeshBuffer::create(f32_buf_cfg, f32_dram_cfg, mesh_device.get());
    auto inp4_dram = distributed::MeshBuffer::create(f16_buf_cfg, f16_dram_cfg, mesh_device.get());
    auto inp5_dram = distributed::MeshBuffer::create(f16_buf_cfg, f16_dram_cfg, mesh_device.get());
    auto out0_dram = distributed::MeshBuffer::create(f16_buf_cfg, f16_dram_cfg, mesh_device.get());
    auto out1_dram = distributed::MeshBuffer::create(f32_buf_cfg, f32_dram_cfg, mesh_device.get());
    auto out2_dram = distributed::MeshBuffer::create(f16_buf_cfg, f16_dram_cfg, mesh_device.get());

    const experimental::DFBSpecName INP0_DFB{"in0"};
    const experimental::DFBSpecName INP1_DFB{"in1"};
    const experimental::DFBSpecName INP2_DFB{"in2"};
    const experimental::DFBSpecName INP3_DFB{"in3"};
    const experimental::DFBSpecName INP4_DFB{"in4"};
    const experimental::DFBSpecName INP5_DFB{"in5"};
    const experimental::DFBSpecName OUT0_DFB{"out0"};
    const experimental::DFBSpecName OUT1_DFB{"out1"};
    const experimental::DFBSpecName OUT2_DFB{"out2"};
    const experimental::KernelSpecName READER{"reader"};
    const experimental::KernelSpecName WRITER0{"writer0"};
    const experimental::KernelSpecName WRITER1{"writer1"};
    const experimental::KernelSpecName WRITER2{"writer2"};
    const experimental::KernelSpecName COMPUTE{"compute"};

    auto make_f16_input_dfb = [&](const experimental::DFBSpecName& name) {
        return experimental::DataflowBufferSpec{
            .unique_id = name,
            .entry_size = f16_tile_size,
            .num_entries = 1,
            .data_format_metadata = tt::DataFormat::Float16_b,
        };
    };
    auto make_f32_input_dfb = [&](const experimental::DFBSpecName& name) {
        return experimental::DataflowBufferSpec{
            .unique_id = name,
            .entry_size = f32_tile_size,
            .num_entries = 1,
            .data_format_metadata = tt::DataFormat::Float32,
        };
    };
    experimental::DataflowBufferSpec inp0_dfb_spec = make_f16_input_dfb(INP0_DFB);
    experimental::DataflowBufferSpec inp1_dfb_spec = make_f16_input_dfb(INP1_DFB);
    experimental::DataflowBufferSpec inp2_dfb_spec = make_f32_input_dfb(INP2_DFB);
    experimental::DataflowBufferSpec inp3_dfb_spec = make_f32_input_dfb(INP3_DFB);
    experimental::DataflowBufferSpec inp4_dfb_spec = make_f16_input_dfb(INP4_DFB);
    experimental::DataflowBufferSpec inp5_dfb_spec = make_f16_input_dfb(INP5_DFB);
    experimental::DataflowBufferSpec out0_dfb_spec{
        .unique_id = OUT0_DFB,
        .entry_size = f16_tile_size,
        .num_entries = 1,
        .data_format_metadata = tt::DataFormat::Float16_b,
    };
    experimental::DataflowBufferSpec out1_dfb_spec{
        .unique_id = OUT1_DFB,
        .entry_size = f32_tile_size,
        .num_entries = 1,
        .data_format_metadata = tt::DataFormat::Float32,
    };
    experimental::DataflowBufferSpec out2_dfb_spec{
        .unique_id = OUT2_DFB,
        .entry_size = f16_tile_size,
        .num_entries = 1,
        .data_format_metadata = tt::DataFormat::Float16_b,
    };

    using DFBEndpoint = experimental::DFBEndpointType;
    using DFBAccess = experimental::DFBAccessPattern;
    auto dfb_binding = [](const experimental::DFBSpecName& name, DFBEndpoint endpoint) {
        return experimental::DFBBinding{
            .dfb_spec_name = name,
            .accessor_name = name.get(),
            .endpoint_type = endpoint,
            .access_pattern = DFBAccess::STRIDED,
        };
    };
    auto make_writer_spec = [&](const experimental::KernelSpecName& writer_id,
                                const experimental::DFBSpecName& out_dfb) {
        return experimental::KernelSpec{
            .unique_id = writer_id,
            .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary_2_0.cpp",
            .num_threads = 1,
            .dfb_bindings = {experimental::ConsumerOf(out_dfb, "in")},
            .runtime_arg_schema = {.runtime_arg_names = {"dst_addr", "bank_id", "num_tiles"}},
            .hw_config = experimental::DataMovementGen2Config{.disable_dfb_implicit_sync_for_all = true},
        };
    };

    experimental::KernelSpec reader_spec{
        .unique_id = READER,
        .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_six_input.cpp",
        .num_threads = 1,
        .dfb_bindings =
            {dfb_binding(INP0_DFB, DFBEndpoint::PRODUCER),
             dfb_binding(INP1_DFB, DFBEndpoint::PRODUCER),
             dfb_binding(INP2_DFB, DFBEndpoint::PRODUCER),
             dfb_binding(INP3_DFB, DFBEndpoint::PRODUCER),
             dfb_binding(INP4_DFB, DFBEndpoint::PRODUCER),
             dfb_binding(INP5_DFB, DFBEndpoint::PRODUCER)},
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"src0_addr",
                  "src0_bank_id",
                  "src1_addr",
                  "src1_bank_id",
                  "src2_addr",
                  "src2_bank_id",
                  "src3_addr",
                  "src3_bank_id",
                  "src4_addr",
                  "src4_bank_id",
                  "src5_addr",
                  "src5_bank_id",
                  "num_tiles"}},
        .hw_config = experimental::DataMovementGen2Config{.disable_dfb_implicit_sync_for_all = true},
    };

    experimental::KernelSpec writer0_spec = make_writer_spec(WRITER0, OUT0_DFB);
    experimental::KernelSpec writer1_spec = make_writer_spec(WRITER1, OUT1_DFB);
    experimental::KernelSpec writer2_spec = make_writer_spec(WRITER2, OUT2_DFB);

    experimental::KernelSpec compute_spec{
        .unique_id = COMPUTE,
        .source = "tests/tt_metal/tt_metal/test_kernels/compute/reconfig_pack_quasar.cpp",
        .num_threads = 1,
        .dfb_bindings =
            {dfb_binding(INP0_DFB, DFBEndpoint::CONSUMER),
             dfb_binding(INP1_DFB, DFBEndpoint::CONSUMER),
             dfb_binding(INP2_DFB, DFBEndpoint::CONSUMER),
             dfb_binding(INP3_DFB, DFBEndpoint::CONSUMER),
             dfb_binding(INP4_DFB, DFBEndpoint::CONSUMER),
             dfb_binding(INP5_DFB, DFBEndpoint::CONSUMER),
             dfb_binding(OUT0_DFB, DFBEndpoint::PRODUCER),
             dfb_binding(OUT1_DFB, DFBEndpoint::PRODUCER),
             dfb_binding(OUT2_DFB, DFBEndpoint::PRODUCER)},
        .hw_config =
            experimental::ComputeGen2Config{
                .math_fidelity = MathFidelity::HiFi4,
                .fp32_dest_acc_en = true,
                .unpack_to_dest_mode =
                    {{INP2_DFB, tt::tt_metal::UnpackToDestMode::Default},
                     {INP3_DFB, tt::tt_metal::UnpackToDestMode::Default}},
            },
    };

    experimental::WorkUnitSpec wu{
        .name = "main",
        .kernels = {READER, WRITER0, WRITER1, WRITER2, COMPUTE},
        .target_nodes = node,
    };

    experimental::ProgramSpec spec{
        .name = "reconfig_pack_quasar",
        .kernels = {reader_spec, writer0_spec, writer1_spec, writer2_spec, compute_spec},
        .dataflow_buffers =
            {inp0_dfb_spec,
             inp1_dfb_spec,
             inp2_dfb_spec,
             inp3_dfb_spec,
             inp4_dfb_spec,
             inp5_dfb_spec,
             out0_dfb_spec,
             out1_dfb_spec,
             out2_dfb_spec},
        .work_units = {wu},
    };

    Program program = experimental::MakeProgramFromSpec(*mesh_device, spec);

    constexpr int kRandMax = 1;
    constexpr uint32_t elems_per_tile = tt::constants::TILE_HW;
    auto src0 = create_random_vector_of_bfloat16(f16_tile_size, kRandMax, /*seed=*/0x2001);
    auto src1 = create_random_vector_of_bfloat16(f16_tile_size, kRandMax, /*seed=*/0x2002);
    auto gen_random_f32 = [&](uint32_t seed) {
        std::vector<uint32_t> packed(elems_per_tile);
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> dist(0.0f, static_cast<float>(kRandMax));
        for (uint32_t i = 0; i < elems_per_tile; ++i) {
            packed[i] = std::bit_cast<uint32_t>(dist(rng));
        }
        return packed;
    };
    auto src2 = gen_random_f32(/*seed=*/0x2003);
    auto src3 = gen_random_f32(/*seed=*/0x2004);
    auto src4 = create_random_vector_of_bfloat16(f16_tile_size, kRandMax, /*seed=*/0x2005);
    auto src5 = create_random_vector_of_bfloat16(f16_tile_size, kRandMax, /*seed=*/0x2006);

    distributed::WriteShard(cq, inp0_dram, src0, zero_coord, false);
    distributed::WriteShard(cq, inp1_dram, src1, zero_coord, false);
    distributed::WriteShard(cq, inp2_dram, src2, zero_coord, false);
    distributed::WriteShard(cq, inp3_dram, src3, zero_coord, false);
    distributed::WriteShard(cq, inp4_dram, src4, zero_coord, false);
    distributed::WriteShard(cq, inp5_dram, src5, zero_coord, false);

    auto in0 = unpack_uint32_vec_into_bfloat16_vec(src0);
    auto in1 = unpack_uint32_vec_into_bfloat16_vec(src1);
    auto unpack_f32 = [](const std::vector<uint32_t>& packed) {
        std::vector<float> out(packed.size());
        for (size_t i = 0; i < packed.size(); ++i) {
            out[i] = std::bit_cast<float>(packed[i]);
        }
        return out;
    };
    auto in2 = unpack_f32(src2);
    auto in3 = unpack_f32(src3);
    auto in4 = unpack_uint32_vec_into_bfloat16_vec(src4);
    auto in5 = unpack_uint32_vec_into_bfloat16_vec(src5);

    auto face_idx = [](uint32_t row, uint32_t col) -> uint32_t {
        const uint32_t face = (row / tt::constants::FACE_HEIGHT) * 2 + (col / tt::constants::FACE_WIDTH);
        const uint32_t r = row % tt::constants::FACE_HEIGHT;
        const uint32_t c = col % tt::constants::FACE_WIDTH;
        return face * tt::constants::FACE_HW + r * tt::constants::FACE_WIDTH + c;
    };

    auto matmul_face = [&](auto& A, auto& B) -> std::vector<bfloat16> {
        std::vector<bfloat16> C(elems_per_tile);
        for (uint32_t i = 0; i < tt::constants::TILE_HEIGHT; ++i) {
            for (uint32_t j = 0; j < tt::constants::TILE_WIDTH; ++j) {
                float sum = 0.0f;
                for (uint32_t k = 0; k < tt::constants::TILE_WIDTH; ++k) {
                    sum += static_cast<float>(A[face_idx(i, k)]) * static_cast<float>(B[face_idx(k, j)]);
                }
                C[face_idx(i, j)] = bfloat16(sum);
            }
        }
        return C;
    };

    auto matmul_face_f32 = [&](const std::vector<float>& A, const std::vector<float>& B) -> std::vector<float> {
        std::vector<float> C(elems_per_tile);
        for (uint32_t i = 0; i < tt::constants::TILE_HEIGHT; ++i) {
            for (uint32_t j = 0; j < tt::constants::TILE_WIDTH; ++j) {
                float sum = 0.0f;
                for (uint32_t k = 0; k < tt::constants::TILE_WIDTH; ++k) {
                    sum += A[face_idx(i, k)] * B[face_idx(k, j)];
                }
                C[face_idx(i, j)] = sum;
            }
        }
        return C;
    };

    auto golden_op0 = matmul_face(in0, in1);
    auto golden_op1_f32 = matmul_face_f32(in2, in3);
    auto golden_op2 = matmul_face(in4, in5);
    auto packed_golden_op0 = pack_vector<uint32_t, bfloat16>(golden_op0);
    auto packed_golden_op2 = pack_vector<uint32_t, bfloat16>(golden_op2);
    std::vector<uint32_t> packed_golden_op1(elems_per_tile);
    for (uint32_t e = 0; e < elems_per_tile; ++e) {
        packed_golden_op1[e] = std::bit_cast<uint32_t>(golden_op1_f32[e]);
    }

    experimental::ProgramRunArgs params;
    params.kernel_run_args = {
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = READER,
            .runtime_arg_values =
                {{node,
                  {{"src0_addr", static_cast<uint32_t>(inp0_dram->address())},
                   {"src0_bank_id", 0u},
                   {"src1_addr", static_cast<uint32_t>(inp1_dram->address())},
                   {"src1_bank_id", 0u},
                   {"src2_addr", static_cast<uint32_t>(inp2_dram->address())},
                   {"src2_bank_id", 0u},
                   {"src3_addr", static_cast<uint32_t>(inp3_dram->address())},
                   {"src3_bank_id", 0u},
                   {"src4_addr", static_cast<uint32_t>(inp4_dram->address())},
                   {"src4_bank_id", 0u},
                   {"src5_addr", static_cast<uint32_t>(inp5_dram->address())},
                   {"src5_bank_id", 0u},
                   {"num_tiles", 1u}}}},
        },
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = WRITER0,
            .runtime_arg_values =
                {{node,
                  {{"dst_addr", static_cast<uint32_t>(out0_dram->address())}, {"bank_id", 0u}, {"num_tiles", 1u}}}},
        },
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = WRITER1,
            .runtime_arg_values =
                {{node,
                  {{"dst_addr", static_cast<uint32_t>(out1_dram->address())}, {"bank_id", 0u}, {"num_tiles", 1u}}}},
        },
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = WRITER2,
            .runtime_arg_values =
                {{node,
                  {{"dst_addr", static_cast<uint32_t>(out2_dram->address())}, {"bank_id", 0u}, {"num_tiles", 1u}}}},
        },
        experimental::ProgramRunArgs::KernelRunArgs{.kernel = COMPUTE},
    };
    experimental::SetProgramRunArgs(program, params);

    auto* dev = mesh_device->get_devices()[0];
    tt_metal::detail::LaunchProgram(dev, program, /*wait_until_cores_done=*/true);

    std::vector<uint32_t> out0_data;
    std::vector<uint32_t> out1_data;
    std::vector<uint32_t> out2_data;
    distributed::ReadShard(cq, out0_data, out0_dram, zero_coord, false);
    distributed::ReadShard(cq, out1_data, out1_dram, zero_coord, false);
    distributed::ReadShard(cq, out2_data, out2_dram, zero_coord, false);

    bool pass = true;

    auto check_bf16_tile =
        [&](const std::vector<uint32_t>& device_data, const std::vector<uint32_t>& golden_packed, uint32_t op_idx) {
            auto device_unpacked = unpack_vector<bfloat16, uint32_t>(device_data);
            auto golden_unpacked = unpack_vector<bfloat16, uint32_t>(golden_packed);
            uint32_t mismatches = 0;
            int first_mismatch_local = -1;
            float worst_absdiff = 0.0f;
            for (uint32_t e = 0; e < elems_per_tile; ++e) {
                const bfloat16 a = device_unpacked[e];
                const bfloat16 b = golden_unpacked[e];
                if (!is_close(a, b, 0.0155f, 0.001f)) {
                    if (first_mismatch_local < 0) {
                        first_mismatch_local = static_cast<int>(e);
                    }
                    ++mismatches;
                    const float absdiff = std::fabs(static_cast<float>(a) - static_cast<float>(b));
                    worst_absdiff = std::fmax(worst_absdiff, absdiff);
                }
            }
            if (mismatches == 0) {
                log_info(tt::LogTest, "OP[{}]: PASS", op_idx);
            } else {
                pass = false;
                log_error(
                    tt::LogTest,
                    "OP[{}]: FAIL ({}/{} mismatches; first idx {}; worst absdiff={:.4f})",
                    op_idx,
                    mismatches,
                    elems_per_tile,
                    first_mismatch_local,
                    worst_absdiff);
            }
        };

    auto check_f32_tile =
        [&](const std::vector<uint32_t>& device_data, const std::vector<uint32_t>& golden_packed, uint32_t op_idx) {
            uint32_t mismatches = 0;
            int first_mismatch_local = -1;
            float worst_absdiff = 0.0f;
            for (uint32_t e = 0; e < elems_per_tile; ++e) {
                const float a = std::bit_cast<float>(device_data[e]);
                const float b = std::bit_cast<float>(golden_packed[e]);
                if (!is_close(a, b, 0.001f, 0.0001f)) {
                    if (first_mismatch_local < 0) {
                        first_mismatch_local = static_cast<int>(e);
                    }
                    ++mismatches;
                    const float absdiff = std::fabs(a - b);
                    worst_absdiff = std::fmax(worst_absdiff, absdiff);
                }
            }
            if (mismatches == 0) {
                log_info(tt::LogTest, "OP[{}]: PASS", op_idx);
            } else {
                pass = false;
                log_error(
                    tt::LogTest,
                    "OP[{}]: FAIL ({}/{} mismatches; first idx {}; worst absdiff={:.4f})",
                    op_idx,
                    mismatches,
                    elems_per_tile,
                    first_mismatch_local,
                    worst_absdiff);
            }
        };

    check_bf16_tile(out0_data, packed_golden_op0, 0);
    check_f32_tile(out1_data, packed_golden_op1, 1);
    check_bf16_tile(out2_data, packed_golden_op2, 2);

    return pass;
}
}  // namespace unit_tests::compute::reconfig

////////////////////////////////////////////////////////////////////////////
//                             Test Description
// ------------------------------------------------------------------------
// These tests aim to cover usage of these API calls:
// - copy_tile_init
// - copy_tile_to_dst_init_short
// - copy_tile_to_dst_init_short_with_dt
// - unpack_reconfig_data_format
// - unpack_reconfig_data_format_srca
// - unpack_reconfig_data_format_srcb
// - pack_reconfig_data_format (Quasar DFB)
// - pack_reconfig_l1_acc
////////////////////////////////////////////////////////////////////////////

TEST_F(LLKMeshDeviceFixture, TensixTileCopyReconfigExplicitSplitDstAcc) {
    for (bool explicit_reconfig : {true, false}) {
        for (bool split_src_reconfig : {true, false}) {
            for (bool fp32_dest_acc_en : {true, false}) {
                for (bool block_copy : {true, false}) {
                    for (bool dst_full_sync_en : {true, false}) {
                        log_info(
                            LogTest,
                            "Block Copy = {}, "
                            "Explicit = {}, "
                            "Split = {}, "
                            "FP32DestAcc = {}"
                            "DstSyncFull = {}.",
                            block_copy,
                            explicit_reconfig,
                            split_src_reconfig,
                            fp32_dest_acc_en,
                            dst_full_sync_en);
                        unit_tests::compute::reconfig::ReconfigConfig test_config = {
                            .num_tiles = 1,
                            .ublock_size_tiles = 1,
                            .explicit_reconfig = explicit_reconfig,
                            .split_src_reconfig = split_src_reconfig,
                            .fp32_dest_acc_en = fp32_dest_acc_en,
                            .block_copy = block_copy,
                            .dst_full_sync_en = dst_full_sync_en};
                        for (unsigned int id = 0; id < num_devices_; id++) {
                            ASSERT_TRUE(
                                unit_tests::compute::reconfig::single_core_reconfig(devices_.at(id), test_config));
                        }
                    }
                }
            }
        }
    }
}

TEST_F(LLKMeshDeviceFixture, TensixTileCopyReconfigL1Acc) {
    for (bool l1_acc : {true, false}) {
        for (bool dst_full_sync_en : {true, false}) {
            log_info(LogTest, "L1 accumulation is {}, DstSyncFull = {}", l1_acc ? "on." : "off.", dst_full_sync_en);
            unit_tests::compute::reconfig::ReconfigConfig test_config = {
                .num_tiles = 1, .ublock_size_tiles = 1, .dst_full_sync_en = dst_full_sync_en};
            for (unsigned int id = 0; id < num_devices_; id++) {
                ASSERT_TRUE(unit_tests::compute::reconfig::single_core_reconfig(devices_.at(id), test_config));
            }
        }
    }
}

TEST_F(LLKQuasarMeshDeviceSingleCardFixture, TensixUnpackReconfigQuasarDfb) {
    for (unsigned int id = 0; id < num_devices_; id++) {
        ASSERT_TRUE(unit_tests::compute::reconfig::single_core_unpack_reconfig_quasar(devices_.at(id)));
    }
}

TEST_F(LLKQuasarMeshDeviceSingleCardFixture, TensixPackReconfigQuasarDfb) {
    for (unsigned int id = 0; id < num_devices_; id++) {
        ASSERT_TRUE(unit_tests::compute::reconfig::single_core_pack_reconfig_quasar(devices_.at(id)));
    }
}

}  // namespace tt::tt_metal
