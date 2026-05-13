// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/base.h>
#include <gtest/gtest.h>
#include <cstddef>
#include <cstdint>
#include <tt-metalium/bfloat8.hpp>
#include <bit>
#include <functional>
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
#include <tt-metalium/experimental/host_api.hpp>
#include <tt-metalium/experimental/dataflow_buffer/dataflow_buffer.hpp>

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
    uint32_t single_tile_size_fp32 = 4 * 32 * 32;        // Single 32x32 tile size for Float32
    uint32_t single_tile_size_bfp16b = 2 * 32 * 32;      // Single 32x32 tile size for Float16_b
    uint32_t single_tile_size_bfp8b = (1 * 32 * 32) + 64;  // Single 32x32 tile size for Bfp8_b
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
    std::vector<uint32_t> packed_golden1 = pack_as_bfp8_tiles(tt::stl::make_const_span(golden1), true, false);

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

// Minimal IEEE 754 binary16 (fp16) <-> float32 helpers used to populate the two
// Float16-formatted DFBs (d2, d3) in single_core_reconfig_quasar and to compute
// their golden contributions. Random stimulus is in [0, 2), so subnormal / inf /
// NaN paths are unreachable; only +/-0 needs the explicit short-circuit.
inline uint16_t float_to_fp16_bits(float f) {
    const uint32_t x = std::bit_cast<uint32_t>(f);
    if ((x & 0x7FFFFFFFu) == 0) {
        return static_cast<uint16_t>((x >> 16) & 0x8000u);
    }
    const uint32_t sign = (x >> 16) & 0x8000u;
    const uint32_t exp = ((x >> 23) & 0xFFu) - 112u;  // bias diff: 127 - 15
    const uint32_t mant = (x >> 13) & 0x3FFu;
    return static_cast<uint16_t>(sign | (exp << 10) | mant);
}

inline float fp16_bits_to_float(uint16_t h) {
    if ((h & 0x7FFFu) == 0) {
        return std::bit_cast<float>(static_cast<uint32_t>(h & 0x8000u) << 16);
    }
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    const uint32_t exp = static_cast<uint32_t>((h >> 10) & 0x1Fu) + 112u;
    const uint32_t mant = static_cast<uint32_t>(h & 0x3FFu);
    return std::bit_cast<float>(sign | (exp << 23) | (mant << 13));
}

bool single_core_reconfig_quasar(const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    constexpr uint32_t kNumOps = 3;
    const uint32_t f16_tile_size = tt::tile_size(tt::DataFormat::Float16_b);
    const uint32_t out_bytes = kNumOps * f16_tile_size;

    const CoreCoord core = {0, 0};
    auto& cq = mesh_device->mesh_command_queue();
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    distributed::MeshWorkload workload;
    tt_metal::Program program = tt_metal::CreateProgram();
    workload.add_program(device_range, std::move(program));
    auto& program_ = workload.get_programs().at(device_range);

    distributed::DeviceLocalBufferConfig f16_dram_cfg{
        .page_size = f16_tile_size, .buffer_type = tt::tt_metal::BufferType::DRAM, .bottom_up = false};
    distributed::ReplicatedBufferConfig f16_buf_cfg{.size = f16_tile_size};
    distributed::ReplicatedBufferConfig out_buf_cfg{.size = out_bytes};

    auto inp0_dram = distributed::MeshBuffer::create(f16_buf_cfg, f16_dram_cfg, mesh_device.get());
    auto inp1_dram = distributed::MeshBuffer::create(f16_buf_cfg, f16_dram_cfg, mesh_device.get());
    auto inp2_dram = distributed::MeshBuffer::create(f16_buf_cfg, f16_dram_cfg, mesh_device.get());
    auto inp3_dram = distributed::MeshBuffer::create(f16_buf_cfg, f16_dram_cfg, mesh_device.get());
    auto inp4_dram = distributed::MeshBuffer::create(f16_buf_cfg, f16_dram_cfg, mesh_device.get());
    auto inp5_dram = distributed::MeshBuffer::create(f16_buf_cfg, f16_dram_cfg, mesh_device.get());
    auto out_dram = distributed::MeshBuffer::create(out_buf_cfg, f16_dram_cfg, mesh_device.get());

    tt_metal::experimental::dfb::DataflowBufferConfig f16_input_dfb_cfg = {
        .entry_size = f16_tile_size,
        .num_entries = 1,
        .num_producers = 1,
        .pap = tt_metal::experimental::dfb::AccessPattern::STRIDED,
        .num_consumers = 1,
        .cap = tt_metal::experimental::dfb::AccessPattern::STRIDED,
        .enable_implicit_sync = false,
        .data_format = tt::DataFormat::Float16_b,
    };
    // d2, d3 use Float16 (fp16) to exercise reconfig_data_format on Quasar.
    // Each reconfig must reprogram the unpacker OUT_DATA_FORMAT register:
    // Float16_b (d0, d1) -> Float16 (d2, d3) -> Float16_b (d4, d5). fp16 and
    // bfloat16 share the same per-element byte size, so entry_size matches.
    tt_metal::experimental::dfb::DataflowBufferConfig fp16_input_dfb_cfg = {
        .entry_size = f16_tile_size,
        .num_entries = 1,
        .num_producers = 1,
        .pap = tt_metal::experimental::dfb::AccessPattern::STRIDED,
        .num_consumers = 1,
        .cap = tt_metal::experimental::dfb::AccessPattern::STRIDED,
        .enable_implicit_sync = false,
        .data_format = tt::DataFormat::Float16,
    };
    tt_metal::experimental::dfb::DataflowBufferConfig out_dfb_cfg = {
        .entry_size = f16_tile_size,
        .num_entries = kNumOps,
        .num_producers = 1,
        .pap = tt_metal::experimental::dfb::AccessPattern::STRIDED,
        .num_consumers = 1,
        .cap = tt_metal::experimental::dfb::AccessPattern::STRIDED,
        .enable_implicit_sync = false,
        .data_format = tt::DataFormat::Float16_b,
    };

    const uint32_t inp0_dfb = tt_metal::experimental::dfb::CreateDataflowBuffer(program_, core, f16_input_dfb_cfg);
    const uint32_t inp1_dfb = tt_metal::experimental::dfb::CreateDataflowBuffer(program_, core, f16_input_dfb_cfg);
    const uint32_t inp2_dfb = tt_metal::experimental::dfb::CreateDataflowBuffer(program_, core, fp16_input_dfb_cfg);
    const uint32_t inp3_dfb = tt_metal::experimental::dfb::CreateDataflowBuffer(program_, core, fp16_input_dfb_cfg);
    const uint32_t inp4_dfb = tt_metal::experimental::dfb::CreateDataflowBuffer(program_, core, f16_input_dfb_cfg);
    const uint32_t inp5_dfb = tt_metal::experimental::dfb::CreateDataflowBuffer(program_, core, f16_input_dfb_cfg);
    const uint32_t out_dfb = tt_metal::experimental::dfb::CreateDataflowBuffer(program_, core, out_dfb_cfg);

    std::vector<uint32_t> reader_cta = {inp0_dfb, inp1_dfb, inp2_dfb, inp3_dfb, inp4_dfb, inp5_dfb};
    std::vector<uint32_t> writer_cta = {out_dfb};
    std::vector<uint32_t> compute_cta = {inp0_dfb, inp1_dfb, inp2_dfb, inp3_dfb, inp4_dfb, inp5_dfb, out_dfb};

    auto reader_kernel = tt_metal::experimental::quasar::CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_six_input.cpp",
        core,
        tt_metal::experimental::quasar::QuasarDataMovementConfig{
            .num_threads_per_cluster = 1, .compile_args = reader_cta});

    auto writer_kernel = tt_metal::experimental::quasar::CreateKernel(
        program_,
        "tt_metal/kernels/dataflow/writer_unary.cpp",
        core,
        tt_metal::experimental::quasar::QuasarDataMovementConfig{
            .num_threads_per_cluster = 1, .compile_args = writer_cta});

    auto compute_kernel = tt_metal::experimental::quasar::CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/compute/reconfig_quasar.cpp",
        core,
        tt_metal::experimental::quasar::QuasarComputeConfig{
            .num_threads_per_cluster = 1, .math_fidelity = MathFidelity::HiFi4, .compile_args = compute_cta});

    tt_metal::experimental::dfb::BindDataflowBufferToProducerConsumerKernels(
        program_, inp0_dfb, reader_kernel, compute_kernel);
    tt_metal::experimental::dfb::BindDataflowBufferToProducerConsumerKernels(
        program_, inp1_dfb, reader_kernel, compute_kernel);
    tt_metal::experimental::dfb::BindDataflowBufferToProducerConsumerKernels(
        program_, inp2_dfb, reader_kernel, compute_kernel);
    tt_metal::experimental::dfb::BindDataflowBufferToProducerConsumerKernels(
        program_, inp3_dfb, reader_kernel, compute_kernel);
    tt_metal::experimental::dfb::BindDataflowBufferToProducerConsumerKernels(
        program_, inp4_dfb, reader_kernel, compute_kernel);
    tt_metal::experimental::dfb::BindDataflowBufferToProducerConsumerKernels(
        program_, inp5_dfb, reader_kernel, compute_kernel);
    tt_metal::experimental::dfb::BindDataflowBufferToProducerConsumerKernels(
        program_, out_dfb, compute_kernel, writer_kernel);

    // Random stimulus: U(0, 2) per element, distinct seeds per input. Inputs 0/1/4/5
    // are bfloat16-packed; inputs 2/3 are fp16-bit-packed to match their DFB format.
    constexpr int kRandMax = 2;
    auto src0 = create_random_vector_of_bfloat16(f16_tile_size, kRandMax, /*seed=*/0x1001);
    auto src1 = create_random_vector_of_bfloat16(f16_tile_size, kRandMax, /*seed=*/0x1002);
    auto gen_fp16_stimulus = [&](uint32_t seed) {
        std::vector<uint32_t> out(f16_tile_size / sizeof(uint32_t), 0);
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> dist(0.0f, static_cast<float>(kRandMax));
        for (auto& word : out) {
            const uint16_t lo = float_to_fp16_bits(dist(rng));
            const uint16_t hi = float_to_fp16_bits(dist(rng));
            word = static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16);
        }
        return out;
    };
    auto src2 = gen_fp16_stimulus(0x1003);
    auto src3 = gen_fp16_stimulus(0x1004);
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
    auto unpack_fp16 = [](const std::vector<uint32_t>& packed) {
        std::vector<float> out;
        out.reserve(packed.size() * 2);
        for (uint32_t w : packed) {
            out.push_back(fp16_bits_to_float(static_cast<uint16_t>(w & 0xFFFFu)));
            out.push_back(fp16_bits_to_float(static_cast<uint16_t>(w >> 16)));
        }
        return out;
    };
    auto in2 = unpack_fp16(src2);
    auto in3 = unpack_fp16(src3);
    auto in4 = unpack_uint32_vec_into_bfloat16_vec(src4);
    auto in5 = unpack_uint32_vec_into_bfloat16_vec(src5);

    const uint32_t elems_per_tile = f16_tile_size / sizeof(bfloat16);
    std::vector<bfloat16> golden(kNumOps * elems_per_tile);
    for (uint32_t e = 0; e < elems_per_tile; ++e) {
        golden[0 * elems_per_tile + e] = bfloat16(static_cast<float>(in0[e]) + static_cast<float>(in1[e]));
        golden[1 * elems_per_tile + e] = bfloat16(static_cast<float>(in2[e]) + static_cast<float>(in3[e]));
        golden[2 * elems_per_tile + e] = bfloat16(static_cast<float>(in4[e]) + static_cast<float>(in5[e]));
    }
    auto packed_golden = pack_vector<uint32_t, bfloat16>(golden);

    tt_metal::SetRuntimeArgs(
        program_,
        reader_kernel,
        core,
        {
            static_cast<uint32_t>(inp0_dram->address()),
            0u,
            static_cast<uint32_t>(inp1_dram->address()),
            0u,
            static_cast<uint32_t>(inp2_dram->address()),
            0u,
            static_cast<uint32_t>(inp3_dram->address()),
            0u,
            static_cast<uint32_t>(inp4_dram->address()),
            0u,
            static_cast<uint32_t>(inp5_dram->address()),
            0u,
            1u,  // num_tiles
        });
    tt_metal::SetRuntimeArgs(program_, writer_kernel, core, {static_cast<uint32_t>(out_dram->address()), 0u, kNumOps});

    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);

    std::vector<uint32_t> dest_buffer_data;
    distributed::ReadShard(cq, dest_buffer_data, out_dram, zero_coord, false);

    int failed_index = -1;
    bool pass = is_close_packed_vectors<bfloat16, uint32_t>(
        dest_buffer_data,
        packed_golden,
        [&](const bfloat16& a, const bfloat16& b) { return is_close(a, b, 0.0155f); },
        &failed_index);
    if (not pass) {
        log_info(tt::LogTest, "Failed Index={}", failed_index);
        log_info(tt::LogTest, "Device output:");
        print_vector_fixed_numel_per_row(unpack_vector<bfloat16, uint32_t>(dest_buffer_data), 32);
        log_info(tt::LogTest, "Golden:");
        print_vector_fixed_numel_per_row(unpack_vector<bfloat16, uint32_t>(packed_golden), 32);
    }
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

TEST_F(LLKQuasarMeshDeviceSingleCardFixture, TensixComputeReconfigQuasarDfb) {
    for (unsigned int id = 0; id < num_devices_; id++) {
        ASSERT_TRUE(unit_tests::compute::reconfig::single_core_reconfig_quasar(devices_.at(id)));
    }
}

}  // namespace tt::tt_metal
