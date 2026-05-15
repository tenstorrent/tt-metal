// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/base.h>
#include <fmt/format.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
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

// TF32 is float32 with the low 13 mantissa bits discarded (1 sign + 8 exp + 10 mant).
// Quantizing the host stimulus matches what the unpacker gasket actually delivers to
// the math engine, so the golden computed in float space stays bit-exact w.r.t. HW.
inline float quantize_to_tf32(float f) { return std::bit_cast<float>(std::bit_cast<uint32_t>(f) & 0xFFFFE000u); }

bool single_core_reconfig_quasar(const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    constexpr uint32_t kNumOps = 3;
    const uint32_t f16_tile_size = tt::tile_size(tt::DataFormat::Float16_b);
    const uint32_t tf32_tile_size = tt::tile_size(tt::DataFormat::Tf32);
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
    distributed::DeviceLocalBufferConfig tf32_dram_cfg{
        .page_size = tf32_tile_size, .buffer_type = tt::tt_metal::BufferType::DRAM, .bottom_up = false};
    distributed::ReplicatedBufferConfig f16_buf_cfg{.size = f16_tile_size};
    distributed::ReplicatedBufferConfig tf32_buf_cfg{.size = tf32_tile_size};
    distributed::ReplicatedBufferConfig out_buf_cfg{.size = out_bytes};

    auto inp0_dram = distributed::MeshBuffer::create(f16_buf_cfg, f16_dram_cfg, mesh_device.get());
    auto inp1_dram = distributed::MeshBuffer::create(f16_buf_cfg, f16_dram_cfg, mesh_device.get());
    auto inp2_dram = distributed::MeshBuffer::create(tf32_buf_cfg, tf32_dram_cfg, mesh_device.get());
    auto inp3_dram = distributed::MeshBuffer::create(tf32_buf_cfg, tf32_dram_cfg, mesh_device.get());
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
    // d2, d3 use Tf32 to exercise reconfig_data_format on Quasar. Each reconfig
    // reprograms the unpacker OUT_DATA_FORMAT register across the sequence:
    // Float16_b (d0, d1) -> Tf32 (d2, d3) -> Float16_b (d4, d5). Tf32 stays in the
    // same exp-B family as Float16_b (so the host JIT consistency check passes) and
    // is one of the few pairs accepted by the Quasar unpacker gasket. Tf32 is stored
    // as 4 bytes per element (vs 2 for bfloat16), so entry_size is 2x.
    tt_metal::experimental::dfb::DataflowBufferConfig tf32_input_dfb_cfg = {
        .entry_size = tf32_tile_size,
        .num_entries = 1,
        .num_producers = 1,
        .pap = tt_metal::experimental::dfb::AccessPattern::STRIDED,
        .num_consumers = 1,
        .cap = tt_metal::experimental::dfb::AccessPattern::STRIDED,
        .enable_implicit_sync = false,
        .data_format = tt::DataFormat::Tf32,
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
    const uint32_t inp2_dfb = tt_metal::experimental::dfb::CreateDataflowBuffer(program_, core, tf32_input_dfb_cfg);
    const uint32_t inp3_dfb = tt_metal::experimental::dfb::CreateDataflowBuffer(program_, core, tf32_input_dfb_cfg);
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
            .num_threads_per_cluster = 1,
            .math_fidelity = MathFidelity::HiFi4,
            // Tf32 -> Tf32 in the Quasar unpacker pair table requires en_32bit_dest, which
            // is driven by DST_ACCUM_MODE (= fp32_dest_acc_en) on the math side.
            .fp32_dest_acc_en = true,
            .compile_args = compute_cta});

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
    // are bfloat16-packed; inputs 2/3 are Tf32 (one float per uint32, low 13 mantissa
    // bits masked to mirror the HW gasket).
    constexpr int kRandMax = 2;
    auto src0 = create_random_vector_of_bfloat16(f16_tile_size, kRandMax, /*seed=*/0x1001);
    auto src1 = create_random_vector_of_bfloat16(f16_tile_size, kRandMax, /*seed=*/0x1002);
    auto gen_tf32_stimulus = [&](uint32_t seed) {
        std::vector<uint32_t> out(tf32_tile_size / sizeof(uint32_t), 0);
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> dist(0.0f, static_cast<float>(kRandMax));
        for (auto& word : out) {
            word = std::bit_cast<uint32_t>(quantize_to_tf32(dist(rng)));
        }
        return out;
    };
    auto src2 = gen_tf32_stimulus(0x1003);
    auto src3 = gen_tf32_stimulus(0x1004);
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
    auto unpack_tf32 = [](const std::vector<uint32_t>& packed) {
        std::vector<float> out;
        out.reserve(packed.size());
        for (uint32_t w : packed) {
            out.push_back(std::bit_cast<float>(w));
        }
        return out;
    };
    auto in2 = unpack_tf32(src2);
    auto in3 = unpack_tf32(src3);
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

    auto device_unpacked = unpack_vector<bfloat16, uint32_t>(dest_buffer_data);
    auto golden_unpacked = unpack_vector<bfloat16, uint32_t>(packed_golden);

    // Per-op diagnostic: one labeled section per op showing format chain, pass/fail, and the
    // result + golden tiles laid out 32x32 as floats. Always prints, regardless of pass/fail,
    // so it's obvious which op produced which tile and how it compares to the golden.
    struct OpStep {
        const char* label;
        tt::DataFormat src_a_fmt;
        tt::DataFormat src_b_fmt;
        tt::DataFormat out_fmt;
    };
    const std::array<OpStep, kNumOps> op_chain = {{
        {"OP[0]: add_tiles(d0, d1) -> dst[0]  [no reconfig, initial path]",
         tt::DataFormat::Float16_b,
         tt::DataFormat::Float16_b,
         tt::DataFormat::Float16_b},
        {"OP[1]: add_tiles(d2, d3) -> dst[1]  [after reconfig Float16_b -> Tf32]",
         tt::DataFormat::Tf32,
         tt::DataFormat::Tf32,
         tt::DataFormat::Float16_b},
        {"OP[2]: add_tiles(d4, d5) -> dst[2]  [after reconfig Tf32 -> Float16_b]",
         tt::DataFormat::Float16_b,
         tt::DataFormat::Float16_b,
         tt::DataFormat::Float16_b},
    }};
    auto fmt_to_str = [](tt::DataFormat f) -> const char* {
        switch (f) {
            case tt::DataFormat::Float16_b: return "Float16_b";
            case tt::DataFormat::Float16: return "Float16";
            case tt::DataFormat::Float32: return "Float32";
            case tt::DataFormat::Tf32: return "Tf32";
            default: return "?";
        }
    };

    constexpr uint32_t kTileR = 32;
    constexpr uint32_t kTileC = 32;
    auto dump_tile_floats = [&](const std::vector<bfloat16>& vec, uint32_t tile_idx) {
        for (uint32_t row = 0; row < kTileR; ++row) {
            std::string row_str;
            for (uint32_t col = 0; col < kTileC; ++col) {
                const float v = static_cast<float>(vec[tile_idx * elems_per_tile + row * kTileC + col]);
                row_str += fmt::format("{:>7.3f} ", v);
            }
            log_info(tt::LogTest, "    {}", row_str);
        }
    };

    bool pass = true;
    for (uint32_t t = 0; t < kNumOps; ++t) {
        const OpStep& s = op_chain[t];

        uint32_t mismatches = 0;
        int first_mismatch_local = -1;
        float worst_absdiff = 0.0f;
        for (uint32_t e = 0; e < elems_per_tile; ++e) {
            const float af = static_cast<float>(device_unpacked[t * elems_per_tile + e]);
            const float bf = static_cast<float>(golden_unpacked[t * elems_per_tile + e]);
            const float absdiff = std::fabs(af - bf);
            const float reldenom = std::fmax(std::fabs(af), std::fabs(bf));
            const bool ok = (absdiff <= 0.001f) || (absdiff <= 0.0155f * reldenom);
            if (!ok) {
                if (first_mismatch_local < 0) {
                    first_mismatch_local = static_cast<int>(e);
                }
                ++mismatches;
                worst_absdiff = std::fmax(worst_absdiff, absdiff);
            }
        }
        const bool tile_pass = (mismatches == 0);
        if (!tile_pass) {
            pass = false;
        }

        log_info(tt::LogTest, "================================================================");
        log_info(tt::LogTest, "{}", s.label);
        log_info(
            tt::LogTest,
            "  format:  srcA={}  srcB={}  output={}",
            fmt_to_str(s.src_a_fmt),
            fmt_to_str(s.src_b_fmt),
            fmt_to_str(s.out_fmt));
        if (tile_pass) {
            log_info(tt::LogTest, "  status:  PASS  ({}/{} elements within tolerance)", elems_per_tile, elems_per_tile);
        } else {
            log_error(
                tt::LogTest,
                "  status:  FAIL  ({}/{} mismatches; first at local idx {} (global {}); worst absdiff={:.4f})",
                mismatches,
                elems_per_tile,
                first_mismatch_local,
                t * elems_per_tile + first_mismatch_local,
                worst_absdiff);
        }
        log_info(tt::LogTest, "  result tile [device] (32x32 floats):");
        dump_tile_floats(device_unpacked, t);
        log_info(tt::LogTest, "  golden tile (32x32 floats):");
        dump_tile_floats(golden_unpacked, t);
    }
    log_info(tt::LogTest, "================================================================");

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
