// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <variant>
#include "device_fixture.hpp"
#include "tt_metal/common/bfloat8.hpp"
#include "tt_metal/test_utils/comparison.hpp"

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
bool single_core_reconfig(tt_metal::Device* device, const ReconfigConfig& test_config) {

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
    uint32_t single_tile_size_fp32 = 4 * 32 * 32;              // Single 32x32 tile size for Float32
    uint32_t single_tile_size_bfp16b = 2 * 32 * 32;            // Single 32x32 tile size for Float16_b
    uint32_t single_tile_size_bfp8b = 1 * 32 * 32 + 64;        // Single 32x32 tile size for Bfp8_b
    uint32_t single_tile_size_out0 = test_config.fp32_dest_acc_en ? single_tile_size_fp32 : single_tile_size_bfp16b;
    const size_t dram_buffer_size_bfp16b = test_config.num_tiles * single_tile_size_bfp16b;
    const size_t dram_buffer_size_bfp8b = test_config.num_tiles * single_tile_size_bfp8b;
    const size_t dram_buffer_size_out0 = test_config.num_tiles * single_tile_size_out0;

    CoreCoord core = {0, 0};
    tt_metal::Program program = tt_metal::CreateProgram();

    tt::tt_metal::InterleavedBufferConfig dram_config_bfp16b{
        .device = device, .size = dram_buffer_size_bfp16b, .page_size = dram_buffer_size_bfp16b, .buffer_type = tt::tt_metal::BufferType::DRAM};

    tt::tt_metal::InterleavedBufferConfig dram_config_bfp8b{
        .device = device, .size = dram_buffer_size_bfp8b, .page_size = dram_buffer_size_bfp8b, .buffer_type = tt::tt_metal::BufferType::DRAM};

    tt::tt_metal::InterleavedBufferConfig dram_config_out0{
        .device = device, .size = dram_buffer_size_out0, .page_size = dram_buffer_size_out0, .buffer_type = tt::tt_metal::BufferType::DRAM};

    // This will be srcB in Bfp8_b
    auto input0_dram_buffer = CreateBuffer(dram_config_bfp8b);
    uint32_t input0_dram_byte_address = input0_dram_buffer->address();
    auto input0_dram_noc_xy = input0_dram_buffer->noc_coordinates();

    // This will be srcA in Float16_b
    auto input1_dram_buffer = CreateBuffer(dram_config_bfp16b);
    uint32_t input1_dram_byte_address = input1_dram_buffer->address();
    auto input1_dram_noc_xy = input1_dram_buffer->noc_coordinates();

    // This will be DEST in Float16_b
    auto input2_dram_buffer = CreateBuffer(dram_config_bfp16b);
    uint32_t input2_dram_byte_address = input2_dram_buffer->address();
    auto input2_dram_noc_xy = input2_dram_buffer->noc_coordinates();

    // This will be Output0 in Float32 or Float16_b depending on fp32_dest_acc_en
    auto output0_dram_buffer = CreateBuffer(dram_config_out0);
    uint32_t output0_dram_byte_address = output0_dram_buffer->address();
    auto output0_dram_noc_xy = output0_dram_buffer->noc_coordinates();

    // This will be Output1 in Bfp8_b
    auto output1_dram_buffer = CreateBuffer(dram_config_bfp8b);
    uint32_t output1_dram_byte_address = output1_dram_buffer->address();
    auto output1_dram_noc_xy = output1_dram_buffer->noc_coordinates();

    tt_metal::CircularBufferConfig l1_input0_cb_config =
        tt_metal::CircularBufferConfig(dram_buffer_size_bfp8b, {{in0_id, tt::DataFormat::Bfp8_b}})
            .set_page_size(in0_id, single_tile_size_bfp8b);
    auto l1_input0_cb = tt_metal::CreateCircularBuffer(program, core, l1_input0_cb_config);

    tt_metal::CircularBufferConfig l1_input1_cb_config =
        tt_metal::CircularBufferConfig(dram_buffer_size_bfp16b, {{in1_id, tt::DataFormat::Float16_b}})
            .set_page_size(in1_id, single_tile_size_bfp16b);
    auto l1_input1_cb = tt_metal::CreateCircularBuffer(program, core, l1_input1_cb_config);

    tt_metal::CircularBufferConfig l1_input2_cb_config =
        tt_metal::CircularBufferConfig(dram_buffer_size_bfp16b, {{in2_id, tt::DataFormat::Float16_b}})
            .set_page_size(in2_id, single_tile_size_bfp16b);
    auto l1_input2_cb = tt_metal::CreateCircularBuffer(program, core, l1_input2_cb_config);

    tt_metal::CircularBufferConfig l1_output0_cb_config =
        tt_metal::CircularBufferConfig(dram_buffer_size_out0, {{out0_id, (test_config.fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b)}})
            .set_page_size(out0_id, single_tile_size_out0);
    auto l1_output0_cb = tt_metal::CreateCircularBuffer(program, core, l1_output0_cb_config);

    tt_metal::CircularBufferConfig l1_output1_cb_config =
        tt_metal::CircularBufferConfig(dram_buffer_size_bfp8b, {{out1_id, tt::DataFormat::Bfp8_b}})
            .set_page_size(out1_id, single_tile_size_bfp8b);
    auto l1_output1_cb = tt_metal::CreateCircularBuffer(program, core, l1_output1_cb_config);

    vector<uint32_t> compute_kernel_args = {};
    std::map<string, string> defines;


    defines["DST_ACCUM_MODE"] = "1"; // Needed always in order for reader kernel to load data from CB2
    defines["EXPLICIT_RECONFIG"] = test_config.explicit_reconfig ? "1" : "0";
    defines["SPLIT_SRC_RECONFIG"] = test_config.split_src_reconfig ? "1" : "0";
    defines["BLOCK_COPY"] = test_config.block_copy ? "1" : "0";
    defines["L1_ACC"] = test_config.l1_acc ? "1" : "0";

    auto reader_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_binary.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default, .defines=defines});

    auto writer_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_binary.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

    auto compute_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/reconfig.cpp",
        core,
        tt_metal::ComputeConfig{
            .fp32_dest_acc_en = test_config.fp32_dest_acc_en,
            .dst_full_sync_en = test_config.dst_full_sync_en,
            .compile_args = compute_kernel_args,
            .defines = defines});

    SetRuntimeArgs(
        program,
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
    std::vector<uint32_t> src0_vec = create_constant_vector_of_bfp8(
            dram_buffer_size_bfp8b,
            in0_val,
            false);
    std::vector<uint32_t> src1_vec = create_constant_vector_of_bfloat16(
            dram_buffer_size_bfp16b,
            in1_val);
    std::vector<uint32_t> src2_vec = create_constant_vector_of_bfloat16(
            dram_buffer_size_bfp16b,
            in2_val);

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
        temp_golden[i] = input1[i].to_float() + bfloat16(input0[i]).to_float();
        // Do temp + DST, store in out0 vector depending on fp32_dest_acc_en:
        if (test_config.fp32_dest_acc_en) {
            golden0_fp32[i] = temp_golden[i] + input2[i].to_float();
        } else {
            golden0_bfp16[i] = bfloat16(temp_golden[i] + input2[i].to_float());
        }
        // Do out1 = temp + DST:
        golden1[i] = bfloat16(temp_golden[i] + input2[i].to_float()).to_float();
        // Do out0[bfp16] = temp + L1, this makes sense only if not fp32_dest_acc_en:
        if (test_config.l1_acc && !test_config.fp32_dest_acc_en) {
            golden0_bfp16[i] = bfloat16(golden0_bfp16[i].to_float() + out0_result_old);
        } else {
            out0_result_old = golden0_bfp16[i].to_float();
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
    std::vector<uint32_t> packed_golden1 = pack_fp32_vec_as_bfp8_tiles(golden1, true, false);

    // ////////////////////////////////////////////////////////////////////////////
    // //                      Compile and Execute Application
    // ////////////////////////////////////////////////////////////////////////////
    tt_metal::detail::WriteToBuffer(input0_dram_buffer, src0_vec);
    tt_metal::detail::WriteToBuffer(input1_dram_buffer, src1_vec);
    tt_metal::detail::WriteToBuffer(input2_dram_buffer, src2_vec);

    tt_metal::SetRuntimeArgs(
        program,
        reader_kernel,
        core,
        {
            (uint32_t)input0_dram_byte_address,
            (uint32_t)input0_dram_noc_xy.x,
            (uint32_t)input0_dram_noc_xy.y,
            (uint32_t)input1_dram_byte_address,
            (uint32_t)input1_dram_noc_xy.x,
            (uint32_t)input1_dram_noc_xy.y,
            (uint32_t)test_config.num_tiles,
            (uint32_t)input2_dram_byte_address,
            (uint32_t)input2_dram_noc_xy.x,
            (uint32_t)input2_dram_noc_xy.y,
        });
    tt_metal::SetRuntimeArgs(
        program,
        writer_kernel,
        core,
        {
            (uint32_t)output0_dram_byte_address,
            (uint32_t)output0_dram_noc_xy.x,
            (uint32_t)output0_dram_noc_xy.y,
            (uint32_t)out0_id,
            (uint32_t)output1_dram_byte_address,
            (uint32_t)output1_dram_noc_xy.x,
            (uint32_t)output1_dram_noc_xy.y,
            (uint32_t)out1_id,
            (uint32_t)test_config.num_tiles,
            (uint32_t)test_config.ublock_size_tiles,
        });

    tt_metal::detail::LaunchProgram(device, program);


    // ////////////////////////////////////////////////////////////////////////////
    // //                      Comparison Checking
    // ////////////////////////////////////////////////////////////////////////////
    std::vector<uint32_t> dest0_buffer_data(src1_vec.size());
    std::vector<uint32_t> dest1_buffer_data(src0_vec.size());
    tt_metal::detail::ReadFromBuffer(output0_dram_buffer, dest0_buffer_data);
    tt_metal::detail::ReadFromBuffer(output1_dram_buffer, dest1_buffer_data);

    pass &= is_close_packed_vectors<bfloat16, uint32_t>(
        dest0_buffer_data,
        packed_golden0,
        [&](const bfloat16& a, const bfloat16& b) {
            return is_close(a, b, 0.0155f);
        });
    pass &= is_close_packed_vectors<bfloat16, uint32_t>(
        dest1_buffer_data,
        packed_golden1,
        [&](const bfloat16& a, const bfloat16& b) {
            return is_close(a, b, 0.0155);
        });

    return pass;
}
}  // namespace unit_tests::compute::binary

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

TEST_F(DeviceFixture, TileCopyReconfigExplicitSplitDstAcc) {
    auto arch = this->arch_;
    if (arch == tt::ARCH::GRAYSKULL) {
        GTEST_SKIP();
    }
    for (bool explicit_reconfig : {true, false}) {
        for (bool split_src_reconfig : {true, false}) {
            for (bool fp32_dest_acc_en : {true, false}) {
                for (bool block_copy : {true, false}) {
                    for (bool dst_full_sync_en : {true, false}) {
                        log_info(LogTest, "Block Copy = {}, "
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
                            .dst_full_sync_en = dst_full_sync_en
                        };
                        for (unsigned int id = 0; id < num_devices_; id++) {
                            ASSERT_TRUE(unit_tests::compute::reconfig::single_core_reconfig(devices_.at(id), test_config));
                        }
                    }
                }
            }
        }
    }
}

TEST_F(DeviceFixture, TileCopyReconfigL1Acc) {
    auto arch = this->arch_;
    if (arch == tt::ARCH::GRAYSKULL) {
        GTEST_SKIP();
    }
    for (bool l1_acc : {true, false}) {
        for (bool dst_full_sync_en : {true, false}) {
            log_info(LogTest, "L1 accumulation is {}, DstSyncFull = {}", l1_acc ? "on." : "off.", dst_full_sync_en);
            unit_tests::compute::reconfig::ReconfigConfig test_config = {
                .num_tiles = 1,
                .ublock_size_tiles = 1,
                .dst_full_sync_en = dst_full_sync_en
            };
            for (unsigned int id = 0; id < num_devices_; id++) {
                ASSERT_TRUE(unit_tests::compute::reconfig::single_core_reconfig(devices_.at(id), test_config));
            }
        }
    }
}
