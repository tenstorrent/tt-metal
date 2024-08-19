// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <functional>
#include <random>

#include "device_fixture.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/hostdevcommon/common_runtime_address_map.h"  // FIXME: Should remove dependency on this
#include "tt_metal/test_utils/comparison.hpp"
#include "tt_metal/test_utils/df/df.hpp"
#include "tt_metal/test_utils/print_helpers.hpp"
#include "tt_metal/test_utils/stimulus.hpp"

using namespace tt;
using namespace tt::test_utils;
using namespace tt::test_utils::df;

namespace unit_tests::compute::binary {
const map<string, string> binary_op_name_to_op_type = {
    {"add", "EltwiseBinaryType::ELWADD"},
    {"sub", "EltwiseBinaryType::ELWSUB"},
    {"mul", "EltwiseBinaryType::ELWMUL"},
    {"add_with_dest_reuse", "EltwiseBinaryType::ELWADD"},
    {"sub_with_dest_reuse", "EltwiseBinaryType::ELWSUB"},
    {"mul_with_dest_reuse", "EltwiseBinaryType::ELWMUL"},
};
const map<string, string> binary_op_name_to_op_kernel = {
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
    CoreCoord core = {};
    std::string binary_op = "";
    bool acc_to_dest = false;
};
/// @brief Does Dramx2 --> Reader --> CB --> Binary Compute --> CB --> Writer --> Dram
/// @param device
/// @param test_config - Configuration of the test -- see struct
/// @return
bool single_core_binary(tt_metal::Device* device, const SingleCoreBinaryConfig& test_config) {
    bool pass = true;
    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    const size_t byte_size = test_config.num_tiles * test_config.tile_byte_size;
    tt_metal::Program program = tt_metal::CreateProgram();

    tt::tt_metal::InterleavedBufferConfig dram_config{
        .device = device, .size = byte_size, .page_size = byte_size, .buffer_type = tt::tt_metal::BufferType::DRAM};
    auto input0_dram_buffer = CreateBuffer(dram_config);
    uint32_t input0_dram_byte_address = input0_dram_buffer->address();
    auto input0_dram_noc_xy = input0_dram_buffer->noc_coordinates();

    auto input1_dram_buffer = CreateBuffer(dram_config);
    uint32_t input1_dram_byte_address = input1_dram_buffer->address();
    auto input1_dram_noc_xy = input1_dram_buffer->noc_coordinates();

    auto input2_dram_buffer = CreateBuffer(dram_config);
    uint32_t input2_dram_byte_address = input2_dram_buffer->address();
    auto input2_dram_noc_xy = input2_dram_buffer->noc_coordinates();

    auto output_dram_buffer = CreateBuffer(dram_config);
    uint32_t output_dram_byte_address = output_dram_buffer->address();
    auto output_dram_noc_xy = output_dram_buffer->noc_coordinates();

    tt_metal::CircularBufferConfig l1_cb_config =
        tt_metal::CircularBufferConfig(byte_size, {{0, test_config.l1_input_data_format}})
            .set_page_size(0, test_config.tile_byte_size);
    auto l1_input0_cb = tt_metal::CreateCircularBuffer(program, test_config.core, l1_cb_config);

    tt_metal::CircularBufferConfig l1_input1_cb_config =
        tt_metal::CircularBufferConfig(byte_size, {{1, test_config.l1_input_data_format}})
            .set_page_size(1, test_config.tile_byte_size);
    auto l1_input1_cb = tt_metal::CreateCircularBuffer(program, test_config.core, l1_input1_cb_config);

    tt_metal::CircularBufferConfig l2_input1_cb_config =
        tt_metal::CircularBufferConfig(byte_size, {{2, test_config.l1_input_data_format}})
            .set_page_size(2, test_config.tile_byte_size);
    auto l1_input2_cb = tt_metal::CreateCircularBuffer(program, test_config.core, l2_input1_cb_config);

    tt_metal::CircularBufferConfig l1_output_cb_config =
        tt_metal::CircularBufferConfig(byte_size, {{16, test_config.l1_output_data_format}})
            .set_page_size(16, test_config.tile_byte_size);
    auto l1_output_cb = tt_metal::CreateCircularBuffer(program, test_config.core, l1_output_cb_config);

    vector<uint32_t> compute_kernel_args = {};
    std::map<string, string> defines = {{"ELTWISE_OP_TYPE", binary_op_name_to_op_type.at(test_config.binary_op)}};

    if (test_config.binary_op.find("_with_dest_reuse") != std::string::npos) {
        defines["ELTWISE_DEST_REUSE_TYPE"] = "EltwiseBinaryReuseDestType::DEST_TO_SRCA";
    } else {
        defines["ELTWISE_OP"] = binary_op_name_to_op_kernel.at(test_config.binary_op);
        if(test_config.acc_to_dest) {
            defines["DST_ACCUM_MODE"] = "1";
            defines["ELTWISE_OP_INIT"] = defines["ELTWISE_OP"] + "_init";
        }
    }

    auto reader_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_binary.cpp",
        test_config.core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default, .defines=defines});

    auto writer_kernel = tt_metal::CreateKernel(
        program,
        "tt_metal/kernels/dataflow/writer_unary.cpp",
        test_config.core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

    auto binary_kernel = tt_metal::CreateKernel(
        program,
        "tt_metal/kernels/compute/eltwise_binary.cpp",
        test_config.core,
        tt_metal::ComputeConfig{.compile_args = compute_kernel_args, .defines = defines});

    SetRuntimeArgs(program, binary_kernel, test_config.core, {uint32_t(test_config.num_tiles), 1});

    ////////////////////////////////////////////////////////////////////////////
    //                      Stimulus Generation
    ////////////////////////////////////////////////////////////////////////////
    std::vector<uint32_t> packed_input0 = generate_packed_uniform_random_vector<uint32_t, tt::test_utils::df::bfloat16>(
        -1.0f,
        1.0f,
        byte_size / tt::test_utils::df::bfloat16::SIZEOF,
        std::chrono::system_clock::now().time_since_epoch().count());
    std::vector<uint32_t> packed_input1 = generate_packed_uniform_random_vector<uint32_t, tt::test_utils::df::bfloat16>(
        -1.0f,
        1.0f,
        byte_size / tt::test_utils::df::bfloat16::SIZEOF,
        std::chrono::system_clock::now().time_since_epoch().count());
    std::vector<uint32_t> packed_input2 = generate_packed_uniform_random_vector<uint32_t, tt::test_utils::df::bfloat16>(
        -1.0f,
        1.0f,
        byte_size / tt::test_utils::df::bfloat16::SIZEOF,
        std::chrono::system_clock::now().time_since_epoch().count());
    ////////////////////////////////////////////////////////////////////////////
    //                      Golden Generation
    ////////////////////////////////////////////////////////////////////////////
    auto input0 = unpack_vector<tt::test_utils::df::bfloat16, uint32_t>(packed_input0);
    auto input1 = unpack_vector<tt::test_utils::df::bfloat16, uint32_t>(packed_input1);
    auto input2 = unpack_vector<tt::test_utils::df::bfloat16, uint32_t>(packed_input2);
    std::vector<float> temp_golden(input0.size());
    std::transform(
        input0.begin(),
        input0.end(),
        input1.begin(),
        temp_golden.begin(),
        [&](const tt::test_utils::df::bfloat16& lhs, const tt::test_utils::df::bfloat16& rhs) {
            if (test_config.binary_op == "add") {
                return (lhs.to_float() + rhs.to_float());
            } else if (test_config.binary_op == "sub") {
                return (lhs.to_float() - rhs.to_float());
            } else if (test_config.binary_op == "mul") {
                return (lhs.to_float() * rhs.to_float());
            } else if (test_config.binary_op.find("with_dest_reuse") != std::string::npos) {
                return lhs.to_float();
            } else {
                TT_THROW("Unsupported binary_op={}", test_config.binary_op);
                return 0.0f;
            }
        });

    std::vector<tt::test_utils::df::bfloat16> golden(input0.size());
    std::transform(
    input2.begin(),
    input2.end(),
    temp_golden.begin(),
    golden.begin(),
    [&](const tt::test_utils::df::bfloat16& lhs, const float& rhs) {
        //acc_to_dest accumulates dest value with binary output, for all binary operations
        if (test_config.acc_to_dest || test_config.binary_op == "add_with_dest_reuse") {
            return (lhs.to_float() + rhs);
        } else if (test_config.binary_op == "sub_with_dest_reuse") {
            return (lhs.to_float() - rhs);
        } else if (test_config.binary_op == "mul_with_dest_reuse") {
            return (lhs.to_float() * rhs);
        } else {
            return rhs;
        }
    });
    auto packed_golden = pack_vector<uint32_t, tt::test_utils::df::bfloat16>(golden);

    ////////////////////////////////////////////////////////////////////////////
    //                      Compile and Execute Application
    ////////////////////////////////////////////////////////////////////////////

    tt_metal::detail::WriteToBuffer(input0_dram_buffer, packed_input0);
    tt_metal::detail::WriteToBuffer(input1_dram_buffer, packed_input1);
    tt_metal::detail::WriteToBuffer(input2_dram_buffer, packed_input2);

    tt_metal::SetRuntimeArgs(
        program,
        reader_kernel,
        test_config.core,
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
        test_config.core,
        {
            (uint32_t)output_dram_byte_address,
            (uint32_t)output_dram_noc_xy.x,
            (uint32_t)output_dram_noc_xy.y,
            (uint32_t)test_config.num_tiles,
        });

    tt_metal::detail::LaunchProgram(device, program);

    ////////////////////////////////////////////////////////////////////////////
    //                      Comparison Checking
    ////////////////////////////////////////////////////////////////////////////
    std::vector<uint32_t> dest_buffer_data;
    tt_metal::detail::ReadFromBuffer(output_dram_buffer, dest_buffer_data);
    pass &= is_close_packed_vectors<tt::test_utils::df::bfloat16, uint32_t>(
        dest_buffer_data,
        packed_golden,
        [&](const tt::test_utils::df::bfloat16& a, const tt::test_utils::df::bfloat16& b) {
            return is_close(a, b, 0.0155f);
        });
    return pass;
}
}  // namespace unit_tests::compute::binary
TEST_F(DeviceFixture, BinaryComputeSingleCoreSingleTileAdd) {
    unit_tests::compute::binary::SingleCoreBinaryConfig test_config = {
        .tile_byte_size = 2 * 32 * 32,
        .l1_input_data_format = tt::DataFormat::Float16_b,
        .l1_output_data_format = tt::DataFormat::Float16_b,
        .core = CoreCoord(0, 0),
        .binary_op = "add"};
    test_config.num_tiles = 1;
    for (unsigned int id = 0; id < num_devices_; id++) {
        ASSERT_TRUE(unit_tests::compute::binary::single_core_binary(devices_.at(id), test_config));
    }
}
TEST_F(DeviceFixture, BinaryComputeSingleCoreSingleTileSub) {
    unit_tests::compute::binary::SingleCoreBinaryConfig test_config = {
        .tile_byte_size = 2 * 32 * 32,
        .l1_input_data_format = tt::DataFormat::Float16_b,
        .l1_output_data_format = tt::DataFormat::Float16_b,
        .core = CoreCoord(0, 0),
        .binary_op = "sub"};
    test_config.num_tiles = 1;
    for (unsigned int id = 0; id < num_devices_; id++) {
        ASSERT_TRUE(unit_tests::compute::binary::single_core_binary(devices_.at(id), test_config));
    }
}
TEST_F(DeviceFixture, BinaryComputeSingleCoreSingleTileMul) {
    unit_tests::compute::binary::SingleCoreBinaryConfig test_config = {
        .tile_byte_size = 2 * 32 * 32,
        .l1_input_data_format = tt::DataFormat::Float16_b,
        .l1_output_data_format = tt::DataFormat::Float16_b,
        .core = CoreCoord(0, 0),
        .binary_op = "mul"};
    test_config.num_tiles = 1;
    for (unsigned int id = 0; id < num_devices_; id++) {
        ASSERT_TRUE(unit_tests::compute::binary::single_core_binary(devices_.at(id), test_config));
    }
}
TEST_F(DeviceFixture, BinaryComputeSingleCoreMultiTileAddWithDestReuse) {
    unit_tests::compute::binary::SingleCoreBinaryConfig test_config = {
        .tile_byte_size = 2 * 32 * 32,
        .l1_input_data_format = tt::DataFormat::Float16_b,
        .l1_output_data_format = tt::DataFormat::Float16_b,
        .core = CoreCoord(0, 0),
        .binary_op = "add_with_dest_reuse"};
    test_config.num_tiles = 4;
    for (unsigned int id = 0; id < num_devices_; id++) {
        ASSERT_TRUE(unit_tests::compute::binary::single_core_binary(devices_.at(id), test_config));
    }
}
TEST_F(DeviceFixture, BinaryComputeSingleCoreMultiTileSubWithDestReuse) {
    unit_tests::compute::binary::SingleCoreBinaryConfig test_config = {
        .tile_byte_size = 2 * 32 * 32,
        .l1_input_data_format = tt::DataFormat::Float16_b,
        .l1_output_data_format = tt::DataFormat::Float16_b,
        .core = CoreCoord(0, 0),
        .binary_op = "sub_with_dest_reuse"};
    test_config.num_tiles = 4;
    for (unsigned int id = 0; id < num_devices_; id++) {
        ASSERT_TRUE(unit_tests::compute::binary::single_core_binary(devices_.at(id), test_config));
    }
}
TEST_F(DeviceFixture, BinaryComputeSingleCoreMultiTileMulWithDestReuse) {
    unit_tests::compute::binary::SingleCoreBinaryConfig test_config = {
        .tile_byte_size = 2 * 32 * 32,
        .l1_input_data_format = tt::DataFormat::Float16_b,
        .l1_output_data_format = tt::DataFormat::Float16_b,
        .core = CoreCoord(0, 0),
        .binary_op = "mul_with_dest_reuse"};
    test_config.num_tiles = 4;
    for (unsigned int id = 0; id < num_devices_; id++) {
        ASSERT_TRUE(unit_tests::compute::binary::single_core_binary(devices_.at(id), test_config));
    }
}
TEST_F(DeviceFixture, BinaryComputeSingleCoreMultiTileAdd) {
    unit_tests::compute::binary::SingleCoreBinaryConfig test_config = {
        .tile_byte_size = 2 * 32 * 32,
        .l1_input_data_format = tt::DataFormat::Float16_b,
        .l1_output_data_format = tt::DataFormat::Float16_b,
        .core = CoreCoord(0, 0),
        .binary_op = "add"};
    test_config.num_tiles = 4;
    for (unsigned int id = 0; id < num_devices_; id++) {
        ASSERT_TRUE(unit_tests::compute::binary::single_core_binary(devices_.at(id), test_config));
    }
}
TEST_F(DeviceFixture, BinaryComputeSingleCoreMultiTileSub) {
    unit_tests::compute::binary::SingleCoreBinaryConfig test_config = {
        .tile_byte_size = 2 * 32 * 32,
        .l1_input_data_format = tt::DataFormat::Float16_b,
        .l1_output_data_format = tt::DataFormat::Float16_b,
        .core = CoreCoord(0, 0),
        .binary_op = "sub"};
    test_config.num_tiles = 4;
    for (unsigned int id = 0; id < num_devices_; id++) {
        ASSERT_TRUE(unit_tests::compute::binary::single_core_binary(devices_.at(id), test_config));
    }
}
TEST_F(DeviceFixture, BinaryComputeSingleCoreMultiTileMul) {
    unit_tests::compute::binary::SingleCoreBinaryConfig test_config = {
        .tile_byte_size = 2 * 32 * 32,
        .l1_input_data_format = tt::DataFormat::Float16_b,
        .l1_output_data_format = tt::DataFormat::Float16_b,
        .core = CoreCoord(0, 0),
        .binary_op = "mul"};
    test_config.num_tiles = 4;
    for (unsigned int id = 0; id < num_devices_; id++) {
        ASSERT_TRUE(unit_tests::compute::binary::single_core_binary(devices_.at(id), test_config));
    }
}

TEST_F(DeviceFixture, BinaryComputeSingleCoreMultiTileAddDestAcc) {
    auto arch = this->arch_;
    if (arch == tt::ARCH::GRAYSKULL) {
        GTEST_SKIP();
    }
    unit_tests::compute::binary::SingleCoreBinaryConfig test_config = {
        .tile_byte_size = 2 * 32 * 32,
        .l1_input_data_format = tt::DataFormat::Float16_b,
        .l1_output_data_format = tt::DataFormat::Float16_b,
        .core = CoreCoord(0, 0),
        .binary_op = "add",
        .acc_to_dest = true};
    test_config.num_tiles = 4;
    for (unsigned int id = 0; id < num_devices_; id++) {
        ASSERT_TRUE(unit_tests::compute::binary::single_core_binary(devices_.at(id), test_config));
    }
}

TEST_F(DeviceFixture, BinaryComputeSingleCoreMultiTileSubDestAcc) {
    auto arch = this->arch_;
    if (arch == tt::ARCH::GRAYSKULL) {
        GTEST_SKIP();
    }

    unit_tests::compute::binary::SingleCoreBinaryConfig test_config = {
        .tile_byte_size = 2 * 32 * 32,
        .l1_input_data_format = tt::DataFormat::Float16_b,
        .l1_output_data_format = tt::DataFormat::Float16_b,
        .core = CoreCoord(0, 0),
        .binary_op = "sub",
        .acc_to_dest = true};
    test_config.num_tiles = 4;
    for (unsigned int id = 0; id < num_devices_; id++) {
        ASSERT_TRUE(unit_tests::compute::binary::single_core_binary(devices_.at(id), test_config));
    }
}
