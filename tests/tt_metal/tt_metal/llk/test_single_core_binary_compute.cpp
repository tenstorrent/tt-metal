// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
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
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/data_types.hpp>
#include "device_fixture.hpp"
#include <tt-metalium/kernel_types.hpp>
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
};

void set_math_fid_masks(
    uint16_t& srca_fid_mask, uint16_t& srcb_fid_mask, MathFidelity math_fidelity = MathFidelity::HiFi4) {
    auto arch = get_arch_from_string(get_umd_arch_name());
    switch (math_fidelity) {
        case MathFidelity::HiFi4:
        case MathFidelity::HiFi3: {
            break;
        }
        case MathFidelity::HiFi2: {
            srcb_fid_mask = (arch == tt::ARCH::GRAYSKULL) ? 0xFFF8 : 0xFFFE;
            ;
            break;
        }
        case MathFidelity::LoFi: {
            srca_fid_mask = 0xFFF8;
            srcb_fid_mask = (arch == tt::ARCH::GRAYSKULL) ? 0xFFF8 : 0xFFFE;
            break;
        }
        default: {
            TT_THROW("Unsupported MathFidelity={}", math_fidelity);
            break;
        }
    }
}

/// @brief Does Dramx2 --> Reader --> CB --> Binary Compute --> CB --> Writer --> Dram
/// @param device
/// @param test_config - Configuration of the test -- see struct
/// @return
bool single_core_binary(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device, const SingleCoreBinaryConfig& test_config) {
    bool pass = true;
    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    const size_t byte_size = test_config.num_tiles * test_config.tile_byte_size;

    auto& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    Program program = tt::tt_metal::CreateProgram();
    workload.add_program(device_range, std::move(program));
    auto& program_ = workload.get_programs().at(device_range);

    distributed::DeviceLocalBufferConfig dram_config{
        .page_size = byte_size, .buffer_type = tt::tt_metal::BufferType::DRAM, .bottom_up = false};
    distributed::ReplicatedBufferConfig buffer_config{.size = byte_size};

    auto input0_dram_buffer = distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());

    // tt::tt_metal::InterleavedBufferConfig dram_config{
    //     .device = device, .size = byte_size, .page_size = byte_size, .buffer_type = tt::tt_metal::BufferType::DRAM};
    // auto input0_dram_buffer = CreateBuffer(dram_config);
    uint32_t input0_dram_byte_address = input0_dram_buffer->address();

    auto input1_dram_buffer = distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());
    // auto input1_dram_buffer = CreateBuffer(dram_config);
    uint32_t input1_dram_byte_address = input1_dram_buffer->address();

    auto input2_dram_buffer = distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());
    // auto input2_dram_buffer = CreateBuffer(dram_config);
    uint32_t input2_dram_byte_address = input2_dram_buffer->address();

    auto output_dram_buffer = distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());
    // auto output_dram_buffer = CreateBuffer(dram_config);
    uint32_t output_dram_byte_address = output_dram_buffer->address();

    tt_metal::CircularBufferConfig l1_cb_config =
        tt_metal::CircularBufferConfig(byte_size, {{0, test_config.l1_input_data_format}})
            .set_page_size(0, test_config.tile_byte_size);
    tt_metal::CreateCircularBuffer(program_, test_config.core, l1_cb_config);

    tt_metal::CircularBufferConfig l1_input1_cb_config =
        tt_metal::CircularBufferConfig(byte_size, {{1, test_config.l1_input_data_format}})
            .set_page_size(1, test_config.tile_byte_size);
    tt_metal::CreateCircularBuffer(program_, test_config.core, l1_input1_cb_config);

    tt_metal::CircularBufferConfig l2_input1_cb_config =
        tt_metal::CircularBufferConfig(byte_size, {{2, test_config.l1_input_data_format}})
            .set_page_size(2, test_config.tile_byte_size);
    tt_metal::CreateCircularBuffer(program_, test_config.core, l2_input1_cb_config);

    tt_metal::CircularBufferConfig l1_output_cb_config =
        tt_metal::CircularBufferConfig(byte_size, {{16, test_config.l1_output_data_format}})
            .set_page_size(16, test_config.tile_byte_size);
    tt_metal::CreateCircularBuffer(program_, test_config.core, l1_output_cb_config);

    vector<uint32_t> compute_kernel_args = {};
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
            defines["DST_ACCUM_MODE"] = "1";
            defines["ELTWISE_OP_INIT"] = defines["ELTWISE_OP"] + "_init";
            if (test_config.binary_op == "mul") {
                defines["MUL_TILES_WITH_DST_ACCUM"] = "1";
            }
        }
    }

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
            .math_fidelity = test_config.math_fidelity, .compile_args = compute_kernel_args, .defines = defines});

    SetRuntimeArgs(program_, binary_kernel, test_config.core, {uint32_t(test_config.num_tiles), 1});

    ////////////////////////////////////////////////////////////////////////////
    //                      Stimulus Generation
    ////////////////////////////////////////////////////////////////////////////
    std::vector<uint32_t> packed_input0 = generate_packed_uniform_random_vector<uint32_t, bfloat16>(
        -1.0f, 1.0f, byte_size / sizeof(bfloat16), std::chrono::system_clock::now().time_since_epoch().count());
    std::vector<uint32_t> packed_input1 = generate_packed_uniform_random_vector<uint32_t, bfloat16>(
        -1.0f, 1.0f, byte_size / sizeof(bfloat16), std::chrono::system_clock::now().time_since_epoch().count());
    std::vector<uint32_t> packed_input2 = generate_packed_uniform_random_vector<uint32_t, bfloat16>(
        -1.0f, 1.0f, byte_size / sizeof(bfloat16), std::chrono::system_clock::now().time_since_epoch().count());
    ////////////////////////////////////////////////////////////////////////////
    //                      Golden Generation
    ////////////////////////////////////////////////////////////////////////////
    auto input0 = unpack_vector<bfloat16, uint32_t>(packed_input0);
    auto input1 = unpack_vector<bfloat16, uint32_t>(packed_input1);
    auto input2 = unpack_vector<bfloat16, uint32_t>(packed_input2);

    std::vector<float> temp_golden(input0.size());
    uint16_t srca_fid_mask = 0xFFFF;
    uint16_t srcb_fid_mask = 0xFFFF;
    set_math_fid_masks(srca_fid_mask, srcb_fid_mask, test_config.math_fidelity);
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
            // acc_to_dest accumulates dest value with binary output, for all binary operations
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
    auto packed_golden = pack_vector<uint32_t, bfloat16>(golden);

    ////////////////////////////////////////////////////////////////////////////
    //                      Compile and Execute Application
    ////////////////////////////////////////////////////////////////////////////

    distributed::WriteShard(cq, input0_dram_buffer, packed_input0, zero_coord, false);
    distributed::WriteShard(cq, input1_dram_buffer, packed_input1, zero_coord, false);
    distributed::WriteShard(cq, input2_dram_buffer, packed_input2, zero_coord, false);
    // tt_metal::detail::WriteToBuffer(input0_dram_buffer, packed_input0);
    // tt_metal::detail::WriteToBuffer(input1_dram_buffer, packed_input1);
    // tt_metal::detail::WriteToBuffer(input2_dram_buffer, packed_input2);

    tt_metal::SetRuntimeArgs(
        program_,
        reader_kernel,
        test_config.core,
        {
            (uint32_t)input0_dram_byte_address,
            (uint32_t)0,  // dram bank id
            (uint32_t)input1_dram_byte_address,
            (uint32_t)0,  // dram bank id
            (uint32_t)test_config.num_tiles,
            (uint32_t)input2_dram_byte_address,
            (uint32_t)0,  // dram bank id
        });
    tt_metal::SetRuntimeArgs(
        program_,
        writer_kernel,
        test_config.core,
        {
            (uint32_t)output_dram_byte_address,
            (uint32_t)0,  // dram bank id
            (uint32_t)test_config.num_tiles,
        });

    distributed::EnqueueMeshWorkload(cq, workload, false);

    ////////////////////////////////////////////////////////////////////////////
    //                      Comparison Checking
    ////////////////////////////////////////////////////////////////////////////
    std::vector<uint32_t> dest_buffer_data;
    distributed::ReadShard(cq, dest_buffer_data, output_dram_buffer, zero_coord, false);
    // tt_metal::detail::ReadFromBuffer(output_dram_buffer, dest_buffer_data);

    pass &= is_close_packed_vectors<bfloat16, uint32_t>(
        dest_buffer_data, packed_golden, [&](const bfloat16& a, const bfloat16& b) { return is_close(a, b, 0.0155f); });
    return pass;
}
}  // namespace unit_tests::compute::binary

TEST_F(MeshDeviceFixture, TensixBinaryComputeSingleCoreSingleTileAdd) {
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
        }
    }
}

TEST_F(MeshDeviceFixture, TensixBinaryComputeSingleCoreSingleTileSub) {
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
        }
    }
}

TEST_F(MeshDeviceFixture, TensixBinaryComputeSingleCoreSingleTileMul) {
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
        }
    }
}

TEST_F(MeshDeviceFixture, TensixBinaryComputeSingleCoreSingleTileAddFullInit) {
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
        }
    }
}

TEST_F(MeshDeviceFixture, TensixBinaryComputeSingleCoreSingleTileSubFullInit) {
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
        }
    }
}

TEST_F(MeshDeviceFixture, TensixBinaryComputeSingleCoreSingleTileMulFullInit) {
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
        }
    }
}

TEST_F(MeshDeviceFixture, TensixBinaryComputeSingleCoreMultiTileAddWithDestReuse) {
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
        }
    }
}

TEST_F(MeshDeviceFixture, TensixBinaryComputeSingleCoreMultiTileSubWithDestReuse) {
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
        }
    }
}

TEST_F(MeshDeviceFixture, TensixBinaryComputeSingleCoreMultiTileMulWithDestReuse) {
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
        }
    }
}

TEST_F(MeshDeviceFixture, TensixBinaryComputeSingleCoreMultiTileAdd) {
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
        }
    }
}

TEST_F(MeshDeviceFixture, TensixBinaryComputeSingleCoreMultiTileSub) {
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
        }
    }
}

TEST_F(MeshDeviceFixture, TensixBinaryComputeSingleCoreMultiTileMul) {
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
        }
    }
}

TEST_F(MeshDeviceFixture, TensixBinaryComputeSingleCoreMultiTileAddDestAcc) {
    auto arch = this->arch_;
    if (arch == tt::ARCH::GRAYSKULL) {
        GTEST_SKIP();
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
        }
    }
}

TEST_F(MeshDeviceFixture, TensixBinaryComputeSingleCoreMultiTileSubDestAcc) {
    auto arch = this->arch_;
    if (arch == tt::ARCH::GRAYSKULL) {
        GTEST_SKIP();
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
        }
    }
}

TEST_F(MeshDeviceFixture, TensixBinaryComputeSingleCoreMultiTileMulDestAcc) {
    auto arch = this->arch_;
    if (arch == tt::ARCH::GRAYSKULL) {
        GTEST_SKIP();
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
        }
    }
}

}  // namespace tt::tt_metal
