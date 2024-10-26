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
#include "tt_metal/test_utils/comparison.hpp"
#include "tt_metal/test_utils/df/df.hpp"
#include "tt_metal/test_utils/print_helpers.hpp"
#include "tt_metal/test_utils/stimulus.hpp"

using namespace tt;
using namespace tt::test_utils;
using namespace tt::test_utils::df;

namespace unit_tests::dram::direct {
/// @brief Does Dram --> Reader --> L1 on a single core
/// @param device
/// @param test_config - Configuration of the test -- see struct
/// @return
bool reader_only(
    tt_metal::Device* device,
    const size_t& byte_size,
    const size_t& l1_byte_address,
    const CoreCoord& reader_core) {

    bool pass = true;
    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::Program program = tt_metal::CreateProgram();

    tt::tt_metal::InterleavedBufferConfig dram_config{
                    .device=device,
                    .size = byte_size,
                    .page_size = byte_size,
                    .buffer_type = tt::tt_metal::BufferType::DRAM
        };

    auto input_dram_buffer = CreateBuffer(dram_config);
    uint32_t dram_byte_address = input_dram_buffer->address();
    auto dram_noc_xy = input_dram_buffer->noc_coordinates();
    // TODO (abhullar): Use L1 buffer after bug with L1 banking and writing to < 1 MB is fixed.
    //                  Try this after KM uplifts TLB setup
    // auto l1_buffer =
    //     CreateBuffer(device, byte_size, l1_byte_address, byte_size, tt_metal::BufferType::L1);

    auto reader_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/dram/direct_reader_dram_to_l1.cpp",
        reader_core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

    ////////////////////////////////////////////////////////////////////////////
    //                      Compile and Execute Application
    ////////////////////////////////////////////////////////////////////////////

    auto inputs = generate_uniform_random_vector<uint32_t>(0, 100, byte_size / sizeof(uint32_t));
    tt_metal::detail::WriteToBuffer(input_dram_buffer, inputs);

    tt_metal::SetRuntimeArgs(
        program,
        reader_kernel,
        reader_core,
        {
            (uint32_t)dram_byte_address,
            (uint32_t)dram_noc_xy.x,
            (uint32_t)dram_noc_xy.y,
            (uint32_t)l1_byte_address,
            (uint32_t)byte_size,
        });

    tt_metal::detail::LaunchProgram(device, program);

    std::vector<uint32_t> dest_core_data;
    // tt_metal::detail::ReadFromBuffer(l1_buffer, dest_core_data);
    tt_metal::detail::ReadFromDeviceL1(device, reader_core, l1_byte_address, byte_size, dest_core_data);
    pass &= (dest_core_data == inputs);
    if (not pass) {
        std::cout << "Mismatch at Core: " << reader_core.str() << std::endl;
    }
    return pass;
}

/// @brief Does L1 --> Writer --> Dram on a single core
/// @param device
/// @param test_config - Configuration of the test -- see struct
/// @return
bool writer_only(
    tt_metal::Device* device,
    const size_t& byte_size,
    const size_t& l1_byte_address,
    const CoreCoord& writer_core) {

    bool pass = true;
    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::Program program = tt_metal::CreateProgram();


    tt_metal::InterleavedBufferConfig dram_config{
                                        .device=device,
                                        .size = byte_size,
                                        .page_size = byte_size,
                                        .buffer_type = tt_metal::BufferType::DRAM
                                        };

    auto output_dram_buffer = CreateBuffer(dram_config);
    uint32_t dram_byte_address = output_dram_buffer->address();
    auto dram_noc_xy = output_dram_buffer->noc_coordinates();
    // TODO (abhullar): Use L1 buffer after bug with L1 banking and writing to < 1 MB is fixed.
    //                  Try this after KM uplifts TLB setup
    // auto l1_buffer =
    //     CreateBuffer(device, byte_size, l1_byte_address, byte_size, tt_metal::BufferType::L1);

    auto writer_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/dram/direct_writer_l1_to_dram.cpp",
        writer_core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

    ////////////////////////////////////////////////////////////////////////////
    //                      Compile and Execute Application
    ////////////////////////////////////////////////////////////////////////////

    auto inputs = generate_uniform_random_vector<uint32_t>(0, 100, byte_size / sizeof(uint32_t));
    tt_metal::detail::WriteToDeviceL1(device, writer_core, l1_byte_address, inputs);
    // tt_metal::detail::WriteToBuffer(l1_buffer, inputs);


    tt_metal::SetRuntimeArgs(
        program,
        writer_kernel,
        writer_core,
        {
            (uint32_t)dram_byte_address,
            (uint32_t)dram_noc_xy.x,
            (uint32_t)dram_noc_xy.y,
            (uint32_t)l1_byte_address,
            (uint32_t)byte_size,
        });

    tt_metal::detail::LaunchProgram(device, program);

    std::vector<uint32_t> dest_buffer_data;
    tt_metal::detail::ReadFromBuffer(output_dram_buffer, dest_buffer_data);
    pass &= (dest_buffer_data == inputs);
    if (not pass) {
        std::cout << "Mismatch at Core: " << writer_core.str() << std::endl;
    }
    return pass;
}

struct ReaderWriterConfig {
    size_t num_tiles = 0;
    size_t tile_byte_size = 0;
    tt::DataFormat l1_data_format = tt::DataFormat::Invalid;
    CoreCoord core = {};
};
/// @brief Does Dram --> Reader --> CB --> Writer --> Dram on a single core
/// @param device
/// @param test_config - Configuration of the test -- see struct
/// @return
bool reader_writer(tt_metal::Device* device, const ReaderWriterConfig& test_config) {

    bool pass = true;

    const uint32_t cb_index = 0;
    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    const size_t byte_size = test_config.num_tiles * test_config.tile_byte_size;
    tt_metal::Program program = tt_metal::CreateProgram();

    tt::tt_metal::InterleavedBufferConfig dram_config{
                    .device=device,
                    .size = byte_size,
                    .page_size = byte_size,
                    .buffer_type = tt::tt_metal::BufferType::DRAM
        };

    auto input_dram_buffer = CreateBuffer(dram_config);
    uint32_t input_dram_byte_address = input_dram_buffer->address();
    auto input_dram_noc_xy = input_dram_buffer->noc_coordinates();
    auto output_dram_buffer = CreateBuffer(dram_config);
    uint32_t output_dram_byte_address = output_dram_buffer->address();
    auto output_dram_noc_xy = output_dram_buffer->noc_coordinates();

    tt_metal::CircularBufferConfig l1_cb_config = tt_metal::CircularBufferConfig(byte_size, {{cb_index, test_config.l1_data_format}})
        .set_page_size(cb_index, test_config.tile_byte_size);
    auto l1_cb = tt_metal::CreateCircularBuffer(program, test_config.core, l1_cb_config);

    auto reader_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/dram/direct_reader_unary.cpp",
        test_config.core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt_metal::NOC::RISCV_1_default,
            .compile_args = {cb_index}});

    auto writer_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/dram/direct_writer_unary.cpp",
        test_config.core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .compile_args = {cb_index}});

    ////////////////////////////////////////////////////////////////////////////
    //                      Stimulus Generation
    ////////////////////////////////////////////////////////////////////////////
    std::vector<uint32_t> inputs = generate_packed_uniform_random_vector<uint32_t, tt::test_utils::df::bfloat16>(
        -1.0f, 1.0f, byte_size / tt::test_utils::df::bfloat16::SIZEOF, std::chrono::system_clock::now().time_since_epoch().count());
    ////////////////////////////////////////////////////////////////////////////
    //                      Compile and Execute Application
    ////////////////////////////////////////////////////////////////////////////

    tt_metal::detail::WriteToBuffer(input_dram_buffer, inputs);


    tt_metal::SetRuntimeArgs(
        program,
        reader_kernel,
        test_config.core,
        {
            (uint32_t)input_dram_byte_address,
            (uint32_t)input_dram_noc_xy.x,
            (uint32_t)input_dram_noc_xy.y,
            (uint32_t)test_config.num_tiles,
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

    std::vector<uint32_t> dest_buffer_data;
    tt_metal::detail::ReadFromBuffer(output_dram_buffer, dest_buffer_data);
    pass &= inputs == dest_buffer_data;
    return pass;
}
struct ReaderDatacopyWriterConfig {
    size_t num_tiles = 0;
    size_t tile_byte_size = 0;
    tt::DataFormat l1_input_data_format = tt::DataFormat::Invalid;
    tt::DataFormat l1_output_data_format = tt::DataFormat::Invalid;
    CoreCoord core = {};
};
/// @brief Does Dram --> Reader --> CB --> Datacopy --> CB --> Writer --> Dram on a single core
/// @param device
/// @param test_config - Configuration of the test -- see struct
/// @return
bool reader_datacopy_writer(tt_metal::Device* device, const ReaderDatacopyWriterConfig& test_config) {

    bool pass = true;

    const uint32_t input0_cb_index = 0;
    const uint32_t output_cb_index = 16;
    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    const size_t byte_size = test_config.num_tiles * test_config.tile_byte_size;
    tt_metal::Program program = tt_metal::CreateProgram();

    tt::tt_metal::InterleavedBufferConfig dram_config{
                    .device=device,
                    .size = byte_size,
                    .page_size = byte_size,
                    .buffer_type = tt::tt_metal::BufferType::DRAM
        };
    auto input_dram_buffer = tt_metal::CreateBuffer(dram_config);
    uint32_t input_dram_byte_address = input_dram_buffer->address();
    auto input_dram_noc_xy = input_dram_buffer->noc_coordinates();
    auto output_dram_buffer = tt_metal::CreateBuffer(dram_config);
    uint32_t output_dram_byte_address = output_dram_buffer->address();
    auto output_dram_noc_xy = output_dram_buffer->noc_coordinates();

    tt_metal::CircularBufferConfig l1_input_cb_config = tt_metal::CircularBufferConfig(byte_size, {{input0_cb_index, test_config.l1_input_data_format}})
        .set_page_size(input0_cb_index, test_config.tile_byte_size);
    auto l1_input_cb = tt_metal::CreateCircularBuffer(program, test_config.core, l1_input_cb_config);

    tt_metal::CircularBufferConfig l1_output_cb_config = tt_metal::CircularBufferConfig(byte_size, {{output_cb_index, test_config.l1_output_data_format}})
        .set_page_size(output_cb_index, test_config.tile_byte_size);
    auto l1_output_cb = tt_metal::CreateCircularBuffer(program, test_config.core, l1_output_cb_config);

    auto reader_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/dram/direct_reader_unary.cpp",
        test_config.core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt_metal::NOC::RISCV_1_default,
            .compile_args = {input0_cb_index}});

    auto writer_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/dram/direct_writer_unary.cpp",
        test_config.core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .compile_args = {output_cb_index}});

    vector<uint32_t> compute_kernel_args = {
        uint(test_config.num_tiles)  // per_core_tile_cnt
    };
    auto datacopy_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/eltwise_copy.cpp",
        test_config.core,
        tt_metal::ComputeConfig{.compile_args = compute_kernel_args});

    ////////////////////////////////////////////////////////////////////////////
    //                      Stimulus Generation
    ////////////////////////////////////////////////////////////////////////////
    std::vector<uint32_t> inputs = generate_packed_uniform_random_vector<uint32_t, tt::test_utils::df::bfloat16>(
        -1.0f, 1.0f, byte_size / tt::test_utils::df::bfloat16::SIZEOF, std::chrono::system_clock::now().time_since_epoch().count());
    ////////////////////////////////////////////////////////////////////////////
    //                      Compile and Execute Application
    ////////////////////////////////////////////////////////////////////////////

    tt_metal::detail::WriteToBuffer(input_dram_buffer, inputs);


    tt_metal::SetRuntimeArgs(
        program,
        reader_kernel,
        test_config.core,
        {
            (uint32_t)input_dram_byte_address,
            (uint32_t)input_dram_noc_xy.x,
            (uint32_t)input_dram_noc_xy.y,
            (uint32_t)test_config.num_tiles,
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

    std::vector<uint32_t> dest_buffer_data;
    tt_metal::detail::ReadFromBuffer(output_dram_buffer, dest_buffer_data);
    pass &= inputs == dest_buffer_data;
    return pass;
}
}  // namespace unit_tests::dram::direct

TEST_F(DeviceFixture, SingleCoreDirectDramReaderOnly) {
    for (unsigned int id = 0; id < num_devices_; id++) {
        uint32_t l1_unreserved_base = devices_.at(id)->get_base_allocator_addr(HalMemType::L1);
        ASSERT_TRUE(
            unit_tests::dram::direct::reader_only(devices_.at(id), 1 * 1024, l1_unreserved_base, CoreCoord(0, 0)));
        ASSERT_TRUE(
            unit_tests::dram::direct::reader_only(devices_.at(id), 2 * 1024, l1_unreserved_base, CoreCoord(0, 0)));
        ASSERT_TRUE(
            unit_tests::dram::direct::reader_only(devices_.at(id), 16 * 1024, l1_unreserved_base, CoreCoord(0, 0)));
    }
}
TEST_F(DeviceFixture, SingleCoreDirectDramWriterOnly) {
    for (unsigned int id = 0; id < num_devices_; id++) {
        uint32_t l1_unreserved_base = devices_.at(id)->get_base_allocator_addr(HalMemType::L1);
        ASSERT_TRUE(
            unit_tests::dram::direct::writer_only(devices_.at(id), 1 * 1024, l1_unreserved_base, CoreCoord(0, 0)));
        ASSERT_TRUE(
            unit_tests::dram::direct::writer_only(devices_.at(id), 2 * 1024, l1_unreserved_base, CoreCoord(0, 0)));
        ASSERT_TRUE(
            unit_tests::dram::direct::writer_only(devices_.at(id), 16 * 1024, l1_unreserved_base, CoreCoord(0, 0)));
    }
}
TEST_F(DeviceFixture, SingleCoreDirectDramReaderWriter) {
    unit_tests::dram::direct::ReaderWriterConfig test_config = {
        .num_tiles = 1,
        .tile_byte_size = 2 * 32 * 32,
        .l1_data_format = tt::DataFormat::Float16_b,
        .core = CoreCoord(0, 0)};
    for (unsigned int id = 0; id < num_devices_; id++) {
        test_config.num_tiles = 1;
        ASSERT_TRUE(unit_tests::dram::direct::reader_writer(devices_.at(id), test_config));
        test_config.num_tiles = 4;
        ASSERT_TRUE(unit_tests::dram::direct::reader_writer(devices_.at(id), test_config));
        test_config.num_tiles = 8;
        ASSERT_TRUE(unit_tests::dram::direct::reader_writer(devices_.at(id), test_config));
    }
}
TEST_F(DeviceFixture, SingleCoreDirectDramReaderDatacopyWriter) {
    unit_tests::dram::direct::ReaderDatacopyWriterConfig test_config = {
        .num_tiles = 1,
        .tile_byte_size = 2 * 32 * 32,
        .l1_input_data_format = tt::DataFormat::Float16_b,
        .l1_output_data_format = tt::DataFormat::Float16_b,
        .core = CoreCoord(0, 0)};
    for (unsigned int id = 0; id < num_devices_; id++) {
        test_config.num_tiles = 1;
        ASSERT_TRUE(unit_tests::dram::direct::reader_datacopy_writer(devices_.at(id), test_config));
        test_config.num_tiles = 4;
        ASSERT_TRUE(unit_tests::dram::direct::reader_datacopy_writer(devices_.at(id), test_config));
        test_config.num_tiles = 8;
        ASSERT_TRUE(unit_tests::dram::direct::reader_datacopy_writer(devices_.at(id), test_config));
    }
}
