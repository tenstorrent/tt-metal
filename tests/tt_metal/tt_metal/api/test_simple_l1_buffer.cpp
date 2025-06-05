// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/base.h>
#include <stddef.h>
#include <stdint.h>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <tt-metalium/assert.hpp>
#include <tt-metalium/buffer_types.hpp>
#include "buffer_test_utils.hpp"
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/data_types.hpp>
#include <tt-metalium/device.hpp>
#include "device_fixture.hpp"
#include "gtest/gtest.h"
#include <tt-metalium/kernel_types.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/program.hpp>
#include <tt_stl/span.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include "tt_metal/test_utils/stimulus.hpp"

using tt::tt_metal::IDevice;
using namespace tt::test_utils;
using namespace tt::test::buffer::detail;

namespace tt::test::buffer::detail {
bool SimpleL1ReadOnly(IDevice* device, size_t local_address, size_t byte_size) {
    std::vector<uint32_t> inputs =
        generate_uniform_random_vector<uint32_t>(0, UINT32_MAX, byte_size / sizeof(uint32_t));
    std::vector<uint32_t> outputs;
    CoreCoord bank0_logical_core = device->allocator()->get_logical_core_from_bank_id(0);
    writeL1Backdoor(device, bank0_logical_core, local_address, inputs);
    readL1Backdoor(device, bank0_logical_core, local_address, byte_size, outputs);
    bool pass = (inputs == outputs);
    if (not pass) {
        log_info(tt::LogTest, "Mismatch at Core={}, Packet Size(in Bytes)={}", bank0_logical_core.str(), byte_size);
    }
    return pass;
}
bool SimpleL1WriteOnly(IDevice* device, size_t local_address, size_t byte_size) {
    std::vector<uint32_t> inputs =
        generate_uniform_random_vector<uint32_t>(0, UINT32_MAX, byte_size / sizeof(uint32_t));
    std::vector<uint32_t> outputs;
    CoreCoord bank0_logical_core = device->allocator()->get_logical_core_from_bank_id(0);
    writeL1Backdoor(device, bank0_logical_core, local_address, inputs);
    readL1Backdoor(device, bank0_logical_core, local_address, byte_size, outputs);
    bool pass = (inputs == outputs);
    if (not pass) {
        log_info(tt::LogTest, "Mismatch at Core={}, Packet Size(in Bytes)={}", bank0_logical_core.str(), byte_size);
    }
    return pass;
}
// input_l1_buffer -->  Reader reads from this location --> CB --> Writer --> output_l1_buffer
bool SimpleTiledL1WriteCBRead(
    IDevice* device, CoreCoord core, size_t input_local_address, size_t output_local_address, size_t byte_size) {
    TT_FATAL(
        (byte_size % (32 * 32 * 2)) == 0,
        "byte_size={} must be multiple of tile size (32x32x2(w*h*datum_byte_size))",
        byte_size);
    int page_size = (32 * 32 * 2);
    int num_tiles = byte_size / page_size;

    std::vector<uint32_t> inputs = generate_uniform_random_vector<uint32_t>(5, 5, byte_size / sizeof(uint32_t));
    std::vector<uint32_t> outputs;

    tt_metal::Program program = tt_metal::CreateProgram();
    const uint32_t cb_index = 0;
    const uint32_t output_cb_index = 16;
    const CoreCoord phys_core = device->worker_core_from_logical_core(core);

    tt_metal::CircularBufferConfig l1_cb_config =
        tt_metal::CircularBufferConfig(byte_size, {{cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(cb_index, page_size);
    auto l1_cb = tt_metal::CreateCircularBuffer(program, core, l1_cb_config);
    std::map<std::string, std::string> defines = {{"INTERFACE_WITH_L1", "1"}};
    uint32_t bank_id = device->allocator()->get_bank_ids_from_logical_core(tt_metal::BufferType::L1, core)[0];
    auto reader_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/direct_reader_unary.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt_metal::NOC::NOC_0,
            .compile_args = {cb_index},
            .defines = defines});
    auto writer_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/direct_writer_unary.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::NOC_1,
            .compile_args = {cb_index},
            .defines = defines});

    tt_metal::SetRuntimeArgs(
        program,
        reader_kernel,
        core,
        {
            (uint32_t)input_local_address,
            bank_id,
            (uint32_t)num_tiles,
        });
    tt_metal::SetRuntimeArgs(
        program,
        writer_kernel,
        core,
        {
            (uint32_t)output_local_address,
            bank_id,
            (uint32_t)num_tiles,
        });

    writeL1Backdoor(device, core, input_local_address, inputs);
    tt_metal::detail::LaunchProgram(device, program);
    readL1Backdoor(device, core, input_local_address, byte_size, outputs);
    log_debug(tt::LogTest, "input readback inputs[0]={} == readback[0]={}", inputs[0], outputs[0]);
    readL1Backdoor(device, core, output_local_address, byte_size, outputs);
    log_debug(tt::LogTest, "inputs[0]={} == outputs[0]={}", inputs[0], outputs[0]);
    bool pass = (inputs == outputs);
    if (not pass) {
        log_info(
            tt::LogTest,
            "Mismatch at Core={}, phys_core={}, Packet Size(in Bytes)={}",
            core.str(),
            phys_core.str(),
            byte_size);
    }
    return pass;
}

}  // namespace tt::test::buffer::detail

namespace tt::tt_metal {

TEST_F(DeviceFixture, TestSimpleL1BufferReadOnlyLo) {
    for (unsigned int id = 0; id < num_devices_; id++) {
        size_t lo_address = this->devices_.at(id)->l1_size_per_core() -
                            this->devices_.at(id)->allocator()->get_bank_size(tt::tt_metal::BufferType::L1);
        ASSERT_TRUE(SimpleL1ReadOnly(this->devices_.at(id), lo_address, 4));
        ASSERT_TRUE(SimpleL1ReadOnly(this->devices_.at(id), lo_address, 8));
        ASSERT_TRUE(SimpleL1ReadOnly(this->devices_.at(id), lo_address, 16));
        ASSERT_TRUE(SimpleL1ReadOnly(this->devices_.at(id), lo_address, 32));
        ASSERT_TRUE(SimpleL1ReadOnly(this->devices_.at(id), lo_address, 1024));
        ASSERT_TRUE(SimpleL1ReadOnly(this->devices_.at(id), lo_address, 16 * 1024));
    }
}
TEST_F(DeviceFixture, TestSimpleL1BufferReadOnlyHi) {
    for (unsigned int id = 0; id < num_devices_; id++) {
        size_t hi_address = this->devices_.at(id)->l1_size_per_core() - (16 * 1024);
        ASSERT_TRUE(SimpleL1ReadOnly(this->devices_.at(id), hi_address, 4));
        ASSERT_TRUE(SimpleL1ReadOnly(this->devices_.at(id), hi_address, 8));
        ASSERT_TRUE(SimpleL1ReadOnly(this->devices_.at(id), hi_address, 16));
        ASSERT_TRUE(SimpleL1ReadOnly(this->devices_.at(id), hi_address, 32));
        ASSERT_TRUE(SimpleL1ReadOnly(this->devices_.at(id), hi_address, 1024));
        ASSERT_TRUE(SimpleL1ReadOnly(this->devices_.at(id), hi_address, 16 * 1024));
    }
}
TEST_F(DeviceFixture, TestSimpleL1BufferWriteOnlyLo) {
    for (unsigned int id = 0; id < num_devices_; id++) {
        size_t lo_address = this->devices_.at(id)->l1_size_per_core() -
                            this->devices_.at(id)->allocator()->get_bank_size(tt::tt_metal::BufferType::L1);
        ASSERT_TRUE(SimpleL1WriteOnly(this->devices_.at(id), lo_address, 4));
        ASSERT_TRUE(SimpleL1WriteOnly(this->devices_.at(id), lo_address, 8));
        ASSERT_TRUE(SimpleL1WriteOnly(this->devices_.at(id), lo_address, 16));
        ASSERT_TRUE(SimpleL1WriteOnly(this->devices_.at(id), lo_address, 32));
        ASSERT_TRUE(SimpleL1WriteOnly(this->devices_.at(id), lo_address, 1024));
        ASSERT_TRUE(SimpleL1WriteOnly(this->devices_.at(id), lo_address, 16 * 1024));
    }
}

TEST_F(DeviceFixture, TestSimpleL1BufferWriteOnlyHi) {
    for (unsigned int id = 0; id < num_devices_; id++) {
        size_t hi_address = this->devices_.at(id)->l1_size_per_core() - (16 * 1024);
        ASSERT_TRUE(SimpleL1WriteOnly(this->devices_.at(id), hi_address, 4));
        ASSERT_TRUE(SimpleL1WriteOnly(this->devices_.at(id), hi_address, 8));
        ASSERT_TRUE(SimpleL1WriteOnly(this->devices_.at(id), hi_address, 16));
        ASSERT_TRUE(SimpleL1WriteOnly(this->devices_.at(id), hi_address, 32));
        ASSERT_TRUE(SimpleL1WriteOnly(this->devices_.at(id), hi_address, 1024));
        ASSERT_TRUE(SimpleL1WriteOnly(this->devices_.at(id), hi_address, 16 * 1024));
    }
}

TEST_F(DeviceFixture, TensixTestSimpleL1ReadWriteTileLo) {
    for (unsigned int id = 0; id < num_devices_; id++) {
        size_t lo_address = 768 * 1024;
        ASSERT_TRUE(SimpleTiledL1WriteCBRead(
            this->devices_.at(id), {0, 0}, lo_address + 8 * 1024, lo_address + 16 * 1024, 2 * 1024));
        ASSERT_TRUE(SimpleTiledL1WriteCBRead(
            this->devices_.at(id), {0, 0}, lo_address + 8 * 1024, lo_address + 16 * 1024, 4 * 1024));
        ASSERT_TRUE(SimpleTiledL1WriteCBRead(
            this->devices_.at(id), {0, 0}, lo_address + 8 * 1024, lo_address + 16 * 1024, 6 * 1024));
    }
}

TEST_F(DeviceFixture, TensixTestSimpleL1ReadWriteTileHi) {
    for (unsigned int id = 0; id < num_devices_; id++) {
        size_t hi_address = this->devices_.at(id)->l1_size_per_core() - (24 * 1024);
        ASSERT_TRUE(SimpleTiledL1WriteCBRead(
            this->devices_.at(id), {0, 0}, hi_address + 8 * 1024, hi_address + 16 * 1024, 2 * 1024));
        ASSERT_TRUE(SimpleTiledL1WriteCBRead(
            this->devices_.at(id), {0, 0}, hi_address + 8 * 1024, hi_address + 16 * 1024, 4 * 1024));
        ASSERT_TRUE(SimpleTiledL1WriteCBRead(
            this->devices_.at(id), {0, 0}, hi_address + 8 * 1024, hi_address + 16 * 1024, 6 * 1024));
    }
}

TEST_F(DeviceFixture, TensixTestSimpleL1ReadWritex2y2TileLo) {
    for (unsigned int id = 0; id < num_devices_; id++) {
        size_t lo_address = 768 * 1024;
        ASSERT_TRUE(SimpleTiledL1WriteCBRead(
            this->devices_.at(id), {2, 2}, lo_address + 8 * 1024, lo_address + 16 * 1024, 2 * 1024));
        ASSERT_TRUE(SimpleTiledL1WriteCBRead(
            this->devices_.at(id), {2, 2}, lo_address + 8 * 1024, lo_address + 16 * 1024, 4 * 1024));
        ASSERT_TRUE(SimpleTiledL1WriteCBRead(
            this->devices_.at(id), {2, 2}, lo_address + 8 * 1024, lo_address + 16 * 1024, 6 * 1024));
    }
}

TEST_F(DeviceFixture, TensixTestSimpleL1ReadWritex2y2TileHi) {
    for (unsigned int id = 0; id < num_devices_; id++) {
        size_t hi_address = this->devices_.at(id)->l1_size_per_core() - (24 * 1024);
        ASSERT_TRUE(SimpleTiledL1WriteCBRead(
            this->devices_.at(id), {2, 2}, hi_address + 8 * 1024, hi_address + 16 * 1024, 2 * 1024));
        ASSERT_TRUE(SimpleTiledL1WriteCBRead(
            this->devices_.at(id), {2, 2}, hi_address + 8 * 1024, hi_address + 16 * 1024, 4 * 1024));
        ASSERT_TRUE(SimpleTiledL1WriteCBRead(
            this->devices_.at(id), {2, 2}, hi_address + 8 * 1024, hi_address + 16 * 1024, 6 * 1024));
    }
}

TEST_F(DeviceFixture, TensixTestBufferL1ReadWriteTileLo) {
    for (unsigned int id = 0; id < num_devices_; id++) {
        size_t lo_address = 768 * 1024;
        ASSERT_TRUE(SimpleTiledL1WriteCBRead(
            this->devices_.at(id), {2, 2}, lo_address + 8 * 1024, lo_address + 16 * 1024, 2 * 1024));
        ASSERT_TRUE(SimpleTiledL1WriteCBRead(
            this->devices_.at(id), {2, 2}, lo_address + 8 * 1024, lo_address + 16 * 1024, 4 * 1024));
        ASSERT_TRUE(SimpleTiledL1WriteCBRead(
            this->devices_.at(id), {2, 2}, lo_address + 8 * 1024, lo_address + 16 * 1024, 6 * 1024));
    }
}

TEST_F(DeviceFixture, TensixTestBufferL1ReadWriteTileHi) {
    for (unsigned int id = 0; id < num_devices_; id++) {
        size_t hi_address = this->devices_.at(id)->l1_size_per_core() - (24 * 1024);
        ASSERT_TRUE(SimpleTiledL1WriteCBRead(
            this->devices_.at(id), {2, 2}, hi_address + 8 * 1024, hi_address + 16 * 1024, 2 * 1024));
        ASSERT_TRUE(SimpleTiledL1WriteCBRead(
            this->devices_.at(id), {2, 2}, hi_address + 8 * 1024, hi_address + 16 * 1024, 4 * 1024));
        ASSERT_TRUE(SimpleTiledL1WriteCBRead(
            this->devices_.at(id), {2, 2}, hi_address + 8 * 1024, hi_address + 16 * 1024, 6 * 1024));
    }
}

}  // namespace tt::tt_metal
