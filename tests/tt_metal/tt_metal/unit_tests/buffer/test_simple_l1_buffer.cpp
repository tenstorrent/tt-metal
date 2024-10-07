// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "device_fixture.hpp"
#include "gtest/gtest.h"
#include "test_buffer_utils.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/impl/program/program_pool.hpp"
#include "tt_metal/test_utils/stimulus.hpp"


using tt::tt_metal::Device;
using namespace tt::test_utils;
using namespace tt::test::buffer::detail;

namespace tt::test::buffer::detail {
    bool SimpleL1ReadOnly (Device* device, size_t local_address, size_t byte_size) {
        std::vector<uint32_t> inputs =
            generate_uniform_random_vector<uint32_t>(0, UINT32_MAX, byte_size / sizeof(uint32_t));
        std::vector<uint32_t> outputs;
        CoreCoord bank0_logical_core = device->logical_core_from_bank_id(0);
        writeL1Backdoor(device, bank0_logical_core, local_address, inputs);
        readL1Backdoor(device, bank0_logical_core, local_address, byte_size, outputs);
        bool pass = (inputs == outputs);
        if (not pass) {
            tt::log_info("Mismatch at Core={}, Packet Size(in Bytes)={}", bank0_logical_core.str(), byte_size);
        }
        return pass;
    }
    bool SimpleL1WriteOnly (Device* device, size_t local_address, size_t byte_size) {
        std::vector<uint32_t> inputs =
            generate_uniform_random_vector<uint32_t>(0, UINT32_MAX, byte_size / sizeof(uint32_t));
        std::vector<uint32_t> outputs;
        CoreCoord bank0_logical_core = device->logical_core_from_bank_id(0);
        writeL1Backdoor(device, bank0_logical_core, local_address, inputs);
        readL1Backdoor(device, bank0_logical_core, local_address, byte_size, outputs);
        bool pass = (inputs == outputs);
        if (not pass) {
            tt::log_info("Mismatch at Core={}, Packet Size(in Bytes)={}", bank0_logical_core.str(), byte_size);
        }
        return pass;
    }
    // input_l1_buffer -->  Reader reads from this location --> CB --> Writer --> output_l1_buffer
    bool SimpleTiledL1WriteCBRead  (Device* device, CoreCoord core, size_t input_local_address, size_t output_local_address, size_t byte_size) {
        TT_FATAL ((byte_size % (32*32*2)) == 0, "byte_size={} must be multiple of tile size (32x32x2(w*h*datum_byte_size))", byte_size);
        int page_size = (32 * 32 * 2);
        int num_tiles = byte_size / page_size;

        std::vector<uint32_t> inputs =
            generate_uniform_random_vector<uint32_t>(5, 5, byte_size / sizeof(uint32_t));
        std::vector<uint32_t> outputs;

        auto program = tt_metal::CreateScopedProgram();
        const uint32_t cb_index = 0;
        const uint32_t output_cb_index = 16;
        const CoreCoord phys_core = device->worker_core_from_logical_core(core);

        tt_metal::CircularBufferConfig l1_cb_config = tt_metal::CircularBufferConfig(byte_size, {{cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(cb_index, page_size);
        auto l1_cb = tt_metal::CreateCircularBuffer(program, core, l1_cb_config);

        auto reader_kernel = tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/dram/direct_reader_unary.cpp",
            core,
            tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::NOC_0, .compile_args = {cb_index}});
        auto writer_kernel = tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/dram/direct_writer_unary.cpp",
            core,
            tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::NOC_1, .compile_args = {cb_index}});



        tt_metal::SetRuntimeArgs(
            program,
            reader_kernel,
            core,
            {
                (uint32_t)input_local_address,
                (uint32_t)phys_core.x,
                (uint32_t)phys_core.y,
                (uint32_t)num_tiles,
            });
        tt_metal::SetRuntimeArgs(
            program,
            writer_kernel,
            core,
            {
                (uint32_t)output_local_address,
                (uint32_t)phys_core.x,
                (uint32_t)phys_core.y,
                (uint32_t)num_tiles,
            });


        writeL1Backdoor(device, core, input_local_address, inputs);
        auto* program_ptr = tt::tt_metal::ProgramPool::instance().get_program(program);
        tt_metal::detail::LaunchProgram(device, *program_ptr);
        readL1Backdoor(device, core, input_local_address, byte_size, outputs);
        tt::log_debug("input readback inputs[0]={} == readback[0]={}", inputs[0], outputs[0]);
        readL1Backdoor(device, core, output_local_address, byte_size, outputs);
        tt::log_debug("inputs[0]={} == outputs[0]={}", inputs[0], outputs[0]);
        bool pass = (inputs == outputs);
        if (not pass) {
            tt::log_info("Mismatch at Core={}, phys_core={}, Packet Size(in Bytes)={}", core.str(), phys_core.str(), byte_size);
        }
        return pass;
    }
}

TEST_F(DeviceFixture, TestSimpleL1BufferReadOnlyLo) {
    for (unsigned int id = 0; id < num_devices_; id++) {
        size_t lo_address =
            this->devices_.at(id)->l1_size_per_core() - this->devices_.at(id)->bank_size(tt::tt_metal::BufferType::L1);
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
        size_t lo_address =
            this->devices_.at(id)->l1_size_per_core() - this->devices_.at(id)->bank_size(tt::tt_metal::BufferType::L1);
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

TEST_F(DeviceFixture, TestSimpleL1ReadWriteTileLo) {
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

TEST_F(DeviceFixture, TestSimpleL1ReadWriteTileHi) {
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

TEST_F(DeviceFixture, TestSimpleL1ReadWritex2y2TileLo) {
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

TEST_F(DeviceFixture, TestSimpleL1ReadWritex2y2TileHi) {
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

TEST_F(DeviceFixture, TestBufferL1ReadWriteTileLo) {
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

TEST_F(DeviceFixture, TestBufferL1ReadWriteTileHi) {
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
