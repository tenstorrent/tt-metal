// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "single_device_fixture.hpp"
#include "gtest/gtest.h"
#include "test_buffer_utils.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/test_utils/comparison.hpp"
#include "tt_metal/test_utils/df/df.hpp"
#include "tt_metal/test_utils/print_helpers.hpp"
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
    bool SimpleTiledL1WriteCBRead  (Device* device, CoreCoord core, size_t input_local_address, size_t intermed_local_address, size_t output_local_address, size_t byte_size) {
        log_assert ((byte_size % (32*32*2)) == 0, "byte_size={} must be multiple of tile size (32x32x2(w*h*datum_byte_size))", byte_size);
        int num_tiles = byte_size / (32*32*2);
        std::vector<uint32_t> inputs =
            generate_uniform_random_vector<uint32_t>(5, 5, byte_size / sizeof(uint32_t));
        std::vector<uint32_t> outputs;

        tt_metal::Program program = tt_metal::Program();
        const uint32_t cb_index = 0;
        const uint32_t output_cb_index = 16;
        const CoreCoord phys_core = device->worker_core_from_logical_core(core);
        auto l1_cb = tt_metal::CreateCircularBuffer(
            program,
            cb_index,
            core,
            num_tiles,
            byte_size,
            tt::DataFormat::Float16_b,
            intermed_local_address);
        auto reader_kernel = tt_metal::CreateDataMovementKernel(
            program,
            "tt_metal/kernels/dataflow/unit_tests/dram/direct_reader_unary.cpp",
            core,
            tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::NOC_0, .compile_args = {cb_index}});
        auto writer_kernel = tt_metal::CreateDataMovementKernel(
            program,
            "tt_metal/kernels/dataflow/unit_tests/dram/direct_writer_unary.cpp",
            core,
            tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::NOC_1, .compile_args = {cb_index}});

        tt_metal::CompileProgram(device, program);
        tt_metal::ConfigureDeviceWithProgram(device, program);
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
        tt_metal::WriteRuntimeArgsToDevice(device, program);

        writeL1Backdoor(device, core, input_local_address, inputs);
        tt_metal::LaunchKernels(device, program);
        readL1Backdoor(device, core, input_local_address, byte_size, outputs);
        tt::log_info("input readback inputs[0]={} == readback[0]={}", inputs[0], outputs[0]);
        readL1Backdoor(device, core, output_local_address, byte_size, outputs);
        tt::log_info("inputs[0]={} == outputs[0]={}", inputs[0], outputs[0]);
        bool pass = (inputs == outputs);
        if (not pass) {
            tt::log_info("Mismatch at Core={}, phys_core={}, Packet Size(in Bytes)={}", core.str(), phys_core.str(), byte_size);
        }
        return pass;
    }
}

TEST_F(SingleDeviceFixture, TestSimpleL1BufferReadOnlyLo) {
    size_t lo_address = this->device_->l1_size() - this->device_->cluster()->get_soc_desc(this->device_->pcie_slot()).l1_bank_size;
    ASSERT_TRUE(SimpleL1ReadOnly(this->device_, lo_address, 4));
    ASSERT_TRUE(SimpleL1ReadOnly(this->device_, lo_address, 8));
    ASSERT_TRUE(SimpleL1ReadOnly(this->device_, lo_address, 16));
    ASSERT_TRUE(SimpleL1ReadOnly(this->device_, lo_address, 32));
    ASSERT_TRUE(SimpleL1ReadOnly(this->device_, lo_address, 1024));
    ASSERT_TRUE(SimpleL1ReadOnly(this->device_, lo_address, 16*1024));
}
TEST_F(SingleDeviceFixture, TestSimpleL1BufferReadOnlyHi) {
    size_t hi_address = this->device_->l1_size() - (16*1024);
    ASSERT_TRUE(SimpleL1ReadOnly(this->device_, hi_address, 4));
    ASSERT_TRUE(SimpleL1ReadOnly(this->device_, hi_address, 8));
    ASSERT_TRUE(SimpleL1ReadOnly(this->device_, hi_address, 16));
    ASSERT_TRUE(SimpleL1ReadOnly(this->device_, hi_address, 32));
    ASSERT_TRUE(SimpleL1ReadOnly(this->device_, hi_address, 1024));
    ASSERT_TRUE(SimpleL1ReadOnly(this->device_, hi_address, 16*1024));
}
TEST_F(SingleDeviceFixture, TestSimpleL1BufferWriteOnlyLo) {
    size_t lo_address = this->device_->l1_size() - this->device_->cluster()->get_soc_desc(this->device_->pcie_slot()).l1_bank_size;
    ASSERT_TRUE(SimpleL1WriteOnly(this->device_, lo_address, 4));
    ASSERT_TRUE(SimpleL1WriteOnly(this->device_, lo_address, 8));
    ASSERT_TRUE(SimpleL1WriteOnly(this->device_, lo_address, 16));
    ASSERT_TRUE(SimpleL1WriteOnly(this->device_, lo_address, 32));
    ASSERT_TRUE(SimpleL1WriteOnly(this->device_, lo_address, 1024));
    ASSERT_TRUE(SimpleL1WriteOnly(this->device_, lo_address, 16*1024));
}

TEST_F(SingleDeviceFixture, TestSimpleL1BufferWriteOnlyHi) {
    size_t hi_address = this->device_->l1_size() - (16*1024);
    ASSERT_TRUE(SimpleL1WriteOnly(this->device_, hi_address, 4));
    ASSERT_TRUE(SimpleL1WriteOnly(this->device_, hi_address, 8));
    ASSERT_TRUE(SimpleL1WriteOnly(this->device_, hi_address, 16));
    ASSERT_TRUE(SimpleL1WriteOnly(this->device_, hi_address, 32));
    ASSERT_TRUE(SimpleL1WriteOnly(this->device_, hi_address, 1024));
    ASSERT_TRUE(SimpleL1WriteOnly(this->device_, hi_address, 16*1024));
}

TEST_F(SingleDeviceFixture, TestSimpleL1ReadWriteTileLo) {
    size_t cb_address = L1_UNRESERVED_BASE;
    size_t lo_address = 768*1024;
    ASSERT_TRUE(SimpleTiledL1WriteCBRead(this->device_, {0, 0}, lo_address + 8*1024, cb_address, lo_address + 16*1024, 2*1024));
    ASSERT_TRUE(SimpleTiledL1WriteCBRead(this->device_, {0, 0}, lo_address + 8*1024, cb_address, lo_address + 16*1024, 4*1024));
    ASSERT_TRUE(SimpleTiledL1WriteCBRead(this->device_, {0, 0}, lo_address + 8*1024, cb_address, lo_address + 16*1024, 6*1024));
}

TEST_F(SingleDeviceFixture, TestSimpleL1ReadWriteTileHi) {
    size_t cb_address = L1_UNRESERVED_BASE;
    size_t hi_address = this->device_->l1_size() - (24*1024);
    ASSERT_TRUE(SimpleTiledL1WriteCBRead(this->device_, {0, 0}, hi_address + 8*1024, cb_address, hi_address + 16*1024, 2*1024));
    ASSERT_TRUE(SimpleTiledL1WriteCBRead(this->device_, {0, 0}, hi_address + 8*1024, cb_address, hi_address + 16*1024, 4*1024));
    ASSERT_TRUE(SimpleTiledL1WriteCBRead(this->device_, {0, 0}, hi_address + 8*1024, cb_address, hi_address + 16*1024, 6*1024));
}

TEST_F(SingleDeviceFixture, TestSimpleL1ReadWritex2y2TileLo) {
    size_t cb_address = L1_UNRESERVED_BASE;
    size_t lo_address = 768*1024;
    ASSERT_TRUE(SimpleTiledL1WriteCBRead(this->device_, {2, 2}, lo_address + 8*1024, cb_address, lo_address + 16*1024, 2*1024));
    ASSERT_TRUE(SimpleTiledL1WriteCBRead(this->device_, {2, 2}, lo_address + 8*1024, cb_address, lo_address + 16*1024, 4*1024));
    ASSERT_TRUE(SimpleTiledL1WriteCBRead(this->device_, {2, 2}, lo_address + 8*1024, cb_address, lo_address + 16*1024, 6*1024));
}

TEST_F(SingleDeviceFixture, TestSimpleL1ReadWritex2y2TileHi) {
    size_t cb_address = L1_UNRESERVED_BASE;
    size_t hi_address = this->device_->l1_size() - (24*1024);
    ASSERT_TRUE(SimpleTiledL1WriteCBRead(this->device_, {2, 2}, hi_address + 8*1024, cb_address, hi_address + 16*1024, 2*1024));
    ASSERT_TRUE(SimpleTiledL1WriteCBRead(this->device_, {2, 2}, hi_address + 8*1024, cb_address, hi_address + 16*1024, 4*1024));
    ASSERT_TRUE(SimpleTiledL1WriteCBRead(this->device_, {2, 2}, hi_address + 8*1024, cb_address, hi_address + 16*1024, 6*1024));
}

TEST_F(SingleDeviceFixture, TestBufferL1ReadWriteTileLo) {
    size_t cb_address = L1_UNRESERVED_BASE;
    size_t lo_address = 768*1024;
    ASSERT_TRUE(SimpleTiledL1WriteCBRead(this->device_, {2, 2}, lo_address + 8*1024, cb_address, lo_address + 16*1024, 2*1024));
    ASSERT_TRUE(SimpleTiledL1WriteCBRead(this->device_, {2, 2}, lo_address + 8*1024, cb_address, lo_address + 16*1024, 4*1024));
    ASSERT_TRUE(SimpleTiledL1WriteCBRead(this->device_, {2, 2}, lo_address + 8*1024, cb_address, lo_address + 16*1024, 6*1024));
}

TEST_F(SingleDeviceFixture, TestBufferL1ReadWriteTileHi) {
    size_t cb_address = L1_UNRESERVED_BASE;
    size_t hi_address = this->device_->l1_size() - (24*1024);
    ASSERT_TRUE(SimpleTiledL1WriteCBRead(this->device_, {2, 2}, hi_address + 8*1024, cb_address, hi_address + 16*1024, 2*1024));
    ASSERT_TRUE(SimpleTiledL1WriteCBRead(this->device_, {2, 2}, hi_address + 8*1024, cb_address, hi_address + 16*1024, 4*1024));
    ASSERT_TRUE(SimpleTiledL1WriteCBRead(this->device_, {2, 2}, hi_address + 8*1024, cb_address, hi_address + 16*1024, 6*1024));
}
