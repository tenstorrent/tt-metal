#include "../basic_harness.hpp"
#include "gtest/gtest.h"
#include "test_buffer_utils.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/test_utils/comparison.hpp"
#include "tt_metal/test_utils/df/df.hpp"
#include "tt_metal/test_utils/print_helpers.hpp"
#include "tt_metal/test_utils/stimulus.hpp"


using namespace tt::test_utils;
using namespace tt::test::buffer::detail;

namespace tt::test::buffer::detail {
    bool SimpleL1ReadOnly (Device* device, size_t local_address, size_t byte_size) {
        std::vector<uint32_t> inputs =
            generate_uniform_random_vector<uint32_t>(0, UINT32_MAX, byte_size / sizeof(uint32_t));
        std::vector<uint32_t> outputs;
        auto buffer = tt::tt_metal::Buffer(device, byte_size, local_address, byte_size, tt::tt_metal::BufferType::L1);
        writeL1Backdoor(device, buffer.logical_core_from_bank_id(0), buffer.address(), inputs);
        tt::tt_metal::ReadFromBuffer(buffer, outputs);
        bool pass = (inputs == outputs);
        if (not pass) {
            tt::log_info("Mismatch at Core={}, Packet Size(in Bytes)={}", buffer.logical_core_from_bank_id(0).str(), byte_size);
        }
        return pass;
    }
    bool SimpleL1WriteOnly (Device* device, size_t local_address, size_t byte_size) {
        std::vector<uint32_t> inputs =
            generate_uniform_random_vector<uint32_t>(0, UINT32_MAX, byte_size / sizeof(uint32_t));
        std::vector<uint32_t> outputs;
        auto buffer = tt::tt_metal::Buffer(device, byte_size, local_address, byte_size, tt::tt_metal::BufferType::L1);
        tt::tt_metal::WriteToBuffer(buffer, inputs);
        readL1Backdoor(device, buffer.logical_core_from_bank_id(0), buffer.address(), byte_size, outputs);
        bool pass = (inputs == outputs);
        if (not pass) {
            tt::log_info("Mismatch at Core={}, Packet Size(in Bytes)={}", buffer.logical_core_from_bank_id(0).str(), byte_size);
        }
        return pass;
    }

    // input_l1_buffer -->  Reader reads from this location --> CB --> Writer --> output_l1_buffer
    bool SimpleL1TileWriteCBRead  (Device* device, size_t input_local_address, size_t intermed_local_address, size_t output_local_address, size_t byte_size) {
        log_assert ((byte_size % (32*32*2)) == 0, "byte_size={} must be multiple of tile size (32x32x2(w*h*datum_byte_size))", byte_size);
        int num_tiles = byte_size / (32*32*2);
        std::vector<uint32_t> inputs =
            generate_uniform_random_vector<uint32_t>(0, UINT32_MAX, byte_size / sizeof(uint32_t));
        std::vector<uint32_t> outputs;

        auto input_buffer = tt::tt_metal::Buffer(device, byte_size, input_local_address, byte_size, tt::tt_metal::BufferType::L1);
        auto output_buffer = tt::tt_metal::Buffer(device, byte_size, output_local_address, byte_size, tt::tt_metal::BufferType::L1);

        tt_metal::Program program = tt_metal::Program();
        const uint32_t cb_index = 0;
        const uint32_t output_cb_index = 16;

        auto l1_cb = tt_metal::CreateCircularBuffer(
            program,
            cb_index,
            input_buffer.logical_core_from_bank_id(0),
            num_tiles,
            byte_size,
            tt::DataFormat::Float16_b,
            intermed_local_address);
        auto reader_kernel = tt_metal::CreateDataMovementKernel(
            program,
            "tt_metal/kernels/dataflow/unit_tests/dram/direct_reader_unary.cpp",
            input_buffer.logical_core_from_bank_id(0),
            {cb_index},
            tt_metal::DataMovementProcessor::RISCV_1,
            tt_metal::NOC::RISCV_1_default);
        auto writer_kernel = tt_metal::CreateDataMovementKernel(
            program,
            "tt_metal/kernels/dataflow/unit_tests/dram/direct_writer_unary.cpp",
            input_buffer.logical_core_from_bank_id(0),
            {cb_index},
            tt_metal::DataMovementProcessor::RISCV_0,
            tt_metal::NOC::RISCV_0_default);

        tt_metal::CompileProgram(device, program);
        tt_metal::ConfigureDeviceWithProgram(device, program);
        tt_metal::SetRuntimeArgs(
            reader_kernel,
            input_buffer.logical_core_from_bank_id(0),
            {
                (uint32_t)input_buffer.address(),
                (uint32_t)input_buffer.noc_coordinates(0).x,
                (uint32_t)input_buffer.noc_coordinates(0).y,
                (uint32_t)num_tiles,
            });
        tt_metal::SetRuntimeArgs(
            writer_kernel,
            input_buffer.logical_core_from_bank_id(0),
            {
                (uint32_t)output_buffer.address(),
                (uint32_t)output_buffer.noc_coordinates(0).x,
                (uint32_t)output_buffer.noc_coordinates(0).y,
                (uint32_t)num_tiles,
            });
        tt_metal::WriteRuntimeArgsToDevice(device, program);

        std::cout << "Write to Buffer " << std::endl;
        tt::tt_metal::WriteToBuffer(input_buffer, inputs);
        tt_metal::LaunchKernels(device, program);
        std::cout << "Read From Buffer " << std::endl;
        tt::tt_metal::ReadFromBuffer(output_buffer, outputs);
        bool pass = (inputs == outputs);
        if (not pass) {
            tt::log_info("Mismatch at Core={}, Packet Size(in Bytes)={}", input_buffer.logical_core_from_bank_id(0).str(), byte_size);
        }
        return pass;
    }

}


TEST_F(DeviceHarness, TestSimpleL1BufferReadOnlyLi) {
    size_t lo_address = UNRESERVED_BASE;
    GTEST_ASSERT_TRUE(SimpleL1ReadOnly(this->device, lo_address, 4));
    GTEST_ASSERT_TRUE(SimpleL1ReadOnly(this->device, lo_address, 8));
    GTEST_ASSERT_TRUE(SimpleL1ReadOnly(this->device, lo_address, 16));
    GTEST_ASSERT_TRUE(SimpleL1ReadOnly(this->device, lo_address, 32));
    GTEST_ASSERT_TRUE(SimpleL1ReadOnly(this->device, lo_address, 1024));
    GTEST_ASSERT_TRUE(SimpleL1ReadOnly(this->device, lo_address, 16*1024));
}
TEST_F(DeviceHarness, TestSimpleL1BufferReadOnlyHi) {
    size_t hi_address = this->device->l1_size() - (16*1024);
    GTEST_ASSERT_TRUE(SimpleL1ReadOnly(this->device, hi_address, 4));
    GTEST_ASSERT_TRUE(SimpleL1ReadOnly(this->device, hi_address, 8));
    GTEST_ASSERT_TRUE(SimpleL1ReadOnly(this->device, hi_address, 16));
    GTEST_ASSERT_TRUE(SimpleL1ReadOnly(this->device, hi_address, 32));
    GTEST_ASSERT_TRUE(SimpleL1ReadOnly(this->device, hi_address, 1024));
    GTEST_ASSERT_TRUE(SimpleL1ReadOnly(this->device, hi_address, 16*1024));
}
TEST_F(DeviceHarness, TestSimpleL1BufferWriteOnlyLo) {
    size_t lo_address = UNRESERVED_BASE;
    GTEST_ASSERT_TRUE(SimpleL1WriteOnly(this->device, lo_address, 4));
    GTEST_ASSERT_TRUE(SimpleL1WriteOnly(this->device, lo_address, 8));
    GTEST_ASSERT_TRUE(SimpleL1WriteOnly(this->device, lo_address, 16));
    GTEST_ASSERT_TRUE(SimpleL1WriteOnly(this->device, lo_address, 32));
    GTEST_ASSERT_TRUE(SimpleL1WriteOnly(this->device, lo_address, 1024));
    GTEST_ASSERT_TRUE(SimpleL1WriteOnly(this->device, lo_address, 16*1024));
}

TEST_F(DeviceHarness, TestSimpleL1BufferWriteOnlyHi) {
    size_t hi_address = this->device->l1_size() - (16*1024);
    GTEST_ASSERT_TRUE(SimpleL1WriteOnly(this->device, hi_address, 4));
    GTEST_ASSERT_TRUE(SimpleL1WriteOnly(this->device, hi_address, 8));
    GTEST_ASSERT_TRUE(SimpleL1WriteOnly(this->device, hi_address, 16));
    GTEST_ASSERT_TRUE(SimpleL1WriteOnly(this->device, hi_address, 32));
    GTEST_ASSERT_TRUE(SimpleL1WriteOnly(this->device, hi_address, 1024));
    GTEST_ASSERT_TRUE(SimpleL1WriteOnly(this->device, hi_address, 16*1024));
}

TEST_F(DeviceHarness, TestSimpleL1ReadDatacopyWriteTileLo) {
    size_t lo_address = UNRESERVED_BASE;
    GTEST_ASSERT_TRUE(SimpleL1TileWriteCBRead(this->device, lo_address + 8*1024, lo_address, lo_address + 16*1024, 2*1024));
    GTEST_ASSERT_TRUE(SimpleL1TileWriteCBRead(this->device, lo_address + 8*1024, lo_address, lo_address + 16*1024, 4*1024));
    GTEST_ASSERT_TRUE(SimpleL1TileWriteCBRead(this->device, lo_address + 8*1024, lo_address, lo_address + 16*1024, 6*1024));
}

TEST_F(DeviceHarness, TestSimpleL1ReadDatacopyWriteTileHi) {
    size_t lo_address = UNRESERVED_BASE;
    size_t hi_address = this->device->l1_size() - (24*1024);
    GTEST_ASSERT_TRUE(SimpleL1TileWriteCBRead(this->device, hi_address + 8*1024, lo_address, hi_address + 16*1024, 2*1024));
    GTEST_ASSERT_TRUE(SimpleL1TileWriteCBRead(this->device, hi_address + 8*1024, lo_address, hi_address + 16*1024, 4*1024));
    GTEST_ASSERT_TRUE(SimpleL1TileWriteCBRead(this->device, hi_address + 8*1024, lo_address, hi_address + 16*1024, 6*1024));
}
