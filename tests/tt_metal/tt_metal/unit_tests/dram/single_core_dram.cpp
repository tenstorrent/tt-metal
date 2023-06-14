#include <algorithm>
#include <functional>
#include <random>

#include "doctest.h"
#include "single_device_fixture.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/hostdevcommon/common_runtime_address_map.h"  // FIXME: Should remove dependency on this
#include "tt_metal/test_utils/comparison.hpp"
#include "tt_metal/test_utils/df/df.hpp"
#include "tt_metal/test_utils/print_helpers.hpp"
#include "tt_metal/test_utils/stimulus.hpp"

using namespace tt;
using namespace tt::test_utils;
using namespace tt::test_utils::df;

namespace unit_tests::single_core_dram {
/// @brief Does Dram --> Reader --> L1 on a single core
/// @param device
/// @param test_config - Configuration of the test -- see struct
/// @return
bool reader_only(
    tt_metal::Device* device,
    const size_t& byte_size,
    const size_t& dram_channel,
    const size_t& dram_byte_address,
    const size_t& l1_byte_address,
    const CoreCoord& reader_core) {
    bool pass = true;
    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::Program program = tt_metal::Program();

    auto input_dram_buffer =
        tt_metal::Buffer(device, byte_size, dram_byte_address, dram_channel, byte_size, tt_metal::BufferType::DRAM);
    auto dram_noc_xy = input_dram_buffer.noc_coordinates();
    auto l1_bank_ids = device->bank_ids_from_logical_core(reader_core);
    auto l1_buffer =
        tt_metal::Buffer(device, byte_size, l1_byte_address, l1_bank_ids.at(0), byte_size, tt_metal::BufferType::L1);

    auto reader_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/dram_to_l1_copy.cpp",
        reader_core,
        tt_metal::DataMovementProcessor::RISCV_0,
        tt_metal::NOC::RISCV_0_default);

    ////////////////////////////////////////////////////////////////////////////
    //                      Compile and Execute Application
    ////////////////////////////////////////////////////////////////////////////
    pass &= tt_metal::CompileProgram(device, program);
    auto inputs = generate_uniform_random_vector<uint32_t>(0, 100, byte_size / sizeof(uint32_t));
    tt_metal::WriteToBuffer(input_dram_buffer, inputs);

    pass &= tt_metal::ConfigureDeviceWithProgram(device, program);
    pass &= tt_metal::WriteRuntimeArgsToDevice(
        device,
        reader_kernel,
        reader_core,
        {
            (uint32_t)dram_byte_address,
            (uint32_t)dram_noc_xy.x,
            (uint32_t)dram_noc_xy.y,
            (uint32_t)l1_byte_address,
            (uint32_t)byte_size,
        });
    pass &= tt_metal::LaunchKernels(device, program);

    std::vector<uint32_t> dest_core_data;
    tt_metal::ReadFromDeviceL1(device, reader_core, l1_byte_address, byte_size, dest_core_data);
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
    const size_t& dram_channel,
    const size_t& dram_byte_address,
    const size_t& l1_byte_address,
    const CoreCoord& writer_core) {
    bool pass = true;
    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::Program program = tt_metal::Program();

    auto output_dram_buffer =
        tt_metal::Buffer(device, byte_size, dram_byte_address, dram_channel, byte_size, tt_metal::BufferType::DRAM);
    auto dram_noc_xy = output_dram_buffer.noc_coordinates();
    auto l1_bank_ids = device->bank_ids_from_logical_core(writer_core);
    auto l1_buffer =
        tt_metal::Buffer(device, byte_size, l1_byte_address, l1_bank_ids.at(0), byte_size, tt_metal::BufferType::L1);

    auto writer_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/l1_to_dram_copy.cpp",
        writer_core,
        tt_metal::DataMovementProcessor::RISCV_0,
        tt_metal::NOC::RISCV_0_default);

    ////////////////////////////////////////////////////////////////////////////
    //                      Compile and Execute Application
    ////////////////////////////////////////////////////////////////////////////
    pass &= tt_metal::CompileProgram(device, program);
    auto inputs = generate_uniform_random_vector<uint32_t>(0, 100, byte_size / sizeof(uint32_t));
    tt_metal::WriteToDeviceL1(device, writer_core, l1_byte_address, inputs);

    pass &= tt_metal::ConfigureDeviceWithProgram(device, program);
    pass &= tt_metal::WriteRuntimeArgsToDevice(
        device,
        writer_kernel,
        writer_core,
        {
            (uint32_t)dram_byte_address,
            (uint32_t)dram_noc_xy.x,
            (uint32_t)dram_noc_xy.y,
            (uint32_t)l1_byte_address,
            (uint32_t)byte_size,
        });
    pass &= tt_metal::LaunchKernels(device, program);

    std::vector<uint32_t> dest_buffer_data;
    tt_metal::ReadFromBuffer(output_dram_buffer, dest_buffer_data);
    pass &= (dest_buffer_data == inputs);
    if (not pass) {
        std::cout << "Mismatch at Core: " << writer_core.str() << std::endl;
    }
    return pass;
}

struct ReaderDatacopyWriterConfig {
    size_t num_tiles = 0;
    size_t tile_byte_size = 0;
    size_t output_dram_channel = 0;
    size_t output_dram_byte_address = 0;
    size_t input_dram_channel = 0;
    size_t input_dram_byte_address = 0;
    size_t l1_input_byte_address = 0;
    tt::DataFormat l1_input_data_format = tt::DataFormat::Invalid;
    size_t l1_output_byte_address = 0;
    tt::DataFormat l1_output_data_format = tt::DataFormat::Invalid;
    CoreCoord core = {};
};
/// @brief Does Dram --> Reader --> CB --> Datacopy --> CB --> Writer --> Dram on a single core
/// @param device
/// @param test_config - Configuration of the test -- see struct
/// @return
bool reader_datacopy_writer(tt_metal::Device* device, const ReaderDatacopyWriterConfig& test_config) {
    bool pass = true;
    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    const size_t byte_size = test_config.num_tiles * test_config.tile_byte_size;
    tt_metal::Program program = tt_metal::Program();
    auto input_dram_buffer = tt_metal::Buffer(
        device,
        byte_size,
        test_config.input_dram_byte_address,
        test_config.input_dram_channel,
        byte_size,
        tt_metal::BufferType::DRAM);
    auto input_dram_noc_xy = input_dram_buffer.noc_coordinates();
    auto output_dram_buffer = tt_metal::Buffer(
        device,
        byte_size,
        test_config.output_dram_byte_address,
        test_config.output_dram_channel,
        byte_size,
        tt_metal::BufferType::DRAM);
    auto output_dram_noc_xy = output_dram_buffer.noc_coordinates();

    auto l1_input_cb = tt_metal::CreateCircularBuffer(
        program,
        device,
        0,
        test_config.core,
        test_config.num_tiles,
        byte_size,
        test_config.l1_input_byte_address,
        test_config.l1_input_data_format);
    auto l1_output_cb = tt_metal::CreateCircularBuffer(
        program,
        device,
        16,
        test_config.core,
        test_config.num_tiles,
        byte_size,
        test_config.l1_output_byte_address,
        test_config.l1_output_data_format);

    auto reader_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/reader_unary.cpp",
        test_config.core,
        tt_metal::DataMovementProcessor::RISCV_1,
        tt_metal::NOC::RISCV_1_default);

    auto writer_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/writer_unary.cpp",
        test_config.core,
        tt_metal::DataMovementProcessor::RISCV_0,
        tt_metal::NOC::RISCV_0_default);

    vector<uint32_t> compute_kernel_args = {
        uint(test_config.num_tiles)  // per_core_tile_cnt
    };
    bool fp32_dest_acc_en = false;
    bool math_approx_mode = false;
    auto datacopy_kernel = tt_metal::CreateComputeKernel(
        program,
        "tt_metal/kernels/compute/eltwise_copy.cpp",
        test_config.core,
        compute_kernel_args,
        MathFidelity::HiFi4,
        fp32_dest_acc_en,
        math_approx_mode);

    ////////////////////////////////////////////////////////////////////////////
    //                      Stimulus Generation
    ////////////////////////////////////////////////////////////////////////////
    std::vector<uint32_t> inputs = generate_packed_uniform_random_vector<uint32_t, bfloat16>(
        -1.0f, 1.0f, byte_size / bfloat16::SIZEOF, std::chrono::system_clock::now().time_since_epoch().count());
    ////////////////////////////////////////////////////////////////////////////
    //                      Compile and Execute Application
    ////////////////////////////////////////////////////////////////////////////
    pass &= tt_metal::CompileProgram(device, program);
    tt_metal::WriteToBuffer(input_dram_buffer, inputs);

    pass &= tt_metal::ConfigureDeviceWithProgram(device, program);
    pass &= tt_metal::WriteRuntimeArgsToDevice(
        device,
        reader_kernel,
        test_config.core,
        {
            (uint32_t)test_config.input_dram_byte_address,
            (uint32_t)input_dram_noc_xy.x,
            (uint32_t)input_dram_noc_xy.y,
            (uint32_t)test_config.num_tiles,
        });
    pass &= tt_metal::WriteRuntimeArgsToDevice(
        device,
        writer_kernel,
        test_config.core,
        {
            (uint32_t)test_config.output_dram_byte_address,
            (uint32_t)output_dram_noc_xy.x,
            (uint32_t)output_dram_noc_xy.y,
            (uint32_t)test_config.num_tiles,
        });
    pass &= tt_metal::LaunchKernels(device, program);

    std::vector<uint32_t> dest_buffer_data;
    tt_metal::ReadFromBuffer(output_dram_buffer, dest_buffer_data);
    pass &= inputs == dest_buffer_data;
    return pass;
}
}  // namespace unit_tests::single_core_dram

TEST_SUITE("SingleCoreDram") {
    TEST_CASE_FIXTURE(unit_tests::SingleDeviceFixture, "ReaderOnly") {
        REQUIRE(unit_tests::single_core_dram::reader_only(device_, 1 * 1024, 0, 0, UNRESERVED_BASE, {.x = 0, .y = 0}));
        REQUIRE(unit_tests::single_core_dram::reader_only(device_, 2 * 1024, 0, 0, UNRESERVED_BASE, {.x = 0, .y = 0}));
        REQUIRE(unit_tests::single_core_dram::reader_only(device_, 16 * 1024, 0, 0, UNRESERVED_BASE, {.x = 0, .y = 0}));
    }
    TEST_CASE_FIXTURE(unit_tests::SingleDeviceFixture, "WriterOnly") {
        REQUIRE(unit_tests::single_core_dram::writer_only(device_, 1 * 1024, 0, 0, UNRESERVED_BASE, {.x = 0, .y = 0}));
        REQUIRE(unit_tests::single_core_dram::writer_only(device_, 2 * 1024, 0, 0, UNRESERVED_BASE, {.x = 0, .y = 0}));
        REQUIRE(unit_tests::single_core_dram::writer_only(device_, 16 * 1024, 0, 0, UNRESERVED_BASE, {.x = 0, .y = 0}));
    }

    // FIXME: We should add two test variants --
    //    1. Single Core Dram Mechanic -- Reader -> CB --> Writer
    //    2. Single Core Compute Datacopy -- WriteToDeviceL1 --> Datacopy (Unpack/Math/Pack) --> ReadFromDeviceL1
    TEST_CASE_FIXTURE(unit_tests::SingleDeviceFixture, "ReaderDatacopyWriter") {
        unit_tests::single_core_dram::ReaderDatacopyWriterConfig test_config = {
            .num_tiles = 1,
            .tile_byte_size = 2 * 32 * 32,
            .output_dram_channel = 0,
            .output_dram_byte_address = 0,
            .input_dram_channel = 0,
            .input_dram_byte_address = 16 * 32 * 32,
            .l1_input_byte_address = UNRESERVED_BASE,
            .l1_input_data_format = tt::DataFormat::Float16_b,
            .l1_output_byte_address = UNRESERVED_BASE + 16 * 32 * 32,
            .l1_output_data_format = tt::DataFormat::Float16_b,
            .core = {.x = 0, .y = 0}};
        SUBCASE("SingleTile") {
            test_config.num_tiles = 1;
            REQUIRE(reader_datacopy_writer(device_, test_config));
        }
        SUBCASE("MultiTile") {
            test_config.num_tiles = 4;
            REQUIRE(reader_datacopy_writer(device_, test_config));
            test_config.num_tiles = 8;
            REQUIRE(reader_datacopy_writer(device_, test_config));
        }
    }
}
