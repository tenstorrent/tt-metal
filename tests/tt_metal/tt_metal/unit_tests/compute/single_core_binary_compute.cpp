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

namespace unit_tests::single_core_binary_compute {
const map<string, string> binary_op_name_to_op_code = {
    {"add", "0"},
    {"sub", "1"},
    {"mul", "2"},
};
const map<string, string> binary_op_name_to_op_kernel = {
    {"add", "add_tiles"},
    {"sub", "sub_tiles"},
    {"mul", "mul_tiles"},
};

struct SingleCoreBinaryConfig {
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
    std::string binary_op = "";
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
    tt_metal::Program program = tt_metal::Program();
    auto input0_dram_buffer = tt_metal::Buffer(
        device,
        byte_size,
        test_config.input_dram_byte_address,
        test_config.input_dram_channel,
        byte_size,
        tt_metal::BufferType::DRAM);
    auto input0_dram_noc_xy = input0_dram_buffer.noc_coordinates();
    auto input1_dram_buffer = tt_metal::Buffer(
        device,
        byte_size,
        test_config.input_dram_byte_address + byte_size,
        test_config.input_dram_channel,
        byte_size,
        tt_metal::BufferType::DRAM);
    auto input1_dram_noc_xy = input1_dram_buffer.noc_coordinates();
    auto output_dram_buffer = tt_metal::Buffer(
        device,
        byte_size,
        test_config.output_dram_byte_address,
        test_config.output_dram_channel,
        byte_size,
        tt_metal::BufferType::DRAM);
    auto output_dram_noc_xy = output_dram_buffer.noc_coordinates();

    auto l1_input0_cb = tt_metal::CreateCircularBuffer(
        program,
        device,
        0,
        test_config.core,
        test_config.num_tiles,
        byte_size,
        test_config.l1_input_byte_address,
        test_config.l1_input_data_format);
    auto l1_input1_cb = tt_metal::CreateCircularBuffer(
        program,
        device,
        1,
        test_config.core,
        test_config.num_tiles,
        byte_size,
        test_config.l1_input_byte_address + byte_size,
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
        "tt_metal/kernels/dataflow/reader_binary.cpp",
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
        uint32_t(test_config.num_tiles),  // per_core_block_cnt
        1                                 // per_core_block_cnt
    };
    bool fp32_dest_acc_en = false;
    bool math_approx_mode = false;
    auto binary_kernel = tt_metal::CreateComputeKernel(
        program,
        "tt_metal/kernels/compute/eltwise_binary.cpp",
        test_config.core,
        compute_kernel_args,
        MathFidelity::HiFi4,
        fp32_dest_acc_en,
        math_approx_mode);
    binary_kernel->add_define("ELTWISE_OP_CODE", binary_op_name_to_op_code.at(test_config.binary_op));
    binary_kernel->add_define("ELTWISE_OP", binary_op_name_to_op_kernel.at(test_config.binary_op));

    ////////////////////////////////////////////////////////////////////////////
    //                      Stimulus Generation
    ////////////////////////////////////////////////////////////////////////////
    std::vector<uint32_t> packed_input0 = generate_packed_uniform_random_vector<uint32_t, bfloat16>(
        -1.0f, 1.0f, byte_size / bfloat16::SIZEOF, std::chrono::system_clock::now().time_since_epoch().count());
    std::vector<uint32_t> packed_input1 = generate_packed_uniform_random_vector<uint32_t, bfloat16>(
        0.1f, 2.0f, byte_size / bfloat16::SIZEOF, std::chrono::system_clock::now().time_since_epoch().count());
    ////////////////////////////////////////////////////////////////////////////
    //                      Golden Generation
    ////////////////////////////////////////////////////////////////////////////
    auto input0 = unpack_vector<bfloat16, uint32_t>(packed_input0);
    auto input1 = unpack_vector<bfloat16, uint32_t>(packed_input1);
    std::vector<bfloat16> golden(input0.size());
    std::transform(
        input0.begin(), input0.end(), input1.begin(), golden.begin(), [&](const bfloat16& lhs, const bfloat16& rhs) {
            if (test_config.binary_op == "add") {
                return (lhs.to_float() + rhs.to_float());
            } else if (test_config.binary_op == "sub") {
                return (lhs.to_float() - rhs.to_float());
            } else if (test_config.binary_op == "mul") {
                return (lhs.to_float() * rhs.to_float());
            } else {
                log_fatal("Unsupported binary_op={}", test_config.binary_op);
                return 0.0f;
            }
        });
    auto packed_golden = pack_vector<uint32_t, bfloat16>(golden);

    ////////////////////////////////////////////////////////////////////////////
    //                      Compile and Execute Application
    ////////////////////////////////////////////////////////////////////////////
    pass &= tt_metal::CompileProgram(device, program);
    tt_metal::WriteToBuffer(input0_dram_buffer, packed_input0);
    tt_metal::WriteToBuffer(input1_dram_buffer, packed_input1);

    pass &= tt_metal::ConfigureDeviceWithProgram(device, program);
    pass &= tt_metal::WriteRuntimeArgsToDevice(
        device,
        reader_kernel,
        test_config.core,
        {
            (uint32_t)test_config.input_dram_byte_address,
            (uint32_t)input0_dram_noc_xy.x,
            (uint32_t)input0_dram_noc_xy.y,
            (uint32_t)(test_config.input_dram_byte_address + byte_size),
            (uint32_t)input1_dram_noc_xy.x,
            (uint32_t)input1_dram_noc_xy.y,
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

    ////////////////////////////////////////////////////////////////////////////
    //                      Comparison Checking
    ////////////////////////////////////////////////////////////////////////////
    std::vector<uint32_t> dest_buffer_data;
    tt_metal::ReadFromBuffer(output_dram_buffer, dest_buffer_data);
    pass &= is_close_packed_vectors<bfloat16, uint32_t>(
        dest_buffer_data, packed_golden, [&](const bfloat16& a, const bfloat16& b) { return is_close(a, b, 0.015f); });
    return pass;
}
}  // namespace unit_tests::single_core_binary_compute
TEST_SUITE("BinaryCompute") {
    TEST_CASE_FIXTURE(unit_tests::SingleDeviceFixture, "SingleCore") {
        unit_tests::single_core_binary_compute::SingleCoreBinaryConfig test_config = {
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
            SUBCASE("add") {
                test_config.binary_op = "add";
                REQUIRE(unit_tests::single_core_binary_compute::single_core_binary(device_, test_config));
            }
            SUBCASE("sub") {
                test_config.binary_op = "sub";
                REQUIRE(unit_tests::single_core_binary_compute::single_core_binary(device_, test_config));
            }
            SUBCASE("sub") {
                test_config.binary_op = "sub";
                REQUIRE(unit_tests::single_core_binary_compute::single_core_binary(device_, test_config));
            }
        }
        SUBCASE("MultiTile") {
            test_config.num_tiles = 4;
            SUBCASE("add") {
                test_config.binary_op = "add";
                REQUIRE(unit_tests::single_core_binary_compute::single_core_binary(device_, test_config));
            }
            SUBCASE("sub") {
                test_config.binary_op = "sub";
                REQUIRE(unit_tests::single_core_binary_compute::single_core_binary(device_, test_config));
            }
            SUBCASE("sub") {
                test_config.binary_op = "sub";
                REQUIRE(unit_tests::single_core_binary_compute::single_core_binary(device_, test_config));
            }
        }
    }
}
