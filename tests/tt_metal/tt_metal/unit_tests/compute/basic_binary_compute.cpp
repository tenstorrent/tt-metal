#include <algorithm>
#include <functional>
#include <random>

#include "bfloat16.hpp"
#include "gtest/gtest.h"
#include "tt_metal/host_api.hpp"
#include "tt_metal/hostdevcommon/common_runtime_address_map.h"
#include "tt_metal/test_utils/env_vars.hpp"
#include "tt_metal/test_utils/print_helpers.hpp"
#include "tt_metal/test_utils/stimulus.hpp"

using namespace tt;
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
class BasicBinaryTest : public ::testing::Test {
   protected:
    void SetUp() override {
        const tt::ARCH arch = tt::get_arch_from_string(tt::test_utils::get_env_arch_name());
        const int pci_express_slot = 0;
        device_ = tt_metal::CreateDevice(arch, pci_express_slot);
        tt_metal::InitializeDevice(device_);
    }

    void TearDown() override { tt_metal::CloseDevice(device_); }
    tt_metal::Device* device_;
};

// Reader reads from single dram to core, writer synchronizes with datacopy kernel and writes to dram
// DRAM --> (Reader Core CB using reader RISCV)
// Reader Core --> Datacopy --> Reader Core
// Reader Core --> Writes to Dram
bool single_core_binary(
    tt_metal::Device* device,
    const size_t& num_tiles,
    const size_t& tile_byte_size,
    const size_t& output_dram_channel,
    const size_t& output_dram_byte_address,
    const size_t& input_dram_channel,
    const size_t& input_dram_byte_address,
    const size_t& local_core_input_byte_address,
    const tt::DataFormat& local_core_input_data_format,
    const size_t& local_core_output_byte_address,
    const tt::DataFormat& local_core_output_data_format,
    const CoreCoord& core,
    const string& binary_op) {
    bool pass = true;
    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    const size_t byte_size = num_tiles * tile_byte_size;
    tt_metal::Program program = tt_metal::Program();
    auto input0_dram_buffer = tt_metal::Buffer(
        device, byte_size, input_dram_byte_address, input_dram_channel, byte_size, tt_metal::BufferType::DRAM);
    auto input0_dram_noc_xy = input0_dram_buffer.noc_coordinates();
    auto input1_dram_buffer = tt_metal::Buffer(
        device,
        byte_size,
        input_dram_byte_address + byte_size,
        input_dram_channel,
        byte_size,
        tt_metal::BufferType::DRAM);
    auto input1_dram_noc_xy = input1_dram_buffer.noc_coordinates();
    auto output_dram_buffer = tt_metal::Buffer(
        device, byte_size, output_dram_byte_address, output_dram_channel, byte_size, tt_metal::BufferType::DRAM);
    auto output_dram_noc_xy = output_dram_buffer.noc_coordinates();

    auto l1_input0_cb = tt_metal::CreateCircularBuffer(
        program, device, 0, core, num_tiles, byte_size, local_core_input_byte_address, local_core_input_data_format);
    auto l1_input1_cb = tt_metal::CreateCircularBuffer(
        program,
        device,
        1,
        core,
        num_tiles,
        byte_size,
        local_core_input_byte_address + byte_size,
        local_core_input_data_format);
    auto l1_output_cb = tt_metal::CreateCircularBuffer(
        program, device, 16, core, num_tiles, byte_size, local_core_output_byte_address, local_core_output_data_format);

    auto reader_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/reader_binary.cpp",
        core,
        tt_metal::DataMovementProcessor::RISCV_1,
        tt_metal::NOC::RISCV_1_default);

    auto writer_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/writer_unary.cpp",
        core,
        tt_metal::DataMovementProcessor::RISCV_0,
        tt_metal::NOC::RISCV_0_default);

    vector<uint32_t> compute_kernel_args = {
        uint32_t(num_tiles),  // per_core_block_cnt
        1                     // per_core_block_cnt
    };
    bool fp32_dest_acc_en = false;
    bool math_approx_mode = false;
    auto binary_kernel = tt_metal::CreateComputeKernel(
        program,
        "tt_metal/kernels/compute/eltwise_binary.cpp",
        core,
        compute_kernel_args,
        MathFidelity::HiFi4,
        fp32_dest_acc_en,
        math_approx_mode);
    binary_kernel->add_define("ELTWISE_OP_CODE", binary_op_name_to_op_code.at(binary_op));
    binary_kernel->add_define("ELTWISE_OP", binary_op_name_to_op_kernel.at(binary_op));

    ////////////////////////////////////////////////////////////////////////////
    //                      Compile Application
    ////////////////////////////////////////////////////////////////////////////
    pass &= tt_metal::CompileProgram(device, program);

    ////////////////////////////////////////////////////////////////////////////
    //                      Execute Application
    ////////////////////////////////////////////////////////////////////////////
    std::vector<uint32_t> packed_input0 = tt::test_utils::generate_packed_uniform_random_vector<uint32_t, bfloat16>(
        -1.0f, 1.0f, byte_size / bfloat16::SIZEOF, std::chrono::system_clock::now().time_since_epoch().count());
    std::vector<uint32_t> packed_input1 = tt::test_utils::generate_packed_uniform_random_vector<uint32_t, bfloat16>(
        0.1f, 2.0f, byte_size / bfloat16::SIZEOF, std::chrono::system_clock::now().time_since_epoch().count());
    tt_metal::WriteToBuffer(input0_dram_buffer, packed_input0);
    tt_metal::WriteToBuffer(input1_dram_buffer, packed_input1);

    pass &= tt_metal::ConfigureDeviceWithProgram(device, program);
    pass &= tt_metal::WriteRuntimeArgsToDevice(
        device,
        reader_kernel,
        core,
        {
            (uint32_t)input_dram_byte_address,
            (uint32_t)input0_dram_noc_xy.x,
            (uint32_t)input0_dram_noc_xy.y,
            (uint32_t)(input_dram_byte_address + byte_size),
            (uint32_t)input1_dram_noc_xy.x,
            (uint32_t)input1_dram_noc_xy.y,
            (uint32_t)num_tiles,
        });
    pass &= tt_metal::WriteRuntimeArgsToDevice(
        device,
        writer_kernel,
        core,
        {
            (uint32_t)output_dram_byte_address,
            (uint32_t)output_dram_noc_xy.x,
            (uint32_t)output_dram_noc_xy.y,
            (uint32_t)num_tiles,
        });
    pass &= tt_metal::LaunchKernels(device, program);

    auto input0 = tt::test_utils::unpack_vector<bfloat16, uint32_t>(packed_input0);
    auto input1 = tt::test_utils::unpack_vector<bfloat16, uint32_t>(packed_input1);
    std::vector<bfloat16> golden(input0.size());
    std::transform(
        input0.begin(), input0.end(), input1.begin(), golden.begin(), [&](const bfloat16& lhs, const bfloat16& rhs) {
            if (binary_op == "add") {
                return (lhs.to_float() + rhs.to_float());
            } else if (binary_op == "sub") {
                return (lhs.to_float() - rhs.to_float());
            } else if (binary_op == "mul") {
                return (lhs.to_float() * rhs.to_float());
            } else {
                log_fatal("Unsupported binary_op={}", binary_op);
                return 0.0f;
            }
        });
    log_info("pack golden");
    auto packed_golden = tt::test_utils::pack_vector<uint32_t, bfloat16>(golden);
    std::vector<uint32_t> dest_buffer_data;
    tt_metal::ReadFromBuffer(output_dram_buffer, dest_buffer_data);
    log_info("compare golden==dest_buffer");
    pass &= packed_uint32_t_vector_comparison(
        dest_buffer_data, packed_golden, [&](const float& a, const float& b) { return is_close(a, b, 0.015f); });
    return pass;
}

TEST_F(BasicBinaryTest, SingleTileAdd) {
    EXPECT_TRUE(single_core_binary(
        device_,
        1,
        2 * 32 * 32,
        0,
        0,
        0,
        32 * 32 * 32,
        UNRESERVED_BASE,
        tt::DataFormat::Float16_b,
        UNRESERVED_BASE + 32 * 32 * 32,
        tt::DataFormat::Float16_b,
        {.x = 0, .y = 0},
        "add"));
}
TEST_F(BasicBinaryTest, SingleTileSub) {
    EXPECT_TRUE(single_core_binary(
        device_,
        1,
        2 * 32 * 32,
        0,
        0,
        0,
        32 * 32 * 32,
        UNRESERVED_BASE,
        tt::DataFormat::Float16_b,
        UNRESERVED_BASE + 32 * 32 * 32,
        tt::DataFormat::Float16_b,
        {.x = 0, .y = 0},
        "sub"));
}
TEST_F(BasicBinaryTest, SingleTileMul) {
    EXPECT_TRUE(single_core_binary(
        device_,
        1,
        2 * 32 * 32,
        0,
        0,
        0,
        32 * 32 * 32,
        UNRESERVED_BASE,
        tt::DataFormat::Float16_b,
        UNRESERVED_BASE + 32 * 32 * 32,
        tt::DataFormat::Float16_b,
        {.x = 0, .y = 0},
        "mul"));
}

TEST_F(BasicBinaryTest, MultiTileAdd) {
    EXPECT_TRUE(single_core_binary(
        device_,
        4,
        2 * 32 * 32,
        0,
        0,
        0,
        32 * 32 * 32,
        UNRESERVED_BASE,
        tt::DataFormat::Float16_b,
        UNRESERVED_BASE + 32 * 32 * 32,
        tt::DataFormat::Float16_b,
        {.x = 0, .y = 0},
        "add"));
}
TEST_F(BasicBinaryTest, MultiTileSub) {
    EXPECT_TRUE(single_core_binary(
        device_,
        4,
        2 * 32 * 32,
        0,
        0,
        0,
        32 * 32 * 32,
        UNRESERVED_BASE,
        tt::DataFormat::Float16_b,
        UNRESERVED_BASE + 32 * 32 * 32,
        tt::DataFormat::Float16_b,
        {.x = 0, .y = 0},
        "sub"));
}
TEST_F(BasicBinaryTest, MultiTileMul) {
    EXPECT_TRUE(single_core_binary(
        device_,
        4,
        2 * 32 * 32,
        0,
        0,
        0,
        32 * 32 * 32,
        UNRESERVED_BASE,
        tt::DataFormat::Float16_b,
        UNRESERVED_BASE + 32 * 32 * 32,
        tt::DataFormat::Float16_b,
        {.x = 0, .y = 0},
        "mul"));
}
