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
#include "tt_metal/detail/tt_metal.hpp"

using namespace tt;
using namespace tt::test_utils;
using namespace tt::test_utils::df;

namespace unit_tests::dram::banked {
struct BankedDramConfig {
    size_t num_tiles = 0;
    size_t size_bytes = 0;
    size_t page_size_bytes = 0;
    size_t input_dram_byte_address = 0;
    size_t output_dram_byte_address = 0;
    size_t l1_byte_address = 0;
    tt::DataFormat l1_data_format = tt::DataFormat::Invalid;
    CoreCoord target_core;
};
/// @brief Does Dram --> Banked Reader --> CB --> Direct Writer --> Dram on a single core
/// @param device
/// @param test_config - Configuration of the test -- see struct
/// @return
bool dram_reader_cb_writer_dram(
    tt_metal::Device* device, const BankedDramConfig& cfg, const bool banked_reader, const bool banked_writer) {

    // Once this test is uplifted to use fast dispatch, this can be removed.
    tt::tt_metal::detail::GLOBAL_CQ.reset();
    char env[] = "TT_METAL_SLOW_DISPATCH_MODE=1";
    putenv(env);

    bool pass = true;

    const uint32_t cb_id = 0;
    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::Program program = tt_metal::Program();

    string reader_kernel_name = "";
    string writer_kernel_name = "";
    size_t input_page_size_bytes = 0;
    size_t output_page_size_bytes = 0;
    std::vector<uint32_t> reader_runtime_args = {};
    std::vector<uint32_t> writer_runtime_args = {};
    if (banked_reader) {
        reader_kernel_name = "tt_metal/kernels/dataflow/unit_tests/dram/banked_reader_dram.cpp";
        input_page_size_bytes = cfg.page_size_bytes;
    } else {
        reader_kernel_name = "tt_metal/kernels/dataflow/unit_tests/dram/direct_reader_unary.cpp";
        input_page_size_bytes = cfg.size_bytes;
    }
    if (banked_writer) {
        writer_kernel_name = "tt_metal/kernels/dataflow/unit_tests/dram/banked_writer_dram.cpp";
        output_page_size_bytes = cfg.page_size_bytes;
    } else {
        writer_kernel_name = "tt_metal/kernels/dataflow/unit_tests/dram/direct_writer_unary.cpp";
        output_page_size_bytes = cfg.size_bytes;
    }

    // input
    auto input_dram_buffer = tt_metal::Buffer(
        device,
        cfg.size_bytes,
        cfg.input_dram_byte_address,
        input_page_size_bytes,
        tt_metal::BufferType::DRAM);

    // output
    auto output_dram_buffer = tt_metal::Buffer(
        device,
        cfg.size_bytes,
        cfg.output_dram_byte_address,
        output_page_size_bytes,
        tt_metal::BufferType::DRAM);

    // buffer_cb CB
    auto buffer_cb = tt_metal::CreateCircularBuffer(
        program,
        cb_id,
        cfg.target_core,
        cfg.num_tiles * 2,
        cfg.size_bytes * 2,
        cfg.l1_data_format,
        cfg.l1_byte_address);
    auto reader_kernel = tt_metal::CreateDataMovementKernel(
        program,
        reader_kernel_name,
        cfg.target_core,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::NOC_0, .compile_args = {cb_id,  uint32_t(input_dram_buffer.page_size())}});
    auto writer_kernel = tt_metal::CreateDataMovementKernel(
        program,
        writer_kernel_name,
        cfg.target_core,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::NOC_1, .compile_args = {cb_id,  uint32_t(output_dram_buffer.page_size())}});

    if (banked_reader) {
        reader_runtime_args = {
            (uint32_t)cfg.input_dram_byte_address,
            (uint32_t)cfg.num_tiles,
        };
    } else {
        reader_runtime_args = {
            (uint32_t)cfg.input_dram_byte_address,
            (uint32_t)input_dram_buffer.noc_coordinates().x,
            (uint32_t)input_dram_buffer.noc_coordinates().y,
            (uint32_t)cfg.num_tiles,
        };
    }
    if (banked_writer) {
        writer_runtime_args = {
            (uint32_t)cfg.output_dram_byte_address,
            (uint32_t)cfg.num_tiles,
        };
    } else {
        writer_runtime_args = {
            (uint32_t)cfg.output_dram_byte_address,
            (uint32_t)output_dram_buffer.noc_coordinates().x,
            (uint32_t)output_dram_buffer.noc_coordinates().y,
            (uint32_t)cfg.num_tiles,
        };
    }
    ////////////////////////////////////////////////////////////////////////////
    //                      Compile and Execute Application
    ////////////////////////////////////////////////////////////////////////////
    pass &= tt_metal::CompileProgram(device, program);
    auto input_packed = generate_uniform_random_vector<uint32_t>(0, 100, cfg.size_bytes / sizeof(uint32_t));
    tt_metal::WriteToBuffer(input_dram_buffer, input_packed);

    pass &= tt_metal::ConfigureDeviceWithProgram(device, program);
    tt_metal::SetRuntimeArgs(program, reader_kernel, cfg.target_core, reader_runtime_args);
    tt_metal::SetRuntimeArgs(program, writer_kernel, cfg.target_core, writer_runtime_args);
    tt_metal::WriteRuntimeArgsToDevice(device, program);
    pass &= tt_metal::LaunchKernels(device, program);

    std::vector<uint32_t> output_packed;
    tt_metal::ReadFromBuffer(output_dram_buffer, output_packed);
    int failed_index;
    pass &= is_close_packed_vectors<bfloat16, uint32_t>(
        output_packed,
        input_packed,
        [&](const bfloat16& a, const bfloat16& b) { return is_close(a, b, 0.005f); },
        &failed_index);
    if (not pass) {
        std::cout << "Failed Index: " << failed_index << endl;
    }
    return pass;
}

}  // namespace unit_tests::dram::banked

TEST_SUITE("SingleCoreBanked") {
    TEST_CASE_FIXTURE(unit_tests::SingleDeviceFixture, "BankedReaderOnly") {
        unit_tests::dram::banked::BankedDramConfig test_config = {
            .num_tiles = 1,
            .size_bytes = 1 * 2 * 32 * 32,
            .page_size_bytes = 2 * 32 * 32,
            .input_dram_byte_address = 0,
            .output_dram_byte_address = 512 * 1024 * 1204,
            .l1_byte_address = 500 * 32 * 32,
            .l1_data_format = tt::DataFormat::Float16_b,
            .target_core = {.x = 0, .y = 0}};
        SUBCASE("SingleTile") {
            REQUIRE(unit_tests::dram::banked::dram_reader_cb_writer_dram(device_, test_config, true, false));
        }
        SUBCASE("MultiTile") {
            test_config.num_tiles = 8;
            test_config.size_bytes = test_config.num_tiles * 2 * 32 * 32;
            REQUIRE(unit_tests::dram::banked::dram_reader_cb_writer_dram(device_, test_config, true, false));
            test_config.num_tiles = 12;
            test_config.size_bytes = test_config.num_tiles * 2 * 32 * 32;
            REQUIRE(unit_tests::dram::banked::dram_reader_cb_writer_dram(device_, test_config, true, false));
            test_config.num_tiles = 16;
            test_config.size_bytes = test_config.num_tiles * 2 * 32 * 32;
            REQUIRE(unit_tests::dram::banked::dram_reader_cb_writer_dram(device_, test_config, true, false));
            test_config.num_tiles = 24;
            test_config.size_bytes = test_config.num_tiles * 2 * 32 * 32;
            REQUIRE(unit_tests::dram::banked::dram_reader_cb_writer_dram(device_, test_config, true, false));
            test_config.num_tiles = 32;
            test_config.size_bytes = test_config.num_tiles * 2 * 32 * 32;
            REQUIRE(unit_tests::dram::banked::dram_reader_cb_writer_dram(device_, test_config, true, false));
            test_config.num_tiles = 36;
            test_config.size_bytes = test_config.num_tiles * 2 * 32 * 32;
            REQUIRE(unit_tests::dram::banked::dram_reader_cb_writer_dram(device_, test_config, true, false));
        }
    }
    TEST_CASE_FIXTURE(unit_tests::SingleDeviceFixture, "BankedWriterOnly") {
        unit_tests::dram::banked::BankedDramConfig test_config = {
            .num_tiles = 1,
            .size_bytes = 1 * 2 * 32 * 32,
            .page_size_bytes = 2 * 32 * 32,
            .input_dram_byte_address = 0,
            .output_dram_byte_address = 128 * 1024 * 1204,
            .l1_byte_address = 256 * 32 * 32,
            .l1_data_format = tt::DataFormat::Float16_b,
            .target_core = {.x = 0, .y = 0}};
        SUBCASE("SingleTile") {
            REQUIRE(unit_tests::dram::banked::dram_reader_cb_writer_dram(device_, test_config, false, true));
        }
        SUBCASE("MultiTile") {
            test_config.num_tiles = 8;
            test_config.size_bytes = test_config.num_tiles * 2 * 32 * 32;
            REQUIRE(unit_tests::dram::banked::dram_reader_cb_writer_dram(device_, test_config, false, true));
            test_config.num_tiles = 12;
            test_config.size_bytes = test_config.num_tiles * 2 * 32 * 32;
            REQUIRE(unit_tests::dram::banked::dram_reader_cb_writer_dram(device_, test_config, false, true));
            test_config.num_tiles = 16;
            test_config.size_bytes = test_config.num_tiles * 2 * 32 * 32;
            REQUIRE(unit_tests::dram::banked::dram_reader_cb_writer_dram(device_, test_config, false, true));
            test_config.num_tiles = 24;
            test_config.size_bytes = test_config.num_tiles * 2 * 32 * 32;
            REQUIRE(unit_tests::dram::banked::dram_reader_cb_writer_dram(device_, test_config, false, true));
            test_config.num_tiles = 32;
            test_config.size_bytes = test_config.num_tiles * 2 * 32 * 32;
            REQUIRE(unit_tests::dram::banked::dram_reader_cb_writer_dram(device_, test_config, false, true));
            test_config.num_tiles = 36;
            test_config.size_bytes = test_config.num_tiles * 2 * 32 * 32;
            REQUIRE(unit_tests::dram::banked::dram_reader_cb_writer_dram(device_, test_config, false, true));
        }
    }
    TEST_CASE_FIXTURE(unit_tests::SingleDeviceFixture, "BankedReaderWriterOnly") {
        unit_tests::dram::banked::BankedDramConfig test_config = {
            .num_tiles = 1,
            .size_bytes = 1 * 2 * 32 * 32,
            .page_size_bytes = 2 * 32 * 32,
            .input_dram_byte_address = 0,
            .output_dram_byte_address = 128 * 1024 * 1204,
            .l1_byte_address = 256 * 32 * 32,
            .l1_data_format = tt::DataFormat::Float16_b,
            .target_core = {.x = 0, .y = 0}};
        SUBCASE("SingleTile") {
            REQUIRE(unit_tests::dram::banked::dram_reader_cb_writer_dram(device_, test_config, true, true));
        }
        SUBCASE("MultiTile") {
            test_config.num_tiles = 8;
            test_config.size_bytes = test_config.num_tiles * 2 * 32 * 32;
            REQUIRE(unit_tests::dram::banked::dram_reader_cb_writer_dram(device_, test_config, true, true));
            test_config.num_tiles = 12;
            test_config.size_bytes = test_config.num_tiles * 2 * 32 * 32;
            REQUIRE(unit_tests::dram::banked::dram_reader_cb_writer_dram(device_, test_config, true, true));
            test_config.num_tiles = 16;
            test_config.size_bytes = test_config.num_tiles * 2 * 32 * 32;
            REQUIRE(unit_tests::dram::banked::dram_reader_cb_writer_dram(device_, test_config, true, true));
            test_config.num_tiles = 24;
            test_config.size_bytes = test_config.num_tiles * 2 * 32 * 32;
            REQUIRE(unit_tests::dram::banked::dram_reader_cb_writer_dram(device_, test_config, true, true));
            test_config.num_tiles = 32;
            test_config.size_bytes = test_config.num_tiles * 2 * 32 * 32;
            REQUIRE(unit_tests::dram::banked::dram_reader_cb_writer_dram(device_, test_config, true, true));
            test_config.num_tiles = 36;
            test_config.size_bytes = test_config.num_tiles * 2 * 32 * 32;
            REQUIRE(unit_tests::dram::banked::dram_reader_cb_writer_dram(device_, test_config, true, true));
        }
    }
    //TODO: Add non tiled unit tests for interleaved
}
