// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <fmt/base.h>
#include <stdint.h>
#include <sys/types.h>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/logger.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <tt-metalium/assert.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/data_types.hpp>
#include <tt-metalium/device.hpp>
#include "device_fixture.hpp"
#include "gtest/gtest.h"
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program.hpp>
#include <tt_stl/span.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include "tt_metal/test_utils/stimulus.hpp"
#include <tt-metalium/utils.hpp>

using std::vector;
using namespace tt::tt_metal;

namespace basic_tests::buffer::banked {

struct BankedConfig {
    size_t num_tiles = 1;
    size_t size_bytes = 1 * 2 * 32 * 32;
    size_t page_size_bytes = 2 * 32 * 32;
    BufferType input_buffer_type = BufferType::L1;
    BufferType output_buffer_type = BufferType::L1;
    CoreCoord logical_core = CoreCoord(0, 0);
    tt::DataFormat l1_data_format = tt::DataFormat::Float16_b;
};

namespace local_test_functions {

/// @brief Does Direct/Banked Reader --> CB --> Direct/Banked Writer on a single core
/// @param device
/// @param test_config - Configuration of the test -- see struct
/// @return
bool reader_cb_writer(IDevice* device, const BankedConfig& cfg, const bool banked_reader, const bool banked_writer) {
    bool pass = true;

    const uint32_t cb_id = 0;
    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    Program program = CreateProgram();

    string reader_kernel_name = "";
    string writer_kernel_name = "";
    size_t input_page_size_bytes = 0;
    size_t output_page_size_bytes = 0;
    std::vector<uint32_t> reader_runtime_args = {};
    std::vector<uint32_t> writer_runtime_args = {};
    if (banked_reader) {
        reader_kernel_name = "tests/tt_metal/tt_metal/test_kernels/dataflow/banked_reader.cpp";
        input_page_size_bytes = cfg.page_size_bytes;
    } else {
        reader_kernel_name = "tests/tt_metal/tt_metal/test_kernels/dataflow/direct_reader_unary.cpp";
        input_page_size_bytes = cfg.size_bytes;
    }

    if (banked_writer) {
        writer_kernel_name = "tests/tt_metal/tt_metal/test_kernels/dataflow/banked_writer.cpp";
        output_page_size_bytes = cfg.page_size_bytes;
    } else {
        writer_kernel_name = "tests/tt_metal/tt_metal/test_kernels/dataflow/direct_writer_unary.cpp";
        output_page_size_bytes = cfg.size_bytes;
    }

    tt::tt_metal::InterleavedBufferConfig in_config{
        .device = device,
        .size = cfg.size_bytes,
        .page_size = input_page_size_bytes,
        .buffer_type = cfg.input_buffer_type};

    tt::tt_metal::InterleavedBufferConfig out_config{
        .device = device,
        .size = cfg.size_bytes,
        .page_size = output_page_size_bytes,
        .buffer_type = cfg.output_buffer_type};

    auto input_buffer = CreateBuffer(in_config);

    auto output_buffer = CreateBuffer(out_config);

    tt::log_debug(
        tt::LogTest, "Input buffer: [address: {} B, size: {} B]", input_buffer->address(), input_buffer->size());
    tt::log_debug(
        tt::LogTest, "Output buffer: [address: {} B, size: {} B]", output_buffer->address(), output_buffer->size());

    TT_FATAL(cfg.num_tiles * cfg.page_size_bytes == cfg.size_bytes, "Error");
    constexpr uint32_t num_pages_cb = 1;
    CircularBufferConfig input_buffer_cb_config =
        CircularBufferConfig(cfg.page_size_bytes, {{cb_id, cfg.l1_data_format}})
            .set_page_size(cb_id, cfg.page_size_bytes);
    auto input_buffer_cb = CreateCircularBuffer(program, cfg.logical_core, input_buffer_cb_config);

    bool input_is_dram = cfg.input_buffer_type == BufferType::DRAM;
    bool output_is_dram = cfg.output_buffer_type == BufferType::DRAM;
    std::map<std::string, std::string> reader_defines = {
        {"INTERFACE_WITH_L1", std::to_string((uint32_t)(not input_is_dram))}};
    std::map<std::string, std::string> writer_defines = {
        {"INTERFACE_WITH_L1", std::to_string((uint32_t)(not output_is_dram))}};

    auto reader_kernel = CreateKernel(
        program,
        reader_kernel_name,
        cfg.logical_core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::NOC_0,
            .compile_args = {cb_id, uint32_t(input_buffer->page_size()), (uint32_t)input_is_dram},
            .defines = reader_defines});
    auto writer_kernel = CreateKernel(
        program,
        writer_kernel_name,
        cfg.logical_core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::NOC_1,
            .compile_args = {cb_id, uint32_t(output_buffer->page_size()), (uint32_t)output_is_dram},
            .defines = writer_defines});

    if (banked_reader) {
        reader_runtime_args = {(uint32_t)input_buffer->address(), (uint32_t)cfg.num_tiles};
    } else {
        reader_runtime_args = {
            (uint32_t)input_buffer->address(),
            0,
            (uint32_t)cfg.num_tiles,
        };
    }
    if (banked_writer) {
        writer_runtime_args = {(uint32_t)output_buffer->address(), (uint32_t)cfg.num_tiles};
    } else {
        writer_runtime_args = {
            (uint32_t)output_buffer->address(),
            0,
            (uint32_t)cfg.num_tiles,
        };
    }
    ////////////////////////////////////////////////////////////////////////////
    //                      Compile and Execute Application
    ////////////////////////////////////////////////////////////////////////////
    auto input_packed =
        tt::test_utils::generate_uniform_random_vector<uint32_t>(0, 100, cfg.size_bytes / sizeof(uint32_t));
    detail::WriteToBuffer(input_buffer, input_packed);
    SetRuntimeArgs(program, reader_kernel, cfg.logical_core, reader_runtime_args);
    SetRuntimeArgs(program, writer_kernel, cfg.logical_core, writer_runtime_args);

    detail::LaunchProgram(device, program);
    std::vector<uint32_t> reread_input_packed;
    detail::ReadFromBuffer(input_buffer, reread_input_packed);

    std::vector<uint32_t> output_packed;
    detail::ReadFromBuffer(output_buffer, output_packed);

    pass &= (output_packed == input_packed);

    return pass;
}

/// @brief Does Interleaved Reader --> CB --> Datacopy --> CB --> Interleaved Writer --> on a single core
/// @param device
/// @param test_config - Configuration of the test -- see struct
/// @return
bool reader_datacopy_writer(IDevice* device, const BankedConfig& cfg) {
    bool pass = true;

    const uint32_t input0_cb_index = 0;
    const uint32_t output_cb_index = 16;
    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    Program program = CreateProgram();

    tt::tt_metal::InterleavedBufferConfig in_config{
        .device = device,
        .size = cfg.size_bytes,
        .page_size = cfg.page_size_bytes,
        .buffer_type = cfg.input_buffer_type};

    tt::tt_metal::InterleavedBufferConfig out_config{
        .device = device,
        .size = cfg.size_bytes,
        .page_size = cfg.page_size_bytes,
        .buffer_type = cfg.output_buffer_type};

    auto input_buffer = CreateBuffer(in_config);
    auto output_buffer = CreateBuffer(out_config);

    TT_FATAL(cfg.num_tiles * cfg.page_size_bytes == cfg.size_bytes, "Error");
    constexpr uint32_t num_pages_cb = 1;
    CircularBufferConfig l1_input_cb_config =
        CircularBufferConfig(cfg.page_size_bytes, {{input0_cb_index, cfg.l1_data_format}})
            .set_page_size(input0_cb_index, cfg.page_size_bytes);
    auto l1_input_cb = CreateCircularBuffer(program, cfg.logical_core, l1_input_cb_config);

    CircularBufferConfig l1_output_cb_config =
        CircularBufferConfig(cfg.page_size_bytes, {{output_cb_index, cfg.l1_data_format}})
            .set_page_size(output_cb_index, cfg.page_size_bytes);
    auto l1_output_cb = CreateCircularBuffer(program, cfg.logical_core, l1_output_cb_config);

    bool input_is_dram = cfg.input_buffer_type == BufferType::DRAM;
    bool output_is_dram = cfg.output_buffer_type == BufferType::DRAM;

    auto reader_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/banked_reader.cpp",
        cfg.logical_core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = {input0_cb_index, uint32_t(input_buffer->page_size()), (uint32_t)input_is_dram}});

    auto writer_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/banked_writer.cpp",
        cfg.logical_core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = {output_cb_index, uint32_t(output_buffer->page_size()), (uint32_t)output_is_dram}});

    vector<uint32_t> compute_kernel_args = {
        uint(cfg.num_tiles)  // per_core_tile_cnt
    };
    auto datacopy_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/eltwise_copy.cpp",
        cfg.logical_core,
        ComputeConfig{.compile_args = compute_kernel_args});

    ////////////////////////////////////////////////////////////////////////////
    //                      Stimulus Generation
    ////////////////////////////////////////////////////////////////////////////
    std::vector<uint32_t> input_packed = tt::test_utils::generate_packed_uniform_random_vector<uint32_t, bfloat16>(
        -1.0f, 1.0f, cfg.size_bytes / bfloat16::SIZEOF, std::chrono::system_clock::now().time_since_epoch().count());

    ////////////////////////////////////////////////////////////////////////////
    //                      Compile and Execute Appli   cation
    ////////////////////////////////////////////////////////////////////////////

    detail::WriteToBuffer(input_buffer, input_packed);

    SetRuntimeArgs(
        program,
        reader_kernel,
        cfg.logical_core,
        {
            (uint32_t)input_buffer->address(),
            (uint32_t)cfg.num_tiles,
        });
    SetRuntimeArgs(
        program,
        writer_kernel,
        cfg.logical_core,
        {
            (uint32_t)output_buffer->address(),
            (uint32_t)cfg.num_tiles,
        });
    detail::LaunchProgram(device, program);
    std::vector<uint32_t> dest_buffer_data;
    detail::ReadFromBuffer(output_buffer, dest_buffer_data);
    pass &= input_packed == dest_buffer_data;

    return pass;
}

}  // end namespace local_test_functions

TEST_F(DeviceFixture, TensixTestSingleCoreSingleTileBankedL1ReaderOnly) {
    for (unsigned int id = 0; id < num_devices_; id++) {
        BankedConfig test_config;
        EXPECT_TRUE(local_test_functions::reader_cb_writer(this->devices_.at(id), test_config, true, false));
    }
}

TEST_F(DeviceFixture, TensixTestSingleCoreMultiTileBankedL1ReaderOnly) {
    for (unsigned int id = 0; id < num_devices_; id++) {
        BankedConfig test_config;
        auto num_banks = devices_.at(id)->allocator()->get_num_banks(BufferType::L1);
        TT_FATAL(num_banks % 2 == 0, "Error");
        size_t num_tiles = num_banks / 2;
        size_t tile_increment = num_tiles;
        uint32_t num_iterations = 3;
        uint32_t index = 0;
        while (index < num_iterations) {
            test_config.num_tiles = num_tiles;
            test_config.size_bytes = test_config.num_tiles * 2 * 32 * 32;
            EXPECT_TRUE(local_test_functions::reader_cb_writer(this->devices_.at(id), test_config, true, false));
            num_tiles += tile_increment;
            index++;
        }
    }
}

TEST_F(DeviceFixture, TensixTestSingleCoreSingleTileBankedDramReaderOnly) {
    for (unsigned int id = 0; id < num_devices_; id++) {
        BankedConfig test_config;
        test_config.input_buffer_type = BufferType::DRAM;
        test_config.output_buffer_type = BufferType::DRAM;
        EXPECT_TRUE(local_test_functions::reader_cb_writer(this->devices_.at(id), test_config, true, false));
    }
}

TEST_F(DeviceFixture, TensixTestSingleCoreMultiTileBankedDramReaderOnly) {
    for (unsigned int id = 0; id < num_devices_; id++) {
        BankedConfig test_config;
        auto num_banks = devices_.at(id)->allocator()->get_num_banks(BufferType::DRAM);
        size_t num_tiles = num_banks / 2;
        size_t tile_increment = num_tiles;
        uint32_t num_iterations = 6;
        uint32_t index = 0;
        test_config.input_buffer_type = BufferType::DRAM;
        test_config.output_buffer_type = BufferType::DRAM;
        while (index < num_iterations) {
            test_config.num_tiles = num_tiles;
            test_config.size_bytes = test_config.num_tiles * 2 * 32 * 32;
            EXPECT_TRUE(local_test_functions::reader_cb_writer(this->devices_.at(id), test_config, true, false));
            num_tiles += tile_increment;
            index++;
        }
    }
}

TEST_F(DeviceFixture, TensixTestSingleCoreSingleTileBankedL1WriterOnly) {
    for (unsigned int id = 0; id < num_devices_; id++) {
        BankedConfig test_config;
        EXPECT_TRUE(local_test_functions::reader_cb_writer(this->devices_.at(id), test_config, false, true));
    }
}

TEST_F(DeviceFixture, TensixTestSingleCoreMultiTileBankedL1WriterOnly) {
    for (unsigned int id = 0; id < num_devices_; id++) {
        BankedConfig test_config;
        auto num_banks = devices_.at(id)->allocator()->get_num_banks(BufferType::L1);
        TT_FATAL(num_banks % 2 == 0, "Error");
        size_t num_tiles = num_banks / 2;
        size_t tile_increment = num_tiles;
        uint32_t num_iterations = 3;
        uint32_t index = 0;
        while (index < num_iterations) {
            test_config.num_tiles = num_tiles;
            test_config.size_bytes = test_config.num_tiles * 2 * 32 * 32;
            EXPECT_TRUE(local_test_functions::reader_cb_writer(this->devices_.at(id), test_config, false, true));
            num_tiles += tile_increment;
            index++;
        }
    }
}

TEST_F(DeviceFixture, TensixTestSingleCoreSingleTileBankedDramWriterOnly) {
    for (unsigned int id = 0; id < num_devices_; id++) {
        BankedConfig test_config;
        test_config.input_buffer_type = BufferType::DRAM;
        test_config.output_buffer_type = BufferType::DRAM;
        EXPECT_TRUE(local_test_functions::reader_cb_writer(this->devices_.at(id), test_config, false, true));
    }
}

TEST_F(DeviceFixture, TensixTestSingleCoreMultiTileBankedDramWriterOnly) {
    for (unsigned int id = 0; id < num_devices_; id++) {
        BankedConfig test_config;
        auto num_banks = devices_.at(id)->allocator()->get_num_banks(BufferType::DRAM);
        size_t num_tiles = num_banks / 2;
        size_t tile_increment = num_tiles;
        uint32_t num_iterations = 6;
        uint32_t index = 0;
        test_config.input_buffer_type = BufferType::DRAM;
        test_config.output_buffer_type = BufferType::DRAM;
        while (index < num_iterations) {
            test_config.num_tiles = num_tiles;
            test_config.size_bytes = test_config.num_tiles * 2 * 32 * 32;
            EXPECT_TRUE(local_test_functions::reader_cb_writer(this->devices_.at(id), test_config, false, true));
            num_tiles += tile_increment;
            index++;
        }
    }
}

TEST_F(DeviceFixture, TensixTestSingleCoreSingleTileBankedL1ReaderAndWriter) {
    for (unsigned int id = 0; id < num_devices_; id++) {
        BankedConfig test_config;
        EXPECT_TRUE(local_test_functions::reader_cb_writer(this->devices_.at(id), test_config, true, true));
    }
}

TEST_F(DeviceFixture, TensixTestSingleCoreMultiTileBankedL1ReaderAndWriter) {
    for (unsigned int id = 0; id < num_devices_; id++) {
        BankedConfig test_config;
        size_t num_tiles = devices_.at(id)->allocator()->get_num_banks(BufferType::L1);
        size_t tile_increment = num_tiles / 2;
        uint32_t num_iterations = 6;
        uint32_t index = 0;
        while (index < num_iterations) {
            test_config.num_tiles = num_tiles;
            test_config.size_bytes = test_config.num_tiles * 2 * 32 * 32;
            EXPECT_TRUE(local_test_functions::reader_cb_writer(this->devices_.at(id), test_config, true, true));
            num_tiles += tile_increment;
            index++;
        }
    }
}

TEST_F(DeviceFixture, TensixTestSingleCoreSingleTileBankedDramReaderAndWriter) {
    for (unsigned int id = 0; id < num_devices_; id++) {
        BankedConfig test_config;
        test_config.input_buffer_type = BufferType::DRAM;
        test_config.output_buffer_type = BufferType::DRAM;
        EXPECT_TRUE(local_test_functions::reader_cb_writer(this->devices_.at(id), test_config, true, true));
    }
}

TEST_F(DeviceFixture, TensixTestSingleCoreMultiTileBankedDramReaderAndWriter) {
    for (unsigned int id = 0; id < num_devices_; id++) {
        BankedConfig test_config;
        size_t num_tiles = devices_.at(id)->allocator()->get_num_banks(BufferType::L1);
        TT_FATAL(num_tiles % 2 == 0, "Error");
        size_t tile_increment = num_tiles / 2;
        uint32_t num_iterations = 6;
        uint32_t index = 0;
        test_config.input_buffer_type = BufferType::DRAM;
        test_config.output_buffer_type = BufferType::DRAM;
        while (index < num_iterations) {
            test_config.num_tiles = num_tiles;
            test_config.size_bytes = test_config.num_tiles * 2 * 32 * 32;
            EXPECT_TRUE(local_test_functions::reader_cb_writer(this->devices_.at(id), test_config, true, true));
            num_tiles += tile_increment;
            index++;
        }
    }
}

TEST_F(DeviceFixture, TensixTestSingleCoreSingleTileBankedDramReaderAndL1Writer) {
    for (unsigned int id = 0; id < num_devices_; id++) {
        BankedConfig test_config;
        test_config.input_buffer_type = BufferType::DRAM;
        EXPECT_TRUE(local_test_functions::reader_cb_writer(this->devices_.at(id), test_config, true, true));
    }
}

TEST_F(DeviceFixture, TensixTestSingleCoreMultiTileBankedDramReaderAndL1Writer) {
    for (unsigned int id = 0; id < num_devices_; id++) {
        BankedConfig test_config;
        test_config.input_buffer_type = BufferType::DRAM;

        size_t num_tiles = devices_.at(id)->allocator()->get_num_banks(BufferType::L1);
        TT_FATAL(num_tiles % 2 == 0, "Error");
        size_t tile_increment = num_tiles / 2;
        uint32_t num_iterations = 6;
        uint32_t index = 0;
        while (index < num_iterations) {
            test_config.num_tiles = num_tiles;
            test_config.size_bytes = test_config.num_tiles * 2 * 32 * 32;
            EXPECT_TRUE(local_test_functions::reader_cb_writer(this->devices_.at(id), test_config, true, true));
            num_tiles += tile_increment;
            index++;
        }
    }
}

TEST_F(DeviceFixture, TensixTestSingleCoreSingleTileBankedL1ReaderAndDramWriter) {
    for (unsigned int id = 0; id < num_devices_; id++) {
        BankedConfig test_config;
        test_config.output_buffer_type = BufferType::DRAM;
        EXPECT_TRUE(local_test_functions::reader_cb_writer(this->devices_.at(id), test_config, true, true));
    }
}

TEST_F(DeviceFixture, TensixTestSingleCoreMultiTileBankedL1ReaderAndDramWriter) {
    for (unsigned int id = 0; id < num_devices_; id++) {
        BankedConfig test_config;
        test_config.output_buffer_type = BufferType::DRAM;

        size_t num_tiles = devices_.at(id)->allocator()->get_num_banks(BufferType::L1);
        TT_FATAL(num_tiles % 2 == 0, "Error");
        size_t tile_increment = num_tiles / 2;
        uint32_t num_iterations = 6;
        uint32_t index = 0;
        while (index < num_iterations) {
            test_config.num_tiles = num_tiles;
            test_config.size_bytes = test_config.num_tiles * 2 * 32 * 32;
            EXPECT_TRUE(local_test_functions::reader_cb_writer(this->devices_.at(id), test_config, true, true));
            num_tiles += tile_increment;
            index++;
        }
    }
}

TEST_F(DeviceFixture, TensixTestSingleCoreMultiTileBankedL1ReaderDataCopyL1Writer) {
    for (unsigned int id = 0; id < num_devices_; id++) {
        BankedConfig test_config;
        size_t num_tiles = devices_.at(id)->allocator()->get_num_banks(BufferType::L1);
        TT_FATAL(num_tiles % 2 == 0, "Error");
        size_t tile_increment = num_tiles / 2;
        uint32_t num_iterations = 6;
        uint32_t index = 0;
        test_config.logical_core = this->devices_.at(id)->allocator()->get_logical_core_from_bank_id(0);
        while (index < num_iterations) {
            test_config.num_tiles = num_tiles;
            test_config.size_bytes = test_config.num_tiles * 2 * 32 * 32;
            EXPECT_TRUE(local_test_functions::reader_datacopy_writer(this->devices_.at(id), test_config));
            num_tiles += tile_increment;
            index++;
        }
    }
}

TEST_F(DeviceFixture, TensixTestSingleCoreMultiTileBankedDramReaderDataCopyDramWriter) {
    for (unsigned int id = 0; id < num_devices_; id++) {
        BankedConfig test_config;
        size_t num_tiles = devices_.at(id)->allocator()->get_num_banks(BufferType::DRAM);
        size_t tile_increment = num_tiles / 2;
        uint32_t num_iterations = 6;
        uint32_t index = 0;
        test_config.input_buffer_type = BufferType::DRAM;
        test_config.output_buffer_type = BufferType::DRAM;
        while (index < num_iterations) {
            test_config.num_tiles = num_tiles;
            test_config.size_bytes = test_config.num_tiles * 2 * 32 * 32;
            EXPECT_TRUE(local_test_functions::reader_datacopy_writer(this->devices_.at(id), test_config));
            num_tiles += tile_increment;
            index++;
        }
    }
}

TEST_F(DeviceFixture, TensixTestSingleCoreMultiTileBankedL1ReaderDataCopyDramWriter) {
    for (unsigned int id = 0; id < num_devices_; id++) {
        BankedConfig test_config;
        size_t num_tiles = devices_.at(id)->allocator()->get_num_banks(BufferType::L1);
        TT_FATAL(num_tiles % 2 == 0, "Error");
        size_t tile_increment = num_tiles / 2;
        uint32_t num_iterations = 6;
        uint32_t index = 0;
        test_config.logical_core = this->devices_.at(id)->allocator()->get_logical_core_from_bank_id(0);
        test_config.input_buffer_type = BufferType::L1;
        test_config.output_buffer_type = BufferType::DRAM;
        while (index < num_iterations) {
            test_config.num_tiles = num_tiles;
            test_config.size_bytes = test_config.num_tiles * 2 * 32 * 32;
            EXPECT_TRUE(local_test_functions::reader_datacopy_writer(this->devices_.at(id), test_config));
            num_tiles += tile_increment;
            index++;
        }
    }
}

TEST_F(DeviceFixture, TensixTestSingleCoreMultiTileBankedDramReaderDataCopyL1Writer) {
    for (unsigned int id = 0; id < num_devices_; id++) {
        BankedConfig test_config;
        size_t num_tiles = devices_.at(id)->allocator()->get_num_banks(BufferType::L1);
        TT_FATAL(num_tiles % 2 == 0, "Error");
        size_t tile_increment = num_tiles / 2;
        uint32_t num_iterations = 6;
        uint32_t index = 0;
        test_config.input_buffer_type = BufferType::DRAM;
        test_config.output_buffer_type = BufferType::L1;
        while (index < num_iterations) {
            test_config.num_tiles = num_tiles;
            test_config.size_bytes = test_config.num_tiles * 2 * 32 * 32;
            EXPECT_TRUE(local_test_functions::reader_datacopy_writer(this->devices_.at(id), test_config));
            num_tiles += tile_increment;
            index++;
        }
    }
}

}  // end namespace basic_tests::buffer::banked
