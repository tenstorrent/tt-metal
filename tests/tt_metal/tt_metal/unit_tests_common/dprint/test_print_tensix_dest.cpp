// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "common/bfloat16.hpp"
#include "dprint_fixture.hpp"
#include "gtest/gtest.h"
#include "test_utils.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/test_utils/comparison.hpp"
#include "tt_metal/test_utils/df/df.hpp"
#include "tt_metal/test_utils/print_helpers.hpp"
#include "tt_metal/test_utils/stimulus.hpp"
//////////////////////////////////////////////////////////////////////////////////////////
// A test for checking dprint
//////////////////////////////////////////////////////////////////////////////////////////
using namespace tt;
using namespace tt::tt_metal;
using namespace tt::test_utils;
using namespace tt::test_utils::df;

// Configuration for Data Flow Test involving Reader, Datacopy, and Writer
struct DestPrintTestConfig {
    size_t num_tiles = 0;
    tt::DataFormat data_format = tt::DataFormat::Invalid;
    CoreCoord core = {};
    bool remap = false;
    bool swizzle = false;
    std::string reader_kernel;
    std::string writer_kernel;
    std::string compute_kernel;

    size_t get_num_elements() const { return 1024 * num_tiles; }
    size_t get_tile_size() const { return tt::tile_size(data_format); }
    // Returns the size of the input buffer
    size_t get_input_buffer_size() const { return num_tiles * get_tile_size(); }
    // Returns the size of the output buffer
    size_t get_output_buffer_size() const { return num_tiles * get_tile_size(); }
};

// Type alias for a shared pointer to a Buffer in DRAM
using DramBuffer = std::shared_ptr<Buffer>;

// Generates the runtime arguments for the DRAM kernel
static std::vector<uint32_t> get_dram_kernel_runtime_arguments(const DramBuffer& dram_buffer, size_t num_tiles) {
    return {
        static_cast<uint32_t>(dram_buffer->address()),
        static_cast<uint32_t>(dram_buffer->noc_coordinates().x),
        static_cast<uint32_t>(dram_buffer->noc_coordinates().y),
        static_cast<uint32_t>(num_tiles),
    };
}

// Creates a circular buffer (L1 cache) for the specified core and data format
static CBHandle create_circular_buffer(
    tt_metal::Program& program,
    const tt_metal::CoreCoord& core_coord,
    uint32_t cb_index,
    tt::DataFormat data_format,
    uint32_t buffer_size) {
    tt_metal::CircularBufferConfig circular_buffer_config =
        tt_metal::CircularBufferConfig(buffer_size, {{cb_index, data_format}})
            .set_page_size(cb_index, tile_size(data_format));
    return tt_metal::CreateCircularBuffer(program, core_coord, circular_buffer_config);
}

// Creates a DRAM interleaved buffer configuration
static tt::tt_metal::InterleavedBufferConfig create_dram_interleaved_config(tt_metal::Device* device, size_t byte_size) {
    return {.device = device, .size = byte_size, .page_size = byte_size, .buffer_type = tt::tt_metal::BufferType::DRAM};
}

constexpr uint32_t DEFAULT_INPUT_CB_INDEX = 0;
constexpr uint32_t DEFAULT_OUTPUT_CB_INDEX = 16;

// Prepares the reader kernel by setting up the DRAM buffer, circular buffer, and kernel
static DramBuffer prepare_reader(tt_metal::Device* device,
                                 tt_metal::Program& program,
                                 const DestPrintTestConfig& config) {
    // Create input DRAM buffer
    auto input_dram_buffer =
        tt_metal::CreateBuffer(create_dram_interleaved_config(device, config.get_input_buffer_size()));

    // Create input circular buffer
    auto input_circular_buffer = create_circular_buffer(
        program, config.core, DEFAULT_INPUT_CB_INDEX, config.data_format, config.get_input_buffer_size());

    // Create reader kernel
    auto reader_kernel = tt_metal::CreateKernel(
        program,
        config.reader_kernel,
        config.core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt_metal::NOC::RISCV_1_default,
            .compile_args = {DEFAULT_INPUT_CB_INDEX}});

    // Set runtime arguments for the reader kernel
    tt_metal::SetRuntimeArgs(
        program, reader_kernel, config.core, get_dram_kernel_runtime_arguments(input_dram_buffer, config.num_tiles));

    return input_dram_buffer;
}

// Prepares the writer kernel by setting up the DRAM buffer, circular buffer, and kernel
static DramBuffer prepare_writer(tt_metal::Device* device, tt_metal::Program& program, const DestPrintTestConfig& config) {
    // Create output DRAM buffer
    auto output_dram_buffer =
        tt_metal::CreateBuffer(create_dram_interleaved_config(device, config.get_output_buffer_size()));

    // Create output circular buffer
    auto output_circular_buffer = create_circular_buffer(
        program, config.core, DEFAULT_OUTPUT_CB_INDEX, config.data_format, config.get_output_buffer_size());

    // Create writer kernel
    auto writer_kernel = tt_metal::CreateKernel(
        program,
        config.writer_kernel,
        config.core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .compile_args = {DEFAULT_OUTPUT_CB_INDEX}});

    // Set runtime arguments for the writer kernel
    tt_metal::SetRuntimeArgs(
        program, writer_kernel, config.core, get_dram_kernel_runtime_arguments(output_dram_buffer, config.num_tiles));
    return output_dram_buffer;
}

// Prepares the compute kernel with the specified program and test configuration
static KernelHandle prepare_compute(tt_metal::Program& program, const DestPrintTestConfig& config) {
    return tt_metal::CreateKernel(
        program,
        config.compute_kernel,
        config.core,
        tt_metal::ComputeConfig{
            .fp32_dest_acc_en = config.data_format == tt::DataFormat::Float32,
            .compile_args = {
                static_cast<uint32_t>(config.num_tiles),
                static_cast<uint32_t>(config.remap),
                static_cast<uint32_t>(config.swizzle)}});
}

// Generates input data based on the test configuration
static std::vector<uint32_t> generate_inputs(const DestPrintTestConfig& config) {
    if (config.data_format == tt::DataFormat::Float16_b)
        return tt::test_utils::generate_packed_increment_vector<uint32_t, tt::test_utils::df::bfloat16>(
            0.0f, config.get_num_elements(), 0.03125f, -1.1875f);

    if (config.data_format == tt::DataFormat::Float32)
        return tt::test_utils::generate_packed_increment_vector<uint32_t, tt::test_utils::df::float32>(
            0.0f, config.get_num_elements());

    ADD_FAILURE() << "Data format (" << config.data_format << ") not implemented!";
    return {};
}

static std::string generate_golden_output(std::vector<uint32_t> data, tt::DataFormat data_format) {
    std::stringstream ss;

    auto print_float = [&ss](uint32_t uvalue) {
        float value;
        memcpy(&value, &uvalue, sizeof(float));
        ss << std::setw(8) << value << " ";
    };

    auto print_new_line = [&ss, data_format](uint32_t i) {
        if (i > 0) {
            ss << std::endl;
        }
        int num_uint32_per_tile = (data_format == tt::DataFormat::Float32) ? 1024 : 512;

        if (i % num_uint32_per_tile == 0) {
            ss << "Tile ID = " << i / num_uint32_per_tile << std::endl;
        }
    };
    ss << std::fixed << std::setprecision(4);
    for (uint32_t i = 0; i < data.size(); ++i) {
        if (data_format == tt::DataFormat::Float32) {
            if (i % 16 == 0) {
                print_new_line(i);
            }
            print_float(data[i]);
        } else {
            if (i % 8 == 0) {
                print_new_line(i);
            }
            print_float((data[i] & 0x0000ffff) << 16);
            print_float(data[i] & 0xffff0000);
        }
    }
    ss << std::endl;
    return ss.str();
}

// Performs DRAM --> Reader --> CB --> Datacopy --> CB --> Writer --> DRAM on a single core
static bool reader_datacopy_writer(
    DPrintFixture* fixture, tt_metal::Device* device, const DestPrintTestConfig& config) {
    // Create program
    tt_metal::Program program = tt_metal::CreateProgram();

    // Prepare reader kernel and get input DRAM buffer
    auto input_dram_buffer = prepare_reader(device, program, config);

    // Prepare writer kernel and get output DRAM buffer
    auto output_dram_buffer = prepare_writer(device, program, config);

    // Prepare compute kernel
    auto compute_kernel = prepare_compute(program, config);

    // Generate input data
    auto input_data = generate_inputs(config);

    // Write input data to input DRAM buffer
    tt_metal::detail::WriteToBuffer(input_dram_buffer, input_data);

    // Run the program
    fixture->RunProgram(device, program);

    // Read output data from output DRAM buffer
    std::vector<uint32_t> output_data;
    tt_metal::detail::ReadFromBuffer(output_dram_buffer, output_data);

    auto golden_output = generate_golden_output(input_data, config.data_format);
    // Check the print log against golden output.
    EXPECT_TRUE(FilesMatchesString(DPrintFixture::dprint_file_name, golden_output));

    // Compare input and output data
    return input_data == output_data;
}

TEST_F(DPrintFixture, TestDestPrintFloat16b) {
    // Setup test configuration
    DestPrintTestConfig test_config = {
        .num_tiles = 2,
        .data_format = tt::DataFormat::Float16_b,
        .core = CoreCoord(0, 0),
        .reader_kernel = "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/dram/direct_reader_unary.cpp",
        .writer_kernel = "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/dram/direct_writer_unary.cpp",
        .compute_kernel = "tests/tt_metal/tt_metal/test_kernels/compute/eltwise_copy_print_dest.cpp"};

    // Run the test on the device
    this->RunTestOnDevice(
        [&](DPrintFixture* fixture, Device* device) { reader_datacopy_writer(fixture, device, test_config); },
        this->devices_[0]);
}

TEST_F(DPrintFixture, TestDestPrintFloat32) {
    // Setup test configuration
    DestPrintTestConfig test_config = {
        .num_tiles = 2,
        .data_format = tt::DataFormat::Float32,
        .core = CoreCoord(0, 0),
        .reader_kernel = "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/dram/direct_reader_unary.cpp",
        .writer_kernel = "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/dram/direct_writer_unary.cpp",
        .compute_kernel = "tests/tt_metal/tt_metal/test_kernels/compute/eltwise_copy_print_dest.cpp"};

    if (this->arch_ == ARCH::GRAYSKULL) {
        GTEST_SKIP() << "Float32 dest is not supported on grayskull.";
    }

    // Run the test on the device
    this->RunTestOnDevice(
        [&](DPrintFixture* fixture, Device* device) { reader_datacopy_writer(fixture, device, test_config); },
        this->devices_[0]);
}

TEST_F(DPrintFixture, TestDestPrintFloat32RemapAndSwizzle) {
    // Setup test configuration
    DestPrintTestConfig test_config = {
        .num_tiles = 3,
        .data_format = tt::DataFormat::Float32,
        .core = CoreCoord(0, 0),
        .remap = true,
        .swizzle = true,
        .reader_kernel = "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/dram/direct_reader_unary.cpp",
        .writer_kernel = "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/dram/direct_writer_unary.cpp",
        .compute_kernel = "tests/tt_metal/tt_metal/test_kernels/compute/eltwise_copy_print_dest.cpp"};

    if (this->arch_ == ARCH::GRAYSKULL) {
        GTEST_SKIP() << "Float32 dest is not supported on grayskull.";
    }

    // Run the test on the device
    this->RunTestOnDevice(
        [&](DPrintFixture* fixture, Device* device) { reader_datacopy_writer(fixture, device, test_config); },
        this->devices_[0]);
}
