// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <string.h>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <functional>
#include <iomanip>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/data_types.hpp>
#include "debug_tools_fixture.hpp"
#include "debug_tools_test_utils.hpp"
#include "fmt/base.h"
#include "gtest/gtest.h"
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program.hpp>
#include <tt_stl/span.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include "tt_metal/test_utils/df/float32.hpp"
#include "tt_metal/test_utils/stimulus.hpp"
#include <umd/device/types/arch.hpp>

namespace tt {
namespace tt_metal {
class IDevice;
}  // namespace tt_metal
}  // namespace tt

namespace {
    constexpr size_t ELEMENTS_PER_TILE = 1024;
    constexpr size_t ELEMENTS_PER_TILE_FLOAT16 = 512;
    constexpr size_t ELEMENTS_PER_LINE_FLOAT32 = 16;
    constexpr size_t ELEMENTS_PER_LINE_INT32 = 16;
    constexpr size_t ELEMENTS_PER_LINE_FLOAT16 = 8;
    constexpr uint32_t DEFAULT_INPUT_CB_INDEX = 0;
    constexpr uint32_t DEFAULT_OUTPUT_CB_INDEX = 16;
}

namespace tt::test_utils::df {
class int32 {
private:
    int32_t value;

public:
    static constexpr size_t SIZEOF = sizeof(int32_t);

    int32(float f) : value(static_cast<int32_t>(f)) {}
    int32(uint32_t u) : value(static_cast<int32_t>(u)) {}

    float to_float() const { return static_cast<float>(value); }
    uint32_t to_packed() const { return static_cast<uint32_t>(value); }
};

class DataFormatHandler {
public:
    virtual ~DataFormatHandler() = default;

    // Template method that defines the algorithm structure
    void print_data(std::stringstream& ss, const std::vector<uint32_t>& data) {
        for (uint32_t i = 0; i < data.size(); ++i) {
            if (i % get_elements_per_line() == 0) {
                print_new_line(ss, i);
            }
            print_datum(ss, data[i]);
        }
        ss << std::endl;
    }

    virtual size_t get_elements_per_line() const = 0;
    virtual size_t get_elements_per_tile() const = 0;

protected:
    // Hook method to be implemented by derived classes
    virtual void print_datum(std::stringstream& ss, uint32_t datum) = 0;

    void print_new_line(std::stringstream& ss, uint32_t i) {
        if (i > 0) {
            ss << std::endl;
        }
        if (i % get_elements_per_tile() == 0) {
            ss << "Tile ID = " << i / get_elements_per_tile() << std::endl;
        }
    }
};

class Float32Handler : public DataFormatHandler {
public:
    size_t get_elements_per_line() const override { return ELEMENTS_PER_LINE_FLOAT32; }
    size_t get_elements_per_tile() const override { return ELEMENTS_PER_TILE; }

protected:
    void print_datum(std::stringstream& ss, uint32_t datum) override {
        float value;
        memcpy(&value, &datum, sizeof(float));
        ss << std::setw(8) << value << " ";
    }
};

class Int32Handler : public DataFormatHandler {
public:
    size_t get_elements_per_line() const override { return ELEMENTS_PER_LINE_INT32; }
    size_t get_elements_per_tile() const override { return ELEMENTS_PER_TILE; }

protected:
    void print_datum(std::stringstream& ss, uint32_t datum) override {
        ss << std::setw(8) << static_cast<int32_t>(datum) << " ";
    }
};

class Float16bHandler : public DataFormatHandler {
public:
    size_t get_elements_per_line() const override { return ELEMENTS_PER_LINE_FLOAT16; }
    size_t get_elements_per_tile() const override { return ELEMENTS_PER_TILE_FLOAT16; }

protected:
    void print_datum(std::stringstream& ss, uint32_t datum) override {
        uint32_t shifted_value1 = (datum & 0x0000ffff) << 16;
        uint32_t shifted_value2 = datum & 0xffff0000;

        float value1, value2;
        memcpy(&value1, &shifted_value1, sizeof(float));
        memcpy(&value2, &shifted_value2, sizeof(float));

        ss << std::setw(8) << value1 << " " << std::setw(8) << value2 << " ";
    }
};

// Factory function to create the appropriate handler
static DataFormatHandler& get_handler(tt::DataFormat data_format) {
    static Float32Handler float32_handler;
    static Int32Handler int32_handler;
    static Float16bHandler float16b_handler;

    switch (data_format) {
        case tt::DataFormat::Float32:
            return float32_handler;
        case tt::DataFormat::Int32:
            return int32_handler;
        case tt::DataFormat::Float16_b:
            return float16b_handler;
        default:
            ADD_FAILURE() << "Data format (" << data_format << ") not implemented!";
            return float32_handler; // Default case, should not be reached
    }
}
}

//////////////////////////////////////////////////////////////////////////////////////////
// A test for checking dprint
//////////////////////////////////////////////////////////////////////////////////////////
using namespace tt;
using namespace tt::tt_metal;
using namespace tt::test_utils;
using namespace tt::test_utils::df;

// Configuration for Data Flow Test involving Reader, Datacopy, and Writer
struct DestPrintTestConfig {
    static constexpr size_t DEFAULT_NUM_TILES = 1;

    size_t num_tiles = DEFAULT_NUM_TILES;
    tt::DataFormat data_format = tt::DataFormat::Invalid;
    CoreCoord core = {};
    bool remap = false;
    bool swizzle = false;
    std::string reader_kernel;
    std::string writer_kernel;
    std::string compute_kernel;

    size_t get_num_elements() const { return ELEMENTS_PER_TILE * num_tiles; }
    size_t get_tile_size() const { return tt::tile_size(data_format); }
    // Returns the size of the input buffer
    size_t get_input_buffer_size() const { return num_tiles * get_tile_size(); }
    // Returns the size of the output buffer
    size_t get_output_buffer_size() const { return num_tiles * get_tile_size(); }

    // Add validation method
    bool is_valid() const {
        return num_tiles > 0 && data_format != tt::DataFormat::Invalid && !reader_kernel.empty() &&
               !writer_kernel.empty() && !compute_kernel.empty();
    }
};

class DestPrintTestConfigBuilder {
public:
    DestPrintTestConfigBuilder() {
        config_.reader_kernel = "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/dram/direct_reader_unary.cpp";
        config_.writer_kernel = "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/dram/direct_writer_unary.cpp";
        config_.compute_kernel = "tests/tt_metal/tt_metal/test_kernels/compute/eltwise_copy_print_dest.cpp";
        config_.core = CoreCoord(0, 0);
    }

    DestPrintTestConfigBuilder& set_num_tiles(size_t num_tiles) {
        config_.num_tiles = num_tiles;
        return *this;
    }

    DestPrintTestConfigBuilder& set_data_format(tt::DataFormat format) {
        config_.data_format = format;
        return *this;
    }

    DestPrintTestConfigBuilder& set_remap(bool remap) {
        config_.remap = remap;
        return *this;
    }

    DestPrintTestConfigBuilder& set_swizzle(bool swizzle) {
        config_.swizzle = swizzle;
        return *this;
    }

    DestPrintTestConfig build() const {
        if (!config_.is_valid()) {
            throw std::runtime_error("Invalid test configuration");
        }
        return config_;
    }

private:
    DestPrintTestConfig config_;
};

// Type alias for a shared pointer to a Buffer in DRAM
using DramBuffer = std::shared_ptr<distributed::MeshBuffer>;

// Generates the runtime arguments for the DRAM kernel
static std::vector<uint32_t> get_dram_kernel_runtime_arguments(const DramBuffer& dram_buffer, size_t num_tiles) {
    return {
        static_cast<uint32_t>(dram_buffer->address()),
        static_cast<uint32_t>(0),
        static_cast<uint32_t>(num_tiles),
    };
}

// Creates a circular buffer (L1 cache) for the specified core and data format
static CBHandle create_circular_buffer(
    distributed::MeshWorkload& workload,
    const tt_metal::CoreCoord& core_coord,
    uint32_t cb_index,
    tt::DataFormat data_format,
    uint32_t buffer_size) {
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    tt_metal::CircularBufferConfig circular_buffer_config =
        tt_metal::CircularBufferConfig(buffer_size, {{cb_index, data_format}})
            .set_page_size(cb_index, tile_size(data_format));
    return tt_metal::CreateCircularBuffer(workload.get_programs().at(device_range), core_coord, circular_buffer_config);
}

// Creates a DRAM interleaved buffer configuration
static DramBuffer create_dram_mesh_buffer(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device, size_t byte_size) {
    distributed::DeviceLocalBufferConfig local_config = {
        .page_size = byte_size, .buffer_type = tt::tt_metal::BufferType::DRAM};
    distributed::ReplicatedBufferConfig buffer_config = {.size = byte_size};
    return distributed::MeshBuffer::create(buffer_config, local_config, mesh_device.get());
}

// Prepares the reader kernel by setting up the DRAM buffer, circular buffer, and kernel
static DramBuffer prepare_reader(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    distributed::MeshWorkload& workload,
    const DestPrintTestConfig& config) {
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    auto& program_ = workload.get_programs().at(device_range);
    // Create input DRAM buffer
    auto input_dram_buffer = create_dram_mesh_buffer(mesh_device, config.get_input_buffer_size());

    // Create input circular buffer
    create_circular_buffer(
        workload, config.core, DEFAULT_INPUT_CB_INDEX, config.data_format, config.get_input_buffer_size());

    // Create reader kernel
    auto reader_kernel = tt_metal::CreateKernel(
        program_,
        config.reader_kernel,
        config.core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt_metal::NOC::RISCV_1_default,
            .compile_args = {DEFAULT_INPUT_CB_INDEX}});

    // Set runtime arguments for the reader kernel
    tt_metal::SetRuntimeArgs(
        program_, reader_kernel, config.core, get_dram_kernel_runtime_arguments(input_dram_buffer, config.num_tiles));

    return input_dram_buffer;
}

// Prepares the writer kernel by setting up the DRAM buffer, circular buffer, and kernel
static DramBuffer prepare_writer(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    distributed::MeshWorkload& workload,
    const DestPrintTestConfig& config) {
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    auto& program_ = workload.get_programs().at(device_range);
    // Create output DRAM buffer
    auto output_dram_buffer = create_dram_mesh_buffer(mesh_device, config.get_output_buffer_size());

    // Create output circular buffer
    create_circular_buffer(
        workload, config.core, DEFAULT_OUTPUT_CB_INDEX, config.data_format, config.get_output_buffer_size());

    // Create writer kernel
    auto writer_kernel = tt_metal::CreateKernel(
        program_,
        config.writer_kernel,
        config.core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .compile_args = {DEFAULT_OUTPUT_CB_INDEX}});

    // Set runtime arguments for the writer kernel
    tt_metal::SetRuntimeArgs(
        program_, writer_kernel, config.core, get_dram_kernel_runtime_arguments(output_dram_buffer, config.num_tiles));
    return output_dram_buffer;
}

// Prepares the compute kernel with the specified program and test configuration
static KernelHandle prepare_compute(distributed::MeshWorkload& workload, const DestPrintTestConfig& config) {
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    auto& program_ = workload.get_programs().at(device_range);
    return tt_metal::CreateKernel(
        program_,
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
    switch (config.data_format) {
        case tt::DataFormat::Float16_b:
            return tt::test_utils::generate_packed_increment_vector<uint32_t, bfloat16>(
                0.0f, config.get_num_elements(), 0.03125f, -1.1875f);
        case tt::DataFormat::Float32:
            return tt::test_utils::generate_packed_increment_vector<uint32_t, tt::test_utils::df::float32>(
                0.0f, config.get_num_elements());
        case tt::DataFormat::Int32:
            return tt::test_utils::generate_packed_increment_vector<uint32_t, tt::test_utils::df::int32>(
                0.0f, config.get_num_elements());
        default:
            ADD_FAILURE() << "Data format (" << config.data_format << ") not implemented!";
            return {};
    }
}

static std::string generate_golden_output(const std::vector<uint32_t>& data, tt::DataFormat data_format) {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(4);

    auto& handler = get_handler(data_format);
    handler.print_data(ss, data);
    return ss.str();
}

// Performs DRAM --> Reader --> CB --> Datacopy --> CB --> Writer --> DRAM on a single core
static bool reader_datacopy_writer(
    DPrintMeshFixture* fixture,
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    const DestPrintTestConfig& config) {
    // Create program
    distributed::MeshWorkload workload;
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    tt_metal::Program program = tt_metal::CreateProgram();
    distributed::AddProgramToMeshWorkload(workload, std::move(program), device_range);
    auto& cq = mesh_device->mesh_command_queue();

    // Prepare reader kernel and get input DRAM buffer
    auto input_dram_buffer = prepare_reader(mesh_device, workload, config);

    // Prepare writer kernel and get output DRAM buffer
    auto output_dram_buffer = prepare_writer(mesh_device, workload, config);

    // Prepare compute kernel
    [[maybe_unused]] auto compute_kernel = prepare_compute(workload, config);

    // Generate input data
    auto input_data = generate_inputs(config);

    // Write input data to input DRAM buffer
    distributed::WriteShard(cq, input_dram_buffer, input_data, zero_coord);

    // Run the program
    fixture->RunProgram(mesh_device, workload);

    // Read output data from output DRAM buffer
    std::vector<uint32_t> output_data;
    distributed::ReadShard(cq, output_data, output_dram_buffer, zero_coord);

    auto golden_output = generate_golden_output(input_data, config.data_format);
    // Check the print log against golden output.
    EXPECT_TRUE(FilesMatchesString(DPrintMeshFixture::dprint_file_name, golden_output));

    // Compare input and output data
    return input_data == output_data;
}

// Helper function to run tests with proper error handling
static void run_test_with_config(
    DPrintMeshFixture* fixture,
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    const DestPrintTestConfig& config) {
    try {
        reader_datacopy_writer(fixture, mesh_device, config);
    } catch (const std::exception& e) {
        FAIL() << "Test failed with error: " << e.what();
    }
}

// Define test parameters
struct TestParams {
    tt::DataFormat data_format;
    size_t num_tiles;
    bool remap;
    bool swizzle;
    std::string test_name;
};

// Parameterized test fixture
class DestPrintTest : public DPrintMeshFixture, public ::testing::WithParamInterface<TestParams> {
protected:
    void SetUp() override { DPrintMeshFixture::SetUp(); }

    void TearDown() override { DPrintMeshFixture::TearDown(); }

    void RunDestPrintTest(const DestPrintTestConfig& config) {
        if (config.data_format == tt::DataFormat::Float32 && this->arch_ == ARCH::GRAYSKULL) {
            GTEST_SKIP() << "Float32 dest is not supported on grayskull.";
        }

        if (config.data_format == tt::DataFormat::Int32 && this->arch_ != ARCH::BLACKHOLE) {
            GTEST_SKIP() << "Int32 dest is not supported on non-blackhole.";
        }

        this->RunTestOnDevice(
            [&](DPrintMeshFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
                run_test_with_config(fixture, mesh_device, config);
            },
            this->devices_[0]);
    }

    DestPrintTestConfig CreateTestConfig(const TestParams& params) {
        return DestPrintTestConfigBuilder()
            .set_num_tiles(params.num_tiles)
            .set_data_format(params.data_format)
            .set_remap(params.remap)
            .set_swizzle(params.swizzle)
            .build();
    }
};

// Overload the output stream operator for TestParams
std::ostream& operator<<(std::ostream& os, const TestParams& params) {
    return os << "DestPrintTest: " << params.test_name
              << " [DataFormat: " << static_cast<int>(params.data_format)
              << ", NumTiles: " << params.num_tiles
              << ", Remap: " << (params.remap ? "true" : "false")
              << ", Swizzle: " << (params.swizzle ? "true" : "false") << "]";
}

// Define test parameters with more combinations
const std::vector<TestParams> kTestParams = {
    // Float16b tests
    {tt::DataFormat::Float16_b, 1, false, false, "Float16b_NoRemapNoSwizzle"},
    {tt::DataFormat::Float16_b, 1, true, false, "Float16b_RemapNoSwizzle"},
    {tt::DataFormat::Float16_b, 1, false, true, "Float16b_NoRemapSwizzle"},
    {tt::DataFormat::Float16_b, 1, true, true, "Float16b_RemapSwizzle"},

    // Float32 tests
    {tt::DataFormat::Float32, 1, false, false, "Float32_NoRemapNoSwizzle"},
    {tt::DataFormat::Float32, 1, true, false, "Float32_RemapNoSwizzle"},
    {tt::DataFormat::Float32, 1, false, true, "Float32_NoRemapSwizzle"},
    {tt::DataFormat::Float32, 1, true, true, "Float32_RemapSwizzle"},

    // Int32 tests
    {tt::DataFormat::Int32, 1, false, false, "Int32_NoRemapNoSwizzle"},
    {tt::DataFormat::Int32, 1, true, false, "Int32_RemapNoSwizzle"},
    {tt::DataFormat::Int32, 1, false, true, "Int32_NoRemapSwizzle"},
    {tt::DataFormat::Int32, 1, true, true, "Int32_RemapSwizzle"},

    // Additional test cases with different tile counts
    {tt::DataFormat::Float32, 3, true, true, "Float32_MultiTile_RemapSwizzle"},
    {tt::DataFormat::Float16_b, 3, true, true, "Float16b_MultiTile_RemapSwizzle"},
    {tt::DataFormat::Int32, 3, true, true, "Int32_MultiTile_RemapSwizzle"}
};

// Parameterized test
TEST_P(DestPrintTest, RunTest) {
    const auto& params = GetParam();
    auto config = CreateTestConfig(params);
    RunDestPrintTest(config);
}

// Register the test cases
INSTANTIATE_TEST_SUITE_P(
    DestPrintTests,
    DestPrintTest,
    ::testing::ValuesIn(kTestParams),
    [](const ::testing::TestParamInfo<TestParams>& info) {
        return info.param.test_name;
    }
);
