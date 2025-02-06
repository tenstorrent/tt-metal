// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <unordered_set>
#include <iostream>

#include <tt-metalium/bfloat16.hpp>
#include "debug_tools_fixture.hpp"
#include "gtest/gtest.h"
#include "debug_tools_test_utils.hpp"
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/host_api.hpp>
#include "tt_metal/test_utils/df/df.hpp"
#include "tt_metal/test_utils/stimulus.hpp"
//////////////////////////////////////////////////////////////////////////////////////////
// A test for checking dprint
//////////////////////////////////////////////////////////////////////////////////////////
using namespace tt;
using namespace tt::tt_metal;
using namespace tt::test_utils;
using namespace tt::test_utils::df;

// Register names
#define ALU_CONFIG 0
#define UNPACK_TILE_DESCRIPTOR 1
#define UNPACK_CONFIG 2
#define PACK_CONFIG 3
#define RELU_CONFIG 4
#define DEST_RD_CTRL 5
#define PACK_EDGE_OFFSET 6
#define PACK_COUNTERS 7
#define PACK_STRIDES 8

// Type of prints
const std::unordered_set<std::string> format_fields = {"ALU_FORMAT_SPEC_REG0_SrcA", "ALU_FORMAT_SPEC_REG1_SrcB",
    "ALU_FORMAT_SPEC_REG2_Dstacc", "in_data_format", "out_data_format"};
const std::unordered_set<std::string> decimal_fields = {"blobs_per_xy_plane", "x_dim", "y_dim", "z_dim", "w_dim", "blobs_y_start",
    "digest_size", "upsample_rate", "shift_amount", "fifo_size"};

// ALU CONFIG
const std::vector<std::string> field_names_alu_config = {"ALU_ROUNDING_MODE_Fpu_srnd_en", "ALU_ROUNDING_MODE_Gasket_srnd_en", "ALU_ROUNDING_MODE_Packer_srnd_en",
    "ALU_ROUNDING_MODE_Padding", "ALU_ROUNDING_MODE_GS_LF", "ALU_ROUNDING_MODE_Bfp8_HF", "ALU_FORMAT_SPEC_REG0_SrcAUnsigned",
    "ALU_FORMAT_SPEC_REG0_SrcBUnsigned", "ALU_FORMAT_SPEC_REG0_SrcA", "ALU_FORMAT_SPEC_REG1_SrcB", "ALU_FORMAT_SPEC_REG2_Dstacc",
    "ALU_ACC_CTRL_Fp32_enabled", "ALU_ACC_CTRL_SFPU_Fp32_enabled", "ALU_ACC_CTRL_INT8_math_enabled"};
const std::vector<uint32_t> field_values_alu_config = {1,0,1,15,0,1,1,0,0,1,0,0,0,1};

#ifdef ARCH_GRAYSKULL
// UNPACK TILE DESCRIPTOR
const std::vector<std::string> field_names_unpack_tile_descriptor = {"in_data_format", "uncompressed", "reserved_0",
    "blobs_per_xy_plane", "reserved_1", "x_dim", "y_dim", "z_dim", "w_dim", "blobs_y_start",
    "digest_type", "digest_size"};
const std::vector<uint32_t> field_values_unpack_tile_descriptor = {5,1,2,10,7,2,4,8,16,32,0,0};

// UNPACK CONFIG
const std::vector<std::string> field_names_unpack_config = {"out_data_format", "throttle_mode", "context_count", "haloize_mode",
    "tileize_mode", "force_shared_exp", "reserved_0", "upsample_rate", "upsample_and_interlave", "shift_amount",
    "uncompress_cntx0_3", "reserved_1", "uncompress_cntx4_7", "reserved_2", "limit_addr", "fifo_size"};
const std::vector<uint32_t> field_values_unpack_config = {0,1,2,0,1,0,0,3,0,16,5,0,2,0,28,29};
#else // ARCH_WORMHOLE or ARCH_BLACKHOLE
// UNPACK TILE DESCRIPTOR
const std::vector<std::string> field_names_unpack_tile_descriptor = {"in_data_format", "uncompressed", "reserved_0",
    "blobs_per_xy_plane", "reserved_1", "x_dim", "y_dim", "z_dim", "w_dim", "blobs_y_start_lo", "blobs_y_start_hi",
    "digest_type", "digest_size"};
const std::vector<uint32_t> field_values_unpack_tile_descriptor = {5,1,0,10,7,2,4,8,16,32,0,0,0};

// UNPACK CONFIG
const std::vector<std::string> field_names_unpack_config = {"out_data_format", "throttle_mode", "context_count", "haloize_mode",
    "tileize_mode", "unpack_src_reg_set_update", "unpack_if_sel", "upsample_rate", "reserved_1", "upsample_and_interlave",
    "shift_amount", "uncompress_cntx0_3", "unpack_if_sel_cntx0_3", "force_shared_exp", "reserved_2", "uncompress_cntx4_7",
    "unpack_if_sel_cntx4_7", "reserved_3", "limit_addr", "reserved_4", "fifo_size", "reserved_5"};
const std::vector<uint32_t> field_values_unpack_config = {0,1,2,0,1,1,0,3,0,0,16,5,6,0,0,2,3,0,28,0,29,0};
#endif

// Configuration for Data Flow Test involving Reader, Datacopy, and Writer
struct ConfigRegPrintTestConfig {
    CoreCoord core = {};
    std::string write_kernel;
    std::string print_kernel;
    int num_of_registers;
    std::vector<std::string> field_names;
    std::vector<uint32_t> field_values;
    uint32_t register_name;
};

// Dprints data format as string given an uint
static std::string data_format_to_string(uint8_t data_format) {
    switch (data_format) {
        case (uint8_t) DataFormat::Float32:
            return "Float32";
        case (uint8_t) DataFormat::Float16:
            return "Float16";
        case (uint8_t) DataFormat::Bfp8:
            return "Bfp8";
        case (uint8_t) DataFormat::Bfp4:
            return "Bfp4";
        case (uint8_t) DataFormat::Bfp2:
            return "Bfp2";
        case (uint8_t) DataFormat::Float16_b:
            return "Float16_b";
        case (uint8_t) DataFormat::Bfp8_b:
            return "Bfp8_b";
        case (uint8_t) DataFormat::Bfp4_b:
            return "Bfp4_b";
        case (uint8_t) DataFormat::Bfp2_b:
            return "Bfp2_b";
        case (uint8_t) DataFormat::Lf8:
            return "Lf8";
        case (uint8_t) DataFormat::Int8:
            return "Int8";
        case (uint8_t) DataFormat::UInt8:
            return "UInt8";
        case (uint8_t) DataFormat::UInt16:
            return "UInt16";
        case (uint8_t) DataFormat::Int32:
            return "Int32";
        case (uint8_t) DataFormat::UInt32:
            return "UInt32";
        case (uint8_t) DataFormat::Tf32:
            return "Tf32";
        default:
            return "INVALID DATA FORMAT";
    }
}

static std::string int_to_hex(int value) {
    std::stringstream ss;
    ss << std::hex << value; // Convert to hexadecimal
    return ss.str();
}

// Prepares the compute kernel with the specified program and test configuration
static KernelHandle prepare_writer(tt_metal::Program& program, const ConfigRegPrintTestConfig& config) {
    return tt_metal::CreateKernel(
        program,
        config.write_kernel,
        config.core,
        tt_metal::ComputeConfig{
            .compile_args = { config.register_name }});
}

// Prepares the compute kernel with the specified program and test configuration
static KernelHandle prepare_print(tt_metal::Program& program, const ConfigRegPrintTestConfig& config) {
    return tt_metal::CreateKernel(
        program,
        config.print_kernel,
        config.core,
        tt_metal::ComputeConfig{});
}

static std::string generate_golden_output(const std::vector<std::string>& field_names, const std::vector<uint32_t>& values, uint num_of_registers) {
    std::string golden_output;
    bool multiple_registers = num_of_registers > 1;
    for (uint reg_id = 1; reg_id <= num_of_registers; reg_id++) {
        if (multiple_registers) golden_output += "REG_ID: " + std::to_string(reg_id) + "\n";
        for (size_t i = 0; i < field_names.size(); i++) {
            if (field_names[i] == "blobs_y_start_lo") continue;
            if (field_names[i] == "blobs_y_start_hi") {
                uint32_t val = (values[i] << 16) | values[i-1];
                golden_output += "blobs_y_start: " + std::to_string(val) + "\n";
                continue;
            }
            if (format_fields.find(field_names[i]) != format_fields.end())
                golden_output += field_names[i] + ": " + data_format_to_string(values[i]) + "\n"; 
            else if (decimal_fields.find(field_names[i]) != format_fields.end())
                golden_output += field_names[i] + ": " + std::to_string(values[i]) + "\n";
            else 
                golden_output += field_names[i] + ": 0x" + int_to_hex(values[i]) + "\n";
        }
        if (reg_id != num_of_registers) golden_output += "\n";
    }
    return golden_output;
}

static void print_config_reg(
    DPrintFixture* fixture, tt_metal::IDevice* device, const ConfigRegPrintTestConfig& config) {
    // Create program
    tt_metal::Program program = tt_metal::CreateProgram();

    // Prepare write kernel
    auto write_kernel = prepare_writer(program, config);

    // Prepare print kernel
    auto print_kernel = prepare_print(program, config);

    // Generate golden output
    std::string golden_output = generate_golden_output(config.field_names, config.field_values, config.num_of_registers);

    // Run the program
    fixture->RunProgram(device, program);
    
    // Check the print log against golden output.
    EXPECT_TRUE(FilesMatchesString(DPrintFixture::dprint_file_name, golden_output));
}

TEST_F(DPrintFixture, ConfigRegAluTestPrint) {
    // Setup test configuration
    ConfigRegPrintTestConfig test_config = {
        .core = CoreCoord(0, 0),
        .write_kernel = "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_config_reg.cpp",
        .print_kernel = "tests/tt_metal/tt_metal/test_kernels/misc/dprint_config_register.cpp",
        .num_of_registers = 1,
        .field_names = field_names_alu_config,
        .field_values = field_values_alu_config,
        .register_name = ALU_CONFIG};
    
    if (this->arch_ == ARCH::GRAYSKULL) {
        GTEST_SKIP() << "Printing ALU CONFIG is not supported on grayskull.";
    }

    // Run the test on the device
    this->RunTestOnDevice(
        [&](DPrintFixture* fixture, IDevice* device) { print_config_reg(fixture, device, test_config); },
        this->devices_[0]);
}

TEST_F(DPrintFixture, ConfigRegTileDescriptorTestPrint) {
    // Setup test configuration
    ConfigRegPrintTestConfig test_config = {
        .core = CoreCoord(0, 0),
        .write_kernel = "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_config_reg.cpp",
        .print_kernel = "tests/tt_metal/tt_metal/test_kernels/misc/dprint_config_register.cpp",
        .num_of_registers = 2,
        .field_names = field_names_unpack_tile_descriptor,
        .field_values = field_values_unpack_tile_descriptor,
        .register_name = UNPACK_TILE_DESCRIPTOR};

    // Run the test on the device
    this->RunTestOnDevice(
        [&](DPrintFixture* fixture, IDevice* device) { print_config_reg(fixture, device, test_config); },
        this->devices_[0]);
}

TEST_F(DPrintFixture, ConfigRegUnpackTestPrint) {
    // Setup test configuration
    ConfigRegPrintTestConfig test_config = {
        .core = CoreCoord(0, 0),
        .write_kernel = "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_config_reg.cpp",
        .print_kernel = "tests/tt_metal/tt_metal/test_kernels/misc/dprint_config_register.cpp",
        .num_of_registers = 2,
        .field_names = field_names_unpack_config,
        .field_values = field_values_unpack_config,
        .register_name = UNPACK_CONFIG};

    // Run the test on the device
    this->RunTestOnDevice(
        [&](DPrintFixture* fixture, IDevice* device) { print_config_reg(fixture, device, test_config); },
        this->devices_[0]);
}