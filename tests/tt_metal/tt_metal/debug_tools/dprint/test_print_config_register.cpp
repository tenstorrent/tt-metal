// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <map>
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

// Type of prints
const std::unordered_set<std::string> format_fields = {"ALU_FORMAT_SPEC_REG0_SrcA", "ALU_FORMAT_SPEC_REG1_SrcB", "ALU_FORMAT_SPEC_REG2_Dstacc"};
const std::unordered_set<std::string> decimal_fields = {};

// ALU CONFIG
const std::vector<std::string> field_names_alu_config = {"ALU_ROUNDING_MODE_Fpu_srnd_en", "ALU_ROUNDING_MODE_Gasket_srnd_en", "ALU_ROUNDING_MODE_Packer_srnd_en",
    "ALU_ROUNDING_MODE_Padding", "ALU_ROUNDING_MODE_GS_LF", "ALU_ROUNDING_MODE_Bfp8_HF", "ALU_FORMAT_SPEC_REG0_SrcAUnsigned",
    "ALU_FORMAT_SPEC_REG0_SrcBUnsigned", "ALU_FORMAT_SPEC_REG0_SrcA", "ALU_FORMAT_SPEC_REG1_SrcB", "ALU_FORMAT_SPEC_REG2_Dstacc",
    "ALU_ACC_CTRL_Fp32_enabled", "ALU_ACC_CTRL_SFPU_Fp32_enabled", "ALU_ACC_CTRL_INT8_math_enabled"};
const std::vector<int> field_values_alu_config = {0,0,0,0,0,0,0,0,5,5,5,0,0,0};

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

// Configuration for Data Flow Test involving Reader, Datacopy, and Writer
struct ConfigRegPrintTestConfig {
    CoreCoord core = {};
//    std::string write_kernel;
    std::string print_kernel;
    std::vector<std::string> field_names;
    std::vector<int> field_values;
};

// Prepares the compute kernel with the specified program and test configuration
static KernelHandle prepare_print(tt_metal::Program& program, const ConfigRegPrintTestConfig& config) {
    return tt_metal::CreateKernel(
        program,
        config.print_kernel,
        config.core,
        tt_metal::ComputeConfig{});
}

static std::string generate_golden_output(const std::vector<std::string>& field_names, const std::vector<int>& values) {
    std::string golden_output;
    for (size_t i = 0; i < field_names.size(); i++) {
        if (format_fields.find(field_names[i]) != format_fields.end())
            golden_output += field_names[i] + ": " + data_format_to_string(values[i]) + "\n"; 
        else if (decimal_fields.find(field_names[i]) != format_fields.end())
            golden_output += field_names[i] + ": " + std::to_string(values[i]) + "\n";
        else 
            golden_output += field_names[i] + ": 0x" + int_to_hex(values[i]) + "\n";
    }

    return golden_output;
}

// Performs DRAM --> Reader --> CB --> Datacopy --> CB --> Writer --> DRAM on a single core
static void print_config_reg(
    DPrintFixture* fixture, tt_metal::IDevice* device, const ConfigRegPrintTestConfig& config) {
    // Create program
    tt_metal::Program program = tt_metal::CreateProgram();

    std::string golden_output = generate_golden_output(config.field_names, config.field_values);

    // Prepare compute kernel
    auto print_kernel = prepare_print(program, config);

    // Run the program
    fixture->RunProgram(device, program);
    
    // Check the print log against golden output.
    EXPECT_TRUE(FilesMatchesString(DPrintFixture::dprint_file_name, golden_output));
}

TEST_F(DPrintFixture, ConfigRegTestPrint) {
    // Setup test configuration
    ConfigRegPrintTestConfig test_config = {
        .core = CoreCoord(0, 0),
        .print_kernel = "tests/tt_metal/tt_metal/test_kernels/misc/dprint_config_register.cpp",
        .field_names = field_names_alu_config,
        .field_values = field_values_alu_config};

    // Run the test on the device
    this->RunTestOnDevice(
        [&](DPrintFixture* fixture, IDevice* device) { print_config_reg(fixture, device, test_config); },
        this->devices_[0]);
}