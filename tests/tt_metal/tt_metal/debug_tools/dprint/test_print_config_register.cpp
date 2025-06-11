// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <sys/types.h>
#include <tt-metalium/host_api.hpp>
#include <functional>
#include <map>
#include <sstream>
#include <string>
#include <unordered_set>
#include <variant>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include "debug_tools_fixture.hpp"
#include "debug_tools_test_utils.hpp"
#include "gtest/gtest.h"
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include "tt_metal/test_utils/df/float32.hpp"
#include "umd/device/types/arch.h"

namespace tt {
namespace tt_metal {
class IDevice;
}  // namespace tt_metal
}  // namespace tt

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
const std::unordered_set<std::string> format_fields = {
    "ALU_FORMAT_SPEC_REG0_SrcA",
    "ALU_FORMAT_SPEC_REG1_SrcB",
    "ALU_FORMAT_SPEC_REG2_Dstacc",
    "in_data_format",
    "out_data_format"};
const std::unordered_set<std::string> decimal_fields = {
    "blobs_per_xy_plane",
    "x_dim",
    "y_dim",
    "z_dim",
    "w_dim",
    "blobs_y_start",
    "digest_size",
    "upsample_rate",
    "shift_amount",
    "fifo_size",
    "row_ptr_section_size",
    "exp_section_size",
    "pack_per_xy_plane",
    "downsample_shift_count",
    "exp_threshold",
    "STACC_RELU_ReluThreshold",
    "pack_reads_per_xy_plane",
    "pack_xys_per_til",
    "pack_per_xy_plane_offset",
    "sub_l1_tile_header_size",
    "add_tile_header_size"};

// ALU CONFIG
const std::vector<std::string> field_names_alu_config_all = {
    "ALU_ROUNDING_MODE_Fpu_srnd_en",
    "ALU_ROUNDING_MODE_Gasket_srnd_en",
    "ALU_ROUNDING_MODE_Packer_srnd_en",
    "ALU_ROUNDING_MODE_Padding",
    "ALU_ROUNDING_MODE_GS_LF",
    "ALU_ROUNDING_MODE_Bfp8_HF",
    "ALU_FORMAT_SPEC_REG0_SrcAUnsigned",
    "ALU_FORMAT_SPEC_REG0_SrcBUnsigned",
    "ALU_FORMAT_SPEC_REG0_SrcA",
    "ALU_FORMAT_SPEC_REG1_SrcB",
    "ALU_FORMAT_SPEC_REG2_Dstacc",
    "ALU_ACC_CTRL_Fp32_enabled",
    "ALU_ACC_CTRL_SFPU_Fp32_enabled",
    "ALU_ACC_CTRL_INT8_math_enabled"};
const std::vector<uint32_t> field_values_alu_config_all = {1, 0, 1, 15, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1};

// PACK_EDGE_OFFSET
const std::vector<std::string> field_names_pack_edge_offset_all = {
    "mask",
    "mode",
    "tile_row_set_select_pack0",
    "tile_row_set_select_pack1",
    "tile_row_set_select_pack2",
    "tile_row_set_select_pack3",
    "reserved"};
const std::vector<uint32_t> field_values_pack_edge_offset_all = {16, 1, 0, 1, 2, 3, 0};

// PACK_COUNTERS
const std::vector<std::string> field_names_pack_counters_all = {
    "pack_per_xy_plane",
    "pack_reads_per_xy_plane",
    "pack_xys_per_til",
    "pack_yz_transposed",
    "pack_per_xy_plane_offset"};
const std::vector<uint32_t> field_values_pack_counters_all = {4, 8, 2, 0, 6};

// RELU_CONFIG
const std::vector<std::string> field_names_relu_config_all = {
    "ALU_ACC_CTRL_Zero_Flag_disabled_src",
    "ALU_ACC_CTRL_Zero_Flag_disabled_dst",
    "STACC_RELU_ApplyRelu",
    "STACC_RELU_ReluThreshold",
    "DISABLE_RISC_BP_Disable_main",
    "DISABLE_RISC_BP_Disable_trisc",
    "DISABLE_RISC_BP_Disable_ncrisc",
    "DISABLE_RISC_BP_Disable_bmp_clear_main",
    "DISABLE_RISC_BP_Disable_bmp_clear_trisc",
    "DISABLE_RISC_BP_Disable_bmp_clear_ncrisc"};
const std::vector<uint32_t> field_values_relu_config_all = {0, 0, 1, 8, 0, 0, 0, 0, 0, 0};

// PACK_DEST_RD_CTRL
const std::vector<std::string> field_names_dest_rd_ctrl_all = {
    "PCK_DEST_RD_CTRL_Read_32b_data",
    "PCK_DEST_RD_CTRL_Read_unsigned",
    "PCK_DEST_RD_CTRL_Read_int8",
    "PCK_DEST_RD_CTRL_Round_10b_mant",
    "PCK_DEST_RD_CTRL_Reserved"};
const std::vector<uint32_t> field_values_dest_rd_ctrl_all = {1, 0, 1, 1, 0};

// UNPACK TILE DESCRIPTOR
const std::vector<std::string> field_names_unpack_tile_descriptor_grayskull = {
    "in_data_format",
    "uncompressed",
    "reserved_0",
    "blobs_per_xy_plane",
    "reserved_1",
    "x_dim",
    "y_dim",
    "z_dim",
    "w_dim",
    "blobs_y_start",
    "digest_type",
    "digest_size"};
const std::vector<uint32_t> field_values_unpack_tile_descriptor_grayskull = {5, 1, 0, 10, 7, 2, 4, 8, 16, 32, 0, 0};

// UNPACK CONFIG
const std::vector<std::string> field_names_unpack_config_grayskull = {
    "out_data_format",
    "throttle_mode",
    "context_count",
    "haloize_mode",
    "tileize_mode",
    "force_shared_exp",
    "reserved_0",
    "upsample_rate",
    "upsample_and_interlave",
    "shift_amount",
    "uncompress_cntx0_3",
    "reserved_1",
    "uncompress_cntx4_7",
    "reserved_2",
    "limit_addr",
    "fifo_size"};
const std::vector<uint32_t> field_values_unpack_config_grayskull = {0, 1, 2, 0, 1, 0, 0, 3, 0, 16, 5, 0, 2, 0, 28, 29};

// PACK CONFIG
const std::vector<std::string> field_names_pack_config_grayskull = {
    "row_ptr_section_size",
    "exp_section_size",
    "l1_dest_addr",
    "uncompress",
    "add_l1_dest_addr_offset",
    "reserved_0",
    "out_data_format",
    "in_data_format",
    "reserved_1",
    "src_if_sel",
    "pack_per_xy_plane",
    "l1_src_addr",
    "downsample_mask",
    "downsample_shift_count",
    "read_mode",
    "exp_threshold_en",
    "reserved_2",
    "exp_threshold"};
const std::vector<uint32_t> field_values_pack_config_grayskull = {
    12, 24, 16, 0, 1, 0, 5, 5, 0, 1, 0, 8, 12, 4, 0, 1, 0, 12};

// UNPACK TILE DESCRIPTOR
const std::vector<std::string> field_names_unpack_tile_descriptor_wormhole_or_blackhole = {
    "in_data_format",
    "uncompressed",
    "reserved_0",
    "blobs_per_xy_plane",
    "reserved_1",
    "x_dim",
    "y_dim",
    "z_dim",
    "w_dim",
    "blobs_y_start_lo",
    "blobs_y_start_hi",
    "digest_type",
    "digest_size"};
const std::vector<uint32_t> field_values_unpack_tile_descriptor_wormhole_or_blackhole = {
    5, 1, 0, 10, 7, 2, 4, 8, 16, 32, 0, 0, 0};

// UNPACK CONFIG
const std::vector<std::string> field_names_unpack_config_wormhole_or_blackhole = {
    "out_data_format",
    "throttle_mode",
    "context_count",
    "haloize_mode",
    "tileize_mode",
    "unpack_src_reg_set_update",
    "unpack_if_sel",
    "upsample_rate",
    "reserved_1",
    "upsample_and_interlave",
    "shift_amount",
    "uncompress_cntx0_3",
    "unpack_if_sel_cntx0_3",
    "force_shared_exp",
    "reserved_2",
    "uncompress_cntx4_7",
    "unpack_if_sel_cntx4_7",
    "reserved_3",
    "limit_addr",
    "reserved_4",
    "fifo_size",
    "reserved_5"};
const std::vector<uint32_t> field_values_unpack_config_wormhole_or_blackhole = {0, 1, 2, 0, 1, 1, 0, 3,  0, 0,  16,
                                                                                5, 6, 0, 0, 2, 3, 0, 28, 0, 29, 0};

const std::vector<std::string> field_names_pack_config_blackhole = {
    "row_ptr_section_size",
    "exp_section_size",
    "l1_dest_addr",
    "uncompress",
    "add_l1_dest_addr_offset",
    "disable_pack_zero_flag",
    "reserved_0",
    "out_data_format",
    "in_data_format",
    "dis_shared_exp_assembler",
    "auto_set_last_pacr_intf_sel",
    "enable_out_fifo",
    "sub_l1_tile_header_size",
    "src_if_sel",
    "pack_start_intf_pos",
    "all_pack_disable_zero_compress_ovrd",
    "add_tile_header_size",
    "pack_dis_y_pos_start_offset",
    "l1_src_addr"};
const std::vector<uint32_t> field_values_pack_config_blackhole = {
    12, 24, 16, 0, 1, 1, 0, 5, 5, 0, 0, 1, 0, 1, 2, 0, 1, 0, 8};
// PACK CONFIG
const std::vector<std::string> field_names_pack_config_wormhole = {
    "row_ptr_section_size",
    "exp_section_size",
    "l1_dest_addr",
    "uncompress",
    "add_l1_dest_addr_offset",
    "reserved_0",
    "out_data_format",
    "in_data_format",
    "reserved_1",
    "src_if_sel",
    "pack_per_xy_plane",
    "l1_src_addr",
    "downsample_mask",
    "downsample_shift_count",
    "read_mode",
    "exp_threshold_en",
    "pack_l1_acc_disable_pack_zero_flag",
    "reserved_2",
    "exp_threshold"};
const std::vector<uint32_t> field_values_pack_config_wormhole = {
    12, 24, 16, 0, 1, 0, 5, 5, 0, 1, 0, 8, 12, 4, 0, 1, 2, 0, 12};

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
        case (uint8_t)DataFormat::Float32: return "Float32";
        case (uint8_t)DataFormat::Float16: return "Float16";
        case (uint8_t)DataFormat::Bfp8: return "Bfp8";
        case (uint8_t)DataFormat::Bfp4: return "Bfp4";
        case (uint8_t)DataFormat::Bfp2: return "Bfp2";
        case (uint8_t)DataFormat::Float16_b: return "Float16_b";
        case (uint8_t)DataFormat::Bfp8_b: return "Bfp8_b";
        case (uint8_t)DataFormat::Bfp4_b: return "Bfp4_b";
        case (uint8_t)DataFormat::Bfp2_b: return "Bfp2_b";
        case (uint8_t)DataFormat::Lf8: return "Lf8";
        case (uint8_t)DataFormat::Int8: return "Int8";
        case (uint8_t)DataFormat::UInt8: return "UInt8";
        case (uint8_t)DataFormat::UInt16: return "UInt16";
        case (uint8_t)DataFormat::Int32: return "Int32";
        case (uint8_t)DataFormat::UInt32: return "UInt32";
        case (uint8_t)DataFormat::Tf32: return "Tf32";
        default: return "INVALID DATA FORMAT";
    }
}

static std::string int_to_hex(int value) {
    std::stringstream ss;
    ss << std::hex << value;  // Convert to hexadecimal
    return ss.str();
}

// Prepares the compute kernel with the specified program and test configuration
static KernelHandle prepare_writer(tt_metal::Program& program, const ConfigRegPrintTestConfig& config) {
    return tt_metal::CreateKernel(
        program, config.write_kernel, config.core, tt_metal::ComputeConfig{.compile_args = {config.register_name}});
}

static std::string generate_golden_output(
    const std::vector<std::string>& field_names,
    const std::vector<uint32_t>& values,
    uint num_of_registers,
    uint32_t register_name) {
    std::string golden_output;
    bool multiple_registers = num_of_registers > 1;
    for (uint reg_id = 1; reg_id <= num_of_registers; reg_id++) {
        if (multiple_registers) {
            golden_output += "REG_ID: " + std::to_string(reg_id) + "\n";
        }
        for (size_t i = 0; i < field_names.size(); i++) {
            if (field_names[i] == "blobs_y_start_lo") {
                continue;
            }
            if (field_names[i] == "blobs_y_start_hi") {
                uint32_t val = (values[i] << 16) | values[i - 1];
                golden_output += "blobs_y_start: " + std::to_string(val) + "\n";
                continue;
            }
            if (format_fields.find(field_names[i]) != format_fields.end()) {
                golden_output += field_names[i] + ": " + data_format_to_string(values[i]) + "\n";
            } else if (decimal_fields.find(field_names[i]) != format_fields.end()) {
                golden_output += field_names[i] + ": " + std::to_string(values[i]) + "\n";
            } else {
                golden_output += field_names[i] + ": 0x" + int_to_hex(values[i]) + "\n";
            }

            if (register_name == PACK_EDGE_OFFSET && reg_id > 1) {
                break;
            }
        }
        if (reg_id != num_of_registers) {
            golden_output += "\n";
        }
    }
    return golden_output;
}

static void print_config_reg(
    DPrintFixture* fixture, tt_metal::IDevice* device, const ConfigRegPrintTestConfig& config) {
    // Create program
    tt_metal::Program program = tt_metal::CreateProgram();

    // Prepare write kernel
    auto write_kernel = prepare_writer(program, config);

    // Generate golden output
    std::string golden_output =
        generate_golden_output(config.field_names, config.field_values, config.num_of_registers, config.register_name);

    // Run the program
    fixture->RunProgram(device, program);

    // Check the print log against golden output.
    EXPECT_TRUE(FilesMatchesString(DPrintFixture::dprint_file_name, golden_output));
}

TEST_F(DPrintFixture, ConfigRegAluTestPrint) {
    std::vector<std::string> field_names_alu_config = field_names_alu_config_all;
    std::vector<uint32_t> field_values_alu_config = field_values_alu_config_all;

    // Setup test configuration
    ConfigRegPrintTestConfig test_config = {
        .core = CoreCoord(0, 0),
        .write_kernel = "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_config_reg.cpp",
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

    std::vector<std::string> field_names_unpack_tile_descriptor;
    std::vector<uint32_t> field_values_unpack_tile_descriptor;

    if (this->arch_ == ARCH::GRAYSKULL) {
        field_names_unpack_tile_descriptor = field_names_unpack_tile_descriptor_grayskull;
        field_values_unpack_tile_descriptor = field_values_unpack_tile_descriptor_grayskull;
    } else {
        field_names_unpack_tile_descriptor = field_names_unpack_tile_descriptor_wormhole_or_blackhole;
        field_values_unpack_tile_descriptor = field_values_unpack_tile_descriptor_wormhole_or_blackhole;
    }

    ConfigRegPrintTestConfig test_config = {
        .core = CoreCoord(0, 0),
        .write_kernel = "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_config_reg.cpp",
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
    std::vector<std::string> field_names_unpack_config;
    std::vector<uint32_t> field_values_unpack_config;

    if (this->arch_ == ARCH::GRAYSKULL) {
        field_names_unpack_config = field_names_unpack_config_grayskull;
        field_values_unpack_config = field_values_unpack_config_grayskull;
    } else {
        field_names_unpack_config = field_names_unpack_config_wormhole_or_blackhole;
        field_values_unpack_config = field_values_unpack_config_wormhole_or_blackhole;
    }

    // Setup test configuration
    ConfigRegPrintTestConfig test_config = {
        .core = CoreCoord(0, 0),
        .write_kernel = "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_config_reg.cpp",
        .num_of_registers = 2,
        .field_names = field_names_unpack_config,
        .field_values = field_values_unpack_config,
        .register_name = UNPACK_CONFIG};

    // Run the test on the device
    this->RunTestOnDevice(
        [&](DPrintFixture* fixture, IDevice* device) { print_config_reg(fixture, device, test_config); },
        this->devices_[0]);
}

TEST_F(DPrintFixture, ConfigRegPackTestPrint) {
    std::vector<std::string> field_names_pack_config;
    std::vector<uint32_t> field_values_pack_config;

    if (this->arch_ == ARCH::GRAYSKULL) {
        field_names_pack_config = field_names_pack_config_grayskull;
        field_values_pack_config = field_values_pack_config_grayskull;
    } else if (this->arch_ == ARCH::WORMHOLE_B0) {
        field_names_pack_config = field_names_pack_config_wormhole;
        field_values_pack_config = field_values_pack_config_wormhole;
    } else {
        field_names_pack_config = field_names_pack_config_blackhole;
        field_values_pack_config = field_values_pack_config_blackhole;
    }

    int num_of_registers;
    if (this->arch_ == ARCH::BLACKHOLE) {
        num_of_registers = 1;
    } else {
        num_of_registers = 4;
    }

    // Setup test configuration
    ConfigRegPrintTestConfig test_config = {
        .core = CoreCoord(0, 0),
        .write_kernel = "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_config_reg.cpp",
        .num_of_registers = num_of_registers,
        .field_names = field_names_pack_config,
        .field_values = field_values_pack_config,
        .register_name = PACK_CONFIG};

    // Run the test on the device
    this->RunTestOnDevice(
        [&](DPrintFixture* fixture, IDevice* device) { print_config_reg(fixture, device, test_config); },
        this->devices_[0]);
}

TEST_F(DPrintFixture, ConfigRegReluTestPrint) {
    std::vector<std::string> field_names_relu_config = field_names_relu_config_all;
    std::vector<uint32_t> field_values_relu_config = field_values_relu_config_all;

    // Setup test configuration
    ConfigRegPrintTestConfig test_config = {
        .core = CoreCoord(0, 0),
        .write_kernel = "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_config_reg.cpp",
        .num_of_registers = 1,
        .field_names = field_names_relu_config,
        .field_values = field_values_relu_config,
        .register_name = RELU_CONFIG};

    if (this->arch_ == ARCH::GRAYSKULL) {
        GTEST_SKIP() << "Printing RELU CONFIG is not supported on grayskull.";
    }

    // Run the test on the device
    this->RunTestOnDevice(
        [&](DPrintFixture* fixture, IDevice* device) { print_config_reg(fixture, device, test_config); },
        this->devices_[0]);
}

TEST_F(DPrintFixture, ConfigRegDestRdCtrlTestPrint) {
    std::vector<std::string> field_names_dest_rd_ctrl = field_names_dest_rd_ctrl_all;
    std::vector<uint32_t> field_values_dest_rd_ctrl = field_values_dest_rd_ctrl_all;

    // Setup test configuration
    ConfigRegPrintTestConfig test_config = {
        .core = CoreCoord(0, 0),
        .write_kernel = "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_config_reg.cpp",
        .num_of_registers = 1,
        .field_names = field_names_dest_rd_ctrl,
        .field_values = field_values_dest_rd_ctrl,
        .register_name = DEST_RD_CTRL};

    if (this->arch_ == ARCH::GRAYSKULL) {
        GTEST_SKIP() << "Printing DEST RD CTRL is not supported on grayskull.";
    }

    // Run the test on the device
    this->RunTestOnDevice(
        [&](DPrintFixture* fixture, IDevice* device) { print_config_reg(fixture, device, test_config); },
        this->devices_[0]);
}

TEST_F(DPrintFixture, ConfigRegPackEdgeOffsetTestPrint) {
    std::vector<std::string> field_names_pack_edge_offset = field_names_pack_edge_offset_all;
    std::vector<uint32_t> field_values_pack_edge_offset = field_values_pack_edge_offset_all;

    int num_of_registers;
    if (this->arch_ == ARCH::BLACKHOLE) {
        num_of_registers = 1;
    } else {
        num_of_registers = 4;
    }

    // Setup test configuration
    ConfigRegPrintTestConfig test_config = {
        .core = CoreCoord(0, 0),
        .write_kernel = "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_config_reg.cpp",
        .num_of_registers = num_of_registers,
        .field_names = field_names_pack_edge_offset,
        .field_values = field_values_pack_edge_offset,
        .register_name = PACK_EDGE_OFFSET};

    // Run the test on the device
    this->RunTestOnDevice(
        [&](DPrintFixture* fixture, IDevice* device) { print_config_reg(fixture, device, test_config); },
        this->devices_[0]);
}

TEST_F(DPrintFixture, ConfigRegPackCountersTestPrint) {
    std::vector<std::string> field_names_pack_counters = field_names_pack_counters_all;
    std::vector<uint32_t> field_values_pack_counters = field_values_pack_counters_all;

    int num_of_registers;
    if (this->arch_ == ARCH::BLACKHOLE) {
        num_of_registers = 1;
    } else {
        num_of_registers = 4;
    }

    // Setup test configuration
    ConfigRegPrintTestConfig test_config = {
        .core = CoreCoord(0, 0),
        .write_kernel = "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_config_reg.cpp",
        .num_of_registers = num_of_registers,
        .field_names = field_names_pack_counters,
        .field_values = field_values_pack_counters,
        .register_name = PACK_COUNTERS};

    // Run the test on the device
    this->RunTestOnDevice(
        [&](DPrintFixture* fixture, IDevice* device) { print_config_reg(fixture, device, test_config); },
        this->devices_[0]);
}
