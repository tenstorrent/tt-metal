// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include "tests/tt_metal/tt_metal/unit_tests_common/common/common_fixture.hpp"
#include "gtest/gtest.h"
#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/test_utils/env_vars.hpp"
#include "tt_metal/impl/dispatch/command_queue.hpp"
#include "tt_metal/test_utils/deprecated/tensor.hpp"
#include "tt_metal/common/bfloat16.hpp"
#include "tt_metal/common/logger.hpp"


using namespace tt;

namespace unit_tests_common::dram::test_dram{
struct DRAMConfig{
    //CoreRange, Kernel, dram_buffer_size
    CoreRange core_range;
    std::string kernel_file;
    std::uint32_t dram_buffer_size;
    std::uint32_t l1_buffer_addr;
    tt_metal::DataMovementConfig data_movement_cfg;
};

bool dram_single_core_db (CommonFixture* fixture, tt_metal::Device *device){
    tt_metal::Program program = tt_metal::CreateProgram();

    CoreCoord core = {0, 0};

    uint32_t single_tile_size = 2 * 1024;
    uint32_t num_tiles = 256;
    uint32_t dram_buffer_size_bytes = single_tile_size * num_tiles;

    // L1 buffer is double buffered
    // We read and write total_l1_buffer_size_tiles / 2 tiles from and to DRAM
    uint32_t l1_buffer_addr = 400 * 1024;
    uint32_t total_l1_buffer_size_tiles = num_tiles / 2;
    TT_FATAL(total_l1_buffer_size_tiles % 2 == 0, "Error");
    uint32_t total_l1_buffer_size_bytes = total_l1_buffer_size_tiles * single_tile_size;

    tt_metal::InterleavedBufferConfig dram_config{
                            .device=device,
                            .size = dram_buffer_size_bytes,
                            .page_size = dram_buffer_size_bytes,
                            .buffer_type = tt_metal::BufferType::DRAM
                            };

    auto input_dram_buffer = CreateBuffer(dram_config);
    uint32_t input_dram_buffer_addr = input_dram_buffer->address();

    auto output_dram_buffer = CreateBuffer(dram_config);
    uint32_t output_dram_buffer_addr = output_dram_buffer->address();

    auto input_dram_noc_xy = input_dram_buffer->noc_coordinates();
    auto output_dram_noc_xy = output_dram_buffer->noc_coordinates();

    auto dram_copy_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/dram_copy_db.cpp",
        core,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

    std::vector<uint32_t> input_vec = create_random_vector_of_bfloat16(
        dram_buffer_size_bytes, 100, std::chrono::system_clock::now().time_since_epoch().count());
    fixture->WriteBuffer(device, input_dram_buffer, input_vec);

    tt_metal::SetRuntimeArgs(
        program,
        dram_copy_kernel,
        core,
        {input_dram_buffer_addr,
        (std::uint32_t)input_dram_noc_xy.x,
        (std::uint32_t)input_dram_noc_xy.y,
        output_dram_buffer_addr,
        (std::uint32_t)output_dram_noc_xy.x,
        (std::uint32_t)output_dram_noc_xy.y,
        dram_buffer_size_bytes,
        num_tiles,
        l1_buffer_addr,
        total_l1_buffer_size_tiles,
        total_l1_buffer_size_bytes});

    fixture->RunProgram(device, program);

    std::vector<uint32_t> result_vec;
    fixture->ReadBuffer(device, output_dram_buffer, result_vec);

    return input_vec == result_vec;
}

bool dram_single_core (CommonFixture* fixture, tt_metal::Device *device, const DRAMConfig &cfg, std::vector<uint32_t> src_vec){
    // Create a program
    tt_metal::Program program = CreateProgram();

    tt_metal::InterleavedBufferConfig dram_config{
                            .device = device,
                            .size = cfg.dram_buffer_size,
                            .page_size = cfg.dram_buffer_size,
                            .buffer_type = tt_metal::BufferType::DRAM
                            };
    auto input_dram_buffer = tt_metal::CreateBuffer(dram_config);
    uint32_t input_dram_buffer_addr = input_dram_buffer->address();

    auto output_dram_buffer = tt_metal::CreateBuffer(dram_config);
    uint32_t output_dram_buffer_addr = output_dram_buffer->address();

    auto input_dram_noc_xy = input_dram_buffer->noc_coordinates();
    auto output_dram_noc_xy = output_dram_buffer->noc_coordinates();
    log_debug(tt::LogVerif, "Creating kernel");
    // Create the kernel
    auto dram_kernel = tt_metal::CreateKernel(
        program,
        cfg.kernel_file,
        cfg.core_range,
        cfg.data_movement_cfg
    );
    fixture->WriteBuffer(device, input_dram_buffer, src_vec);

    tt_metal::SetRuntimeArgs(
            program,
            dram_kernel,
            cfg.core_range,
            {cfg.l1_buffer_addr,
            input_dram_buffer_addr,
            (std::uint32_t)input_dram_noc_xy.x,
            (std::uint32_t)input_dram_noc_xy.y,
            output_dram_buffer_addr,
            (std::uint32_t)output_dram_noc_xy.x,
            (std::uint32_t)output_dram_noc_xy.y,
            cfg.dram_buffer_size});

    fixture->RunProgram(device, program);

    std::vector<uint32_t> result_vec;
    fixture->ReadBuffer(device, output_dram_buffer, result_vec);
    return result_vec == src_vec;
}
}

TEST_F(CommonFixture, DRAMLoopbackSingleCore){
    uint32_t buffer_size = 2 * 1024 * 25;
    std::vector<uint32_t> src_vec = create_random_vector_of_bfloat16(
        buffer_size, 100, std::chrono::system_clock::now().time_since_epoch().count());
    unit_tests_common::dram::test_dram::DRAMConfig dram_test_config = {
        .core_range = {{0, 0}, {0, 0}},
        .kernel_file = "tests/tt_metal/tt_metal/test_kernels/dataflow/dram_copy.cpp",
        .dram_buffer_size = buffer_size,
        .l1_buffer_addr = 400 * 1024,
        .data_movement_cfg = {.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default},
    };
    for (unsigned int id=0; id < devices_.size(); id++){
        ASSERT_TRUE(unit_tests_common::dram::test_dram::dram_single_core(this, devices_.at(id), dram_test_config, src_vec));
    }
}

TEST_F(CommonFixture, DRAMLoopbackSingleCoreDB){
    if (!getenv("TT_METAL_SLOW_DISPATCH_MODE")){
        tt::log_info(tt::LogTest, "This test is only supported in slow dispatch mode");
        GTEST_SKIP();
    }
    for (unsigned int id=0; id < devices_.size(); id++){
        ASSERT_TRUE(unit_tests_common::dram::test_dram::dram_single_core_db(this, devices_.at(id)));
    }
}
