// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include "tests/tt_metal/tt_metal/unit_tests_common/common/common_fixture.hpp"
#include "tests/tt_metal/tt_metal/unit_tests_fast_dispatch/common/command_queue_fixture.hpp"
#include "gtest/gtest.h"
#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/test_utils/env_vars.hpp"
#include "tt_metal/impl/dispatch/command_queue.hpp"
#include "common/bfloat16.hpp"
#include "common/logger.hpp"


using namespace tt;

namespace unit_tests_common::dram::test_dram{
struct DRAMConfig{
    //CoreRange, Kernel, dram_buffer_size
    CoreRange core_range;
    std::string kernel_file;
    uint32_t dram_buffer_size;
    uint32_t l1_buffer_addr;
    tt_metal::DataMovementConfig data_movement_cfg;
};

bool dram_test_single_core (CommonFixture* fixture, tt_metal::Device *device, const DRAMConfig &cfg, std::vector<uint32_t> src_vec){
    // Create a program
    tt_metal::Program program = CreateProgram();

    tt_metal::InterleavedBufferConfig dram_config{
                            .device = device,
                            .size = cfg.dram_buffer_size,
                            .page_size = cfg.dram_buffer_size,
                            .buffer_type = tt_metal::BufferType::DRAM
                            };
    auto input_dram_buffer = tt_metal::CreateBuffer(dram_config);
    uint32_t input_dram_buffer_addr = input_dram_buffer.address();

    auto output_dram_buffer = tt_metal::CreateBuffer(dram_config);
    uint32_t output_dram_buffer_addr = output_dram_buffer.address();

    auto input_dram_noc_xy = input_dram_buffer.noc_coordinates();
    auto output_dram_noc_xy = output_dram_buffer.noc_coordinates();
    log_info(tt::LogVerif, "Creating kernel");
    // Create the kernel
    auto dram_kernel = tt_metal::CreateKernel(
        program,
        cfg.kernel_file,
        cfg.core_range,
        cfg.data_movement_cfg
    );
    // log_info(tt::LogVerif, "Writing to buffer");
    // EnqueueWriteBuffer(cq, input_dram_buffer, cfg.src_vec, false);
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
    // log_info(tt::LogVerif, "Lauching program");
    fixture->RunProgram(device, program);
    // EnqueueProgram(cq, program, false);
    // Finish(cq);

    std::vector<uint32_t> result_vec;
    // log_info(tt::LogVerif, "Reading buffer");
    // EnqueueReadBuffer(cq, output_dram_buffer, result_vec, true);
    fixture->ReadBuffer(device, output_dram_buffer, result_vec);
    return result_vec == src_vec;
}
}

TEST_F(CommonFixture, DRAMLoopBackSingleCore){
    uint32_t buffer_size = 2 * 1024 * 50;
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
        ASSERT_TRUE(unit_tests_common::dram::test_dram::dram_test_single_core(this, devices_.at(id), dram_test_config, src_vec));
    }
}

// TEST_F(CommonFixture, DRAMLoopBackSingleCoreDB){
//     uint32_t buffer_size = 2 * 1024 * 256;
//     std::vector<uint32_t> src_vec = create_random_vector_of_bfloat16(
//         buffer_size, 100, std::chrono::system_clock::now().time_since_epoch().count());
//     unit_tests_common::dram::test_dram::DRAMConfig dram_test_config = {
//         .core_range = {{0, 0}, {0, 0}},
//         .kernel_file = "tests/tt_metal/tt_metal/test_kernels/dataflow/dram_copy_db.cpp",
//         .dram_buffer_size = buffer_size,
//         .l1_buffer_addr = 400 * 1024,
//         .data_movement_cfg = {.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default},
//     };
//     for (unsigned int id=0; id < devices_.size(); id++){
//         ASSERT_TRUE(unit_tests_common::dram::test_dram::dram_test_single_core(this, devices_.at(id), dram_test_config, src_vec));
//     }
// }
