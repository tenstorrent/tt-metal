// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <memory>
#include "command_queue_fixture.hpp"
#include "command_queue_test_utils.hpp"
#include "gtest/gtest.h"
#include "impl/buffers/buffer.hpp"
#include "impl/device/device.hpp"
#include "tt_metal/common/bfloat16.hpp"
#include "tt_metal/common/scoped_timer.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/tt_metal.hpp"

using namespace tt::tt_metal;

struct CBConfig {
    uint32_t cb_id;
    uint32_t num_pages;
    uint32_t page_size;
    tt::DataFormat data_format;
};

struct DummyProgramConfig {
    CoreRangeSet cr_set;
    CBConfig cb_config;
    uint32_t num_cbs;
    uint32_t num_sems;
};

struct DummyProgramMultiCBConfig {
    CoreRangeSet cr_set;
    std::vector<CBConfig> cb_config_vector;
    uint32_t num_sems;
};


namespace local_test_functions {

// Create randomly sized pair of unique and common runtime args vectors, with careful not to exceed max between the two.
// Optionally force the max size for one of the vectors.
std::pair<std::vector<uint32_t>, std::vector<uint32_t>> create_runtime_args(bool force_max_size = false, uint32_t unique_base = 0, uint32_t common_base = 100){

    constexpr uint32_t MAX_RUNTIME_ARGS = 255;

    // Generate Unique Runtime Args. Common RT args starting address must be L1 Aligned, so account for that here via padding
    uint32_t num_rt_args_unique = num_rt_args_unique = rand() % (MAX_RUNTIME_ARGS + 1);
    uint32_t num_rt_args_unique_padded = align(num_rt_args_unique, L1_ALIGNMENT / sizeof(uint32_t));
    uint32_t num_rt_args_common = num_rt_args_unique_padded < MAX_RUNTIME_ARGS ? rand() % (MAX_RUNTIME_ARGS - num_rt_args_unique_padded + 1) : 0;

    if (force_max_size) {
        if (rand() % 2) {
            num_rt_args_unique = MAX_RUNTIME_ARGS;
            num_rt_args_common = 0;
        } else {
            num_rt_args_common = MAX_RUNTIME_ARGS;
            num_rt_args_unique = 0;
        }
    }

    vector<uint32_t> rt_args_common;
    for (uint32_t i = 0; i < num_rt_args_common; i++) {
        rt_args_common.push_back(common_base + i);
    }

    vector<uint32_t> rt_args_unique;
    for (uint32_t i = 0; i < num_rt_args_unique; i++) {
        rt_args_unique.push_back(unique_base + i);
    }

    log_trace(tt::LogTest, "{} - num_rt_args_unique: {} num_rt_args_common: {} force_max_size: {}", __FUNCTION__, num_rt_args_unique, num_rt_args_common, force_max_size);
    return std::make_pair(rt_args_unique, rt_args_common);
}


}  // namespace local_test_functions

namespace stress_tests {

TEST_F(MultiCommandQueueSingleDeviceFixture, TestRandomizedProgram) {
    uint32_t NUM_PROGRAMS = 100;
    uint32_t MAX_LOOP = 100;
    uint32_t page_size = 1024;

    if (this->arch_ == tt::ARCH::BLACKHOLE) {
        GTEST_SKIP(); // Running on second CQ is hanging on CI
    }

    // Make random
    auto random_seed = 0; // (unsigned int)time(NULL);
    uint32_t seed = tt::parse_env("SEED", random_seed);
    log_info(tt::LogTest, "Using Test Seed: {}", seed);
    srand(seed);

    CoreCoord worker_grid_size = this->device_->compute_with_storage_grid_size();
    CoreRange cr({0, 0}, {worker_grid_size.x - 1, worker_grid_size.y - 1});
    CoreRangeSet cr_set({cr});

    log_info(tt::LogTest, "Starting compile of {} programs now.", NUM_PROGRAMS);

    vector<Program> programs;
    for (uint32_t i = 0; i < NUM_PROGRAMS; i++) {
        programs.push_back(Program());
        Program& program = programs.back();

        std::map<string, string> data_movement_defines = {{"DATA_MOVEMENT", "1"}};
        std::map<string, string> compute_defines = {{"COMPUTE", "1"}};

        // brisc
        uint32_t BRISC_OUTER_LOOP, BRISC_MIDDLE_LOOP, BRISC_INNER_LOOP, NUM_CBS, NUM_SEMS;
        bool USE_MAX_RT_ARGS;

        if (i == 0) {
            // Ensures that we get at least one compilation with the max amount to
            // ensure it compiles and runs
            BRISC_OUTER_LOOP = MAX_LOOP;
            BRISC_MIDDLE_LOOP = MAX_LOOP;
            BRISC_INNER_LOOP = MAX_LOOP;
            NUM_CBS = NUM_CIRCULAR_BUFFERS;
            NUM_SEMS = NUM_SEMAPHORES;
            USE_MAX_RT_ARGS = true;
        } else {
            BRISC_OUTER_LOOP = rand() % (MAX_LOOP) + 1;
            BRISC_MIDDLE_LOOP = rand() % (MAX_LOOP) + 1;
            BRISC_INNER_LOOP = rand() % (MAX_LOOP) + 1;
            NUM_CBS = rand() % (NUM_CIRCULAR_BUFFERS) + 1;
            NUM_SEMS = rand() % (NUM_SEMAPHORES) + 1;
            USE_MAX_RT_ARGS = false;
        }

        log_debug(tt::LogTest, "Compiling program {}/{} w/ BRISC_OUTER_LOOP: {} BRISC_MIDDLE_LOOP: {} BRISC_INNER_LOOP: {} NUM_CBS: {} NUM_SEMS: {} USE_MAX_RT_ARGS: {}",
                 i+1, NUM_PROGRAMS, BRISC_OUTER_LOOP, BRISC_MIDDLE_LOOP, BRISC_INNER_LOOP, NUM_CBS, NUM_SEMS, USE_MAX_RT_ARGS);

        for (uint32_t j = 0; j < NUM_CBS; j++) {
            CircularBufferConfig cb_config = CircularBufferConfig(page_size * (j + 1), {{j, tt::DataFormat::Float16_b}}).set_page_size(j, page_size * (j + 1));
            auto cb = CreateCircularBuffer(program, cr_set, cb_config);
        }

        for (uint32_t j = 0; j < NUM_SEMS; j++) {
            CreateSemaphore(program, cr_set, j + 1);
        }

        auto [brisc_unique_rtargs, brisc_common_rtargs] = local_test_functions::create_runtime_args(USE_MAX_RT_ARGS);
        uint32_t num_brisc_unique_rtargs = brisc_unique_rtargs.size();
        uint32_t num_brisc_common_rtargs = brisc_common_rtargs.size();
        vector<uint32_t> brisc_compile_args = {BRISC_OUTER_LOOP, BRISC_MIDDLE_LOOP, BRISC_INNER_LOOP, NUM_CBS, NUM_SEMS, num_brisc_unique_rtargs, num_brisc_common_rtargs, page_size};

        // ncrisc
        uint32_t NCRISC_OUTER_LOOP, NCRISC_MIDDLE_LOOP, NCRISC_INNER_LOOP;
        if (i == 0) {
            NCRISC_OUTER_LOOP = MAX_LOOP;
            NCRISC_MIDDLE_LOOP = MAX_LOOP;
            NCRISC_INNER_LOOP = MAX_LOOP;
        } else {
            NCRISC_OUTER_LOOP = rand() % (MAX_LOOP) + 1;
            NCRISC_MIDDLE_LOOP = rand() % (MAX_LOOP) + 1;
            NCRISC_INNER_LOOP = rand() % (MAX_LOOP) + 1;
        }

        auto [ncrisc_unique_rtargs, ncrisc_common_rtargs] = local_test_functions::create_runtime_args(USE_MAX_RT_ARGS);
        uint32_t num_ncrisc_unique_rtargs = ncrisc_unique_rtargs.size();
        uint32_t num_ncrisc_common_rtargs = ncrisc_common_rtargs.size();
        vector<uint32_t> ncrisc_compile_args = {NCRISC_OUTER_LOOP, NCRISC_MIDDLE_LOOP, NCRISC_INNER_LOOP, NUM_CBS, NUM_SEMS, num_ncrisc_unique_rtargs, num_ncrisc_common_rtargs, page_size};

        // trisc
        uint32_t TRISC_OUTER_LOOP, TRISC_MIDDLE_LOOP, TRISC_INNER_LOOP;
        if (i == 0) {
            TRISC_OUTER_LOOP = MAX_LOOP;
            TRISC_MIDDLE_LOOP = MAX_LOOP;
            TRISC_INNER_LOOP = MAX_LOOP;
        } else {
            TRISC_OUTER_LOOP = rand() % (MAX_LOOP) + 1;
            TRISC_MIDDLE_LOOP = rand() % (MAX_LOOP) + 1;
            TRISC_INNER_LOOP = rand() % (MAX_LOOP) + 1;
        }

        auto [trisc_unique_rtargs, trisc_common_rtargs] = local_test_functions::create_runtime_args(USE_MAX_RT_ARGS);
        uint32_t num_trisc_unique_rtargs = trisc_unique_rtargs.size();
        uint32_t num_trisc_common_rtargs = trisc_common_rtargs.size();
        vector<uint32_t> trisc_compile_args = {TRISC_OUTER_LOOP, TRISC_MIDDLE_LOOP, TRISC_INNER_LOOP, NUM_CBS, NUM_SEMS, num_trisc_unique_rtargs, num_trisc_common_rtargs, page_size};

        bool at_least_one_kernel = false;
        if (i == 0 or ((rand() % 2) == 0)) {
            auto dummy_brisc_kernel = CreateKernel(
                program, "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/command_queue/random_program.cpp", cr_set, DataMovementConfig{
                    .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = brisc_compile_args, .defines = data_movement_defines});
            SetRuntimeArgs(program, dummy_brisc_kernel, cr_set, brisc_unique_rtargs);
            SetCommonRuntimeArgs(program, dummy_brisc_kernel, brisc_common_rtargs);
            at_least_one_kernel = true;
        }

        if (i == 0 or ((rand() % 2) == 0)) {
            auto dummy_ncrisc_kernel = CreateKernel(
                program, "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/command_queue/random_program.cpp", cr_set, DataMovementConfig{
                    .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = ncrisc_compile_args, .defines = data_movement_defines});
            SetRuntimeArgs(program, dummy_ncrisc_kernel, cr_set, ncrisc_unique_rtargs);
            SetCommonRuntimeArgs(program, dummy_ncrisc_kernel, ncrisc_common_rtargs);
            at_least_one_kernel = true;
        }

        if (i == 0 or ((rand() % 2) == 0)) {
            auto dummy_trisc_kernel = CreateKernel(
                program, "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/command_queue/random_program.cpp", cr_set, ComputeConfig{
                    .math_approx_mode = false,
                    .compile_args = trisc_compile_args,
                    .defines = compute_defines
                });
            SetRuntimeArgs(program, dummy_trisc_kernel, cr_set, trisc_unique_rtargs);
            SetCommonRuntimeArgs(program, dummy_trisc_kernel, trisc_common_rtargs);
            at_least_one_kernel = true;
        }

        if (not at_least_one_kernel) {
            uint32_t random_risc = rand() % 3 + 1;
            if (random_risc == 1) {
                auto dummy_brisc_kernel = CreateKernel(
                    program, "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/command_queue/random_program.cpp", cr_set, DataMovementConfig{
                        .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = brisc_compile_args, .defines = data_movement_defines});
                SetRuntimeArgs(program, dummy_brisc_kernel, cr_set, brisc_unique_rtargs);
                SetCommonRuntimeArgs(program, dummy_brisc_kernel, brisc_common_rtargs);
            } else if (random_risc == 2) {
                auto dummy_ncrisc_kernel = CreateKernel(
                    program, "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/command_queue/random_program.cpp", cr_set, DataMovementConfig{
                        .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = ncrisc_compile_args, .defines = data_movement_defines});
                SetRuntimeArgs(program, dummy_ncrisc_kernel, cr_set, ncrisc_unique_rtargs);
                SetCommonRuntimeArgs(program, dummy_ncrisc_kernel, ncrisc_common_rtargs);
            } else if (random_risc == 3) {
                auto dummy_trisc_kernel = CreateKernel(
                    program, "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/command_queue/random_program.cpp", cr_set, ComputeConfig{
                        .math_approx_mode = false,
                        .compile_args = trisc_compile_args,
                        .defines = compute_defines
                    });
                SetRuntimeArgs(program, dummy_trisc_kernel, cr_set, trisc_unique_rtargs);
                SetCommonRuntimeArgs(program, dummy_trisc_kernel, trisc_common_rtargs);
            } else {
                TT_ASSERT("Invalid");
            }
        }

        tt::tt_metal::detail::CompileProgram(this->device_, program);
    }

    for (uint8_t cq_id = 0; cq_id < this->device_->num_hw_cqs(); ++cq_id) {
        log_info(tt::LogTest, "Running {} programs on cq {} for cache warmup.", programs.size(), (uint32_t)cq_id);
        // This loop caches program and runs
        for (Program& program: programs) {
            EnqueueProgram(this->device_->command_queue(cq_id), program, false);
        }

        // This loops assumes already cached
        uint32_t NUM_ITERATIONS = 500; // TODO(agrebenisan): Bump this to 5000, saw hangs for very large number of iterations, need to come back to that

        log_info(tt::LogTest, "Running {} programs on cq {} for {} iterations now.", programs.size(), (uint32_t)cq_id, NUM_ITERATIONS);
        for (uint32_t i = 0; i < NUM_ITERATIONS; i++) {
            auto rng = std::default_random_engine {};
            std::shuffle(std::begin(programs), std::end(programs), rng);
            if (i % 10 == 0) {
                log_debug(tt::LogTest, "Enqueueing {} programs on cq {} for iter: {}/{} now.", programs.size(), (uint32_t)cq_id, i+1, NUM_ITERATIONS);
            }
            for (Program& program: programs) {
                EnqueueProgram(this->device_->command_queue(cq_id), program, false);
            }
        }

        log_info(tt::LogTest, "Calling Finish.");
        Finish(this->device_->command_queue(cq_id));
    }
}

}  // namespace stress_tests
