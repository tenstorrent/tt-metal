// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tests/tt_metal/tt_metal/unit_tests_common/common/common_fixture.hpp"
#include "gtest/gtest.h"
#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/test_utils/env_vars.hpp"
#include "tt_metal/impl/dispatch/command_queue.hpp"
#include "tt_metal/common/logger.hpp"


using namespace tt;

// Ensures we can successfully create kernels on available compute grid
TEST_F(CommonFixture, CreateKernelsOnComputeCores) {
    for (unsigned int id = 0; id < devices_.size(); id++) {
        tt_metal::Program program = CreateProgram();
        CoreCoord compute_grid = devices_.at(id)->compute_with_storage_grid_size();
        EXPECT_NO_THROW(
            auto test_kernel = tt_metal::CreateKernel(
                program,
                "tests/tt_metal/tt_metal/test_kernels/dataflow/dram_copy.cpp",
                CoreRange(CoreCoord(0, 0), CoreCoord(compute_grid.x, compute_grid.y)),
                {.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default}
            );
        );
    }
}

// Ensure we cannot create kernels on storage cores
TEST_F(CommonFixture, CreateKernelsOnStorageCores) {
    for (unsigned int id=0; id < devices_.size(); id++) {
        if (devices_.at(id)->storage_only_cores().empty()) {
            GTEST_SKIP() << "This test only runs on devices with storage only cores";
        }
        CoreRangeSet storage_core_range_set = CoreRangeSet(devices_.at(id)->storage_only_cores());
        EXPECT_ANY_THROW(
            auto test_kernel = tt_metal::CreateKernel(
                program,
                "tests/tt_metal/tt_metal/test_kernels/dataflow/dram_copy.cpp",
                storage_core_range_set,
                {.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default}
            );
        );
    }
}

TEST_F(CommonFixture, CreateKernelsOnDispatchCores) {
    if (getenv("TT_METAL_SLOW_DISPATCH_MODE")) {
        GTEST_SKIP() << "This test is only supported in fast dispatch mode";
    }
    for (unsigned int id=0; id < devices_.size(); id++) {
        std::vector<CoreCoord> dispatch_cores = tt::get_logical_dispatch_cores(device->id(), device->num_hw_cqs());
        CoreType dispatch_core_type = dispatch_core_manager::instance().get_dispatch_core_type(device->id());
        std::set<CoreCoord> dispatch_core_range_set(dispatch_cores.begin(), dispatch_cores.end());

        if (dispatch_core_type == CoreType::WORKER) {
            EXPECT_ANY_THROW(
                auto test_kernel = tt_metal::CreateKernel(
                    program,
                    "tests/tt_metal/tt_metal/test_kernels/dataflow/dram_copy.cpp",
                    dispatch_core_range_set,
                    {.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default}
                );
            );
        } else if (dispatch_core_type == CoreType::ETH) {
            EXPECT_ANY_THROW(
                auto test_kernel = tt_metal::CreateKernel(
                    program,
                    "tests/tt_metal/tt_metal/test_kernels/misc/erisc_print.cpp",
                    dispatch_core_range_set,
                    {.noc = tt_metal::NOC::NOC_0, .eth_mode = Eth::IDLE}
                );
            );
        }
    }
}
