// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// This file contains dispatch tests that are (generally) dispatch mode agnostic

#include "tests/tt_metal/tt_metal/unit_tests_common/common/common_fixture.hpp"

// Test sync w/ semaphores betweeen eth/tensix cores
// Test will hang in the kernel if the sync doesn't work properly
static void test_sems_across_core_types(CommonFixture *fixture,
                                        vector<tt::tt_metal::Device*>& devices,
                                        bool active_eth) {
    // just something unique...
    constexpr uint32_t eth_sem_init_val = 33;
    constexpr uint32_t tensix_sem_init_val = 102;

    vector<uint32_t> compile_args;
    if (active_eth) {
        compile_args.push_back(static_cast<uint32_t>(HalProgrammableCoreType::ACTIVE_ETH));
    } else {
        compile_args.push_back(static_cast<uint32_t>(HalProgrammableCoreType::IDLE_ETH));
    }

    for (Device *device : devices) {
        if (not device->is_mmio_capable()) continue;

        const auto &eth_cores = active_eth ?
            device->get_active_ethernet_cores() :
            device->get_inactive_ethernet_cores();
        if (eth_cores.size() > 0) {
            Program program = CreateProgram();

            CoreCoord eth_core = *eth_cores.begin();
            CoreCoord phys_eth_core = device->physical_core_from_logical_core(eth_core, CoreType::ETH);
            uint32_t eth_sem_id = CreateSemaphore(program, eth_core, eth_sem_init_val, CoreType::ETH);
            auto eth_kernel = CreateKernel(
                program,
                "tests/tt_metal/tt_metal/test_kernels/dataflow/semaphore_across_core_types.cpp",
                eth_core,
                tt::tt_metal::EthernetConfig {
                    .eth_mode = active_eth ? Eth::RECEIVER : Eth::IDLE,
                    .noc = NOC::NOC_0,
                    .compile_args = compile_args,
                });

            CoreCoord tensix_core(0, 0);
            CoreCoord phys_tensix_core = device->worker_core_from_logical_core(tensix_core);
            uint32_t tensix_sem_id = CreateSemaphore(program, tensix_core, tensix_sem_init_val, CoreType::WORKER);
            auto tensix_kernel = CreateKernel(
                program,
                "tests/tt_metal/tt_metal/test_kernels/dataflow/semaphore_across_core_types.cpp",
                tensix_core,
                DataMovementConfig {
                    .processor = DataMovementProcessor::RISCV_0,
                    .noc = NOC::RISCV_0_default,
                    .compile_args = compile_args,
                });

            // Set up args
            vector<uint32_t> eth_rtas = {
                NOC_XY_ENCODING(phys_tensix_core.x, phys_tensix_core.y),
                eth_sem_id,
                tensix_sem_id,
                eth_sem_init_val,
                0, // dummy so eth/tensix are different sizes w/ different offsets
                0, // dummy so eth/tensix are different sizes w/ different offsets
                0, // dummy so eth/tensix are different sizes w/ different offsets
                0, // dummy so eth/tensix are different sizes w/ different offsets
                0, // dummy so eth/tensix are different sizes w/ different offsets
                0, // dummy so eth/tensix are different sizes w/ different offsets
                0, // dummy so eth/tensix are different sizes w/ different offsets
                0, // dummy so eth/tensix are different sizes w/ different offsets
            };
            SetRuntimeArgs(program, eth_kernel, eth_core, eth_rtas);

            vector<uint32_t> tensix_rtas = {
                NOC_XY_ENCODING(phys_eth_core.x, phys_eth_core.y),
                tensix_sem_id,
                eth_sem_id,
                tensix_sem_init_val,
            };
            SetRuntimeArgs(program, tensix_kernel, tensix_core, tensix_rtas);

            fixture->RunProgram(device, program);
        }
    }
}

TEST_F(CommonFixture, TestSemaphoresTensixActiveEth) {
    test_sems_across_core_types(this, this->devices_, true);
}

TEST_F(CommonFixture, TestSemaphoresTensixIdleEth) {
    if (not this->slow_dispatch_) {
        GTEST_SKIP();
    }

    test_sems_across_core_types(this, this->devices_, false);
}
