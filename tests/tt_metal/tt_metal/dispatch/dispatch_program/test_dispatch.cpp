// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// This file contains dispatch tests that are (generally) dispatch mode agnostic

#include <fmt/base.h>
#include <gtest/gtest.h>
#include <stdint.h>
#include <tt-metalium/allocator.hpp>
#include <map>
#include <memory>
#include <unordered_set>
#include <variant>
#include <vector>

#include <tt-metalium/circular_buffer_constants.h>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/data_types.hpp>
#include <tt-metalium/device.hpp>
#include "dispatch_fixture.hpp"
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/logger.hpp>
#include <tt-metalium/program.hpp>
#include <tt_stl/span.hpp>
#include "impl/context/metal_context.hpp"
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "umd/device/tt_core_coordinates.h"
#include "umd/device/types/xy_pair.h"

namespace tt::tt_metal {

using std::vector;

// Test sync w/ semaphores betweeen eth/tensix cores
// Test will hang in the kernel if the sync doesn't work properly
static void test_sems_across_core_types(
    tt::tt_metal::DispatchFixture* fixture, vector<tt::tt_metal::IDevice*>& devices, bool active_eth) {
    // just something unique...
    constexpr uint32_t eth_sem_init_val = 33;
    constexpr uint32_t tensix_sem_init_val = 102;

    vector<uint32_t> compile_args;
    if (active_eth) {
        compile_args.push_back(static_cast<uint32_t>(tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH));
    } else {
        compile_args.push_back(static_cast<uint32_t>(tt::tt_metal::HalProgrammableCoreType::IDLE_ETH));
    }

    for (tt::tt_metal::IDevice* device : devices) {
        if (not device->is_mmio_capable()) {
            continue;
        }

        const auto& eth_cores =
            active_eth ? device->get_active_ethernet_cores() : device->get_inactive_ethernet_cores();
        if (eth_cores.size() > 0) {
            auto program = tt::tt_metal::CreateProgram();

            CoreCoord eth_core = *eth_cores.begin();
            CoreCoord phys_eth_core = device->virtual_core_from_logical_core(eth_core, CoreType::ETH);
            uint32_t eth_sem_id = CreateSemaphore(program, eth_core, eth_sem_init_val, CoreType::ETH);
            auto eth_kernel = CreateKernel(
                program,
                "tests/tt_metal/tt_metal/test_kernels/dataflow/semaphore_across_core_types.cpp",
                eth_core,
                tt::tt_metal::EthernetConfig{
                    .eth_mode = active_eth ? tt::tt_metal::Eth::RECEIVER : tt::tt_metal::Eth::IDLE,
                    .noc = tt::tt_metal::NOC::NOC_0,
                    .compile_args = compile_args,
                });

            CoreCoord tensix_core(0, 0);
            CoreCoord phys_tensix_core = device->worker_core_from_logical_core(tensix_core);
            uint32_t tensix_sem_id = CreateSemaphore(program, tensix_core, tensix_sem_init_val, CoreType::WORKER);
            auto tensix_kernel = CreateKernel(
                program,
                "tests/tt_metal/tt_metal/test_kernels/dataflow/semaphore_across_core_types.cpp",
                tensix_core,
                tt::tt_metal::DataMovementConfig{
                    .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                    .noc = tt::tt_metal::NOC::RISCV_0_default,
                    .compile_args = compile_args,
                });

            // Set up args
            vector<uint32_t> eth_rtas = {
                tt::tt_metal::MetalContext::instance().hal().noc_xy_encoding(phys_tensix_core.x, phys_tensix_core.y),
                eth_sem_id,
                tensix_sem_id,
                eth_sem_init_val,
                0,  // dummy so eth/tensix are different sizes w/ different offsets
                0,  // dummy so eth/tensix are different sizes w/ different offsets
                0,  // dummy so eth/tensix are different sizes w/ different offsets
                0,  // dummy so eth/tensix are different sizes w/ different offsets
                0,  // dummy so eth/tensix are different sizes w/ different offsets
                0,  // dummy so eth/tensix are different sizes w/ different offsets
                0,  // dummy so eth/tensix are different sizes w/ different offsets
                0,  // dummy so eth/tensix are different sizes w/ different offsets
            };
            SetRuntimeArgs(program, eth_kernel, eth_core, eth_rtas);

            vector<uint32_t> tensix_rtas = {
                tt::tt_metal::MetalContext::instance().hal().noc_xy_encoding(phys_eth_core.x, phys_eth_core.y),
                tensix_sem_id,
                eth_sem_id,
                tensix_sem_init_val,
            };
            SetRuntimeArgs(program, tensix_kernel, tensix_core, tensix_rtas);

            fixture->RunProgram(device, program);
        }
    }
}

TEST_F(DispatchFixture, EthTestBlank) {
    IDevice* device = devices_[0];
    Program program = CreateProgram();

    // TODO: tweak when FD supports idle eth
    const auto& eth_cores =
        this->slow_dispatch_ ? device->get_inactive_ethernet_cores() : device->get_active_ethernet_cores();

    if (eth_cores.size() > 0) {
        CoreCoord eth_core = *eth_cores.begin();
        CoreCoord phys_eth_core = device->virtual_core_from_logical_core(eth_core, CoreType::ETH);
        CreateKernel(
            program,
            "tt_metal/kernels/dataflow/blank.cpp",
            eth_core,
            tt::tt_metal::EthernetConfig{
                .eth_mode = this->slow_dispatch_ ? Eth::IDLE : Eth::RECEIVER,
            });

        this->RunProgram(device, program);
    }
}

TEST_F(DispatchFixture, TensixTestInitLocalMemory) {
    // This test will hang/assert if there is a failure

    IDevice* device = devices_[0];
    CoreCoord core = {0, 0};
    Program program;

    CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/local_mem.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/local_mem.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

    CreateKernel(program, "tests/tt_metal/tt_metal/test_kernels/misc/local_mem.cpp", core, ComputeConfig{});

    this->RunProgram(device, program);
}

TEST_F(DispatchFixture, EthTestInitLocalMemory) {
    // This test will hang/assert if there is a failure

    if (not this->slow_dispatch_) {
        tt::log_warning("Skipping fast dispatch test until active eth memory map is fixed");
        return;
    }

    IDevice* device = devices_[0];
    Program program = CreateProgram();

    // TODO: tweak when FD supports idle eth
    const auto& eth_cores =
        this->slow_dispatch_ ? device->get_inactive_ethernet_cores() : device->get_active_ethernet_cores();

    if (eth_cores.size() > 0) {
        CoreCoord eth_core = *eth_cores.begin();
        CoreCoord phys_eth_core = device->virtual_core_from_logical_core(eth_core, CoreType::ETH);
        CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/misc/local_mem.cpp",
            eth_core,
            tt::tt_metal::EthernetConfig{.eth_mode = this->slow_dispatch_ ? Eth::IDLE : Eth::RECEIVER});

        this->RunProgram(device, program);
    }
}

TEST_F(DispatchFixture, TensixActiveEthTestSemaphores) { test_sems_across_core_types(this, this->devices_, true); }

TEST_F(DispatchFixture, TensixIdleEthTestSemaphores) {
    if (not this->slow_dispatch_) {
        GTEST_SKIP();
    }

    test_sems_across_core_types(this, this->devices_, false);
}

// This test was written to cover issue #12738 (CBs for workers showing up on
// active eth cores)
TEST_F(DispatchFixture, TensixActiveEthTestCBsAcrossDifferentCoreTypes) {
    uint32_t intermediate_cb = 24;
    uint32_t out_cb = 16;
    std::map<uint8_t, tt::DataFormat> intermediate_and_out_data_format_spec = {
        {intermediate_cb, tt::DataFormat::Float16_b}, {out_cb, tt::DataFormat::Float16_b}};
    uint32_t num_bytes_for_df = 2;
    uint32_t single_tile_size = num_bytes_for_df * 1024;
    uint32_t num_tiles = 2;
    uint32_t cb_size = num_tiles * single_tile_size;

    uint32_t cb_config_buffer_size =
        NUM_CIRCULAR_BUFFERS * UINT32_WORDS_PER_LOCAL_CIRCULAR_BUFFER_CONFIG * sizeof(uint32_t);

    for (IDevice* device : devices_) {
        CoreCoord worker_grid_size = device->compute_with_storage_grid_size();
        bool found_overlapping_core = false;
        CoreCoord core_coord;
        for (const auto& eth_core : device->get_active_ethernet_cores(true)) {
            if (eth_core.x < worker_grid_size.x && eth_core.y < worker_grid_size.y) {
                core_coord = eth_core;
                found_overlapping_core = true;
                break;
            }
        }

        if (not found_overlapping_core) {
            log_info(tt::LogTest, "No core overlaps worker and eth core ranges, skipping");
            return;
        }

        Program program;
        CircularBufferConfig cb_config = CircularBufferConfig(cb_size, intermediate_and_out_data_format_spec)
                                             .set_page_size(intermediate_cb, single_tile_size)
                                             .set_page_size(out_cb, single_tile_size);
        auto cb = CreateCircularBuffer(program, core_coord, cb_config);

        CreateKernel(
            program,
            "tt_metal/kernels/dataflow/blank.cpp",
            core_coord,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

        CreateKernel(
            program,
            "tt_metal/kernels/dataflow/blank.cpp",
            core_coord,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

        CreateKernel(
            program,
            "tt_metal/kernels/dataflow/blank.cpp",
            core_coord,
            EthernetConfig{.eth_mode = Eth::RECEIVER, .noc = NOC::NOC_0});

        this->RunProgram(device, program);

        vector<uint32_t> cb_config_vector;

        tt::tt_metal::detail::ReadFromDeviceL1(
            device,
            core_coord,
            program.get_cb_base_addr(device, core_coord, CoreType::WORKER),
            cb_config_buffer_size,
            cb_config_vector);

        // ETH core doesn't have CB
        EXPECT_TRUE(program.get_cb_size(device, core_coord, CoreType::ETH) == 0);

        uint32_t cb_addr = device->allocator()->get_base_allocator_addr(HalMemType::L1);
        uint32_t intermediate_index = intermediate_cb * sizeof(uint32_t);

        bool addr_match_intermediate = cb_config_vector.at(intermediate_index) == cb_addr;
        bool size_match_intermediate = cb_config_vector.at(intermediate_index + 1) == cb_size;
        bool num_pages_match_intermediate = cb_config_vector.at(intermediate_index + 2) == num_tiles;
        bool pass_intermediate = (addr_match_intermediate and size_match_intermediate and num_pages_match_intermediate);
        EXPECT_TRUE(pass_intermediate);

        uint32_t out_index = out_cb * sizeof(uint32_t);
        bool addr_match_out = cb_config_vector.at(out_index) == cb_addr;
        bool size_match_out = cb_config_vector.at(out_index + 1) == cb_size;
        bool num_pages_match_out = cb_config_vector.at(out_index + 2) == num_tiles;
        bool pass_out = (addr_match_out and size_match_out and num_pages_match_out);
        EXPECT_TRUE(pass_out);
    }
}

class EarlyReturnFixture : public DispatchFixture {
    void SetUp() override {
        tt::tt_metal::MetalContext::instance().rtoptions().set_kernels_early_return(true);
        DispatchFixture::SetUp();
    }
    void TearDown() override {
        DispatchFixture::TearDown();
        tt::tt_metal::MetalContext::instance().rtoptions().set_kernels_early_return(false);
    }
};

TEST_F(EarlyReturnFixture, TensixKernelEarlyReturn) {
    for (IDevice* device : devices_) {
        CoreCoord worker{0, 0};
        Program program;
        // Kernel will block if it doesn't early return.
        auto writer_kernel = CreateKernel(
            program,
            "tt_metal/kernels/dataflow/writer_unary.cpp",
            worker,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

        this->RunProgram(device, program);
    }
}

}  // namespace tt::tt_metal
