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

TEST_F(CommonFixture, TestEthBlank) {

    Device *device = devices_[0];
    Program program = CreateProgram();

    // TODO: tweak when FD supports idle eth
    const auto &eth_cores = this->slow_dispatch_ ?
        device->get_inactive_ethernet_cores() :
        device->get_active_ethernet_cores();

    if (eth_cores.size() > 0) {
        CoreCoord eth_core = *eth_cores.begin();
        CoreCoord phys_eth_core = device->physical_core_from_logical_core(eth_core, CoreType::ETH);
        CreateKernel(
            program, "tt_metal/kernels/dataflow/blank.cpp", eth_core,
            tt::tt_metal::EthernetConfig {
                .eth_mode = this->slow_dispatch_ ? Eth::IDLE : Eth::RECEIVER,
            }
        );

        this->RunProgram(device, program);
    }
}

TEST_F(CommonFixture, TestTensixInitLocalMemory) {

    // This test will hang/assert if there is a failure

    Device *device = devices_[0];
    CoreCoord core = {0, 0};
    Program program;

    CreateKernel(
        program, "tests/tt_metal/tt_metal/test_kernels/misc/local_mem.cpp", core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    CreateKernel(
        program, "tests/tt_metal/tt_metal/test_kernels/misc/local_mem.cpp", core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

    CreateKernel(
        program, "tests/tt_metal/tt_metal/test_kernels/misc/local_mem.cpp", core,
        ComputeConfig{});

    this->RunProgram(device, program);
}

TEST_F(CommonFixture, TestEthInitLocalMemory) {

    // This test will hang/assert if there is a failure

    if (not this->slow_dispatch_) {
        tt::log_warning("Skipping fast dispatch test until active eth memory map is fixed");
        return;
    }

    Device *device = devices_[0];
    Program program = CreateProgram();

    // TODO: tweak when FD supports idle eth
    const auto &eth_cores = this->slow_dispatch_ ?
        device->get_inactive_ethernet_cores() :
        device->get_active_ethernet_cores();

    if (eth_cores.size() > 0) {
        CoreCoord eth_core = *eth_cores.begin();
        CoreCoord phys_eth_core = device->physical_core_from_logical_core(eth_core, CoreType::ETH);
        CreateKernel(
            program, "tests/tt_metal/tt_metal/test_kernels/misc/local_mem.cpp", eth_core,
            tt::tt_metal::EthernetConfig {
                .eth_mode = this->slow_dispatch_ ? Eth::IDLE : Eth::RECEIVER
            }
        );

        this->RunProgram(device, program);
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

// This test was written to cover issue #12738 (CBs for workers showing up on
// active eth cores)
TEST_F(CommonFixture, TestCBsAcrossWorkerEth) {

    uint32_t intermediate_cb = 24;
    uint32_t out_cb = 16;
    std::map<uint8_t, tt::DataFormat> intermediate_and_out_data_format_spec = {
        {intermediate_cb, tt::DataFormat::Float16_b},
        {out_cb, tt::DataFormat::Float16_b}
    };
    uint32_t num_bytes_for_df = 2;
    uint32_t single_tile_size = num_bytes_for_df * 1024;
    uint32_t num_tiles = 2;
    uint32_t cb_size = num_tiles * single_tile_size;

    uint32_t cb_config_buffer_size = NUM_CIRCULAR_BUFFERS * UINT32_WORDS_PER_CIRCULAR_BUFFER_CONFIG * sizeof(uint32_t);

    for (Device *device : devices_) {
        CoreCoord worker_grid_size = device->compute_with_storage_grid_size();
        bool found_overlapping_core = false;
        CoreCoord core_coord;
        for (const auto &eth_core : device->get_active_ethernet_cores(true)) {
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
            program, "tt_metal/kernels/dataflow/blank.cpp", core_coord,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

        CreateKernel(
            program, "tt_metal/kernels/dataflow/blank.cpp", core_coord,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

        CreateKernel(
            program, "tt_metal/kernels/dataflow/blank.cpp", core_coord,
            EthernetConfig{.eth_mode = Eth::RECEIVER, .noc = NOC::NOC_0});

        this->RunProgram(device, program);

        vector<uint32_t> cb_config_vector;

        tt::tt_metal::detail::ReadFromDeviceL1(
            device, core_coord,
            program.get_cb_base_addr(device, core_coord, CoreType::WORKER), cb_config_buffer_size, cb_config_vector);

        // ETH core doesn't have CB
        EXPECT_TRUE(program.get_cb_size(device, core_coord, CoreType::ETH) == 0);

        uint32_t cb_addr = L1_UNRESERVED_BASE;
        uint32_t intermediate_index = intermediate_cb * sizeof(uint32_t);

        bool addr_match_intermediate = cb_config_vector.at(intermediate_index) == ((cb_addr) >> 4);
        bool size_match_intermediate = cb_config_vector.at(intermediate_index + 1) == (cb_size >> 4);
        bool num_pages_match_intermediate = cb_config_vector.at(intermediate_index + 2) == num_tiles;
        bool pass_intermediate = (addr_match_intermediate and size_match_intermediate and num_pages_match_intermediate);
        EXPECT_TRUE(pass_intermediate);

        uint32_t out_index = out_cb * sizeof(uint32_t);
        bool addr_match_out = cb_config_vector.at(out_index) == ((cb_addr) >> 4);
        bool size_match_out = cb_config_vector.at(out_index + 1) == (cb_size >> 4);
        bool num_pages_match_out = cb_config_vector.at(out_index + 2) == num_tiles;
        bool pass_out = (addr_match_out and size_match_out and num_pages_match_out);
        EXPECT_TRUE(pass_out);
    }
}
