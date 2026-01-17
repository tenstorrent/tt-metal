// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// This file contains dispatch tests that are (generally) dispatch mode agnostic

#include <fmt/base.h>
#include <gtest/gtest.h>
#include <cstdint>
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
#include "mesh_dispatch_fixture.hpp"
#include <distributed.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/program.hpp>
#include <tt_stl/span.hpp>
#include "impl/context/metal_context.hpp"
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <umd/device/types/core_coordinates.hpp>
#include <umd/device/types/xy_pair.hpp>

// Access to internal API: ProgramImpl::get_cb_base_addr, ProgramImpl::get_cb_size
#include "impl/program/program_impl.hpp"

namespace tt::tt_metal {

using std::vector;

// Test sync w/ semaphores betweeen eth/tensix cores
// Test will hang in the kernel if the sync doesn't work properly
static void test_sems_across_core_types(
    tt::tt_metal::MeshDispatchFixture* fixture,
    vector<std::shared_ptr<distributed::MeshDevice>>& devices,
    bool active_eth) {
    // just something unique...
    constexpr uint32_t eth_sem_init_val = 33;
    constexpr uint32_t tensix_sem_init_val = 102;

    vector<uint32_t> compile_args;
    if (active_eth) {
        compile_args.push_back(static_cast<uint32_t>(tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH));
    } else {
        compile_args.push_back(static_cast<uint32_t>(tt::tt_metal::HalProgrammableCoreType::IDLE_ETH));
    }

    for (const auto& mesh_device : devices) {
        auto* device = mesh_device->get_devices()[0];
        if (not device->is_mmio_capable()) {
            continue;
        }

        auto erisc_count = tt::tt_metal::MetalContext::instance().hal().get_num_risc_processors(
            tt::tt_metal::HalProgrammableCoreType::IDLE_ETH);
        if (active_eth) {
            erisc_count = tt::tt_metal::MetalContext::instance().hal().get_num_risc_processors(
                tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH);
        }
        for (uint32_t erisc_idx = 0; erisc_idx < erisc_count; erisc_idx++) {
            log_info(tt::LogTest, "Test {} ethernet DM{}", active_eth ? "active" : "idle", erisc_idx);
            DataMovementProcessor dm_processor = static_cast<DataMovementProcessor>(erisc_idx);

            const auto& eth_cores_unordered =
                active_eth ? device->get_active_ethernet_cores(true) : device->get_inactive_ethernet_cores();

            std::set<CoreCoord> eth_cores(eth_cores_unordered.begin(), eth_cores_unordered.end());
            if (eth_cores.empty()) {
                log_info(
                    tt::LogTest,
                    "No {} ethernet cores found on device {}, skipping",
                    active_eth ? "active" : "idle",
                    device->id());
                continue;
            }

            distributed::MeshWorkload workload;
            auto zero_coord = distributed::MeshCoordinate(0, 0);
            auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
            auto program = tt::tt_metal::CreateProgram();

            CoreCoord eth_core = *eth_cores.begin();
            CoreCoord phys_eth_core = mesh_device->virtual_core_from_logical_core(eth_core, CoreType::ETH);
            uint32_t eth_sem_id = CreateSemaphore(program, eth_core, eth_sem_init_val, CoreType::ETH);
            auto eth_kernel = CreateKernel(
                program,
                "tests/tt_metal/tt_metal/test_kernels/dataflow/semaphore_across_core_types.cpp",
                eth_core,
                tt::tt_metal::EthernetConfig{
                    .eth_mode = active_eth ? tt::tt_metal::Eth::RECEIVER : tt::tt_metal::Eth::IDLE,
                    .noc = static_cast<tt_metal::NOC>(dm_processor),
                    .processor = dm_processor,
                    .compile_args = compile_args,
                });

            CoreCoord tensix_core(0, 0);
            CoreCoord phys_tensix_core = mesh_device->worker_core_from_logical_core(tensix_core);
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
            workload.add_program(device_range, std::move(program));
            fixture->RunProgram(mesh_device, workload);
        }
    }
}

TEST_F(MeshDispatchFixture, EthTestBlank) {
    auto mesh_device = devices_[0];
    auto* device = mesh_device->get_devices()[0];
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);

    // TODO: tweak when FD supports idle eth
    const auto& eth_cores_unordered =
        this->slow_dispatch_ ? device->get_inactive_ethernet_cores() : device->get_active_ethernet_cores(true);

    std::set<CoreCoord> eth_cores(eth_cores_unordered.begin(), eth_cores_unordered.end());

    if (!eth_cores.empty()) {
        const auto prog_core_type = this->slow_dispatch_ ? tt::tt_metal::HalProgrammableCoreType::IDLE_ETH
                                                         : tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH;
        const auto erisc_count = tt::tt_metal::MetalContext::instance().hal().get_num_risc_processors(prog_core_type);
        for (uint32_t erisc_idx = 0; erisc_idx < erisc_count; erisc_idx++) {
            distributed::MeshWorkload workload;
            log_info(tt::LogTest, "Add ethernet DM{}", erisc_idx);
            DataMovementProcessor dm_processor = static_cast<DataMovementProcessor>(erisc_idx);
            Program program = CreateProgram();

            CoreCoord eth_core = *eth_cores.begin();
            CreateKernel(
                program,
                "tt_metal/kernels/dataflow/blank.cpp",
                eth_core,
                tt::tt_metal::EthernetConfig{
                    .eth_mode = this->slow_dispatch_ ? Eth::IDLE : Eth::RECEIVER,
                    .noc = static_cast<NOC>(erisc_idx),
                    .processor = dm_processor,
                });

            workload.add_program(device_range, std::move(program));
            this->RunProgram(mesh_device, workload);
        }
    }
}

TEST_F(MeshDispatchFixture, TensixTestInitLocalMemory) {
    // This test will hang/assert if there is a failure

    auto mesh_device = devices_[0];
    distributed::MeshWorkload workload;
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
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

    workload.add_program(device_range, std::move(program));
    this->RunProgram(mesh_device, workload);
}

TEST_F(MeshDispatchFixture, EthTestInitLocalMemory) {
    // This test will hang/assert if there is a failure

    if (not this->slow_dispatch_) {
        log_warning(tt::LogTest, "Skipping fast dispatch test until active eth memory map is fixed");
        return;
    }

    auto mesh_device = devices_[0];
    auto* device = mesh_device->get_devices()[0];
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);

    // TODO: tweak when FD supports idle eth
    const bool is_idle_eth = this->slow_dispatch_;
    const auto& eth_cores = is_idle_eth ? device->get_inactive_ethernet_cores() : device->get_active_ethernet_cores(true);

    if (eth_cores.empty()) {
        log_info(
            tt::LogTest,
            "No {} ethernet cores found on device {}, skipping",
            this->slow_dispatch_ ? "idle" : "active",
            device->id());
        return;
    }

    auto erisc_count = tt::tt_metal::MetalContext::instance().hal().get_num_risc_processors(
        tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH);
    if (is_idle_eth) {
        erisc_count = tt::tt_metal::MetalContext::instance().hal().get_num_risc_processors(
            tt::tt_metal::HalProgrammableCoreType::IDLE_ETH);
    }
    for (uint32_t erisc_idx = 0; erisc_idx < erisc_count; erisc_idx++) {
        DataMovementProcessor dm_processor = static_cast<DataMovementProcessor>(erisc_idx);
        CoreCoord eth_core = *eth_cores.begin();
        log_info(tt::LogTest, "Adding {} ethernet DM{} {}", this->slow_dispatch_ ? "idle" : "active", erisc_idx, eth_core.str());
        distributed::MeshWorkload workload;
        Program program = CreateProgram();
        CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/misc/local_mem.cpp",
            eth_core,
            tt::tt_metal::EthernetConfig{.eth_mode = this->slow_dispatch_ ? Eth::IDLE : Eth::RECEIVER, .noc = static_cast<NOC>(erisc_idx), .processor = dm_processor});

        workload.add_program(device_range, std::move(program));
        this->RunProgram(mesh_device, workload);
    }
}

TEST_F(MeshDispatchFixture, TensixActiveEthTestSemaphores) { test_sems_across_core_types(this, this->devices_, true); }

TEST_F(MeshDispatchFixture, TensixIdleEthTestSemaphores) {
    if (not this->slow_dispatch_) {
        GTEST_SKIP();
    }

    test_sems_across_core_types(this, this->devices_, false);
}

// This test was written to cover issue #12738 (CBs for workers showing up on
// active eth cores)
TEST_F(MeshDispatchFixture, TensixActiveEthTestCBsAcrossDifferentCoreTypes) {
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

    for (const auto& mesh_device : devices_) {
        auto* device = mesh_device->get_devices()[0];

        CoreCoord worker_grid_size = mesh_device->compute_with_storage_grid_size();
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

        const auto erisc_count = tt::tt_metal::MetalContext::instance().hal().get_num_risc_processors(
            tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH);

        for (uint32_t erisc_idx = 0; erisc_idx < erisc_count; erisc_idx++) {
            log_info(tt::LogTest, "Test active ethernet DM{}", erisc_idx);
            DataMovementProcessor dm_processor = static_cast<DataMovementProcessor>(erisc_idx);
            distributed::MeshWorkload workload;

            auto zero_coord = distributed::MeshCoordinate(0, 0);
            auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
            Program program;

            CircularBufferConfig cb_config = CircularBufferConfig(cb_size, intermediate_and_out_data_format_spec)
                                                .set_page_size(intermediate_cb, single_tile_size)
                                                .set_page_size(out_cb, single_tile_size);
            CreateCircularBuffer(program, core_coord, cb_config);

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
                EthernetConfig{.eth_mode = Eth::RECEIVER, .noc = static_cast<NOC>(erisc_idx), .processor = dm_processor});

            workload.add_program(device_range, std::move(program));
            this->RunProgram(mesh_device, workload);

            auto& program_ = workload.get_programs().at(device_range);

            vector<uint32_t> cb_config_vector;

            this->RunProgram(mesh_device, workload);

            // ETH core doesn't have CB
            EXPECT_TRUE(program_.impl().get_cb_size(device, core_coord, CoreType::ETH) == 0);

            // Skip L1 readback validation for mock devices (L1 memory not populated)
            if (tt::tt_metal::MetalContext::instance().get_cluster().get_target_device_type() !=
                tt::TargetDevice::Mock) {
                tt::tt_metal::detail::ReadFromDeviceL1(
                    device,
                    core_coord,
                    program_.impl().get_cb_base_addr(device, core_coord, CoreType::WORKER),
                    cb_config_buffer_size,
                    cb_config_vector);

                uint32_t cb_addr = mesh_device->allocator()->get_base_allocator_addr(HalMemType::L1);
                uint32_t intermediate_index = intermediate_cb * sizeof(uint32_t);

                EXPECT_EQ(cb_config_vector.at(intermediate_index), cb_addr);
                EXPECT_EQ(cb_config_vector.at(intermediate_index + 1), cb_size);
                EXPECT_EQ(cb_config_vector.at(intermediate_index + 2), num_tiles);

                uint32_t out_index = out_cb * sizeof(uint32_t);
                EXPECT_EQ(cb_config_vector.at(out_index), cb_addr);
                EXPECT_EQ(cb_config_vector.at(out_index + 1), cb_size);
                EXPECT_EQ(cb_config_vector.at(out_index + 2), num_tiles);
            }
        }
    }
}

class EarlyReturnFixture : public MeshDispatchFixture {
    void SetUp() override {
        tt::tt_metal::MetalContext::instance().rtoptions().set_kernels_early_return(true);
        MeshDispatchFixture::SetUp();
    }
    void TearDown() override {
        MeshDispatchFixture::TearDown();
        tt::tt_metal::MetalContext::instance().rtoptions().set_kernels_early_return(false);
    }
};

TEST_F(EarlyReturnFixture, TensixKernelEarlyReturn) {
    for (const auto& mesh_device : devices_) {
        CoreCoord worker{0, 0};
        distributed::MeshWorkload workload;
        auto zero_coord = distributed::MeshCoordinate(0, 0);
        auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
        Program program;
        // Kernel will block if it doesn't early return.
        CreateKernel(
            program,
            "tt_metal/kernels/dataflow/writer_unary.cpp",
            worker,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

        workload.add_program(device_range, std::move(program));
        this->RunProgram(mesh_device, workload);
    }
}

TEST_F(MeshDispatchFixture, TensixCircularBufferInitFunction) {
    for (const auto& mesh_device : devices_) {
        for (bool use_assembly : {true, false}) {
            for (uint32_t mask : {0xffffffffu, 0xaaaaaaaau}) {
                CoreCoord core{0, 0};
                distributed::MeshWorkload workload;
                auto zero_coord = distributed::MeshCoordinate(0, 0);
                auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
                Program program;

                std::map<std::string, std::string> defines;
                if (!use_assembly) {
                    defines["DISABLE_CB_ASSEMBLY"] = "1";
                }
                KernelHandle kernel = CreateKernel(
                    program,
                    "tests/tt_metal/tt_metal/test_kernels/misc/circular_buffer/cb_init.cpp",
                    core,
                    DataMovementConfig{
                        .processor = DataMovementProcessor::RISCV_1,
                        .noc = NOC::RISCV_1_default,
                        .defines = defines,
                        .opt_level = KernelBuildOptLevel::O2});
                uint32_t l1_unreserved_base = mesh_device->allocator()->get_base_allocator_addr(HalMemType::L1);
                std::vector<uint32_t> runtime_args{mask, l1_unreserved_base};
                SetRuntimeArgs(program, kernel, core, runtime_args);
                workload.add_program(device_range, std::move(program));
                this->RunProgram(mesh_device, workload);
            }
        }
    }
}

}  // namespace tt::tt_metal
