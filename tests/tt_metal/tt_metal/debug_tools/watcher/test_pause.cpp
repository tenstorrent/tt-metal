// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/base.h>
#include <gtest/gtest.h>
#include <cstdint>
#include <functional>
#include <optional>
#include <set>
#include <string>
#include <unordered_set>
#include <variant>
#include <vector>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/kernel_types.hpp>
#include "debug_tools_fixture.hpp"
#include <tt-metalium/device.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/program.hpp>
#include <tt_stl/span.hpp>
#include "impl/context/metal_context.hpp"
#include "impl/kernels/kernel.hpp"
#include <umd/device/types/xy_pair.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>

//////////////////////////////////////////////////////////////////////////////////////////
// A test for checking watcher pause feature.
//////////////////////////////////////////////////////////////////////////////////////////
using std::vector;
using namespace tt;
using namespace tt::tt_metal;

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {
void RunTest(MeshWatcherFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    // Set up program
    distributed::MeshWorkload workload;
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    auto* device = mesh_device->get_devices()[0];
    const auto& hal = MetalContext::instance().hal();
    const bool is_quasar = hal.get_arch() == tt::ARCH::QUASAR;
    // Watcher pause kernels use Metal 2.0 on all architectures.
    const std::string path_metal2 = "tests/tt_metal/tt_metal/test_kernels/misc/watcher_pause_2_0.cpp";

    CoreCoord xy_start = {0, 0}, xy_end;

    if (is_quasar) {
        // Quasar only supports single cluster currently
        // TODO: Once SW supports multiple quasar clusters/cores, expand to multiple CoreRangeSets
        xy_end = {0, 0};
    } else {
        // Test runs on a 5x5 grid on BH/WH
        xy_end = {4, 4};
    }

    // Write runtime args
    uint32_t clk_mhz = tt::tt_metal::MetalContext::instance().get_cluster().get_device_aiclk(device->id());
    uint32_t delay_cycles = clk_mhz * 500000;  // .5 seconds
    std::vector<experimental::KernelSpec> kernel_specs;
    std::vector<experimental::KernelSpecName> kernel_names;
    experimental::ProgramRunArgs params;
    auto add_dm_kernel =
        [&](const char* name, uint32_t num_threads, std::optional<tt::tt_metal::DataMovementProcessor> gen1_processor) {
            auto gen1_proc = gen1_processor.value_or(tt::tt_metal::DataMovementProcessor::RISCV_0);
            auto gen1_noc = (gen1_proc == tt::tt_metal::DataMovementProcessor::RISCV_1)
                                ? tt::tt_metal::NOC::RISCV_1_default
                                : tt::tt_metal::NOC::RISCV_0_default;
            experimental::DataMovementHardwareConfig dm_cfg;
            if (is_quasar) {
                dm_cfg = experimental::DataMovementGen2Config{};
            } else {
                dm_cfg = experimental::DataMovementGen1Config{.processor = gen1_proc, .noc = gen1_noc};
            }
            kernel_specs.push_back(experimental::KernelSpec{
                .unique_id = experimental::KernelSpecName{name},
                .source = path_metal2,
                .num_threads = num_threads,
                .runtime_arg_schema = {.common_runtime_arg_names = {"wait_cycles"}},
                .hw_config = dm_cfg,
            });
            kernel_names.emplace_back(name);
            params.kernel_run_args.push_back(experimental::ProgramRunArgs::KernelRunArgs{
                .kernel = experimental::KernelSpecName{name},
                .common_runtime_arg_values = {{"wait_cycles", delay_cycles}},
            });
        };

    if (is_quasar) {
        // On Quasar, launch kernel on all user DMs (DM2..DM7). DM0/DM1 are reserved for internal use.
        // TODO: Watcher features for ERISCs and TRISCs are temporarily skipped on Quasar until basic runtime bring-up.
        constexpr uint32_t kQuasarUserDmCores = 6;
        add_dm_kernel("pause_dm", static_cast<uint8_t>(kQuasarUserDmCores), std::nullopt);
    } else {
        add_dm_kernel("pause_brisc", 1, tt::tt_metal::DataMovementProcessor::RISCV_0);
        add_dm_kernel("pause_ncrisc", 1, tt::tt_metal::DataMovementProcessor::RISCV_1);

        const experimental::KernelSpecName compute_kernel_name{"pause_compute"};
        kernel_specs.push_back(experimental::KernelSpec{
            .unique_id = compute_kernel_name,
            .source = path_metal2,
            .num_threads = 1,
            .runtime_arg_schema = {.common_runtime_arg_names = {"wait_cycles"}},
            .hw_config = experimental::ComputeGen1Config{},
        });
        kernel_names.emplace_back(compute_kernel_name);
        params.kernel_run_args.push_back(experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = compute_kernel_name,
            .common_runtime_arg_values = {{"wait_cycles", delay_cycles}},
        });
    }
    experimental::WorkUnitSpec wu{
        .name = "main",
        .kernels = kernel_names,
        .target_nodes = experimental::NodeRange{CoreRange(xy_start, xy_end)},
    };
    experimental::ProgramSpec spec{
        .name = "watcher_pause",
        .kernels = kernel_specs,
        .work_units = {wu},
    };
    Program program = experimental::MakeProgramFromSpec(*mesh_device, spec);
    experimental::SetProgramRunArgs(program, params);
    workload.add_program(device_range, std::move(program));

    // Run the program
    fixture->RunProgram(mesh_device, workload);

    // Check that the pause message is present for each core in the watcher log.
    vector<std::string> expected_strings;
    for (uint32_t x = xy_start.x; x <= xy_end.x; x++) {
        for (uint32_t y = xy_start.y; y <= xy_end.y; y++) {
            CoreCoord virtual_core = device->worker_core_from_logical_core({x, y});
            uint32_t num_processors =
                is_quasar ? hal.get_processor_types_count(HalProgrammableCoreType::TENSIX, 0)  // DMs only
                          : hal.get_num_risc_processors(HalProgrammableCoreType::TENSIX);      // all 5
            // On Quasar, DM0/DM1 are reserved for internal use and don't run the pause kernel,
            // so the pause message only appears for DM2..DM7.
            uint32_t first_processor_idx = is_quasar ? 2u : 0u;
            for (uint32_t processor_idx = first_processor_idx; processor_idx < num_processors; processor_idx++) {
                const std::string& risc_str =
                    hal.get_processor_class_name(HalProgrammableCoreType::TENSIX, processor_idx, false);
                std::string expected = fmt::format("{}:{}", virtual_core.str(), risc_str);
                expected_strings.push_back(expected);
            }
        }
    }

    EXPECT_TRUE(FileContainsAllStrings(fixture->log_file_name, expected_strings));
}

void RunEthTest(MeshWatcherFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    auto* device = mesh_device->get_devices()[0];
    if (MetalContext::instance().hal().get_arch() == tt::ARCH::QUASAR) {
        GTEST_SKIP() << "Metal 2.0 does not yet support ETH KernelSpecs";
    }

    bool has_active_eth_cores = !device->get_active_ethernet_cores(true).empty();
    bool has_idle_eth_cores = fixture->IsSlowDispatch() && !device->get_inactive_ethernet_cores().empty();
    if (!has_active_eth_cores && !has_idle_eth_cores) {
        GTEST_SKIP() << "No supported ETH cores available";
    }

    const std::string kernel_path = "tests/tt_metal/tt_metal/test_kernels/misc/watcher_pause.cpp";
    uint32_t clk_mhz = MetalContext::instance().get_cluster().get_device_aiclk(device->id());
    std::vector<uint32_t> args = {clk_mhz * 500000};  // .5 seconds
    Program program = CreateProgram();

    auto create_eth_kernels = [&](bool is_active) {
        std::set<CoreRange> eth_core_ranges;
        auto eth_cores = is_active ? device->get_active_ethernet_cores(true) : device->get_inactive_ethernet_cores();
        for (const auto& core : eth_cores) {
            log_info(
                LogTest,
                "Running on {} eth core {}({})",
                is_active ? "active" : "inactive",
                core.str(),
                device->ethernet_core_from_logical_core(core).str());
            eth_core_ranges.insert(CoreRange(core, core));
        }
        EthernetConfig eth_config{.noc = NOC::NOC_0};
        if (!is_active) {
            eth_config.eth_mode = Eth::IDLE;
        }
        KernelHandle erisc_kid = CreateKernel(program, kernel_path, eth_core_ranges, eth_config);
        SetCommonRuntimeArgs(program, erisc_kid, args);
    };

    if (has_active_eth_cores) {
        create_eth_kernels(true);
    }
    if (has_idle_eth_cores) {
        create_eth_kernels(false);
    }

    distributed::MeshWorkload workload;
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    workload.add_program(distributed::MeshCoordinateRange(zero_coord, zero_coord), std::move(program));
    fixture->RunProgram(mesh_device, workload);

    std::vector<std::string> expected_strings;
    auto add_expected_messages = [&](bool is_active) {
        auto eth_cores = is_active ? device->get_active_ethernet_cores(true) : device->get_inactive_ethernet_cores();
        for (const auto& core : eth_cores) {
            CoreCoord virtual_core = device->ethernet_core_from_logical_core(core);
            expected_strings.push_back(fmt::format("{}:{}", virtual_core.str(), is_active ? "erisc" : "ierisc"));
        }
    };
    if (has_active_eth_cores) {
        add_expected_messages(true);
    }
    if (has_idle_eth_cores) {
        add_expected_messages(false);
    }
    EXPECT_TRUE(FileContainsAllStrings(fixture->log_file_name, expected_strings));
}
}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

TEST_F(MeshWatcherFixture, TensixTestWatcherPause) {
    for (auto& mesh_device : this->devices_) {
        this->RunTestOnDevice(CMAKE_UNIQUE_NAMESPACE::RunTest, mesh_device);
    }
}

TEST_F(MeshWatcherFixture, EthernetTestWatcherPause) {
    for (auto& mesh_device : this->devices_) {
        this->RunTestOnDevice(CMAKE_UNIQUE_NAMESPACE::RunEthTest, mesh_device);
    }
}
