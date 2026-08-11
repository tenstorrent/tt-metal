// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <gtest/gtest.h>
#include "debug_tools_fixture.hpp"
#include "debug_tools_test_utils.hpp"
#include "hal_types.hpp"
#include "impl/context/metal_context.hpp"
#include <fmt/base.h>
#include <string>
#include <vector>
#include <optional>
#include "impl/kernels/kernel.hpp"

using namespace tt;
using namespace tt::tt_metal;

namespace {

void RunOneTest(
    MeshWatcherFixture* fixture,
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    unsigned free,
    std::optional<uint32_t> quasar_dms_per_kernel = std::nullopt) {
    const auto& hal = MetalContext::instance().hal();
    const bool is_quasar = hal.get_arch() == tt::ARCH::QUASAR;
    // TENSIX cores use the Metal 2.0 variant; idle-ETH cores use the legacy kernel.
    const std::string path_metal2 = "tests/tt_metal/tt_metal/test_kernels/misc/watcher_stack_2_0.cpp";
    const std::string path_legacy = "tests/tt_metal/tt_metal/test_kernels/misc/watcher_stack.cpp";

    // Set up program
    distributed::MeshWorkload workload;
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    auto* device = mesh_device->get_devices()[0];

    CoreCoord coord = {0, 0};
    std::vector<uint32_t> compile_args{free};
    std::vector<std::string> expected{"Stack usage summary:"};

    // Helper to add expected message in watcher logs. The "kernel" path reflects whichever
    // source file was compiled for that processor (Metal 2.0 for TENSIX, legacy for idle-ETH).
    auto add_expected_msg = [&](const std::string& cpu, const std::string& kernel_path) {
        expected.push_back(fmt::format(
            "{} highest stack usage: {} bytes free, on core "
            "* running kernel {} ({})",
            cpu,
            free,
            kernel_path,
            !free ? "OVERFLOW" : "Close to overflow"));
    };

    // Create DM kernels
    auto num_dms = hal.get_processor_types_count(HalProgrammableCoreType::TENSIX, 0);
    auto num_compute_types = hal.get_processor_types_count(HalProgrammableCoreType::TENSIX, 1);

    // TENSIX kernels are launched via Metal 2.0 on both gen1 (WH/BH) and gen2 (Quasar).
    std::vector<experimental::KernelSpec> kernel_specs;
    std::vector<experimental::KernelSpecName> kernel_names;

    if (is_quasar) {
        // On Quasar, DM0/DM1 are reserved for internal use; user kernels can only run on DM2..DM7.
        constexpr uint32_t kQuasarReservedDmCores = 2;
        const uint32_t num_user_dms = num_dms - kQuasarReservedDmCores;
        // quasar_dms_per_kernel = num_user_dms, num_kernels = 1. Same kernel launched on all user DMs (default)
        // quasar_dms_per_kernel = 1, num_kernels = num_user_dms. A single kernel launched on a unique user DM
        uint32_t dms_per_kernel = quasar_dms_per_kernel.value_or(num_user_dms);
        TT_FATAL(
            dms_per_kernel > 0 && num_user_dms % dms_per_kernel == 0,
            "dms_per_kernel ({}) must be positive and evenly divide num_user_dms ({})",
            dms_per_kernel,
            num_user_dms);
        uint32_t num_kernels = num_user_dms / dms_per_kernel;

        kernel_specs.reserve(num_kernels + 1);
        kernel_names.reserve(num_kernels + 1);

        for (uint32_t i = 0; i < num_kernels; i++) {
            std::string name = fmt::format("dm_{}", i);
            kernel_specs.push_back(experimental::KernelSpec{
                .unique_id = experimental::KernelSpecName{name},
                .source = path_metal2,
                .num_threads = dms_per_kernel,
                .compile_time_args = {{"usage", free}},
                .hw_config = experimental::DataMovementGen2Config{},
            });
            kernel_names.emplace_back(name);
        }
        const experimental::KernelSpecName COMPUTE_NAME{"compute"};
        kernel_specs.push_back(experimental::KernelSpec{
            .unique_id = COMPUTE_NAME,
            .source = path_metal2,
            // One thread per Neo (Quasar Tensix has 4) so the compute kernel fans out across
            // all Neos; each Neo internally runs the kernel on its 4 TRISCs.
            .num_threads = 4,
            .compile_time_args = {{"usage", free}},
            .hw_config = experimental::ComputeGen2Config{},
        });
        kernel_names.push_back(COMPUTE_NAME);
    } else {
        // WH/BH gen1: BRISC and NCRISC are separate KernelSpecs (one DM processor each).
        kernel_specs.reserve(num_dms + 1);
        kernel_names.reserve(num_dms + 1);
        for (uint32_t type_idx = 0; type_idx < num_dms; type_idx++) {
            std::string name = fmt::format("dm_{}", type_idx);
            auto processor = static_cast<tt::tt_metal::DataMovementProcessor>(type_idx);
            auto noc = (type_idx == 1) ? tt::tt_metal::NOC::RISCV_1_default : tt::tt_metal::NOC::RISCV_0_default;
            kernel_specs.push_back(experimental::KernelSpec{
                .unique_id = experimental::KernelSpecName{name},
                .source = path_metal2,
                .num_threads = 1,
                .compile_time_args = {{"usage", free}},
                .hw_config = experimental::DataMovementGen1Config{.processor = processor, .noc = noc},
            });
            kernel_names.emplace_back(name);
        }
        const experimental::KernelSpecName COMPUTE_NAME{"compute"};
        kernel_specs.push_back(experimental::KernelSpec{
            .unique_id = COMPUTE_NAME,
            .source = path_metal2,
            .num_threads = 1,
            .compile_time_args = {{"usage", free}},
            .hw_config = experimental::ComputeHardwareConfig{},
        });
        kernel_names.push_back(COMPUTE_NAME);
    }

    experimental::WorkUnitSpec wu{
        .name = "main",
        .kernels = kernel_names,
        .target_nodes = experimental::NodeCoord{coord},
    };
    experimental::ProgramSpec spec{
        .name = "watcher_stack",
        .kernels = kernel_specs,
        .work_units = {wu},
    };
    Program program = experimental::MakeProgramFromSpec(*mesh_device, spec);

    // Idle-ETH cores: invoke the original (legacy) kernel via the legacy host API.
    if (!is_quasar) {
        const auto& inactive_eth_cores = device->get_inactive_ethernet_cores();
        if (!inactive_eth_cores.empty() && fixture->IsSlowDispatch()) {
            CoreCoord idle_coord = CoreCoord(*inactive_eth_cores.begin());
            CreateKernel(
                program,
                path_legacy,
                idle_coord,
                tt_metal::EthernetConfig{
                    .eth_mode = Eth::IDLE,
                    .noc = tt_metal::NOC::NOC_0,
                    .processor = DataMovementProcessor::RISCV_0,
                    .compile_args = compile_args});
        }
    }
    workload.add_program(device_range, std::move(program));

    // Add expected messages for the DMs that ran a user kernel. On Quasar DM0/DM1 are reserved
    // for internal use, so user kernels run on DM2..DM7 (6 user DMs).
    const uint32_t dm_start = is_quasar ? 2u : 0u;
    for (uint32_t type_idx = dm_start; type_idx < num_dms; type_idx++) {
        uint32_t processor_idx =
            hal.get_processor_index(HalProgrammableCoreType::TENSIX, HalProcessorClassType::DM, type_idx);
        add_expected_msg(
            hal.get_processor_class_name(HalProgrammableCoreType::TENSIX, processor_idx, false), path_metal2);
    }
    for (uint32_t type_idx = 0; type_idx < num_compute_types; type_idx++) {
        uint32_t processor_idx =
            hal.get_processor_index(HalProgrammableCoreType::TENSIX, HalProcessorClassType::COMPUTE, type_idx);
        add_expected_msg(
            hal.get_processor_class_name(HalProgrammableCoreType::TENSIX, processor_idx, false), path_metal2);
    }
    const auto& inactive_eth_cores = device->get_inactive_ethernet_cores();
    if (!inactive_eth_cores.empty() && fixture->IsSlowDispatch()) {
        // TODO: replace string literal "ierisc" with hal.get_processor_class_name() after
        // unifying all tests + watcher_device_reader::get_riscv_name() with same method
        add_expected_msg("ierisc", path_legacy);
    }

    fixture->RunProgram(mesh_device, workload, true);
    EXPECT_TRUE(FileContainsAllStringsInOrder(fixture->log_file_name, expected));
}

} // namespace

struct StackUsageTestParams {
    std::string test_name;
    unsigned free_bytes;
    std::optional<uint32_t> quasar_dms_per_kernel;  // nullopt = default (all DMs), value = multi-kernel mode
    bool quasar_only;                               // If true, skip on non-Quasar
};

class StackUsageTest : public MeshWatcherFixture, public ::testing::WithParamInterface<StackUsageTestParams> {};

TEST_P(StackUsageTest, TestWatcherStackUsage) {
    const auto& params = GetParam();

    // Skip Quasar-only tests on other architectures
    if (params.quasar_only && MetalContext::instance().hal().get_arch() != tt::ARCH::QUASAR) {
        GTEST_SKIP() << "Test only applicable to Quasar";
    }

    for (auto& mesh_device : this->devices_) {
        this->RunTestOnDevice(
            [&params](MeshWatcherFixture* f, const std::shared_ptr<distributed::MeshDevice>& d) {
                RunOneTest(f, d, params.free_bytes, params.quasar_dms_per_kernel);
            },
            mesh_device);
    }
}

INSTANTIATE_TEST_SUITE_P(
    WatcherStackUsageTests,
    StackUsageTest,
    ::testing::Values(
        // Standard tests (all architectures, default Quasar uses single kernel with all user DMs)
        StackUsageTestParams{"StackUsage0", 0, std::nullopt, false},
        StackUsageTestParams{"StackUsage16", 16, std::nullopt, false},
        // Quasar only: multi-kernel mode (launch one kernel per user DM, i.e. 6 kernels on Quasar)
        StackUsageTestParams{"StackUsage0_QuasarMultiKernel", 0, 1, true},
        StackUsageTestParams{"StackUsage16_QuasarMultiKernel", 16, 1, true}),
    [](const ::testing::TestParamInfo<StackUsageTestParams>& info) { return info.param.test_name; });
