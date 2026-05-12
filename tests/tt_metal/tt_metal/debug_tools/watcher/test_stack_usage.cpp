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
    const std::string path = "tests/tt_metal/tt_metal/test_kernels/misc/watcher_stack.cpp";

    // Set up program
    distributed::MeshWorkload workload;
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    auto* device = mesh_device->get_devices()[0];

    CoreCoord coord = {0, 0};
    std::vector<uint32_t> compile_args{free};
    std::vector<std::string> expected{"Stack usage summary:"};

    // Helper to add expected message in watcher logs
    auto add_expected_msg = [&](const std::string& cpu) {
        expected.push_back(fmt::format(
            "{} highest stack usage: {} bytes free, on core "
            "* running kernel {} ({})",
            cpu,
            free,
            path,
            !free ? "OVERFLOW" : "Close to overflow"));
    };

    // Create DM kernels
    auto num_dms = hal.get_processor_types_count(HalProgrammableCoreType::TENSIX, 0);
    auto num_compute_types = hal.get_processor_types_count(HalProgrammableCoreType::TENSIX, 1);

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

        std::vector<experimental::metal2_host_api::KernelSpec> kernel_specs;
        std::vector<experimental::metal2_host_api::KernelSpecName> kernel_names;
        kernel_specs.reserve(num_kernels + 1);
        kernel_names.reserve(num_kernels + 1);

        for (uint32_t i = 0; i < num_kernels; i++) {
            std::string name = fmt::format("dm_{}", i);
            kernel_specs.push_back(experimental::metal2_host_api::KernelSpec{
                .unique_id = name,
                .source = experimental::metal2_host_api::KernelSpec::SourceFilePath{path},
                .num_threads = static_cast<uint8_t>(dms_per_kernel),
                .compile_time_arg_bindings = {{"usage", free}},
                .config_spec =
                    experimental::metal2_host_api::DataMovementConfiguration{
                        .gen2_data_movement_config =
                            experimental::metal2_host_api::DataMovementConfiguration::Gen2DataMovementConfig{}},
            });
            kernel_names.push_back(name);
        }
        constexpr const char* COMPUTE_NAME = "compute";
        kernel_specs.push_back(experimental::metal2_host_api::KernelSpec{
            .unique_id = COMPUTE_NAME,
            .source = experimental::metal2_host_api::KernelSpec::SourceFilePath{path},
            // One thread per Neo (Quasar Tensix has 4) so the compute kernel fans out across
            // all Neos; each Neo internally runs the kernel on its 4 TRISCs.
            .num_threads = 4,
            .compile_time_arg_bindings = {{"usage", free}},
            .config_spec = experimental::metal2_host_api::ComputeConfiguration{},
        });
        kernel_names.push_back(COMPUTE_NAME);

        experimental::metal2_host_api::WorkUnitSpec wu{
            .unique_id = "main",
            .kernels = kernel_names,
            .target_nodes = experimental::metal2_host_api::NodeCoord{coord},
        };
        experimental::metal2_host_api::ProgramSpec spec{
            .program_id = "watcher_stack",
            .kernels = kernel_specs,
            .work_units = {wu},
        };
        Program program = experimental::metal2_host_api::MakeProgramFromSpec(*mesh_device, spec);
        workload.add_program(device_range, std::move(program));
    } else {
        // BH/WH legacy path
        workload.add_program(device_range, {});
        auto& program_ = workload.get_programs().at(device_range);
        for (uint32_t type_idx = 0; type_idx < num_dms; type_idx++) {
            DataMovementConfig dm_config{};
            dm_config.processor = static_cast<DataMovementProcessor>(type_idx);
            dm_config.noc = (type_idx == 1) ? NOC::RISCV_1_default : NOC::RISCV_0_default;
            dm_config.compile_args = compile_args;
            CreateKernel(program_, path, coord, dm_config);
        }
        CreateKernel(program_, path, coord, ComputeConfig{.compile_args = compile_args});

        // Also run on idle ethernet, if present
        const auto& inactive_eth_cores = device->get_inactive_ethernet_cores();
        if (!inactive_eth_cores.empty() && fixture->IsSlowDispatch()) {
            // Just pick the first core
            CoreCoord idle_coord = CoreCoord(*inactive_eth_cores.begin());
            CreateKernel(
                program_,
                path,
                idle_coord,
                tt_metal::EthernetConfig{
                    .eth_mode = Eth::IDLE,
                    .noc = tt_metal::NOC::NOC_0,
                    .processor = DataMovementProcessor::RISCV_0,
                    .compile_args = compile_args});
        }
    }

    // Add expected messages for the DMs that ran a user kernel. On Quasar DM0/DM1 are reserved
    // for internal use, so user kernels run on DM2..DM7 (6 user DMs).
    const uint32_t dm_start = is_quasar ? 2u : 0u;
    for (uint32_t type_idx = dm_start; type_idx < num_dms; type_idx++) {
        uint32_t processor_idx =
            hal.get_processor_index(HalProgrammableCoreType::TENSIX, HalProcessorClassType::DM, type_idx);
        add_expected_msg(hal.get_processor_class_name(HalProgrammableCoreType::TENSIX, processor_idx, false));
    }
    for (uint32_t type_idx = 0; type_idx < num_compute_types; type_idx++) {
        uint32_t processor_idx =
            hal.get_processor_index(HalProgrammableCoreType::TENSIX, HalProcessorClassType::COMPUTE, type_idx);
        add_expected_msg(hal.get_processor_class_name(HalProgrammableCoreType::TENSIX, processor_idx, false));
    }
    const auto& inactive_eth_cores = device->get_inactive_ethernet_cores();
    if (!inactive_eth_cores.empty() && fixture->IsSlowDispatch()) {
        // TODO: replace string literal "ierisc" with hal.get_processor_class_name() after
        // unifying all tests + watcher_device_reader::get_riscv_name() with same method
        add_expected_msg("ierisc");
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
