// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/experimental/host_api.hpp>
#include <gtest/gtest.h>
#include "debug_tools_fixture.hpp"
#include "debug_tools_test_utils.hpp"
#include "hal_types.hpp"
#include "impl/context/metal_context.hpp"
#include <fmt/base.h>
#include <string>
#include <vector>
#include <optional>

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
    workload.add_program(device_range, {});
    auto& program_ = workload.get_programs().at(device_range);
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
    if (is_quasar) {
        // quasar_dms_per_kernel = 8, num_kernels = 1. Same Kernel launched on all DMs (default)
        // quasar_dms_per_kernel = 1, num_kernels = 8. A single kernel launched on a unique DM
        uint32_t dms_per_kernel = quasar_dms_per_kernel.value_or(num_dms);
        TT_FATAL(
            dms_per_kernel > 0 && num_dms % dms_per_kernel == 0,
            "dms_per_kernel ({}) must be positive and evenly divide num_dms ({})",
            dms_per_kernel,
            num_dms);
        uint32_t num_kernels = num_dms / dms_per_kernel;
        for (uint32_t i = 0; i < num_kernels; i++) {
            experimental::quasar::CreateKernel(
                program_,
                path,
                coord,
                experimental::quasar::QuasarDataMovementConfig{
                    .num_processors_per_cluster = dms_per_kernel, .compile_args = compile_args});
        }
    } else {
        // BH/WH:
        for (uint32_t type_idx = 0; type_idx < num_dms; type_idx++) {
            DataMovementConfig dm_config{};
            dm_config.processor = static_cast<DataMovementProcessor>(type_idx);
            dm_config.noc = (type_idx == 1) ? NOC::RISCV_1_default : NOC::RISCV_0_default;
            dm_config.compile_args = compile_args;
            CreateKernel(program_, path, coord, dm_config);
        }
    }
    // Add expected messages for all DMs
    for (uint32_t type_idx = 0; type_idx < num_dms; type_idx++) {
        uint32_t processor_idx =
            hal.get_processor_index(HalProgrammableCoreType::TENSIX, HalProcessorClassType::DM, type_idx);
        add_expected_msg(hal.get_processor_class_name(HalProgrammableCoreType::TENSIX, processor_idx, false));
    }

    // Create compute kernels
    // TODO: Watcher features are temporarily skipped on Quasar until basic runtime bring-up for TRISCs is complete
    auto num_processor_classes = hal.get_processor_classes_count(HalProgrammableCoreType::TENSIX);
    if (!is_quasar && num_processor_classes > 1) {
        auto num_compute_types = hal.get_processor_types_count(HalProgrammableCoreType::TENSIX, 1);
        CreateKernel(program_, path, coord, ComputeConfig{.compile_args = compile_args});
        for (uint32_t type_idx = 0; type_idx < num_compute_types; type_idx++) {
            uint32_t processor_idx =
                hal.get_processor_index(HalProgrammableCoreType::TENSIX, HalProcessorClassType::COMPUTE, type_idx);
            add_expected_msg(hal.get_processor_class_name(HalProgrammableCoreType::TENSIX, processor_idx, false));
        }
    }

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
        // Standard tests (all architectures, default Quasar uses single kernel with all DMs)
        StackUsageTestParams{"StackUsage0", 0, std::nullopt, false},
        StackUsageTestParams{"StackUsage16", 16, std::nullopt, false},
        // Quasar only: multi-kernel mode (launch 8 kernels in total, each mapped to a unique DM each)
        StackUsageTestParams{"StackUsage0_QuasarMultiKernel", 0, 1, true},
        StackUsageTestParams{"StackUsage16_QuasarMultiKernel", 16, 1, true}),
    [](const ::testing::TestParamInfo<StackUsageTestParams>& info) { return info.param.test_name; });
