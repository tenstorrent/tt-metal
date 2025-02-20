// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <magic_enum/magic_enum.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/device_pool.hpp>
#include <tt-metalium/logger.hpp>
#include <vector>
#include "core_coord.hpp"
#include "hostdevcommon/common_values.hpp"
#include "tt_cluster.hpp"

auto get_all_device_ids() {
    auto& cluster = tt::Cluster::instance();
    tt::log_info("Cluster type {}", magic_enum::enum_name(cluster.get_cluster_type()));
    std::vector<int> device_ids;
    for (int i = 0; i < cluster.number_of_devices(); ++i) {
        device_ids.push_back(i);
    }
    return device_ids;
}

int main(int argc, char** argv) {
    // Force SD for now
    setenv("TT_METAL_SLOW_DISPATCH_MODE", "true", 1);

    // 1 - Configure Fabric to initialize
    tt::tt_metal::detail::InitializeFabricSetting(tt::tt_metal::detail::FabricSetting::FABRIC);

    // 2 - Create Devices
    // All devices need to be enabled for Fabric
    auto devices =
        tt::tt_metal::detail::CreateDevices(get_all_device_ids(), 1, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE);

    // 3 - Make a kernel on the MMIO and remote device talk to each other
    static const std::string k_DummyKernelSrc =
        "tests/tt_metal/tt_metal/perf_microbenchmark/dispatch/kernels/dummy_kernel.cpp";

    auto pgm_0 = tt::tt_metal::CreateProgram();
    auto pgm_1 = tt::tt_metal::CreateProgram();
    CoreCoord core{0, 0};

    // Upstream
    tt::tt_metal::CreateKernel(
        pgm_0,
        k_DummyKernelSrc,
        core,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
            .compile_args =
                {
                    false,
                },
        });

    // Downstream
    tt::tt_metal::CreateKernel(
        pgm_1,
        k_DummyKernelSrc,
        core,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
            .compile_args =
                {
                    true,
                },
        });

    tt::tt_metal::detail::LaunchProgram(devices[0], pgm_0);
    tt::tt_metal::detail::LaunchProgram(devices[1], pgm_1);
    tt::tt_metal::detail::WaitProgramDone(devices[0], pgm_0);
    tt::tt_metal::detail::WaitProgramDone(devices[1], pgm_1);

    // Teardown
    tt::tt_metal::detail::CloseDevices(devices);
    return 0;
}
