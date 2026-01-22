// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/base.h>
#include <cstdint>
#include <cstdlib>
#include <tt-metalium/device.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <map>
#include <string>
#include <variant>

#include <tt_stl/assert.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/data_types.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_metal_profiler.hpp>
#include "impl/context/metal_context.hpp"
#include <impl/dispatch/dispatch_core_manager.hpp>
#include <llrt/tt_cluster.hpp>

using namespace tt;

void measure_latency(const std::string& kernel_name) {
    const int device_id = 0;
    tt_metal::IDevice* device = tt_metal::CreateDevice(device_id);

    uint16_t channel =
        tt::tt_metal::MetalContext::instance().get_cluster().get_assigned_channel_for_device(device->id());
    CoreCoord producer_logical_core =
        tt_metal::MetalContext::instance().get_dispatch_core_manager().prefetcher_core(device->id(), channel, 0);
    CoreCoord consumer_logical_core =
        tt_metal::MetalContext::instance().get_dispatch_core_manager().dispatcher_core(device->id(), channel, 0);

    TT_FATAL(
        producer_logical_core != consumer_logical_core,
        "Producer and consumer core are {}. They should not be the same!",
        producer_logical_core.str());

    auto first_worker_physical_core = device->worker_core_from_logical_core({0, 0});

    std::map<std::string, std::string> defines = {
        {"WORKER_NOC_X", std::to_string(first_worker_physical_core.x)},
        {"WORKER_NOC_Y", std::to_string(first_worker_physical_core.y)},
    };

    tt_metal::Program program = tt_metal::CreateProgram();
    tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/perf_microbenchmark/noc/kernels/" + kernel_name + ".cpp",
        consumer_logical_core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .defines = defines});

    tt::tt_metal::detail::SetDeviceProfilerDir(kernel_name + "_microbenchmark");
    tt::tt_metal::detail::FreshProfilerDeviceLog();
    tt::tt_metal::detail::CompileProgram(device, program);
    tt_metal::detail::LaunchProgram(device, program);
    tt_metal::CloseDevice(device);
}

int main() {
    if (getenv("TT_METAL_SLOW_DISPATCH_MODE") == nullptr) {
        TT_THROW("Test not supported w/ fast dispatch, exiting");
    }

    measure_latency("multicast_to_single_core");
    measure_latency("unicast_to_single_core");
}
