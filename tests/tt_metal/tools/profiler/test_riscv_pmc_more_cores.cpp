// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Launches one kernel per idle ethernet core and one DRISC kernel, then reads the device profiler. Run it with
// check_riscv_pmc_more_cores.py, which compares the profiler log with TT_METAL_PROFILE_RISCV_PMC off and on.
// Idle ethernet and DRISC kernels run under slow dispatch only here, like the active ethernet profiler test.

#include <cstdint>
#include <map>
#include <string>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include "distributed/mesh_device_impl.hpp"
#include "impl/context/metal_context.hpp"
#include "llrt/hal.hpp"
#include "llrt/tt_cluster.hpp"
#include "tt_metal/impl/kernels/kernel.hpp"

using namespace tt;
using namespace tt::tt_metal;

namespace {

constexpr int kLaunches = 2;

// CreateKernel(DramConfig) indexes dram_bank_endpoint_coords, which lists the preferred endpoint first.
CoreCoord logical_dram_endpoint_for_noc(const metal_SocDescriptor& soc_desc, uint32_t dram_view, NOC noc) {
    const CoreCoord pref = soc_desc.get_preferred_worker_core_for_dram_view(dram_view, static_cast<uint8_t>(noc));
    const auto& endpoints = soc_desc.dram_bank_endpoint_coords.at(dram_view);
    for (uint32_t i = 0; i < endpoints.size(); i++) {
        if (endpoints[i] == pref) {
            return CoreCoord{dram_view, i};
        }
    }
    TT_THROW("Preferred DRAM endpoint ({}, {}) for bank {} not found", pref.x, pref.y, dram_view);
}

// The NOC0 endpoint of a bank is left in reset by the system firmware, so a DRISC kernel goes to another one.
CoreCoord usable_dram_core(const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    const auto& soc_desc = MetalContext::instance().get_cluster().get_soc_desc(mesh_device->build_id());
    const uint32_t noc0_sub = logical_dram_endpoint_for_noc(soc_desc, 0, NOC::NOC_0).y;
    return CoreCoord{0, noc0_sub == 0 ? 1u : 0u};
}

}  // namespace

int main() {
    bool pass = true;

    try {
        TT_FATAL(
            std::getenv("TT_METAL_SLOW_DISPATCH_MODE") != nullptr,
            "Idle ethernet and DRISC kernels require slow dispatch; set TT_METAL_SLOW_DISPATCH_MODE=1");

        std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(0);
        auto* device = mesh_device->impl().get_devices()[0];

        const auto idle_cores = device->get_inactive_ethernet_cores();
        TT_FATAL(!idle_cores.empty(), "Device {} has no idle ethernet cores", device->id());
        fmt::print("idle ethernet cores: {}\n", idle_cores.size());

        const std::map<std::string, std::string> eth_defines = {{"LOOP_COUNT", "20"}, {"LOOP_SIZE", "200"}};
        const bool has_drisc = MetalContext::instance().hal().has_programmable_core_type(HalProgrammableCoreType::DRAM);
        fmt::print("drisc cores: {}\n", has_drisc ? "yes" : "no");

        distributed::MeshCoordinateRange device_range(mesh_device->shape());
        for (int launch = 0; launch < kLaunches; launch++) {
            Program program = CreateProgram(*mesh_device);
            for (const CoreCoord& core : idle_cores) {
                CreateKernel(
                    program,
                    "tests/tt_metal/tools/profiler/kernels/timestamped_events.cpp",
                    core,
                    EthernetConfig{.eth_mode = Eth::IDLE, .noc = NOC::NOC_0, .defines = eth_defines});
            }
            if (has_drisc) {
                CreateKernel(
                    program,
                    "tests/tt_metal/tools/profiler/kernels/drisc_nops.cpp",
                    usable_dram_core(mesh_device),
                    DramConfig{.noc = NOC::NOC_0, .defines = {{"LOOP_COUNT", "1000"}}});
            }
            distributed::MeshWorkload workload;
            workload.add_program(device_range, std::move(program));
            distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload, false);
            distributed::Finish(mesh_device->mesh_command_queue());
        }

        ReadMeshDeviceProfilerResults(*mesh_device);
        pass &= mesh_device->close();
    } catch (const std::exception& e) {
        pass = false;
        fmt::print(stderr, "{}\n", e.what());
    }

    if (!pass) {
        TT_THROW("Test Failed");
    }
    fmt::print("Test Passed\n");
    return 0;
}
