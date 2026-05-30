// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Verification and usage example of dispatch telemetry.

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>

#include <sys/utsname.h>

#include <fmt/core.h>

#include <hostdevcommon/common_values.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/dispatch_core_common.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/experimental/dispatch_telemetry.hpp>

using namespace tt;
using namespace tt::tt_metal;

namespace {
struct KernelVersion {
    unsigned major = 0;
    unsigned minor = 0;
    std::string release;
};

bool get_linux_kernel_version(KernelVersion& version) {
    utsname info{};
    if (uname(&info) != 0) {
        return false;
    }

    version.release = info.release;
    char* end = nullptr;
    version.major = static_cast<unsigned>(std::strtoul(version.release.c_str(), &end, 10));
    if (end == version.release.c_str() || *end != '.') {
        return false;
    }

    const char* minor_start = end + 1;
    version.minor = static_cast<unsigned>(std::strtoul(minor_start, &end, 10));
    return end != minor_start;
}

bool require_supported_kernel() {
    KernelVersion version;
    if (!get_linux_kernel_version(version)) {
        fmt::print(stderr, "Unable to determine Linux kernel version; refusing to run dispatch_telemetry_dump.\n");
        return false;
    }

    if (version.major > 5 || (version.major == 5 && version.minor >= 15)) {
        return true;
    }

    fmt::print(
        stderr,
        "dispatch_telemetry_dump requires Linux kernel 5.15 or newer; found {}.\n"
        "tt-kmd memory.c::is_pin_pages_size_safe() documents that with IOMMU enabled on Linux 5.4,\n"
        "large page pinnings can soft-lock during unpin in:\n"
        "  tt_cdev_release/ioctl_unpin_pages -> unmap_sg -> __unmap_single -> iommu_unmap_page\n",
        version.release);
    return false;
}

bool print_snapshot(IDevice* device, DispatchTelemetry& telemetry) {
    auto infos = telemetry.read_info();
    fmt::print(
        "dispatch_telemetry_dump  chip={}  num_hw_cqs={}  telemetry_api_version={}  ts={}s\n",
        device->id(),
        device->num_hw_cqs(),
        telemetry.version(),
        std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now().time_since_epoch()).count());

    if (infos.empty()) {
        fmt::print(stderr, "Failed to read dispatch telemetry info; see log warnings for validation details.\n");
        return false;
    }

    fmt::print(
        "{:>3} {:<9} {:>8} {:>26} {:>24}\n",
        "cq",
        "component",
        "waiting",
        "blocked_since_last_read",
        "work_since_last_read");
    for (const auto& info : infos) {
        fmt::print(
            "{:>3} {:<9} {:>8} {:>26} {:>24}\n",
            info.cq_id,
            "prefetch",
            info.prefetch_waiting_on_upstream ? "yes" : "no",
            info.prefetch_blocked_count_since_last_read,
            info.prefetch_command_count_since_last_read);
        fmt::print(
            "{:>3} {:<9} {:>8} {:>26} {:>24}\n",
            info.cq_id,
            "dispatch",
            info.dispatch_waiting_on_upstream ? "yes" : "no",
            info.dispatch_blocked_count_since_last_read,
            info.dispatch_program_count_since_last_read);
    }
    std::cout.flush();
    return true;
}

Program create_blank_program(const CoreCoord& core) {
    Program program = CreateProgram();
    CreateKernel(
        program,
        "tt_metal/kernels/dataflow/blank.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    return program;
}

void issue_workloads(distributed::MeshDevice* mesh_device, uint32_t num_programs) {
    if (num_programs == 0) {
        return;
    }

    auto& cq = mesh_device->mesh_command_queue();
    const auto target_devices = distributed::MeshCoordinateRange(mesh_device->shape());
    const CoreCoord worker_core{0, 0};

    fmt::print("Issuing {} blank workload program(s).\n", num_programs);
    for (uint32_t i = 0; i < num_programs; ++i) {
        distributed::MeshWorkload workload;
        workload.add_program(target_devices, create_blank_program(worker_core));
        distributed::EnqueueMeshWorkload(cq, workload, false);
    }
    distributed::Finish(cq);
}

}  // namespace

int main(int argc, char** argv) {
    if (!require_supported_kernel()) {
        return 1;
    }

    ChipId device_id = 0;
    uint32_t num_programs = 4;

    auto parse_u32 = [](const char* s) -> uint32_t {
        const long value = std::atol(s);
        return value > 0 ? static_cast<uint32_t>(value) : 0;
    };

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if ((a == "--device" || a == "-d") && i + 1 < argc) {
            device_id = std::atoi(argv[++i]);
        } else if ((a == "--programs") && i + 1 < argc) {
            num_programs = parse_u32(argv[++i]);
        } else if (a == "--help" || a == "-h") {
            fmt::print("Usage: {} [--device N] [--programs N]\n", argv[0]);
            return 0;
        }
    }

    auto mesh_device = distributed::MeshDevice::create_unit_mesh(
        device_id, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, 1, DispatchCoreConfig{DispatchCoreType::WORKER});
    IDevice* device = mesh_device->get_devices().front();

    DispatchTelemetry telemetry(*device);
    if (telemetry.read_info().empty()) {
        fmt::print(
            stderr, "Failed to read initial dispatch telemetry info; see log warnings for validation details.\n");
        return 1;
    }

    // TODO: Inspect device while it runs an independent workload instead of launching our own
    //       once the SMC/ARC region supports sharing dispatch core locations.
    issue_workloads(mesh_device.get(), num_programs);

    if (!print_snapshot(device, telemetry)) {
        return 1;
    }

    return 0;
}
