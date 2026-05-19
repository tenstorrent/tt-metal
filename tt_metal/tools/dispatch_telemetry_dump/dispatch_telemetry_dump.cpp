// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Verification and usage example of dispatch telemetry.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#include <sys/utsname.h>

#include <fmt/core.h>

#include <hostdevcommon/common_values.hpp>
#include <host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/dispatch_core_common.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/experimental/dispatch_telemetry.hpp>

#include "impl/context/metal_context.hpp"
#include "impl/dispatch/dispatch_core_manager.hpp"
#include "llrt/tt_cluster.hpp"

using namespace tt;
using namespace tt::tt_metal;

namespace {
constexpr uint32_t MB = 1 << 20;

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

enum class CoreRole : uint8_t {
    PREFETCH,
    PREFETCH_D,
    DISPATCH,
    DISPATCH_D,
    DISPATCH_S,
};

constexpr std::string_view to_string(CoreRole role) {
    switch (role) {
        case CoreRole::PREFETCH: return "PREFETCH";
        case CoreRole::PREFETCH_D: return "PREFETCH_D";
        case CoreRole::DISPATCH: return "DISPATCH";
        case CoreRole::DISPATCH_D: return "DISPATCH_D";
        case CoreRole::DISPATCH_S: return "DISPATCH_S";
    }
    return "UNKNOWN";
}

constexpr bool is_prefetch_role(CoreRole role) { return role == CoreRole::PREFETCH || role == CoreRole::PREFETCH_D; }

struct CoreEntry {
    CoreRole role;
    tt_cxy_pair cxy;
    uint8_t cq_id;
};

// Walk dispatch_core_manager and collect every allocated dispatch/prefetch core for a device.
std::vector<CoreEntry> collect_cores(IDevice* device) {
    std::vector<CoreEntry> entries;
    auto& dcm = MetalContext::instance().get_dispatch_core_manager();
    const auto& cluster = MetalContext::instance().get_cluster();
    ChipId chip = device->id();
    uint16_t channel = cluster.get_assigned_channel_for_device(chip);
    uint8_t num_cqs = device->num_hw_cqs();

    for (uint8_t cq = 0; cq < num_cqs; ++cq) {
        if (dcm.is_prefetcher_core_allocated(chip, channel, cq)) {
            entries.push_back({CoreRole::PREFETCH, dcm.prefetcher_core(chip, channel, cq), cq});
        }
        if (dcm.is_prefetcher_d_core_allocated(chip, channel, cq)) {
            entries.push_back({CoreRole::PREFETCH_D, dcm.prefetcher_d_core(chip, channel, cq), cq});
        }
        if (dcm.is_dispatcher_core_allocated(chip, channel, cq)) {
            entries.push_back({CoreRole::DISPATCH, dcm.dispatcher_core(chip, channel, cq), cq});
        }
        if (dcm.is_dispatcher_d_core_allocated(chip, channel, cq)) {
            entries.push_back({CoreRole::DISPATCH_D, dcm.dispatcher_d_core(chip, channel, cq), cq});
        }
        if (dcm.is_dispatcher_s_core_allocated(chip, channel, cq)) {
            entries.push_back({CoreRole::DISPATCH_S, dcm.dispatcher_s_core(chip, channel, cq), cq});
        }
    }
    return entries;
}

void print_snapshot(IDevice* device, const std::vector<CoreEntry>& entries) {
    auto core_type = MetalContext::instance().get_dispatch_core_manager().get_dispatch_core_type();

    fmt::print(
        "dispatch_telemetry_dump  chip={}  num_hw_cqs={}  cores={}  ts={}s\n",
        device->id(),
        device->num_hw_cqs(),
        entries.size(),
        std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
    fmt::print("{:<11} {:>3} {:>10} {:>10} {:>12} {:>10}\n", "role", "cq", "core", "blocked", "unblocked", "cmds");

    for (const auto& e : entries) {
        CoreCoord logical{e.cxy.x, e.cxy.y};
        std::string core_str = fmt::format("({},{})", logical.x, logical.y);
        std::string_view role_str = to_string(e.role);
        if (is_prefetch_role(e.role)) {
            auto t = read_prefetch_telemetry(device, logical, core_type);
            if (t) {
                fmt::print(
                    "{:<11} {:>3} {:>10} {:>10} {:>12} {:>10}\n",
                    role_str,
                    e.cq_id,
                    core_str,
                    t->blocked_by_host_count,
                    t->unblocked_by_host_count,
                    t->command_count);
            } else {
                fmt::print("{:<11} {:>3} {:>10} <invalid>\n", role_str, e.cq_id, core_str);
            }
        } else {
            auto t = read_dispatch_telemetry(device, logical, core_type);
            if (t) {
                fmt::print(
                    "{:<11} {:>3} {:>10} {:>10} {:>12} {:>10}\n",
                    role_str,
                    e.cq_id,
                    core_str,
                    t->blocked_by_host_count,
                    t->unblocked_by_host_count,
                    "-");
            } else {
                fmt::print("{:<11} {:>3} {:>10} <invalid>\n", role_str, e.cq_id, core_str);
            }
        }
    }
    std::cout.flush();
}

void flood_completion_queue(distributed::MeshDevice* mesh_device, uint32_t num_reads, uint32_t bytes_per_read) {
    if (num_reads == 0 || bytes_per_read == 0) {
        return;
    }

    auto& cq = mesh_device->mesh_command_queue();

    // bytes_per_read is used both as DRAM page size and total buffer size (one page = one buffer)
    distributed::DeviceLocalBufferConfig dram_config{.page_size = bytes_per_read, .buffer_type = BufferType::DRAM};
    distributed::ReplicatedBufferConfig buffer_config{.size = bytes_per_read};

    auto src_buffer = distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device);

    fmt::print(
        "Flooding completion queue: {} non-blocking reads x {} bytes = {} bytes total payload.\n",
        num_reads,
        bytes_per_read,
        (static_cast<uint64_t>(num_reads) * bytes_per_read));

    const distributed::MeshCoordinate shard_coord(0, 0);
    std::vector<std::vector<uint8_t>> dst_pool(num_reads, std::vector<uint8_t>(bytes_per_read));
    for (uint32_t i = 0; i < num_reads; ++i) {
        std::vector<distributed::ShardDataTransfer> transfers = {
            distributed::ShardDataTransfer{shard_coord}.host_data(dst_pool[i].data())};
        // Non-blocking to delay finish to the end. This allows host blocking of the dispatch completion queue.
        cq.enqueue_read_shards(transfers, src_buffer, false);
    }
    distributed::Finish(cq);
}

}  // namespace

int main(int argc, char** argv) {
    if (!require_supported_kernel()) {
        return 1;
    }

    ChipId device_id = 0;
    uint32_t num_reads = 512;
    uint32_t bytes_per_read = 1 * MB;

    auto parse_u32 = [](const char* s) -> uint32_t { return static_cast<uint32_t>(std::max<long>(0, std::atol(s))); };

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if ((a == "--device" || a == "-d") && i + 1 < argc) {
            device_id = std::atoi(argv[++i]);
        } else if (a == "--reads" && i + 1 < argc) {
            num_reads = parse_u32(argv[++i]);
        } else if (a == "--read-bytes" && i + 1 < argc) {
            bytes_per_read = parse_u32(argv[++i]);
        } else if (a == "--help" || a == "-h") {
            fmt::print("Usage: {} [--device N] [--reads N] [--read-bytes B]\n", argv[0]);
            return 0;
        }
    }

    // Verify bytes_per_read is a valid DRAM page size
    const uint32_t dram_alignment = MetalContext::instance().hal().get_alignment(HalMemType::DRAM);
    TT_FATAL(
        bytes_per_read != 0 && bytes_per_read % dram_alignment == 0,
        "--read-bytes ({}) must be a positive multiple of the DRAM alignment ({})",
        bytes_per_read,
        dram_alignment);

    auto mesh_device = distributed::MeshDevice::create_unit_mesh(
        device_id, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, 1, DispatchCoreConfig{DispatchCoreType::WORKER});
    IDevice* device = mesh_device->get_devices().front();

    auto entries = collect_cores(device);
    if (entries.empty()) {
        fmt::print(stderr, "No dispatch/prefetch cores found on chip {}\n", device_id);
        mesh_device->close();
        return 1;
    }

    flood_completion_queue(mesh_device.get(), num_reads, bytes_per_read);

    print_snapshot(device, entries);

    mesh_device->close();
    return 0;
}
