// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/ethernet_cores.hpp>

#include <algorithm>
#include <tt-metalium/execution/slow_dispatch.hpp>
#include <tt-metalium/vocabulary/hal_types.hpp>
#include <tt_stl/assert.hpp>

#include "context/metal_context.hpp"
#include "distributed/mesh_device_impl.hpp"

namespace tt::tt_metal::experimental::ethernet {

namespace {
IDevice* local_device(const distributed::MeshDevice& mesh_device, const distributed::MeshCoordinate& device_coord) {
    TT_FATAL(mesh_device.impl().is_local(device_coord), "device {} is not local", device_coord);
    return mesh_device.impl().get_device(device_coord);
}
}  // namespace

std::vector<EthernetCore> get_ethernet_cores(
    const distributed::MeshDevice& mesh_device, const distributed::MeshCoordinate& device_coord) {
    IDevice* device = local_device(mesh_device, device_coord);
    const auto& soc_desc = MetalContext::instance().get_cluster().get_soc_desc(device->id());
    const auto active = device->get_active_ethernet_cores(true);

    std::vector<EthernetCore> cores;
    for (const auto& logical : device->ethernet_cores()) {
        cores.push_back({
            .logical = logical,
            .virtual_core = device->ethernet_core_from_logical_core(logical),
            .active = active.contains(logical),
            .channel = soc_desc.logical_eth_core_to_chan_map.at(logical),
        });
    }
    std::sort(cores.begin(), cores.end(), [](const auto& a, const auto& b) { return a.channel < b.channel; });
    return cores;
}

EthernetL1Range get_common_unreserved_l1_range() {
    const auto& hal = MetalContext::instance().hal();
    const auto range = [&](HalProgrammableCoreType type) {
        const auto start = hal.get_dev_addr(type, HalL1MemAddrType::UNRESERVED);
        return std::pair<uint64_t, uint64_t>{start, start + hal.get_dev_size(type, HalL1MemAddrType::UNRESERVED)};
    };
    const auto [active_start, active_end] = range(HalProgrammableCoreType::ACTIVE_ETH);
    const auto [idle_start, idle_end] = range(HalProgrammableCoreType::IDLE_ETH);
    return {
        .start = static_cast<uint32_t>(std::max(active_start, idle_start)),
        .end = static_cast<uint32_t>(std::min(active_end, idle_end)),
    };
}

EthernetRiscInfo get_risc_info() {
    const auto& hal = MetalContext::instance().hal();
    return {
        .active_fw_cooperative = hal.get_eth_fw_is_cooperative(),
        .active_eth_riscs = hal.get_num_risc_processors(HalProgrammableCoreType::ACTIVE_ETH),
        .idle_eth_riscs = hal.get_num_risc_processors(HalProgrammableCoreType::IDLE_ETH),
    };
}

std::vector<uint32_t> read_l1(
    const distributed::MeshDevice& mesh_device,
    const distributed::MeshCoordinate& device_coord,
    const CoreCoord& logical_core,
    bool is_ethernet,
    uint32_t address,
    uint32_t size_bytes) {
    TT_FATAL(size_bytes % sizeof(uint32_t) == 0, "read_l1 size {} is not a multiple of 4", size_bytes);
    std::vector<uint32_t> data;
    const bool ok = tt::tt_metal::detail::ReadFromDeviceL1(
        local_device(mesh_device, device_coord),
        logical_core,
        address,
        size_bytes,
        data,
        is_ethernet ? CoreType::ETH : CoreType::WORKER);
    TT_FATAL(ok, "read_l1 of core {} at {:#x} failed", logical_core, address);
    return data;
}

void write_l1(
    const distributed::MeshDevice& mesh_device,
    const distributed::MeshCoordinate& device_coord,
    const CoreCoord& logical_core,
    bool is_ethernet,
    uint32_t address,
    std::vector<uint32_t> words) {
    const bool ok = tt::tt_metal::detail::WriteToDeviceL1(
        local_device(mesh_device, device_coord),
        logical_core,
        address,
        words,
        is_ethernet ? CoreType::ETH : CoreType::WORKER);
    TT_FATAL(ok, "write_l1 to core {} at {:#x} failed", logical_core, address);
}

}  // namespace tt::tt_metal::experimental::ethernet
