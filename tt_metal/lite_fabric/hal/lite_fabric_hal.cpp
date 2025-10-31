// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "lite_fabric_hal.hpp"
#include "hw/inc/host_interface.hpp"
#include "wormhole_impl.hpp"
#include "blackhole_impl.hpp"
#include "tt_metal/impl/context/metal_context.hpp"
#include <tt-metalium/control_plane.hpp>
#include <umd/device/types/cluster_descriptor_types.hpp>

namespace {

uint32_t GetEthChannelMask(tt::ChipId device_id) {
    auto& cp = tt::tt_metal::MetalContext::instance().get_control_plane();

    uint32_t mask = 0;
    for (const auto& core : cp.get_active_ethernet_cores(device_id)) {
        mask |= 0x1 << core.y;
    }

    return mask;
}

lite_fabric::SystemDescriptor GetSystemDescriptor() {
    auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    lite_fabric::SystemDescriptor desc;

    const auto mmio_chip_ids  = cluster.mmio_chip_ids();

    for (const auto& mmio_device_id : mmio_chip_ids) {
        // Get the eth mask for each device
        desc.enabled_eth_channels[mmio_device_id] = GetEthChannelMask(mmio_device_id);

        // Find the correct ethernet core to connect mmio device to connected device id
        const auto connected_id = cluster.get_ethernet_connected_device_ids(mmio_device_id);
        for (const auto& dev_id : connected_id) {
                desc.enabled_eth_channels[dev_id] = GetEthChannelMask(dev_id);
                // We only support 1 hop
                constexpr int hop_count = 1;
                for (const auto& mmio_eth_core : cluster.get_ethernet_sockets(mmio_device_id, dev_id)) {
                    const auto& [other_device, other_core] =
                        cluster.get_connected_ethernet_core({mmio_device_id, mmio_eth_core});
                    desc.tunnels_from_mmio.push_back(lite_fabric::TunnelDescriptor{
                        .mmio_id = mmio_device_id,
                        .mmio_core_virtual = cluster.get_virtual_coordinate_from_logical_coordinates(
                            mmio_device_id, mmio_eth_core, tt::CoreType::ETH),
                        .mmio_core_logical = mmio_eth_core,
                        .connected_id = other_device,
                        .connected_core_virtual = cluster.get_virtual_coordinate_from_logical_coordinates(
                            other_device, other_core, tt::CoreType::ETH),
                        .connected_core_logical = other_core,
                        .num_hops = hop_count,
                    });
                    log_info(
                        tt::LogMetal,
                        "Add tunnel from {} {} to {} {} ({} hops)",
                        mmio_device_id,
                        mmio_eth_core,
                        other_device,
                        other_core,
                        hop_count);
                }
        }
    }

    return desc;
}
}

namespace lite_fabric {

LiteFabricHal::LiteFabricHal() : system_descriptor_(GetSystemDescriptor()) {}

std::shared_ptr<LiteFabricHal> LiteFabricHal::create() {
    auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    auto arch = cluster.arch();
    switch (arch) {
        case tt::ARCH::WORMHOLE_B0: return std::make_shared<WormholeLiteFabricHal>();
        case tt::ARCH::BLACKHOLE: return std::make_shared<BlackholeLiteFabricHal>();
        default: TT_THROW("Unsupported architecture {}", arch);
    }
}

void LiteFabricHal::set_pc(uint32_t pc_val) {
    for (auto tunnel_1x : system_descriptor_.tunnels_from_mmio) {
        set_pc(tunnel_1x.mmio_cxy_virtual(), pc_val);
    }
}

void LiteFabricHal::set_reset_state(bool assert_reset) {
    for (auto tunnel_1x : system_descriptor_.tunnels_from_mmio) {
        set_reset_state(tunnel_1x.mmio_cxy_virtual(), assert_reset);
    }
}

void LiteFabricHal::wait_for_state(lite_fabric::InitState state) {
    for (auto tunnel_1x : system_descriptor_.tunnels_from_mmio) {
        wait_for_state(tunnel_1x.mmio_cxy_virtual(), state);
    }
}

}  // namespace lite_fabric
