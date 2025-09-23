// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_metal/lite_fabric/hw/inc/host_interface.hpp"

#include "tt_cluster.hpp"
#include "core_coord.hpp"
#include <umd/device/types/xy_pair.hpp>
#include <tt-metalium/control_plane.hpp>

namespace lite_fabric {

struct TunnelDescriptor {
    chip_id_t mmio_id{0xffffffff};
    CoreCoord mmio_core_virtual{-1, -1};
    CoreCoord mmio_core_logical{-1, -1};

    chip_id_t connected_id{0xffffffff};
    CoreCoord connected_core_virtual{-1, -1};
    CoreCoord connected_core_logical{-1, -1};

    int num_hops{0};

    tt_cxy_pair mmio_cxy_virtual() const { return {mmio_id, mmio_core_virtual}; }

    tt_cxy_pair connected_cxy_virtual() const { return {connected_id, connected_core_virtual}; }
};

struct SystemDescriptor {
    std::map<chip_id_t, uint32_t> enabled_eth_channels;
    std::vector<TunnelDescriptor> tunnels_from_mmio;
};

uint32_t GetEthChannelMask(chip_id_t device_id);

SystemDescriptor GetSystemDescriptorFromMmio(
    tt::Cluster& cluster, chip_id_t mmio_device_id);

void SetResetState(tt::Cluster& cluster, tt_cxy_pair virtual_core, bool assert_reset);

void SetResetState(tt::Cluster& cluster, const SystemDescriptor& desc, bool assert_reset);

void SetPC(tt::Cluster& cluster, tt_cxy_pair virtual_core, uint32_t pc_addr, uint32_t pc_val);

void SetPC(tt::Cluster& cluster, const SystemDescriptor& desc, uint32_t pc_addr, uint32_t pc_val);

void WaitForState(tt::Cluster& cluster, tt_cxy_pair virtual_core, uint32_t addr, lite_fabric::InitState state);

void WaitForState(tt::Cluster& cluster, const SystemDescriptor& desc, uint32_t addr, lite_fabric::InitState state);

void LaunchLiteFabric(
    tt::Cluster& cluster,
    const tt::tt_metal::Hal& hal,
    const SystemDescriptor& desc,
    const std::filesystem::path& elf_path);

void LaunchLiteFabric(tt::Cluster& cluster, const tt::tt_metal::Hal& hal, const SystemDescriptor& desc);

// Resume the execution of Lite Fabric which is already on the device
// Resets the PC to execute from the start of the Lite Fabric firmware and updates the host interface pointers
// WARNING: This function is only valid if all work as already been completed.
// WARNING: This function is only valid if the lite fabric binary and config on the device are still intact. If they are
// corrupted, a full
//          reinit with LaunchLiteFabric() is required.
template <typename HOST_INTERFACE>
void ResumeLiteFabric(
    tt::Cluster& cluster, const tt::tt_metal::Hal& hal, const SystemDescriptor& desc, HOST_INTERFACE& host_interface);

void TerminateLiteFabric(tt::Cluster& cluster, const SystemDescriptor& desc);

}  // namespace lite_fabric
