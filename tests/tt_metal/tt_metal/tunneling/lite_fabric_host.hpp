// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <umd/device/types/xy_pair.h>
#include <utility>
#include <vector>
#include <memory>
#include "fabric_lite.hpp"
#include "kernel_types.hpp"
#include "program.hpp"
#include "tt_cluster.hpp"
#include "tt_memory.h"

namespace lite_fabric {

// Depth of 1
struct TunnelDescriptor {
    chip_id_t mmio_id{0xffffffff};
    CoreCoord mmio_core_virtual{-1, -1};
    CoreCoord mmio_core_logical{-1, -1};
    tt::tt_metal::KernelHandle mmio_kernel = 0;

    chip_id_t connected_id{0xffffffff};
    CoreCoord connected_core_virtual{-1, -1};
    CoreCoord connected_core_logical{-1, -1};

    tt_cxy_pair mmio_cxy_virtual() const { return {mmio_id, mmio_core_virtual}; }

    tt_cxy_pair connected_cxy_virtual() const { return {connected_id, connected_core_virtual}; }
};

struct SystemDescriptor {
    std::unordered_map<chip_id_t, uint32_t> enabled_eth_channels;
    std::vector<TunnelDescriptor> tunnels_from_mmio;
};

uint32_t GetEthChannelMask(chip_id_t device_id);

void SetResetState(std::shared_ptr<tt::Cluster> cluster, tt_cxy_pair virtual_core, bool assert_reset);

void SetPC(std::shared_ptr<tt::Cluster> cluster, tt_cxy_pair virtual_core, uint32_t pc_addr, uint32_t pc_val);

SystemDescriptor GetSystemDescriptor2Devices(
    const std::map<chip_id_t, tt::tt_metal::IDevice*>& devices,
    chip_id_t mmio_device_id,
    chip_id_t connected_device_id);

// Returns the binary and local init scratch address for a kernel
std::pair<const ll_api::memory&, uint32_t> GetBinaryMetadata(
    uint32_t build_id, tt::tt_metal::Program& pgm, tt::tt_metal::KernelHandle kernel_handle);

void wait_for_state(tt::Cluster& cluster, tt_cxy_pair virtual_core, lite_fabric::InitState state);

std::unique_ptr<tt::tt_metal::Program> LaunchLiteFabricWithMetal(
    std::map<chip_id_t, tt::tt_metal::IDevice*>& devices, const SystemDescriptor& desc);

void TerminateLiteFabric(tt::Cluster& cluster, const SystemDescriptor& desc);

}  // namespace lite_fabric
