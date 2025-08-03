// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <umd/device/types/xy_pair.h>
#include <vector>
#include <memory>
#include "tt_cluster.hpp"

namespace lite_fabric {

// Depth of 1
struct TunnelDescriptor {
    chip_id_t mmio_id;
    // Virtual core
    CoreCoord mmio_core;

    chip_id_t connected_id;
    // Virtual core
    CoreCoord connected_core;

    tt_cxy_pair mmio_cxy() const { return {mmio_id, mmio_core}; }

    tt_cxy_pair connected_cxy() const { return {connected_id, connected_core}; }
};

struct SystemDescriptor {
    std::unordered_map<chip_id_t, uint32_t> enabled_eth_channels;
    std::vector<TunnelDescriptor> tunnels_from_mmio;
};

uint32_t GetEthChannelMask(chip_id_t device_id);

void SetResetState(std::shared_ptr<tt::Cluster> cluster, tt_cxy_pair virtual_core, bool assert_reset);

void SetPC(std::shared_ptr<tt::Cluster> cluster, tt_cxy_pair virtual_core, uint32_t pc_addr, uint32_t pc_val);

}  // namespace lite_fabric
