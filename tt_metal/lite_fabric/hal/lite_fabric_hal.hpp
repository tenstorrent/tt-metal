// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "hw/inc/host_interface.hpp"
#include "tt_cluster.hpp"
#include "core_coord.hpp"
#include <memory>
#include <umd/device/types/arch.hpp>
#include <umd/device/types/cluster_types.hpp>
#include <umd/device/types/xy_pair.hpp>

namespace lite_fabric {

struct TunnelDescriptor {
    tt::ChipId mmio_id{0xffffffff};
    CoreCoord mmio_core_virtual{-1, -1};
    CoreCoord mmio_core_logical{-1, -1};

    tt::ChipId connected_id{0xffffffff};
    CoreCoord connected_core_virtual{-1, -1};
    CoreCoord connected_core_logical{-1, -1};

    int num_hops{0};

    tt_cxy_pair mmio_cxy_virtual() const { return {mmio_id, mmio_core_virtual}; }

    tt_cxy_pair connected_cxy_virtual() const { return {connected_id, connected_core_virtual}; }
};

struct SystemDescriptor {
    std::map<tt::ChipId, uint32_t> enabled_eth_channels;
    std::vector<TunnelDescriptor> tunnels_from_mmio;
};

// Abstract interface for the Lite Fabric cluster
class LiteFabricHal {
private:
    SystemDescriptor system_descriptor_;

public:
    LiteFabricHal(tt::Cluster& cluster);

    static std::shared_ptr<LiteFabricHal> create();

    virtual ~LiteFabricHal() = default;

    virtual void set_reset_state(tt::Cluster& cluster, tt_cxy_pair virtual_core, bool assert_reset) = 0;

    virtual void set_pc(tt::Cluster& cluster, tt_cxy_pair virtual_core, uint32_t pc_addr, uint32_t pc_val) = 0;

    virtual void launch(tt::Cluster& cluster, const SystemDescriptor& desc, const std::filesystem::path& bin_path) = 0;

    virtual void terminate(tt::Cluster& cluster, const SystemDescriptor& desc) = 0;

    virtual tt::umd::tt_version get_binary_version() = 0;

    virtual void wait_for_state(tt::Cluster& cluster, tt_cxy_pair virtual_core, lite_fabric::InitState state) = 0;

    virtual std::vector<std::filesystem::path> build_includes(const std::filesystem::path& root_dir) = 0;

    virtual std::vector<std::string> build_defines() = 0;

    virtual std::vector<std::filesystem::path> build_linker(const std::filesystem::path& root_dir) = 0;

    void wait_for_state(tt::Cluster& cluster, const SystemDescriptor& desc, lite_fabric::InitState state);

    void set_pc(tt::Cluster& cluster, const SystemDescriptor& desc, uint32_t pc_addr, uint32_t pc_val);

    void set_reset_state(tt::Cluster& cluster, const SystemDescriptor& desc, bool assert_reset);

};

}  // namespace lite_fabric
