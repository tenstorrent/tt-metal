// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include "hw/inc/host_interface.hpp"
#include "core_coord.hpp"
#include <umd/device/types/arch.hpp>
#include <umd/device/types/cluster_descriptor_types.hpp>
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
protected:
    SystemDescriptor system_descriptor_;

public:
    LiteFabricHal();

    static std::shared_ptr<LiteFabricHal> create();

    virtual ~LiteFabricHal() = default;

    virtual void set_reset_state(tt_cxy_pair virtual_core, bool assert_reset) = 0;

    virtual void set_pc(tt_cxy_pair virtual_core, uint32_t pc_val) = 0;

    virtual void launch(const std::filesystem::path& bin_path) = 0;

    virtual void terminate() = 0;

    virtual tt::umd::tt_version get_binary_version() = 0;

    virtual void wait_for_state(tt_cxy_pair virtual_core, lite_fabric::InitState state) = 0;

    virtual std::vector<std::filesystem::path> build_srcs(const std::filesystem::path& root_dir) = 0;

    virtual std::vector<std::filesystem::path> build_includes(const std::filesystem::path& root_dir) = 0;

    virtual std::vector<std::string> build_defines() = 0;

    virtual std::vector<std::filesystem::path> build_linker(const std::filesystem::path& root_dir) = 0;

    virtual std::optional<std::filesystem::path> build_asm_startup(const std::filesystem::path& root_dir) = 0;

    virtual std::string build_target_name() = 0;

    void wait_for_state(lite_fabric::InitState state);

    void set_pc(uint32_t pc_val);

    void set_reset_state(bool assert_reset);

    const SystemDescriptor& get_system_descriptor() { return system_descriptor_; }
};

}  // namespace lite_fabric
