// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "lite_fabric_hal.hpp"

namespace lite_fabric {

class WormholeLiteFabricHal : public LiteFabricHal {
public:
    WormholeLiteFabricHal(tt::Cluster& cluster) : LiteFabricHal(cluster) {};

    void set_reset_state(tt::Cluster& cluster, tt_cxy_pair virtual_core, bool assert_reset) override;

    void set_pc(tt::Cluster& cluster, tt_cxy_pair virtual_core, uint32_t pc_addr, uint32_t pc_val) override;

    tt::umd::tt_version get_binary_version() override;

    void launch(tt::Cluster& cluster, const SystemDescriptor& desc, const std::filesystem::path& bin_path) override;

    void terminate(tt::Cluster& cluster, const SystemDescriptor& desc) override;

    void wait_for_state(tt::Cluster& cluster, tt_cxy_pair virtual_core, lite_fabric::InitState state) override;

    std::vector<std::filesystem::path> build_includes(const std::filesystem::path& root_dir) override;

    std::vector<std::string> build_defines() override;

    std::vector<std::filesystem::path> build_linker(const std::filesystem::path& root_dir) override;
};

}  // namespace lite_fabric
