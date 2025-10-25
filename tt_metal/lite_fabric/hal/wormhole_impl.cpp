// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "wormhole_impl.hpp"

namespace lite_fabric {

void WormholeLiteFabricHal::set_reset_state(tt::Cluster& cluster, tt_cxy_pair virtual_core, bool assert_reset) {
}

void WormholeLiteFabricHal::set_pc(tt::Cluster& cluster, tt_cxy_pair virtual_core, uint32_t pc_addr, uint32_t pc_val) {
}

tt::umd::tt_version WormholeLiteFabricHal::get_binary_version() {
    return tt::umd::tt_version{0, 0, 0};
}

void WormholeLiteFabricHal::launch(tt::Cluster& cluster, const SystemDescriptor& desc, const std::filesystem::path& bin_path) {
}

void WormholeLiteFabricHal::terminate(tt::Cluster& cluster, const SystemDescriptor& desc) {
}

std::vector<std::filesystem::path> WormholeLiteFabricHal::build_includes(const std::filesystem::path& root_dir) {
    return {};
}

std::vector<std::string> WormholeLiteFabricHal::build_defines() {
    return {};
}

std::vector<std::filesystem::path> WormholeLiteFabricHal::build_linker(const std::filesystem::path& root_dir) {
    return {};
}

}  // namespace lite_fabric