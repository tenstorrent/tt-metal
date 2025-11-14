// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "lite_fabric_hal.hpp"

namespace lite_fabric {

class BlackholeLiteFabricHal : public LiteFabricHal {
public:
    BlackholeLiteFabricHal() = default;

    void set_reset_state(tt_cxy_pair virtual_core, bool assert_reset) override;

    void set_pc(tt_cxy_pair virtual_core, uint32_t pc_val) override;

    void launch(const std::filesystem::path& bin_path) override;

    void terminate() override;

    void wait_for_state(tt_cxy_pair virtual_core, lite_fabric::InitState state) override;

    tt::umd::tt_version get_binary_version() override;

    std::vector<std::filesystem::path> build_includes(const std::filesystem::path& root_dir) override;

    std::vector<std::string> build_defines() override;

    std::vector<std::filesystem::path> build_linker(const std::filesystem::path& root_dir) override;

    std::string build_target_name() override { return "blackhole"; }
};

}  // namespace lite_fabric
