// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/ranges.h>
#include <gtest/gtest.h>
#include <umd/device/types/arch.h>
#include <umd/device/types/cluster_descriptor_types.h>
#include <umd/device/types/xy_pair.h>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/control_plane.hpp>
#include "kernel.hpp"
#include "tt_memory.h"
#include "build.hpp"

TEST(Tunneling, LiteFabricBuild) {
    std::string root_dir = std::getenv("TT_METAL_HOME");
    std::string lite_fabric_dir = fmt::format("{}/{}", root_dir, "lite_fabric");

    lite_fabric::CompileLiteFabric(root_dir, lite_fabric_dir);
    lite_fabric::LinkLiteFabric(root_dir, lite_fabric_dir);

    const std::string k_FabricLitePath = fmt::format("{}/lite_fabric/lite_fabric.elf", root_dir);
    ll_api::memory bin = ll_api::memory(k_FabricLitePath, ll_api::memory::Loading::DISCRETE);

    ASSERT_GE(bin.get_text_size(), 0);
    ASSERT_GE(bin.get_text_addr(), 0);
}
