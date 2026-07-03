// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <filesystem>
#include <gtest/gtest.h>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <umd/device/soc_arch_descriptor.hpp>
#include <umd/device/soc_descriptor.hpp>
#include <umd/device/types/core_coordinates.hpp>

#include "tt_metal/test_utils/env_vars.hpp"

namespace {

std::filesystem::path repo_root_from_test_file() {
    return std::filesystem::path(__FILE__).parent_path().parent_path().parent_path().parent_path().parent_path();
}

tt::umd::SocDescriptor load_quasar_2x3_soc() {
    const std::filesystem::path soc_yaml =
        repo_root_from_test_file() / "tt_metal/third_party/umd/tests/soc_descs/quasar_simulation_2x3.yaml";
    return tt::umd::SocDescriptor(
        std::make_shared<tt::umd::SocArchDescriptor>(soc_yaml.string()), {.noc_translation_enabled = true});
}

}  // namespace

TEST(DispatchEngineCoordinates, QuasarSimulation2x3SocDispatchList) {
    if (tt::get_arch_from_string(tt::test_utils::get_umd_arch_name()) != tt::ARCH::QUASAR) {
        GTEST_SKIP() << "Dispatch-engine coordinate tests require Quasar";
    }

    const tt::umd::SocDescriptor soc = load_quasar_2x3_soc();

    const auto dispatch_noc0 = soc.get_cores(tt::CoreType::DISPATCH, tt::CoordSystem::NOC0);
    ASSERT_EQ(dispatch_noc0.size(), 1u);
    EXPECT_EQ(dispatch_noc0[0].x, 0);
    EXPECT_EQ(dispatch_noc0[0].y, 2);

    // Synthetic logical index CoreCoord(index, 0) maps to dispatch: list entry at index
    // (see dispatch_engine_cores.cpp / metal_soc_descriptor.cpp).
    const CoreCoord logical_dispatch_index(0, 0);
    ASSERT_LT(logical_dispatch_index.x, dispatch_noc0.size());
    EXPECT_EQ(dispatch_noc0[logical_dispatch_index.x].x, 0);
    EXPECT_EQ(dispatch_noc0[logical_dispatch_index.x].y, 2);

    const tt::umd::CoreCoord dispatch_noc0_coord(0, 2, tt::CoreType::DISPATCH, tt::CoordSystem::NOC0);
    const tt::umd::CoreCoord translated =
        soc.translate_coord_to(dispatch_noc0_coord, tt::CoordSystem::NOC0, tt::CoordSystem::TRANSLATED);
    EXPECT_EQ(translated.core_type, tt::CoreType::DISPATCH);
    EXPECT_EQ(translated.x, 0);
    EXPECT_EQ(translated.y, 2);
}
