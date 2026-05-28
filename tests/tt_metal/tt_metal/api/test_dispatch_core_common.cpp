// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <optional>

#include <tt-metalium/dispatch_core_common.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <tt-metalium/hal.hpp>

namespace tt::tt_metal {

TEST(DispatchCoreConfigTest, CreateDispatchCoreConfigConstructorRules) {
    EXPECT_ANY_THROW(DispatchCoreConfig::create_dispatch_core_config(DispatchCoreType::ETH, DispatchCoreAxis::COL));

    const auto col_axis_worker_config =
        DispatchCoreConfig::create_dispatch_core_config(std::nullopt, DispatchCoreAxis::COL);
    EXPECT_EQ(col_axis_worker_config.get_dispatch_core_type(), DispatchCoreType::WORKER);
    EXPECT_EQ(col_axis_worker_config.get_dispatch_core_axis(), DispatchCoreAxis::COL);

    if (tt::tt_metal::hal::get_arch() == tt::ARCH::BLACKHOLE) {
        EXPECT_ANY_THROW(DispatchCoreConfig::create_dispatch_core_config(
            std::nullopt, DispatchCoreAxis::ROW, tt::tt_fabric::FabricTensixConfig::DISABLED));

        const auto mux_config = DispatchCoreConfig::create_dispatch_core_config(
            std::nullopt, std::nullopt, tt::tt_fabric::FabricTensixConfig::MUX);
        EXPECT_EQ(mux_config.get_dispatch_core_axis(), DispatchCoreAxis::ROW);
    }
}

}  // namespace tt::tt_metal
