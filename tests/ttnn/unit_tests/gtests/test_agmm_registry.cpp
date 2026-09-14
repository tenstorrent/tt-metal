// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "ttnn/operations/experimental/ccl/all_gather_minimal_matmul_async/registry/agmm_config_registry.hpp"
#include "ttnn/operations/experimental/ccl/all_gather_minimal_matmul_async/registry/agmm_registry_data.hpp"

namespace {
namespace registry = ttnn::experimental::all_gather_minimal_matmul_registry;

TEST(AgmmRegistry, CohortsDoNotCrossDeviceCounts) {
    EXPECT_TRUE(registry::entries_for_device_count(8).empty());
    EXPECT_EQ(registry::entries_for_device_count(32).size(), 40U);
    EXPECT_TRUE(registry::entries_for_device_count(1).empty());
    EXPECT_TRUE(registry::entries_for_device_count(16).empty());
}

TEST(AgmmRegistry, EveryBh32EntryRoundTripsAndMaterializes) {
    const auto entries = registry::generated::bh32_entries();
    ASSERT_EQ(entries.size(), 40U);
    for (const auto& entry : entries) {
        EXPECT_EQ(entry.key.device, registry::generated::kBh32Device);
        EXPECT_EQ(registry::lookup(entry.key), &entry);
        EXPECT_TRUE(registry::materialize_recipe(entry).has_value());
    }
}

TEST(AgmmRegistry, LiveGridIsCheckedAsCapabilityNotIdentity) {
    auto live_key = registry::generated::bh32_entries().front().key;
    live_key.device.compute_grid_x += 1;
    live_key.device.compute_grid_y += 1;
    EXPECT_NE(registry::lookup(live_key), nullptr);

    live_key.device.compute_grid_x = registry::generated::kBh32Device.compute_grid_x - 1;
    EXPECT_EQ(registry::lookup(live_key), nullptr);
}

TEST(AgmmRegistry, SelectionUsesTheSharedMatmulMode) {
    const auto& entry = registry::generated::bh32_entries().front();
    const auto& key = entry.key;
    auto facts = registry::RegistryRequestFacts{
        .device = key.device,
        .workload = key.workload,
        .operation = key.operation,
        .input = key.input,
        .weight = key.weight,
        .bias = key.bias,
        .ternary_input_a = key.ternary_input_a,
        .ternary_input_b = key.ternary_input_b,
        .persistent_output = key.persistent_output,
        .persistent_weight = key.persistent_weight};
    EXPECT_FALSE(registry::select_recipe(ttnn::MatmulRegistryMode::Off, facts).has_value());
    EXPECT_FALSE(registry::select_recipe(ttnn::MatmulRegistryMode::Shadow, facts).has_value());
    EXPECT_TRUE(registry::select_recipe(ttnn::MatmulRegistryMode::On, facts).has_value());
    facts.device.compute_grid_x += 1;
    facts.device.compute_grid_y += 1;
    ASSERT_EQ(registry::build_registry_key(facts)->device, facts.device);
    EXPECT_TRUE(registry::select_recipe(ttnn::MatmulRegistryMode::On, facts).has_value());
}

}  // namespace
