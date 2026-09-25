// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// The ControlPlane's link-health surface with no factory system descriptor configured, which is the state
// every caller is in today.
//
// This walks the whole surface deliberately. Each accessor forwards to a `LinkHealth` that does not exist
// without a factory descriptor, so a forwarder missing its null check is a segfault in the default
// configuration rather than an edge case, and only calling all of them finds it.

#include <gtest/gtest.h>

#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <tt-metalium/experimental/fabric/link_health.hpp>

#include "fabric_fixture.hpp"
#include "impl/context/metal_context.hpp"
#include "llrt/tt_cluster.hpp"

namespace tt::tt_fabric::fabric_router_tests {
namespace {

const ControlPlane& control_plane_without_factory_descriptor() {
    tt::tt_metal::MetalContext::instance().set_default_fabric_topology();
    tt::tt_metal::MetalContext::instance().set_fabric_config(
        tt::tt_fabric::FabricConfig::FABRIC_2D, tt::tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE);
    tt::tt_metal::MetalContext::instance().initialize_fabric_config();
    return tt::tt_metal::MetalContext::instance().get_control_plane();
}

TEST_F(ControlPlaneFixture, NoFactoryDescriptorReportsNone) {
    const auto& control_plane = control_plane_without_factory_descriptor();

    EXPECT_FALSE(control_plane.has_factory_descriptor());
    EXPECT_EQ(control_plane.get_link_health(), nullptr);
    EXPECT_FALSE(control_plane.fsd_rerouting_active());
}

// Healthy, not unknown. With no descriptor saying what should be cabled, there is no expectation for a
// cable to be missing from, so callers gating on this must not start failing when the feature ships.
TEST_F(ControlPlaneFixture, NoFactoryDescriptorReportsEveryLinkHealthy) {
    const auto& control_plane = control_plane_without_factory_descriptor();

    for (const auto& mesh_id : control_plane.get_mesh_graph().get_mesh_ids()) {
        for (const auto& [_, chip_id] : control_plane.get_mesh_graph().get_chip_ids(mesh_id)) {
            const FabricNodeId fabric_node_id(mesh_id, chip_id);
            for (chan_id_t chan = 0; chan < 16; ++chan) {
                EXPECT_TRUE(control_plane.is_link_healthy(fabric_node_id, chan))
                    << fabric_node_id << " chan " << static_cast<int>(chan);
            }
        }
    }
}

TEST_F(ControlPlaneFixture, NoFactoryDescriptorReportsNoDownedLinks) {
    const auto& control_plane = control_plane_without_factory_descriptor();

    EXPECT_TRUE(control_plane.get_downed_links().empty());
    EXPECT_TRUE(control_plane.get_locally_unhealthy_links().empty());
}

// The empty vectors must be the same object every call, since they are returned by reference and a
// per-call temporary would leave callers holding a dangling reference.
TEST_F(ControlPlaneFixture, NoFactoryDescriptorEmptyResultsAreStable) {
    const auto& control_plane = control_plane_without_factory_descriptor();

    EXPECT_EQ(&control_plane.get_downed_links(), &control_plane.get_downed_links());
    EXPECT_EQ(&control_plane.get_locally_unhealthy_links(), &control_plane.get_locally_unhealthy_links());
}

TEST_F(ControlPlaneFixture, RefreshWithoutAFactoryDescriptorIsANoOp) {
    tt::tt_metal::MetalContext::instance().set_default_fabric_topology();
    tt::tt_metal::MetalContext::instance().set_fabric_config(
        tt::tt_fabric::FabricConfig::FABRIC_2D, tt::tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE);
    tt::tt_metal::MetalContext::instance().initialize_fabric_config();
    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();

    control_plane.refresh_connectivity_diff();
    control_plane.refresh_connectivity_diff();

    EXPECT_FALSE(control_plane.has_factory_descriptor());
    EXPECT_TRUE(control_plane.get_downed_links().empty());
}

// The factory-descriptor path. Gated on the environment because the descriptor path is read from
// RTOptions when MetalContext is first built, so it cannot be set from inside a test; the driver runs this
// with TT_METAL_FACTORY_SYSTEM_DESCRIPTOR_PATH and a mock cluster descriptor for one of its hosts.
class FactoryDescriptorControlPlaneFixture : public ControlPlaneFixture {
protected:
    void SetUp() override {
        if (getenv("TT_METAL_FACTORY_SYSTEM_DESCRIPTOR_PATH") == nullptr) {
            GTEST_SKIP() << "needs TT_METAL_FACTORY_SYSTEM_DESCRIPTOR_PATH";
        }
        ControlPlaneFixture::SetUp();
    }
};

// A descriptor that agrees with the hardware produces no downed links. This is the case that has to be
// silent: if ingesting a matching descriptor reported holes, the feature would be unusable on a healthy
// machine.
TEST_F(FactoryDescriptorControlPlaneFixture, AMatchingDescriptorReportsNoDownedLinks) {
    const auto& control_plane = control_plane_without_factory_descriptor();

    ASSERT_TRUE(control_plane.has_factory_descriptor());
    ASSERT_NE(control_plane.get_link_health(), nullptr);

    EXPECT_FALSE(control_plane.fsd_rerouting_active());
    EXPECT_TRUE(control_plane.get_downed_links().empty());
    EXPECT_TRUE(control_plane.get_locally_unhealthy_links().empty());

    // The comparison actually ran, rather than finding nothing to compare.
    EXPECT_GT(control_plane.get_link_health()->fsd_expected_count(), 0u);
}

// The mesh is placed on the factory topology, so every node still resolves to a real chip. Solving on a
// descriptor that carries no UMD ids and forgetting to resolve them would leave these all zero.
TEST_F(FactoryDescriptorControlPlaneFixture, EveryMappedNodeResolvesToARealChip) {
    const auto& control_plane = control_plane_without_factory_descriptor();
    ASSERT_TRUE(control_plane.has_factory_descriptor());

    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    const auto local_chips = cluster.user_exposed_chip_ids();
    ASSERT_FALSE(local_chips.empty());

    std::size_t resolved = 0;
    for (const auto& mesh_id : control_plane.get_mesh_graph().get_mesh_ids()) {
        for (const auto& [_, chip_id] : control_plane.get_mesh_graph().get_chip_ids(mesh_id)) {
            const FabricNodeId fabric_node_id(mesh_id, chip_id);
            const auto chip = control_plane.try_get_physical_chip_id_from_fabric_node_id(fabric_node_id);
            if (!chip.has_value()) {
                continue;
            }
            EXPECT_TRUE(local_chips.contains(*chip)) << fabric_node_id << " resolved to unknown chip " << *chip;
            ++resolved;
        }
    }
    EXPECT_EQ(resolved, local_chips.size());
}

// Refreshing against an unchanged live descriptor must not invent holes.
TEST_F(FactoryDescriptorControlPlaneFixture, RefreshAgainstUnchangedLiveIsStable) {
    tt::tt_metal::MetalContext::instance().set_default_fabric_topology();
    tt::tt_metal::MetalContext::instance().set_fabric_config(
        tt::tt_fabric::FabricConfig::FABRIC_2D, tt::tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE);
    tt::tt_metal::MetalContext::instance().initialize_fabric_config();
    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    ASSERT_TRUE(control_plane.has_factory_descriptor());

    const auto before = control_plane.get_link_health()->fsd_expected_count();
    control_plane.refresh_connectivity_diff();

    EXPECT_EQ(control_plane.get_link_health()->fsd_expected_count(), before);
    EXPECT_TRUE(control_plane.get_downed_links().empty());
    EXPECT_FALSE(control_plane.fsd_rerouting_active());
}

}  // namespace
}  // namespace tt::tt_fabric::fabric_router_tests
