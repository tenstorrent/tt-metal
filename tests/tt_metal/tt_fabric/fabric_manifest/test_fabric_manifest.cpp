// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Checks the fabric manifest written during fabric init against the live fabric: ControlPlane, the cluster and
// what the router builders published. Each check covers one part of the manifest and runs under 1D and 2D.

#include <fmt/format.h>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt-metalium/hal.hpp>

#include "fabric_fixture.hpp"
#include "hostdevcommon/fabric_common.h"
#include "impl/context/metal_context.hpp"
#include "tt_metal/fabric/builder/fabric_edge_capability.hpp"
#include "tt_metal/fabric/builder/fabric_manifest_model.hpp"
#include "tt_metal/fabric/fabric_builder_context.hpp"
#include "tt_metal/fabric/fabric_context.hpp"
#include "tt_metal/fabric/fabric_manifest.hpp"
#include "tt_metal/fabric/fabric_manifest_names.hpp"

namespace tt::tt_fabric::fabric_router_tests {
namespace {

using json = nlohmann::json;

const ControlPlane& control_plane() { return tt::tt_metal::MetalContext::instance().get_control_plane(); }

const FabricBuilderContext& builder_context() { return control_plane().get_fabric_context().get_builder_context(); }

// ============ Helpers ============

using manifest::chip_key;
using manifest::enum_name;
using manifest::lower_enum_name;
using manifest::mesh_key;
using manifest::router_key;

std::set<std::string> keys_of(const json& object) {
    std::set<std::string> keys;
    for (const auto& [key, _] : object.items()) {
        keys.insert(key);
    }
    return keys;
}

// The active channel ControlPlane keys as `key` on `node`, e.g. the plane-0 east channel for "E0".
std::optional<chan_id_t> channel_for_key(FabricNodeId node, const std::string& key) {
    for (const auto& [chan, direction] : control_plane().get_active_fabric_eth_channels(node)) {
        if (router_key(direction, control_plane().get_routing_plane_id(node, chan)) == key) {
            return chan;
        }
    }
    return std::nullopt;
}

// The router at a manifest path such as "M0/C7/E0", or null when this manifest does not have it (the peer
// is on another host).
const json* find_router(const json& manifest, const std::string& path) {
    const auto first = path.find('/');
    const auto second = path.find('/', first + 1);
    const json::json_pointer pointer(fmt::format(
        "/meshes/{}/chips/{}/routers/{}",
        path.substr(0, first),
        path.substr(first + 1, second - first - 1),
        path.substr(second + 1)));
    return manifest.contains(pointer) ? &manifest.at(pointer) : nullptr;
}

// ============ Fixture ============

// One router in the manifest, with what its key names and what the builder published for it.
struct RouterEntry {
    std::string path;  // e.g. "M0/C7/E0"
    std::string key;   // e.g. "E0"
    FabricNodeId node;
    ChipId physical_chip_id;
    chan_id_t eth_chan;  // ControlPlane's channel for the router's key
    const json* router;
    const manifest::Router* published;
};

// Brings up fabric with manifest generation on, then reads this rank's manifest and indexes its routers.
template <FabricConfig kFabricConfig>
class FabricManifestFixture : public BaseFabricFixture {
protected:
    static constexpr FabricConfig fabric_config = kFabricConfig;

    static void SetUpTestSuite() {
        auto& rtoptions = tt::tt_metal::MetalContext::instance().rtoptions();
        generate_manifest_before_ = rtoptions.get_generate_fabric_manifest();
        rtoptions.set_generate_fabric_manifest(true);
        DoSetUpTestSuite(kFabricConfig);
    }

    static void TearDownTestSuite() {
        DoTearDownTestSuite();
        tt::tt_metal::MetalContext::instance().rtoptions().set_generate_fabric_manifest(generate_manifest_before_);
    }

    void SetUp() override {
        BaseFabricFixture::SetUp();
        if (IsSkipped()) {
            return;
        }

        manifest_path_ = fabric_manifest_path(tt::tt_metal::MetalContext::instance().rtoptions());
        std::ifstream stream(manifest_path_);
        ASSERT_TRUE(stream.is_open()) << manifest_path_;
        manifest_ = json::parse(stream);

        for (const auto& [m_key, mesh] : manifest_.at("meshes").items()) {
            for (const auto& [c_key, chip] : mesh.at("chips").items()) {
                if (!chip.at("is_local").get<bool>()) {
                    continue;
                }
                const FabricNodeId node(
                    MeshId{static_cast<uint32_t>(std::stoul(m_key.substr(1)))},
                    static_cast<uint32_t>(std::stoul(c_key.substr(1))));
                const auto physical_chip_id = chip.at("physical_chip_id").get<ChipId>();
                for (const auto& [r_key, router] : chip.at("routers").items()) {
                    const std::string path = fmt::format("{}/{}/{}", m_key, c_key, r_key);
                    const auto eth_chan = channel_for_key(node, r_key);
                    ASSERT_TRUE(eth_chan.has_value()) << path << " is not one of ControlPlane's active channels";
                    const auto& published_routers = builder_context().get_manifest_chip(physical_chip_id).routers;
                    const auto published = std::ranges::find(
                        published_routers, *eth_chan, [](const manifest::Router& r) { return r.identity.eth_chan; });
                    ASSERT_NE(published, published_routers.end()) << path << " has no published router";
                    routers_.push_back({path, r_key, node, physical_chip_id, *eth_chan, &router, &*published});
                }
            }
        }
    }

    // Used for restoring the generate_manifest runtime arg value to its initial value after the test.
    inline static bool generate_manifest_before_ = false;
    std::filesystem::path manifest_path_;
    json manifest_;
    std::vector<RouterEntry> routers_;
};

using Manifest1DFixture = FabricManifestFixture<FabricConfig::FABRIC_1D>;
using Manifest2DFixture = FabricManifestFixture<FabricConfig::FABRIC_2D>;

// ============ Checks ============

// The manifest has exactly the top-level blocks the writer emits, and the write left no temporary file.
void check_top_level(const json& manifest, const std::filesystem::path& manifest_path, FabricConfig fabric_config) {
    EXPECT_EQ(
        keys_of(manifest), (std::set<std::string>{"manifest_version", "kind", "run", "fabric_context", "meshes"}));
    EXPECT_EQ(manifest.at("manifest_version"), FABRIC_MANIFEST_VERSION);
    EXPECT_EQ(manifest.at("kind"), "fabric_manifest");

    const auto& run = manifest.at("run");
    EXPECT_EQ(run.at("fabric_config"), enum_name(fabric_config));
    EXPECT_EQ(run.at("arch"), enum_name(tt::tt_metal::MetalContext::instance().get_cluster().arch()));
    EXPECT_FALSE(run.at("written_at").get<std::string>().empty());

    const auto& fabric_context = control_plane().get_fabric_context();
    const auto& block = manifest.at("fabric_context");
    EXPECT_EQ(block.at("is_2d_routing"), fabric_context.is_2D_routing_enabled());
    EXPECT_EQ(block.at("channel_buffer_size_bytes"), fabric_context.get_fabric_channel_buffer_size_bytes());
    EXPECT_EQ(block.at("packet_header_size_bytes"), fabric_context.get_fabric_packet_header_size_bytes());
    EXPECT_EQ(block.at("max_payload_size_bytes"), fabric_context.get_fabric_max_payload_size_bytes());

    for (const auto& entry : std::filesystem::directory_iterator(manifest_path.parent_path())) {
        EXPECT_EQ(entry.path().string().find(".tmp."), std::string::npos) << entry.path();
    }
}

// Every mesh in the mesh graph appears under its M key, with its shape and express setting.
void check_meshes(const json& manifest) {
    const auto& mesh_graph = control_plane().get_mesh_graph();
    std::set<std::string> expected_keys;
    for (const auto& mesh_id : mesh_graph.get_all_mesh_ids()) {
        expected_keys.insert(mesh_key(mesh_id));
        SCOPED_TRACE(mesh_key(mesh_id));
        ASSERT_TRUE(manifest.at("meshes").contains(mesh_key(mesh_id)));
        const auto& mesh = manifest.at("meshes").at(mesh_key(mesh_id));

        const auto shape = mesh_graph.get_mesh_shape(mesh_id);
        ASSERT_EQ(mesh.at("shape").size(), shape.dims());
        for (size_t dim = 0; dim < shape.dims(); ++dim) {
            EXPECT_EQ(mesh.at("shape").at(dim), shape[dim]);
        }
        EXPECT_EQ(mesh.contains("torus"), shape.dims() == 2);
        EXPECT_EQ(mesh.at("express_routing"), control_plane().express_routing_enabled(mesh_id));
    }
    EXPECT_EQ(keys_of(manifest.at("meshes")), expected_keys);
}

// Every chip in each mesh appears under its C key. Local chips carry their device ids, Z-port role and routers;
// the others only say where they are, so the viewer can still draw the whole mesh.
void check_chips(const json& manifest) {
    const auto& mesh_graph = control_plane().get_mesh_graph();
    for (const auto& mesh_id : mesh_graph.get_all_mesh_ids()) {
        const auto& chips = manifest.at("meshes").at(mesh_key(mesh_id)).at("chips");
        std::set<std::string> expected_keys;
        for (const auto& [_, fabric_chip_id] : mesh_graph.get_chip_ids(mesh_id)) {
            const FabricNodeId node(mesh_id, fabric_chip_id);
            expected_keys.insert(chip_key(fabric_chip_id));
            SCOPED_TRACE(fmt::format("{}/{}", mesh_key(mesh_id), chip_key(fabric_chip_id)));
            ASSERT_TRUE(chips.contains(chip_key(fabric_chip_id)));
            const auto& chip = chips.at(chip_key(fabric_chip_id));

            const auto coord = mesh_graph.chip_to_coordinate(mesh_id, fabric_chip_id);
            ASSERT_EQ(chip.at("mesh_coord").size(), coord.dims());
            for (size_t dim = 0; dim < coord.dims(); ++dim) {
                EXPECT_EQ(chip.at("mesh_coord").at(dim), coord[dim]);
            }

            const auto physical_chip_id = control_plane().try_get_physical_chip_id_from_fabric_node_id(node);
            ASSERT_EQ(chip.at("is_local"), physical_chip_id.has_value());
            if (!physical_chip_id.has_value()) {
                EXPECT_TRUE(chip.at("physical_chip_id").is_null());
                EXPECT_TRUE(chip.at("asic_id").is_null());
                EXPECT_FALSE(chip.contains("z_port_role"));
                EXPECT_FALSE(chip.contains("routers"));
                continue;
            }

            EXPECT_EQ(chip.at("physical_chip_id"), *physical_chip_id);
            EXPECT_EQ(
                chip.at("asic_id"), fmt::format("0x{:016x}", *control_plane().get_asic_id_from_fabric_node_id(node)));
            // A chip that built no routers publishes nothing, and has no Z port in use.
            const auto expected_role = builder_context().has_manifest_chip(*physical_chip_id)
                                           ? builder_context().get_manifest_chip(*physical_chip_id).z_port_role
                                           : ZPortRole::NONE;
            EXPECT_EQ(chip.at("z_port_role"), lower_enum_name(expected_role));
        }
        EXPECT_EQ(keys_of(chips), expected_keys);
    }
}

// A local chip's routers are exactly ControlPlane's active fabric channels, each keyed by its direction and
// routing plane.
void check_routers_match_active_channels(const json& manifest) {
    for (const auto& mesh_id : control_plane().get_mesh_graph().get_all_mesh_ids()) {
        for (const auto& [_, fabric_chip_id] : control_plane().get_mesh_graph().get_chip_ids(mesh_id)) {
            const FabricNodeId node(mesh_id, fabric_chip_id);
            if (!control_plane().try_get_physical_chip_id_from_fabric_node_id(node).has_value()) {
                continue;
            }
            SCOPED_TRACE(fmt::format("{}/{}", mesh_key(mesh_id), chip_key(fabric_chip_id)));
            std::set<std::string> expected_keys;
            for (const auto& [chan, direction] : control_plane().get_active_fabric_eth_channels(node)) {
                expected_keys.insert(router_key(direction, control_plane().get_routing_plane_id(node, chan)));
            }
            const auto& chip = manifest.at("meshes").at(mesh_key(mesh_id)).at("chips").at(chip_key(fabric_chip_id));
            EXPECT_EQ(keys_of(chip.at("routers")), expected_keys);
        }
    }
}

// Identity names the router's node, channel, architecture and cores.
void check_router_identity(const std::vector<RouterEntry>& routers) {
    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    for (const auto& entry : routers) {
        SCOPED_TRACE(entry.path);
        const auto& identity = entry.router->at("identity");
        EXPECT_EQ(identity.at("mesh_id"), *entry.node.mesh_id);
        EXPECT_EQ(identity.at("chip_id"), entry.node.chip_id);
        EXPECT_EQ(identity.at("eth_chan"), entry.eth_chan);
        EXPECT_EQ(identity.at("arch"), lower_enum_name(tt::tt_metal::hal::get_arch()));

        const auto logical_core =
            cluster.get_soc_desc(entry.physical_chip_id).get_eth_core_for_channel(entry.eth_chan, CoordSystem::LOGICAL);
        EXPECT_EQ(identity.at("logical_core"), json::array({logical_core.x, logical_core.y}));
        const auto virtual_core = cluster.get_virtual_coordinate_from_logical_coordinates(
            entry.physical_chip_id, tt::tt_metal::CoreCoord(logical_core.x, logical_core.y), CoreType::ETH);
        EXPECT_EQ(identity.at("virtual_core"), json::array({virtual_core.x, virtual_core.y}));
    }
}

// Link agrees with ControlPlane (direction and plane, which the key names; cross-host; peer) and with what the
// builder published (edge capability, dispatch link).
void check_router_link(const std::vector<RouterEntry>& routers) {
    for (const auto& entry : routers) {
        SCOPED_TRACE(entry.path);
        const auto& link = entry.router->at("link");
        const auto& published = entry.published->link;
        EXPECT_EQ(link.at("direction"), entry.key.substr(0, 1));
        EXPECT_EQ(link.at("routing_plane"), control_plane().get_routing_plane_id(entry.node, entry.eth_chan));
        EXPECT_EQ(link.at("edge_capability"), lower_enum_name(published.edge_capability));
        EXPECT_EQ(link.at("dispatch_link"), published.dispatch_link);
        EXPECT_EQ(
            link.at("cross_host"), control_plane().is_cross_host_eth_link(entry.physical_chip_id, entry.eth_chan));

        // The peer can be null even with a cable: the far channel may be trimmed from the peer's planes.
        const auto peer = control_plane().try_get_connected_mesh_chip_chan_ids(entry.node, entry.eth_chan);
        if (!peer.has_value()) {
            EXPECT_TRUE(link.at("peer").is_null());
        } else if (!link.at("peer").is_null()) {
            const auto peer_chip = fmt::format("{}/{}/", mesh_key(peer->first.mesh_id), chip_key(peer->first.chip_id));
            EXPECT_TRUE(link.at("peer").get<std::string>().starts_with(peer_chip)) << link.at("peer");
        }
    }
}

// The two ends of a link, when both are in this manifest, name each other and agree on what the link is.
void check_peers_are_symmetric(const json& manifest, const std::vector<RouterEntry>& routers) {
    for (const auto& entry : routers) {
        const auto& link = entry.router->at("link");
        if (link.at("peer").is_null()) {
            continue;
        }
        SCOPED_TRACE(fmt::format("{} -> {}", entry.path, link.at("peer").get<std::string>()));
        const json* peer = find_router(manifest, link.at("peer").get<std::string>());
        if (peer == nullptr) {
            continue;  // The peer is in another host's manifest.
        }
        const auto& peer_link = peer->at("link");
        EXPECT_EQ(peer_link.at("peer"), entry.path);
        EXPECT_EQ(peer_link.at("routing_plane"), link.at("routing_plane"));
        EXPECT_EQ(peer_link.at("edge_capability"), link.at("edge_capability"));
        EXPECT_EQ(peer_link.at("cross_host"), link.at("cross_host"));
        EXPECT_EQ(peer_link.at("wrap"), link.at("wrap"));
    }
}

// Shape is what the builder published.
void check_router_shape(const std::vector<RouterEntry>& routers) {
    for (const auto& entry : routers) {
        SCOPED_TRACE(entry.path);
        const auto& shape = entry.router->at("shape");
        const auto& published = entry.published->shape;
        EXPECT_EQ(shape.at("num_vcs"), published.num_vcs);
        EXPECT_EQ(shape.at("senders_per_vc"), json(published.senders_per_vc));
        EXPECT_EQ(shape.at("receivers_per_vc"), json(published.receivers_per_vc));
        EXPECT_EQ(shape.at("num_active_eriscs"), published.num_active_eriscs);
        EXPECT_EQ(shape.at("channel_trimming_overrides_applied"), published.channel_trimming_overrides_applied);
        EXPECT_EQ(shape.at("vc0_bubble_flow_control"), published.vc0_bubble_flow_control);
    }
}

}  // namespace

// ============ Tests ============

// The spellings the manifest's readers depend on. The checks below build their expected values with the same
// helpers, so a spelling only fails here.
TEST(ManifestNames, Spellings) {
    EXPECT_EQ(mesh_key(MeshId{0}), "M0");
    EXPECT_EQ(chip_key(7), "C7");

    EXPECT_EQ(router_key(eth_chan_directions::EAST, 0), "E0");
    EXPECT_EQ(router_key(eth_chan_directions::WEST, 1), "W1");
    EXPECT_EQ(router_key(eth_chan_directions::NORTH, 2), "N2");
    EXPECT_EQ(router_key(eth_chan_directions::SOUTH, 3), "S3");
    EXPECT_EQ(router_key(eth_chan_directions::Z, 0), "Z0");

    EXPECT_EQ(lower_enum_name(EdgeCapability::INTRAMESH_CARDINAL), "intramesh_cardinal");
    EXPECT_EQ(lower_enum_name(EdgeCapability::INTRAMESH_EXPRESS), "intramesh_express");
    EXPECT_EQ(lower_enum_name(EdgeCapability::INTERMESH), "intermesh");

    EXPECT_EQ(lower_enum_name(ZPortRole::NONE), "none");
    EXPECT_EQ(lower_enum_name(ZPortRole::INTERMESH_BOUNDARY), "intermesh_boundary");
    EXPECT_EQ(lower_enum_name(ZPortRole::EXPRESS_CHORD), "express_chord");

    EXPECT_EQ(lower_enum_name(tt::ARCH::WORMHOLE_B0), "wormhole_b0");
    EXPECT_EQ(lower_enum_name(tt::ARCH::BLACKHOLE), "blackhole");

    EXPECT_EQ(enum_name(FabricConfig::FABRIC_1D), "FABRIC_1D");
    EXPECT_EQ(enum_name(FabricConfig::FABRIC_2D), "FABRIC_2D");
}

TEST_F(Manifest1DFixture, TopLevel) { check_top_level(manifest_, manifest_path_, fabric_config); }
TEST_F(Manifest2DFixture, TopLevel) { check_top_level(manifest_, manifest_path_, fabric_config); }

TEST_F(Manifest1DFixture, Meshes) { check_meshes(manifest_); }
TEST_F(Manifest2DFixture, Meshes) { check_meshes(manifest_); }

TEST_F(Manifest1DFixture, Chips) { check_chips(manifest_); }
TEST_F(Manifest2DFixture, Chips) { check_chips(manifest_); }

TEST_F(Manifest1DFixture, RoutersMatchActiveChannels) { check_routers_match_active_channels(manifest_); }
TEST_F(Manifest2DFixture, RoutersMatchActiveChannels) { check_routers_match_active_channels(manifest_); }

TEST_F(Manifest1DFixture, RouterIdentity) { check_router_identity(routers_); }
TEST_F(Manifest2DFixture, RouterIdentity) { check_router_identity(routers_); }

TEST_F(Manifest1DFixture, RouterLink) { check_router_link(routers_); }
TEST_F(Manifest2DFixture, RouterLink) { check_router_link(routers_); }

TEST_F(Manifest1DFixture, PeersAreSymmetric) { check_peers_are_symmetric(manifest_, routers_); }
TEST_F(Manifest2DFixture, PeersAreSymmetric) { check_peers_are_symmetric(manifest_, routers_); }

TEST_F(Manifest1DFixture, RouterShape) { check_router_shape(routers_); }
TEST_F(Manifest2DFixture, RouterShape) { check_router_shape(routers_); }

}  // namespace tt::tt_fabric::fabric_router_tests
