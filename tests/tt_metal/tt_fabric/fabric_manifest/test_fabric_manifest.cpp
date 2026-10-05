// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Checks the fabric manifest written during fabric init against the live fabric. Each check covers one part of the
// manifest and runs under 1D and 2D.

#include <gtest/gtest.h>
#include <fmt/format.h>
#include <nlohmann/json.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>

#include <filesystem>
#include <fstream>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "fabric_fixture.hpp"
#include "impl/context/metal_context.hpp"
#include "llrt/hal.hpp"
#include "tt_metal/fabric/builder/fabric_builder_config.hpp"
#include "tt_metal/fabric/builder/fabric_edge_capability.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_names.hpp"

namespace tt::tt_fabric::fabric_router_tests {
namespace {

using json = nlohmann::json;

const ControlPlane& control_plane() { return tt::tt_metal::MetalContext::instance().get_control_plane(); }

// ============ Helpers ============

using manifest::chip_key;
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

// One router in the manifest, with what its key names.
struct RouterEntry {
    std::string path;        // e.g. "M0/C7/E0"
    std::string router_key;  // e.g. "E0"
    FabricNodeId node;
    ChipId physical_chip_id;
    chan_id_t eth_chan;  // ControlPlane's channel for the router's key
    const json* router;
};

// Every router on this manifest's local chips.
void index_routers(const json& manifest, std::vector<RouterEntry>& routers) {
    for (const auto& [m_key, mesh] : manifest.at("meshes").items()) {
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
                routers.push_back({path, r_key, node, physical_chip_id, *eth_chan, &router});
            }
        }
    }
}

// Brings up fabric with manifest generation on, then reads this rank's manifest and indexes its routers.
template <FabricConfig kFabricConfig>
class FabricManifestFixture : public BaseFabricFixture {
protected:
    static constexpr FabricConfig fabric_config = kFabricConfig;

    static void SetUpTestSuite() {
        auto& rtoptions = tt::tt_metal::MetalContext::instance().rtoptions();
        generate_manifest_before_ = rtoptions.get_generate_fabric_manifest();
        rtoptions.set_generate_fabric_manifest(true);
        suite_start_ = std::filesystem::file_time_type::clock::now();
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
        index_routers(manifest_, routers_);
    }

    // Used for restoring the generate_manifest runtime arg value to its initial value after the test.
    inline static bool generate_manifest_before_ = false;
    // When this suite started bringing up fabric. A manifest written before it is a stale one.
    inline static std::filesystem::file_time_type suite_start_;
    std::filesystem::path manifest_path_;
    json manifest_;
    std::vector<RouterEntry> routers_;
};

using Fabric1DManifestFixture = FabricManifestFixture<FabricConfig::FABRIC_1D>;
using Fabric2DManifestFixture = FabricManifestFixture<FabricConfig::FABRIC_2D>;

// ============ Checks ============

// The manifest has exactly the top-level blocks the writer emits, was written by this run, and the write left no
// temporary file. Values are checked against the fixture's own settings, not the getters the writer reads.
void check_top_level(
    const json& manifest,
    const std::filesystem::path& manifest_path,
    FabricConfig fabric_config,
    std::filesystem::file_time_type suite_start) {
    EXPECT_EQ(
        keys_of(manifest), (std::set<std::string>{"manifest_version", "kind", "run", "fabric_context", "meshes"}));
    EXPECT_EQ(manifest.at("manifest_version"), FABRIC_MANIFEST_VERSION);
    EXPECT_EQ(manifest.at("kind"), "fabric_manifest");
    EXPECT_GE(std::filesystem::last_write_time(manifest_path), suite_start);

    const auto& run = manifest.at("run");
    EXPECT_EQ(
        keys_of(run),
        (std::set<std::string>{
            "arch",
            "fabric_config",
            "reliability_mode",
            "tensix_config",
            "udm_mode",
            "host_rank",
            "mpi_rank",
            "world_size",
            "written_at"}));
    EXPECT_EQ(run.at("fabric_config"), lower_enum_name(fabric_config));
    EXPECT_EQ(run.at("arch"), lower_enum_name(BaseFabricFixture::arch_));
    EXPECT_FALSE(run.at("written_at").get<std::string>().empty());

    const bool is_2d = fabric_config == FabricConfig::FABRIC_2D;
    const auto& block = manifest.at("fabric_context");
    EXPECT_EQ(
        keys_of(block),
        (std::set<std::string>{
            "topology",
            "is_2d_routing",
            "packet_header_size_bytes",
            "max_payload_size_bytes",
            "channel_buffer_size_bytes",
            is_2d ? "routing_2d_route_buffer_size" : "routing_1d_extension_words"}));
    EXPECT_EQ(block.at("is_2d_routing"), is_2d);

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
        EXPECT_EQ(
            keys_of(mesh),
            (shape.dims() == 2 ? std::set<std::string>{"shape", "torus", "express_routing", "chips"}
                               : std::set<std::string>{"shape", "express_routing", "chips"}));
        ASSERT_EQ(mesh.at("shape").size(), shape.dims());
        for (size_t dim = 0; dim < shape.dims(); ++dim) {
            EXPECT_EQ(mesh.at("shape").at(dim), shape[dim]);
        }
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
                EXPECT_EQ(
                    keys_of(chip), (std::set<std::string>{"mesh_coord", "physical_chip_id", "asic_id", "is_local"}));
                EXPECT_TRUE(chip.at("physical_chip_id").is_null());
                EXPECT_TRUE(chip.at("asic_id").is_null());
                continue;
            }

            EXPECT_EQ(
                keys_of(chip),
                (std::set<std::string>{
                    "mesh_coord", "physical_chip_id", "asic_id", "is_local", "z_port_role", "routers"}));
            EXPECT_EQ(chip.at("physical_chip_id"), *physical_chip_id);
            EXPECT_EQ(
                chip.at("asic_id"), fmt::format("0x{:016x}", *control_plane().get_asic_id_from_fabric_node_id(node)));
            // Read from the neighbor graph, not from the builder's per-chip facts the writer uses.
            EXPECT_EQ(chip.at("z_port_role"), lower_enum_name(z_port_role(control_plane(), node)));
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

// Every router has exactly the blocks the writer emits.
void check_router_blocks(const std::vector<RouterEntry>& routers) {
    for (const auto& entry : routers) {
        SCOPED_TRACE(entry.path);
        EXPECT_EQ(keys_of(*entry.router), (std::set<std::string>{"identity", "link", "shape"}));
    }
}

// Identity names the router's channel and cores. Its mesh and chip are the path.
void check_router_identity(const std::vector<RouterEntry>& routers) {
    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    for (const auto& entry : routers) {
        SCOPED_TRACE(entry.path);
        const auto& identity = entry.router->at("identity");
        EXPECT_EQ(keys_of(identity), (std::set<std::string>{"eth_chan", "logical_core", "virtual_core"}));
        EXPECT_EQ(identity.at("eth_chan"), entry.eth_chan);

        const auto logical_core =
            cluster.get_soc_desc(entry.physical_chip_id).get_eth_core_for_channel(entry.eth_chan, CoordSystem::LOGICAL);
        EXPECT_EQ(identity.at("logical_core"), json::array({logical_core.x, logical_core.y}));
        const auto virtual_core = cluster.get_virtual_coordinate_from_logical_coordinates(
            entry.physical_chip_id, tt::tt_metal::CoreCoord(logical_core.x, logical_core.y), CoreType::ETH);
        EXPECT_EQ(identity.at("virtual_core"), json::array({virtual_core.x, virtual_core.y}));
    }
}

// Link agrees with ControlPlane: the edge capability follows from where the peer is and which way the router
// faces, cross-host is ControlPlane's, and a wrap link spans a torus axis. Direction and plane are the key, which
// RoutersMatchActiveChannels checks.
void check_router_link(const json& manifest, const std::vector<RouterEntry>& routers) {
    for (const auto& entry : routers) {
        SCOPED_TRACE(entry.path);
        const auto& link = entry.router->at("link");
        EXPECT_EQ(
            keys_of(link), (std::set<std::string>{"edge_capability", "peer", "cross_host", "wrap", "dispatch_link"}));
        EXPECT_TRUE(link.at("dispatch_link").is_boolean());
        EXPECT_EQ(
            link.at("cross_host"), control_plane().is_cross_host_eth_link(entry.physical_chip_id, entry.eth_chan));

        const bool faces_z = entry.router_key.front() == 'Z';
        const auto peer = control_plane().try_get_connected_mesh_chip_chan_ids(entry.node, entry.eth_chan);
        if (peer.has_value()) {
            // A same-mesh Z edge is always an express chord: classification rejects one on a mesh without express.
            const EdgeCapability expected = peer->first.mesh_id != entry.node.mesh_id ? EdgeCapability::INTERMESH
                                            : faces_z ? EdgeCapability::INTRAMESH_EXPRESS
                                                      : EdgeCapability::INTRAMESH_CARDINAL;
            EXPECT_EQ(link.at("edge_capability"), lower_enum_name(expected));
        }

        // The peer can be null even with a cable: the far channel may be trimmed from the peer's planes.
        if (!peer.has_value()) {
            EXPECT_TRUE(link.at("peer").is_null());
        } else if (!link.at("peer").is_null()) {
            const auto peer_chip = fmt::format("{}/{}/", mesh_key(peer->first.mesh_id), chip_key(peer->first.chip_id));
            EXPECT_TRUE(link.at("peer").get<std::string>().starts_with(peer_chip)) << link.at("peer");
        }

        if (link.at("wrap").get<bool>()) {
            EXPECT_TRUE(peer.has_value());
            EXPECT_FALSE(faces_z);
            const auto& mesh = manifest.at("meshes").at(mesh_key(entry.node.mesh_id));
            ASSERT_TRUE(mesh.contains("torus"));
            const bool is_east_west = entry.router_key.front() == 'E' || entry.router_key.front() == 'W';
            EXPECT_TRUE(mesh.at("torus").at(is_east_west ? "x" : "y").get<bool>());
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
        const auto peer_path = link.at("peer").get<std::string>();
        SCOPED_TRACE(fmt::format("{} -> {}", entry.path, peer_path));
        // Both ends are on the same routing plane: the keys differ only in direction, e.g. E1 and W1.
        EXPECT_EQ(peer_path.substr(peer_path.rfind('/') + 2), entry.router_key.substr(1));
        const json* peer = find_router(manifest, peer_path);
        if (peer == nullptr) {
            continue;  // The peer is in another host's manifest.
        }
        const auto& peer_link = peer->at("link");
        EXPECT_EQ(peer_link.at("peer"), entry.path);
        EXPECT_EQ(peer_link.at("edge_capability"), link.at("edge_capability"));
        EXPECT_EQ(peer_link.at("cross_host"), link.at("cross_host"));
        EXPECT_EQ(peer_link.at("wrap"), link.at("wrap"));
    }
}

// Shape stays within what the hardware and the builder's limits allow. Its exact counts are checked when the
// router is collected, against the compile-time arguments the kernel receives.
void check_router_shape(const std::vector<RouterEntry>& routers) {
    const auto& hal = tt::tt_metal::MetalContext::instance().hal();
    const uint32_t max_eriscs = hal.get_num_risc_processors(tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH);
    const bool has_trimming_profile = tt::tt_metal::MetalContext::instance().rtoptions().has_fabric_trimming_profile();
    for (const auto& entry : routers) {
        SCOPED_TRACE(entry.path);
        const auto& shape = entry.router->at("shape");
        EXPECT_EQ(
            keys_of(shape),
            (std::set<std::string>{
                "num_vcs",
                "senders_per_vc",
                "receivers_per_vc",
                "num_active_eriscs",
                "channel_trimming_overrides_applied",
                "vc0_bubble_flow_control"}));

        const auto num_vcs = shape.at("num_vcs").get<uint32_t>();
        EXPECT_GE(num_vcs, 1u);
        EXPECT_LE(num_vcs, builder_config::MAX_NUM_VCS);
        const auto& senders = shape.at("senders_per_vc");
        const auto& receivers = shape.at("receivers_per_vc");
        ASSERT_EQ(senders.size(), builder_config::MAX_NUM_VCS);
        ASSERT_EQ(receivers.size(), builder_config::MAX_NUM_VCS);
        EXPECT_GE(senders.at(0).get<uint32_t>(), 1u);
        EXPECT_GE(receivers.at(0).get<uint32_t>(), 1u);
        for (uint32_t vc = num_vcs; vc < builder_config::MAX_NUM_VCS; ++vc) {
            EXPECT_EQ(senders.at(vc), 0) << "vc" << vc;
            EXPECT_EQ(receivers.at(vc), 0) << "vc" << vc;
        }

        const auto num_active_eriscs = shape.at("num_active_eriscs").get<uint32_t>();
        EXPECT_GE(num_active_eriscs, 1u);
        EXPECT_LE(num_active_eriscs, max_eriscs);
        if (!has_trimming_profile) {
            EXPECT_EQ(shape.at("channel_trimming_overrides_applied"), false);
        }
        EXPECT_TRUE(shape.at("vc0_bubble_flow_control").is_boolean());
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

    EXPECT_EQ(lower_enum_name(FabricConfig::FABRIC_1D), "fabric_1d");
    EXPECT_EQ(lower_enum_name(FabricConfig::FABRIC_2D), "fabric_2d");
}

TEST_F(Fabric1DManifestFixture, TopLevel) { check_top_level(manifest_, manifest_path_, fabric_config, suite_start_); }
TEST_F(Fabric2DManifestFixture, TopLevel) { check_top_level(manifest_, manifest_path_, fabric_config, suite_start_); }

TEST_F(Fabric1DManifestFixture, Meshes) { check_meshes(manifest_); }
TEST_F(Fabric2DManifestFixture, Meshes) { check_meshes(manifest_); }

TEST_F(Fabric1DManifestFixture, Chips) { check_chips(manifest_); }
TEST_F(Fabric2DManifestFixture, Chips) { check_chips(manifest_); }

TEST_F(Fabric1DManifestFixture, RoutersMatchActiveChannels) { check_routers_match_active_channels(manifest_); }
TEST_F(Fabric2DManifestFixture, RoutersMatchActiveChannels) { check_routers_match_active_channels(manifest_); }

TEST_F(Fabric1DManifestFixture, RouterBlocks) { check_router_blocks(routers_); }
TEST_F(Fabric2DManifestFixture, RouterBlocks) { check_router_blocks(routers_); }

TEST_F(Fabric1DManifestFixture, RouterIdentity) { check_router_identity(routers_); }
TEST_F(Fabric2DManifestFixture, RouterIdentity) { check_router_identity(routers_); }

TEST_F(Fabric1DManifestFixture, RouterLink) { check_router_link(manifest_, routers_); }
TEST_F(Fabric2DManifestFixture, RouterLink) { check_router_link(manifest_, routers_); }

TEST_F(Fabric1DManifestFixture, PeersAreSymmetric) { check_peers_are_symmetric(manifest_, routers_); }
TEST_F(Fabric2DManifestFixture, PeersAreSymmetric) { check_peers_are_symmetric(manifest_, routers_); }

TEST_F(Fabric1DManifestFixture, RouterShape) { check_router_shape(routers_); }
TEST_F(Fabric2DManifestFixture, RouterShape) { check_router_shape(routers_); }

}  // namespace tt::tt_fabric::fabric_router_tests
