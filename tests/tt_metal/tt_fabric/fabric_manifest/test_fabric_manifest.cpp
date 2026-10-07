// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Checks the fabric manifest written during fabric init against the live fabric. Each check covers one part of the
// manifest and runs under 1D and 2D.

#include <gtest/gtest.h>
#include <enchantum/enchantum.hpp>
#include <fmt/format.h>
#include <nlohmann/json.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>
#include <tt_stl/overloaded.hpp>

#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "fabric_fixture.hpp"
#include "impl/context/metal_context.hpp"
#include "llrt/hal.hpp"
#include "tt_metal/fabric/builder/fabric_builder_config.hpp"
#include "tt_metal/fabric/builder/fabric_builder_helpers.hpp"
#include "tt_metal/fabric/builder/fabric_edge_capability.hpp"
#include "tt_metal/fabric/builder/fabric_stream_assignment.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_fields.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_model.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_names.hpp"

namespace tt::tt_fabric::fabric_router_tests {
namespace {

using json = nlohmann::json;

const ControlPlane& control_plane() { return tt::tt_metal::MetalContext::instance().get_control_plane(); }

// ============ Helpers ============

using manifest::chip_key;
using manifest::kind_name;
using manifest::lower_enum_name;
using manifest::mesh_key;
using manifest::router_key;
using manifest::schema_name;
using manifest::schema_name_of;

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
        keys_of(manifest),
        (std::set<std::string>{"manifest_version", "kind", "run", "fabric_context", "vocabulary", "meshes"}));
    EXPECT_EQ(keys_of(manifest.at("vocabulary")), (std::set<std::string>{"categories", "kinds"}));
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
            is_2d ? "routing_2d_route_buffer_size" : "routing_1d_extension_words",
            "multi_txq"}));
    EXPECT_EQ(block.at("is_2d_routing"), is_2d);
    EXPECT_TRUE(block.at("multi_txq").is_boolean());

    // Other ranks write their manifests into the same directory, and may still be writing.
    const auto temp_prefix = manifest_path.filename().string() + ".tmp.";
    for (const auto& entry : std::filesystem::directory_iterator(manifest_path.parent_path())) {
        EXPECT_FALSE(entry.path().filename().string().starts_with(temp_prefix)) << entry.path();
    }
}

bool is_local_mesh(MeshId mesh_id) {
    const auto local_mesh_ids = control_plane().get_local_mesh_id_bindings();
    return std::ranges::find(local_mesh_ids, mesh_id) != local_mesh_ids.end();
}

// Every mesh in the mesh graph appears under its M key, with its shape and express setting. Only this host's meshes
// have a credit transport; other hosts' meshes are described by their own manifests.
void check_meshes(const json& manifest) {
    const auto& mesh_graph = control_plane().get_mesh_graph();
    std::set<std::string> expected_keys;
    for (const auto& mesh_id : mesh_graph.get_all_mesh_ids()) {
        expected_keys.insert(mesh_key(mesh_id));
        SCOPED_TRACE(mesh_key(mesh_id));
        ASSERT_TRUE(manifest.at("meshes").contains(mesh_key(mesh_id)));
        const auto& mesh = manifest.at("meshes").at(mesh_key(mesh_id));

        const auto shape = mesh_graph.get_mesh_shape(mesh_id);
        std::set<std::string> expected_mesh_keys = {"shape", "express_routing", "chips"};
        if (shape.dims() == 2) {
            expected_mesh_keys.insert("torus");
        }
        if (is_local_mesh(mesh_id)) {
            expected_mesh_keys.insert("credit_transport");
        }
        EXPECT_EQ(keys_of(mesh), expected_mesh_keys);
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
                    "mesh_coord", "physical_chip_id", "asic_id", "is_local", "z_port_role", "local_sync", "routers"}));
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
        EXPECT_EQ(
            keys_of(*entry.router),
            (std::set<std::string>{
                "identity",
                "link",
                "shape",
                "credit_counters",
                "channels",
                "intra_chip_downstream_edges",
                "fields",
                "eriscs"}));
        EXPECT_EQ(keys_of(entry.router->at("channels")), (std::set<std::string>{"senders", "receivers"}));
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

// A local mesh lists the credit backing of every VC its routers send on, with a reason exactly when the VC is on L1
// counters. Each reason sits only on the VCs it applies to, and matches the fact it names: multi_txq on VC0 and VC1
// with two TX queues, express on VC1 with express routing, and no_completion_register on VC2 wherever VC2 exists.
void check_credit_transport(const json& manifest, const std::vector<RouterEntry>& routers) {
    const bool multi_txq = manifest.at("fabric_context").at("multi_txq").get<bool>();
    for (const auto& mesh_id : control_plane().get_mesh_graph().get_all_mesh_ids()) {
        if (!is_local_mesh(mesh_id)) {
            continue;
        }
        SCOPED_TRACE(mesh_key(mesh_id));
        const auto& mesh = manifest.at("meshes").at(mesh_key(mesh_id));
        const auto& transport = mesh.at("credit_transport");
        const bool express = mesh.at("express_routing").get<bool>();

        std::set<std::string> vcs_with_senders;
        for (const auto& entry : routers) {
            if (entry.node.mesh_id != mesh_id) {
                continue;
            }
            const auto& senders = entry.router->at("shape").at("senders_per_vc");
            for (uint32_t vc = 0; vc < senders.size(); ++vc) {
                if (senders.at(vc).get<uint32_t>() > 0) {
                    vcs_with_senders.insert(fmt::format("vc{}", vc));
                }
            }
        }
        const auto listed = keys_of(transport);
        EXPECT_TRUE(std::ranges::includes(listed, vcs_with_senders));
        std::set<std::string> all_vcs;
        for (uint32_t vc = 0; vc < builder_config::MAX_NUM_VCS; ++vc) {
            all_vcs.insert(fmt::format("vc{}", vc));
        }
        EXPECT_TRUE(std::ranges::includes(all_vcs, listed));

        for (uint32_t vc = 0; vc < builder_config::MAX_NUM_VCS; ++vc) {
            const auto vc_key = fmt::format("vc{}", vc);
            if (!transport.contains(vc_key)) {
                continue;
            }
            SCOPED_TRACE(vc_key);
            const auto& entry = transport.at(vc_key);
            EXPECT_EQ(keys_of(entry), (std::set<std::string>{"backing", "reasons"}));
            std::set<std::string> reasons;
            for (const auto& reason : entry.at("reasons")) {
                EXPECT_TRUE(reasons.insert(reason.get<std::string>()).second) << "repeated reason " << reason;
            }
            EXPECT_EQ(entry.at("backing"), reasons.empty() ? "stream_register" : "l1_counter");

            std::set<std::string> expected_reasons;
            if (multi_txq && vc <= 1) {
                expected_reasons.insert(lower_enum_name(L1CreditCounterReason::MULTI_TXQ));
            }
            if (express && vc == 1) {
                expected_reasons.insert(lower_enum_name(L1CreditCounterReason::EXPRESS));
            }
            if (vc == 2) {
                expected_reasons.insert(lower_enum_name(L1CreditCounterReason::NO_COMPLETION_REGISTER));
            }
            EXPECT_EQ(reasons, expected_reasons);
        }
    }
}

// The four counter arrays are u32 arrays of one size, back to back in the order the kernel sends them in, each long
// enough for every sender channel it is indexed by, and the host clears all of them or none.
void check_router_credit_counters(const std::vector<RouterEntry>& routers) {
    const std::array<std::pair<const char*, const char*>, 4> arrays = {{
        {"to_sender_ack", "own_sender_compact"},
        {"to_sender_completion", "own_sender_compact"},
        {"receiver_ack", "peer_sender_compact"},
        {"receiver_completion", "peer_sender_compact"},
    }};
    for (const auto& entry : routers) {
        SCOPED_TRACE(entry.path);
        const auto& counters = entry.router->at("credit_counters");
        EXPECT_EQ(
            keys_of(counters),
            (std::set<std::string>{"to_sender_ack", "to_sender_completion", "receiver_ack", "receiver_completion"}));

        uint32_t num_senders = 0;
        for (const auto& count : entry.router->at("shape").at("senders_per_vc")) {
            num_senders += count.get<uint32_t>();
        }

        std::optional<uint32_t> next_address;
        std::optional<uint32_t> first_size;
        std::optional<bool> first_cleared_by_host;
        for (const auto& [name, index_space] : arrays) {
            SCOPED_TRACE(name);
            const auto& array = counters.at(name);
            EXPECT_EQ(
                keys_of(array),
                (std::set<std::string>{
                    "address",
                    "size",
                    "num_elements",
                    "size_per_element",
                    "schema",
                    "cleared_by_host",
                    "index_space"}));
            EXPECT_EQ(array.at("index_space"), index_space);
            EXPECT_EQ(array.at("schema"), "u32");
            EXPECT_EQ(array.at("size_per_element"), sizeof(uint32_t));

            const auto address = array.at("address").get<uint32_t>();
            const auto size = array.at("size").get<uint32_t>();
            const auto num_elements = array.at("num_elements").get<uint32_t>();
            EXPECT_EQ(size, num_elements * sizeof(uint32_t));
            // The receiver arrays are indexed by the peer router's sender channels; both ends are built from the same
            // config.
            EXPECT_GE(num_elements, num_senders);
            if (next_address.has_value()) {
                EXPECT_EQ(address, *next_address);
            }
            next_address = address + size;
            EXPECT_EQ(size, first_size.value_or(size));
            first_size = size;
            const auto cleared_by_host = array.at("cleared_by_host").get<bool>();
            EXPECT_EQ(cleared_by_host, first_cleared_by_host.value_or(cleared_by_host));
            first_cleared_by_host = cleared_by_host;
        }
    }
}

const std::set<std::string> k_region_keys = {"address", "size", "schema", "cleared_by_host"};
const std::set<std::string> k_array_region_keys = {
    "address", "size", "num_elements", "size_per_element", "schema", "cleared_by_host"};

void expect_value_region(const json& region, const std::string& schema, uint32_t size) {
    EXPECT_EQ(keys_of(region), k_region_keys);
    EXPECT_EQ(region.at("schema"), schema);
    EXPECT_EQ(region.at("size"), size);
}

// Stream ids past the hardware's registers are the builder's "not allocated" sentinel.
void expect_stream(const json& stream, bool must_be_allocated) {
    EXPECT_EQ(keys_of(stream), (std::set<std::string>{"stream_id", "register", "schema"}));
    EXPECT_EQ(stream.at("register"), lower_enum_name(manifest::StreamRegister::BUF_SPACE_AVAILABLE));
    EXPECT_EQ(stream.at("schema"), "u32");
    if (must_be_allocated) {
        EXPECT_LT(stream.at("stream_id").get<uint32_t>(), StreamRegAssignments::num_eth_stream_registers);
    }
}

// On L1 counters, a credit is the sender's own element of the router's counter array, which is indexed by the
// router's sender compact index. On stream registers, it is a register.
void expect_credit(
    const json& credit,
    const json& router,
    manifest::CreditCounterArray array,
    bool uses_counters,
    uint32_t compact,
    bool serviced) {
    if (!uses_counters) {
        expect_stream(credit, serviced);
        return;
    }
    EXPECT_EQ(keys_of(credit), (std::set<std::string>{"array", "index"}));
    EXPECT_EQ(credit.at("array"), fmt::format("credit_counters/{}", lower_enum_name(array)));
    EXPECT_EQ(credit.at("index"), compact);
    const json::json_pointer pointer("/" + credit.at("array").get<std::string>());
    ASSERT_TRUE(router.contains(pointer)) << credit.at("array");
    EXPECT_LT(compact, router.at(pointer).at("num_elements").get<uint32_t>());
}

// Every object under `node` with an address and a size, as (address, size, path).
void collect_regions(
    const json& node, const std::string& path, std::vector<std::tuple<uint32_t, uint32_t, std::string>>& regions) {
    if (!node.is_object()) {
        return;
    }
    if (node.contains("address") && node.contains("size")) {
        regions.emplace_back(node.at("address").get<uint32_t>(), node.at("size").get<uint32_t>(), path);
        return;
    }
    for (const auto& [key, child] : node.items()) {
        collect_regions(child, path.empty() ? key : path + "/" + key, regions);
    }
}

// No two of a router's L1 regions overlap.
void check_router_regions_disjoint(const std::vector<RouterEntry>& routers) {
    for (const auto& entry : routers) {
        SCOPED_TRACE(entry.path);
        std::vector<std::tuple<uint32_t, uint32_t, std::string>> regions;
        collect_regions(*entry.router, "", regions);
        std::ranges::sort(regions);
        for (size_t i = 1; i < regions.size(); ++i) {
            const auto& [address, size, name] = regions[i - 1];
            const auto& [next_address, next_size, next_name] = regions[i];
            EXPECT_LE(address + size, next_address) << name << " overlaps " << next_name;
        }
    }
}

// The last component of a router path, e.g. "E0" for "M0/C7/E0".
std::string key_of_path(const std::string& path) { return path.substr(path.rfind('/') + 1); }

// ERISC names in ascending order, each one active. Returns whether any ERISC services the channel.
bool expect_serviced_by(const json& serviced_by, uint32_t num_active_eriscs) {
    std::optional<uint32_t> previous_risc;
    for (const auto& risc : serviced_by) {
        const auto name = risc.get<std::string>();
        EXPECT_TRUE(name.starts_with("erisc")) << name;
        const auto risc_id = static_cast<uint32_t>(std::stoul(name.substr(5)));
        EXPECT_LT(risc_id, num_active_eriscs);
        if (previous_risc.has_value()) {
            EXPECT_GT(risc_id, *previous_risc);
        }
        previous_risc = risc_id;
    }
    return !serviced_by.empty();
}

// A channel is active exactly when an ERISC runs it. Every VC0 channel is in a VC the kernel runs, only VC0
// senders other than the worker channel are carried by the mux, and only a router with a trimming profile has
// trimmed channels. Returns whether the channel is active.
bool expect_channel_status(
    const json& channel, const json& shape, bool is_sender, uint32_t vc, uint32_t ch, uint32_t num_active_eriscs) {
    const bool serviced = expect_serviced_by(channel.at("serviced_by"), num_active_eriscs);
    const auto status = channel.at("status").get<std::string>();
    EXPECT_EQ(status == lower_enum_name(manifest::ChannelStatus::ACTIVE), serviced) << status;
    if (status == lower_enum_name(manifest::ChannelStatus::VC_NOT_SERVICED)) {
        EXPECT_GE(vc, 1u);
    } else if (status == lower_enum_name(manifest::ChannelStatus::MUX)) {
        EXPECT_TRUE(is_sender);
        EXPECT_EQ(vc, 0u);
        EXPECT_NE(ch, 0u);
    } else if (status == lower_enum_name(manifest::ChannelStatus::TRIMMED)) {
        EXPECT_TRUE(shape.at("channel_trimming_overrides_applied").get<bool>());
    } else {
        EXPECT_EQ(status, lower_enum_name(manifest::ChannelStatus::ACTIVE));
    }
    return serviced;
}

// An active channel has slots. In mux mode the VC0 senders the mux carries have none.
void expect_ring_buffer(const json& ring, uint32_t channel_buffer_size, bool active) {
    EXPECT_EQ(keys_of(ring), k_array_region_keys);
    EXPECT_EQ(ring.at("schema"), "packet_ring");
    EXPECT_EQ(ring.at("size_per_element"), channel_buffer_size);
    if (active) {
        EXPECT_GE(ring.at("num_elements").get<uint32_t>(), 1u);
    }
    EXPECT_EQ(
        ring.at("size").get<uint32_t>(),
        ring.at("num_elements").get<uint32_t>() * ring.at("size_per_element").get<uint32_t>());
}

// A field is written as its table entry says: its category and kind, then a region, a stream register or a value.
void expect_field(const json& entry, const manifest::RouterField& field) {
    EXPECT_EQ(entry.at("category"), lower_enum_name(field.category.value));
    EXPECT_EQ(entry.at("kind"), kind_name(field.kind.value));
    const auto schema = [](const manifest::ElementType& element) { return schema_name(element.type, element.size); };
    std::set<std::string> keys = {"category", "kind"};
    std::visit(
        ttsl::overloaded{
            [&](const manifest::kind::L1Value& kind) {
                keys.insert(k_region_keys.begin(), k_region_keys.end());
                EXPECT_EQ(entry.at("schema"), schema(kind.element));
                EXPECT_EQ(entry.at("size"), kind.element.size);
            },
            [&](const manifest::kind::Stream& kind) {
                keys.insert({"stream_id", "register", "schema"});
                EXPECT_EQ(entry.at("register"), lower_enum_name(kind.reg));
                EXPECT_EQ(entry.at("schema"), schema(kind.element));
                const auto stream_id = entry.at("stream_id").get<uint32_t>();
                EXPECT_TRUE(
                    stream_id < StreamRegAssignments::num_eth_stream_registers || stream_id == k_unused_stream_id)
                    << stream_id;
            },
            [&](const manifest::kind::Number&) {
                keys.insert("value");
                EXPECT_TRUE(entry.at("value").is_number_unsigned());
            },
            [&](const manifest::kind::Flag&) {
                keys.insert("value");
                EXPECT_TRUE(entry.at("value").is_boolean());
            },
            [&](const manifest::kind::Enum& kind) {
                keys.insert({"value", "schema"});
                EXPECT_EQ(entry.at("schema"), schema(kind.element));
                EXPECT_TRUE(kind.is_enumerator(entry.at("value").get<uint32_t>())) << entry.at("value");
            },
        },
        field.kind.value);
    EXPECT_EQ(keys_of(entry), keys);
}

// Whether `field`'s argument is emitted on this fabric: a 2D-only argument is not emitted in 1D.
bool is_emitted(const manifest::RouterField& field, bool is_2d) {
    const auto* arg = std::get_if<manifest::NamedArg>(&field.source);
    return is_2d || arg == nullptr || arg->emitted != manifest::Emitted::FABRIC_2D;
}

// `fields` holds exactly `table`'s fields emitted on this fabric, each written as its entry says, with a category and
// kind the manifest's vocabulary lists.
void expect_fields(
    const json& fields, std::span<const manifest::RouterField> table, const json& vocabulary, bool is_2d) {
    const auto listed = [&](const char* list, const json& name) {
        return std::ranges::find(vocabulary.at(list), name) != vocabulary.at(list).end();
    };
    std::set<std::string> expected_keys;
    for (const auto& field : table) {
        if (!is_emitted(field, is_2d)) {
            continue;
        }
        const std::string key(field.key.value);
        expected_keys.insert(key);
        if (fields.contains(key)) {
            SCOPED_TRACE(key);
            expect_field(fields.at(key), field);
            EXPECT_TRUE(listed("categories", fields.at(key).at("category")));
            EXPECT_TRUE(listed("kinds", fields.at(key).at("kind")));
        }
    }
    EXPECT_EQ(keys_of(fields), expected_keys);
}

// A router's sender channels, over its shape. Each one's producer is the chip's worker on a VC's first channel (the
// tensix mux on VC0 in mux mode), or a different router on the same chip and routing plane, feeding at most one
// channel per VC; in 1D a router and its producer feed each other. Nothing feeds a channel the mux carries. Credits
// are on the backing the mesh's credit transport names, and only VC0 with bubble flow control gets first-level acks.
void check_router_senders(const json& manifest, const std::vector<RouterEntry>& routers) {
    const auto& fabric_context = manifest.at("fabric_context");
    const bool is_2d = fabric_context.at("is_2d_routing").get<bool>();
    const auto channel_buffer_size = fabric_context.at("channel_buffer_size_bytes").get<uint32_t>();

    for (const auto& entry : routers) {
        SCOPED_TRACE(entry.path);
        const auto& router = *entry.router;
        const auto& shape = router.at("shape");
        const auto& senders = router.at("channels").at("senders");
        const auto& transport = manifest.at("meshes").at(mesh_key(entry.node.mesh_id)).at("credit_transport");
        const auto num_active_eriscs = shape.at("num_active_eriscs").get<uint32_t>();
        const bool vc0_bubble_flow_control = shape.at("vc0_bubble_flow_control").get<bool>();
        const auto chip_prefix = entry.path.substr(0, entry.path.rfind('/') + 1);

        std::set<std::string> expected_vc_keys;
        uint32_t compact = 0;
        for (uint32_t vc = 0; vc < builder_config::MAX_NUM_VCS; ++vc) {
            const auto count = shape.at("senders_per_vc").at(vc).get<uint32_t>();
            if (count == 0) {
                continue;
            }
            const auto vc_key = fmt::format("vc{}", vc);
            expected_vc_keys.insert(vc_key);
            ASSERT_TRUE(senders.contains(vc_key));
            ASSERT_EQ(senders.at(vc_key).size(), count);
            const bool uses_counters = transport.at(vc_key).at("backing") == "l1_counter";

            std::set<std::string> producers;
            for (uint32_t ch = 0; ch < count; ++ch, ++compact) {
                const auto ch_key = fmt::format("ch{}", ch);
                SCOPED_TRACE(fmt::format("senders/{}/{}", vc_key, ch_key));
                ASSERT_TRUE(senders.at(vc_key).contains(ch_key));
                const auto& sender = senders.at(vc_key).at(ch_key);
                EXPECT_EQ(
                    keys_of(sender),
                    (std::set<std::string>{"status", "serviced_by", "producer", "ring_buffer", "credits", "fields"}));

                const bool serviced = expect_channel_status(sender, shape, true, vc, ch, num_active_eriscs);

                const auto& producer = sender.at("producer");
                if (sender.at("status") == lower_enum_name(manifest::ChannelStatus::MUX)) {
                    EXPECT_TRUE(producer.is_null()) << producer;
                }
                if (producer == "worker") {
                    EXPECT_EQ(ch, 0u);
                    EXPECT_TRUE(vc == 0 || vc == 2);
                } else if (producer == "tensix_mux") {
                    EXPECT_EQ(ch, 0u);
                    EXPECT_EQ(vc, 0u);
                } else if (!producer.is_null()) {
                    const auto producer_path = producer.get<std::string>();
                    EXPECT_TRUE(producer_path.starts_with(chip_prefix)) << producer_path;
                    EXPECT_NE(producer_path, entry.path);
                    EXPECT_EQ(key_of_path(producer_path).substr(1), entry.router_key.substr(1)) << producer_path;
                    EXPECT_TRUE(producers.insert(producer_path).second) << producer_path << " feeds two channels";
                    const json* producer_router = find_router(manifest, producer_path);
                    ASSERT_NE(producer_router, nullptr) << producer_path;
                    if (!is_2d) {
                        const json::json_pointer pointer(
                            fmt::format("/channels/senders/{}/{}/producer", vc_key, ch_key));
                        ASSERT_TRUE(producer_router->contains(pointer)) << producer_path;
                        EXPECT_EQ(producer_router->at(pointer), entry.path);
                    }
                }

                expect_ring_buffer(sender.at("ring_buffer"), channel_buffer_size, serviced);
                expect_fields(sender.at("fields"), manifest::k_sender_channel_fields, manifest.at("vocabulary"), is_2d);

                const auto& credits = sender.at("credits");
                const bool acked = vc == 0 && vc0_bubble_flow_control;
                EXPECT_EQ(
                    keys_of(credits),
                    acked ? (std::set<std::string>{"acked", "completed"}) : (std::set<std::string>{"completed"}));
                if (acked) {
                    expect_credit(
                        credits.at("acked"),
                        router,
                        manifest::CreditCounterArray::TO_SENDER_ACK,
                        uses_counters,
                        compact,
                        serviced);
                }
                expect_credit(
                    credits.at("completed"),
                    router,
                    manifest::CreditCounterArray::TO_SENDER_COMPLETION,
                    uses_counters,
                    compact,
                    serviced);
            }
        }
        EXPECT_EQ(keys_of(senders), expected_vc_keys);
    }
}

// A router's receiver channels, over its shape. A receiver forwards on its own VC, except that VC0's may cross over
// to VC1, and VC2's forwards on none.
void check_router_receivers(const json& manifest, const std::vector<RouterEntry>& routers) {
    const auto& fabric_context = manifest.at("fabric_context");
    const bool is_2d = fabric_context.at("is_2d_routing").get<bool>();
    const auto channel_buffer_size = fabric_context.at("channel_buffer_size_bytes").get<uint32_t>();

    for (const auto& entry : routers) {
        SCOPED_TRACE(entry.path);
        const auto& router = *entry.router;
        const auto& shape = router.at("shape");
        const auto& receivers = router.at("channels").at("receivers");
        const auto num_active_eriscs = shape.at("num_active_eriscs").get<uint32_t>();

        std::set<std::string> expected_vc_keys;
        for (uint32_t vc = 0; vc < builder_config::MAX_NUM_VCS; ++vc) {
            const auto count = shape.at("receivers_per_vc").at(vc).get<uint32_t>();
            if (count == 0) {
                continue;
            }
            const auto vc_key = fmt::format("vc{}", vc);
            expected_vc_keys.insert(vc_key);
            ASSERT_TRUE(receivers.contains(vc_key));
            ASSERT_EQ(receivers.at(vc_key).size(), count);

            for (uint32_t ch = 0; ch < count; ++ch) {
                const auto ch_key = fmt::format("ch{}", ch);
                SCOPED_TRACE(fmt::format("receivers/{}/{}", vc_key, ch_key));
                ASSERT_TRUE(receivers.at(vc_key).contains(ch_key));
                const auto& receiver = receivers.at(vc_key).at(ch_key);
                EXPECT_EQ(
                    keys_of(receiver),
                    (std::set<std::string>{"status", "serviced_by", "forwards_on", "ring_buffer", "fields"}));

                const bool serviced = expect_channel_status(receiver, shape, false, vc, ch, num_active_eriscs);

                const auto& forwards_on = receiver.at("forwards_on");
                if (!serviced || vc == 2) {
                    EXPECT_TRUE(forwards_on.is_null()) << forwards_on;
                } else if (!forwards_on.is_null()) {
                    const auto forward_vc = forwards_on.get<std::string>();
                    if (vc == 0) {
                        EXPECT_TRUE(forward_vc == "vc0" || forward_vc == "vc1") << forward_vc;
                    } else {
                        EXPECT_EQ(forward_vc, vc_key);
                    }
                }

                expect_ring_buffer(receiver.at("ring_buffer"), channel_buffer_size, serviced);
                expect_fields(
                    receiver.at("fields"), manifest::k_receiver_channel_fields, manifest.at("vocabulary"), is_2d);
            }
        }
        EXPECT_EQ(keys_of(receivers), expected_vc_keys);
    }
}

// A router's edges, on VC0 and VC1 only, and only VC0 in 1D. In 2D, edge n goes to the sibling at compact index
// n - 1 among the router's other directions; 1D has only edge 1. Each lands on a sender channel, on the edge's VC, of
// a different router on the same chip and plane: one whose producer is this router, or, through the tensix mux, one
// the mux carries. Its free-slots register is allocated when a serviced receiver forwards on its VC.
void check_router_edges(const json& manifest, const std::vector<RouterEntry>& routers) {
    const bool is_2d = manifest.at("fabric_context").at("is_2d_routing").get<bool>();

    for (const auto& entry : routers) {
        SCOPED_TRACE(entry.path);
        const auto& router = *entry.router;
        const auto chip_prefix = entry.path.substr(0, entry.path.rfind('/') + 1);
        const auto plane = control_plane().get_routing_plane_id(entry.node, entry.eth_chan);
        const auto facing = control_plane().get_eth_chan_direction(entry.node, entry.eth_chan);

        std::set<std::string> forwarded_vcs;
        for (const auto& [vc_key, vc_receivers] : router.at("channels").at("receivers").items()) {
            for (const auto& [ch_key, receiver] : vc_receivers.items()) {
                const auto& forwards_on = receiver.at("forwards_on");
                if (!forwards_on.is_null()) {
                    forwarded_vcs.insert(forwards_on.get<std::string>());
                }
            }
        }

        const auto& edges = router.at("intra_chip_downstream_edges");
        for (const auto& [vc_key, vc_edges] : edges.items()) {
            SCOPED_TRACE(vc_key);
            EXPECT_TRUE(vc_key == "vc0" || (is_2d && vc_key == "vc1")) << vc_key;
            EXPECT_FALSE(vc_edges.empty());
            for (const auto& [edge_key, edge] : vc_edges.items()) {
                SCOPED_TRACE(edge_key);
                EXPECT_EQ(
                    keys_of(edge),
                    (std::set<std::string>{"downstream_channel", "through_tensix_mux", "free_slots", "teardown_sem"}));
                ASSERT_TRUE(edge_key.starts_with("edge"));
                const auto n = std::stoul(edge_key.substr(4));

                const auto target = edge.at("downstream_channel").get<std::string>();
                const auto senders = target.find("/channels/senders/" + vc_key + "/ch");
                ASSERT_NE(senders, std::string::npos) << target << " is not a " << vc_key << " sender channel";
                const auto target_router_path = target.substr(0, senders);
                EXPECT_TRUE(target_router_path.starts_with(chip_prefix)) << target;
                EXPECT_NE(target_router_path, entry.path);
                if (is_2d) {
                    ASSERT_TRUE(n >= 1 && n <= 4) << edge_key;
                    EXPECT_EQ(
                        key_of_path(target_router_path),
                        router_key(builder::direction_from_compact_index(facing, n - 1), plane));
                } else {
                    EXPECT_EQ(n, 1u);
                    EXPECT_EQ(key_of_path(target_router_path).substr(1), entry.router_key.substr(1));
                }

                const json* target_router = find_router(manifest, target_router_path);
                ASSERT_NE(target_router, nullptr) << target;
                const json::json_pointer channel_pointer(target.substr(senders));
                ASSERT_TRUE(target_router->contains(channel_pointer)) << target;
                const auto& channel = target_router->at(channel_pointer);
                if (edge.at("through_tensix_mux").get<bool>()) {
                    EXPECT_EQ(vc_key, "vc0");
                    EXPECT_EQ(channel.at("status"), lower_enum_name(manifest::ChannelStatus::MUX)) << target;
                } else {
                    EXPECT_EQ(channel.at("producer"), entry.path) << target;
                }

                expect_stream(edge.at("free_slots"), forwarded_vcs.contains(vc_key));
                expect_value_region(edge.at("teardown_sem"), schema_name_of<uint32_t>(), sizeof(uint32_t));
            }
        }
    }
}

// Each router has exactly the table's fields it should, router-wide and on each ERISC, each written as its entry says.
void check_router_fields(const json& manifest, const std::vector<RouterEntry>& routers) {
    const auto& vocabulary = manifest.at("vocabulary");
    const bool is_2d = manifest.at("fabric_context").at("is_2d_routing").get<bool>();
    for (const auto& entry : routers) {
        SCOPED_TRACE(entry.path);
        const auto& router = *entry.router;
        const auto num_active_eriscs = router.at("shape").at("num_active_eriscs").get<uint32_t>();

        expect_fields(router.at("fields"), manifest::k_router_fields, vocabulary, is_2d);
        std::set<std::string> expected_eriscs;
        for (uint32_t risc_id = 0; risc_id < num_active_eriscs; ++risc_id) {
            expected_eriscs.insert(fmt::format("erisc{}", risc_id));
        }
        EXPECT_EQ(keys_of(router.at("eriscs")), expected_eriscs);
        const std::set<std::string> processors = {
            lower_enum_name(tt::tt_metal::DataMovementProcessor::RISCV_0),
            lower_enum_name(tt::tt_metal::DataMovementProcessor::RISCV_1)};
        std::set<std::string> used_processors;
        for (const auto& [risc_key, erisc] : router.at("eriscs").items()) {
            SCOPED_TRACE(risc_key);
            EXPECT_EQ(keys_of(erisc), (std::set<std::string>{"processor", "fields"}));
            const auto processor = erisc.at("processor").get<std::string>();
            EXPECT_TRUE(processors.contains(processor)) << processor;
            EXPECT_TRUE(used_processors.insert(processor).second) << "two ERISCs run on " << processor;
            expect_fields(erisc.at("fields"), manifest::k_erisc_fields, vocabulary, is_2d);
        }
    }
}

// Each local chip's local_sync names its routers, and the one ERISC that leads the sync is on its master router.
void check_local_sync(const json& manifest) {
    for (const auto& [m_key, mesh] : manifest.at("meshes").items()) {
        for (const auto& [c_key, chip] : mesh.at("chips").items()) {
            if (!chip.at("is_local").get<bool>()) {
                continue;
            }
            SCOPED_TRACE(fmt::format("{}/{}", m_key, c_key));
            const auto& routers = chip.at("routers");
            const auto& local_sync = chip.at("local_sync");
            if (routers.empty()) {
                EXPECT_TRUE(local_sync.is_null());
                continue;
            }
            ASSERT_EQ(
                keys_of(local_sync), (std::set<std::string>{"master_eth_chan", "num_routers", "router_channels_mask"}));
            const FabricNodeId node(
                MeshId{static_cast<uint32_t>(std::stoul(m_key.substr(1)))},
                static_cast<uint32_t>(std::stoul(c_key.substr(1))));
            const auto master_eth_chan = local_sync.at("master_eth_chan").get<uint32_t>();
            uint32_t mask = 0;
            bool syncs = false;
            uint32_t num_masters = 0;
            for (const auto& [r_key, router] : routers.items()) {
                const auto eth_chan = channel_for_key(node, r_key);
                ASSERT_TRUE(eth_chan.has_value()) << r_key;
                mask |= 1u << *eth_chan;
                // The kernel reads the local sync arguments only with wait_for_host_signal.
                if (!router.at("fields").at("wait_for_host_signal").at("value").get<bool>()) {
                    continue;
                }
                syncs = true;
                for (const auto& [risc_key, erisc] : router.at("eriscs").items()) {
                    if (erisc.at("fields").at("local_handshake_master").at("value").get<bool>()) {
                        EXPECT_EQ(*eth_chan, master_eth_chan) << r_key << "/" << risc_key;
                        ++num_masters;
                    }
                }
            }
            EXPECT_EQ(local_sync.at("num_routers"), routers.size());
            EXPECT_EQ(local_sync.at("router_channels_mask"), mask);
            EXPECT_NE(mask & (1u << master_eth_chan), 0u) << "the master is not one of the chip's routers";
            EXPECT_EQ(num_masters, syncs ? 1u : 0u);
        }
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

    EXPECT_EQ(lower_enum_name(L1CreditCounterReason::MULTI_TXQ), "multi_txq");
    EXPECT_EQ(lower_enum_name(L1CreditCounterReason::EXPRESS), "express");
    EXPECT_EQ(lower_enum_name(L1CreditCounterReason::NO_COMPLETION_REGISTER), "no_completion_register");

    EXPECT_EQ(lower_enum_name(manifest::CreditCounterArray::TO_SENDER_ACK), "to_sender_ack");
    EXPECT_EQ(lower_enum_name(manifest::CreditCounterArray::TO_SENDER_COMPLETION), "to_sender_completion");
    EXPECT_EQ(lower_enum_name(manifest::CreditCounterArray::RECEIVER_ACK), "receiver_ack");
    EXPECT_EQ(lower_enum_name(manifest::CreditCounterArray::RECEIVER_COMPLETION), "receiver_completion");

    EXPECT_EQ(lower_enum_name(manifest::StreamRegister::BUF_SPACE_AVAILABLE), "buf_space_available");
    EXPECT_EQ(lower_enum_name(manifest::StreamRegister::REMOTE_SRC), "remote_src");

    EXPECT_EQ(lower_enum_name(manifest::ChannelStatus::ACTIVE), "active");
    EXPECT_EQ(lower_enum_name(manifest::ChannelStatus::VC_NOT_SERVICED), "vc_not_serviced");
    EXPECT_EQ(lower_enum_name(manifest::ChannelStatus::MUX), "mux");
    EXPECT_EQ(lower_enum_name(manifest::ChannelStatus::TRIMMED), "trimmed");

    EXPECT_EQ(lower_enum_name(manifest::NocCmdBuf::WR_CMD_BUF), "wr_cmd_buf");
    EXPECT_EQ(lower_enum_name(manifest::NocCmdBuf::RD_CMD_BUF), "rd_cmd_buf");
    EXPECT_EQ(lower_enum_name(manifest::NocCmdBuf::WR_REG_CMD_BUF), "wr_reg_cmd_buf");
    EXPECT_EQ(lower_enum_name(manifest::NocCmdBuf::AT_CMD_BUF), "at_cmd_buf");

    EXPECT_EQ(lower_enum_name(tt::tt_metal::DataMovementProcessor::RISCV_0), "riscv_0");
    EXPECT_EQ(lower_enum_name(tt::tt_metal::DataMovementProcessor::RISCV_1), "riscv_1");

    EXPECT_EQ(lower_enum_name(manifest::FieldCategory::LIFECYCLE), "lifecycle");
    EXPECT_EQ(lower_enum_name(manifest::FieldCategory::KERNEL_PARAMS), "kernel_params");
    EXPECT_EQ(lower_enum_name(manifest::FieldCategory::FLOW_CONTROL), "flow_control");
    EXPECT_EQ(lower_enum_name(manifest::FieldCategory::CONTROL_INFO), "control_info");

    EXPECT_EQ(kind_name<manifest::kind::L1Value>(), "l1_value");
    EXPECT_EQ(kind_name<manifest::kind::Stream>(), "stream");
    EXPECT_EQ(kind_name<manifest::kind::Number>(), "number");
    EXPECT_EQ(kind_name<manifest::kind::Flag>(), "flag");
    EXPECT_EQ(kind_name<manifest::kind::Enum>(), "enum");

    EXPECT_EQ(schema_name(field::Uint{}, 4), "u32");
    EXPECT_EQ(schema_name(field::Uint{}, 1), "u8");
    EXPECT_EQ(schema_name(field::Int{}, 2), "i16");
    EXPECT_EQ(schema_name(field::Enum{"RouterState"}, 4), "enum:RouterState");
    EXPECT_EQ(schema_name(field::Struct{"WorkerXY"}, 4), "struct:WorkerXY");
    EXPECT_EQ(schema_name(field::Packed{"direction_table", 2}, 16), "packed:direction_table");
    EXPECT_EQ(schema_name(field::Bytes{}, 1), "bytes");
    EXPECT_EQ(schema_name(field::Pad{}, 1), "pad");
    EXPECT_EQ(schema_name_of<uint32_t>(), "u32");
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

TEST_F(Fabric1DManifestFixture, CreditTransport) { check_credit_transport(manifest_, routers_); }
TEST_F(Fabric2DManifestFixture, CreditTransport) { check_credit_transport(manifest_, routers_); }

TEST_F(Fabric1DManifestFixture, RouterCreditCounters) { check_router_credit_counters(routers_); }
TEST_F(Fabric2DManifestFixture, RouterCreditCounters) { check_router_credit_counters(routers_); }

TEST_F(Fabric1DManifestFixture, RouterSenders) { check_router_senders(manifest_, routers_); }
TEST_F(Fabric2DManifestFixture, RouterSenders) { check_router_senders(manifest_, routers_); }

TEST_F(Fabric1DManifestFixture, RouterReceivers) { check_router_receivers(manifest_, routers_); }
TEST_F(Fabric2DManifestFixture, RouterReceivers) { check_router_receivers(manifest_, routers_); }

TEST_F(Fabric1DManifestFixture, RouterDownstreamEdges) { check_router_edges(manifest_, routers_); }
TEST_F(Fabric2DManifestFixture, RouterDownstreamEdges) { check_router_edges(manifest_, routers_); }

TEST_F(Fabric1DManifestFixture, RouterFields) { check_router_fields(manifest_, routers_); }
TEST_F(Fabric2DManifestFixture, RouterFields) { check_router_fields(manifest_, routers_); }

TEST_F(Fabric1DManifestFixture, LocalSync) { check_local_sync(manifest_); }
TEST_F(Fabric2DManifestFixture, LocalSync) { check_local_sync(manifest_); }

TEST_F(Fabric1DManifestFixture, RouterRegionsDisjoint) { check_router_regions_disjoint(routers_); }
TEST_F(Fabric2DManifestFixture, RouterRegionsDisjoint) { check_router_regions_disjoint(routers_); }

}  // namespace tt::tt_fabric::fabric_router_tests
