// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Checks the fabric manifest written during fabric init against the live fabric: ControlPlane, the cluster and
// what the router builders published. Each check covers one part of the manifest and runs under 1D and 2D.

#include <fmt/format.h>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <variant>
#include <vector>

#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <umd/device/types/arch.hpp>

#include "fabric_fixture.hpp"
#include "hostdevcommon/fabric_common.h"
#include "impl/context/metal_context.hpp"
#include "tt_metal/fabric/builder/fabric_edge_capability.hpp"
#include "tt_metal/fabric/builder/fabric_manifest_model.hpp"
#include "tt_metal/fabric/builder/fabric_stream_assignment.hpp"
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

using Fabric1DManifestFixture = FabricManifestFixture<FabricConfig::FABRIC_1D>;
using Fabric2DManifestFixture = FabricManifestFixture<FabricConfig::FABRIC_2D>;

// ============ Checks ============

// The manifest has exactly the top-level blocks the writer emits, and the write left no temporary file.
void check_top_level(const json& manifest, const std::filesystem::path& manifest_path, FabricConfig fabric_config) {
    EXPECT_EQ(
        keys_of(manifest), (std::set<std::string>{"manifest_version", "kind", "run", "fabric_context", "meshes"}));
    EXPECT_EQ(manifest.at("manifest_version"), FABRIC_MANIFEST_VERSION);
    EXPECT_EQ(manifest.at("kind"), "fabric_manifest");

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
    EXPECT_EQ(run.at("arch"), lower_enum_name(tt::tt_metal::MetalContext::instance().get_cluster().arch()));
    EXPECT_FALSE(run.at("written_at").get<std::string>().empty());

    const auto& fabric_context = control_plane().get_fabric_context();
    const auto& block = manifest.at("fabric_context");
    std::set<std::string> context_keys{
        "topology",
        "is_2d_routing",
        "packet_header_size_bytes",
        "max_payload_size_bytes",
        "channel_buffer_size_bytes",
        "multi_txq"};
    context_keys.insert(
        fabric_context.is_2D_routing_enabled() ? "routing_2d_route_buffer_size" : "routing_1d_extension_words");
    EXPECT_EQ(keys_of(block), context_keys);
    const auto& router_config = builder_context().get_fabric_router_config();
    EXPECT_EQ(block.at("multi_txq"), router_config.sender_txq_id != router_config.receiver_txq_id);
    EXPECT_EQ(block.at("topology"), lower_enum_name(fabric_context.get_fabric_topology()));
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

// A local mesh states each fabric VC's credit backing as the builder planned it, with a reason exactly when
// the VC is on L1 counters. Other hosts' meshes are described by their own manifests.
void check_credit_transport(const json& manifest) {
    const auto local_mesh_ids = control_plane().get_local_mesh_id_bindings();
    for (const auto& mesh_id : control_plane().get_mesh_graph().get_all_mesh_ids()) {
        SCOPED_TRACE(mesh_key(mesh_id));
        const auto& mesh = manifest.at("meshes").at(mesh_key(mesh_id));
        if (std::ranges::find(local_mesh_ids, mesh_id) == local_mesh_ids.end()) {
            EXPECT_FALSE(mesh.contains("credit_transport"));
            continue;
        }

        const auto& plan = builder_context().get_stream_assignment(mesh_id).plan();
        const auto& transport = mesh.at("credit_transport");
        std::set<std::string> expected_keys;
        for (uint32_t vc = 0; vc < builder_config::MAX_NUM_VCS; ++vc) {
            if (builder_context().get_max_sender_channels_per_vc()[vc] == 0) {
                continue;
            }
            const auto vc_key = fmt::format("vc{}", vc);
            expected_keys.insert(vc_key);
            SCOPED_TRACE(vc_key);
            ASSERT_TRUE(transport.contains(vc_key));
            const auto& entry = transport.at(vc_key);
            EXPECT_EQ(keys_of(entry), (std::set<std::string>{"backing", "reasons"}));
            EXPECT_EQ(entry.at("backing"), plan.vc_uses_counters(vc) ? "l1_counter" : "stream_register");
            json expected_reasons = json::array();
            for (const auto reason : plan.reasons(vc)) {
                expected_reasons.push_back(lower_enum_name(reason));
            }
            EXPECT_EQ(entry.at("reasons"), expected_reasons);
        }
        EXPECT_EQ(keys_of(transport), expected_keys);
    }
}

// A local mesh lists each allocated stream register under its id, with the role the builder assigned it and
// the keys that role is indexed by. Every use is a buf_space_available use until the scratch uses are added.
void check_stream_registers(const json& manifest) {
    const auto local_mesh_ids = control_plane().get_local_mesh_id_bindings();
    for (const auto& mesh_id : control_plane().get_mesh_graph().get_all_mesh_ids()) {
        SCOPED_TRACE(mesh_key(mesh_id));
        const auto& mesh = manifest.at("meshes").at(mesh_key(mesh_id));
        if (std::ranges::find(local_mesh_ids, mesh_id) == local_mesh_ids.end()) {
            EXPECT_FALSE(mesh.contains("stream_registers"));
            continue;
        }

        const auto& registers = mesh.at("stream_registers");
        std::set<std::string> expected_keys;
        for (const auto& use : builder_context().get_stream_assignment(mesh_id).uses()) {
            const auto id_key = std::to_string(use.stream_id);
            expected_keys.insert(id_key);
            SCOPED_TRACE(fmt::format("stream {}", id_key));
            ASSERT_TRUE(registers.contains(id_key));
            ASSERT_EQ(registers.at(id_key).size(), 1u);
            const auto& entry = registers.at(id_key).at(0);

            std::set<std::string> entry_keys{"role", "register"};
            if (use.vc.has_value()) {
                entry_keys.insert("vc");
                EXPECT_EQ(entry.at("vc"), *use.vc);
            }
            if (use.index.has_value()) {
                entry_keys.insert("index");
                EXPECT_EQ(entry.at("index"), *use.index);
            }
            EXPECT_EQ(keys_of(entry), entry_keys);
            EXPECT_EQ(entry.at("role"), lower_enum_name(use.role));
            EXPECT_EQ(entry.at("register"), "buf_space_available");
            EXPECT_LT(use.stream_id, StreamRegAssignments::num_eth_stream_registers);
        }
        EXPECT_EQ(keys_of(registers), expected_keys);
    }
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

// Link agrees with ControlPlane (cross-host, peer) and with what the builder published (edge capability,
// dispatch link). Direction and plane are the key, which RoutersMatchActiveChannels checks.
void check_router_link(const std::vector<RouterEntry>& routers) {
    for (const auto& entry : routers) {
        SCOPED_TRACE(entry.path);
        const auto& link = entry.router->at("link");
        const auto& published = entry.published->link;
        EXPECT_EQ(
            keys_of(link),
            (std::set<std::string>{"edge_capability", "peer", "cross_host", "wrap", "dispatch_link"}));
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
        const auto peer_path = link.at("peer").get<std::string>();
        SCOPED_TRACE(fmt::format("{} -> {}", entry.path, peer_path));
        // Both ends are on the same routing plane: the keys differ only in direction, e.g. E1 and W1.
        EXPECT_EQ(peer_path.substr(peer_path.rfind('/') + 2), entry.key.substr(1));
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

// Shape is what the builder published.
void check_router_shape(const std::vector<RouterEntry>& routers) {
    for (const auto& entry : routers) {
        SCOPED_TRACE(entry.path);
        const auto& shape = entry.router->at("shape");
        const auto& published = entry.published->shape;
        EXPECT_EQ(
            keys_of(shape),
            (std::set<std::string>{
                "num_vcs",
                "senders_per_vc",
                "receivers_per_vc",
                "num_active_eriscs",
                "channel_trimming_overrides_applied",
                "vc0_bubble_flow_control"}));
        EXPECT_EQ(shape.at("num_vcs"), published.num_vcs);
        EXPECT_EQ(shape.at("senders_per_vc"), json(published.senders_per_vc));
        EXPECT_EQ(shape.at("receivers_per_vc"), json(published.receivers_per_vc));
        EXPECT_EQ(shape.at("num_active_eriscs"), published.num_active_eriscs);
        EXPECT_EQ(shape.at("channel_trimming_overrides_applied"), published.channel_trimming_overrides_applied);
        EXPECT_EQ(shape.at("vc0_bubble_flow_control"), published.vc0_bubble_flow_control);
    }
}

// The region is what the builder published.
void expect_region(const json& region, const manifest::L1Region& published) {
    EXPECT_EQ(region.at("address"), published.address);
    EXPECT_EQ(region.at("size"), published.size);
    EXPECT_EQ(region.contains("num_elements"), published.num_elements.has_value());
    if (published.num_elements.has_value()) {
        EXPECT_EQ(region.at("num_elements"), *published.num_elements);
    }
    EXPECT_EQ(region.contains("size_per_element"), published.size_per_element.has_value());
    if (published.size_per_element.has_value()) {
        EXPECT_EQ(region.at("size_per_element"), *published.size_per_element);
    }
    EXPECT_EQ(region.at("schema"), published.schema);
    EXPECT_EQ(region.at("host_cleared"), published.host_cleared);
}

// The four counter arrays are what the builder published, back to back and u32 each, and the host clears
// them exactly when it says so.
void check_router_credit_counters(const std::vector<RouterEntry>& routers) {
    const auto addresses_to_clear = builder_context().get_fabric_router_addresses_to_clear();
    for (const auto& entry : routers) {
        SCOPED_TRACE(entry.path);
        const auto& counters = entry.router->at("credit_counters");
        const auto& published = entry.published->credit_counters;
        const std::array<std::tuple<const char*, const manifest::L1Region*, const char*>, 4> arrays = {{
            {"to_sender_ack", &published.to_sender_ack, "own_sender_compact"},
            {"to_sender_completion", &published.to_sender_completion, "own_sender_compact"},
            {"receiver_ack", &published.receiver_ack, "peer_sender_compact"},
            {"receiver_completion", &published.receiver_completion, "peer_sender_compact"},
        }};
        EXPECT_EQ(
            keys_of(counters),
            (std::set<std::string>{"to_sender_ack", "to_sender_completion", "receiver_ack", "receiver_completion"}));

        std::optional<uint32_t> next_address;
        for (const auto& [name, published_array, index_space] : arrays) {
            SCOPED_TRACE(name);
            const auto& array = counters.at(name);
            EXPECT_EQ(
                keys_of(array),
                (std::set<std::string>{
                    "address", "size", "num_elements", "size_per_element", "schema", "host_cleared", "index_space"}));
            expect_region(array, *published_array);
            EXPECT_EQ(array.at("index_space"), index_space);
            EXPECT_EQ(array.at("schema"), "u32");
            EXPECT_EQ(
                array.at("size").get<uint32_t>(),
                array.at("num_elements").get<uint32_t>() * array.at("size_per_element").get<uint32_t>());
            EXPECT_EQ(
                array.at("host_cleared"),
                std::ranges::find(addresses_to_clear, array.at("address").get<size_t>()) != addresses_to_clear.end());
            if (next_address.has_value()) {
                EXPECT_EQ(array.at("address"), *next_address);
            }
            next_address = array.at("address").get<uint32_t>() + array.at("size").get<uint32_t>();
        }
    }
}

void expect_stream(const json& stream, const manifest::StreamRef& published) {
    EXPECT_EQ(keys_of(stream), (std::set<std::string>{"stream_id", "register", "schema"}));
    EXPECT_EQ(stream.at("stream_id"), published.stream_id);
    EXPECT_EQ(stream.at("register"), lower_enum_name(published.reg));
    EXPECT_EQ(stream.at("schema"), published.schema);
}

// A stream-backed credit is its register; a counter-backed one is its element of the router's to_sender arrays.
void expect_credit(const json& credit, const manifest::CreditRef& published, bool uses_counters) {
    if (const auto* stream = std::get_if<manifest::StreamRef>(&published)) {
        EXPECT_FALSE(uses_counters);
        expect_stream(credit, *stream);
        return;
    }
    const auto& array = std::get<manifest::ArrayRef>(published);
    EXPECT_TRUE(uses_counters);
    EXPECT_EQ(keys_of(credit), (std::set<std::string>{"array", "index"}));
    EXPECT_EQ(credit.at("array"), array.array);
    EXPECT_EQ(credit.at("index"), array.index);
}

json expected_serviced_by(const std::vector<uint32_t>& risc_ids) {
    json out = json::array();
    for (const auto risc_id : risc_ids) {
        out.push_back(fmt::format("erisc{}", risc_id));
    }
    return out;
}

// The set of vc<N> keys for the VCs that have channels.
template <typename Channel>
std::set<std::string> expected_vc_keys(const std::vector<std::vector<Channel>>& channels) {
    std::set<std::string> keys;
    for (size_t vc = 0; vc < channels.size(); ++vc) {
        if (!channels[vc].empty()) {
            keys.insert(fmt::format("vc{}", vc));
        }
    }
    return keys;
}

// A sibling is the router facing `direction` on this router's chip and routing plane.
std::string expected_sibling_path(const RouterEntry& entry, eth_chan_directions direction) {
    const auto plane = control_plane().get_routing_plane_id(entry.node, entry.eth_chan);
    return fmt::format(
        "{}/{}/{}", mesh_key(entry.node.mesh_id), chip_key(entry.node.chip_id), router_key(direction, plane));
}

json expected_producer(const RouterEntry& entry, const std::optional<manifest::SenderChannelProducer>& producer) {
    if (!producer.has_value()) {
        return nullptr;
    }
    if (std::holds_alternative<manifest::LocalWorker>(*producer)) {
        return "worker";
    }
    return expected_sibling_path(entry, std::get<manifest::SiblingRouterRef>(*producer).direction);
}

// Each edge is what the builder published: keyed by its kernel edge number, landing on a sender channel of the
// sibling it targets that is in this manifest.
void expect_downstream_edges(
    const json& manifest,
    const json& edges,
    const RouterEntry& entry,
    const std::vector<manifest::DownstreamEdge>& published,
    bool serviced) {
    std::set<std::string> keys;
    for (const auto& edge : published) {
        keys.insert(fmt::format("edge{}", edge.edge));
    }
    EXPECT_EQ(keys_of(edges), keys);
    ASSERT_EQ(keys.size(), published.size());

    for (const auto& expected : published) {
        SCOPED_TRACE(fmt::format("edge{}", expected.edge));
        const auto& edge = edges.at(fmt::format("edge{}", expected.edge));
        EXPECT_EQ(keys_of(edge), (std::set<std::string>{"downstream_channel", "free_slots", "teardown_sem"}));

        const auto target = expected_sibling_path(entry, expected.target.direction);
        EXPECT_EQ(
            edge.at("downstream_channel"),
            fmt::format("{}/senders/vc{}/ch{}", target, expected.landing_vc, expected.landing_channel));
        const json* target_router = find_router(manifest, target);
        ASSERT_NE(target_router, nullptr) << target;
        EXPECT_TRUE(target_router->at("channels")
                        .at("senders")
                        .at(fmt::format("vc{}", expected.landing_vc))
                        .contains(fmt::format("ch{}", expected.landing_channel)));

        expect_stream(edge.at("free_slots"), expected.free_slots);
        if (serviced) {
            EXPECT_NE(expected.free_slots.stream_id, k_unused_stream_id);
        }
        expect_region(edge.at("teardown_sem"), expected.teardown_sem);
        EXPECT_EQ(edge.at("teardown_sem").at("schema"), "u32");
    }
}

// Each sender channel is what the builder published, over the router's shape. Its credits are on the backing
// the mesh's credit transport names, a sibling producer is a router in this manifest, and only worker-fed
// channels have a buffer index semaphore.
void check_router_senders(const json& manifest, const std::vector<RouterEntry>& routers) {
    for (const auto& entry : routers) {
        SCOPED_TRACE(entry.path);
        const auto& senders = entry.router->at("channels").at("senders");
        const auto& published = entry.published->channels.senders;
        const auto& transport = manifest.at("meshes").at(mesh_key(entry.node.mesh_id)).at("credit_transport");

        for (uint32_t vc = 0; vc < builder_config::MAX_NUM_VCS; ++vc) {
            ASSERT_EQ(published.at(vc).size(), entry.published->shape.senders_per_vc[vc]);
        }
        EXPECT_EQ(keys_of(senders), expected_vc_keys(published));

        for (uint32_t vc = 0; vc < builder_config::MAX_NUM_VCS; ++vc) {
            if (published[vc].empty()) {
                continue;
            }
            const auto vc_key = fmt::format("vc{}", vc);
            const bool uses_counters = transport.at(vc_key).at("backing") == "l1_counter";
            ASSERT_EQ(senders.at(vc_key).size(), published[vc].size());
            for (uint32_t ch = 0; ch < published[vc].size(); ++ch) {
                SCOPED_TRACE(fmt::format("{}/ch{}", vc_key, ch));
                const auto& sender = senders.at(vc_key).at(fmt::format("ch{}", ch));
                const auto& expected = published[vc][ch];
                EXPECT_EQ(
                    keys_of(sender),
                    (std::set<std::string>{
                        "serviced_by",
                        "producer",
                        "is_injection_channel",
                        "producer_credit_return",
                        "ring_buffer",
                        "free_slots",
                        "credits",
                        "control_info"}));

                EXPECT_EQ(sender.at("serviced_by"), expected_serviced_by(expected.serviced_by));

                const json producer = expected_producer(entry, expected.producer);
                EXPECT_EQ(sender.at("producer"), producer);
                if (producer.is_string() && producer != "worker") {
                    EXPECT_NE(find_router(manifest, producer.get<std::string>()), nullptr);
                }

                EXPECT_EQ(sender.at("is_injection_channel"), expected.is_injection_channel);
                EXPECT_EQ(
                    sender.at("producer_credit_return"),
                    json({{"noc", static_cast<uint32_t>(expected.producer_credit_return.noc)},
                          {"cmd_buf", lower_enum_name(expected.producer_credit_return.cmd_buf)}}));

                const auto& ring = sender.at("ring_buffer");
                expect_region(ring, expected.ring_buffer);
                EXPECT_EQ(ring.at("schema"), "packet_ring");
                EXPECT_EQ(
                    ring.at("size").get<uint32_t>(),
                    ring.at("num_elements").get<uint32_t>() * ring.at("size_per_element").get<uint32_t>());

                expect_stream(sender.at("free_slots"), expected.free_slots);
                if (!expected.serviced_by.empty()) {
                    EXPECT_NE(expected.free_slots.stream_id, k_unused_stream_id);
                }

                const auto& credits = sender.at("credits");
                const bool acked = vc == 0 && entry.published->shape.vc0_bubble_flow_control;
                EXPECT_EQ(expected.credits.acked.has_value(), acked);
                EXPECT_EQ(credits.contains("acked"), acked);
                if (acked) {
                    expect_credit(credits.at("acked"), *expected.credits.acked, uses_counters);
                }
                expect_credit(credits.at("completed"), expected.credits.completed, uses_counters);

                const auto& control_info = sender.at("control_info");
                const bool worker_fed = producer == "worker";
                EXPECT_EQ(expected.control_info.buffer_index_sem.has_value(), worker_fed);
                EXPECT_EQ(
                    keys_of(control_info),
                    worker_fed ? (std::set<std::string>{"connection", "conn_info", "buffer_index_sem"})
                               : (std::set<std::string>{"connection", "conn_info"}));
                expect_region(control_info.at("connection"), expected.control_info.connection);
                expect_region(control_info.at("conn_info"), expected.control_info.conn_info);
                if (worker_fed) {
                    expect_region(control_info.at("buffer_index_sem"), *expected.control_info.buffer_index_sem);
                }
            }
        }
    }
}

// Each receiver channel is what the builder published, over the router's shape. Its producer is the link's
// peer, only VC2's receiver has a free-slots register, and VC2's receiver never forwards.
void check_router_receivers(const json& manifest, const std::vector<RouterEntry>& routers) {
    for (const auto& entry : routers) {
        SCOPED_TRACE(entry.path);
        const auto& receivers = entry.router->at("channels").at("receivers");
        const auto& published = entry.published->channels.receivers;

        for (uint32_t vc = 0; vc < builder_config::MAX_NUM_VCS; ++vc) {
            ASSERT_EQ(published.at(vc).size(), entry.published->shape.receivers_per_vc[vc]);
        }
        EXPECT_EQ(keys_of(receivers), expected_vc_keys(published));

        for (uint32_t vc = 0; vc < builder_config::MAX_NUM_VCS; ++vc) {
            if (published[vc].empty()) {
                continue;
            }
            const auto vc_key = fmt::format("vc{}", vc);
            ASSERT_EQ(receivers.at(vc_key).size(), published[vc].size());
            for (uint32_t ch = 0; ch < published[vc].size(); ++ch) {
                SCOPED_TRACE(fmt::format("{}/ch{}", vc_key, ch));
                const auto& receiver = receivers.at(vc_key).at(fmt::format("ch{}", ch));
                const auto& expected = published[vc][ch];
                const bool has_free_slots = vc == 2;

                std::set<std::string> keys = {
                    "serviced_by",
                    "producer",
                    "forwarding_disabled",
                    "intermesh_ingress",
                    "forward_noc",
                    "local_write_noc",
                    "ring_buffer",
                    "pkts_sent",
                    "downstream_edges"};
                if (has_free_slots) {
                    keys.insert("free_slots");
                }
                EXPECT_EQ(keys_of(receiver), keys);

                EXPECT_EQ(receiver.at("serviced_by"), expected_serviced_by(expected.serviced_by));
                EXPECT_EQ(receiver.at("producer"), entry.router->at("link").at("peer"));
                EXPECT_EQ(receiver.at("forwarding_disabled"), expected.forwarding_disabled);
                EXPECT_EQ(receiver.at("intermesh_ingress"), expected.intermesh_ingress);
                EXPECT_EQ(
                    receiver.at("forward_noc"),
                    json({{"noc", static_cast<uint32_t>(expected.forward_noc.noc)},
                          {"data_cmd_buf", lower_enum_name(expected.forward_noc.data_cmd_buf)},
                          {"sync_cmd_buf", lower_enum_name(expected.forward_noc.sync_cmd_buf)}}));
                EXPECT_EQ(
                    receiver.at("local_write_noc"),
                    json({{"noc", static_cast<uint32_t>(expected.local_write_noc.noc)},
                          {"cmd_buf", lower_enum_name(expected.local_write_noc.cmd_buf)}}));

                const auto& ring = receiver.at("ring_buffer");
                expect_region(ring, expected.ring_buffer);
                EXPECT_EQ(ring.at("schema"), "packet_ring");
                EXPECT_EQ(
                    ring.at("size").get<uint32_t>(),
                    ring.at("num_elements").get<uint32_t>() * ring.at("size_per_element").get<uint32_t>());

                expect_stream(receiver.at("pkts_sent"), expected.pkts_sent);
                EXPECT_EQ(expected.free_slots.has_value(), has_free_slots);
                if (has_free_slots) {
                    expect_stream(receiver.at("free_slots"), *expected.free_slots);
                }
                if (!expected.serviced_by.empty()) {
                    EXPECT_NE(expected.pkts_sent.stream_id, k_unused_stream_id);
                    if (has_free_slots) {
                        EXPECT_NE(expected.free_slots->stream_id, k_unused_stream_id);
                    }
                }

                if (vc == 2) {
                    EXPECT_TRUE(expected.downstream_edges.empty());
                }
                expect_downstream_edges(
                    manifest,
                    receiver.at("downstream_edges"),
                    entry,
                    expected.downstream_edges,
                    !expected.serviced_by.empty());
            }
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

    EXPECT_EQ(lower_enum_name(StreamRole::RECEIVER_PKTS_SENT), "receiver_pkts_sent");
    EXPECT_EQ(lower_enum_name(StreamRole::SENDER_PKTS_ACKED), "sender_pkts_acked");
    EXPECT_EQ(lower_enum_name(StreamRole::SENDER_PKTS_COMPLETED), "sender_pkts_completed");
    EXPECT_EQ(lower_enum_name(StreamRole::DOWNSTREAM_FREE_SLOTS), "downstream_free_slots");
    EXPECT_EQ(lower_enum_name(StreamRole::SENDER_FREE_SLOTS), "sender_free_slots");
    EXPECT_EQ(lower_enum_name(StreamRole::VC2_SENDER_FREE_SLOTS), "vc2_sender_free_slots");
    EXPECT_EQ(lower_enum_name(StreamRole::VC2_RECEIVER_FREE_SLOTS), "vc2_receiver_free_slots");
    EXPECT_EQ(lower_enum_name(StreamRole::TENSIX_RELAY_FREE_SLOTS), "tensix_relay_free_slots");

    EXPECT_EQ(lower_enum_name(manifest::StreamRegister::BUF_SPACE_AVAILABLE), "buf_space_available");
    EXPECT_EQ(lower_enum_name(manifest::StreamRegister::REMOTE_SRC), "remote_src");

    EXPECT_EQ(lower_enum_name(manifest::NocCmdBuf::WR_CMD_BUF), "wr_cmd_buf");
    EXPECT_EQ(lower_enum_name(manifest::NocCmdBuf::RD_CMD_BUF), "rd_cmd_buf");
    EXPECT_EQ(lower_enum_name(manifest::NocCmdBuf::WR_REG_CMD_BUF), "wr_reg_cmd_buf");
    EXPECT_EQ(lower_enum_name(manifest::NocCmdBuf::AT_CMD_BUF), "at_cmd_buf");
}

TEST_F(Fabric1DManifestFixture, TopLevel) { check_top_level(manifest_, manifest_path_, fabric_config); }
TEST_F(Fabric2DManifestFixture, TopLevel) { check_top_level(manifest_, manifest_path_, fabric_config); }

TEST_F(Fabric1DManifestFixture, Meshes) { check_meshes(manifest_); }
TEST_F(Fabric2DManifestFixture, Meshes) { check_meshes(manifest_); }

TEST_F(Fabric1DManifestFixture, CreditTransport) { check_credit_transport(manifest_); }
TEST_F(Fabric2DManifestFixture, CreditTransport) { check_credit_transport(manifest_); }

TEST_F(Fabric1DManifestFixture, StreamRegisters) { check_stream_registers(manifest_); }
TEST_F(Fabric2DManifestFixture, StreamRegisters) { check_stream_registers(manifest_); }

TEST_F(Fabric1DManifestFixture, Chips) { check_chips(manifest_); }
TEST_F(Fabric2DManifestFixture, Chips) { check_chips(manifest_); }

TEST_F(Fabric1DManifestFixture, RoutersMatchActiveChannels) { check_routers_match_active_channels(manifest_); }
TEST_F(Fabric2DManifestFixture, RoutersMatchActiveChannels) { check_routers_match_active_channels(manifest_); }

TEST_F(Fabric1DManifestFixture, RouterIdentity) { check_router_identity(routers_); }
TEST_F(Fabric2DManifestFixture, RouterIdentity) { check_router_identity(routers_); }

TEST_F(Fabric1DManifestFixture, RouterLink) { check_router_link(routers_); }
TEST_F(Fabric2DManifestFixture, RouterLink) { check_router_link(routers_); }

TEST_F(Fabric1DManifestFixture, PeersAreSymmetric) { check_peers_are_symmetric(manifest_, routers_); }
TEST_F(Fabric2DManifestFixture, PeersAreSymmetric) { check_peers_are_symmetric(manifest_, routers_); }

TEST_F(Fabric1DManifestFixture, RouterShape) { check_router_shape(routers_); }
TEST_F(Fabric2DManifestFixture, RouterShape) { check_router_shape(routers_); }

TEST_F(Fabric1DManifestFixture, RouterCreditCounters) { check_router_credit_counters(routers_); }
TEST_F(Fabric2DManifestFixture, RouterCreditCounters) { check_router_credit_counters(routers_); }

TEST_F(Fabric1DManifestFixture, RouterSenders) { check_router_senders(manifest_, routers_); }
TEST_F(Fabric2DManifestFixture, RouterSenders) { check_router_senders(manifest_, routers_); }

TEST_F(Fabric1DManifestFixture, RouterReceivers) { check_router_receivers(manifest_, routers_); }
TEST_F(Fabric2DManifestFixture, RouterReceivers) { check_router_receivers(manifest_, routers_); }

}  // namespace tt::tt_fabric::fabric_router_tests
