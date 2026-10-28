// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Checks the fabric manifest written during fabric init against the live fabric. Each check covers one part of the
// manifest and runs under every fabric configuration a fixture brings up.

#include <gtest/gtest.h>
#include <fmt/format.h>
#include <nlohmann/json.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>

#include <algorithm>
#include <bit>
#include <filesystem>
#include <fstream>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <variant>
#include <vector>

#include "fabric_fixture.hpp"
#include "impl/context/metal_context.hpp"
#include "llrt/hal.hpp"
#include "tt_metal/fabric/builder/fabric_builder_config.hpp"
#include "tt_metal/fabric/builder/fabric_builder_helpers.hpp"
#include "tt_metal/fabric/builder/fabric_edge_capability.hpp"
#include "tt_metal/fabric/fabric_builder_context.hpp"
#include "tt_metal/fabric/fabric_context.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_router_pass.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_fields.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_model.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_names.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_type_collector.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_struct_layouts.hpp"

namespace tt::tt_fabric::fabric_router_tests {
namespace {

using json = nlohmann::json;

const ControlPlane& control_plane() { return tt::tt_metal::MetalContext::instance().get_control_plane(); }

const FabricBuilderContext& builder_context() { return control_plane().get_fabric_context().get_builder_context(); }

// ============ Helpers ============

using manifest::chip_key;
using manifest::kind_name;
using manifest::lower_enum_name;
using manifest::mesh_key;
using manifest::router_key;
using manifest::schema_name;

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
template <FabricConfig kFabricConfig, FabricTensixConfig kTensixConfig = FabricTensixConfig::DISABLED>
class FabricManifestFixture : public BaseFabricFixture {
protected:
    static constexpr FabricConfig fabric_config = kFabricConfig;

    static void SetUpTestSuite() {
        auto& rtoptions = tt::tt_metal::MetalContext::instance().rtoptions();
        generate_manifest_before_ = rtoptions.get_generate_fabric_manifest();
        rtoptions.set_generate_fabric_manifest(true);
        suite_start_ = std::filesystem::file_time_type::clock::now();
        DoSetUpTestSuite(kFabricConfig, std::nullopt, kTensixConfig);
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
// We use mock cluster descriptors for torus and mux fabric configs, so we can test the manifest generation for these
// configs without needing physical hardware. These names intentionally don't hit the hardware CI filters so that they
// don't run on hardware CI as they are meant to be host-only.
using FabricTorusXManifestFixture = FabricManifestFixture<FabricConfig::FABRIC_2D_TORUS_X>;
using FabricTorusYManifestFixture = FabricManifestFixture<FabricConfig::FABRIC_2D_TORUS_Y>;
using FabricTorusXYManifestFixture = FabricManifestFixture<FabricConfig::FABRIC_2D_TORUS_XY>;
using FabricMuxManifestFixture = FabricManifestFixture<FabricConfig::FABRIC_2D, FabricTensixConfig::MUX>;

// ============ Checks ============
//
// Generation already fails on what each pass can check about itself, and the schema check pins every key set and value
// type. These checks cover what neither can: the manifest against an independent source (the fixture's own settings,
// the HAL, the builder's buffer map, ControlPlane's view of the neighbor graph), and facts that only hold across
// routers, chips or blocks.

// The manifest describes this run's fabric config, was written by this run, and the write left no temporary file.
void check_top_level(
    const json& manifest,
    const std::filesystem::path& manifest_path,
    FabricConfig fabric_config,
    std::filesystem::file_time_type suite_start) {
    EXPECT_EQ(manifest.at("fabric").at("config").at("fabric_config"), lower_enum_name(fabric_config));
    EXPECT_EQ(manifest.at("fabric").at("is_2d_routing"), is_2d_fabric_config(fabric_config));
    EXPECT_GE(std::filesystem::last_write_time(manifest_path), suite_start);

    // Other ranks write their manifests into the same directory, and may still be writing.
    const auto temp_prefix = manifest_path.filename().string() + ".tmp.";
    for (const auto& entry : std::filesystem::directory_iterator(manifest_path.parent_path())) {
        EXPECT_FALSE(entry.path().filename().string().starts_with(temp_prefix)) << entry.path();
    }
}

// A local chip's Z port role, which the builder takes from its per-direction capabilities, matches the one the neighbor
// graph gives.
void check_chip_z_port_role(const json& manifest) {
    for (const auto& [m_key, mesh] : manifest.at("meshes").items()) {
        for (const auto& [c_key, chip] : mesh.at("chips").items()) {
            if (!chip.at("is_local").get<bool>()) {
                continue;
            }
            SCOPED_TRACE(fmt::format("{}/{}", m_key, c_key));
            const FabricNodeId node(
                MeshId{static_cast<uint32_t>(std::stoul(m_key.substr(1)))},
                static_cast<uint32_t>(std::stoul(c_key.substr(1))));
            EXPECT_EQ(chip.at("z_port_role"), lower_enum_name(z_port_role(control_plane(), node)));
        }
    }
}

// The edge capability the builder classified follows from where ControlPlane says the peer is and which way the router
// faces, and a wrap link spans an axis the mesh's torus wraps.
void check_router_link(const json& manifest, const std::vector<RouterEntry>& routers) {
    for (const auto& entry : routers) {
        SCOPED_TRACE(entry.path);
        const auto& link = entry.router->at("link");
        const bool faces_z = entry.router_key.front() == 'Z';
        const auto peer = control_plane().try_get_connected_mesh_chip_chan_ids(entry.node, entry.eth_chan);
        if (peer.has_value()) {
            // A same-mesh Z edge is always an express chord: classification rejects one on a mesh without express.
            EdgeCapability expected = EdgeCapability::INTRAMESH_CARDINAL;
            if (peer->first.mesh_id != entry.node.mesh_id) {
                expected = EdgeCapability::INTERMESH;
            } else if (faces_z) {
                expected = EdgeCapability::INTRAMESH_EXPRESS;
            }
            EXPECT_EQ(link.at("edge_capability"), lower_enum_name(expected));
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

// Every VC some router of a local mesh has senders on has a credit transport entry, so each sender's credits can be
// read by the backing its VC names.
void check_credit_transport(const json& manifest, const std::vector<RouterEntry>& routers) {
    for (const auto& [m_key, mesh] : manifest.at("meshes").items()) {
        if (!mesh.contains("credit_transport")) {
            continue;
        }
        SCOPED_TRACE(m_key);
        std::set<std::string> vcs_with_senders;
        for (const auto& entry : routers) {
            if (mesh_key(entry.node.mesh_id) != m_key) {
                continue;
            }
            const auto& senders = entry.router->at("shape").at("senders_per_vc");
            for (uint32_t vc = 0; vc < senders.size(); ++vc) {
                if (senders.at(vc).get<uint32_t>() > 0) {
                    vcs_with_senders.insert(fmt::format("vc{}", vc));
                }
            }
        }
        EXPECT_TRUE(std::ranges::includes(keys_of(mesh.at("credit_transport")), vcs_with_senders));
    }
}

// A stream register the builder did not allocate is null. One it did is one of the streams the arch lists.
void expect_stream(const json& stream, bool must_be_allocated, uint32_t num_streams) {
    if (stream.is_null()) {
        EXPECT_FALSE(must_be_allocated);
        return;
    }
    EXPECT_LT(stream.at("stream_id").get<uint32_t>(), num_streams);
}

// On L1 counters, a credit is the sender's own element of the router's counter array, which is indexed by the router's
// sender compact index and must be long enough for it. On stream registers, it is a register.
void expect_credit(
    const json& credit,
    const json& router,
    manifest::CreditCounterArray array,
    bool uses_counters,
    uint32_t compact,
    bool serviced,
    uint32_t num_streams) {
    if (!uses_counters) {
        ASSERT_FALSE(credit.contains("array")) << credit;
        expect_stream(credit, serviced, num_streams);
        return;
    }
    ASSERT_TRUE(credit.is_object() && credit.contains("array")) << credit;
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

// The arch's L1 areas, which every router's core has.
const json& arch_areas(const json& manifest) { return manifest.at("arch").at("l1_areas"); }

// No two of a router's L1 regions overlap, counting the arch areas on its core.
void check_router_regions_disjoint(const json& manifest, const std::vector<RouterEntry>& routers) {
    std::vector<std::tuple<uint32_t, uint32_t, std::string>> areas;
    collect_regions(arch_areas(manifest), "l1_areas", areas);
    for (const auto& entry : routers) {
        SCOPED_TRACE(entry.path);
        auto regions = areas;
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

// A ring buffer's slots are each one channel buffer, and fill its size.
void expect_ring_buffer(const json& ring, uint32_t channel_buffer_size) {
    if (ring.is_null()) {
        return;
    }
    EXPECT_EQ(ring.at("size_per_element"), channel_buffer_size);
    EXPECT_EQ(
        ring.at("size").get<uint32_t>(),
        ring.at("num_elements").get<uint32_t>() * ring.at("size_per_element").get<uint32_t>());
}

// A router's sender channels, over its shape. Each one's producer is the chip's worker on a VC's first channel (the
// tensix mux on VC0 in mux mode), or a different router on the same chip and routing plane, feeding at most one
// channel per VC; in 1D a router and its producer feed each other. Nothing feeds a channel the mux carries. Only the
// worker channel has a producer cursor (buffer_index_sem): a sibling router keeps none. Each credit is on the backing
// the mesh's credit transport names for its VC. A router has trimmed channels only when a trimming profile is set.
void check_router_senders(const json& manifest, const std::vector<RouterEntry>& routers) {
    const bool is_2d = manifest.at("fabric").at("is_2d_routing").get<bool>();
    const auto channel_buffer_size = manifest.at("fabric").at("channel_buffer_size_bytes").get<uint32_t>();
    const auto num_streams = manifest.at("arch").at("stream_registers").at("count").get<uint32_t>();
    const bool has_trimming_profile = tt::tt_metal::MetalContext::instance().rtoptions().has_fabric_trimming_profile();

    for (const auto& entry : routers) {
        SCOPED_TRACE(entry.path);
        const auto& router = *entry.router;
        const auto& shape = router.at("shape");
        const auto& senders = router.at("channels").at("senders");
        const auto& transport = manifest.at("meshes").at(mesh_key(entry.node.mesh_id)).at("credit_transport");
        const auto chip_prefix = entry.path.substr(0, entry.path.rfind('/') + 1);
        if (!has_trimming_profile) {
            EXPECT_FALSE(shape.at("channel_trimming_overrides_applied").get<bool>());
        }

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
            ASSERT_TRUE(transport.contains(vc_key));
            const bool uses_counters = transport.at(vc_key).at("backing") == "l1_counter";

            std::set<std::string> producers;
            for (uint32_t ch = 0; ch < count; ++ch, ++compact) {
                const auto ch_key = fmt::format("ch{}", ch);
                SCOPED_TRACE(fmt::format("senders/{}/{}", vc_key, ch_key));
                ASSERT_TRUE(senders.at(vc_key).contains(ch_key));
                const auto& sender = senders.at(vc_key).at(ch_key);
                const bool serviced = !sender.at("serviced_by").empty();

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
                const bool worker_channel = producer == "worker" || producer == "tensix_mux";
                EXPECT_EQ(sender.at("fields").at("buffer_index_sem").is_null(), !worker_channel);

                expect_ring_buffer(sender.at("ring_buffer"), channel_buffer_size);
                const auto& credits = sender.at("credits");
                if (credits.contains("acked")) {
                    expect_credit(
                        credits.at("acked"),
                        router,
                        manifest::CreditCounterArray::TO_SENDER_ACK,
                        uses_counters,
                        compact,
                        serviced,
                        num_streams);
                }
                expect_credit(
                    credits.at("completed"),
                    router,
                    manifest::CreditCounterArray::TO_SENDER_COMPLETION,
                    uses_counters,
                    compact,
                    serviced,
                    num_streams);
            }
        }
        EXPECT_EQ(keys_of(senders), expected_vc_keys);
    }
}

// A router's edges land on a sibling's sender channel on the same chip and routing plane: in 2D the sibling the edge
// number names (the builder's compact index among the router's other directions), in 1D the router facing the other
// way. An edge through the tensix mux lands on a channel the mux carries.
void check_router_edges(const json& manifest, const std::vector<RouterEntry>& routers) {
    const bool is_2d = manifest.at("fabric").at("is_2d_routing").get<bool>();

    for (const auto& entry : routers) {
        SCOPED_TRACE(entry.path);
        const auto chip_prefix = entry.path.substr(0, entry.path.rfind('/') + 1);
        const auto plane = control_plane().get_routing_plane_id(entry.node, entry.eth_chan);
        const auto facing = control_plane().get_eth_chan_direction(entry.node, entry.eth_chan);

        for (const auto& [vc_key, vc_edges] : entry.router->at("intra_chip_downstream_edges").items()) {
            SCOPED_TRACE(vc_key);
            for (const auto& [edge_key, edge] : vc_edges.items()) {
                SCOPED_TRACE(edge_key);
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
                if (edge.at("through_tensix_mux").get<bool>()) {
                    EXPECT_EQ(
                        target_router->at(channel_pointer).at("status"), lower_enum_name(manifest::ChannelStatus::MUX))
                        << target;
                }
            }
        }
    }
}

// Whether `field`'s argument is emitted on this fabric: a 2D-only argument is not emitted in 1D, and a capture-only
// argument only when the builder allocated the channel trimming capture.
bool is_emitted(const manifest::RouterField& field, bool is_2d) {
    const auto* arg = std::get_if<manifest::NamedArg>(&field.source);
    if (arg == nullptr) {
        return true;
    }
    switch (arg->emitted) {
        case manifest::Emitted::ALWAYS: return true;
        case manifest::Emitted::FABRIC_2D: return is_2d;
        case manifest::Emitted::CHANNEL_TRIMMING_CAPTURE:
            return builder_context().get_telemetry_and_metadata_buffer_map().channel_trimming_capture.is_enabled();
    }
    return false;
}

// `fields` holds exactly `table`'s fields emitted on this fabric, and only an L1 or stream field, a buffer or register
// the builder did not allocate, is null. Emission is worked out from the fabric and the builder's buffer map, not from
// the kernel arguments the router pass reads.
void expect_fields(const json& fields, std::span<const manifest::RouterField> table, bool is_2d) {
    std::set<std::string> expected_keys;
    for (const auto& field : table) {
        if (!is_emitted(field, is_2d)) {
            continue;
        }
        const std::string key(field.key.value);
        expected_keys.insert(key);
        if (fields.contains(key) && fields.at(key).is_null()) {
            EXPECT_TRUE(
                std::holds_alternative<manifest::content::L1>(field.content.value) ||
                std::holds_alternative<manifest::content::Stream>(field.content.value))
                << key;
        }
    }
    EXPECT_EQ(keys_of(fields), expected_keys);
}

// Every fields object holds exactly its table's fields emitted on this fabric: the router's, each ERISC's, and each
// channel's and edge's. The kernel's direction, a field, is the one ControlPlane gives the router's channel.
void check_router_fields(const json& manifest, const std::vector<RouterEntry>& routers) {
    const bool is_2d = manifest.at("fabric").at("is_2d_routing").get<bool>();
    const auto expect_channel_fields = [&](const json& channels, std::span<const manifest::RouterField> table) {
        for (const auto& [vc_key, vc_channels] : channels.items()) {
            for (const auto& [ch_key, channel] : vc_channels.items()) {
                SCOPED_TRACE(fmt::format("{}/{}", vc_key, ch_key));
                expect_fields(channel.at("fields"), table, is_2d);
            }
        }
    };
    for (const auto& entry : routers) {
        SCOPED_TRACE(entry.path);
        const auto& router = *entry.router;
        expect_fields(router.at("fields"), manifest::k_router_fields, is_2d);
        EXPECT_EQ(
            router.at("fields").at("my_direction").at("value").get<uint32_t>(),
            static_cast<uint32_t>(control_plane().get_eth_chan_direction(entry.node, entry.eth_chan)));
        for (const auto& [risc_key, erisc] : router.at("eriscs").items()) {
            SCOPED_TRACE(risc_key);
            expect_fields(erisc.at("fields"), manifest::k_erisc_fields, is_2d);
        }
        expect_channel_fields(router.at("channels").at("senders"), manifest::k_sender_channel_fields);
        expect_channel_fields(router.at("channels").at("receivers"), manifest::k_receiver_channel_fields);
        expect_channel_fields(router.at("intra_chip_downstream_edges"), manifest::k_downstream_edge_fields);
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
            const FabricNodeId node(
                MeshId{static_cast<uint32_t>(std::stoul(m_key.substr(1)))},
                static_cast<uint32_t>(std::stoul(c_key.substr(1))));
            const auto master = local_sync.at("master").get<std::string>();
            const auto chip_prefix = fmt::format("{}/{}/", m_key, c_key);
            ASSERT_TRUE(master.starts_with(chip_prefix)) << master << " is not on this chip";
            ASSERT_TRUE(routers.contains(key_of_path(master))) << master;
            const auto master_eth_chan = channel_for_key(node, key_of_path(master));
            ASSERT_TRUE(master_eth_chan.has_value()) << master;
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
                        EXPECT_EQ(chip_prefix + r_key, master) << risc_key;
                        ++num_masters;
                    }
                }
            }
            EXPECT_EQ(local_sync.at("num_routers"), routers.size());
            EXPECT_EQ(local_sync.at("router_channels_mask"), mask);
            EXPECT_NE(mask & (1u << *master_eth_chan), 0u) << "the master is not in router_channels_mask";
            EXPECT_EQ(num_masters, syncs ? 1u : 0u);
        }
    }
}

// A diagnostic buffer is where the builder's buffer map puts it, or null when the builder did not allocate it.
void expect_diagnostic_buffer(const json& field, const FabricRouterDiagnosticBufferMap::BufferRegion& region) {
    if (!region.is_enabled()) {
        EXPECT_TRUE(field.is_null()) << field;
        return;
    }
    ASSERT_FALSE(field.is_null());
    EXPECT_EQ(field.at("address").get<size_t>(), region.l1_address);
    EXPECT_EQ(field.at("size").get<size_t>(), region.size_bytes);
}

// The diagnostic buffers the kernel is fed are where the builder's buffer map puts them, and the router records a
// channel trimming capture exactly when the builder allocated one. notify_worker_src is where the builder's config puts
// it.
void check_router_diagnostics(const std::vector<RouterEntry>& routers) {
    const auto& context = builder_context();
    const auto buffers = context.get_telemetry_and_metadata_buffer_map();
    const auto notify_address = context.get_fabric_router_config().notify_worker_of_read_counter_update_src_address;
    for (const auto& entry : routers) {
        SCOPED_TRACE(entry.path);
        const auto& fields = entry.router->at("fields");
        expect_diagnostic_buffer(fields.at("perf_telemetry"), buffers.perf_telemetry);
        expect_diagnostic_buffer(fields.at("code_profiling"), buffers.code_profiling);
        const bool capture = buffers.channel_trimming_capture.is_enabled();
        EXPECT_EQ(fields.at("channel_trimming_capture_enabled").at("value").get<bool>(), capture);
        if (capture) {
            expect_diagnostic_buffer(fields.at("channel_trimming_capture"), buffers.channel_trimming_capture);
        }
        const auto& notify = fields.at("notify_worker_src");
        if (notify_address == 0) {
            EXPECT_TRUE(notify.is_null()) << notify;
        } else {
            ASSERT_FALSE(notify.is_null());
            EXPECT_EQ(notify.at("address").get<size_t>(), notify_address);
        }
    }
}

// The addresses of every region under `node` that the manifest marks cleared_by_host.
void collect_cleared_addresses(const json& node, std::set<size_t>& addresses) {
    if (!node.is_object()) {
        return;
    }
    if (node.contains("cleared_by_host")) {
        if (node.at("cleared_by_host").get<bool>()) {
            addresses.insert(node.at("address").get<size_t>());
        }
        return;
    }
    for (const auto& [key, child] : node.items()) {
        collect_cleared_addresses(child, addresses);
    }
}

// Every address the host zeroes before launch starts a region the manifest describes as cleared_by_host, and no other
// region is.
void check_router_host_cleared(const std::vector<RouterEntry>& routers) {
    const auto to_clear = builder_context().get_fabric_router_addresses_to_clear();
    const std::set<size_t> expected(to_clear.begin(), to_clear.end());
    for (const auto& entry : routers) {
        SCOPED_TRACE(entry.path);
        std::set<size_t> cleared;
        collect_cleared_addresses(*entry.router, cleared);
        EXPECT_EQ(cleared, expected);
    }
}

// Base firmware's link status words exist on Blackhole and not on Wormhole, and where base firmware has a heartbeat
// word, the router's heartbeat is that word.
void check_arch(const json& manifest) {
    using tt::tt_metal::FWMailboxMsg;
    const auto& hal = tt::tt_metal::MetalContext::instance().hal();
    const auto& areas = arch_areas(manifest);
    const bool is_blackhole = BaseFabricFixture::arch_ == tt::ARCH::BLACKHOLE;
    for (const char* key :
         {"base_fw_port_status",
          "base_fw_train_status",
          "base_fw_retrain_count",
          "base_fw_rx_link_up",
          "base_fw_postcode"}) {
        EXPECT_EQ(areas.at(key).is_null(), !is_blackhole) << key;
    }
    const uint32_t base_fw_heartbeat = hal.get_eth_fw_mailbox_val(FWMailboxMsg::HEARTBEAT);
    if (base_fw_heartbeat != 0) {
        EXPECT_EQ(areas.at("heartbeat").at("address").get<uint32_t>(), base_fw_heartbeat);
    }
}

// Every register of every stream is in the HAL's register space.
void check_arch_stream_registers(const json& manifest) {
    const auto& hal = tt::tt_metal::MetalContext::instance().hal();
    const auto& streams = manifest.at("arch").at("stream_registers");
    const auto base = streams.at("base").get<uint32_t>();
    const auto stride = streams.at("stride").get<uint32_t>();
    const auto count = streams.at("count").get<uint32_t>();
    for (const auto& [name, reg] : streams.at("registers").items()) {
        SCOPED_TRACE(name);
        const auto offset = reg.at("offset").get<uint32_t>();
        EXPECT_LT(offset, stride);
        for (uint32_t stream_id = 0; stream_id < count; ++stream_id) {
            EXPECT_TRUE(hal.valid_reg_addr(base + (stream_id * stride) + offset)) << stream_id;
        }
    }
}

// soft_reset has a distinct one-bit reset mask for each physical ERISC: one on Wormhole, two on Blackhole. Each router
// ERISC runs on the physical ERISC of its processor, except in Blackhole single-ERISC mode, where base firmware keeps
// ERISC0 and the one processor runs on ERISC1.
void check_arch_core_registers(const json& manifest, const std::vector<RouterEntry>& routers) {
    const auto arch = BaseFabricFixture::arch_;
    const auto& soft_reset = manifest.at("arch").at("core_registers").at(0);
    ASSERT_EQ(soft_reset.at("name"), "soft_reset");
    const auto masks = soft_reset.at("reset_masks").get<std::vector<uint32_t>>();
    EXPECT_EQ(masks.size(), arch == tt::ARCH::BLACKHOLE ? 2u : 1u);
    uint32_t seen = 0;
    for (const auto mask : masks) {
        EXPECT_EQ(std::popcount(mask), 1) << mask;
        EXPECT_EQ(seen & mask, 0u) << mask;
        seen |= mask;
    }

    const bool single_erisc_blackhole =
        arch == tt::ARCH::BLACKHOLE && !tt::tt_metal::MetalContext::instance().rtoptions().get_enable_2_erisc_mode();
    const auto riscv_1 = lower_enum_name(tt::tt_metal::DataMovementProcessor::RISCV_1);
    for (const auto& entry : routers) {
        SCOPED_TRACE(entry.path);
        std::set<uint32_t> used;
        for (const auto& [risc_key, erisc] : entry.router->at("eriscs").items()) {
            SCOPED_TRACE(risc_key);
            const auto physical = erisc.at("physical_erisc").get<uint32_t>();
            const bool on_riscv_1 = erisc.at("processor").get<std::string>() == riscv_1;
            EXPECT_EQ(physical, single_erisc_blackhole || on_riscv_1 ? 1u : 0u);
            EXPECT_LT(physical, masks.size());
            EXPECT_TRUE(used.insert(physical).second) << "two ERISCs run on physical ERISC " << physical;
        }
    }
}

}  // namespace

// ============ Tests ============

// The spellings the manifest's readers depend on, one case for each rule.
TEST(ManifestNames, Spellings) {
    EXPECT_EQ(mesh_key(MeshId{0}), "M0");
    EXPECT_EQ(chip_key(7), "C7");
    EXPECT_EQ(router_key(eth_chan_directions::EAST, 0), "E0");
    EXPECT_EQ(router_key(eth_chan_directions::WEST, 1), "W1");
    EXPECT_EQ(router_key(eth_chan_directions::NORTH, 2), "N2");
    EXPECT_EQ(router_key(eth_chan_directions::SOUTH, 3), "S3");
    EXPECT_EQ(router_key(eth_chan_directions::Z, 0), "Z0");

    // An enumerator in lower case, and a value with no enumerator as its number.
    EXPECT_EQ(lower_enum_name(FabricConfig::FABRIC_2D), "fabric_2d");
    EXPECT_EQ(lower_enum_name(static_cast<manifest::ChannelStatus>(200)), "200");

    // A content type's name in snake case.
    EXPECT_EQ(kind_name<manifest::content::L1>(), "l1");
    EXPECT_EQ(kind_name<manifest::content::Stream>(), "stream");

    namespace element = layout::element;
    EXPECT_EQ(schema_name({element::Uint{}, 4, 0}), "u32");
    EXPECT_EQ(schema_name({element::Int{}, 2, 0}), "i16");
    // An array's integers take their width from one element.
    EXPECT_EQ(schema_name({element::Uint{}, 8, 2}), "u32");
    EXPECT_EQ(schema_name({element::Enum{&layout::enum_description<RouterState>}, 4, 0}), "enum:RouterState");
    EXPECT_EQ(schema_name({element::Struct{&layout::struct_description<WorkerXY>}, 4, 0}), "struct:WorkerXY");
    EXPECT_EQ(
        schema_name(
            {element::TaggedUnion{&layout::tagged_union_description<NocCommandFields>, "noc_send_type"}, 40, 0}),
        "union:NocCommandFields");
    EXPECT_EQ(schema_name({element::Bytes{}, 1, 0}), "bytes");
    EXPECT_EQ(schema_name({element::Pad{}, 1, 0}), "pad");
}

// Collecting a struct should also collect every struct and enum its members name, and theirs in turn, so the types
// block describes everything a schema leads to.
TEST(ManifestTypes, CollectorFollowsMembers) {
    manifest::TypeCollector types;
    types.add(layout::type_of<FabricTelemetry>());
    for (const std::string_view name :
         {"FabricTelemetry",
          "StaticInfo",
          "DynamicInfo",
          "BandwidthTelemetry",
          "EriscDynamicEntry",
          "RiscTimestampV2"}) {
        EXPECT_TRUE(types.get_structs().contains(name)) << name;
    }
    EXPECT_TRUE(types.get_enums().contains("RouterState"));
    EXPECT_TRUE(types.get_tagged_unions().empty());

    for (const auto& [name, description] : types.get_structs()) {
        for (const auto& member : description->members) {
            if (const auto* s = std::get_if<layout::element::Struct>(&member.type.element)) {
                EXPECT_TRUE(types.get_structs().contains(s->description->name)) << name << "." << member.name;
            } else if (const auto* e = std::get_if<layout::element::Enum>(&member.type.element)) {
                EXPECT_TRUE(types.get_enums().contains(e->description->name)) << name << "." << member.name;
            }
        }
    }
}

// Collecting a tagged union should collect its tag enum and every type its cases name, since decode reads the tag and
// then the case it picks.
TEST(ManifestTypes, CollectorFollowsTaggedUnionCases) {
    manifest::TypeCollector types;
    types.add(
        {layout::element::TaggedUnion{&layout::tagged_union_description<NocCommandFields>, "noc_send_type"},
         sizeof(NocCommandFields),
         0});
    EXPECT_TRUE(types.get_tagged_unions().contains("NocCommandFields"));
    EXPECT_TRUE(types.get_enums().contains("NocSendType"));
    EXPECT_TRUE(types.get_structs().contains("NocUnicastCommandHeader"));
    EXPECT_TRUE(types.get_structs().contains("NocUnicastScatterCommandHeader"));
}

// A type with nothing to describe, such as an integer, should add nothing.
TEST(ManifestTypes, CollectorSkipsIntegers) {
    manifest::TypeCollector types;
    types.add(layout::type_of<uint32_t>());
    types.add(layout::type_of<uint8_t[4]>());
    EXPECT_TRUE(types.get_structs().empty());
    EXPECT_TRUE(types.get_enums().empty());
    EXPECT_TRUE(types.get_tagged_unions().empty());
}

// Two different types with one name would make that name's schema ambiguous, so collecting the second should fail.
TEST(ManifestTypes, CollectorRejectsTwoTypesWithOneName) {
    static constexpr layout::StructDescription first{"Twin", 4, {}};
    static constexpr layout::StructDescription second{"Twin", 4, {}};
    manifest::TypeCollector types;
    types.add({layout::element::Struct{&first}, 4, 0});
    types.add({layout::element::Struct{&first}, 4, 0});
    EXPECT_ANY_THROW(types.add({layout::element::Struct{&second}, 4, 0}));
}

// A chip is published once, with every router its builder initialized, and only within the chips the context holds.
TEST(ManifestPublishedChips, RejectsWhatItCannotHold) {
    manifest::PublishedChips chips(2);
    EXPECT_ANY_THROW(chips.publish(0, manifest::Chip{.routers = {manifest::Router{}}}, 2));
    EXPECT_ANY_THROW(chips.publish(2, manifest::Chip{}, 0));
    chips.publish(0, manifest::Chip{}, 0);
    EXPECT_ANY_THROW(chips.publish(0, manifest::Chip{}, 0));
}

TEST(ManifestArgs, SameFamily) {
    EXPECT_TRUE(manifest::same_arg_family("SENDER_CH_7_IS_INJECTION", "SENDER_CH_{}_IS_INJECTION"));
    EXPECT_TRUE(manifest::same_arg_family("SENDER_CH_7_IS_INJECTION", "SENDER_CH_12_IS_INJECTION"));
    EXPECT_TRUE(manifest::same_arg_family(
        "VC1_FREE_SLOTS_FROM_DOWNSTREAM_EDGE_3_STREAM_ID", "VC{}_FREE_SLOTS_FROM_DOWNSTREAM_EDGE_{}_STREAM_ID"));
    EXPECT_FALSE(manifest::same_arg_family("SENDER_CH_IS_INJECTION", "SENDER_CH_{}_IS_INJECTION"));
    EXPECT_FALSE(manifest::same_arg_family("SENDER_CH_7_IS_INJECTION", "SENDER_CH_{}_IS_INJECTIONS"));
    EXPECT_FALSE(manifest::same_arg_family("SENDER_CH_7", "SENDER_CH_{}_IS_INJECTION"));
}

// A router's inputs pass when each is read or listed as unrecorded, and fail on one that is neither.
TEST(ManifestArgs, KernelInputsAccounted) {
    const RouterKernelInputs kernel{
        .defines = {{"FABRIC_2D", ""}, {"FABRIC_2D_VC1_SERVICED", ""}},
        .processors = {},
        .named_ct_args = {{
            {"EDM_STATUS_PTR_ADDR", 0},
            {"SENDER_CH_7_IS_INJECTION", 0},
            {"NUM_ACTIVE_ERISCS", 1},
            {"MAX_NUM_VCS", 3},
        }},
    };
    EXPECT_NO_THROW(check_kernel_inputs_accounted(kernel));

    auto new_arg = kernel;
    new_arg.named_ct_args.front()["SOME_NEW_ARG"] = 0;
    EXPECT_THROW(check_kernel_inputs_accounted(new_arg), std::exception);

    auto new_define = kernel;
    new_define.defines["SOME_NEW_DEFINE"] = "";
    EXPECT_THROW(check_kernel_inputs_accounted(new_define), std::exception);
}

// Every check, one test each, for `Fixture`.
#define TEST_MANIFEST_CHECKS(Fixture)                                                                      \
    TEST_F(Fixture, TopLevel) { check_top_level(manifest_, manifest_path_, fabric_config, suite_start_); } \
    TEST_F(Fixture, ChipZPortRole) { check_chip_z_port_role(manifest_); }                                  \
    TEST_F(Fixture, RouterLink) { check_router_link(manifest_, routers_); }                                \
    TEST_F(Fixture, PeersAreSymmetric) { check_peers_are_symmetric(manifest_, routers_); }                 \
    TEST_F(Fixture, CreditTransport) { check_credit_transport(manifest_, routers_); }                      \
    TEST_F(Fixture, RouterSenders) { check_router_senders(manifest_, routers_); }                          \
    TEST_F(Fixture, RouterDownstreamEdges) { check_router_edges(manifest_, routers_); }                    \
    TEST_F(Fixture, RouterFields) { check_router_fields(manifest_, routers_); }                            \
    TEST_F(Fixture, LocalSync) { check_local_sync(manifest_); }                                            \
    TEST_F(Fixture, RouterRegionsDisjoint) { check_router_regions_disjoint(manifest_, routers_); }         \
    TEST_F(Fixture, Arch) { check_arch(manifest_); }                                                       \
    TEST_F(Fixture, ArchStreamRegisters) { check_arch_stream_registers(manifest_); }                       \
    TEST_F(Fixture, ArchCoreRegisters) { check_arch_core_registers(manifest_, routers_); }                 \
    TEST_F(Fixture, RouterDiagnostics) { check_router_diagnostics(routers_); }                             \
    TEST_F(Fixture, RouterHostCleared) { check_router_host_cleared(routers_); }

TEST_MANIFEST_CHECKS(Fabric1DManifestFixture)
TEST_MANIFEST_CHECKS(Fabric2DManifestFixture)
TEST_MANIFEST_CHECKS(FabricTorusXManifestFixture)
TEST_MANIFEST_CHECKS(FabricTorusYManifestFixture)
TEST_MANIFEST_CHECKS(FabricTorusXYManifestFixture)
TEST_MANIFEST_CHECKS(FabricMuxManifestFixture)

}  // namespace tt::tt_fabric::fabric_router_tests
