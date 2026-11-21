// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Checks the fabric manifest written during fabric init against the live fabric. Each check covers one part of the
// manifest and runs under every fabric configuration a fixture brings up. CapturePlan checks capture's reading of the
// same manifest, so a writer change that capture doesn't follow fails here, on every configuration.

#include <gtest/gtest.h>
#include <fmt/format.h>
#include <nlohmann/json.hpp>
#include <tt-metalium/distributed_context.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <regex>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
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
#include "tt_metal/fabric/debug/visualizer/capture/fabric_capture.hpp"
#include "tt_metal/fabric/debug/visualizer/capture/fabric_capture_device.hpp"
#include "tt_metal/fabric/debug/visualizer/capture/fabric_capture_load.hpp"
#include "tt_metal/fabric/debug/visualizer/capture/fabric_capture_manifest.hpp"
#include "tt_metal/fabric/debug/visualizer/capture/fabric_capture_record.hpp"
#include "tt_metal/fabric/debug/visualizer/capture/fabric_capture_targets.hpp"
#include "tt_metal/fabric/debug/visualizer/capture/fabric_capture_umd.hpp"
#include "tt_metal/fabric/debug/visualizer/capture/fabric_capture_utils.hpp"
#include "tt_metal/fabric/debug/visualizer/common/fabric_visualizer_files.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_published_chips.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_router_pass.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_fields.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_model.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_names.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_paths.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_type_registry.hpp"

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
        const manifest::RouterKey chan_key{
            .direction = direction, .routing_plane = control_plane().get_routing_plane_id(node, chan)};
        if (router_key(chan_key) == key) {
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

// Calls f(where, holder) for every node on the manifest's local chips that holds a fields key, in the order they're in
// the file: each router's sender and receiver channels, its downstream edges, then the router itself. `where` is the
// holder's path, for messages.
template <typename Manifest, typename F>
void for_each_fields_holder(Manifest& manifest, F&& f) {
    for (auto& [m_key, mesh] : manifest.at("meshes").items()) {
        for (auto& [c_key, chip] : mesh.at("chips").items()) {
            if (!chip.at("is_local").template get<bool>()) {
                continue;
            }
            for (auto& [r_key, router] : chip.at("routers").items()) {
                const std::string router_path = fmt::format("{}/{}/{}", m_key, c_key, r_key);
                for (const char* kind : {"senders", "receivers"}) {
                    for (auto& [vc_key, channels] : router.at("channels").at(kind).items()) {
                        for (auto& [ch_key, channel] : channels.items()) {
                            f(fmt::format("{}/channels/{}/{}/{}", router_path, kind, vc_key, ch_key), channel);
                        }
                    }
                }
                for (auto& [vc_key, edges] : router.at("intra_chip_downstream_edges").items()) {
                    for (auto& [edge_key, edge] : edges.items()) {
                        f(fmt::format("{}/intra_chip_downstream_edges/{}/{}", router_path, vc_key, edge_key), edge);
                    }
                }
                f(router_path, router);
            }
        }
    }
}

// `written` as its readers see it: each fields id replaced by its block from field_sets, and field_sets dropped.
json expand_field_sets(const nlohmann::ordered_json& written) {
    json manifest(written);
    const json field_sets = manifest.at("field_sets");
    manifest.erase("field_sets");
    for_each_fields_holder(manifest, [&](const std::string& where, json& holder) {
        const std::string id = holder.at("fields").get<std::string>();
        if (!field_sets.contains(id)) {
            throw std::runtime_error(fmt::format("{}'s fields {} isn't in field_sets", where, id));
        }
        holder["fields"] = field_sets.at(id);
    });
    return manifest;
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

        // This rank's manifest, in the newest run's folder: the one this suite's fabric init just wrote.
        const auto world = tt::tt_metal::distributed::multihost::DistributedContext::get_world_context();
        const std::string manifest_file_name = visualizer::make_manifest_file_name(
            static_cast<uint32_t>(*world->rank()), static_cast<uint32_t>(*world->size()));
        const auto run_directory = visualizer::find_latest_run_directory(
            tt::tt_metal::MetalContext::instance().rtoptions().get_logs_dir(), manifest_file_name);
        ASSERT_TRUE(run_directory.has_value()) << "no run folder holds " << manifest_file_name;
        manifest_path_ = *run_directory / manifest_file_name;
        std::ifstream stream(manifest_path_);
        ASSERT_TRUE(stream.is_open()) << manifest_path_;
        written_manifest_ = nlohmann::ordered_json::parse(stream);
        manifest_ = expand_field_sets(written_manifest_);
        index_routers(manifest_, routers_);
    }

    // Used for restoring the generate_manifest runtime arg value to its initial value after the test.
    inline static bool generate_manifest_before_ = false;
    // When this suite started bringing up fabric. A manifest written before it is a stale one.
    inline static std::filesystem::file_time_type suite_start_;
    std::filesystem::path manifest_path_;
    // The manifest as written, and with its field_sets expanded, which every check but FieldSets reads.
    nlohmann::ordered_json written_manifest_;
    json manifest_;
    std::vector<RouterEntry> routers_;
};

using Manifest1DFixture = FabricManifestFixture<FabricConfig::FABRIC_1D>;
using Manifest2DFixture = FabricManifestFixture<FabricConfig::FABRIC_2D>;
using ManifestTorusXFixture = FabricManifestFixture<FabricConfig::FABRIC_2D_TORUS_X>;
using ManifestTorusYFixture = FabricManifestFixture<FabricConfig::FABRIC_2D_TORUS_Y>;
using ManifestTorusXYFixture = FabricManifestFixture<FabricConfig::FABRIC_2D_TORUS_XY>;
using ManifestMuxFixture = FabricManifestFixture<FabricConfig::FABRIC_2D, FabricTensixConfig::MUX>;

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

    // The run id is a UTC time and 8 hex digits, and it names the folder the manifest is in.
    const std::string run_id = manifest.at("run").at("run_id").get<std::string>();
    EXPECT_TRUE(std::regex_match(run_id, std::regex("[0-9]{8}T[0-9]{6}Z-[0-9a-f]{8}"))) << run_id;
    EXPECT_EQ(manifest_path.parent_path().filename(), run_id);

    // Other ranks write their manifests into the same directory, and may still be writing.
    const auto temp_prefix = manifest_path.filename().string() + ".tmp.";
    for (const auto& entry : std::filesystem::directory_iterator(manifest_path.parent_path())) {
        EXPECT_FALSE(entry.path().filename().string().starts_with(temp_prefix)) << entry.path();
    }
}

// The manifest as written holds each distinct fields block once: every fields id names a block in field_sets, every
// block is named, no two blocks are the same, and the blocks are keyed F0, F1, ... in the order they're first named.
void check_field_sets(const nlohmann::ordered_json& written_manifest) {
    const nlohmann::ordered_json& field_sets = written_manifest.at("field_sets");
    std::vector<std::string> first_named;
    std::set<std::string> named;
    for_each_fields_holder(written_manifest, [&](const std::string& where, const nlohmann::ordered_json& holder) {
        const std::string id = holder.at("fields").get<std::string>();
        EXPECT_TRUE(field_sets.contains(id)) << where << "'s fields " << id << " isn't in field_sets";
        if (named.insert(id).second) {
            first_named.push_back(id);
        }
    });
    std::vector<std::string> keys;
    std::set<std::string> blocks;
    for (const auto& [key, block] : field_sets.items()) {
        keys.push_back(key);
        EXPECT_TRUE(blocks.insert(block.dump()).second) << key << " is the same as an earlier block";
    }
    std::vector<std::string> expected_keys;
    for (size_t index = 0; index < first_named.size(); ++index) {
        expected_keys.push_back(manifest::field_set_key(index));
    }
    EXPECT_EQ(first_named, expected_keys);
    EXPECT_EQ(keys, expected_keys);
}

// The first operations of the patch from `expected` to `actual`, each naming the pointer that differs.
std::string describe_json_diff(const nlohmann::ordered_json& expected, const nlohmann::ordered_json& actual) {
    constexpr size_t k_max_operations = 20;
    const nlohmann::ordered_json patch = nlohmann::ordered_json::diff(expected, actual);
    std::string out = fmt::format("{} differences", patch.size());
    for (size_t index = 0; index < std::min(patch.size(), k_max_operations); ++index) {
        std::string operation = patch[index].dump();
        if (operation.size() > 300) {
            operation.resize(300);
            operation += "...";
        }
        out += "\n  " + operation;
    }
    return out;
}

// Reading the manifest and writing it again gives back the same JSON, in each fields layout: the writer and the reader
// agree on every key and value this run's fabric has.
void check_manifest_writer_parser_round_trip(const nlohmann::ordered_json& written_manifest) {
    // Shared fields layout check
    const manifest::ManifestFacts facts = parse_manifest_json(written_manifest);
    const nlohmann::ordered_json rewritten = make_manifest_json(facts, FieldsLayout::SHARED);
    EXPECT_TRUE(rewritten == written_manifest) << describe_json_diff(written_manifest, rewritten);

    // Per-router fields layout check
    const nlohmann::ordered_json per_router = make_manifest_json(facts, FieldsLayout::PER_ROUTER);
    const nlohmann::ordered_json per_router_rewritten =
        make_manifest_json(parse_manifest_json(per_router), FieldsLayout::PER_ROUTER);
    EXPECT_TRUE(per_router_rewritten == per_router) << describe_json_diff(per_router, per_router_rewritten);
}

// Every chip names the host rank that owns it. With one local mesh, the local chips are this process's: their host rank
// is the one ControlPlane binds this process to.
void check_chip_host_ranks(const json& manifest) {
    const bool one_local_mesh = control_plane().get_local_mesh_id_bindings().size() == 1;
    const auto local_host_rank = *control_plane().get_local_host_rank_id_binding();
    for (const auto& [m_key, mesh] : manifest.at("meshes").items()) {
        for (const auto& [c_key, chip] : mesh.at("chips").items()) {
            SCOPED_TRACE(m_key + "/" + c_key);
            ASSERT_FALSE(chip.at("host_rank").is_null());
            if (one_local_mesh && chip.at("is_local").get<bool>()) {
                EXPECT_EQ(chip.at("host_rank").get<uint32_t>(), local_host_rank);
            }
        }
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

// Every VC some router of a local mesh has senders on has a credit transport entry, backed by L1 counters: the router
// takes every VC's credits through them (router_sender.hpp static_asserts it), and each sender's credits name them.
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
        const auto& transport = mesh.at("credit_transport");
        EXPECT_TRUE(std::ranges::includes(keys_of(transport), vcs_with_senders));
        for (const auto& vc_key : vcs_with_senders) {
            if (transport.contains(vc_key)) {
                EXPECT_EQ(transport.at(vc_key).at("backing"), lower_enum_name(manifest::CreditBacking::L1_COUNTER))
                    << vc_key;
            }
        }
    }
}

// A router has VC1 senders only when the fabric's inter-mesh VC setup requires VC1, and VC2 senders only when it
// requires VC2.
void check_intermesh_vc(const json& manifest, const std::vector<RouterEntry>& routers) {
    const auto& intermesh_vc = manifest.at("fabric").at("intermesh_vc");
    for (const auto& entry : routers) {
        SCOPED_TRACE(entry.path);
        const auto& senders = entry.router->at("shape").at("senders_per_vc");
        if (senders.at(1).get<uint32_t>() > 0) {
            EXPECT_TRUE(intermesh_vc.at("requires_vc1").get<bool>());
        }
        if (senders.at(2).get<uint32_t>() > 0) {
            EXPECT_TRUE(intermesh_vc.at("requires_vc2").get<bool>());
        }
    }
}

// A credit is the sender's own element of the router's counter array, which is indexed by the router's sender compact
// index and must be long enough for it.
void expect_credit(const json& credit, const json& router, manifest::CreditCounterArray array, uint32_t compact) {
    ASSERT_TRUE(credit.is_object() && credit.contains("array")) << credit;
    EXPECT_EQ(credit.at("array"), fmt::format("credit_counters/{}", lower_enum_name(array)));
    EXPECT_EQ(credit.at("index"), compact);
    const json::json_pointer pointer("/" + credit.at("array").get<std::string>());
    ASSERT_TRUE(router.contains(pointer)) << credit.at("array");
    EXPECT_LT(compact, router.at(pointer).at("num_elements").get<uint32_t>());
}

// The bytes an L1 region spans.
uint32_t get_region_size(const json& region) {
    return region.at("num_elements").get<uint32_t>() * region.at("element_size").get<uint32_t>();
}

// Every L1 region under `node`, as (address, size, path). Stream and core registers have an address too, but no
// element_size.
void collect_regions(
    const json& node, const std::string& path, std::vector<std::tuple<uint32_t, uint32_t, std::string>>& regions) {
    if (!node.is_object()) {
        return;
    }
    if (node.contains("address") && node.contains("element_size")) {
        regions.emplace_back(node.at("address").get<uint32_t>(), get_region_size(node), path);
        return;
    }
    for (const auto& [key, child] : node.items()) {
        collect_regions(child, path.empty() ? key : path + "/" + key, regions);
    }
}

// The arch's L1 areas, which every router's core has.
const json& arch_areas(const json& manifest) { return manifest.at("arch").at("l1_areas"); }

// A region's path with its ERISC left out, e.g. "fields/perf_telemetry" for "fields/erisc1/perf_telemetry".
std::string remove_erisc_from_path(const std::string& path) {
    std::string out;
    std::string_view rest = path;
    while (!rest.empty()) {
        const size_t slash = rest.find('/');
        const std::string_view part = rest.substr(0, slash);
        if (!(part.starts_with("erisc") && part.size() > 5 &&
              std::ranges::all_of(part.substr(5), [](char c) { return c >= '0' && c <= '9'; }))) {
            out += out.empty() ? std::string(part) : "/" + std::string(part);
        }
        rest = slash == std::string_view::npos ? std::string_view{} : rest.substr(slash + 1);
    }
    return out;
}

// No two of a router's L1 regions overlap, counting the arch areas on its core. A field each ERISC is fed the same
// region for is one region, so it counts once.
void check_router_regions_disjoint(const json& manifest, const std::vector<RouterEntry>& routers) {
    std::vector<std::tuple<uint32_t, uint32_t, std::string>> areas;
    collect_regions(arch_areas(manifest), "l1_areas", areas);
    for (const auto& entry : routers) {
        SCOPED_TRACE(entry.path);
        std::vector<std::tuple<uint32_t, uint32_t, std::string>> router_regions;
        collect_regions(*entry.router, "", router_regions);
        auto regions = areas;
        std::set<std::tuple<uint32_t, uint32_t, std::string>> seen;
        for (const auto& [address, size, path] : router_regions) {
            if (seen.emplace(address, size, remove_erisc_from_path(path)).second) {
                regions.emplace_back(address, size, path);
            }
        }
        std::ranges::sort(regions);
        for (size_t i = 1; i < regions.size(); ++i) {
            const auto& [address, size, name] = regions[i - 1];
            const auto& [next_address, next_size, next_name] = regions[i];
            EXPECT_LE(address + size, next_address) << name << " overlaps " << next_name;
        }
    }
}

// Every L1 region a router names, and every arch area, lies inside the L1 of the Ethernet core it runs on.
void check_router_regions_in_l1(const json& manifest, const std::vector<RouterEntry>& routers) {
    const auto& l1 = manifest.at("arch").at("l1");
    const uint64_t l1_start = l1.at("address").get<uint32_t>();
    const uint64_t l1_end = l1_start + l1.at("size_bytes").get<uint32_t>();
    std::vector<std::tuple<uint32_t, uint32_t, std::string>> regions;
    collect_regions(arch_areas(manifest), "l1_areas", regions);
    for (const auto& entry : routers) {
        collect_regions(*entry.router, entry.path, regions);
    }
    for (const auto& [address, size, path] : regions) {
        EXPECT_GE(address, l1_start) << path;
        EXPECT_LE(uint64_t{address} + size, l1_end) << path;
    }
}

// The last component of a router path, e.g. "E0" for "M0/C7/E0".
std::string key_of_path(const std::string& path) { return path.substr(path.rfind('/') + 1); }

// A ring buffer's elements are its slots, each one channel buffer that starts with the run's packet header.
void expect_ring_buffer(const json& ring, uint32_t channel_buffer_size, const json& packet_header) {
    if (ring.is_null()) {
        return;
    }
    EXPECT_EQ(ring.at("element_size"), channel_buffer_size);
    EXPECT_EQ(ring.at("type"), packet_header);
}

// A router's sender channels, over its shape. Each one's producer is the chip's worker on a VC's first channel (the
// tensix mux on VC0 in mux mode), or a different router on the same chip and routing plane, feeding at most one
// channel per VC; in 1D a router and its producer feed each other. Nothing feeds a channel the mux carries. Only the
// worker channel has a producer cursor (buffer_index_sem): a sibling router keeps none. Each credit is the channel's
// element of the router's to_sender counter arrays. A router has trimmed channels only when a trimming profile is set.
void check_router_senders(const json& manifest, const std::vector<RouterEntry>& routers) {
    const bool is_2d = manifest.at("fabric").at("is_2d_routing").get<bool>();
    const auto channel_buffer_size = manifest.at("fabric").at("channel_buffer_size_bytes").get<uint32_t>();
    const bool has_trimming_profile = tt::tt_metal::MetalContext::instance().rtoptions().has_fabric_trimming_profile();

    for (const auto& entry : routers) {
        SCOPED_TRACE(entry.path);
        const auto& router = *entry.router;
        const auto& shape = router.at("shape");
        const auto& senders = router.at("channels").at("senders");
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

            std::set<std::string> producers;
            for (uint32_t ch = 0; ch < count; ++ch, ++compact) {
                const auto ch_key = fmt::format("ch{}", ch);
                SCOPED_TRACE(fmt::format("senders/{}/{}", vc_key, ch_key));
                ASSERT_TRUE(senders.at(vc_key).contains(ch_key));
                const auto& sender = senders.at(vc_key).at(ch_key);

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
                for (const auto& [risc_key, fields] : sender.at("fields").items()) {
                    EXPECT_EQ(fields.at("buffer_index_sem").is_null(), !worker_channel) << risc_key;
                }

                expect_ring_buffer(
                    sender.at("ring_buffer"), channel_buffer_size, manifest.at("fabric").at("packet_header"));
                // Each ERISC's credits: acked only on VC0, on an ERISC with bubble flow control.
                const auto& credits_by_erisc = sender.at("credits");
                EXPECT_EQ(keys_of(credits_by_erisc), keys_of(router.at("eriscs")));
                for (const auto& [risc_key, credits] : credits_by_erisc.items()) {
                    SCOPED_TRACE(risc_key);
                    const bool bubble_flow_control =
                        router.at("fields").at(risc_key).at("vc0_bubble_flow_control").at("value").get<bool>();
                    EXPECT_EQ(credits.contains("acked"), vc == 0 && bubble_flow_control);
                    if (credits.contains("acked")) {
                        expect_credit(
                            credits.at("acked"), router, manifest::CreditCounterArray::TO_SENDER_ACK, compact);
                    }
                    expect_credit(
                        credits.at("completed"), router, manifest::CreditCounterArray::TO_SENDER_COMPLETION, compact);
                }
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
                        router_key(
                            {.direction = builder::direction_from_compact_index(facing, n - 1),
                             .routing_plane = plane}));
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

// Whether `field`'s argument is emitted on this fabric: a 2D-only argument is not emitted in 1D, a capture-only
// argument only when the builder allocated the channel trimming capture, a progress-log-only argument only when it
// placed the channel progress log, and a deadlock-avoidance argument only on an ERISC with deadlock avoidance on.
bool is_emitted(const manifest::RouterField& field, bool is_2d, bool deadlock_avoidance) {
    const auto* arg = std::get_if<manifest::NamedArg>(&field.source);
    if (arg == nullptr) {
        return true;
    }
    switch (arg->emitted) {
        case manifest::Emitted::ALWAYS: return true;
        case manifest::Emitted::FABRIC_2D: return is_2d;
        case manifest::Emitted::CHANNEL_TRIMMING_CAPTURE:
            return builder_context().get_telemetry_and_metadata_buffer_map().channel_trimming_capture.is_enabled();
        case manifest::Emitted::CHANNEL_PROGRESS_LOG:
            return builder_context().get_telemetry_and_metadata_buffer_map().channel_progress_log.is_enabled();
        case manifest::Emitted::DEADLOCK_AVOIDANCE: return deadlock_avoidance;
    }
    return false;
}

// `fields` holds exactly `table`'s fields emitted on this fabric, and only an L1 or stream field, a buffer or register
// the builder did not allocate, is null. Emission is worked out from the fabric and the builder's buffer map, not from
// the kernel arguments the router pass reads, except deadlock avoidance: the builder decides it per router, partly from
// the trimming fast path, so a deadlock-avoidance field is expected exactly where the ERISC's own
// vc0_bubble_flow_control field is set.
void expect_fields(const json& fields, std::span<const manifest::RouterField> table, bool is_2d) {
    const bool deadlock_avoidance =
        fields.contains("vc0_bubble_flow_control") && fields.at("vc0_bubble_flow_control").at("value").get<bool>();
    std::set<std::string> expected_keys;
    for (const auto& field : table) {
        if (!is_emitted(field, is_2d, deadlock_avoidance)) {
            continue;
        }
        const std::string key(field.key.value);
        expected_keys.insert(key);
        if (fields.contains(key) && fields.at(key).is_null()) {
            EXPECT_TRUE(
                std::holds_alternative<manifest::L1Field>(field.content.value) ||
                std::holds_alternative<manifest::StreamRegField>(field.content.value))
                << key;
        }
    }
    EXPECT_EQ(keys_of(fields), expected_keys);
}

// A fields object has one entry per ERISC of the router, each holding exactly `table`'s fields emitted on this fabric.
void expect_fields_by_erisc(
    const json& fields_by_erisc, const json& eriscs, std::span<const manifest::RouterField> table, bool is_2d) {
    EXPECT_EQ(keys_of(fields_by_erisc), keys_of(eriscs));
    for (const auto& [risc_key, fields] : fields_by_erisc.items()) {
        SCOPED_TRACE(risc_key);
        expect_fields(fields, table, is_2d);
    }
}

// Every fields object holds each ERISC's fields, exactly its table's fields emitted on this fabric: the router's, and
// each channel's and edge's. The kernel's direction, a field, is the one ControlPlane gives the router's channel on
// every ERISC.
void check_router_fields(const json& manifest, const std::vector<RouterEntry>& routers) {
    const bool is_2d = manifest.at("fabric").at("is_2d_routing").get<bool>();
    for (const auto& entry : routers) {
        SCOPED_TRACE(entry.path);
        const auto& router = *entry.router;
        const auto& eriscs = router.at("eriscs");
        const auto expect_channel_fields = [&](const json& channels, std::span<const manifest::RouterField> table) {
            for (const auto& [vc_key, vc_channels] : channels.items()) {
                for (const auto& [ch_key, channel] : vc_channels.items()) {
                    SCOPED_TRACE(fmt::format("{}/{}", vc_key, ch_key));
                    expect_fields_by_erisc(channel.at("fields"), eriscs, table, is_2d);
                }
            }
        };
        expect_fields_by_erisc(router.at("fields"), eriscs, manifest::k_router_fields, is_2d);
        for (const auto& [risc_key, fields] : router.at("fields").items()) {
            EXPECT_EQ(
                fields.at("my_direction").at("value").get<uint32_t>(),
                static_cast<uint32_t>(control_plane().get_eth_chan_direction(entry.node, entry.eth_chan)))
                << risc_key;
            // Bubble flow control's injection floor is the bubble's two slots, at least.
            if (fields.contains("injection_min_free_slots")) {
                EXPECT_GE(fields.at("injection_min_free_slots").at("value").get<uint32_t>(), 2u) << risc_key;
            }
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
                // An ERISC reads the local sync arguments only with its wait_for_host_signal.
                for (const auto& [risc_key, fields] : router.at("fields").items()) {
                    if (!fields.at("wait_for_host_signal").at("value").get<bool>()) {
                        continue;
                    }
                    syncs = true;
                    if (fields.at("local_handshake_master").at("value").get<bool>()) {
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
    EXPECT_EQ(get_region_size(field), region.size_bytes);
}

// The diagnostic buffers the kernel is fed are where the builder's buffer map puts them, and the router records a
// channel trimming capture and a channel progress log exactly when the builder allocated them. notify_worker_src is
// where the builder's config puts it.
void check_router_diagnostics(const std::vector<RouterEntry>& routers) {
    const auto& context = builder_context();
    const auto buffers = context.get_telemetry_and_metadata_buffer_map();
    const auto notify_address = context.get_fabric_router_config().notify_worker_of_read_counter_update_src_address;
    for (const auto& entry : routers) {
        for (const auto& [risc_key, fields] : entry.router->at("fields").items()) {
            SCOPED_TRACE(fmt::format("{} {}", entry.path, risc_key));
            expect_diagnostic_buffer(fields.at("perf_telemetry"), buffers.perf_telemetry);
            expect_diagnostic_buffer(fields.at("code_profiling"), buffers.code_profiling);
            const bool capture = buffers.channel_trimming_capture.is_enabled();
            EXPECT_EQ(fields.at("channel_trimming_capture_enabled").at("value").get<bool>(), capture);
            if (capture) {
                expect_diagnostic_buffer(fields.at("channel_trimming_capture"), buffers.channel_trimming_capture);
            }
            const bool progress_log = buffers.channel_progress_log.is_enabled();
            EXPECT_EQ(fields.at("channel_progress_log_enabled").at("value").get<bool>(), progress_log);
            if (progress_log) {
                expect_diagnostic_buffer(fields.at("channel_progress_log"), buffers.channel_progress_log);
            }
            const auto& notify = fields.at("notify_worker_src");
            if (notify_address == 0) {
                EXPECT_TRUE(notify.is_null()) << notify;
            } else {
                ASSERT_FALSE(notify.is_null());
                EXPECT_EQ(notify.at("address").get<size_t>(), notify_address);
            }
            // A gated region is active exactly when its gate is nonzero on the same ERISC, as Blackhole's perf
            // telemetry buffer is allocated with bandwidth telemetry off but not written.
            for (const auto& field : manifest::k_router_fields) {
                if (!field.written_when.has_value()) {
                    continue;
                }
                const auto region = fields.find(std::string(field.key.value));
                if (region == fields.end() || region->is_null()) {
                    continue;
                }
                const auto& gate = fields.at(std::string(*field.written_when)).at("value");
                const bool gate_on = gate.is_boolean() ? gate.get<bool>() : gate.get<uint32_t>() != 0;
                EXPECT_EQ(region->at("active").get<bool>(), gate_on) << field.key.value;
            }
        }
    }
}

// Leftover L1 is never active, and a router's credit counter arrays are active exactly when its mesh takes some VC's
// credits through L1 counters.
void check_router_regions_active(const json& manifest, const std::vector<RouterEntry>& routers) {
    for (const auto& entry : routers) {
        SCOPED_TRACE(entry.path);
        const auto& router = *entry.router;
        if (!router.at("leftover_l1").is_null()) {
            EXPECT_FALSE(router.at("leftover_l1").at("active").get<bool>());
        }
        const auto& mesh = manifest.at("meshes").at(mesh_key(entry.node.mesh_id));
        bool mesh_uses_counters = false;
        for (const auto& [vc_key, vc_transport] : mesh.at("credit_transport").items()) {
            mesh_uses_counters = mesh_uses_counters ||
                                 vc_transport.at("backing") == lower_enum_name(manifest::CreditBacking::L1_COUNTER);
        }
        for (const auto& [array_key, array] : router.at("credit_counters").items()) {
            EXPECT_EQ(array.at("active").get<bool>(), mesh_uses_counters) << array_key;
        }
    }
}

// The start address of every L1 region under `node`.
void collect_region_addresses(const json& node, std::set<size_t>& addresses) {
    if (!node.is_object()) {
        return;
    }
    if (node.contains("element_size")) {
        addresses.insert(node.at("address").get<size_t>());
        return;
    }
    for (const auto& [key, child] : node.items()) {
        collect_region_addresses(child, addresses);
    }
}

// The host clears are the builder context's clear list, each the size of fabric init's zero buffer, and each starts a
// region every router describes.
void check_host_clears(const json& manifest, const std::vector<RouterEntry>& routers) {
    const auto to_clear = builder_context().get_fabric_router_addresses_to_clear();
    const auto clear_size =
        builder_context().get_fabric_router_config().router_buffer_clear_size_words * sizeof(uint32_t);
    const auto& host_clears = manifest.at("fabric").at("host_clears");
    ASSERT_EQ(host_clears.size(), to_clear.size());
    for (size_t i = 0; i < to_clear.size(); ++i) {
        EXPECT_EQ(host_clears.at(i).at("address"), to_clear[i]);
        EXPECT_EQ(host_clears.at(i).at("size_bytes"), clear_size);
    }
    for (const auto& entry : routers) {
        SCOPED_TRACE(entry.path);
        std::set<size_t> regions;
        collect_region_addresses(*entry.router, regions);
        for (const size_t address : to_clear) {
            EXPECT_TRUE(regions.contains(address)) << fmt::format("{:#x}", address);
        }
    }
}

// Every region and stream register under `node` that capture reads in its rounds, as path to role.
void collect_multi_sampled(const json& node, const std::string& path, std::map<std::string, std::string>& sampled) {
    if (!node.is_object()) {
        return;
    }
    if (const auto it = node.find("multi_sampled"); it != node.end() && !it->is_null()) {
        sampled.emplace(path, it->get<std::string>());
    }
    for (const auto& [key, child] : node.items()) {
        collect_multi_sampled(child, path.empty() ? key : path + "/" + key, sampled);
    }
}

// Base firmware's link status words exist on Blackhole and not on Wormhole, and where base firmware has a heartbeat
// word, the router's heartbeat is that word. Capture reads the heartbeat and the telemetry area in every round, and
// nothing else.
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
    // Every router core writes an arch area.
    for (const auto& [key, area] : areas.items()) {
        if (!area.is_null()) {
            EXPECT_TRUE(area.at("active").get<bool>()) << key;
        }
    }
    std::map<std::string, std::string> sampled;
    collect_multi_sampled(manifest, "", sampled);
    EXPECT_EQ(
        sampled,
        (std::map<std::string, std::string>{
            {"arch/l1_areas/heartbeat", "loop_heartbeat"},
            {"arch/l1_areas/fabric_telemetry", "fabric_telemetry"},
        }));
}

// Every stream register under `node`, with its path.
void collect_stream_registers(
    const json& node, const std::string& path, std::vector<std::pair<const json*, std::string>>& streams) {
    if (!node.is_object()) {
        return;
    }
    if (node.contains("stream_id")) {
        streams.emplace_back(&node, path);
        return;
    }
    for (const auto& [key, child] : node.items()) {
        collect_stream_registers(child, path.empty() ? key : path + "/" + key, streams);
    }
}

// The read mask the router reads a stream register with role name `role` under.
uint32_t get_read_mask_of_role_name(const std::string& role) {
    if (const auto stream_role = manifest::parse_lower_enum_name<StreamRole>(role)) {
        return manifest::get_stream_register_read_mask(*stream_role);
    }
    const auto scratch_role = manifest::parse_lower_enum_name<manifest::ScratchRole>(role);
    EXPECT_TRUE(scratch_role.has_value()) << role << " names no stream or scratch role";
    return scratch_role.has_value() ? manifest::get_stream_register_read_mask(*scratch_role) : 0;
}

// Every stream register a router names is in the HAL's register space, under the read mask of its role. Every count
// is active; the ERISC sync is active exactly on a router running two ERISCs, and the context switch exactly where one
// of its ERISCs context switches.
void check_router_stream_registers(const std::vector<RouterEntry>& routers) {
    const auto& hal = tt::tt_metal::MetalContext::instance().hal();
    for (const auto& entry : routers) {
        SCOPED_TRACE(entry.path);
        const auto& router = *entry.router;
        std::vector<std::pair<const json*, std::string>> streams;
        collect_stream_registers(router, "", streams);
        EXPECT_FALSE(streams.empty());
        for (const auto& [stream, path] : streams) {
            const auto address = stream->at("address").get<uint32_t>();
            EXPECT_TRUE(hal.valid_reg_addr(address)) << path << " at " << address;
            const auto role = stream->at("role").get<std::string>();
            EXPECT_EQ(stream->at("read_mask").get<uint32_t>(), get_read_mask_of_role_name(role)) << path;
            if (manifest::parse_lower_enum_name<StreamRole>(role).has_value()) {
                EXPECT_TRUE(stream->at("active").get<bool>()) << path;
            }
        }
        const bool two_eriscs = router.at("shape").at("num_active_eriscs").get<uint32_t>() > 1;
        bool context_switches = false;
        for (const auto& [risc_key, fields] : router.at("fields").items()) {
            context_switches = context_switches || fields.at("context_switch_enabled").at("value").get<bool>();
        }
        for (const auto& [risc_key, fields] : router.at("fields").items()) {
            SCOPED_TRACE(risc_key);
            EXPECT_EQ(fields.at("erisc_sync").at("role"), "multi_risc_teardown_sync");
            EXPECT_EQ(fields.at("erisc_sync").at("active").get<bool>(), two_eriscs);
            EXPECT_EQ(fields.at("retrain_sync").at("role"), "eth_retrain_link_sync");
            EXPECT_EQ(fields.at("retrain_sync").at("active").get<bool>(), context_switches);
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

// The chips capture finds on this rank, through capture's own UMD code, on the mock cluster's driver.
capture::ClusterFacts collect_cluster_facts() {
    return capture::collect_umd_cluster_facts(*tt::tt_metal::MetalContext::instance().get_cluster().get_driver());
}

// Capture's reading of this manifest matches the live fabric. Capture reads the file itself, in the manifest's order,
// as it will after a real run, and its plan holds:
//   - this rank's routers, each on ControlPlane's channel, on a chip with the cluster's unique id for it
//   - each router's stream registers, as the fixture's own walk over the router finds them
//   - the HAL's L1 range and core registers, with the low wall clock word before the high one
//   - the multi-sampled arch areas, the heartbeat word and then the telemetry area, and no router region besides
// and matching the plan to the cluster's chips finds every router at its manifest core.
void check_capture_plan(
    // Parse manifest file using capture's own reader
    const std::filesystem::path& manifest_path,
    const std::vector<RouterEntry>& routers) {
    const capture::ManifestFile file = capture::read_manifest_file(manifest_path);
    const capture::CapturePlan plan = capture::collect_capture_plan(file.facts);
    const capture::ManifestIdentity identity = capture::collect_manifest_identity(file);

    // Validate mpi / distributed context
    const auto world = tt::tt_metal::distributed::multihost::DistributedContext::get_world_context();
    EXPECT_EQ(identity.mpi_rank, static_cast<uint32_t>(*world->rank()));
    EXPECT_EQ(identity.world_size, static_cast<uint32_t>(*world->size()));

    // Validate HAL
    const auto& hal = tt::tt_metal::MetalContext::instance().hal();
    using tt::tt_metal::HalL1MemAddrType;
    using tt::tt_metal::HalProgrammableCoreType;
    EXPECT_EQ(plan.arch, hal.get_arch());
    EXPECT_EQ(plan.l1_address, hal.get_dev_addr(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::BASE));
    EXPECT_EQ(plan.l1_size_bytes, hal.get_dev_size(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::BASE));
    const std::vector<capture::CoreRegister> core_registers = {
        {.name = "soft_reset", .address = hal.get_risc_soft_reset_addr()},
        {.name = "wall_clock_low", .address = hal.get_wall_clock_low_addr()},
        {.name = "wall_clock_high", .address = hal.get_wall_clock_high_addr()},
    };
    EXPECT_EQ(plan.core_registers, core_registers);
    const std::vector<capture::MultiSampledRegion> multi_sampled_arch_areas = {
        {.kind = capture::RegionKind::L1,
         .address = hal.get_arch() == tt::ARCH::BLACKHOLE ? FABRIC_KERNEL_HEARTBEAT_ADDR_BLACKHOLE
                                                          : FABRIC_KERNEL_HEARTBEAT_ADDR_WORMHOLE,
         .size_bytes = sizeof(uint32_t)},
        {.kind = capture::RegionKind::L1,
         .address = hal.get_dev_addr(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::FABRIC_TELEMETRY),
         .size_bytes = hal.get_dev_size(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::FABRIC_TELEMETRY)},
    };
    EXPECT_EQ(plan.multi_sampled_arch_areas, multi_sampled_arch_areas);

    // Validate planned routers match routers listed in manifest
    const auto& unique_chip_ids = tt::tt_metal::MetalContext::instance().get_cluster().get_unique_chip_ids();
    size_t num_planned_routers = 0;
    for (const capture::PlannedChip& chip : plan.chips) {
        for (const capture::PlannedRouter& router : chip.routers) {
            SCOPED_TRACE(router.path);
            ++num_planned_routers;
            const auto entry = std::ranges::find(routers, router.path, &RouterEntry::path);
            ASSERT_NE(entry, routers.end());
            EXPECT_EQ(router.eth_chan, entry->eth_chan);
            EXPECT_EQ(chip.asic_id, unique_chip_ids.at(entry->physical_chip_id));

            std::vector<std::pair<const json*, std::string>> streams;
            collect_stream_registers(*entry->router, "", streams);
            std::set<uint64_t> addresses;
            for (const auto& [stream, path] : streams) {
                addresses.insert(stream->at("address").get<uint32_t>());
            }
            EXPECT_EQ(router.stream_register_addresses, std::vector<uint64_t>(addresses.begin(), addresses.end()));
            EXPECT_TRUE(router.multi_sampled_regions.empty());
        }
    }
    EXPECT_EQ(num_planned_routers, routers.size());

    // Collect cluster facts reads the mock cluster's UMD driver to find physical chips and cores, so we should never
    // see any missing chips
    for (const capture::ChipTarget& chip : capture::resolve_capture_targets(plan, collect_cluster_facts())) {
        EXPECT_EQ(chip.location, capture::ChipLocation::FOUND) << chip.path;
        for (const capture::RouterTarget& target : chip.routers) {
            EXPECT_EQ(target.location, capture::CoreLocation::FOUND)
                << target.router.path << " at " << target.router.virtual_core.x << "," << target.router.virtual_core.y;
        }
    }
}

// ============ Capture, end to end ============
//
// Mock clusters drop device reads, so these capture the run's real manifest through a reader that serves made-up bytes:
// every L1 byte and register value is worked out from its chip, core and address.

std::byte get_fake_l1_byte(int32_t chip, capture::CoreXY core, uint64_t address) {
    return static_cast<std::byte>((address + static_cast<uint64_t>(chip) + core.x) & 0xff);
}

uint32_t get_fake_register(int32_t chip, capture::CoreXY core, uint64_t address) {
    return static_cast<uint32_t>(address) ^ (static_cast<uint32_t>(chip) << 24) ^ (core.x << 16) ^ (core.y << 8);
}

// The bytes FakeDeviceReader gives for one recorded read: L1 bytes for a read inside the plan's L1, else a register.
std::vector<std::byte> make_fake_read_bytes(
    const capture::CapturePlan& plan, int32_t chip, capture::CoreXY core, const capture::ReadRecord& read) {
    if (read.address >= plan.l1_address && read.address < plan.l1_address + plan.l1_size_bytes) {
        std::vector<std::byte> bytes(read.size_bytes);
        for (uint32_t i = 0; i < read.size_bytes; ++i) {
            bytes[i] = get_fake_l1_byte(chip, core, read.address + i);
        }
        return bytes;
    }
    const auto value =
        std::bit_cast<std::array<std::byte, sizeof(uint32_t)>>(get_fake_register(chip, core, read.address));
    return {value.begin(), value.end()};
}

// A DeviceReader that serves the made-up bytes above, and fails one core's L1 reads of one size when told to.
class FakeDeviceReader final : public capture::DeviceReader {
public:
    void read_l1(int32_t chip, capture::CoreXY core, uint64_t address, std::span<std::byte> out) override {
        if (failing_l1_ == std::tuple(chip, core, out.size())) {
            throw std::runtime_error("fake L1 read failure");
        }
        for (size_t i = 0; i < out.size(); ++i) {
            out[i] = get_fake_l1_byte(chip, core, address + i);
        }
    }

    uint32_t read_register(int32_t chip, capture::CoreXY core, uint64_t address) override {
        return get_fake_register(chip, core, address);
    }

    // Fails `core`'s L1 reads of `size_bytes` bytes, such as its full read or one multi-sampled region's. Its other
    // reads still work.
    void fail_l1_reads_at(int32_t chip, capture::CoreXY core, size_t size_bytes) {
        failing_l1_ = std::tuple(chip, core, size_bytes);
    }

private:
    std::optional<std::tuple<int32_t, capture::CoreXY, size_t>> failing_l1_;
};

// The record capture_rank wrote reads back and writes again as the same JSON: the record's writer and reader agree on
// every key and value a real capture has.
void check_capture_record_writer_parser_round_trip(const std::filesystem::path& record_path) {
    std::ifstream in(record_path);
    const nlohmann::ordered_json written_record = nlohmann::ordered_json::parse(in);
    const nlohmann::ordered_json rewritten =
        capture::make_capture_record_json(capture::parse_capture_record(written_record));
    EXPECT_TRUE(rewritten == written_record) << describe_json_diff(written_record, rewritten);
}

capture::CaptureRequest make_capture_request(
    const std::filesystem::path& manifest_path, const std::string& capture_id, uint32_t num_rounds = 2) {
    return capture::CaptureRequest{
        .manifest_path = manifest_path,
        .capture_id = capture_id,
        .round_options = {.num_rounds = num_rounds, .interval = std::chrono::milliseconds(1)},
        .capture_git_sha = "0123456789abcdef0123456789abcdef01234567",
    };
}

}  // namespace

// ============ Tests ============

// The spellings the manifest's readers depend on, one case for each rule.
TEST(ManifestNames, Spellings) {
    EXPECT_EQ(mesh_key(MeshId{0}), "M0");
    EXPECT_EQ(chip_key(7), "C7");
    EXPECT_EQ(router_key({eth_chan_directions::EAST, 0}), "E0");
    EXPECT_EQ(router_key({eth_chan_directions::WEST, 1}), "W1");
    EXPECT_EQ(router_key({eth_chan_directions::NORTH, 2}), "N2");
    EXPECT_EQ(router_key({eth_chan_directions::SOUTH, 3}), "S3");
    EXPECT_EQ(router_key({eth_chan_directions::Z, 0}), "Z0");

    // An enumerator in lower case, and a value with no enumerator as its number.
    EXPECT_EQ(lower_enum_name(FabricConfig::FABRIC_2D), "fabric_2d");
    EXPECT_EQ(lower_enum_name(static_cast<manifest::ChannelStatus>(200)), "200");

    // A content type's name in snake case.
    EXPECT_EQ(kind_name<manifest::content::L1>(), "l1");
    EXPECT_EQ(kind_name<manifest::content::StreamReg>(), "stream_reg");
}

// Each key and path parses back to what it was written from, and anything the writer wouldn't write fails.
TEST(ManifestNames, KeysAndPathsParseBack) {
    EXPECT_EQ(manifest::parse_mesh_key("M12"), MeshId{12});
    EXPECT_EQ(manifest::parse_chip_key("C7"), 7u);
    const manifest::RouterKey key = manifest::parse_router_key("S3");
    EXPECT_EQ(key.direction, eth_chan_directions::SOUTH);
    EXPECT_EQ(key.routing_plane, 3);
    for (const std::string_view bad : {"", "M", "C7", "M07", "M-1", "M1x", "m1"}) {
        EXPECT_ANY_THROW(manifest::parse_mesh_key(bad)) << bad;
    }
    EXPECT_ANY_THROW(manifest::parse_router_key("X0"));
    EXPECT_ANY_THROW(manifest::parse_router_key("E256"));

    const FabricNodeId node(MeshId{1}, 7);
    const manifest::RouterRef router{.node = node, .key = {.direction = eth_chan_directions::WEST, .routing_plane = 2}};
    EXPECT_EQ(manifest::router_path(router), "M1/C7/W2");
    const manifest::RouterRef parsed = manifest::parse_router_path("M1/C7/W2");
    EXPECT_EQ(parsed.node, router.node);
    EXPECT_EQ(parsed.key.direction, router.key.direction);
    EXPECT_EQ(parsed.key.routing_plane, router.key.routing_plane);
    for (const std::string_view bad : {"M1/C7", "M1/C7/W2/", "/M1/C7/W2", "M1/C7/W2/E0"}) {
        EXPECT_ANY_THROW(manifest::parse_router_path(bad)) << bad;
    }

    // A sibling is on the referring router's chip and routing plane.
    const manifest::RouterIdentity identity{.key = {.direction = eth_chan_directions::EAST, .routing_plane = 2}};
    const manifest::DownstreamChannel landing{
        .target = {.direction = eth_chan_directions::NORTH}, .landing_channel = 3};
    EXPECT_EQ(manifest::sibling_router_path(node, identity, landing.target), "M1/C7/N2");
    EXPECT_EQ(manifest::parse_sibling_router_path("M1/C7/N2", node, identity).direction, eth_chan_directions::NORTH);
    EXPECT_ANY_THROW(manifest::parse_sibling_router_path("M1/C7/N1", node, identity));
    EXPECT_ANY_THROW(manifest::parse_sibling_router_path("M1/C6/N2", node, identity));

    const std::string channel = manifest::downstream_channel_path(node, identity, 1, landing);
    EXPECT_EQ(channel, "M1/C7/N2/channels/senders/vc1/ch3");
    const auto parsed_channel = manifest::parse_downstream_channel_path(channel, node, identity, 1);
    EXPECT_EQ(parsed_channel.target.direction, eth_chan_directions::NORTH);
    EXPECT_EQ(parsed_channel.landing_channel, 3u);
    EXPECT_ANY_THROW(manifest::parse_downstream_channel_path(channel, node, identity, 0));
    EXPECT_ANY_THROW(manifest::parse_downstream_channel_path("M1/C7/N2/channels/receivers/vc1/ch3", node, identity, 1));

    // An enum's spelling parses back, and a number only where no enumerator names the value.
    EXPECT_EQ(manifest::parse_lower_enum_name<FabricConfig>("fabric_2d"), FabricConfig::FABRIC_2D);
    EXPECT_EQ(
        manifest::parse_lower_enum_name<manifest::ChannelStatus>("200"), static_cast<manifest::ChannelStatus>(200));
    EXPECT_EQ(manifest::parse_lower_enum_name<FabricConfig>("FABRIC_2D"), std::nullopt);
    EXPECT_EQ(manifest::parse_lower_enum_name<manifest::ChannelStatus>("0"), std::nullopt);
    EXPECT_EQ(manifest::parse_lower_enum_name<manifest::ChannelStatus>(""), std::nullopt);
}

// A chip is published once, with every router its builder initialized, and only within the chips the context holds.
// An unpublished chip can't be read.
TEST(ManifestPublishedChips, RejectsWhatItCannotHold) {
    manifest::PublishedChips chips(2);
    EXPECT_ANY_THROW(chips.publish(0, {.chip = {.routers = {manifest::Router{}}}}, 2));
    EXPECT_ANY_THROW(chips.publish(2, {}, 0));
    EXPECT_FALSE(chips.has(0));
    EXPECT_ANY_THROW(chips.get(0));
    chips.publish(0, {}, 0);
    EXPECT_TRUE(chips.has(0));
    EXPECT_ANY_THROW(chips.publish(0, {}, 0));
    EXPECT_FALSE(chips.has(2));
    EXPECT_FALSE(chips.has(-1));
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
    const std::vector<KernelInputs> kernel = {{
        .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
        .noc = tt::tt_metal::NOC::NOC_0,
        .opt_level = tt::tt_metal::KernelBuildOptLevel::O3,
        .compile_args = {},
        .named_compile_args =
            {
                {"EDM_STATUS_PTR_ADDR", 0},
                {"SENDER_CH_7_IS_INJECTION", 0},
                {"NUM_ACTIVE_ERISCS", 1},
                {"MAX_NUM_VCS", 3},
                {"LINK_SYNC_ROLE", 0},
            },
        .defines = {{"FABRIC_2D", ""}, {"FABRIC_2D_VC1_SERVICED", ""}},
        .runtime_args = {},
    }};
    EXPECT_NO_THROW(check_kernel_inputs_accounted(kernel));

    auto new_arg = kernel;
    new_arg.front().named_compile_args["SOME_NEW_ARG"] = 0;
    EXPECT_THROW(check_kernel_inputs_accounted(new_arg), std::exception);

    auto new_define = kernel;
    new_define.front().defines["SOME_NEW_DEFINE"] = "";
    EXPECT_THROW(check_kernel_inputs_accounted(new_define), std::exception);
}

// Every check, one test each, for `Fixture`.
#define TEST_MANIFEST_CHECKS(Fixture)                                                                      \
    TEST_F(Fixture, TopLevel) { check_top_level(manifest_, manifest_path_, fabric_config, suite_start_); } \
    TEST_F(Fixture, FieldSets) { check_field_sets(written_manifest_); }                                    \
    TEST_F(Fixture, WriterParserRoundTrip) { check_manifest_writer_parser_round_trip(written_manifest_); } \
    TEST_F(Fixture, ChipZPortRole) { check_chip_z_port_role(manifest_); }                                  \
    TEST_F(Fixture, ChipHostRanks) { check_chip_host_ranks(manifest_); }                                   \
    TEST_F(Fixture, RouterLink) { check_router_link(manifest_, routers_); }                                \
    TEST_F(Fixture, PeersAreSymmetric) { check_peers_are_symmetric(manifest_, routers_); }                 \
    TEST_F(Fixture, CreditTransport) { check_credit_transport(manifest_, routers_); }                      \
    TEST_F(Fixture, IntermeshVc) { check_intermesh_vc(manifest_, routers_); }                              \
    TEST_F(Fixture, RouterSenders) { check_router_senders(manifest_, routers_); }                          \
    TEST_F(Fixture, RouterDownstreamEdges) { check_router_edges(manifest_, routers_); }                    \
    TEST_F(Fixture, RouterFields) { check_router_fields(manifest_, routers_); }                            \
    TEST_F(Fixture, LocalSync) { check_local_sync(manifest_); }                                            \
    TEST_F(Fixture, RouterRegionsDisjoint) { check_router_regions_disjoint(manifest_, routers_); }         \
    TEST_F(Fixture, RouterRegionsInL1) { check_router_regions_in_l1(manifest_, routers_); }                \
    TEST_F(Fixture, Arch) { check_arch(manifest_); }                                                       \
    TEST_F(Fixture, RouterStreamRegisters) { check_router_stream_registers(routers_); }                    \
    TEST_F(Fixture, ArchCoreRegisters) { check_arch_core_registers(manifest_, routers_); }                 \
    TEST_F(Fixture, RouterDiagnostics) { check_router_diagnostics(routers_); }                             \
    TEST_F(Fixture, HostClears) { check_host_clears(manifest_, routers_); }                                \
    TEST_F(Fixture, RouterRegionsActive) { check_router_regions_active(manifest_, routers_); }             \
    TEST_F(Fixture, CapturePlan) { check_capture_plan(manifest_path_, routers_); }

TEST_MANIFEST_CHECKS(Manifest1DFixture)
TEST_MANIFEST_CHECKS(Manifest2DFixture)
TEST_MANIFEST_CHECKS(ManifestTorusXFixture)
TEST_MANIFEST_CHECKS(ManifestTorusYFixture)
TEST_MANIFEST_CHECKS(ManifestTorusXYFixture)
TEST_MANIFEST_CHECKS(ManifestMuxFixture)

namespace {

// The plan capture_rank reads this fixture's manifest into.
capture::CapturePlan read_capture_plan(const std::filesystem::path& manifest_path) {
    return capture::collect_capture_plan(capture::read_manifest_file(manifest_path).facts);
}

// The addresses `reads` were made at, in order.
std::vector<uint64_t> collect_read_addresses(const std::vector<capture::ReadRecord>& reads) {
    std::vector<uint64_t> addresses;
    addresses.reserve(reads.size());
    for (const capture::ReadRecord& read : reads) {
        addresses.push_back(read.address);
    }
    return addresses;
}

// What a round of `router` reads, in order: the core registers, then the multi-sampled arch areas and the router's own.
std::vector<uint64_t> make_round_addresses(const capture::CapturePlan& plan, const capture::PlannedRouter& router) {
    std::vector<uint64_t> addresses;
    for (const capture::CoreRegister& core_register : plan.core_registers) {
        addresses.push_back(core_register.address);
    }
    for (const auto* regions : {&plan.multi_sampled_arch_areas, &router.multi_sampled_regions}) {
        for (const capture::MultiSampledRegion& region : *regions) {
            addresses.push_back(region.address);
        }
    }
    return addresses;
}

// What a full read of `router` reads, in order: the core registers, the whole L1, then each stream register once.
std::vector<uint64_t> make_full_read_addresses(const capture::CapturePlan& plan, const capture::PlannedRouter& router) {
    std::vector<uint64_t> addresses;
    for (const capture::CoreRegister& core_register : plan.core_registers) {
        addresses.push_back(core_register.address);
    }
    addresses.push_back(plan.l1_address);
    addresses.insert(addresses.end(), router.stream_register_addresses.begin(), router.stream_register_addresses.end());
    return addresses;
}

// The bytes of each of `reads` come back from the loaded .bin exactly as the fake reader gave them.
void expect_reads_give_back_their_bytes(
    const capture::LoadedCapture& loaded,
    const capture::CapturePlan& plan,
    int32_t chip,
    capture::CoreXY core,
    const std::vector<capture::ReadRecord>& reads) {
    for (const capture::ReadRecord& read : reads) {
        const std::span<const std::byte> bytes = loaded.get_read_bytes(read);
        EXPECT_EQ(std::vector<std::byte>(bytes.begin(), bytes.end()), make_fake_read_bytes(plan, chip, core, read))
            << fmt::format("{:#x}", read.address);
    }
}

// The core of the first router on `chip` in `routers`, as the manifest names it.
capture::CoreXY find_first_router_core(const std::vector<RouterEntry>& routers, int32_t chip) {
    const auto entry = std::ranges::find(routers, static_cast<ChipId>(chip), &RouterEntry::physical_chip_id);
    if (entry == routers.end()) {
        throw std::runtime_error(fmt::format("No router on chip {}", chip));
    }
    const json& core = entry->router->at("identity").at("virtual_core");
    return capture::CoreXY{core.at(0).get<uint32_t>(), core.at(1).get<uint32_t>()};
}

}  // namespace

// Captures the run's real manifest end to end, then loads it back. Every router has its rounds, each reading the core
// registers and the multi-sampled regions, and then its full read: the core registers, the whole L1, then each stream
// register once. Every read's bytes, a round's or the full read's, come back exactly as the reader returned them, and
// the record names the run and the capture. Loading checks the .bin is the one the record names. Only on a 1D T3K: on a
// galaxy, the .bin is hundreds of MB.
TEST_F(Manifest1DFixture, CaptureRankWritesEveryRoutersReadsWhereTheRecordSays) {
    FakeDeviceReader reader;
    capture::SteadyCaptureClock clock;
    const capture::CaptureResult result =
        capture::capture_rank(make_capture_request(manifest_path_, "complete"), reader, collect_cluster_facts(), clock);

    EXPECT_EQ(result.outcome, capture::CaptureOutcome::COMPLETE);
    EXPECT_EQ(result.paths.record_path.parent_path(), manifest_path_.parent_path() / "captures" / "complete");
    const capture::LoadedCapture loaded = capture::load_capture(result.paths.record_path);
    const capture::CaptureRecord& record = loaded.get_record();
    EXPECT_EQ(record.manifest.run_id, manifest_.at("run").at("run_id").get<std::string>());
    EXPECT_EQ(record.capture_id, "complete");
    EXPECT_EQ(record.round_options.num_rounds, 2u);
    check_capture_record_writer_parser_round_trip(result.paths.record_path);

    const capture::CapturePlan plan = read_capture_plan(manifest_path_);
    size_t num_routers = 0;
    for (const capture::ChipRecord& chip : record.chips) {
        ASSERT_EQ(chip.location, capture::ChipLocation::FOUND) << chip.path;
        for (const capture::RouterRecord& router : chip.routers) {
            SCOPED_TRACE(router.target.router.path);
            ++num_routers;
            const capture::CoreXY core = *router.target.reader_core;
            ASSERT_EQ(router.rounds.size(), 2u);
            for (uint32_t round = 0; round < router.rounds.size(); ++round) {
                EXPECT_EQ(router.rounds[round].round, round);
                const std::vector<capture::ReadRecord>& reads = router.rounds[round].reads;
                EXPECT_EQ(collect_read_addresses(reads), make_round_addresses(plan, router.target.router));
                expect_reads_give_back_their_bytes(loaded, plan, *chip.chip, core, reads);
            }
            EXPECT_EQ(collect_read_addresses(router.reads), make_full_read_addresses(plan, router.target.router));
            expect_reads_give_back_their_bytes(loaded, plan, *chip.chip, core, router.reads);
        }
    }
    EXPECT_EQ(num_routers, routers_.size());
}

// With zero rounds, capture reads each core in full only.
TEST_F(Manifest1DFixture, CaptureRankWithNoRoundsReadsEachCoreInFull) {
    FakeDeviceReader reader;
    capture::SteadyCaptureClock clock;
    const capture::CaptureResult result = capture::capture_rank(
        make_capture_request(manifest_path_, "no_rounds", 0), reader, collect_cluster_facts(), clock);

    EXPECT_EQ(result.outcome, capture::CaptureOutcome::COMPLETE);
    check_capture_record_writer_parser_round_trip(result.paths.record_path);
    const capture::LoadedCapture loaded = capture::load_capture(result.paths.record_path);
    for (const capture::ChipRecord& chip : loaded.get_record().chips) {
        for (const capture::RouterRecord& router : chip.routers) {
            EXPECT_TRUE(router.rounds.empty()) << router.target.router.path;
            EXPECT_FALSE(router.reads.empty()) << router.target.router.path;
        }
    }
}

// A chip the reader didn't find and a core whose full L1 read failed are recorded, not fatal: the files are still
// written, and the outcome says something is missing. The failing core's rounds, which read only small regions, still
// work.
TEST_F(Manifest1DFixture, CaptureRankRecordsWhatItCouldNotRead) {
    capture::ClusterFacts facts = collect_cluster_facts();
    ASSERT_GE(facts.chips.size(), 2u);
    const uint64_t missing_asic_id = facts.chips.back().asic_id;
    facts.chips.pop_back();
    // The first router on the first chip the reader still finds fails its full L1 read.
    const capture::CapturePlan plan = read_capture_plan(manifest_path_);
    const int32_t failing_chip = facts.chips.front().chip;
    const capture::CoreXY failing_core = find_first_router_core(routers_, failing_chip);
    FakeDeviceReader reader;
    reader.fail_l1_reads_at(failing_chip, failing_core, plan.l1_size_bytes);
    capture::SteadyCaptureClock clock;

    const capture::CaptureResult result =
        capture::capture_rank(make_capture_request(manifest_path_, "problems"), reader, facts, clock);

    EXPECT_EQ(result.outcome, capture::CaptureOutcome::PROBLEMS_RECORDED);
    check_capture_record_writer_parser_round_trip(result.paths.record_path);
    const capture::LoadedCapture loaded = capture::load_capture(result.paths.record_path);
    const capture::CaptureRecord& record = loaded.get_record();
    size_t num_missing_chips = 0;
    size_t num_failed_routers = 0;
    for (const capture::ChipRecord& chip : record.chips) {
        if (chip.asic_id == missing_asic_id) {
            ++num_missing_chips;
            EXPECT_EQ(chip.location, capture::ChipLocation::MISSING);
            for (const capture::RouterRecord& router : chip.routers) {
                EXPECT_TRUE(router.reads.empty() && router.rounds.empty()) << router.target.router.path;
            }
            continue;
        }
        for (const capture::RouterRecord& router : chip.routers) {
            SCOPED_TRACE(router.target.router.path);
            EXPECT_EQ(router.rounds.size(), 2u);
            const bool is_failing = chip.chip == failing_chip && router.target.reader_core == failing_core;
            if (!is_failing) {
                EXPECT_TRUE(std::ranges::none_of(
                    router.reads, [](const capture::ReadRecord& read) { return read.error.has_value(); }));
                continue;
            }
            // The failed L1 read is the core's last: it has no offset, and no stream register is read after it.
            ++num_failed_routers;
            ASSERT_FALSE(router.reads.empty());
            const capture::ReadRecord& last = router.reads.back();
            EXPECT_EQ(last.address, plan.l1_address);
            EXPECT_EQ(last.size_bytes, plan.l1_size_bytes);
            EXPECT_TRUE(last.error.has_value());
            EXPECT_FALSE(last.bin_offset.has_value());
            EXPECT_ANY_THROW(loaded.get_read_bytes(last));
        }
    }
    EXPECT_EQ(num_missing_chips, 1u);
    EXPECT_EQ(num_failed_routers, 1u);
}

// A round read that fails ends the router's round and its rounds: it isn't sampled again. Its full read still happens,
// and the outcome says something is missing.
TEST_F(Manifest1DFixture, CaptureRankStopsARoutersRoundsAtAFailedRead) {
    const capture::CapturePlan plan = read_capture_plan(manifest_path_);
    ASSERT_EQ(plan.multi_sampled_arch_areas.size(), 2u);
    const capture::MultiSampledRegion& telemetry = plan.multi_sampled_arch_areas.back();
    const capture::ClusterFacts facts = collect_cluster_facts();
    const int32_t failing_chip = facts.chips.front().chip;
    const capture::CoreXY failing_core = find_first_router_core(routers_, failing_chip);
    FakeDeviceReader reader;
    reader.fail_l1_reads_at(failing_chip, failing_core, telemetry.size_bytes);
    capture::SteadyCaptureClock clock;

    const capture::CaptureResult result =
        capture::capture_rank(make_capture_request(manifest_path_, "failed_round"), reader, facts, clock);

    EXPECT_EQ(result.outcome, capture::CaptureOutcome::PROBLEMS_RECORDED);
    check_capture_record_writer_parser_round_trip(result.paths.record_path);
    const capture::LoadedCapture loaded = capture::load_capture(result.paths.record_path);
    size_t num_failed_routers = 0;
    for (const capture::ChipRecord& chip : loaded.get_record().chips) {
        for (const capture::RouterRecord& router : chip.routers) {
            SCOPED_TRACE(router.target.router.path);
            EXPECT_TRUE(std::ranges::none_of(
                router.reads, [](const capture::ReadRecord& read) { return read.error.has_value(); }));
            if (chip.chip != failing_chip || router.target.reader_core != failing_core) {
                EXPECT_EQ(router.rounds.size(), 2u);
                continue;
            }
            ++num_failed_routers;
            ASSERT_EQ(router.rounds.size(), 1u);
            const capture::ReadRecord& last = router.rounds.front().reads.back();
            EXPECT_EQ(last.address, telemetry.address);
            EXPECT_TRUE(last.error.has_value());
            EXPECT_FALSE(last.bin_offset.has_value());
        }
    }
    EXPECT_EQ(num_failed_routers, 1u);
}

}  // namespace tt::tt_fabric::fabric_router_tests
