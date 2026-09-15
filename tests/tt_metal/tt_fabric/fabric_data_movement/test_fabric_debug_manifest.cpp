// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <enchantum/enchantum.hpp>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <map>
#include <numeric>
#include <optional>
#include <regex>
#include <set>
#include <tuple>

#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt-metalium/hal.hpp>

#include "fabric_fixture.hpp"
#include "hostdevcommon/fabric_common.h"
#include "impl/context/metal_context.hpp"
#include "tt_metal/fabric/fabric_builder_context.hpp"
#include "tt_metal/fabric/fabric_context.hpp"
#include "tt_metal/fabric/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/fabric_host_utils.hpp"

namespace tt::tt_fabric::fabric_router_tests {
namespace {

// Fabric-native router identity. Links, snapshots, and the viewer join on this triple.
// ASIC id is a per-chip device identity used later by capture to look up ttexalens devices;
// it is not unique per ethernet channel and is null on non-local chips.
using RouterKey = std::tuple<uint32_t, uint32_t, uint32_t>;  // mesh, chip, eth channel

RouterKey router_key(const nlohmann::json& endpoint) {
    return {
        endpoint.at("mesh_id").get<uint32_t>(),
        endpoint.at("chip_id").get<uint32_t>(),
        endpoint.at("eth_chan").get<uint32_t>()};
}

std::optional<std::string> stream_arg_for_region(const std::string& id) {
    std::smatch match;
    if (std::regex_match(id, match, std::regex(R"(sender\.([0-9]+)\.free_slots)"))) {
        return fmt::format("SENDER_CHANNEL_{}_FREE_SLOTS_STREAM_ID", match[1].str());
    }
    if (std::regex_match(id, match, std::regex(R"(sender\.([0-9]+)\.credits\.acked)"))) {
        return fmt::format("TO_SENDER_{}_PKTS_ACKED_ID", match[1].str());
    }
    if (std::regex_match(id, match, std::regex(R"(sender\.([0-9]+)\.credits\.completed)"))) {
        return fmt::format("TO_SENDER_{}_PKTS_COMPLETED_ID", match[1].str());
    }
    if (std::regex_match(id, match, std::regex(R"(receiver\.([0-9]+)\.pkts_sent)"))) {
        return fmt::format("TO_RECEIVER_{}_PKTS_SENT_ID", match[1].str());
    }
    if (std::regex_match(id, match, std::regex(R"(credits\.downstream\.vc([0-9]+)\.edge([0-9]+)\.free_slots)"))) {
        return fmt::format("VC{}_FREE_SLOTS_FROM_DOWNSTREAM_EDGE_{}_STREAM_ID", match[1].str(), match[2].str());
    }
    if (id == "credits.vc2_receiver.free_slots") {
        return "VC2_RECEIVER_FREE_SLOTS_STREAM_ID";
    }
    if (id == "credits.tensix_relay.free_slots") {
        return "TENSIX_RELAY_LOCAL_FREE_SLOTS_STREAM_ID";
    }
    return std::nullopt;
}

void check_layout_structure(const nlohmann::json& layout, uint32_t unreserved_base, uint32_t unreserved_size) {
    const auto& regions = layout.at("regions");
    std::set<std::string> ids;
    struct Interval {
        uint32_t begin;
        uint32_t end;
        const nlohmann::json* region;
    };
    std::vector<Interval> intervals;

    for (const auto& region : regions) {
        const std::string id = region.at("id");
        EXPECT_TRUE(ids.insert(id).second) << "duplicate region " << id;
        if (!region.at("parent").get<std::string>().empty()) {
            // Parent records are emitted before their children.
            EXPECT_TRUE(ids.contains(region.at("parent").get<std::string>())) << id;
        }
        if (region.at("backing") == "stream_reg" && region.at("allocated").get<bool>()) {
            EXPECT_LT(region.at("stream_id").get<uint32_t>(), 32U) << id;
        }
        if (region.value("schema", "") == "packet_ring") {
            EXPECT_EQ(
                region.at("size").get<uint32_t>(),
                region.at("count").get<uint32_t>() * region.at("stride").get<uint32_t>())
                << id;
        }
        if (region.at("backing") == "unreserved_l1" && region.at("allocated").get<bool>() &&
            region.at("size").get<uint32_t>() != 0) {
            const uint32_t begin = region.at("address");
            const uint32_t end = begin + region.at("size").get<uint32_t>();
            EXPECT_GE(begin, unreserved_base) << id;
            EXPECT_LE(end, unreserved_base + unreserved_size) << id;
            intervals.push_back({begin, end, &region});
        }
    }

    std::ranges::sort(intervals, {}, &Interval::begin);
    uint32_t covered_end = unreserved_base;
    std::vector<const nlohmann::json*> active;
    for (const auto& interval : intervals) {
        active.erase(
            std::remove_if(
                active.begin(),
                active.end(),
                [&](const nlohmann::json* previous) {
                    return previous->at("address").get<uint32_t>() + previous->at("size").get<uint32_t>() <=
                           interval.begin;
                }),
            active.end());
        if (interval.begin < covered_end) {
            bool declared_overlap = false;
            for (const auto* previous : active) {
                const auto declares = [](const nlohmann::json& lhs, const nlohmann::json& rhs) {
                    if (!lhs.contains("overlaps")) {
                        return false;
                    }
                    const auto values = lhs.at("overlaps").get<std::vector<std::string>>();
                    return std::ranges::find(values, rhs.at("id").get<std::string>()) != values.end();
                };
                declared_overlap |= declares(*interval.region, *previous) || declares(*previous, *interval.region);
            }
            EXPECT_TRUE(declared_overlap) << interval.region->at("id");
        } else {
            EXPECT_EQ(interval.begin, covered_end) << "gap before " << interval.region->at("id");
        }
        covered_end = std::max(covered_end, interval.end);
        active.push_back(interval.region);
    }
    EXPECT_EQ(covered_end, unreserved_base + unreserved_size);
}

// Compare the generated fabric manifest with the live fabric state.
void check_manifest_matches_live_fabric(FabricConfig expected_config) {
    auto& metal_context = tt::tt_metal::MetalContext::instance();
    const auto manifest_path = fabric_debug_manifest_path(metal_context.rtoptions());
    ASSERT_TRUE(std::filesystem::exists(manifest_path)) << manifest_path;
    for (const auto& entry : std::filesystem::directory_iterator(manifest_path.parent_path())) {
        EXPECT_TRUE(entry.path().string().find(".tmp.") == std::string::npos) << entry.path();
    }

    std::ifstream manifest_stream(manifest_path);
    ASSERT_TRUE(manifest_stream.is_open());
    const nlohmann::json manifest = nlohmann::json::parse(manifest_stream);

    ASSERT_EQ(manifest.at("manifest_version"), FABRIC_DEBUG_MANIFEST_VERSION);
    ASSERT_EQ(manifest.at("kind"), "fabric_debug_manifest");
    ASSERT_TRUE(manifest.contains("run"));
    ASSERT_TRUE(manifest.contains("hal"));
    ASSERT_TRUE(manifest.contains("heartbeat"));
    ASSERT_TRUE(manifest.contains("fabric_context"));
    ASSERT_TRUE(manifest.contains("router_template"));
    ASSERT_TRUE(manifest.contains("stream_assignment"));
    ASSERT_TRUE(manifest.contains("enums"));
    ASSERT_TRUE(manifest.contains("layouts"));
    ASSERT_TRUE(manifest.contains("meshes"));
    ASSERT_TRUE(manifest.contains("links"));
    ASSERT_EQ(manifest.at("run").at("fabric_config"), enchantum::to_string(expected_config));
    ASSERT_EQ(manifest.at("run").at("arch"), enchantum::to_string(metal_context.get_cluster().arch()));

    const auto& control_plane = metal_context.get_control_plane();
    ASSERT_EQ(control_plane.get_fabric_config(), expected_config);
    const auto& fabric_context = control_plane.get_fabric_context();
    ASSERT_TRUE(fabric_context.has_builder_context());
    const auto& builder_context = fabric_context.get_builder_context();
    const auto& hal = metal_context.hal();
    using tt::tt_metal::HalL1MemAddrType;
    using tt::tt_metal::HalProgrammableCoreType;

    EXPECT_FALSE(manifest.at("run").at("written_at").get<std::string>().empty());
    EXPECT_TRUE(manifest.at("layouts").is_object());
    EXPECT_FALSE(manifest.at("layouts").empty());

    const auto expected_heartbeat_address = metal_context.get_cluster().arch() == tt::ARCH::BLACKHOLE
                                                ? FABRIC_KERNEL_HEARTBEAT_ADDR_BLACKHOLE
                                                : FABRIC_KERNEL_HEARTBEAT_ADDR_WORMHOLE;
    EXPECT_EQ(manifest.at("heartbeat").at("address"), expected_heartbeat_address);
    EXPECT_EQ(manifest.at("heartbeat").at("magic"), FABRIC_KERNEL_HEARTBEAT_MAGIC);

    auto expect_hal_region = [&](const char* name, HalL1MemAddrType addr_type) {
        EXPECT_EQ(
            manifest.at("hal").at(name).at("base"), hal.get_dev_addr(HalProgrammableCoreType::ACTIVE_ETH, addr_type));
        EXPECT_EQ(
            manifest.at("hal").at(name).at("size"), hal.get_dev_size(HalProgrammableCoreType::ACTIVE_ETH, addr_type));
    };
    expect_hal_region("unreserved", HalL1MemAddrType::UNRESERVED);
    expect_hal_region("go_msg", HalL1MemAddrType::GO_MSG);
    expect_hal_region("launch", HalL1MemAddrType::LAUNCH);
    expect_hal_region("fabric_telemetry", HalL1MemAddrType::FABRIC_TELEMETRY);
    expect_hal_region("routing_table", HalL1MemAddrType::ROUTING_TABLE);
    expect_hal_region("router_state", HalL1MemAddrType::ROUTER_STATE);
    expect_hal_region("router_command", HalL1MemAddrType::ROUTER_COMMAND);
    expect_hal_region("eth_fw_mailbox", HalL1MemAddrType::ETH_FW_MAILBOX);

    EXPECT_EQ(manifest.at("fabric_context").at("is_2d_routing"), fabric_context.is_2D_routing_enabled());
    EXPECT_EQ(
        manifest.at("fabric_context").at("channel_buffer_size_bytes"),
        fabric_context.get_fabric_channel_buffer_size_bytes());
    EXPECT_EQ(
        manifest.at("fabric_context").at("packet_header_size_bytes"),
        fabric_context.get_fabric_packet_header_size_bytes());
    EXPECT_EQ(
        manifest.at("fabric_context").at("max_payload_size_bytes"), fabric_context.get_fabric_max_payload_size_bytes());

    const auto [status_address, _] = builder_context.get_fabric_router_sync_address_and_status();
    const auto [termination_address, termination_signal] =
        builder_context.get_fabric_router_termination_address_and_signal();
    EXPECT_EQ(manifest.at("router_template").at("edm_status_address"), status_address);
    EXPECT_EQ(manifest.at("router_template").at("termination_signal_address"), termination_address);
    EXPECT_EQ(
        manifest.at("router_template").at("addresses_to_clear"),
        nlohmann::json(builder_context.get_fabric_router_addresses_to_clear()));
    const auto expected_handshake_address = tt::round_up(
        hal.get_dev_addr(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::UNRESERVED),
        FabricEriscDatamoverConfig::eth_channel_sync_size);
    EXPECT_EQ(manifest.at("router_template").at("handshake_address"), expected_handshake_address);
    EXPECT_EQ(
        manifest.at("router_template").at("unused_config_handshake_address"),
        builder_context.get_fabric_router_config().handshake_addr);

    EXPECT_EQ(manifest.at("enums").at("EDMStatus").at("READY_FOR_TRAFFIC"), EDMStatus::READY_FOR_TRAFFIC);
    EXPECT_EQ(manifest.at("enums").at("TerminationSignal").at("IMMEDIATELY_TERMINATE"), termination_signal);
    EXPECT_EQ(manifest.at("enums").at("RunMsg").at("RUN_MSG_GO"), 0x80);

    for (const auto& mesh_id : control_plane.get_local_mesh_id_bindings()) {
        const auto named_args = builder_context.get_stream_assignment(mesh_id).named_args();
        const auto& assignment = manifest.at("stream_assignment").at(std::to_string(*mesh_id));
        ASSERT_EQ(assignment.size(), named_args.size());
        for (const auto& [name, value] : named_args) {
            EXPECT_EQ(assignment.at(name), value) << name;
        }
    }

    std::set<RouterKey> declared_routers;
    size_t local_router_count = 0;
    std::map<std::string, size_t> observed_layout_counts;
    std::set<std::string> checked_layouts;
    const uint32_t unreserved_base = manifest.at("hal").at("unreserved").at("base");
    const uint32_t unreserved_size = manifest.at("hal").at("unreserved").at("size");
    for (const auto& mesh : manifest.at("meshes")) {
        const MeshId mesh_id{mesh.at("mesh_id").get<uint32_t>()};
        const auto live_shape = control_plane.get_mesh_graph().get_mesh_shape(mesh_id);
        ASSERT_EQ(mesh.at("shape").size(), live_shape.dims());
        for (size_t dim = 0; dim < live_shape.dims(); ++dim) {
            EXPECT_EQ(mesh.at("shape").at(dim), live_shape[dim]);
        }

        for (const auto& chip : mesh.at("chips")) {
            const uint32_t chip_id = chip.at("fabric_chip_id").get<uint32_t>();
            const FabricNodeId node{mesh_id, chip_id};
            const auto physical_chip_id = control_plane.try_get_physical_chip_id_from_fabric_node_id(node);
            ASSERT_EQ(chip.at("is_local").get<bool>(), physical_chip_id.has_value());

            if (!physical_chip_id.has_value()) {
                EXPECT_TRUE(chip.at("physical_chip_id").is_null());
                EXPECT_TRUE(chip.at("asic_id").is_null());
                EXPECT_TRUE(chip.at("master_router_chan").is_null());
                EXPECT_TRUE(chip.at("routers").empty());
                continue;
            }

            EXPECT_EQ(chip.at("physical_chip_id"), *physical_chip_id);
            EXPECT_EQ(
                chip.at("asic_id"), fmt::format("0x{:016x}", *control_plane.get_asic_id_from_fabric_node_id(node)));
            EXPECT_EQ(chip.at("master_router_chan"), builder_context.get_fabric_master_router_chan(*physical_chip_id));
            EXPECT_EQ(chip.at("routers").size(), builder_context.get_num_fabric_initialized_routers(*physical_chip_id));
            ASSERT_TRUE(builder_context.has_router_debug_instances(*physical_chip_id));
            const auto& published_instances = builder_context.get_router_debug_instances(*physical_chip_id);
            const auto live_routers = control_plane.get_active_fabric_eth_channels(node);
            std::set<uint32_t> manifest_channels;
            for (const auto& router : chip.at("routers")) {
                const uint32_t eth_chan = router.at("eth_chan").get<uint32_t>();
                manifest_channels.insert(eth_chan);
                declared_routers.emplace(*mesh_id, chip_id, eth_chan);
                ++local_router_count;

                const std::string layout_id = router.at("layout_id");
                ASSERT_TRUE(manifest.at("layouts").contains(layout_id));
                ++observed_layout_counts[layout_id];
                const auto& layout = manifest.at("layouts").at(layout_id);
                if (checked_layouts.insert(layout_id).second) {
                    check_layout_structure(layout, unreserved_base, unreserved_size);
                }

                const auto published =
                    std::ranges::find(published_instances, eth_chan, &FabricRouterDebugInstance::eth_chan);
                ASSERT_NE(published, published_instances.end());
                const auto& instance = router.at("instance");
                EXPECT_EQ(instance.at("num_active_eriscs"), published->num_active_eriscs);
                EXPECT_EQ(instance.at("sender_channels_per_vc"), nlohmann::json(published->sender_channels_per_vc));
                EXPECT_EQ(instance.at("receiver_channels_per_vc"), nlohmann::json(published->receiver_channels_per_vc));
                EXPECT_EQ(instance.at("downstream_edm_mask_vc0"), published->downstream_edm_mask_vc0);
                EXPECT_EQ(instance.at("downstream_edm_mask_vc1"), published->downstream_edm_mask_vc1);

                const auto& assignment = manifest.at("stream_assignment").at(std::to_string(*mesh_id));
                size_t enabled_sender_rings = 0;
                for (const auto& region : layout.at("regions")) {
                    if (region.value("schema", "") == "packet_ring" &&
                        region.at("id").get<std::string>().starts_with("sender.") && region.at("enabled").get<bool>()) {
                        ++enabled_sender_rings;
                        EXPECT_EQ(region.at("stride"), manifest.at("fabric_context").at("channel_buffer_size_bytes"));
                    }
                    if (region.at("backing") != "stream_reg" || !region.at("allocated").get<bool>()) {
                        continue;
                    }
                    const auto arg_name = stream_arg_for_region(region.at("id"));
                    ASSERT_TRUE(arg_name.has_value()) << region.at("id");
                    ASSERT_TRUE(assignment.contains(*arg_name)) << *arg_name;
                    EXPECT_EQ(region.at("stream_id"), assignment.at(*arg_name)) << region.at("id");
                }
                EXPECT_EQ(
                    enabled_sender_rings,
                    std::accumulate(
                        published->sender_channels_per_vc.begin(), published->sender_channels_per_vc.end(), size_t{0}));
            }

            std::set<uint32_t> live_channels;
            for (const auto& [eth_chan, _] : live_routers) {
                live_channels.insert(eth_chan);
            }
            EXPECT_EQ(manifest_channels, live_channels) << "router mismatch for " << node;
        }
    }

    EXPECT_LE(manifest.at("layouts").size(), local_router_count);
    for (const auto& [layout_id, count] : observed_layout_counts) {
        EXPECT_EQ(manifest.at("layouts").at(layout_id).at("router_count"), count);
    }

    ASSERT_EQ(manifest.at("links").size(), local_router_count);
    std::map<RouterKey, const nlohmann::json*> links_by_source;
    for (const auto& link : manifest.at("links")) {
        const RouterKey source = router_key(link.at("src"));
        EXPECT_TRUE(declared_routers.contains(source));
        EXPECT_TRUE(links_by_source.emplace(source, &link).second) << "duplicate directed link source";
    }

    for (const auto& link : manifest.at("links")) {
        if (link.at("dst").is_null()) {
            continue;
        }
        const RouterKey source = router_key(link.at("src"));
        const RouterKey destination = router_key(link.at("dst"));
        const auto reverse = links_by_source.find(destination);
        if (reverse == links_by_source.end()) {
            continue;  // The peer belongs to another host's rank-local manifest.
        }
        EXPECT_EQ(router_key(reverse->second->at("dst")), source);
        EXPECT_EQ(reverse->second->at("routing_plane"), link.at("routing_plane"));
    }
}

}  // namespace

TEST_F(Fabric1DFixture, DebugManifestMatchesLiveFabric) { check_manifest_matches_live_fabric(FabricConfig::FABRIC_1D); }

TEST_F(Fabric2DFixture, DebugManifestMatchesLiveFabric) { check_manifest_matches_live_fabric(FabricConfig::FABRIC_2D); }

}  // namespace tt::tt_fabric::fabric_router_tests
