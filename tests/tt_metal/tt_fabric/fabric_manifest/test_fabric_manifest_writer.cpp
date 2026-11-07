// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The manifest writer, from hand-built facts: the conventions readers rely on, which the fixtures only see through
// whatever the fabric they bring up happens to produce. No device.

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <optional>
#include <string>
#include <vector>

#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_model.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_type_registry.hpp"

namespace tt::tt_fabric::manifest {
namespace {

using json = nlohmann::ordered_json;

content::L1 word_at(uint32_t address) { return register_l1_region<uint32_t>(address); }

// One mesh, M0, of two chips. C0 is this rank's, with one router, E0, whose link reaches W0 on C1, another rank's chip.
ManifestFacts make_facts() {
    const FabricNodeId local_node{MeshId{0}, 0};
    const FabricNodeId remote_node{MeshId{0}, 1};

    Router router{
        .identity = {.eth_chan = 0, .direction = eth_chan_directions::EAST, .routing_plane = 0},
        .link = {.peer = RouterRef{.node = remote_node, .direction = eth_chan_directions::WEST, .routing_plane = 0}},
        .shape = {.num_vcs = 1, .senders_per_vc = {1, 0, 0}, .receivers_per_vc = {1, 0, 0}, .num_active_eriscs = 1},
        .credit_counters = {word_at(0x100), word_at(0x104), word_at(0x108), word_at(0x10c)},
        .channels =
            {.senders = {{SenderChannel{
                 .serviced_by = {0},
                 .producer = LocalWorker{},
                 .credits_by_erisc = {SenderChannelCredits{.completed = std::optional<content::StreamReg>{}}},
             }}},
             .receivers = {{ReceiverChannel{.serviced_by = {0}, .forwards_on_by_erisc = {0}}}}},
        .intra_chip_downstream_edges = {},
        .fields_by_erisc = {{
            Field{
                .key = "edm_status",
                .category = FieldCategory::LIFECYCLE,
                .content = register_l1_region<EDMStatus>(0x200)},
            Field{
                .key = "retrain_sync",
                .category = FieldCategory::LIFECYCLE,
                .content =
                    content::StreamReg{
                        .stream_id = 30,
                        .address = 0xffb5e000,
                        .value_bits = 24,
                        .type = get_manifest_type_name<CoordinatedEriscContextSwitchState>()}},
            Field{
                .key = "my_direction",
                .category = FieldCategory::KERNEL_PARAMS,
                .content = content::Enum{.type = "eth_chan_directions", .value = 0, .value_name = "EAST"}},
            Field{.key = "perf_telemetry", .category = FieldCategory::DIAGNOSTICS, .content = std::nullopt},
        }},
        .eriscs = {Erisc{.processor = tt::tt_metal::DataMovementProcessor::RISCV_0, .physical_erisc = 0}},
        .leftover_l1 = std::nullopt,
    };

    Chip local_chip{
        .fabric_node_id = local_node,
        .mesh_coord = {0, 0},
        .physical_chip_id = 0,
        .asic_id = 0x1234,
        .local_sync =
            LocalSync{
                .master_eth_chan = 0,
                .master = RouterRef{.node = local_node, .direction = eth_chan_directions::EAST, .routing_plane = 0},
                .num_routers = 1,
                .router_channels_mask = 1},
        .routers = {router},
    };
    Chip remote_chip{.fabric_node_id = remote_node, .mesh_coord = {0, 1}, .host_rank = 1};

    return ManifestFacts{
        .run = {},
        .fabric =
            {.config = {.fabric_config = FabricConfig::FABRIC_1D},
             .packet_header = get_manifest_type_name<LowLatencyPacketHeaderT<0>>(),
             .packet_header_size_bytes = sizeof(LowLatencyPacketHeaderT<0>),
             .routing_1d_extension_words = 0,
             .host_clears = {{.address = 0x100, .size_bytes = 4}}},
        .arch =
            {.arch = tt::ARCH::WORMHOLE_B0,
             .areas =
                 {.heartbeat = {.word = word_at(0x1f80)},
                  .fabric_telemetry = register_l1_region<FabricTelemetry>(0x3000),
                  .routing_table = word_at(0x4000)},
             .core_registers =
                 {.soft_reset = 0x1, .reset_masks = {0x800}, .wall_clock_low = 0x2, .wall_clock_high = 0x3}},
        .meshes = {Mesh{
            .mesh_id = MeshId{0},
            .shape = {1, 2},
            .torus = MeshTorus{},
            .express_routing = false,
            .credit_transport = std::vector<VcCreditTransport>{{.vc = 0}},
            .chips = {local_chip, remote_chip},
        }},
    };
}

const json& router_e0(const json& manifest) {
    return manifest.at("meshes").at("M0").at("chips").at("C0").at("routers").at("E0");
}

// A remote chip says only where it is in its mesh: the ids this rank can't read are null, and it has no routers.
TEST(ManifestWriter, RemoteChipHasOnlyItsPlace) {
    const auto manifest = make_manifest_json(make_facts());
    EXPECT_EQ(
        manifest.at("meshes").at("M0").at("chips").at("C1"),
        json::parse(
            R"({"mesh_coord": [0, 1], "host_rank": 1, "physical_chip_id": null, "asic_id": null, "is_local": false})"));
}

// Something the builder didn't allocate is null, and a key that doesn't apply to this run is left out.
TEST(ManifestWriter, UnallocatedIsNullAndInapplicableIsLeftOut) {
    const auto manifest = make_manifest_json(make_facts());
    const auto& router = router_e0(manifest);
    EXPECT_TRUE(router.at("fields").at("erisc0").at("perf_telemetry").is_null());
    EXPECT_TRUE(router.at("leftover_l1").is_null());
    const auto& sender = router.at("channels").at("senders").at("vc0").at("ch0");
    EXPECT_TRUE(sender.at("ring_buffer").is_null());
    EXPECT_TRUE(sender.at("credits").at("erisc0").at("completed").is_null());
    EXPECT_FALSE(sender.at("credits").at("erisc0").contains("acked"));
    EXPECT_FALSE(router.at("channels").at("senders").contains("vc1"));
    EXPECT_FALSE(router.at("intra_chip_downstream_edges").contains("vc0"));
    EXPECT_TRUE(manifest.at("fabric").contains("routing_1d_extension_words"));
    EXPECT_FALSE(manifest.at("fabric").contains("routing_2d_route_buffer_size"));
}

// A router another part refers to is written as its path, and an ASIC id as a hex string, since it can exceed what a
// JSON number holds exactly.
TEST(ManifestWriter, ReferencesAndIds) {
    const auto manifest = make_manifest_json(make_facts());
    const auto& chip = manifest.at("meshes").at("M0").at("chips").at("C0");
    EXPECT_EQ(router_e0(manifest).at("link").at("peer"), "M0/C1/W0");
    EXPECT_EQ(chip.at("local_sync").at("master"), "M0/C0/E0");
    EXPECT_EQ(chip.at("asic_id"), "0x0000000000001234");
}

// A region and a stream register name the registered type their bytes hold, and a configuration enum names its
// enumerator. The manifest describes no type's members: decode gets the type back from its name.
TEST(ManifestWriter, ContentsNameTheirType) {
    const auto manifest = make_manifest_json(make_facts());
    const auto& fields = router_e0(manifest).at("fields").at("erisc0");
    EXPECT_EQ(
        fields.at("edm_status"),
        json(
            {{"category", "lifecycle"},
             {"kind", "l1"},
             {"address", 0x200},
             {"num_elements", 1},
             {"element_size", sizeof(EDMStatus)},
             {"type", "EDMStatus"}}));
    EXPECT_EQ(
        fields.at("retrain_sync"),
        json(
            {{"category", "lifecycle"},
             {"kind", "stream_reg"},
             {"stream_id", 30},
             {"address", 0xffb5e000},
             {"value_bits", 24},
             {"type", "CoordinatedEriscContextSwitchState"}}));
    // The enum's own name isn't written: it stays in the model, for FIELDS.md and errors.
    EXPECT_EQ(
        fields.at("my_direction"),
        json({{"category", "kernel_params"}, {"kind", "enum"}, {"value", 0}, {"value_name", "east"}}));
    // A counter array is its elements' type and count.
    const auto& counters = router_e0(manifest).at("credit_counters").at("to_sender_ack");
    EXPECT_EQ(counters.at("type"), get_manifest_type_name<uint32_t>());
    EXPECT_EQ(counters.at("element_size"), sizeof(uint32_t));
    // The run's packet header, with its template argument even though it equals the default.
    EXPECT_EQ(manifest.at("fabric").at("packet_header"), "LowLatencyPacketHeaderT<0>");
    EXPECT_EQ(manifest.at("fabric").at("packet_header_size_bytes"), sizeof(LowLatencyPacketHeaderT<0>));
    // What fabric init zeroes, as an address and a size.
    EXPECT_EQ(manifest.at("fabric").at("host_clears"), json::parse(R"([{"address": 256, "size_bytes": 4}])"));
    EXPECT_FALSE(manifest.contains("types"));
}

// Each ERISC's values are written under its own key, as the builder gave them, even where two ERISCs differ.
TEST(ManifestWriter, WritesEachEriscsValuesUnderItsOwnKey) {
    auto facts = make_facts();
    auto& router = facts.meshes.at(0).chips.at(0).routers.at(0);
    router.shape.num_active_eriscs = 2;
    router.eriscs.push_back(Erisc{.processor = tt::tt_metal::DataMovementProcessor::RISCV_1, .physical_erisc = 1});
    router.fields_by_erisc.push_back({Field{
        .key = "speedy_vc0", .category = FieldCategory::KERNEL_PARAMS, .content = content::Flag{.value = true}}});
    auto& sender = router.channels.senders.at(0).at(0);
    sender.credits_by_erisc.push_back(SenderChannelCredits{
        .acked = CounterRef{.array = CreditCounterArray::TO_SENDER_ACK, .index = 0},
        .completed = CounterRef{.array = CreditCounterArray::TO_SENDER_COMPLETION, .index = 0}});
    router.channels.receivers.at(0).at(0).forwards_on_by_erisc.emplace_back(std::nullopt);

    const auto manifest = make_manifest_json(facts);
    const auto& router_json = router_e0(manifest);
    EXPECT_TRUE(router_json.at("fields").at("erisc0").contains("edm_status"));
    EXPECT_EQ(router_json.at("fields").at("erisc1").at("speedy_vc0").at("value"), true);
    const auto& credits = router_json.at("channels").at("senders").at("vc0").at("ch0").at("credits");
    EXPECT_FALSE(credits.at("erisc0").contains("acked"));
    EXPECT_EQ(credits.at("erisc1").at("acked").at("array"), "credit_counters/to_sender_ack");
    const auto& forwards_on = router_json.at("channels").at("receivers").at("vc0").at("ch0").at("forwards_on");
    EXPECT_EQ(forwards_on, json::parse(R"({"erisc0": "vc0", "erisc1": null})"));
    EXPECT_FALSE(router_json.at("eriscs").at("erisc1").contains("fields"));
}

// Two fields with one key would leave readers one of them, so the writer refuses.
TEST(ManifestWriter, FailsOnTwoFieldsWithOneKey) {
    auto facts = make_facts();
    auto& fields = facts.meshes.at(0).chips.at(0).routers.at(0).fields_by_erisc.at(0);
    fields.push_back(fields.front());
    EXPECT_ANY_THROW(make_manifest_json(facts));
}

}  // namespace
}  // namespace tt::tt_fabric::manifest
