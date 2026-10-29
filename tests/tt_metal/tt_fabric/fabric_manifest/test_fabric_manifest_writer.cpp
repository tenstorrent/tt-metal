// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The manifest writer, from hand-built facts: the conventions readers rely on, which the fixtures only see through
// whatever the fabric they bring up happens to produce. No device.

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <optional>
#include <string>
#include <vector>

#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_model.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_struct_layouts.hpp"

namespace tt::tt_fabric::manifest {
namespace {

using json = nlohmann::ordered_json;

content::L1 word_at(uint32_t address) { return {.address = address, .type = layout::type_of<uint32_t>()}; }

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
                 .credits = {.completed = std::optional<content::Stream>{}},
             }}},
             .receivers = {{ReceiverChannel{.serviced_by = {0}, .forwards_on = 0}}}},
        .intra_chip_downstream_edges = {},
        .fields =
            {
                Field{
                    .key = "edm_status",
                    .category = FieldCategory::LIFECYCLE,
                    .content = content::L1{.address = 0x200, .type = layout::type_of<EDMStatus>()}},
                Field{.key = "perf_telemetry", .category = FieldCategory::DIAGNOSTICS, .content = std::nullopt},
            },
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
    Chip remote_chip{.fabric_node_id = remote_node, .mesh_coord = {0, 1}};

    return ManifestFacts{
        .run = {},
        .fabric =
            {.config = {.fabric_config = FabricConfig::FABRIC_1D},
             .packet_header = layout::type_of<LowLatencyPacketHeaderT<0>>(),
             .routing_1d_extension_words = 0},
        .arch =
            {.arch = tt::ARCH::WORMHOLE_B0,
             .areas =
                 {.heartbeat = {.word = word_at(0x1f80)},
                  .fabric_telemetry = {.address = 0x3000, .type = layout::type_of<FabricTelemetry>()},
                  .routing_table = word_at(0x4000)},
             .stream_registers =
                 {.base = 0x10000,
                  .stride = 0x1000,
                  .count = 32,
                  .registers = {{StreamRegister::BUF_SPACE_AVAILABLE, {.offset = 0x100, .bits = 17}}}},
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

// Every schema string under `node`.
void collect_schemas(const json& node, std::vector<std::string>& schemas) {
    if (node.is_array()) {
        for (const auto& element : node) {
            collect_schemas(element, schemas);
        }
    } else if (node.is_object()) {
        for (const auto& [key, value] : node.items()) {
            if (key == "schema" && value.is_string()) {
                schemas.push_back(value.get<std::string>());
            } else {
                collect_schemas(value, schemas);
            }
        }
    }
}

// A remote chip says only where it is in its mesh: the ids this rank can't read are null, and it has no routers.
TEST(ManifestWriter, RemoteChipHasOnlyItsPlace) {
    const auto manifest = make_manifest_json(make_facts());
    EXPECT_EQ(
        manifest.at("meshes").at("M0").at("chips").at("C1"),
        json::parse(R"({"mesh_coord": [0, 1], "physical_chip_id": null, "asic_id": null, "is_local": false})"));
}

// Something the builder didn't allocate is null, and a key that doesn't apply to this run is left out.
TEST(ManifestWriter, UnallocatedIsNullAndInapplicableIsLeftOut) {
    const auto manifest = make_manifest_json(make_facts());
    const auto& router = router_e0(manifest);
    EXPECT_TRUE(router.at("fields").at("perf_telemetry").is_null());
    EXPECT_TRUE(router.at("leftover_l1").is_null());
    const auto& sender = router.at("channels").at("senders").at("vc0").at("ch0");
    EXPECT_TRUE(sender.at("ring_buffer").is_null());
    EXPECT_TRUE(sender.at("credits").at("completed").is_null());
    EXPECT_FALSE(sender.at("credits").contains("acked"));
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

// types holds every type a schema names, and the types those name, sorted by name, and nothing a schema doesn't reach.
TEST(ManifestWriter, TypesHoldExactlyWhatSchemasReach) {
    const auto manifest = make_manifest_json(make_facts());
    const auto& structs = manifest.at("types").at("structs");
    const auto& enums = manifest.at("types").at("enums");
    const auto& tagged_unions = manifest.at("types").at("tagged_unions");

    std::vector<std::string> schemas;
    collect_schemas(manifest, schemas);
    for (const auto& schema : schemas) {
        if (schema.starts_with("struct:")) {
            EXPECT_TRUE(structs.contains(schema.substr(7))) << schema;
        } else if (schema.starts_with("enum:")) {
            EXPECT_TRUE(enums.contains(schema.substr(5))) << schema;
        } else if (schema.starts_with("union:")) {
            EXPECT_TRUE(tagged_unions.contains(schema.substr(6))) << schema;
        }
    }
    EXPECT_TRUE(structs.contains("FabricTelemetry"));
    EXPECT_TRUE(enums.contains("EDMStatus"));
    // The run's packet header, named outside the meshes, with its template argument.
    EXPECT_EQ(manifest.at("fabric").at("packet_header").at("schema"), "struct:LowLatencyPacketHeaderT<0>");
    EXPECT_TRUE(structs.contains("LowLatencyPacketHeaderT<0>"));
    // A described struct no schema reaches.
    EXPECT_FALSE(structs.contains("WorkerXY"));

    std::vector<std::string> names;
    for (const auto& [name, _] : structs.items()) {
        names.push_back(name);
    }
    EXPECT_TRUE(std::ranges::is_sorted(names));
}

// Two fields with one key would leave readers one of them, so the writer refuses.
TEST(ManifestWriter, FailsOnTwoFieldsWithOneKey) {
    auto facts = make_facts();
    auto& fields = facts.meshes.at(0).chips.at(0).routers.at(0).fields;
    fields.push_back(fields.front());
    EXPECT_ANY_THROW(make_manifest_json(facts));
}

// A struct member holding a tagged union names the member that holds its tag, and the union lists the enum its tag
// holds and its cases, each picked by one of that enum's values.
TEST(ManifestWriter, TaggedUnionNamesItsTag) {
    const auto manifest = make_manifest_json(make_facts());
    const auto& types = manifest.at("types");
    const auto& members = types.at("structs").at("LowLatencyPacketHeaderT<0>").at("members");
    const auto command =
        std::find_if(members.begin(), members.end(), [](const json& m) { return m.at("name") == "command_fields"; });
    ASSERT_NE(command, members.end());
    EXPECT_EQ(command->at("schema"), "union:NocCommandFields");
    EXPECT_EQ(command->at("tag_member"), "noc_send_type");

    const auto& command_fields = types.at("tagged_unions").at("NocCommandFields");
    EXPECT_EQ(command_fields.at("tag").at("schema"), "enum:NocSendType");
    const auto& tag_values = types.at("enums").at("NocSendType").at("values");
    ASSERT_FALSE(command_fields.at("cases").empty());
    for (const auto& union_case : command_fields.at("cases")) {
        EXPECT_TRUE(tag_values.contains(union_case.at("tag_value").get<std::string>())) << union_case.at("name");
    }
}

}  // namespace
}  // namespace tt::tt_fabric::manifest
