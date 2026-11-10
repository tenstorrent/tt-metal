// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The manifest writer and reader, from hand-built facts: the conventions readers rely on, which the fixtures only see
// through whatever the fabric they bring up happens to produce, and the reader's strictness. No device.

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <cstddef>
#include <exception>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "tt_metal/fabric/debug/visualizer/common/fabric_visualizer_files.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_model.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_type_registry.hpp"

namespace tt::tt_fabric::manifest {
namespace {

using json = nlohmann::ordered_json;

content::L1 word_at(uint32_t address) { return register_l1_region<uint32_t>(address); }

content::L1 multi_sampled_as(content::L1 region, MultiSampleRole role) {
    region.multi_sampled = role;
    return region;
}

// The areas every architecture has, marked as describe marks them; the rest are null, as on Wormhole.
ArchAreas make_arch_areas() {
    ArchAreas areas;
    areas.get(ArchArea::HEARTBEAT) = multi_sampled_as(word_at(0x1f80), MultiSampleRole::LOOP_HEARTBEAT);
    areas.get(ArchArea::FABRIC_TELEMETRY) =
        multi_sampled_as(register_l1_region<FabricTelemetry>(0x3000), MultiSampleRole::FABRIC_TELEMETRY);
    areas.get(ArchArea::ROUTING_TABLE) = word_at(0x4000);
    return areas;
}

// One mesh, M0, of two chips. C0 is this rank's, with one router, E0, whose link reaches W0 on C1, another rank's chip.
ManifestFacts make_facts() {
    const FabricNodeId local_node{MeshId{0}, 0};
    const FabricNodeId remote_node{MeshId{0}, 1};

    Router router{
        .identity = {.eth_chan = 0, .key = {.direction = eth_chan_directions::EAST, .routing_plane = 0}},
        .link = {.peer = RouterRef{.node = remote_node, .key = {.direction = eth_chan_directions::WEST}}},
        .shape = {.num_vcs = 1, .senders_per_vc = {1, 0, 0}, .receivers_per_vc = {1, 0, 0}, .num_active_eriscs = 1},
        .credit_counters = {word_at(0x100), word_at(0x104), word_at(0x108), word_at(0x10c)},
        .channels =
            {.senders = {{SenderChannel{
                 .serviced_by = {0},
                 .producer = LocalWorker{},
                 .credits_by_erisc = {SenderChannelCredits{
                     .completed = CounterRef{.array = CreditCounterArray::TO_SENDER_COMPLETION, .index = 0}}},
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
                        .type = std::string(get_manifest_type_name<CoordinatedEriscContextSwitchState>())}},
            Field{
                .key = "my_direction",
                .category = FieldCategory::KERNEL_PARAMS,
                .content = content::Enum{.value = 0, .value_name = "east"}},
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
                .master = RouterRef{.node = local_node, .key = {.direction = eth_chan_directions::EAST}},
                .num_routers = 1,
                .router_channels_mask = 1},
        .routers = {router},
    };
    Chip remote_chip{.fabric_node_id = remote_node, .mesh_coord = {0, 1}, .host_rank = 1};

    return ManifestFacts{
        // The reader reads only what this build wrote.
        .run = {.build = visualizer::get_build_identity()},
        .fabric =
            {.config = {.fabric_config = FabricConfig::FABRIC_1D},
             .packet_header = std::string(get_manifest_type_name<LowLatencyPacketHeaderT<0>>()),
             .packet_header_size_bytes = sizeof(LowLatencyPacketHeaderT<0>),
             .routing_1d_extension_words = 0,
             .host_clears = {{.address = 0x100, .size_bytes = 4}}},
        .arch =
            {.arch = tt::ARCH::WORMHOLE_B0,
             .areas = make_arch_areas(),
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

json& router_e0(json& manifest) { return manifest.at("meshes").at("M0").at("chips").at("C0").at("routers").at("E0"); }

// A remote chip says only where it is in its mesh: the ids this rank can't read are null, and it has no routers.
TEST(ManifestWriter, RemoteChipHasOnlyItsPlace) {
    const auto manifest = make_manifest_json(make_facts());
    EXPECT_EQ(
        manifest.at("meshes").at("M0").at("chips").at("C1"),
        json::parse(
            R"({"mesh_coord": [0, 1], "host_rank": 1, "physical_chip_id": null, "asic_id": null, "is_local": false})"));
}

// Each fields id in a SHARED manifest replaced by its block from field_sets, and field_sets dropped: the manifest as
// PER_ROUTER writes it.
json expand_field_sets(json manifest) {
    const json field_sets = manifest.at("field_sets");
    manifest.erase("field_sets");
    const auto expand = [&](auto& self, json& node) -> void {
        if (!node.is_object()) {
            return;
        }
        for (auto& [key, value] : node.items()) {
            if (key == "fields" && value.is_string()) {
                value = field_sets.at(value.get<std::string>());
            } else {
                self(self, value);
            }
        }
    };
    expand(expand, manifest);
    return manifest;
}

// SHARED writes each distinct fields block once, in field_sets, keyed in the order each is first written, and its id
// where the block was. The sender and receiver channels here have the same (empty) fields, so they share one.
TEST(ManifestWriter, SharedLayoutWritesEachDistinctBlockOnce) {
    const auto facts = make_facts();
    const auto manifest = make_manifest_json(facts);
    const auto& router = router_e0(manifest);
    EXPECT_EQ(router.at("channels").at("senders").at("vc0").at("ch0").at("fields"), "F0");
    EXPECT_EQ(router.at("channels").at("receivers").at("vc0").at("ch0").at("fields"), "F0");
    EXPECT_EQ(router.at("fields"), "F1");
    const auto per_router = make_manifest_json(facts, FieldsLayout::PER_ROUTER);
    EXPECT_EQ(manifest.at("field_sets"), json({{"F0", json::object()}, {"F1", router_e0(per_router).at("fields")}}));
}

// PER_ROUTER writes each block where it is, with no field_sets; otherwise the two layouts write the same manifest.
TEST(ManifestWriter, LayoutsDifferOnlyInWhereFieldsAre) {
    const auto facts = make_facts();
    const auto per_router = make_manifest_json(facts, FieldsLayout::PER_ROUTER);
    EXPECT_FALSE(per_router.contains("field_sets"));
    EXPECT_TRUE(router_e0(per_router).at("fields").at("erisc0").contains("edm_status"));
    EXPECT_EQ(expand_field_sets(make_manifest_json(facts)), per_router);
}

// Something the builder didn't allocate is null, and a key that doesn't apply to this run is left out.
TEST(ManifestWriter, UnallocatedIsNullAndInapplicableIsLeftOut) {
    const auto manifest = make_manifest_json(make_facts(), FieldsLayout::PER_ROUTER);
    const auto& router = router_e0(manifest);
    EXPECT_TRUE(router.at("fields").at("erisc0").at("perf_telemetry").is_null());
    EXPECT_TRUE(router.at("leftover_l1").is_null());
    const auto& sender = router.at("channels").at("senders").at("vc0").at("ch0");
    EXPECT_TRUE(sender.at("ring_buffer").is_null());
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
    // A credit names its counter array by its path in the router.
    EXPECT_EQ(
        router_e0(manifest).at("channels").at("senders").at("vc0").at("ch0").at("credits").at("erisc0").at("completed"),
        json({{"array", "credit_counters/to_sender_completion"}, {"index", 0}}));
}

// A region and a stream register name the registered type their bytes hold, and a configuration enum names its
// enumerator. The manifest describes no type's members: decode gets the type back from its name.
TEST(ManifestWriter, ContentsNameTheirType) {
    const auto manifest = make_manifest_json(make_facts(), FieldsLayout::PER_ROUTER);
    const auto& fields = router_e0(manifest).at("fields").at("erisc0");
    EXPECT_EQ(
        fields.at("edm_status"),
        json(
            {{"category", "lifecycle"},
             {"kind", "l1"},
             {"address", 0x200},
             {"num_elements", 1},
             {"element_size", sizeof(EDMStatus)},
             {"type", "EDMStatus"},
             {"multi_sampled", nullptr},
             {"active", true}}));
    EXPECT_EQ(
        fields.at("retrain_sync"),
        json(
            {{"category", "lifecycle"},
             {"kind", "stream_reg"},
             {"stream_id", 30},
             {"address", 0xffb5e000},
             {"value_bits", 24},
             {"type", "CoordinatedEriscContextSwitchState"},
             {"multi_sampled", nullptr}}));
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

// A multi-sampled region names its role, and every other region and stream register is null: capture reads it only in
// its full read.
TEST(ManifestWriter, MultiSampledRegionsNameTheirRole) {
    const auto manifest = make_manifest_json(make_facts(), FieldsLayout::PER_ROUTER);
    const auto& areas = manifest.at("arch").at("l1_areas");
    EXPECT_EQ(areas.at("heartbeat").at("multi_sampled"), "loop_heartbeat");
    EXPECT_EQ(areas.at("fabric_telemetry").at("multi_sampled"), "fabric_telemetry");
    EXPECT_TRUE(areas.at("routing_table").at("multi_sampled").is_null());
    const auto& router = router_e0(manifest);
    EXPECT_TRUE(router.at("credit_counters").at("to_sender_ack").at("multi_sampled").is_null());
    EXPECT_TRUE(router.at("fields").at("erisc0").at("retrain_sync").at("multi_sampled").is_null());
    EXPECT_EQ(manifest.at("vocabulary").at("multi_sample_role"), json({"loop_heartbeat", "fabric_telemetry"}));
}

// Every L1 region says whether anything writes it: leftover L1 doesn't, and the rest here do. A stream register has no
// active: the kernel or the hardware always holds its value.
TEST(ManifestWriter, ActiveSaysWhetherAnythingWritesTheRegion) {
    auto facts = make_facts();
    auto& leftover = facts.meshes.at(0).chips.at(0).routers.at(0).leftover_l1;
    leftover = register_l1_region<std::byte>(0x8000, 0x100, 1);
    leftover->active = false;
    const auto manifest = make_manifest_json(facts, FieldsLayout::PER_ROUTER);
    const auto& router = router_e0(manifest);
    EXPECT_EQ(router.at("leftover_l1").at("active"), false);
    EXPECT_EQ(router.at("fields").at("erisc0").at("edm_status").at("active"), true);
    EXPECT_EQ(router.at("credit_counters").at("to_sender_ack").at("active"), true);
    EXPECT_EQ(manifest.at("arch").at("l1_areas").at("heartbeat").at("active"), true);
    EXPECT_FALSE(router.at("fields").at("erisc0").at("retrain_sync").contains("active"));
}

// Each ERISC's values are written under its own key, as the builder gave them, even where two ERISCs differ.
TEST(ManifestWriter, WritesEachEriscsValuesUnderItsOwnKey) {
    auto facts = make_facts();
    auto& router = facts.meshes.at(0).chips.at(0).routers.at(0);
    router.shape.num_active_eriscs = 2;
    router.eriscs.push_back(Erisc{
        .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
        .physical_erisc = 1,
        .opt_level = tt::tt_metal::KernelBuildOptLevel::Os});
    router.fields_by_erisc.push_back({Field{
        .key = "speedy_vc0", .category = FieldCategory::KERNEL_PARAMS, .content = content::Flag{.value = true}}});
    auto& sender = router.channels.senders.at(0).at(0);
    sender.credits_by_erisc.push_back(SenderChannelCredits{
        .acked = CounterRef{.array = CreditCounterArray::TO_SENDER_ACK, .index = 0},
        .completed = CounterRef{.array = CreditCounterArray::TO_SENDER_COMPLETION, .index = 0}});
    router.channels.receivers.at(0).at(0).forwards_on_by_erisc.emplace_back(std::nullopt);

    const auto manifest = make_manifest_json(facts, FieldsLayout::PER_ROUTER);
    const auto& router_json = router_e0(manifest);
    EXPECT_TRUE(router_json.at("fields").at("erisc0").contains("edm_status"));
    EXPECT_EQ(router_json.at("fields").at("erisc1").at("speedy_vc0").at("value"), true);
    const auto& credits = router_json.at("channels").at("senders").at("vc0").at("ch0").at("credits");
    EXPECT_FALSE(credits.at("erisc0").contains("acked"));
    EXPECT_EQ(credits.at("erisc1").at("acked").at("array"), "credit_counters/to_sender_ack");
    const auto& forwards_on = router_json.at("channels").at("receivers").at("vc0").at("ch0").at("forwards_on");
    EXPECT_EQ(forwards_on, json::parse(R"({"erisc0": "vc0", "erisc1": null})"));
    EXPECT_FALSE(router_json.at("eriscs").at("erisc1").contains("fields"));
    EXPECT_EQ(router_json.at("eriscs").at("erisc0").at("opt_level"), "o3");
    EXPECT_EQ(router_json.at("eriscs").at("erisc1").at("opt_level"), "os");
}

// Two fields with one key would leave readers one of them, so the writer refuses.
TEST(ManifestWriter, FailsOnTwoFieldsWithOneKey) {
    auto facts = make_facts();
    auto& fields = facts.meshes.at(0).chips.at(0).routers.at(0).fields_by_erisc.at(0);
    fields.push_back(fields.front());
    EXPECT_ANY_THROW(make_manifest_json(facts));
}

// ============ Model ============

// A CounterRef's array resolves to its region, and each array says whose sender channel indexes it.
TEST(ManifestModel, CreditCounterArraysResolve) {
    const L1CreditCounters counters{word_at(0x100), word_at(0x104), word_at(0x108), word_at(0x10c)};
    EXPECT_EQ(credit_counter_region(counters, CreditCounterArray::TO_SENDER_ACK).address, 0x100u);
    EXPECT_EQ(credit_counter_region(counters, CreditCounterArray::TO_SENDER_COMPLETION).address, 0x104u);
    EXPECT_EQ(credit_counter_region(counters, CreditCounterArray::RECEIVER_ACK).address, 0x108u);
    EXPECT_EQ(credit_counter_region(counters, CreditCounterArray::RECEIVER_COMPLETION).address, 0x10cu);
    EXPECT_EQ(
        credit_counter_indexed_by(CreditCounterArray::TO_SENDER_COMPLETION),
        CreditCounterIndexedBy::OWN_SENDER_CHANNEL);
    EXPECT_EQ(credit_counter_indexed_by(CreditCounterArray::RECEIVER_ACK), CreditCounterIndexedBy::PEER_SENDER_CHANNEL);
}

// ============ Reader ============

// make_facts with what it leaves out: a sub-context, a sibling producer, an acked credit, a downstream edge, live
// downstreams, fields of every kind, a multi-sampled stream register, and leftover L1, which nothing writes.
ManifestFacts make_facts_with_every_part() {
    auto facts = make_facts();
    facts.run.subcontext = Subcontext{.id = 1, .rank = 0, .size = 2};
    auto& router = facts.meshes.at(0).chips.at(0).routers.at(0);
    auto& sender = router.channels.senders.at(0).at(0);
    sender.producer = SiblingRouterRef{.direction = eth_chan_directions::NORTH};
    sender.credits_by_erisc.at(0).acked = CounterRef{.array = CreditCounterArray::TO_SENDER_ACK, .index = 0};
    router.intra_chip_downstream_edges.resize(builder_config::MAX_NUM_VCS);
    router.intra_chip_downstream_edges.at(0).push_back(DownstreamEdge{
        .edge = 1,
        .downstream_channel = {.target = {.direction = eth_chan_directions::NORTH}, .landing_channel = 0},
        .fields_by_erisc = {{Field{
            .key = "edge_count", .category = FieldCategory::FLOW_CONTROL, .content = content::Number{.value = 4}}}},
    });
    router.live_downstreams = LiveDownstreamMasks{
        Routing2DCodec::ACTION_WEST,
        Routing2DCodec::ACTION_WEST | Routing2DCodec::ACTION_NORTH | Routing2DCodec::ACTION_SOUTH,
    };
    router.fields_by_erisc.at(0).push_back(
        Field{.key = "speedy", .category = FieldCategory::KERNEL_PARAMS, .content = content::Flag{.value = true}});
    router.fields_by_erisc.at(0).push_back(Field{
        .key = "sampled_stream",
        .category = FieldCategory::FLOW_CONTROL,
        .content = content::StreamReg{
            .stream_id = 31,
            .address = 0xffb5f000,
            .value_bits = 24,
            .type = std::string(get_manifest_type_name<uint32_t>()),
            .multi_sampled = MultiSampleRole::LOOP_HEARTBEAT}});
    router.leftover_l1 = register_l1_region<std::byte>(0x8000, 0x100, 1);
    router.leftover_l1->active = false;
    return facts;
}

// What reading `manifest` fails with, or empty when it doesn't fail.
std::string parse_error(const json& manifest) {
    try {
        parse_manifest_json(manifest);
    } catch (const std::exception& error) {
        return error.what();
    }
    return {};
}

void expect_parse_fails_with(const json& manifest, std::string_view expected) {
    const std::string message = parse_error(manifest);
    EXPECT_FALSE(message.empty()) << "expected a failure with: " << expected;
    EXPECT_NE(message.find(expected), std::string::npos) << message;
}

// Writing the facts, reading them back and writing again gives the same JSON, in each layout.
TEST(ManifestReader, RoundTripsInEachLayout) {
    for (const auto& facts : {make_facts(), make_facts_with_every_part()}) {
        for (const FieldsLayout layout : {FieldsLayout::SHARED, FieldsLayout::PER_ROUTER}) {
            const json written = make_manifest_json(facts, layout);
            EXPECT_EQ(make_manifest_json(parse_manifest_json(written), layout), written);
        }
    }
}

// Either layout reads into the same facts.
TEST(ManifestReader, ReadsEitherLayoutIntoTheSameFacts) {
    const auto facts = make_facts_with_every_part();
    const json per_router = make_manifest_json(facts, FieldsLayout::PER_ROUTER);
    EXPECT_EQ(make_manifest_json(parse_manifest_json(make_manifest_json(facts)), FieldsLayout::PER_ROUTER), per_router);
}

// A key the writer writes that the reader doesn't read fails, named by its pointer, whatever its value.
TEST(ManifestReader, FailsOnAKeyItDoesNotRead) {
    json manifest = make_manifest_json(make_facts());
    manifest["run"]["pid"] = 0;
    expect_parse_fails_with(manifest, "Fabric manifest: /run has keys this reader doesn't read: /run/pid");
}

// A key the reader reads that the writer didn't write fails, named by its pointer.
TEST(ManifestReader, FailsOnAMissingKey) {
    json manifest = make_manifest_json(make_facts());
    router_e0(manifest).at("shape").erase("num_vcs");
    expect_parse_fails_with(manifest, "Fabric manifest: /meshes/M0/chips/C0/routers/E0/shape/num_vcs is missing");
}

TEST(ManifestReader, FailsOnAnotherFileKindVersionOrBuild) {
    json capture = make_manifest_json(make_facts());
    capture["file_kind"] = "fabric_capture";
    expect_parse_fails_with(capture, R"(Fabric manifest: /file_kind is "fabric_capture", not "fabric_manifest")");

    json newer = make_manifest_json(make_facts());
    newer["manifest_version"] = visualizer::k_manifest_version + 1;
    expect_parse_fails_with(newer, "Fabric manifest: /manifest_version is ");

    json other_build = make_manifest_json(make_facts());
    other_build["run"]["build"]["git_sha"] = "0000000";
    expect_parse_fails_with(other_build, "Fabric manifest: /run/build is ");
}

// A type name this build doesn't register is one decode couldn't read bytes as.
TEST(ManifestReader, FailsOnATypeThisBuildDoesNotRegister) {
    json manifest = make_manifest_json(make_facts(), FieldsLayout::PER_ROUTER);
    router_e0(manifest).at("fields").at("erisc0").at("edm_status").at("type") = "NotAType";
    expect_parse_fails_with(
        manifest,
        R"(Fabric manifest: /meshes/M0/chips/C0/routers/E0/fields/erisc0/edm_status/type is "NotAType", a type this )"
        "build doesn't register");
}

// A multi_sampled that names no role fails, as does a region without the key.
TEST(ManifestReader, FailsOnAMultiSampledItCannotRead) {
    json unknown = make_manifest_json(make_facts());
    unknown.at("arch").at("l1_areas").at("heartbeat").at("multi_sampled") = "sometimes";
    expect_parse_fails_with(
        unknown, R"(Fabric manifest: /arch/l1_areas/heartbeat/multi_sampled is "sometimes", which names no )");

    json missing = make_manifest_json(make_facts());
    router_e0(missing).at("credit_counters").at("to_sender_ack").erase("multi_sampled");
    expect_parse_fails_with(
        missing,
        "Fabric manifest: /meshes/M0/chips/C0/routers/E0/credit_counters/to_sender_ack/multi_sampled is missing");
}

// Every router core writes an arch area, so one that says it's inactive fails.
TEST(ManifestReader, FailsOnAnInactiveArchArea) {
    json manifest = make_manifest_json(make_facts());
    manifest.at("arch").at("l1_areas").at("routing_table").at("active") = false;
    expect_parse_fails_with(
        manifest,
        "Fabric manifest: /arch/l1_areas/routing_table is inactive, but every router core writes an arch area");
}

// Sampling a region nothing writes says nothing, so an inactive multi-sampled region fails.
TEST(ManifestReader, FailsOnAnInactiveMultiSampledRegion) {
    json manifest = make_manifest_json(make_facts());
    auto& counters = router_e0(manifest).at("credit_counters").at("to_sender_ack");
    counters.at("multi_sampled") = "loop_heartbeat";
    counters.at("active") = false;
    expect_parse_fails_with(
        manifest,
        "Fabric manifest: /meshes/M0/chips/C0/routers/E0/credit_counters/to_sender_ack/active is false, but the region "
        "is multi_sampled, and nothing writes it");
}

// A fields id names a block in field_sets, and in the PER_ROUTER layout there are no ids.
TEST(ManifestReader, FailsOnAFieldsIdItCannotResolve) {
    json shared = make_manifest_json(make_facts());
    router_e0(shared).at("fields") = "F9";
    expect_parse_fails_with(
        shared, R"(Fabric manifest: /meshes/M0/chips/C0/routers/E0/fields is "F9", which field_sets doesn't have)");

    json per_router = make_manifest_json(make_facts(), FieldsLayout::PER_ROUTER);
    router_e0(per_router).at("fields") = "F0";
    expect_parse_fails_with(
        per_router, R"(Fabric manifest: /meshes/M0/chips/C0/routers/E0/fields is "F0", not an object)");
}

// A remote chip has only its place, and says so: is_local is whether the chip has this rank's ids.
TEST(ManifestReader, FailsWhenIsLocalDisagreesWithTheIds) {
    json manifest = make_manifest_json(make_facts());
    manifest.at("meshes").at("M0").at("chips").at("C1").at("is_local") = true;
    expect_parse_fails_with(manifest, "Fabric manifest: /meshes/M0/chips/C1/is_local is true, not false");
}

}  // namespace
}  // namespace tt::tt_fabric::manifest
