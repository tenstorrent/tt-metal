// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/fabric/debug/visualizer/decode/fabric_decode_stream_registers.hpp"

#include <bit>
#include <type_traits>
#include <variant>
#include <vector>

#include <fmt/format.h>

#include <enchantum/enchantum.hpp>

#include "tt_metal/fabric/debug/visualizer/decode/fabric_decode_interpret.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_regions.hpp"

namespace tt::tt_fabric::decode {

namespace {

using json = nlohmann::ordered_json;

// The slots of a channel, from its ring buffer. Null when it has none.
std::optional<uint32_t> find_channel_depth(const std::optional<manifest::content::L1>& ring_buffer) {
    return ring_buffer.has_value() ? std::optional(ring_buffer->num_elements) : std::nullopt;
}

// The router on `ref`'s chip and routing plane that faces `sibling`'s direction. Null when the manifest has none.
const manifest::Router* find_sibling_router(
    const manifest::ManifestFacts& facts, const manifest::RouterRef& ref, const manifest::SiblingRouterRef& sibling) {
    const manifest::Chip* chip = manifest::find_chip(facts, ref.node);
    if (chip == nullptr) {
        return nullptr;
    }
    for (const manifest::Router& router : chip->routers) {
        if (router.identity.key.direction == sibling.direction &&
            router.identity.key.routing_plane == ref.key.routing_plane) {
            return &router;
        }
    }
    return nullptr;
}

// The slots of the sibling sender channel `edge`, on `vc`, lands on. Null for an edge landing on the tensix mux, or a
// sibling or channel the manifest doesn't have.
std::optional<uint32_t> find_landing_channel_depth(
    const manifest::ManifestFacts& facts,
    const manifest::RouterRef& ref,
    const manifest::DownstreamEdge& edge,
    size_t vc) {
    if (edge.through_tensix_mux) {
        return std::nullopt;
    }
    const manifest::Router* sibling = find_sibling_router(facts, ref, edge.downstream_channel.target);
    if (sibling == nullptr || vc >= sibling->channels.senders.size()) {
        return std::nullopt;
    }
    const std::vector<manifest::SenderChannel>& senders = sibling->channels.senders[vc];
    const uint32_t landing = edge.downstream_channel.landing_channel;
    return landing < senders.size() ? find_channel_depth(senders[landing].ring_buffer) : std::nullopt;
}

// Adds `facts_for_erisc(erisc)` for each stream register among `fields_by_erisc`.
template <typename FactsForErisc>
void add_field_stream_registers(
    const manifest::FieldsByErisc& fields_by_erisc,
    const FactsForErisc& facts_for_erisc,
    StreamRegisterFactsByRegister& out) {
    for (size_t erisc = 0; erisc < fields_by_erisc.size(); ++erisc) {
        for (const manifest::Field& field : fields_by_erisc[erisc]) {
            if (!field.content.has_value()) {
                continue;
            }
            if (const auto* stream = std::get_if<manifest::content::StreamReg>(&*field.content)) {
                out.emplace(stream, facts_for_erisc(erisc));
            }
        }
    }
}

// The producer's edge on `vc` that lands on sender channel `channel` of the sibling facing `consumer_direction`. Null
// when it has none, or the edge lands on the tensix mux.
const manifest::DownstreamEdge* find_edge_into_channel(
    const manifest::Router& producer, size_t vc, eth_chan_directions consumer_direction, size_t channel) {
    if (vc >= producer.intra_chip_downstream_edges.size()) {
        return nullptr;
    }
    for (const manifest::DownstreamEdge& edge : producer.intra_chip_downstream_edges[vc]) {
        if (!edge.through_tensix_mux && edge.downstream_channel.target.direction == consumer_direction &&
            edge.downstream_channel.landing_channel == channel) {
            return &edge;
        }
    }
    return nullptr;
}

}  // namespace

std::optional<bool> find_first_level_ack(const manifest::Router& router, size_t vc, size_t erisc) {
    if (vc >= router.channels.senders.size()) {
        return std::nullopt;
    }
    for (const manifest::SenderChannel& sender : router.channels.senders[vc]) {
        if (erisc < sender.credits_by_erisc.size()) {
            return sender.credits_by_erisc[erisc].acked.has_value();
        }
    }
    return std::nullopt;
}

const StreamCountRead* find_field_count_read(
    const manifest::FieldsByErisc& fields_by_erisc, std::string_view key, const StreamCountReads& reads) {
    for (const std::vector<manifest::Field>& fields : fields_by_erisc) {
        for (const manifest::Field& field : fields) {
            if (field.key != key || !field.content.has_value()) {
                continue;
            }
            const auto* stream = std::get_if<manifest::content::StreamReg>(&*field.content);
            if (const auto read = stream == nullptr ? reads.end() : reads.find(stream); read != reads.end()) {
                return &read->second;
            }
        }
    }
    return nullptr;
}

StreamRegisterFactsByRegister collect_stream_register_facts(
    const manifest::ManifestFacts& facts, const manifest::RouterRef& ref, const manifest::Router& router) {
    StreamRegisterFactsByRegister out;
    for (const auto& senders : router.channels.senders) {
        for (const manifest::SenderChannel& sender : senders) {
            const StreamRegisterFacts channel_facts{.depth = find_channel_depth(sender.ring_buffer)};
            add_field_stream_registers(sender.fields_by_erisc, [&](size_t) { return channel_facts; }, out);
        }
    }
    for (size_t vc = 0; vc < router.channels.receivers.size(); ++vc) {
        for (const manifest::ReceiverChannel& receiver : router.channels.receivers[vc]) {
            add_field_stream_registers(
                receiver.fields_by_erisc,
                [&](size_t erisc) {
                    return StreamRegisterFacts{
                        .depth = find_channel_depth(receiver.ring_buffer),
                        .first_level_ack = find_first_level_ack(router, vc, erisc)};
                },
                out);
        }
    }
    for (size_t vc = 0; vc < router.intra_chip_downstream_edges.size(); ++vc) {
        for (const manifest::DownstreamEdge& edge : router.intra_chip_downstream_edges[vc]) {
            const StreamRegisterFacts edge_facts{.depth = find_landing_channel_depth(facts, ref, edge, vc)};
            add_field_stream_registers(edge.fields_by_erisc, [&](size_t) { return edge_facts; }, out);
        }
    }
    add_field_stream_registers(router.fields_by_erisc, [](size_t) { return StreamRegisterFacts{}; }, out);
    return out;
}

StreamCountReads collect_stream_count_reads(const RouterValues& values) {
    StreamCountReads out;
    manifest::for_each_router_region(
        *values.get_snapshot().get_router().router, [&](std::string_view path, const auto& region) {
            if constexpr (std::is_same_v<std::decay_t<decltype(region)>, manifest::content::StreamReg>) {
                if (!std::holds_alternative<StreamRole>(region.role) ||
                    region.type != manifest::manifest_type_name<uint32_t>) {
                    return;
                }
                if (const RegionRead<uint32_t>* read = values.find_read<uint32_t>(path, std::nullopt)) {
                    out.emplace(
                        &region,
                        StreamCountRead{.path = std::string(path), .count = read->elements.at(0), .t_ns = read->t_ns});
                }
            }
        });
    return out;
}

std::optional<UncreditedAtPeer> find_uncredited_at_peer(
    const RouterCountReads& consumer, size_t vc, size_t channel, const RouterCountReads& producer) {
    const manifest::SenderChannel& sender = consumer.router.channels.senders.at(vc).at(channel);
    const std::optional<uint32_t> depth = find_channel_depth(sender.ring_buffer);
    const manifest::DownstreamEdge* edge =
        find_edge_into_channel(producer.router, vc, consumer.router.identity.key.direction, channel);
    if (!depth.has_value() || edge == nullptr) {
        return std::nullopt;
    }
    const StreamCountRead* edge_read = find_field_count_read(edge->fields_by_erisc, "free_slots", producer.reads);
    const StreamCountRead* channel_read = find_field_count_read(sender.fields_by_erisc, "free_slots", consumer.reads);
    if (edge_read == nullptr || channel_read == nullptr) {
        return std::nullopt;
    }
    const std::optional<uint32_t> producer_uncredited = find_slots_not_counted(edge_read->count, depth);
    const std::optional<uint32_t> unsent = find_slots_not_counted(channel_read->count, depth);
    if (!producer_uncredited.has_value() || !unsent.has_value()) {
        return std::nullopt;
    }
    return UncreditedAtPeer{
        .value = static_cast<int32_t>(*producer_uncredited) - static_cast<int32_t>(*unsent),
        .producer_edge = fmt::format("{}/{}", producer.path, edge_read->path),
        .producer_uncredited = *producer_uncredited,
        .unsent = *unsent,
        .read_apart_ns = edge_read->t_ns > channel_read->t_ns ? edge_read->t_ns - channel_read->t_ns
                                                              : channel_read->t_ns - edge_read->t_ns,
    };
}

int32_t sign_extend_stream_count(uint32_t count) {
    constexpr int width = std::bit_width(stream_count_mask);
    const uint32_t value = count & stream_count_mask;
    const uint32_t sign_bit = uint32_t{1} << (width - 1);
    return (value & sign_bit) != 0 ? static_cast<int32_t>(value) - static_cast<int32_t>(stream_count_mask + 1)
                                   : static_cast<int32_t>(value);
}

bool is_stream_count_in_range(uint32_t count, std::optional<uint32_t> depth) {
    return !depth.has_value() || count <= *depth;
}

std::optional<uint32_t> find_slots_not_counted(uint32_t count, std::optional<uint32_t> depth) {
    if (!depth.has_value() || count > *depth) {
        return std::nullopt;
    }
    return *depth - count;
}

EriscSyncState read_erisc_sync(uint32_t value) {
    if (value == static_cast<uint32_t>(EriscSyncStep::IDLE)) {
        return {.step = EriscSyncStep::IDLE};
    }
    for (const EriscSyncPhase phase : enchantum::values<EriscSyncPhase>) {
        for (const EriscSyncStep step : {EriscSyncStep::START, EriscSyncStep::ANSWER}) {
            if (value == make_erisc_sync_value(step, phase)) {
                return {.phase = phase, .step = step};
            }
        }
    }
    return {};
}

json make_stream_count_json(uint32_t count, StreamRole role, const StreamRegisterFacts& register_facts) {
    const std::optional<uint32_t> depth = register_facts.depth;
    const bool in_range = is_stream_count_in_range(count, depth);
    json out;
    out["count"] = count;
    if (!in_range) {
        out["signed"] = sign_extend_stream_count(count);
    }
    out["depth"] = depth.has_value() ? json(*depth) : json(nullptr);
    const auto optional_json = [](std::optional<uint32_t> value) {
        return value.has_value() ? json(*value) : json(nullptr);
    };
    switch (role) {
        case StreamRole::SENDER_FREE_SLOTS:
        case StreamRole::VC2_SENDER_FREE_SLOTS:
            out["unsent"] = optional_json(find_slots_not_counted(count, depth));
            break;
        case StreamRole::DOWNSTREAM_FREE_SLOTS:
            out["uncredited"] = optional_json(find_slots_not_counted(count, depth));
            break;
        case StreamRole::RECEIVER_PKTS_SENT:
            if (register_facts.first_level_ack.has_value()) {
                out["taken_at"] = *register_facts.first_level_ack ? "first_level_ack" : "retire";
            } else {
                out["taken_at"] = nullptr;
            }
            break;
        case StreamRole::SENDER_PKTS_ACKED:
        case StreamRole::SENDER_PKTS_COMPLETED:
        case StreamRole::VC2_RECEIVER_FREE_SLOTS:
        case StreamRole::TENSIX_RELAY_FREE_SLOTS: break;
    }
    return out;
}

json make_erisc_sync_json(uint32_t value) {
    const EriscSyncState state = read_erisc_sync(value);
    json out;
    out["value"] = value;
    out["phase"] = state.phase.has_value() ? detail::make_enum_json(*state.phase) : json(nullptr);
    const std::optional<std::string> step_name =
        state.step.has_value() ? find_enum_name(*state.step) : std::optional<std::string>();
    out["step"] = step_name.has_value() ? json(*step_name) : json(nullptr);
    return out;
}

json make_uncredited_at_peer_json(const std::optional<UncreditedAtPeer>& uncredited_at_peer) {
    if (!uncredited_at_peer.has_value()) {
        return json(nullptr);
    }
    json out;
    out["value"] = uncredited_at_peer->value;
    out["producer_edge"] = uncredited_at_peer->producer_edge;
    out["producer_uncredited"] = uncredited_at_peer->producer_uncredited;
    out["unsent"] = uncredited_at_peer->unsent;
    out["read_apart_ns"] = uncredited_at_peer->read_apart_ns;
    return out;
}

}  // namespace tt::tt_fabric::decode
