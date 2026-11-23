// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Stream registers: what the value the router reads from each means, by the register's role (the manifest's `role`),
// against the facts about the channel the role names. Values only: what a count implies about flow waits for what the
// router doesn't publish yet.

#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>

#include <hostdevcommon/fabric_common.h>
#include <nlohmann/json.hpp>

#include "tt_metal/fabric/debug/visualizer/decode/fabric_decode_values.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_model.hpp"

namespace tt::tt_fabric::decode {

// What a stream register's value is read against, beyond the register itself: the manifest's facts about the channel
// its role names.
struct StreamRegisterFacts {
    // The slots of the channel a count is about: a sender channel's own (sender_free_slots), a receiver channel's
    // (receiver_pkts_sent), or the sibling's sender channel an edge lands on (downstream_free_slots). Null for a
    // scratch register, and where the manifest doesn't give it: an edge landing on the tensix mux, or a channel with
    // no slots.
    std::optional<uint32_t> depth;
    // A receiver's packets-sent count: whether the ERISC that names it takes packets at first-level ack, as its
    // router's sender channels on the receiver's VC are credited (credits.acked). Null for every other role, and when
    // the router has no sender channel on that VC to say.
    std::optional<bool> first_level_ack;
};

// Every stream register a router names, keyed by its content in the manifest's model: the reference
// for_each_router_region hands over. A stream register named by several fields comes once per field.
using StreamRegisterFactsByRegister = std::map<const manifest::content::StreamReg*, StreamRegisterFacts>;

// The facts for every stream register `router` names. The router is `ref` in `facts`, from which an edge's sibling is
// found on the router's chip and routing plane.
StreamRegisterFactsByRegister collect_stream_register_facts(
    const manifest::ManifestFacts& facts, const manifest::RouterRef& ref, const manifest::Router& router);

// Whether ERISC `erisc` of `router` takes VC `vc`'s packets at first-level ack: the builder feeds each ERISC
// ENABLE_FIRST_LEVEL_ACK_VC<v>, which the manifest records as the acked credits of the router's sender channels on
// that VC. Null when the router has no sender channel there.
std::optional<bool> find_first_level_ack(const manifest::Router& router, size_t vc, size_t erisc);

// -- Counts (every StreamRole).

// A count as the router reads it, sign-extended from stream_count_mask's width: a count taken below zero wraps
// within the hardware's count and reads as a large number under the mask.
int32_t sign_extend_stream_count(uint32_t count);

// Whether `count` can be the count of a channel of `depth` slots: at most depth. True when the depth isn't known.
bool is_stream_count_in_range(uint32_t count, std::optional<uint32_t> depth);

// The channel's slots the count doesn't hold, depth - count: a sender channel's packets not sent yet, or an edge's
// slots in the sibling not credited back yet. Null when the depth isn't known or the count is out of range.
std::optional<uint32_t> find_slots_not_counted(uint32_t count, std::optional<uint32_t> depth);

// -- The ERISC sync (MULTI_RISC_TEARDOWN_SYNC).

// What the ERISC sync's value says. Both are null for a value the router never writes.
struct EriscSyncState {
    std::optional<EriscSyncPhase> phase;
    std::optional<EriscSyncStep> step;
};

// The step and phase the ERISC sync's value says, as the router writes it (sync_erisc, make_erisc_sync_value).
EriscSyncState read_erisc_sync(uint32_t value);

// -- One value from two routers' registers.

// A count register's full read, kept past its router's decode for analysis that needs another router's registers.
struct StreamCountRead {
    // The register's path below its router's node (for_each_router_region's).
    std::string path;
    // The count the router reads, under the register's read mask.
    uint32_t count = 0;
    // When capture read it, on capture's clock.
    uint64_t t_ns = 0;
};

// Each active count register's full read, keyed by its content in the manifest's model, as StreamRegisterFacts are.
using StreamCountReads = std::map<const manifest::content::StreamReg*, StreamCountRead>;

// The full read of every active count register (any StreamRole) `values` has one of.
StreamCountReads collect_stream_count_reads(const RouterValues& values);

// The full read of the count register the field `key` names, on the first ERISC of `fields_by_erisc` whose field has
// one in `reads`. Null when none does.
const StreamCountRead* find_field_count_read(
    const manifest::FieldsByErisc& fields_by_erisc, std::string_view key, const StreamCountReads& reads);

// A router with its count registers' full reads.
struct RouterCountReads {
    const manifest::Router& router;
    // Such as "M0/C7/E0".
    std::string_view path;
    const StreamCountReads& reads;
};

// A sibling-fed sender channel's packets sent over Ethernet whose credits haven't come back to the producer yet: in
// the peer's receiver, or their credits still on the wire. The producer's uncredited slots in the channel, from its
// edge into it, less the channel's packets not sent yet, from its own free slots. Capture reads the two registers at
// different times, so on routers that are moving it can be off, even negative.
struct UncreditedAtPeer {
    int32_t value = 0;
    // The producer's edge register: its router's path, then the register's path below the router.
    std::string producer_edge;
    uint32_t producer_uncredited = 0;
    uint32_t unsent = 0;
    // How far apart capture read the two registers.
    uint64_t read_apart_ns = 0;
};

// uncredited_at_peer for sender channel `channel` on VC `vc` of `consumer`, whose producer is the sibling `producer`.
// Null when `producer` has no edge on the VC landing on the channel (or it lands on the tensix mux), the channel has
// no slots, or either register has no full read with a count in range.
std::optional<UncreditedAtPeer> find_uncredited_at_peer(
    const RouterCountReads& consumer, size_t vc, size_t channel, const RouterCountReads& producer);

// -- Shown.

// A count's read, for a register whose role is `role`: `count`, its `depth`, and by role `unsent`
// (sender_free_slots), `taken_at` (receiver_pkts_sent) or `uncredited` (downstream_free_slots). Out of range, also
// `signed`, and the derived value null.
nlohmann::ordered_json make_stream_count_json(
    uint32_t count, StreamRole role, const StreamRegisterFacts& register_facts);

// The ERISC sync's read: its `value`, and the `phase` and `step` it says (read_erisc_sync).
nlohmann::ordered_json make_erisc_sync_json(uint32_t value);

// A sender channel's analysis.uncredited_at_peer: null, or its value and what it's made from.
nlohmann::ordered_json make_uncredited_at_peer_json(const std::optional<UncreditedAtPeer>& uncredited_at_peer);

}  // namespace tt::tt_fabric::decode
