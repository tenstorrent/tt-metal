// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Stream registers: what the value the router reads from each means, by the register's role (the manifest's `role`),
// against the facts about the channel the role names. Values only: what a count implies about flow waits for what the
// router doesn't publish yet.

#pragma once

#include <cstdint>
#include <map>
#include <optional>

#include <hostdevcommon/fabric_common.h>
#include <nlohmann/json.hpp>

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

// -- Shown.

// A count's read, for a register whose role is `role`: `count`, its `depth`, and by role `unsent`
// (sender_free_slots), `taken_at` (receiver_pkts_sent) or `uncredited` (downstream_free_slots). Out of range, also
// `signed`, and the derived value null.
nlohmann::ordered_json make_stream_count_json(
    uint32_t count, StreamRole role, const StreamRegisterFacts& register_facts);

// The ERISC sync's read: its `value`, and the `phase` and `step` it says (read_erisc_sync).
nlohmann::ordered_json make_erisc_sync_json(uint32_t value);

}  // namespace tt::tt_fabric::decode
