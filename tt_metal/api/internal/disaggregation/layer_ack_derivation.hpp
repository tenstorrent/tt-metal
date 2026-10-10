// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Labels a rank's k-th D2H ack record. Records arrive in emission order with no
// layer index of their own, so chunk and layer come from a counter in ACK space (a
// hybrid stack acks only on KV-writing layers); the record's identity words are used
// to catch a lost record, which would otherwise mislabel everything after it.

#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <variant>
#include <vector>

namespace tt::tt_metal::internal {

struct AckSpace {
    uint32_t num_ack_layers = 0;      // records one chunk produces across all ranks (seq stride)
    uint32_t ack_first_idx = 0;       // this rank's first ACK index
    uint32_t ack_local_count = 0;     // records this rank emits per chunk
    std::vector<uint32_t> layer_ids;  // global layer of each local record; empty = ack index is the layer
};

using AckIdentity = std::array<uint32_t, 3>;  // {slot_id, pos_start, pos_end}

struct AckLabel {
    uint64_t seq = 0;      // chunk * num_ack_layers + ack_idx: dense across ranks
    uint32_t chunk = 0;    // this rank's chunk ordinal
    uint32_t ack_idx = 0;  // ACK-space index
    uint32_t layer = 0;    // global layer
    AckIdentity identity{};
};

struct AckDesync {
    uint64_t record = 0;     // k
    uint32_t slice_idx = 0;  // where in the chunk the new identity appeared (never 0)
    AckIdentity identity{};
};

class AckDerivation {
public:
    explicit AckDerivation(AckSpace space) : space_(std::move(space)) {}

    // Every record of one chunk carries the same identity, so a change must land on slice 0.
    std::variant<AckLabel, AckDesync> step(std::optional<AckIdentity> identity) {
        const uint64_t k = count_++;
        const uint32_t slice_idx = static_cast<uint32_t>(k % space_.ack_local_count);
        if (identity && prev_ && *identity != *prev_ && slice_idx != 0) {
            prev_ = identity;
            return AckDesync{k, slice_idx, *identity};
        }
        if (identity) {
            prev_ = identity;
        }
        const uint32_t ack_idx = space_.ack_first_idx + slice_idx;
        AckLabel label;
        label.chunk = static_cast<uint32_t>(k / space_.ack_local_count);
        label.ack_idx = ack_idx;
        label.seq = static_cast<uint64_t>(label.chunk) * space_.num_ack_layers + ack_idx;
        label.layer = space_.layer_ids.empty() ? ack_idx : space_.layer_ids[slice_idx];
        label.identity = identity.value_or(AckIdentity{});
        return label;
    }

    uint64_t records() const { return count_; }

private:
    AckSpace space_;
    uint64_t count_ = 0;
    std::optional<AckIdentity> prev_;
};

}  // namespace tt::tt_metal::internal
