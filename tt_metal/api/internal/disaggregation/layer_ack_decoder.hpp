// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Decodes which global layer a D2H ack record is for. Records carry only the chunk identity, so the layer is the
// record's position in the rank's per-chunk record sequence (a hybrid stack acks only on KV-writing layers); an
// identity change mid-sequence means a record was lost, which would mislabel everything after it.

#pragma once

#include <cstdint>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

namespace tt::tt_metal::internal {

struct LayerAckLayout {
    uint32_t first_record = 0;        // index of this rank's first record in a chunk's record sequence
    uint32_t record_count = 0;        // records this rank emits per chunk
    std::vector<uint32_t> layer_ids;  // global layer of each local record; empty = record index is the layer
};

struct AckIdentity {
    uint32_t slot_id = 0;
    uint32_t pos_start = 0;
    uint32_t pos_end = 0;
    bool operator==(const AckIdentity&) const = default;
};
static_assert(sizeof(AckIdentity) == 3 * sizeof(uint32_t));

struct DecodedAck {
    uint32_t layer = 0;  // global layer
    AckIdentity identity{};
};

struct LostAckRecord {
    uint64_t record = 0;     // k
    uint32_t slice_idx = 0;  // where in the chunk the new identity appeared (never 0)
    AckIdentity identity{};
};

class LayerAckDecoder {
public:
    explicit LayerAckDecoder(LayerAckLayout layout) : layout_(std::move(layout)) {}

    // Every record of one chunk carries the same identity, so a change must land on slice 0.
    std::variant<DecodedAck, LostAckRecord> decode(const AckIdentity& identity) {
        const uint64_t k = count_++;
        const uint32_t slice_idx = static_cast<uint32_t>(k % layout_.record_count);
        if (prev_ && identity != *prev_ && slice_idx != 0) {
            prev_ = identity;
            return LostAckRecord{k, slice_idx, identity};
        }
        prev_ = identity;
        DecodedAck ack;
        ack.layer = layout_.layer_ids.empty() ? layout_.first_record + slice_idx : layout_.layer_ids[slice_idx];
        ack.identity = identity;
        return ack;
    }

private:
    LayerAckLayout layout_;
    uint64_t count_ = 0;
    std::optional<AckIdentity> prev_;
};

}  // namespace tt::tt_metal::internal
