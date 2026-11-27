// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Decodes which global layer a D2H ack record is for. Records carry only the chunk identity, so the layer is the
// record's position in that chunk's record sequence on the rank (a hybrid stack acks only on KV-writing layers).
// Up to open_chunks chunks may be in progress at once: a packed forward acks every one of its chunks on a layer, in
// pack order, before the next layer, so their sequences interleave. That order is checked, since a lost record would
// mislabel everything after it:
//   - a chunk starts (its first record) only while every chunk in progress has exactly one record, and at most
//     open_chunks are in progress;
//   - no chunk gets more records than one that started before it.

#pragma once

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <utility>
#include <variant>
#include <vector>

namespace tt::tt_metal::internal {

struct LayerAckLayout {
    uint32_t first_record = 0;        // index of this rank's first record in a chunk's record sequence
    uint32_t record_count = 0;        // records this rank emits per chunk
    std::vector<uint32_t> layer_ids;  // global layer of each local record; empty = record index is the layer
    uint32_t open_chunks = 1;         // chunks whose records may interleave (a packed forward's chunk count)
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
    uint32_t slice_idx = 0;  // records of the chunk in progress that this record shows to be short
    AckIdentity identity{};  // this record's chunk
};

// With open_chunks > 1, a record naming no chunk (pos_end == pos_start): an empty position of a traced packed
// forward. Not counted.
struct NoChunkAck {};

using DecodeResult = std::variant<DecodedAck, LostAckRecord, NoChunkAck>;

class LayerAckDecoder {
public:
    explicit LayerAckDecoder(LayerAckLayout layout) : layout_(std::move(layout)) {}

    DecodeResult decode(const AckIdentity& identity) {
        if (layout_.open_chunks > 1 && identity.pos_end == identity.pos_start) {
            return NoChunkAck{};
        }
        const uint64_t k = count_++;
        auto it = std::find_if(
            open_.begin(), open_.end(), [&](const InProgress& chunk) { return chunk.identity == identity; });
        if (it == open_.end()) {
            const auto short_chunk =
                std::find_if(open_.begin(), open_.end(), [](const InProgress& chunk) { return chunk.records != 1; });
            if (short_chunk != open_.end() || open_.size() == layout_.open_chunks) {
                const uint32_t records = short_chunk != open_.end() ? short_chunk->records : open_.front().records;
                open_.clear();
                return LostAckRecord{k, records, identity};
            }
            open_.push_back(InProgress{identity, 0});
            it = std::prev(open_.end());
        }
        const uint32_t slice_idx = it->records++;
        const auto behind =
            std::find_if(open_.begin(), it, [&](const InProgress& older) { return older.records < it->records; });
        if (behind != it) {
            const uint32_t records = behind->records;
            open_.clear();
            return LostAckRecord{k, records, identity};
        }
        if (it->records == layout_.record_count) {
            open_.erase(it);
        }
        DecodedAck ack;
        ack.layer = layout_.layer_ids.empty() ? layout_.first_record + slice_idx : layout_.layer_ids[slice_idx];
        ack.identity = identity;
        return ack;
    }

private:
    struct InProgress {
        AckIdentity identity{};
        uint32_t records = 0;  // decoded so far
    };

    LayerAckLayout layout_;
    uint64_t count_ = 0;
    std::vector<InProgress> open_;  // oldest first
};

}  // namespace tt::tt_metal::internal
