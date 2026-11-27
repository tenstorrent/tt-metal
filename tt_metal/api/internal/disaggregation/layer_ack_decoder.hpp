// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Decodes which global layer a D2H ack record is for. Records carry only the chunk identity, so the layer is the
// record's position in that chunk's record sequence on the rank (a hybrid stack acks only on KV-writing layers).
// Up to open_chunks chunks interleave: a packed forward acks each of its chunks on a layer, in pack order, before the
// next layer. A lost record would mislabel every later one, so the order is checked:
//   - a chunk starts only while fewer than open_chunks are in progress, each with exactly one record (a rank's first
//     acking layer sends one per chunk);
//   - no chunk gets more records than one that started before it.
// A lost record shows once a later chunk overtakes its chunk, or starts while its chunk has more than one record;
// until then its chunk's records name the layer before the one that finished (a late completion, never an early
// one). Not caught: losing the only record of a chunk, or a chunk identity restarting while its earlier run is short.

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

// With open_chunks > 1: a record naming no chunk (pos_end == pos_start), from a traced packed forward's empty position.
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
