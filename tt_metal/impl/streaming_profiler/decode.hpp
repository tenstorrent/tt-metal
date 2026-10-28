// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>
#include <span>
#include <type_traits>
#include <vector>

#include <tt-metalium/experimental/streaming_profiler.hpp>

#include "impl/streaming_profiler/spsc_marker_decode.hpp"
#include "impl/streaming_profiler/capture_context.hpp"

namespace tt::tt_metal::streaming_profiler {

static_assert(sizeof(experimental::streaming_profiler::Zone) == profiler::kSpscZoneBytes);
static_assert(sizeof(experimental::streaming_profiler::Event) == profiler::kSpscEventBytes);
static_assert(sizeof(experimental::streaming_profiler::TimestampedData) == profiler::kSpscDataBytes);
static_assert(std::is_trivially_copyable_v<experimental::streaming_profiler::Zone>);
static_assert(std::is_trivially_copyable_v<experimental::streaming_profiler::Event>);

struct SpscLane {
    uint64_t cursor = 0;   // the last absolute zone end, which a delta16 zone's end delta counts from
    // The lane's latest timestamp, before any repair. Lanes emit in end order, so a step back is a torn read.
    uint64_t last_ts = 0;
    // The lane's latest record in the output. A later record with an earlier time means this record's wall-clock read
    // was torn and came out 2^32 ticks too high, so the decoder repairs it in place. The pointer is valid only until
    // the batch numbered last_rec_seq is delivered.
    uint8_t* last_rec = nullptr;
    uint64_t last_rec_seq = 0;
    // Whether last_rec is a zone. A repair may fix a zone's duration in word 4, where data keeps its value count.
    bool last_rec_zone = false;
    bool seeded = false;
    bool need_state = true;
    // Set until the lane's next ZONE_ATOMIC, whose absolute end the following ZONE_S zones count from. The producer
    // writes one as the first zone after each launch or rewind, and after a gap the decoder waits for the next one.
    bool need_anchor = true;
    profiler::SpscLaneConsts consts{};
};

class StreamDecoder {
public:
    struct Out {
        uint8_t* zones;
        uint8_t* events;
        uint8_t* data;
        uint64_t* values;
    };
    struct Produced {
        uint32_t zones, events, data, values;
        // The latest timestamp among the last records of the lanes this call advanced, or INT64_MIN if none of them has
        // a record yet. It counts wall ticks of the chip's wall-clock core, whose clock the clock sync maps to host
        // time.
        int64_t newest_ticks;
        uint64_t stalls;
        uint64_t order_regressions;
    };
    struct Capacity {
        size_t zone_bytes, event_bytes, data_bytes, value_bytes;
    };
    static_assert(std::ranges::all_of(profiler::kFormats, [](const profiler::PacketFormat& format) {
        return format.kind == profiler::Kind::Sticky || format.words >= 2;
    }));
    // Returns the output a call needs room for, given `words` payload words in `frames` frames. A packet that becomes a
    // record is at least two words, so each record kind gets room for words / 2 records, plus kSpscSinkSlackRecs for
    // the extra records the vector decoders write when they store a last group of four. Each value comes from two
    // payload words, plus 32 bytes per frame for what spsc_point() stores past its last value.
    static constexpr Capacity out_capacity(size_t words, uint32_t frames) {
        const size_t records = words / 2 + profiler::kSpscSinkSlackRecs;
        return Capacity{
            .zone_bytes = records * profiler::kSpscZoneBytes,
            .event_bytes = records * profiler::kSpscEventBytes,
            .data_bytes = records * profiler::kSpscDataBytes,
            .value_bytes = words * 4 + size_t{32} * frames};
    }

    // `dev` must outlive the decoder. It produces only records of `types`, and steps over the other packets.
    StreamDecoder(const CaptureContext::Device& dev, uint32_t types);
    // Decodes the frames into `out` and returns what it produced. Size `out` with out_capacity(). `undelivered` is how
    // many of the earlier outputs the caller still holds, which it delivers oldest first. A call may repair a lane's
    // last record in place in any of those, so they must stay writable until delivered.
    Produced decode_frames(size_t undelivered, const uint32_t* frames, std::span<const uint32_t> frame_words, Out out);

private:
    std::vector<SpscLane> lanes_;
    // Each core's ring heads, one per RISC. A head is where the lane's previous frame ended. decode_frames reads a
    // core's heads with one 256-bit load.
    static constexpr size_t kHeadsPerLoad = 8;
    static_assert(profiler::kSpscNRiscDecode <= kHeadsPerLoad);
    std::vector<std::array<uint32_t, kHeadsPerLoad>> heads_;
    const CaptureContext::Device& dev_;
    // One bit per wire type the decoder steps over instead of producing.
    uint32_t skipped_types_ = 0;
    uint64_t decoded_seq_ = 0;
};

// Starts a TimestampedData's lifetime in the decoded bytes at `record`, keeping their values. TimestampedData has a
// destructor, so the memmove that creates zones and events in place can't create it. The record borrows its values from
// the consumer's arena, which reclaims both without running ~TimestampedData.
inline void construct_timestamped_data(uint8_t* record) {
    std::array<std::byte, sizeof(experimental::streaming_profiler::TimestampedData)> bytes;
    std::memcpy(bytes.data(), record, bytes.size());
    new (record) experimental::streaming_profiler::TimestampedData;
    std::memcpy(record, bytes.data(), bytes.size());
}

}  // namespace tt::tt_metal::streaming_profiler
