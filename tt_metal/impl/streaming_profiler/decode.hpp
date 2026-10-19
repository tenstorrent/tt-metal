// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <memory>
#include <memory_resource>
#include <optional>
#include <span>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include <tt-metalium/experimental/streaming_profiler.hpp>

#include "hostdev/record_site.h"
#include "impl/streaming_profiler/spsc_marker_decode.hpp"
#include "impl/streaming_profiler/capture_context.hpp"
#include "impl/streaming_profiler/sync/clock_map.hpp"

namespace tt::llrt {
struct CallSite;
}

namespace tt::tt_metal::streaming_profiler {

template <typename T>
T load_unaligned(const std::byte* at) {
    T value;
    std::memcpy(&value, at, sizeof(value));
    return value;
}

// The `width` bits from bit `bit` of `bytes`, sign-extended for a signed field.
inline uint64_t field_value(const std::byte* bytes, uint32_t bit, uint32_t width, uint16_t code) {
    const std::byte* const at = bytes + bit / 8;
    const uint32_t shift = bit % 8;
    uint64_t value = 0;
    if (shift == 0 && width == 8 * experimental::streaming_profiler::detail::code_bytes(code)) {
        // A field in whole bytes of its own size, as most are, loads just those bytes.
        switch (width) {
            case 8: value = std::to_integer<uint8_t>(*at); break;
            case 16: value = load_unaligned<uint16_t>(at); break;
            case 32: value = load_unaligned<uint32_t>(at); break;
            default: value = load_unaligned<uint64_t>(at); break;
        }
    } else {
        std::array<std::byte, 16> window{};
        std::memcpy(window.data(), at, (shift + width + 7) / 8);
        uint64_t low = 0, high = 0;
        std::memcpy(&low, window.data(), sizeof(low));
        std::memcpy(&high, window.data() + sizeof(low), sizeof(high));
        value = shift == 0 ? low : low >> shift | high << (64 - shift);
    }
    if (width < 64) {
        const uint64_t mask = (uint64_t{1} << width) - 1;
        const bool negative = (code >> 8) == 'i' && (value >> (width - 1) & 1) != 0;
        value = negative ? value | ~mask : value & mask;
    }
    return value;
}

// A wall-clock read that tore comes out this many ticks high.
inline constexpr uint64_t kEpoch = uint64_t{1} << 32;

// How a form's payload fills a record the decoder writes directly: each data field from its bits in the payload to its
// bytes in the record.
struct DirectForm {
    enum class Op : uint8_t {
        Copy,       // the field's bits
        Distance,   // a Timestamp sent as its distance back from the record's time
        Time,       // a Timestamp sent as the time itself, in data whose Timestamps don't travel as distances
        Published,  // the Timestamp that is the record's own time, which no form sends
    };
    struct Field {
        Op op;
        uint16_t code;
        uint32_t to;          // byte offset in the record
        uint32_t bit, width;  // in the payload
    };
    std::vector<Field> fields;
    std::vector<uint32_t> follows_time;  // byte offsets of the Timestamps computed from the record's time
};
// The core read the Timestamp before the record, and a read that tore came out 2^32 ticks high, which puts it after the
// record by up to 2^32 ticks.
inline uint64_t timestamp_ticks(DirectForm::Op op, uint64_t time, uint64_t sent) {
    const uint64_t ticks = op == DirectForm::Op::Published ? time : op == DirectForm::Op::Distance ? time - sent : sent;
    return ticks > time && ticks - time <= kEpoch ? ticks - kEpoch : ticks;
}
// A call site's forms for its direct types, indexed by TimeForm. Those types all hold the site's data, so they share
// one layout.
using DirectForms = std::array<DirectForm, 3>;

inline bool has_type(uint64_t types, uint32_t type) { return (types >> type & 1) != 0; }

struct SpscLane {
    static constexpr uint64_t kNoCursor = ~uint64_t{0};
    // The time of the lane's last short or medium record from a call site with a short form, which a short record's
    // time counts from. kNoCursor until the first medium record after a launch, a rewind or a gap.
    uint64_t cursor = kNoCursor;
    // The lane's latest timestamp, before any repair. Lanes emit in time order, so a step back is a torn read.
    uint64_t last_ts = 0;
    // The time slot of the lane's latest record in the output. A later record with an earlier time means this record's
    // wall-clock read was torn and came out 2^32 ticks too high, so the decoder repairs it in place. The pointer is
    // valid only until the batch numbered last_time_seq is delivered.
    uint8_t* last_time = nullptr;
    uint64_t last_time_seq = 0;
    const DirectForm* last_form = nullptr;  // the last record's form, when the decoder wrote it directly
    // Where the time of a record no type takes goes, so that a repair of it has somewhere to write. Nothing reads it.
    uint64_t unrouted_time = 0;
    bool seeded = false;
    bool need_state = true;
    profiler::SpscLaneConsts consts{};
};

struct PublishedRecord {
    experimental::streaming_profiler::detail::Source source;
    uint64_t ticks = 0, clock_offset = 0;
    uint32_t byte_count = 0, length = 0;
    const std::byte* bytes = nullptr;
};
static_assert(sizeof(PublishedRecord) == profiler::kSpscPublishedBytes);
static_assert(std::is_trivially_copyable_v<PublishedRecord>);
// The decoder writes each record's head, its source and time, where Record lays them out.
static_assert(sizeof(experimental::streaming_profiler::detail::Source) == 16);
static_assert(sizeof(experimental::streaming_profiler::Record<void, void>) == profiler::kSpscHeadBytes);

class RecordRoutes {
public:
    static constexpr uint32_t kMaxTypes = experimental::streaming_profiler::detail::kMaxRecordTypes;
    static_assert(kMaxTypes <= 64, "a type mask is a uint64_t");
    struct Route {
        uint64_t types = 0;                         // a bit per type of the consumer's the call site sends
        uint8_t short_length = 0, full_length = 0;  // its short and full forms' body lengths, 0 for a form it lacks
        bool resolved = false;                      // set when set_route() takes it
        const DirectForms* forms = nullptr;         // for its direct types, null when its payload is their data as is
    };
    // A direct type's short-record layout, which all its call sites share, and the first site's short form, whose
    // Timestamps that follow the record's time a repair moves for every site.
    struct ShortRun {
        profiler::SpscShortLayout layout;
        const DirectForm* form;
    };
    explicit RecordRoutes(std::vector<experimental::streaming_profiler::detail::Kind> kinds);

    Route resolve(uint32_t id);
    // Sets call site `id`'s route, as resolve() does.
    void set_route(uint32_t id, Route route) {
        route.resolved = true;
        routes_[id] = route;
        short_run_types_[id] = short_run_type(route);
    }
    size_t type_count() const { return kinds_.size(); }
    const experimental::streaming_profiler::detail::Kind& kind(uint32_t type) const { return kinds_[type]; }
    // Whether a type's records are PublishedRecords until its batch builds them.
    bool built(uint32_t type) const { return !has_type(direct_types_, type); }
    std::span<const uint32_t> time_offsets(uint32_t type) const { return time_offsets_[type]; }
    uint32_t stride(uint32_t type) const { return strides_[type]; }
    // For each value call site `id` passes, which of built type `type`'s alternatives it is.
    std::span<const uint8_t> alternatives(uint32_t id, uint32_t type) const;

private:
    friend class StreamDecoder;
    const DirectForms* direct_forms(const tt::llrt::CallSite& site, std::span<const uint32_t> field_offsets);
    // The bit of the type whose short records spsc_short_records() writes in runs for a route's call site, or 0. A site
    // takes the run kernel when it sends one direct type and its short records are 2 words.
    uint64_t short_run_type(const Route& route);

    std::vector<experimental::streaming_profiler::detail::Kind> kinds_;
    std::vector<Route> routes_;
    std::vector<uint64_t> short_run_types_;
    std::array<uint32_t, kMaxTypes> strides_{};
    // The types the decoder writes directly, and each one's data fields' and Timestamps' byte offsets in a record.
    uint64_t direct_types_ = 0;
    std::array<std::vector<uint32_t>, kMaxTypes> field_offsets_;
    std::array<std::vector<uint32_t>, kMaxTypes> time_offsets_;
    std::deque<DirectForms> forms_;
    std::array<std::optional<ShortRun>, kMaxTypes> short_runs_;
    std::unordered_map<uint64_t, std::vector<uint8_t>> alternatives_;
};

// The time slots of the other types' copies of a lane's last record, which repair_last() moves with it while batch
// `seq` is undelivered.
struct RecordCopies {
    const uint8_t* copied_time = nullptr;
    uint64_t seq = 0;
    uint32_t count = 0;
    uint32_t direct_count = 0;  // of the copies, the first direct_count are of direct types
    std::array<uint8_t*, RecordRoutes::kMaxTypes> time_slots{};
};

class StreamDecoder {
public:
    struct Out {
        uint64_t* values;
        std::span<uint8_t* const> types;
        // For a consumer of more than one type, each record's type, once per type it goes to, in the order the lanes
        // emitted them.
        uint32_t* order;
    };
    struct Produced {
        uint32_t values;
        std::array<uint32_t, RecordRoutes::kMaxTypes> type_bytes;
        uint32_t order_entries;
        // The latest timestamp among the last records of the lanes this call advanced, or INT64_MIN if none of them has
        // a record yet. It counts wall ticks of the chip's wall-clock core, whose clock the clock sync maps to host
        // time.
        int64_t newest_ticks;
        uint64_t stalls;
        uint64_t order_regressions;
    };

    // `dev` and `routes` must outlive the decoder.
    StreamDecoder(const CaptureContext::Device& dev, RecordRoutes& routes);
    // Decodes the frames into `out` and returns what it produced. ConsumerBatches sizes `out`. `undelivered` is how
    // many of the earlier outputs the caller still holds, which it delivers oldest first. A call may repair a lane's
    // last record in place in any of those, so they must stay writable until delivered.
    Produced decode_frames(size_t undelivered, const uint32_t* frames, std::span<const uint32_t> frame_words, Out out);

private:
    struct Written {
        uint8_t* time_slot;
        uint32_t value_count;
    };
    struct Packet {
        const uint32_t* src;
        uint32_t payload_at;
        uint64_t time;
        const DirectForm* form;
    };
    // Aligned to 64 bytes, since where the loop falls across cache lines moves its time per record by 3%.
    template <bool kWritesOrder>
    [[gnu::aligned(64)]] Produced decode(
        size_t undelivered, const uint32_t* frames, std::span<const uint32_t> frame_words, Out out);
    // repair_last() and write_record_routed() take the lane rather than its index, which the decode loop would then
    // keep in a register.
    bool repair_last(const SpscLane& lane, uint64_t next_ts);
    Written write_record_to(
        uint32_t type, const Packet& packet, const profiler::SpscLaneConsts& consts, uint64_t* values);
    Written write_record_routed(
        uint64_t types, const Packet& packet, const profiler::SpscLaneConsts& consts, uint64_t* values, SpscLane& lane);

    std::vector<SpscLane> lanes_;
    std::vector<RecordCopies> copies_;
    // Each core's ring heads, one per RISC. A head is where the lane's previous frame ended. decode_frames reads a
    // core's heads with one 256-bit load.
    static constexpr size_t kHeadsPerLoad = 8;
    static_assert(profiler::kSpscNRiscDecode <= kHeadsPerLoad);
    std::vector<std::array<uint32_t, kHeadsPerLoad>> heads_;
    const CaptureContext::Device& dev_;
    uint64_t decoded_seq_ = 0, delivered_seq_ = 0;
    RecordRoutes& routes_;
    // Copies of the routes' tables, which the decode loop reads as members: as its locals they spill.
    const RecordRoutes::Route* site_routes_;
    const uint64_t* short_run_types_;
    uint64_t direct_types_;
    std::array<uint32_t, RecordRoutes::kMaxTypes> strides_;
    std::array<uint8_t*, RecordRoutes::kMaxTypes> types_out_{};
    std::array<uint32_t, RecordRoutes::kMaxTypes> type_bytes_{};
    bool writes_order_;
};

// Builds the records of built type `type` in `arena`.
const std::byte* build_records(
    std::span<const PublishedRecord> published,
    uint32_t type,
    const RecordRoutes& routes,
    std::pmr::memory_resource& arena,
    const ClockMap& map,
    ClockMap::Reader& reader,
    uint32_t dev);

struct DecodedFrames {
    std::unique_ptr<uint8_t[]> buffer;
    StreamDecoder::Produced produced;
};

class ConsumerBatches {
public:
    explicit ConsumerBatches(std::vector<experimental::streaming_profiler::detail::Kind> kinds);

    RecordRoutes& routes() { return routes_; }
    // Decodes the frames with `decoder` into a buffer, reusing the most recently released one, which is still in cache
    // and already faulted in.
    DecodedFrames decode(
        StreamDecoder& decoder, size_t undelivered, const uint32_t* frames, std::span<const uint32_t> frame_words);
    // Builds the batch of `frames`, whose records it places on host time in place, so it runs once per frames. The
    // batch is good until the next call.
    const experimental::streaming_profiler::detail::BatchData& build_batch(
        DecodedFrames& frames, const ClockMap& map, ClockMap::Reader& reader, uint32_t dev, uint64_t dropped_bytes);
    void release(DecodedFrames&& frames) { free_buffers_.push_back(std::move(frames.buffer)); }

private:
    // The output's type regions' addresses are kept in type_out_, so it is good until the next call.
    StreamDecoder::Out out_in(uint8_t* buffer);
    // Puts the batch's records on host time through the clock map. The decoder leaves each record's clock offset in its
    // tsc_ slot, and this overwrites it with the host time. Built records are placed as they are built.
    void place(
        const StreamDecoder::Out& out,
        const StreamDecoder::Produced& produced,
        const ClockMap& map,
        ClockMap::Reader& reader,
        uint32_t dev);

    RecordRoutes routes_;
    std::array<uint8_t*, RecordRoutes::kMaxTypes> type_out_{};
    std::vector<std::unique_ptr<uint8_t[]>> free_buffers_;
    experimental::streaming_profiler::detail::BatchData batch_;
    std::vector<const std::byte*> records_;
    std::vector<std::size_t> counts_;
    std::unique_ptr<std::byte[]> arena_buf_;
    std::pmr::monotonic_buffer_resource arena_;
};

}  // namespace tt::tt_metal::streaming_profiler
