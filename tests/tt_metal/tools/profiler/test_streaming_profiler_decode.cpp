// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Host-only tests of the streaming profiler's record decoder on hand-built wire. They cover the record forms, routing
// records to types, the order across types and the repair of torn wall-clock reads, and check ClockMap::place_host4
// against place_host.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <new>
#include <random>
#include <span>
#include <utility>
#include <vector>

#include "impl/streaming_profiler/decode.hpp"
#include "llrt/call_sites.hpp"

using namespace tt::tt_metal;
namespace api = experimental::streaming_profiler;

constexpr uint64_t kEpoch = 1ull << 32;  // a wall-clock time is its high word * kEpoch + its low word
constexpr uint32_t kNear = 0xfffffff0u;  // a low word just below the wrap
constexpr uint32_t kTileXy = 0x30002;
constexpr uint32_t kEventId = 5;

using kernel_profiler::PacketType;
using kernel_profiler::ppfmt;
uint32_t word(PacketType type, uint32_t value) { return ppfmt::w0(type, value); }
uint32_t sticky(uint32_t hi) { return word(PacketType::TimerHigh, hi); }
uint32_t record(uint32_t packet_words, uint32_t id) { return ppfmt::record_w0(packet_words) | id; }
uint32_t event() { return record(2, kEventId); }
// A short record's body: its time's distance from the lane's cursor, then its 16-bit payload.
uint32_t short_body(uint32_t since_cursor, uint32_t payload) { return since_cursor << 16 | payload; }

void check(bool v, const char* why) {
    if (!v) {
        std::fprintf(stderr, "FAIL: %s\n", why);
        std::exit(1);
    }
}

// A record of the harness's default type, whose payload is the packet's payload as sent.
struct Raw {
    api::detail::RecordOrigin origin;
    api::Timestamp time;
    std::array<std::byte, 16> data;
};
static_assert(sizeof(Raw) == profiler::kSpscPublishedBytes);
constexpr api::detail::RecordTypeDescriptor kRawRecordType{.size = sizeof(Raw)};

// A Seeded lane first gets one frame holding a zero timer sticky, so a test's own frames decode as the wire would after
// a capture's first frame.
enum class Lane { Seeded, Fresh };

streaming_profiler::CaptureContext::Device one_tile_device() {
    streaming_profiler::CaptureContext::Device dev{
        .lanes = std::vector<api::Core>(profiler::kSpscNRiscDecode), .clock_offsets = {0}};
    dev.core_of_xy[kTileXy] = 0;
    return dev;
}

struct Harness {
    // A buffer for the built types' values and one per record type, which the frames of one batch append to.
    std::vector<uint64_t> value_buffer = std::vector<uint64_t>(64 * 1024);
    std::array<std::vector<uint64_t>, 2> type_buffers{value_buffer, value_buffer};
    std::vector<uint32_t> order_buffer = std::vector<uint32_t>(1024);
    std::array<size_t, 2> type_bytes{};
    size_t value_count = 0;
    uint32_t order_entries = 0;
    uint64_t order_regressions = 0, stalls = 0;
    int64_t newest_ticks = 0;
    uint32_t parked = 0;  // batches decoded and not yet delivered
    streaming_profiler::CaptureContext::Device dev = one_tile_device();
    tt::llrt::CallSites call_sites;  // none: the tests set their routes themselves
    std::vector<api::detail::RecordLane> lanes = std::vector<api::detail::RecordLane>(profiler::kSpscNRiscDecode);
    streaming_profiler::RecordTypes types;
    streaming_profiler::RecordRoutes routes;
    streaming_profiler::StreamDecoder dec;
    uint32_t head = 0;

    // The state slots a frame carries for lane 0.
    uint32_t slot_timer = 0, slot_prog = 0;

    explicit Harness(
        std::vector<api::detail::RecordTypeDescriptor> record_types = {kRawRecordType}, Lane lane = Lane::Seeded) :
        types(std::move(record_types)), routes(types, call_sites), dec(dev, lanes, routes) {
        routes.set_route(kEventId, {.types = 1});
        if (lane == Lane::Seeded) {
            feed({word(PacketType::TimerHigh, 0)});
            deliver_oldest();
        }
    }
    // Decodes one frame carrying `words` as lane 0's new words.
    void feed(const std::vector<uint32_t>& words) {
        std::vector<uint32_t> frame(profiler::kSpscMaxFrameWords);
        frame[0] = kernel_profiler::spsc_span_w0();
        uint32_t* control = frame.data() + kernel_profiler::SPSC_SPAN_PREFIX_WORDS;
        frame[kernel_profiler::SPSC_PREFIX_XY] = kTileXy;
        frame[kernel_profiler::SPSC_PREFIX_HEAD_0] = head;
        control[kernel_profiler::SPSC_WIRE_TAIL_0] = head + words.size();
        control[kernel_profiler::SPSC_WIRE_TIMER_0] = slot_timer;
        control[kernel_profiler::spsc_wire_prog_word(0)] = slot_prog;
        uint32_t offset = kernel_profiler::SPSC_SPAN_PREFIX_WORDS + kernel_profiler::SPSC_SPAN_WIRE_CTRL_WORDS;
        offset += kernel_profiler::spsc_span_pack_pad(head, offset);
        std::copy(words.begin(), words.end(), frame.begin() + offset);
        frame[1] = offset + words.size() - kernel_profiler::SPSC_SPAN_PREFIX_WORDS;
        const uint32_t frame_words = kernel_profiler::spsc_span_frame_words(frame[1]);
        uint8_t* const type_outs[] = {
            reinterpret_cast<uint8_t*>(type_buffers[0].data()) + type_bytes[0],
            reinterpret_cast<uint8_t*>(type_buffers[1].data()) + type_bytes[1]};
        const auto decoded = dec.decode_frames(
            parked,
            frame.data(),
            {&frame_words, 1},
            {.values = value_buffer.data() + value_count,
             .types = {type_outs, types.type_count()},
             .order = order_buffer.data()});
        type_bytes[0] += decoded.type_bytes[0];
        type_bytes[1] += decoded.type_bytes[1];
        value_count += decoded.values;
        order_regressions += decoded.order_regressions;
        order_entries = decoded.order_entries;
        stalls += decoded.stalls;
        newest_ticks = decoded.newest_ticks;
        head += words.size();
        parked++;
    }
    // Delivers the oldest batch, as the service does once the clock sync has placed everything up to that batch's
    // newest record on the host timeline. Each feed() call decodes one batch, and the decoder repairs records only in
    // batches not yet delivered.
    void deliver_oldest() { parked--; }
    // Delivers every held batch, so the next one starts over in the buffers.
    void new_batch() {
        parked = 0;
        type_bytes = {};
        value_count = 0;
    }
    template <typename R = Raw>
    std::span<const R> records(uint32_t type = 0) const {
        return {reinterpret_cast<const R*>(type_buffers[type].data()), type_bytes[type] / sizeof(R)};
    }
    std::vector<uint32_t> order() const { return {order_buffer.begin(), order_buffer.begin() + order_entries}; }
};

int main() {
    {
        Harness h;
        const uint32_t payload[] = {0x04030201u, 0x08070605u, 0x0c0b0a09u};
        h.feed({record(5, kEventId), 60, payload[0], payload[1], payload[2]});
        check(
            h.records().size() == 1 &&
                std::ranges::equal(
                    std::span(h.records()[0].data).first(sizeof(payload)), std::as_bytes(std::span(payload))) &&
                h.records()[0].time.device_cycles() == 60 && h.records()[0].origin.call_site_id == kEventId &&
                h.records()[0].origin.lane == &h.lanes[0],
            "a DevicePublish record keeps its id and its bytes in order");
    }
    {
        Harness h({kRawRecordType, kRawRecordType});
        h.routes.set_route(kEventId, {.types = 0b11});
        h.routes.set_route(9, {.types = 0});
        h.feed({sticky(1), record(3, kEventId), kNear, 7, record(2, 9), 4});
        check(
            h.records(1).size() == 1 && h.records()[0].time.device_cycles() == kEpoch - 16 &&
                h.records(1)[0].time.device_cycles() == kEpoch - 16,
            "a torn record that two types take is repaired in both copies");
    }
    {
        Harness h({kRawRecordType, kRawRecordType});
        h.routes.set_route(6, {.types = 0b10});
        h.routes.set_route(7, {.types = 0b11});
        h.feed({event(), 10, event(), 11, record(2, 6), 12, record(2, 7), 13, event(), 14});
        check(
            h.order() == std::vector<uint32_t>{0, 0, 1, 0, 1, 0},
            "a consumer of more than one type gets an order entry per record of each type it goes to, in the order the "
            "lane sent them");
    }
    {
        // The decoder writes each record of a type without a fixed layout as a PublishedRecord, its time, length and
        // payload as sent, which the batch builds later.
        Harness h({{.fixed_layout = false}});
        h.routes.set_route(8, {.types = 1, .bits16_packet_words = 2, .bits64_packet_words = 5});
        h.routes.set_route(9, {.types = 0, .bits16_packet_words = 2, .bits64_packet_words = 5});
        // clang-format off
        h.feed({sticky(2),
                record(2, 8), short_body(5, 3),
                record(3, 8), 100, 7,
                record(2, 8), short_body(5, 3),
                record(5, 8), 200, 2, 0x11, 0x22,
                record(2, 8), short_body(100, 4),
                record(3, 9), 300, 9,
                record(2, 8), short_body(2, 5)});
        // clang-format on
        const auto published = h.records<streaming_profiler::PublishedRecord>();
        const auto payload_is = [&](size_t k, std::initializer_list<uint32_t> words) {
            return std::ranges::equal(
                std::span(published[k].bytes, published[k].byte_count), std::as_bytes(std::span(words)));
        };
        check(published.size() == 5, "a short record before the lane's cursor is set makes no record");
        check(
            published[0].ticks == 2 * kEpoch + 100 && published[0].packet_words == 3 && payload_is(0, {7}),
            "a medium record's time is its time word and its payload follows it");
        check(
            published[1].ticks == 2 * kEpoch + 105 && published[1].packet_words == 2 &&
                payload_is(1, {short_body(5, 3)}),
            "a short record's time counts from the cursor and its payload is its body");
        check(
            published[2].ticks == 2 * kEpoch + 200 && published[2].packet_words == 5 && payload_is(2, {0x11, 0x22}),
            "a full record's payload follows its whole time");
        check(published[3].ticks == 2 * kEpoch + 300, "a full record moves the cursor too");
        check(published[4].ticks == 2 * kEpoch + 302, "a record no type takes still moves the cursor");
        h.feed({record(5, 8), 400, 3, 0x11, 0x22});
        check(
            h.records<streaming_profiler::PublishedRecord>()[5].ticks == 3 * kEpoch + 400,
            "a full record's time is its own, not the lane's timer");
    }
    {
        Harness h;
        h.feed({record(3, profiler::kSpscStallZoneId), 40, 8, event(), 50});
        check(h.stalls == 1, "a stall is a record of the stall id");
    }
    {
        struct Direct {
            api::detail::RecordOrigin origin;
            api::Timestamp time, start, end;
        };
        static constexpr uint16_t kTimes[] = {
            api::detail::type_code<api::Timestamp>(), api::detail::type_code<api::Timestamp>()};
        const auto build = [](std::byte* at,
                              const api::detail::RecordOrigin& origin,
                              const api::Timestamp& time,
                              api::detail::FieldReader& data) {
            auto* record = new (at) Direct{origin, time, {}, {}};
            record->start = data.leaf<api::Timestamp>();
            record->end = data.leaf<api::Timestamp>();
        };
        using Op = streaming_profiler::PayloadToRecord::Op;
        streaming_profiler::PayloadToRecordByWidth by_width;
        // The short and medium forms send the start as a 16- or 32-bit distance back, and the full form sends it whole.
        constexpr uint32_t kStartBits[] = {16, 32, 64};
        for (uint32_t width = 0; width < 3; width++) {
            by_width[width] = {
                .fields =
                    {{width == 2 ? Op::Time : Op::Delta, kTimes[0], offsetof(Direct, start), 0, kStartBits[width]},
                     {Op::Published, kTimes[1], offsetof(Direct, end), 0, 0}},
                .follows_time = width == 2 ? std::vector<uint32_t>{offsetof(Direct, end)}
                                           : std::vector<uint32_t>{offsetof(Direct, start), offsetof(Direct, end)}};
        }
        const api::detail::RecordTypeDescriptor record_type{
            .payload_codes = kTimes, .size = sizeof(Direct), .build = build};
        const auto route = [&](Harness& h, uint32_t id, uint64_t types) {
            h.routes.set_route(
                id,
                {.types = types, .bits16_packet_words = 2, .bits64_packet_words = 5, .payload_to_record = &by_width});
        };
        {
            Harness h({record_type});
            route(h, 8, 1);
            const uint64_t whole_start = kEpoch + 150;
            // clang-format off
            h.feed({sticky(2),
                    record(3, 8), 100, 30,
                    record(2, 8), short_body(5, 3),
                    record(5, 8), 200, 2, uint32_t(whole_start), uint32_t(whole_start >> 32)});
            // clang-format on
            check(h.type_bytes[0] == 3 * sizeof(Direct), "a direct type's records take its own size");
            const std::span<const Direct> direct = h.records<Direct>();
            check(
                direct[0].time.device_cycles() == 2 * kEpoch + 100 &&
                    direct[0].start.device_cycles() == 2 * kEpoch + 70 &&
                    direct[0].end.device_cycles() == 2 * kEpoch + 100 && direct[0].origin.call_site_id == 8,
                "a medium record's Timestamp is its 32-bit distance back from the record's time");
            check(
                direct[1].time.device_cycles() == 2 * kEpoch + 105 &&
                    direct[1].start.device_cycles() == 2 * kEpoch + 102 &&
                    direct[1].end.device_cycles() == 2 * kEpoch + 105,
                "a short record's Timestamp is its 16-bit distance back");
            check(
                direct[2].time.device_cycles() == 2 * kEpoch + 200 && direct[2].start.device_cycles() == kEpoch + 150 &&
                    direct[2].end.device_cycles() == 2 * kEpoch + 200,
                "a full record's Timestamp is sent whole");
        }
        {
            Harness h({record_type});
            route(h, 8, 1);
            h.feed({sticky(1), record(3, 8), kNear, 8, record(3, 8), 4, 0});
            const Direct& torn = h.records<Direct>()[0];
            check(
                torn.time.device_cycles() == kEpoch - 16 && torn.start.device_cycles() == kEpoch - 24 &&
                    torn.end.device_cycles() == kEpoch - 16,
                "a torn direct record is repaired with the Timestamps computed from its time");
        }
        // The decoder writes runs of short records 16 at a time. Sites 8 and 9 share a layout, so their records run
        // together, and site 10's records go to a second type and end the run. 53 records make three full blocks of 16
        // and part of a fourth.
        {
            constexpr uint32_t kShortRecords = 53;
            Harness h({record_type, record_type});
            for (const uint32_t id : {8u, 9u, 10u}) {
                route(h, id, id == 10 ? 2 : 1);
            }
            struct Expected {
                uint32_t id;
                uint64_t time, start;
            };
            std::vector<Expected> expected[2];
            std::vector<uint32_t> words = {sticky(2), record(3, 8), 1000, 5};
            std::vector<uint32_t> types = {0};
            uint64_t time = 2 * kEpoch + 1000;
            expected[0].push_back({8, time, time - 5});
            for (uint32_t k = 0; k < kShortRecords; k++) {
                const uint32_t id = k >= 20 && k < 23 ? 10 : 8 + k % 2;
                const uint32_t delta = 100 + k * 37, distance = k * 11;
                time += delta;
                words.insert(words.end(), {record(2, id), short_body(delta, distance)});
                expected[id == 10].push_back({id, time, time - distance});
                types.push_back(id == 10);
            }
            h.feed(words);
            const auto matches = [](const Direct& direct, const Expected& want) {
                return direct.origin.call_site_id == want.id && direct.time.device_cycles() == want.time &&
                       direct.start.device_cycles() == want.start && direct.end.device_cycles() == want.time;
            };
            check(
                std::ranges::equal(h.records<Direct>(0), expected[0], matches) &&
                    std::ranges::equal(h.records<Direct>(1), expected[1], matches),
                "a run of short records gives each its time, its distance back and its site");
            check(h.order() == types, "a run of short records takes an entry of the order each");
        }
        {
            Harness h({kRawRecordType, record_type});
            route(h, 8, 0b10);
            // clang-format off
            h.feed({sticky(2), record(3, 8), 0xffffff00u, 5,
                    sticky(3), event(), 0xffffff10u,
                    sticky(2), record(2, 8), short_body(0x20, 3)});
            // clang-format on
            check(
                h.records()[0].time.device_cycles() == 2 * kEpoch + 0xffffff10 && h.order_regressions == 0,
                "the first record of a run of short records witnesses a torn record");
        }
    }
    {
        // kSegments segments of kSegmentTicks wall ticks, each of whose groups of four spans up to kGroupSpread ticks.
        constexpr int64_t kSegmentTicks = 1'000'000, kSegments = 20, kGroupSpread = 150'000;
        streaming_profiler::ClockMap map(1, {.root_refclk = 0, .tsc = 1'000'000});
        for (int64_t k = 0; k <= kSegments; k++) {
            map.append(
                0,
                {.at = kSegmentTicks * k + 5,
                 .value = 0.3 + 999'000.0 * static_cast<double>(k),
                 .tangent = 0.999 + 1e-6 * static_cast<double>(k)});
            map.append_host(
                {.at = 1'000'000.0 * static_cast<double>(k),
                 .value = 2'700'000.25 * static_cast<double>(k),
                 .tangent = 2.7 + 1e-5 * static_cast<double>(k)},
                1'000'000.0 * static_cast<double>(k + 1));
        }
        map.finish(0);
        streaming_profiler::ClockMap::Reader together = map.reader(), alone = map.reader();
        std::mt19937_64 rng(7);
        bool same = true;
        // Groups walk the map as a capture's records do, so most fall within the segments the last group found.
        int64_t near = 5;
        for (int group = 0; same && group < 100'000; group++) {
            near = near < kSegments * kSegmentTicks - 2 * kGroupSpread ? near + static_cast<int64_t>(rng() % 1'000) : 5;
            alignas(32) int64_t walls[4], hosts[4];
            for (int64_t& wall : walls) {
                wall = near + static_cast<int64_t>(rng() % kGroupSpread);
            }
            simde_mm256_store_si256(
                reinterpret_cast<simde__m256i*>(hosts),
                map.place_host4(together, 0, simde_mm256_load_si256(reinterpret_cast<const simde__m256i*>(walls))));
            for (int k = 0; same && k < 4; k++) {
                const int64_t host = map.place_host(alone, 0, walls[k]);
                if (hosts[k] != host) {
                    std::fprintf(
                        stderr,
                        "wall tick %lld: place_host4() gives %lld, place_host() %lld\n",
                        static_cast<long long>(walls[k]),
                        static_cast<long long>(hosts[k]),
                        static_cast<long long>(host));
                    same = false;
                }
            }
        }
        check(same, "four wall ticks placed together land where place_host() puts each");
    }
    // Lane state from a frame's slots.
    {
        Harness h({kRawRecordType}, Lane::Fresh);
        h.slot_timer = 5;
        h.slot_prog = 7;
        h.feed({event(), 40});
        check(
            h.records()[0].time.device_cycles() == 5 * kEpoch + 40 && h.records()[0].origin.runtime_id == 7,
            "a lane's first frame seeds timer and runtime id from its slots");
    }
    {
        Harness h({kRawRecordType}, Lane::Fresh);
        h.slot_timer = 5;
        h.feed({event(), 4, sticky(9), event(), 8});
        check(
            h.records().size() == 1 && h.records()[0].time.device_cycles() == 9 * kEpoch + 8,
            "a reseeding run is taken up from its last sticky");
    }
    {
        Harness h;
        h.feed({sticky(2), event(), 32});
        h.head += 40;  // the wire moved on without us
        h.slot_timer = 3;
        h.slot_prog = 9;
        h.feed({event(), 64});
        check(
            h.records().size() == 2 && h.records()[1].time.device_cycles() == 3 * kEpoch + 64 &&
                h.records()[1].origin.runtime_id == 9,
            "after a gap the next frame's slots reseed the lane");
    }
    // A record whose next record's time is up to 2^32 ticks earlier read its high word one too high, and moves down
    // 2^32 ticks while its batch is undelivered.
    {
        Harness h;
        h.feed({sticky(1), event(), kNear, event(), 4});
        check(
            h.records()[0].time.device_cycles() == kEpoch - 16 && h.records()[1].time.device_cycles() == kEpoch + 4,
            "a torn record moves down an epoch when the next record is earlier");
    }
    {
        Harness h;
        h.routes.set_route(9, {.types = 0});
        h.feed({sticky(1), record(2, 9), kNear, event(), 4});
        check(h.order_regressions == 0, "a torn record no type takes is a tear, not an order regression");
    }
    {
        Harness h;
        h.feed({sticky(1), record(6, kEventId), kNear, 0x11, 0x22, 0x33, 0x44, event(), 4});
        check(h.records()[0].time.device_cycles() == kEpoch - 16, "a torn record with a payload moves down an epoch");
        uint32_t payload[4];
        std::memcpy(payload, h.records()[0].data.data(), sizeof(payload));
        check(
            payload[0] == 0x11 && payload[1] == 0x22 && payload[2] == 0x33 && payload[3] == 0x44,
            "a repair leaves the record's payload as sent");
    }
    const struct {
        std::vector<uint32_t> words;
        uint64_t cycles;
        const char* why;
    } kFirstRecord[] = {
        {{sticky(1), event(), 100, event(), 200}, kEpoch + 100, "records in order keep their times"},
        {{sticky(1), event(), 0x10000000u, event(), 4},
         0x10000000u,
         "a torn record far below the wrap is still repaired"},
        {{sticky(1), event(), kNear, word(PacketType::RuntimeId, 2), sticky(1), event(), 4},
         kEpoch - 16,
         "stickies between a torn record and the next record leave the repair intact"},
        {{sticky(1), event(), kNear}, 2 * kEpoch - 16, "a torn record with no record after it keeps its time"},
    };
    for (const auto& c : kFirstRecord) {
        Harness h;
        h.feed(c.words);
        check(h.records()[0].time.device_cycles() == c.cycles, c.why);
    }
    {
        Harness h;
        h.feed({sticky(1), event(), kNear});
        h.feed({event(), 4});
        check(
            h.records()[0].time.device_cycles() == kEpoch - 16,
            "the next record in a later frame of the same batch repairs a torn record");
    }
    {
        // A full record of a site whose payload holds Timestamps carries its time whole.
        Harness h;
        h.routes.set_route(8, {.types = 0, .bits16_packet_words = 2, .bits64_packet_words = 5});
        h.feed({sticky(3), event(), kNear});
        h.feed({record(5, 8), 4, 3, 0, 0});
        check(h.records()[0].time.device_cycles() == 3 * kEpoch - 16, "a full record witnesses a torn record");
    }
    {
        // The witness goes to a second type's region, so the delivered record stays readable.
        Harness h({kRawRecordType, kRawRecordType});
        h.routes.set_route(6, {.types = 0b10});
        h.feed({sticky(1), event(), kNear});
        const Raw& delivered = h.records()[0];
        h.new_batch();
        h.feed({record(2, 6), 4});
        check(
            h.order_regressions == 1 && h.records(1)[0].time.device_cycles() == kEpoch + 4 &&
                delivered.time.device_cycles() == 2 * kEpoch - 16,
            "a torn record already delivered counts a regression and stays as it is");
    }
    {
        Harness h;
        h.feed({sticky(1), event(), kNear});
        h.new_batch();
        h.feed({event(), 4});
        h.feed({event(), 8});
        check(
            h.order_regressions == 1 && h.records()[0].time.device_cycles() == kEpoch + 4,
            "a failed repair does not retarget the next record");
    }
    {
        Harness h;
        h.feed({event(), 100});
        h.feed({sticky(1), event(), kNear});
        h.deliver_oldest();
        h.feed({event(), 4});
        check(
            h.order_regressions == 0 && h.records()[0].time.device_cycles() == 100 &&
                h.records()[1].time.device_cycles() == kEpoch - 16,
            "a repair lands in a batch still parked behind delivered ones");
    }
    {
        Harness h;
        h.feed({sticky(1), event(), 100});
        check(h.newest_ticks == static_cast<int64_t>(kEpoch + 100), "a lane's first record counts toward the cover");
        h.feed({sticky(2), event(), kNear});
        check(
            h.newest_ticks == static_cast<int64_t>(2 * kEpoch - 16),
            "a torn last record holds its batch to the time it would repair to");
    }
    {
        Harness h;
        h.feed({sticky(1), event(), 100});
        h.feed({sticky(2), event(), 200});
        check(
            h.newest_ticks == static_cast<int64_t>(2 * kEpoch + 200),
            "a lane back from an idle gap of 2^32 ticks holds its batch to its own last record");
    }
    {
        Harness h;
        h.feed({event(), kNear, event(), 4});
        check(
            h.order_regressions == 1 && h.records()[0].time.device_cycles() == kNear,
            "a record before the first 2^32 ticks cannot have read its high word too high");
    }
    {
        Harness h;
        h.feed({sticky(3), event(), kNear});
        h.feed({sticky(2), event(), 32});
        check(
            h.order_regressions == 1 && h.records()[0].time.device_cycles() == 4 * kEpoch - 16 &&
                h.records()[1].time.device_cycles() == 2 * kEpoch + 32,
            "a regression beyond one epoch is not a tear");
    }
    std::puts("PASS");
}
