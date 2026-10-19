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

using namespace tt::tt_metal;
namespace sp = experimental::streaming_profiler;

constexpr uint64_t kEpoch = 1ull << 32;  // a wall-clock time is its high word * kEpoch + its low word
constexpr uint32_t kNear = 0xfffffff0u;  // a low word just below the wrap
constexpr uint32_t kTileXy = 0x30002;
constexpr uint32_t kEventId = 5;

uint32_t word(uint32_t type, uint32_t low27) { return (type << PP_TYPE_SHIFT) | low27; }
uint32_t sticky(uint32_t hi) { return word(PP_STICKY_TIMER, hi); }
uint32_t record(uint32_t length, uint32_t id) { return word(PP_RECORD, id | length << PP_LENGTH_SHIFT); }
uint32_t event() { return record(1, kEventId); }
// A short record's body: its time's distance from the lane's cursor, then its 16-bit payload.
uint32_t short_body(uint32_t since_cursor, uint32_t payload) { return since_cursor << 16 | payload; }

void check(bool v, const char* why) {
    if (!v) {
        std::fprintf(stderr, "FAIL: %s\n", why);
        std::exit(1);
    }
}

// A record of the harness's default type, whose data is the record's payload as sent.
struct Raw {
    sp::detail::Source source;
    sp::Timestamp time;
    std::array<std::byte, 16> data;
};
static_assert(sizeof(Raw) == profiler::kSpscPublishedBytes);
constexpr sp::detail::Kind kRawKind{.size = sizeof(Raw)};

// A Seeded lane first gets one frame holding a zero timer sticky, so a test's own frames decode as the wire would after
// a capture's first frame.
enum class Lane { Seeded, Fresh };

streaming_profiler::CaptureContext::Device one_tile_device() {
    streaming_profiler::CaptureContext::Device dev{
        .lanes = std::vector<sp::Core>(profiler::kSpscNRiscDecode), .clock_offsets = {0}};
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
    streaming_profiler::RecordRoutes routes;
    streaming_profiler::StreamDecoder dec;
    uint32_t head = 0;

    // The state slots a frame carries for lane 0.
    uint32_t slot_timer = 0, slot_prog = 0;

    explicit Harness(std::vector<sp::detail::Kind> kinds = {kRawKind}, Lane lane = Lane::Seeded) :
        routes(std::move(kinds)), dec(dev, routes) {
        routes.set_route(kEventId, {.types = 1});
        if (lane == Lane::Seeded) {
            feed({word(PP_STICKY_TIMER, 0)});
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
             .types = {type_outs, routes.type_count()},
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
        h.feed({record(4, kEventId), 60, payload[0], payload[1], payload[2]});
        check(
            h.records().size() == 1 &&
                std::ranges::equal(
                    std::span(h.records()[0].data).first(sizeof(payload)), std::as_bytes(std::span(payload))) &&
                h.records()[0].time.device_cycles() == 60 && h.records()[0].source.zone_id == kEventId,
            "a DevicePublish record keeps its id and its bytes in order");
    }
    {
        Harness h({kRawKind, kRawKind});
        h.routes.set_route(kEventId, {.types = 0b11});
        h.feed({sticky(1), record(2, kEventId), kNear, 7, record(1, 9), 4});
        check(
            h.records(1).size() == 1 && h.records()[0].time.device_cycles() == kEpoch - 16 &&
                h.records(1)[0].time.device_cycles() == kEpoch - 16,
            "a torn record that two types take is repaired in both copies");
    }
    {
        Harness h({kRawKind, kRawKind});
        h.routes.set_route(6, {.types = 0b10});
        h.routes.set_route(7, {.types = 0b11});
        h.feed(
            {record(1, kEventId),
             10,
             record(1, kEventId),
             11,
             record(1, 6),
             12,
             record(1, 7),
             13,
             record(1, kEventId),
             14});
        check(
            h.order() == std::vector<uint32_t>{0, 0, 1, 0, 1, 0},
            "a consumer of more than one type gets an order entry per record of each type it goes to, in the order the "
            "lane sent them");
    }
    {
        // The decoder writes each record of a type with values as a PublishedRecord, its time, length and payload as
        // sent, which the batch builds later.
        static constexpr sp::detail::ValueType kAnyValue[] = {{}};
        Harness h({{.value_types = kAnyValue}});
        h.routes.set_route(8, {.types = 1, .short_length = 1, .full_length = 4});
        h.routes.set_route(9, {.types = 0, .short_length = 1, .full_length = 4});
        // clang-format off
        h.feed({sticky(2),
                record(1, 8), short_body(5, 3),
                record(2, 8), 100, 7,
                record(1, 8), short_body(5, 3),
                record(4, 8), 200, 2, 0x11, 0x22,
                record(1, 8), short_body(100, 4),
                record(2, 9), 300, 9,
                record(1, 8), short_body(2, 5)});
        // clang-format on
        const auto published = h.records<streaming_profiler::PublishedRecord>();
        const auto payload_is = [&](size_t k, std::initializer_list<uint32_t> words) {
            return std::ranges::equal(
                std::span(published[k].bytes, published[k].byte_count), std::as_bytes(std::span(words)));
        };
        check(published.size() == 5, "a short record before the lane's cursor is set makes no record");
        check(
            published[0].ticks == 2 * kEpoch + 100 && published[0].length == 2 && payload_is(0, {7}),
            "a medium record's time is its time word and its payload follows it");
        check(
            published[1].ticks == 2 * kEpoch + 105 && published[1].length == 1 && payload_is(1, {short_body(5, 3)}),
            "a short record's time counts from the cursor and its payload is its body");
        check(
            published[2].ticks == 2 * kEpoch + 200 && published[2].length == 4 && payload_is(2, {0x11, 0x22}),
            "a full record's payload follows its whole time");
        check(published[3].ticks == 2 * kEpoch + 205, "a full record does not move the cursor");
        check(published[4].ticks == 2 * kEpoch + 302, "a record no type takes still moves the cursor");
        h.feed({record(4, 8), 400, 3, 0x11, 0x22});
        check(
            h.records<streaming_profiler::PublishedRecord>()[5].ticks == 3 * kEpoch + 400,
            "a full record's time is its own, not the lane's timer");
    }
    {
        Harness h;
        h.feed({record(2, profiler::kSpscStallZoneId), 40, 8, event(), 50});
        check(h.stalls == 1, "a stall is a record of the stall id");
    }
    {
        struct Direct {
            sp::detail::Source source;
            sp::Timestamp time, start, end;
        };
        static constexpr uint16_t kTimes[] = {
            sp::detail::type_code<sp::Timestamp>(), sp::detail::type_code<sp::Timestamp>()};
        const auto build =
            [](void* at, const sp::detail::Source& source, const sp::Timestamp& time, sp::detail::FieldReader& data) {
                auto* record = new (at) Direct{source, time, {}, {}};
                record->start = data.leaf<sp::Timestamp>();
                record->end = data.leaf<sp::Timestamp>();
            };
        using Op = streaming_profiler::DirectForm::Op;
        streaming_profiler::DirectForms forms;
        constexpr uint32_t kDistanceBits[] = {16, 32, 64};
        for (uint32_t form = 0; form < 3; form++) {
            forms[form] = {
                .fields =
                    {{Op::Distance, kTimes[0], offsetof(Direct, start), 0, kDistanceBits[form]},
                     {Op::Published, kTimes[1], offsetof(Direct, end), 0, 0}},
                .follows_time = {offsetof(Direct, start), offsetof(Direct, end)}};
        }
        const sp::detail::Kind kind{.data_codes = kTimes, .size = sizeof(Direct), .build = build};
        {
            Harness h({kind});
            h.routes.set_route(8, {.types = 1, .short_length = 1, .full_length = 4, .forms = &forms});
            const uint64_t far = kEpoch + 50;
            h.feed(
                {sticky(2),
                 record(2, 8),
                 100,
                 30,
                 record(1, 8),
                 short_body(5, 3),
                 record(4, 8),
                 200,
                 2,
                 uint32_t(far),
                 uint32_t(far >> 32)});
            check(h.type_bytes[0] == 3 * sizeof(Direct), "a direct type's records take its own size");
            const std::span<const Direct> direct = h.records<Direct>();
            check(
                direct[0].time.device_cycles() == 2 * kEpoch + 100 &&
                    direct[0].start.device_cycles() == 2 * kEpoch + 70 &&
                    direct[0].end.device_cycles() == 2 * kEpoch + 100 && direct[0].source.zone_id == 8,
                "a medium record's Timestamp is its 32-bit distance back from the record's time");
            check(
                direct[1].time.device_cycles() == 2 * kEpoch + 105 &&
                    direct[1].start.device_cycles() == 2 * kEpoch + 102 &&
                    direct[1].end.device_cycles() == 2 * kEpoch + 105,
                "a short record's Timestamp is its 16-bit distance back");
            check(
                direct[2].time.device_cycles() == 2 * kEpoch + 200 && direct[2].start.device_cycles() == kEpoch + 150 &&
                    direct[2].end.device_cycles() == 2 * kEpoch + 200,
                "a full record's Timestamp is its whole distance back");
        }
        {
            Harness h({kind});
            h.routes.set_route(8, {.types = 1, .short_length = 1, .full_length = 4, .forms = &forms});
            h.feed({sticky(1), record(2, 8), kNear, 8, record(2, 8), 4, 0});
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
            Harness h({kind, kind});
            for (const uint32_t id : {8u, 9u, 10u}) {
                h.routes.set_route(
                    id, {.types = id == 10 ? 2u : 1u, .short_length = 1, .full_length = 4, .forms = &forms});
            }
            struct Expected {
                uint32_t id;
                uint64_t time, start;
            };
            std::vector<Expected> expected[2];
            std::vector<uint32_t> words = {sticky(2), record(2, 8), 1000, 5};
            std::vector<uint32_t> types = {0};
            uint64_t time = 2 * kEpoch + 1000;
            expected[0].push_back({8, time, time - 5});
            for (uint32_t k = 0; k < kShortRecords; k++) {
                const uint32_t id = k >= 20 && k < 23 ? 10 : 8 + k % 2;
                const uint32_t delta = 100 + k * 37, distance = k * 11;
                time += delta;
                words.insert(words.end(), {record(1, id), short_body(delta, distance)});
                expected[id == 10].push_back({id, time, time - distance});
                types.push_back(id == 10);
            }
            h.feed(words);
            const auto matches = [](const Direct& direct, const Expected& want) {
                return direct.source.zone_id == want.id && direct.time.device_cycles() == want.time &&
                       direct.start.device_cycles() == want.start && direct.end.device_cycles() == want.time;
            };
            check(
                std::ranges::equal(h.records<Direct>(0), expected[0], matches) &&
                    std::ranges::equal(h.records<Direct>(1), expected[1], matches),
                "a run of short records gives each its time, its distance back and its site");
            check(h.order() == types, "a run of short records takes an entry of the order each");
        }
        {
            Harness h({kRawKind, kind});
            h.routes.set_route(8, {.types = 0b10, .short_length = 1, .full_length = 4, .forms = &forms});
            h.feed(
                {sticky(2),
                 record(2, 8),
                 0xffffff00u,
                 5,
                 sticky(3),
                 event(),
                 0xffffff10u,
                 sticky(2),
                 record(1, 8),
                 short_body(0x20, 3)});
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
        Harness h({kRawKind}, Lane::Fresh);
        h.slot_timer = 5;
        h.slot_prog = 7;
        h.feed({event(), 40});
        check(
            h.records()[0].time.device_cycles() == 5 * kEpoch + 40 && h.records()[0].source.runtime_id == 7,
            "a lane's first frame seeds timer and runtime id from its slots");
    }
    {
        Harness h({kRawKind}, Lane::Fresh);
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
                h.records()[1].source.runtime_id == 9,
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
        h.feed({sticky(1), record(1, 9), kNear, event(), 4});
        check(h.order_regressions == 0, "a torn record no type takes is a tear, not an order regression");
    }
    {
        Harness h;
        h.feed({sticky(1), record(5, kEventId), kNear, 0x11, 0x22, 0x33, 0x44, event(), 4});
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
        {{sticky(1), event(), kNear, word(PP_STICKY_PROG, 2), sticky(1), event(), 4},
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
        // A full record of a site whose data holds Timestamps carries its time whole.
        Harness h;
        h.routes.set_route(8, {.types = 0, .short_length = 1, .full_length = 4});
        h.feed({sticky(3), event(), kNear});
        h.feed({record(4, 8), 4, 3, 0, 0});
        check(h.records()[0].time.device_cycles() == 3 * kEpoch - 16, "a full record witnesses a torn record");
    }
    {
        // The witness goes to a second type's region, so the delivered record stays readable.
        Harness h({kRawKind, kRawKind});
        h.routes.set_route(6, {.types = 0b10});
        h.feed({sticky(1), event(), kNear});
        const Raw& delivered = h.records()[0];
        h.new_batch();
        h.feed({record(1, 6), 4});
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
