// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <vector>

#include "impl/streaming_profiler/decode.hpp"

using namespace tt::tt_metal;

constexpr uint64_t kEpoch = 1ull << 32;  // a wall-clock time is its high word * kEpoch + its low word
constexpr uint32_t kNear = 0xfffffff0u;  // a low word just below the wrap
constexpr uint32_t kTileXy = 0x30002;

uint32_t word(uint32_t t, uint32_t id = 1) { return (t << PP_TYPE_SHIFT) | id; }
uint32_t sticky(uint32_t hi) { return word(PP_STICKY_TIMER, hi); }

void check(bool v, const char* why) {
    if (!v) {
        std::fprintf(stderr, "FAIL: %s\n", why);
        std::exit(1);
    }
}

streaming_profiler::CaptureContext::Device one_tile_device() {
    streaming_profiler::CaptureContext::Device dev{
        .lanes = std::vector<experimental::streaming_profiler::Core>(profiler::kSpscNRiscDecode), .clock_offsets = {0}};
    dev.core_of_xy[kTileXy] = 0;
    return dev;
}

struct Harness {
    // One buffer per record kind; the frames of one batch append to them.
    std::vector<uint64_t> zone_buffer = std::vector<uint64_t>(64 * 1024), event_buffer = zone_buffer,
                          data_buffer = zone_buffer, value_buffer = zone_buffer;
    size_t zones = 0, events = 0, data_records = 0, value_count = 0;
    uint64_t order_regressions = 0, stalls = 0;
    int64_t newest_ticks = 0;
    uint32_t parked = 0;  // batches decoded and not yet delivered
    streaming_profiler::CaptureContext::Device dev = one_tile_device();
    streaming_profiler::StreamDecoder dec;
    uint32_t head = 0;
    const experimental::streaming_profiler::TimestampedData* first_data = nullptr;

    // The state slots a frame carries for lane 0.
    uint32_t slot_timer = 0, slot_prog = 0;

    // `warm` feeds one frame holding a zero timer sticky, so a test's own frames find the lane seeded and decode as
    // the wire would after a capture's first frame.
    explicit Harness(
        bool warm = true,
        experimental::streaming_profiler::RecordType types = experimental::streaming_profiler::RecordType::All) :
        dec(dev, types) {
        if (warm) {
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
        const auto decoded = dec.decode_frames(
            parked,
            frame.data(),
            {&frame_words, 1},
            {reinterpret_cast<uint8_t*>(zone_buffer.data()) + zones * profiler::kSpscZoneBytes,
             reinterpret_cast<uint8_t*>(event_buffer.data()) + events * profiler::kSpscEventBytes,
             reinterpret_cast<uint8_t*>(data_buffer.data()) + data_records * profiler::kSpscDataBytes,
             value_buffer.data() + value_count});
        zones += decoded.zones;
        events += decoded.events;
        data_records += decoded.data;
        value_count += decoded.values;
        order_regressions += decoded.order_regressions;
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
        zones = events = data_records = value_count = 0;
        first_data = nullptr;
    }
    const experimental::streaming_profiler::Zone& zone(size_t k) const {
        return reinterpret_cast<const experimental::streaming_profiler::Zone*>(zone_buffer.data())[k];
    }
    uint64_t zone_start(size_t k) const { return zone(k).start_device_cycles(); }
    uint32_t zone_runtime_id(size_t k) const { return zone(k).runtime_id(); }
    uint64_t zone_duration(size_t k) const { return zone(k).end_device_cycles() - zone(k).start_device_cycles(); }
    uint64_t event_cycles(size_t k) const {
        return reinterpret_cast<const experimental::streaming_profiler::Event*>(event_buffer.data())[k].device_cycles();
    }
    // Returns the first data record. The decoder writes only raw bytes, so this constructs the TimestampedData in them
    // on first use.
    const experimental::streaming_profiler::TimestampedData& data() {
        if (first_data == nullptr) {
            streaming_profiler::construct_timestamped_data(reinterpret_cast<uint8_t*>(data_buffer.data()), 0);
            first_data = std::launder(
                reinterpret_cast<const experimental::streaming_profiler::TimestampedData*>(data_buffer.data()));
        }
        return *first_data;
    }
    uint64_t data_cycles() { return data().device_cycles(); }
    uint64_t data_value(size_t k) { return data().payload()[k]; }
};

int main() {
    // A decoder built for some record kinds produces only those, with the same times a decoder of every kind gives
    // them.
    {
        Harness h(true, experimental::streaming_profiler::RecordType::Zones);
        h.feed({word(PP_ZONE_ATOMIC), 40, 8, word(PP_EVENT), 50, word(PP_ZONE_ATOMIC), 70, 4});
        check(
            h.zones == 2 && h.events == 0 && h.zone_start(0) == 32 && h.zone_start(1) == 66,
            "a zones-only decoder produces the zones and no events");
    }
    {
        Harness h(true, experimental::streaming_profiler::RecordType::Events);
        h.feed({word(PP_ZONE_ATOMIC), 40, 8, word(PP_EVENT), 50, word(PP_ZONE_ATOMIC), 70, 4});
        check(
            h.zones == 0 && h.events == 1 && h.event_cycles(0) == 50,
            "an events-only decoder produces the event alone");
    }
    {
        Harness h(true, experimental::streaming_profiler::RecordType::Events);
        h.feed({word(PP_ZONE_ATOMIC, profiler::kSpscStallZoneId), 40, 8, word(PP_EVENT), 50});
        check(h.zones == 0 && h.stalls == 1, "a decoder that skips zones still counts stalls");
    }
    // Lane state from a frame's slots.
    {
        Harness h(false);
        h.slot_timer = 5;
        h.slot_prog = 7;
        h.feed({word(PP_ZONE_ATOMIC), 40, 8});
        check(
            h.zones == 1 && h.zone_start(0) == 5 * kEpoch + 40 - 8 && h.zone_runtime_id(0) == 7,
            "a lane's first frame seeds timer and runtime id from its slots");
    }
    {
        Harness h(false);
        h.slot_timer = 5;
        h.feed({word(PP_ZONE_ATOMIC), 4, 1, sticky(9), word(PP_ZONE_ATOMIC), 8, 1});
        check(
            h.zones == 1 && h.zone_start(0) == 9 * kEpoch + 8 - 1, "a reseeding run is taken up from its last sticky");
    }
    {
        Harness h(false);
        h.slot_timer = 2;
        h.feed({word(PP_ZONE_S), (5u << 16) | 1u, word(PP_ZONE_ATOMIC), 100, 8, word(PP_ZONE_S), (5u << 16) | 1u});
        check(
            h.zones == 2 && h.zone_start(0) == 2 * kEpoch + 100 - 8 && h.zone_start(1) == 2 * kEpoch + 105 - 1,
            "ZONE_S before the first absolute zone is skipped, not placed");
    }
    {
        Harness h;
        h.feed({sticky(2), word(PP_ZONE_ATOMIC), 32, 8});
        h.head += 40;  // the wire moved on without us
        h.slot_timer = 3;
        h.slot_prog = 9;
        h.feed({word(PP_ZONE_ATOMIC), 64, 8});
        check(
            h.zones == 2 && h.zone_start(1) == 3 * kEpoch + 64 - 8 && h.zone_runtime_id(1) == 9,
            "after a gap the next frame's slots reseed the lane");
    }
    {
        Harness h;
        h.feed({sticky(1), word(PP_EVENT), kNear, word(PP_EVENT), 4});
        check(h.event_cycles(0) == kEpoch - 16 && h.event_cycles(1) == kEpoch + 4, "torn EVENT, EVENT witness");
    }
    {
        Harness h;
        h.feed(
            {sticky(1),
             word(PP_DATA),
             kNear,
             4u << PP_DATA_SIZE_SHIFT,
             0x11,
             0x22,
             0x33,
             0x44,
             word(PP_ZONE_ATOMIC),
             4,
             2});
        check(h.data_cycles() == kEpoch - 16, "torn DATA, zone witness");
        check(
            h.data_value(0) == (0x11ull << 32 | 0x22) && h.data_value(1) == (0x33ull << 32 | 0x44),
            "payload untouched");
    }
    {
        Harness h;
        h.feed({sticky(1), word(PP_DATA), 4, 10u << PP_DATA_SIZE_SHIFT, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
        const experimental::streaming_profiler::TimestampedData kept = h.data();
        std::ranges::fill(h.data_buffer, 0);
        std::ranges::fill(h.value_buffer, 0);
        check(
            kept.device_cycles() == kEpoch + 4 && kept.payload().size() == 5 && kept.payload()[0] == (1ull << 32 | 2) &&
                kept.payload()[4] == (9ull << 32 | 10),
            "a copy keeps its values past the batch");
    }
    {
        Harness h;
        h.feed(
            {sticky(1),
             word(PP_ZONE_ATOMIC),
             kNear,
             8,
             sticky(1),
             word(PP_DATA),
             4,
             2u << PP_DATA_SIZE_SHIFT,
             0x11,
             0x22});
        check(h.zone_start(0) == kEpoch - 24 && h.zone_duration(0) == 8, "torn zone, DATA witness");
    }
    {
        Harness h;
        h.feed({sticky(1), word(PP_EVENT), 100, word(PP_EVENT), 200});
        check(h.event_cycles(0) == kEpoch + 100, "ordered events untouched");
    }
    {
        Harness h;
        h.feed({sticky(1), word(PP_EVENT), 0x10000000u, word(PP_EVENT), 4});
        check(h.event_cycles(0) == 0x10000000u, "regression far from the wrap still repairs");
    }
    {
        Harness h;
        h.feed({sticky(1), word(PP_EVENT), kNear, word(PP_STICKY_PROG, 2), sticky(1), word(PP_EVENT), 4});
        check(h.event_cycles(0) == kEpoch - 16, "metadata between point and witness");
    }
    {
        Harness h;
        h.feed({sticky(1), word(PP_EVENT), kNear});
        h.feed({word(PP_ZONE_ATOMIC), 4, 8});
        check(h.event_cycles(0) == kEpoch - 16, "witness in a later frame of the same batch");
    }
    {
        Harness h;
        h.feed({sticky(3), word(PP_EVENT), kNear});
        const uint64_t duration = kEpoch + 100;
        h.feed({word(PP_ZONE_L), 4, 3, uint32_t(duration), uint32_t(duration >> 32)});
        check(h.event_cycles(0) == 3 * kEpoch - 16, "a ZONE_L witnesses a torn point");
    }
    {
        Harness h;
        h.feed({sticky(1), word(PP_EVENT), kNear});
        check(h.event_cycles(0) == 2 * kEpoch - 16, "uncovered: final point with no witness");
    }
    {
        Harness h;
        h.feed({sticky(1), word(PP_EVENT), kNear});
        h.new_batch();
        h.feed({word(PP_ZONE_ATOMIC), 4, 0});
        check(
            h.order_regressions == 1 && h.zone_start(0) == kEpoch + 4 && h.event_cycles(0) == 2 * kEpoch - 16,
            "uncovered: target already delivered counts a regression, touches nothing");
    }
    {
        Harness h;
        h.feed({sticky(1), word(PP_ZONE_ATOMIC), kNear, 8});
        h.new_batch();
        h.feed({word(PP_ZONE_ATOMIC), 4, 2});
        h.feed({word(PP_ZONE_ATOMIC), 8, 2});
        check(
            h.order_regressions == 1 && h.zone_start(0) == kEpoch + 2,
            "a failed repair does not retarget the next record");
    }
    {
        Harness h;
        h.feed({word(PP_EVENT), 100});
        h.feed({sticky(1), word(PP_EVENT), kNear});
        h.deliver_oldest();
        h.feed({word(PP_ZONE_ATOMIC), 4, 0});
        check(
            h.order_regressions == 0 && h.event_cycles(0) == 100 && h.event_cycles(1) == kEpoch - 16,
            "a repair lands in a batch still parked behind delivered ones");
    }
    {
        Harness h;
        h.feed({sticky(1), word(PP_EVENT), 100});
        check(h.newest_ticks == static_cast<int64_t>(kEpoch + 100), "a lane's first record counts toward the cover");
        h.feed({sticky(2), word(PP_EVENT), kNear});
        check(
            h.newest_ticks == static_cast<int64_t>(2 * kEpoch - 16),
            "a torn last record holds its batch to the time it would repair to");
    }
    {
        Harness h;
        h.feed({sticky(1), word(PP_EVENT), 100});
        h.feed({sticky(2), word(PP_EVENT), 200});
        check(
            h.newest_ticks == static_cast<int64_t>(2 * kEpoch + 200),
            "a lane back from an idle gap of 2^32 ticks holds its batch to its own last record");
    }
    {
        Harness h;
        h.feed({sticky(2), word(PP_ZONE_ATOMIC), 16, 32});
        check(h.zone_start(0) == 2 * kEpoch - 16, "uncovered: torn start with no later record");
    }
    {
        Harness h;
        const uint64_t duration = uint64_t(20) - kEpoch;
        h.feed({word(PP_ZONE_L), 4, 1, uint32_t(duration), uint32_t(duration >> 32)});
        check(h.zone_start(0) == kEpoch - 16 && h.zone_duration(0) == 20, "ZONE_L torn start: negative elapsed");
    }
    {
        Harness h;
        const uint64_t duration = uint64_t(0) - 2 * kEpoch;
        h.feed({word(PP_ZONE_L), 4, 3, uint32_t(duration), uint32_t(duration >> 32)});
        check(h.zone_duration(0) == duration, "an elapsed time below -2^32 is not a tear");
    }
    {
        Harness h;
        const uint64_t duration = uint64_t(0) - 20;
        h.feed({word(PP_ZONE_L), 4, 0, uint32_t(duration), uint32_t(duration >> 32)});
        check(
            h.order_regressions == 1 && h.zone_start(0) == 24 && h.zone_duration(0) == duration,
            "a repair cannot place a start before zero");
    }
    {
        Harness h;
        const uint64_t duration = kEpoch + 100;
        h.feed(
            {sticky(3),
             word(PP_ZONE_L),
             kNear,
             3,
             uint32_t(duration),
             uint32_t(duration >> 32),
             word(PP_ZONE_ATOMIC),
             4,
             8});
        check(
            h.zone_duration(0) == 100 && h.zone_start(0) == 3 * kEpoch - 116,
            "ZONE_L torn end: the inflated duration moves, the start stays");
    }
    {
        Harness h;
        h.feed({sticky(3), word(PP_ZONE_ATOMIC, 1), kNear, 8, word(PP_ZONE_L), kNear + 8, 2, 100, 0});
        check(
            h.order_regressions == 0 && h.zone_start(0) == 3 * kEpoch - 24,
            "a ZONE_L behind a torn ordinary zone repairs it");
    }
    {
        Harness h;
        h.feed({sticky(2), word(PP_ZONE_ATOMIC), kNear, 0xffffffffu, word(PP_ZONE_ATOMIC), 4, 8});
        check(h.zone_start(0) == kEpoch - 15 && h.zone_duration(0) == 0xffffffffu, "saturated stall, torn end");
    }
    {
        Harness h;
        h.feed({sticky(2), word(PP_ZONE_ATOMIC), 32, 8});
        h.feed({sticky(3), word(PP_EVENT), kNear, word(PP_ZONE_S), (1u << 16) | 1u});
        check(
            h.order_regressions == 1 && h.event_cycles(0) == 4 * kEpoch - 16 && h.zone_start(1) == 2 * kEpoch + 32 &&
                h.zone_duration(1) == 1,
            "a regression beyond one epoch is not a tear");
    }
    {
        Harness h;
        std::vector<uint32_t> words = {sticky(3)};
        for (unsigned k = 0; k < 16; k++) {
            words.push_back(word(PP_ZONE_ATOMIC));
            words.push_back(k == 7 ? 0xfffffff8u : 100 + k);
            words.push_back(8);
        }
        h.feed(words);
        check(h.zone_start(7) == 3 * kEpoch - 16, "in-block tear across the wrap repairs");
    }
    std::puts("PASS");
}
