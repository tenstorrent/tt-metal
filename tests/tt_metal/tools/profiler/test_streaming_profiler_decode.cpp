// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Host-only: the decoder's wall-clock epoch repair against hand-built wire, no device.

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

#include "impl/streaming_profiler/decode.hpp"

using namespace tt::tt_metal;
namespace kp = kernel_profiler;
namespace api = experimental::streaming_profiler;

using ZoneBatch = api::Batch<api::RecordType::Zones>;
static_assert(api::BatchCallable<decltype([](const ZoneBatch&) {})>);
static_assert(api::BatchCallable<decltype([](ZoneBatch) {})>);
static_assert(api::BatchCallable<std::reference_wrapper<decltype([](const ZoneBatch&) {})>>);
static_assert(!api::BatchCallable<decltype([](const auto&) {})>);
static_assert(!api::BatchCallable<decltype([](int) {})>);
static_assert(!api::BatchCallable<decltype([owned = std::unique_ptr<int>()](const ZoneBatch&) {})>);

constexpr uint64_t M = 1ull << 32;
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
        .lanes = std::vector<api::Core>(profiler::kSpscNRiscDecode), .tiles = {{.xy = kTileXy}}};
    dev.core_of_xy[kTileXy] = 0;
    return dev;
}

struct Harness {
    // One buffer per record kind; the frames of one batch append to them.
    std::vector<uint64_t> zbuf = std::vector<uint64_t>(64 * 1024), ebuf = zbuf, dbuf = zbuf, vbuf = zbuf;
    size_t zones = 0, events = 0, data_records = 0, value_count = 0;
    uint64_t order_regressions = 0;
    int64_t newest_ticks = 0;  // the last feed's
    uint32_t parked = 0;       // feeds decoded and not yet delivered
    streaming_profiler::CaptureContext::Device dev = one_tile_device();
    streaming_profiler::StreamDecoder dec{dev};
    uint32_t head = 0;
    const api::TimestampedData* first_data = nullptr;

    // The state slots a frame carries for lane 0.
    uint32_t slot_timer = 0, slot_prog = 0;

    // `warm` feeds one frame holding a zero timer sticky, so a test's own frames find the lane seeded and decode as
    // the wire would after a capture's first frame.
    explicit Harness(bool warm = true) {
        if (warm) {
            feed({word(PP_STICKY_TIMER, 0)});
        }
    }
    // One frame carrying `w` as lane 0's new words.
    void feed(const std::vector<uint32_t>& w) {
        std::vector<uint32_t> f(profiler::kSpscMaxFrameWords);
        f[0] = kp::spsc_span_w0();
        uint32_t* c = f.data() + kp::SPSC_SPAN_PREFIX_WORDS;
        f[kp::SPSC_PREFIX_XY] = kTileXy;
        f[kp::SPSC_PREFIX_HEAD_0] = head;
        c[kp::SPSC_WIRE_TAIL_0] = head + w.size();
        c[kp::SPSC_WIRE_TIMER_0] = slot_timer;
        c[kp::spsc_wire_prog_word(0)] = slot_prog;
        uint32_t off = kp::SPSC_SPAN_PREFIX_WORDS + kp::SPSC_SPAN_WIRE_CTRL_WORDS;
        off += kp::spsc_span_pack_pad(head, off);
        std::copy(w.begin(), w.end(), f.begin() + off);
        f[1] = off + w.size() - kp::SPSC_SPAN_PREFIX_WORDS;
        const uint32_t frame_words = kp::spsc_span_frame_words(f[1]);
        const auto decoded = dec.decode_frames(
            f.data(),
            {&frame_words, 1},
            {reinterpret_cast<uint8_t*>(zbuf.data()) + zones * profiler::kSpscZoneBytes,
             reinterpret_cast<uint8_t*>(ebuf.data()) + events * profiler::kSpscEventBytes,
             reinterpret_cast<uint8_t*>(dbuf.data()) + data_records * profiler::kSpscDataBytes,
             reinterpret_cast<uint8_t*>(vbuf.data() + value_count)});
        zones += decoded.zones;
        events += decoded.events;
        data_records += decoded.data;
        value_count += decoded.values;
        order_regressions += decoded.order_regressions;
        newest_ticks = decoded.newest_ticks;
        head += w.size();
        parked++;
    }
    // Each feed is a batch the service parks; it delivers the oldest once the cover passes it.
    void deliver_oldest() {
        dec.commit();
        parked--;
    }
    // Every feed so far is delivered: the next one starts over in the buffers.
    void new_batch() {
        while (parked != 0) {
            deliver_oldest();
        }
        zones = events = data_records = value_count = 0;
        first_data = nullptr;
    }
    const api::Zone& zone(size_t k) const { return reinterpret_cast<const api::Zone*>(zbuf.data())[k]; }
    uint64_t zs(size_t k) const { return zone(k).start_device_cycles(); }
    uint32_t zprog(size_t k) const { return zone(k).runtime_id(); }
    uint64_t zd(size_t k) const { return zone(k).end_device_cycles() - zone(k).start_device_cycles(); }
    uint64_t ev(size_t k) const { return reinterpret_cast<const api::Event*>(ebuf.data())[k].device_cycles(); }
    // Built once, as the service does at delivery: a later feed may still repair the decoder's bytes before then.
    const api::TimestampedData& data() {
        if (first_data == nullptr) {
            streaming_profiler::construct_timestamped_data(reinterpret_cast<uint8_t*>(dbuf.data()), 0);
            first_data = std::launder(reinterpret_cast<const api::TimestampedData*>(dbuf.data()));
        }
        return *first_data;
    }
    uint64_t dt() { return data().device_cycles(); }
    uint64_t pl(size_t k) { return data().payload()[k]; }
};

int main() {
    // Lane state from a frame's slots.
    {
        Harness h(false);
        h.slot_timer = 5;
        h.slot_prog = 7;
        h.feed({word(PP_ZONE_ATOMIC), 40, 8});
        check(
            h.zones == 1 && h.zs(0) == 5 * M + 40 - 8 && h.zprog(0) == 7,
            "a lane's first frame seeds timer and runtime id from its slots");
    }
    {
        Harness h(false);
        h.slot_timer = 5;
        h.feed({word(PP_ZONE_ATOMIC), 4, 1, sticky(9), word(PP_ZONE_ATOMIC), 8, 1});
        check(
            h.zones == 1 && h.zs(0) == 9 * M + 8 - 1,
            "a reseeding run is taken up from its last sticky");
    }
    {
        Harness h(false);
        h.slot_timer = 2;
        h.feed({word(PP_ZONE_S), (5u << 16) | 1u, word(PP_ZONE_ATOMIC), 100, 8, word(PP_ZONE_S), (5u << 16) | 1u});
        check(
            h.zones == 2 && h.zs(0) == 2 * M + 100 - 8 &&
                h.zs(1) == 2 * M + 105 - 1,
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
            h.zones == 2 && h.zs(1) == 3 * M + 64 - 8 &&
                h.zprog(1) == 9,
            "after a gap the next frame's slots reseed the lane");
    }
    {
        Harness h;
        h.feed({sticky(1), word(PP_EVENT), kNear, word(PP_EVENT), 4});
        check(h.ev(0) == M - 16 && h.ev(1) == M + 4, "torn EVENT, EVENT witness");
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
        check(h.dt() == M - 16, "torn DATA, zone witness");
        check(h.pl(0) == (0x11ull << 32 | 0x22) && h.pl(1) == (0x33ull << 32 | 0x44), "payload untouched");
    }
    {
        Harness h;
        h.feed({sticky(1), word(PP_DATA), 4, 10u << PP_DATA_SIZE_SHIFT, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
        const api::TimestampedData kept = h.data();
        std::ranges::fill(h.dbuf, 0);
        std::ranges::fill(h.vbuf, 0);
        check(
            kept.device_cycles() == M + 4 && kept.payload().size() == 5 && kept.payload()[0] == (1ull << 32 | 2) &&
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
        check(h.zs(0) == M - 24 && h.zd(0) == 8, "torn zone, DATA witness");
    }
    {
        Harness h;
        h.feed({sticky(1), word(PP_EVENT), 100, word(PP_EVENT), 200});
        check(h.ev(0) == M + 100, "ordered events untouched");
    }
    {
        Harness h;
        h.feed({sticky(1), word(PP_EVENT), 0x10000000u, word(PP_EVENT), 4});
        check(h.ev(0) == 0x10000000u, "regression far from the wrap still repairs");
    }
    {
        Harness h;
        h.feed({sticky(1), word(PP_EVENT), kNear, word(PP_STICKY_PROG, 2), sticky(1), word(PP_EVENT), 4});
        check(h.ev(0) == M - 16, "metadata between point and witness");
    }
    {
        Harness h;
        h.feed({sticky(1), word(PP_EVENT), kNear});
        h.feed({word(PP_ZONE_ATOMIC), 4, 8});
        check(h.ev(0) == M - 16, "witness in a later frame of the same batch");
    }
    {
        Harness h;
        h.feed({sticky(3), word(PP_EVENT), kNear});
        const uint64_t dur = M + 100;
        h.feed({word(PP_ZONE_L), 4, 3, uint32_t(dur), uint32_t(dur >> 32)});
        check(h.ev(0) == 3 * M - 16, "a ZONE_L witnesses a torn point");
    }
    {
        Harness h;
        h.feed({sticky(1), word(PP_EVENT), kNear});
        check(h.ev(0) == 2 * M - 16, "uncovered: final point with no witness");
    }
    {
        Harness h;
        h.feed({sticky(1), word(PP_EVENT), kNear});
        h.new_batch();
        h.feed({word(PP_ZONE_ATOMIC), 4, 0});
        check(
            h.order_regressions == 1 && h.zs(0) == M + 4,
            "uncovered: target already delivered counts a regression, touches nothing");
    }
    {
        Harness h;
        h.feed({sticky(1), word(PP_ZONE_ATOMIC), kNear, 8});
        h.new_batch();
        h.feed({word(PP_ZONE_ATOMIC), 4, 2});
        h.feed({word(PP_ZONE_ATOMIC), 8, 2});
        check(h.order_regressions == 1 && h.zs(0) == M + 2, "a failed repair does not retarget the next record");
    }
    {
        Harness h;
        h.feed({word(PP_EVENT), 100});
        h.feed({sticky(1), word(PP_EVENT), kNear});
        h.deliver_oldest();
        h.deliver_oldest();
        h.feed({word(PP_ZONE_ATOMIC), 4, 0});
        check(
            h.order_regressions == 0 && h.ev(0) == 100 && h.ev(1) == M - 16,
            "a repair lands in a batch still parked behind delivered ones");
    }
    {
        Harness h;
        h.feed({sticky(1), word(PP_EVENT), 100});
        check(h.newest_ticks == static_cast<int64_t>(M + 100), "a lane's first record counts toward the cover");
        h.feed({sticky(2), word(PP_EVENT), kNear});
        check(
            h.newest_ticks == static_cast<int64_t>(2 * M - 16),
            "a torn last record holds its batch to the time it would repair to");
    }
    {
        Harness h;
        h.feed({sticky(2), word(PP_ZONE_ATOMIC), 16, 32});
        check(h.zs(0) == 2 * M - 16, "uncovered: torn start with no later record");
    }
    {
        Harness h;
        const uint64_t dur = uint64_t(20) - M;
        h.feed({word(PP_ZONE_L), 4, 1, uint32_t(dur), uint32_t(dur >> 32)});
        check(h.zs(0) == M - 16 && h.zd(0) == 20, "ZONE_L torn start: negative elapsed");
    }
    {
        Harness h;
        const uint64_t dur = uint64_t(0) - 2 * M;
        h.feed({word(PP_ZONE_L), 4, 3, uint32_t(dur), uint32_t(dur >> 32)});
        check(h.zd(0) == dur, "an elapsed time below -2^32 is not a tear");
    }
    {
        Harness h;
        const uint64_t dur = uint64_t(0) - 20;
        h.feed({word(PP_ZONE_L), 4, 0, uint32_t(dur), uint32_t(dur >> 32)});
        check(h.order_regressions == 1 && h.zs(0) == 24 && h.zd(0) == dur, "a repair cannot place a start before zero");
    }
    {
        Harness h;
        const uint64_t dur = M + 100;
        h.feed({sticky(3), word(PP_ZONE_L), kNear, 3, uint32_t(dur), uint32_t(dur >> 32), word(PP_ZONE_ATOMIC), 4, 8});
        check(
            h.zd(0) == 100 && h.zs(0) == 3 * M - 116, "ZONE_L torn end: the inflated duration moves, the start stays");
    }
    {
        Harness h;
        h.feed({sticky(3), word(PP_ZONE_ATOMIC, 1), kNear, 8, word(PP_ZONE_L), kNear + 8, 2, 100, 0});
        check(h.order_regressions == 0 && h.zs(0) == 3 * M - 24, "a ZONE_L behind a torn ordinary zone repairs it");
    }
    {
        Harness h;
        h.feed({sticky(2), word(PP_ZONE_ATOMIC), kNear, 0xffffffffu, word(PP_ZONE_ATOMIC), 4, 8});
        check(h.zs(0) == M - 15 && h.zd(0) == 0xffffffffu, "saturated stall, torn end");
    }
    {
        Harness h;
        h.feed({sticky(2), word(PP_ZONE_ATOMIC), 32, 8});
        h.feed({sticky(3), word(PP_EVENT), kNear, word(PP_ZONE_S), (1u << 16) | 1u});
        check(
            h.order_regressions == 1 && h.ev(0) == 4 * M - 16 && h.zs(1) == 2 * M + 32 && h.zd(1) == 1,
            "a regression beyond one epoch is not a tear");
    }
    {
        Harness h;
        std::vector<uint32_t> w = {sticky(3)};
        for (unsigned k = 0; k < 16; k++) {
            w.push_back(word(PP_ZONE_ATOMIC));
            w.push_back(k == 7 ? 0xfffffff8u : 100 + k);
            w.push_back(8);
        }
        h.feed(w);
        check(h.zs(7) == 3 * M - 16, "in-block tear across the wrap repairs");
    }
    std::puts("PASS");
}
