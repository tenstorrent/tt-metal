// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Host-only: the decoder's wall-clock epoch repair against hand-built wire, no device.

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "impl/streaming_profiler/streaming_profiler_decode.hpp"

using namespace tt::tt_metal;
namespace kp = kernel_profiler;

constexpr uint64_t M = 1ull << 32;
constexpr uint32_t kNear = 0xfffffff0u;  // a low word just below the wrap

uint32_t word(uint32_t t, uint32_t id = 1) { return (t << PP_TYPE_SHIFT) | id; }
uint32_t sticky(uint32_t hi) { return word(PP_STICKY_TIMER, hi); }

void check(bool v, const char* why) {
    if (!v) {
        std::fprintf(stderr, "FAIL: %s\n", why);
        std::exit(1);
    }
}

struct Harness {
    profiler::SpanDecodeState st;
    // One buffer per record kind; the frames of one batch append to them.
    std::vector<uint64_t> zbuf = std::vector<uint64_t>(64 * 1024), ebuf = zbuf, dbuf = zbuf;
    size_t zones = 0, events = 0, data_bytes = 0;
    streaming_profiler::StreamDecoder dec;
    uint32_t head = 0;

    static constexpr profiler::SpscRecConsts kLanes[5]{};

    // The state slots a frame carries for lane 0.
    uint32_t slot_timer = 0, slot_prog = 0;

    // `warm` feeds one frame holding a zero timer sticky, so a test's own frames find the lane seeded and decode as
    // the wire would after a capture's first frame.
    explicit Harness(bool warm = true) {
        st.reset(1);
        st.core_of_xy[0x30002] = 0;
        dec.st = &st;
        dec.lanes = kLanes;
        if (warm) {
            feed({word(PP_STICKY_TIMER, 0)});
            dec.stats = {};
        }
    }
    // One frame carrying `w` as lane 0's new words.
    void feed(const std::vector<uint32_t>& w) {
        std::vector<uint32_t> f(profiler::kSpscMaxFrameWords);
        f[0] = kp::spsc_span_w0();
        uint32_t* c = f.data() + kp::SPSC_SPAN_PREFIX_WORDS;
        f[kp::SPSC_PREFIX_XY] = 0x30002;
        f[kp::SPSC_PREFIX_HEAD_0] = head;
        c[kp::SPSC_WIRE_TAIL_0] = head + w.size();
        c[kp::SPSC_WIRE_TIMER_0] = slot_timer;
        c[kp::spsc_wire_prog_word(0)] = slot_prog;
        uint32_t off = kp::SPSC_SPAN_PREFIX_WORDS + kp::SPSC_SPAN_WIRE_CTRL_WORDS;
        off += kp::spsc_span_pack_pad(head, off);
        std::copy(w.begin(), w.end(), f.begin() + off);
        f[1] = off + w.size() - kp::SPSC_SPAN_PREFIX_WORDS;
        const auto p = dec.decode_frame(
            f.data(),
            kp::spsc_span_frame_words(f[1]),
            {reinterpret_cast<uint8_t*>(zbuf.data()) + zones * profiler::kSpscRecBytes,
             reinterpret_cast<uint8_t*>(ebuf.data()) + events * profiler::kSpscRecBytes,
             reinterpret_cast<uint8_t*>(dbuf.data()) + data_bytes});
        zones += p.zones;
        events += p.events;
        data_bytes += p.data_bytes;
        head += w.size();
    }
    // The batch so far is delivered: the next one starts over in the buffers.
    void new_batch() {
        dec.commit();
        zones = events = data_bytes = 0;
    }
    static constexpr size_t kQ = profiler::kSpscRecBytes / 8;                                 // qwords per record
    uint64_t zs(size_t k) const { return zbuf[kQ * k]; }                                      // zone k: start, duration
    uint32_t zprog(size_t k) const { return static_cast<uint32_t>(zbuf[kQ * k + 2] >> 32); }  // zone k: runtime id
    uint64_t zd(size_t k) const { return zbuf[kQ * k + 1]; }
    uint64_t ev(size_t k) const { return ebuf[kQ * k]; }  // event k: timestamp
    uint64_t dt(size_t k) const { return dbuf[kQ * k]; }  // the first data record's timestamp (k = 0 only)
    uint64_t pl(size_t k) const { return dbuf[kQ + k]; }  // the first data record's value k
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
        check(h.dec.stats.epoch_fixes == 1 && h.ev(0) == M - 16 && h.ev(1) == M + 4, "torn EVENT, EVENT witness");
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
        check(h.dec.stats.epoch_fixes == 1 && h.dt(0) == M - 16, "torn DATA, zone witness");
        check(h.pl(0) == (0x11ull << 32 | 0x22) && h.pl(1) == (0x33ull << 32 | 0x44), "payload untouched");
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
        check(h.dec.stats.epoch_fixes == 1 && h.zs(0) == M - 24 && h.zd(0) == 8, "torn zone, DATA witness");
    }
    {
        Harness h;
        h.feed({sticky(1), word(PP_EVENT), 100, word(PP_EVENT), 200});
        check(h.dec.stats.epoch_fixes == 0 && h.ev(0) == M + 100, "ordered events untouched");
    }
    {
        Harness h;
        h.feed({sticky(1), word(PP_EVENT), 0x10000000u, word(PP_EVENT), 4});
        check(h.dec.stats.epoch_fixes == 1 && h.ev(0) == 0x10000000u, "regression far from the wrap still repairs");
    }
    {
        Harness h;
        h.feed({sticky(1), word(PP_EVENT), kNear, word(PP_STICKY_PROG, 2), sticky(1), word(PP_EVENT), 4});
        check(h.dec.stats.epoch_fixes == 1 && h.ev(0) == M - 16, "metadata between point and witness");
    }
    {
        Harness h;
        h.feed({sticky(1), word(PP_EVENT), kNear});
        h.feed({word(PP_ZONE_ATOMIC), 4, 8});
        check(h.dec.stats.epoch_fixes == 1 && h.ev(0) == M - 16, "witness in a later frame of the same batch");
    }
    {
        Harness h;
        h.feed({sticky(3), word(PP_EVENT), kNear});
        const uint64_t dur = M + 100;
        h.feed({word(PP_ZONE_L), 4, 3, uint32_t(dur), uint32_t(dur >> 32)});
        check(h.dec.stats.epoch_fixes == 1 && h.ev(0) == 3 * M - 16, "a ZONE_L witnesses a torn point");
    }
    {
        Harness h;
        h.feed({sticky(1), word(PP_EVENT), kNear});
        check(h.dec.stats.epoch_fixes == 0 && h.ev(0) == 2 * M - 16, "uncovered: final point with no witness");
    }
    {
        Harness h;
        h.feed({sticky(1), word(PP_EVENT), kNear});
        h.new_batch();
        h.feed({word(PP_ZONE_ATOMIC), 4, 0});
        check(
            h.dec.stats.epoch_fixes == 0 && h.dec.stats.order_regressions == 1 && h.zs(0) == M + 4,
            "uncovered: target already delivered counts a regression, touches nothing");
    }
    {
        Harness h;
        h.feed({sticky(1), word(PP_ZONE_ATOMIC), kNear, 8});
        h.new_batch();
        h.feed({word(PP_ZONE_ATOMIC), 4, 2});
        h.feed({word(PP_ZONE_ATOMIC), 8, 2});
        check(
            h.dec.stats.epoch_fixes == 0 && h.dec.stats.order_regressions == 1 && h.zs(0) == M + 2,
            "a failed repair does not retarget the next record");
    }
    {
        Harness h;
        h.feed({sticky(2), word(PP_ZONE_ATOMIC), 16, 32});
        check(h.dec.stats.epoch_fixes == 0 && h.zs(0) == 2 * M - 16, "uncovered: torn start with no later record");
    }
    {
        Harness h;
        const uint64_t dur = uint64_t(20) - M;
        h.feed({word(PP_ZONE_L), 4, 1, uint32_t(dur), uint32_t(dur >> 32)});
        check(
            h.dec.stats.epoch_fixes == 1 && h.zs(0) == M - 16 && h.zd(0) == 20, "ZONE_L torn start: negative elapsed");
    }
    {
        Harness h;
        const uint64_t dur = uint64_t(0) - 2 * M;
        h.feed({word(PP_ZONE_L), 4, 3, uint32_t(dur), uint32_t(dur >> 32)});
        check(h.dec.stats.epoch_fixes == 0 && h.zd(0) == dur, "an elapsed time below -2^32 is not a tear");
    }
    {
        Harness h;
        const uint64_t dur = uint64_t(0) - 20;
        h.feed({word(PP_ZONE_L), 4, 0, uint32_t(dur), uint32_t(dur >> 32)});
        check(
            h.dec.stats.epoch_fixes == 0 && h.dec.stats.order_regressions == 1,
            "a repair cannot place a start before zero");
    }
    {
        Harness h;
        const uint64_t dur = M + 100;
        h.feed({sticky(3), word(PP_ZONE_L), kNear, 3, uint32_t(dur), uint32_t(dur >> 32), word(PP_ZONE_ATOMIC), 4, 8});
        check(
            h.dec.stats.epoch_fixes == 1 && h.zd(0) == 100 && h.zs(0) == 3 * M - 116,
            "ZONE_L torn end: the inflated duration moves, the start stays");
    }
    {
        Harness h;
        const uint64_t dur = M + 100;
        h.feed(
            {sticky(3),
             word(PP_ZONE_ATOMIC, profiler::kSpscStallZoneId),
             kNear,
             8,
             word(PP_ZONE_L),
             kNear - 32,
             3,
             uint32_t(dur),
             uint32_t(dur >> 32)});
        check(
            h.dec.stats.epoch_fixes == 0 && h.dec.stats.order_regressions == 1 && h.zs(0) == 4 * M - 24,
            "a ZONE_L behind a later stall zone is ordering, not a tear");
    }
    {
        Harness h;
        h.feed({sticky(3), word(PP_ZONE_ATOMIC, 1), kNear, 8, word(PP_ZONE_L), kNear + 8, 2, 100, 0});
        check(
            h.dec.stats.epoch_fixes == 1 && h.dec.stats.order_regressions == 0 && h.zs(0) == 3 * M - 24,
            "a ZONE_L behind a torn ordinary zone repairs it");
    }
    {
        Harness h;
        h.feed({sticky(2), word(PP_ZONE_ATOMIC), kNear, 0xffffffffu, word(PP_ZONE_ATOMIC), 4, 8});
        check(h.dec.stats.epoch_fixes == 1 && h.zs(0) == M - 15 && h.zd(0) == 0xffffffffu, "saturated stall, torn end");
    }
    {
        Harness h;
        h.feed({sticky(2), word(PP_ZONE_ATOMIC), 32, 8});
        h.feed({sticky(3), word(PP_EVENT), kNear, word(PP_ZONE_S), (1u << 16) | 1u});
        check(
            h.dec.stats.epoch_fixes == 0 && h.dec.stats.order_regressions == 1,
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
        check(h.dec.stats.epoch_fixes == 1 && h.zs(7) == 3 * M - 16, "in-block tear across the wrap repairs");
    }
    std::puts("PASS");
}
