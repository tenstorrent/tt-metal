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
    std::vector<uint64_t> buf = std::vector<uint64_t>(2048 * 4);
    streaming_profiler::StreamDecoder dec;
    uint32_t head = 0;

    Harness() {
        st.reset(1);
        st.core_of_xy[0x30002] = 0;
        dec.st = &st;
        dec.sink.buf = reinterpret_cast<uint8_t*>(buf.data());
    }
    // One frame carrying `w` as lane 0's new words.
    void feed(const std::vector<uint32_t>& w) {
        std::vector<uint32_t> f(profiler::kSpscMaxFrameWords);
        f[0] = kp::spsc_span_w0();
        uint32_t* c = f.data() + kp::SPSC_SPAN_PREFIX_WORDS;
        c[kp::SPSC_WIRE_XY] = 0x30002;
        c[kp::SPSC_WIRE_HEAD_0] = head;
        c[kp::SPSC_WIRE_TAIL_0] = head + w.size();
        uint32_t off = kp::SPSC_SPAN_PREFIX_WORDS + kp::SPSC_SPAN_WIRE_CTRL_WORDS;
        off += kp::spsc_span_pack_pad(head, off);
        std::copy(w.begin(), w.end(), f.begin() + off);
        f[1] = off + w.size() - kp::SPSC_SPAN_PREFIX_WORDS;
        dec.decode_frame(f.data(), kp::spsc_span_frame_words(f[1]));
        check(dec.stats.anomalies == 0, "wire geometry");
        head += w.size();
    }
    void new_batch() { dec.end_batch(); }
    uint64_t start(size_t rec) const { return buf[4 * rec]; }
    uint64_t dur(size_t rec) const { return buf[4 * rec + 1]; }
};

int main() {
    {
        Harness h;
        h.feed({sticky(1), word(PP_EVENT), kNear, word(PP_EVENT), 4});
        check(h.dec.stats.epoch_fixes == 1 && h.start(0) == M - 16 && h.start(1) == M + 4, "torn EVENT, EVENT witness");
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
        check(h.dec.stats.epoch_fixes == 1 && h.start(0) == M - 16, "torn DATA, zone witness");
        check(h.start(1) == (0x11ull << 32 | 0x22) && h.start(2) == (0x33ull << 32 | 0x44), "payload untouched");
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
        check(h.dec.stats.epoch_fixes == 1 && h.start(0) == M - 24 && h.dur(0) == 8, "torn zone, DATA witness");
    }
    {
        Harness h;
        h.feed({sticky(1), word(PP_EVENT), 100, word(PP_EVENT), 200});
        check(h.dec.stats.epoch_fixes == 0 && h.start(0) == M + 100, "ordered events untouched");
    }
    {
        Harness h;
        h.feed({sticky(1), word(PP_EVENT), 0x10000000u, word(PP_EVENT), 4});
        check(h.dec.stats.epoch_fixes == 1 && h.start(0) == 0x10000000u, "regression far from the wrap still repairs");
    }
    {
        Harness h;
        h.feed({sticky(1), word(PP_EVENT), kNear, word(PP_STICKY_PROG, 2), sticky(1), word(PP_EVENT), 4});
        check(h.dec.stats.epoch_fixes == 1 && h.start(0) == M - 16, "metadata between point and witness");
    }
    {
        Harness h;
        h.feed({sticky(1), word(PP_EVENT), kNear});
        h.feed({word(PP_ZONE_ATOMIC), 4, 8});
        check(h.dec.stats.epoch_fixes == 1 && h.start(0) == M - 16, "witness in a later frame of the same batch");
    }
    {
        Harness h;
        h.feed({sticky(3), word(PP_EVENT), kNear});
        const uint64_t dur = M + 100;
        h.feed({word(PP_ZONE_L), 4, 3, uint32_t(dur), uint32_t(dur >> 32)});
        check(h.dec.stats.epoch_fixes == 1 && h.start(0) == 3 * M - 16, "a ZONE_L witnesses a torn point");
    }
    {
        Harness h;
        h.feed({sticky(1), word(PP_EVENT), kNear});
        check(h.dec.stats.epoch_fixes == 0 && h.start(0) == 2 * M - 16, "uncovered: final point with no witness");
    }
    {
        Harness h;
        h.feed({sticky(1), word(PP_EVENT), kNear});
        h.new_batch();
        h.feed({word(PP_ZONE_ATOMIC), 4, 0});
        check(
            h.dec.stats.epoch_fixes == 0 && h.dec.stats.order_regressions == 1 && h.start(0) == M + 4,
            "uncovered: target already delivered counts a regression, touches nothing");
    }
    {
        Harness h;
        h.feed({sticky(1), word(PP_ZONE_ATOMIC), kNear, 8});
        h.new_batch();
        h.feed({word(PP_ZONE_ATOMIC), 4, 2});
        h.feed({word(PP_ZONE_ATOMIC), 8, 2});
        check(
            h.dec.stats.epoch_fixes == 0 && h.dec.stats.order_regressions == 1 && h.start(0) == M + 2,
            "a failed repair does not retarget the next record");
    }
    {
        Harness h;
        h.feed({sticky(2), word(PP_ZONE_ATOMIC), 16, 32});
        check(h.dec.stats.epoch_fixes == 0 && h.start(0) == 2 * M - 16, "uncovered: torn start with no later record");
    }
    {
        Harness h;
        const uint64_t dur = uint64_t(20) - M;
        h.feed({word(PP_ZONE_L), 4, 1, uint32_t(dur), uint32_t(dur >> 32)});
        check(
            h.dec.stats.epoch_fixes == 1 && h.start(0) == M - 16 && h.dur(0) == 20,
            "ZONE_L torn start: negative elapsed");
    }
    {
        Harness h;
        const uint64_t dur = uint64_t(0) - 2 * M;
        h.feed({word(PP_ZONE_L), 4, 3, uint32_t(dur), uint32_t(dur >> 32)});
        check(h.dec.stats.epoch_fixes == 0 && h.dur(0) == dur, "an elapsed time below -2^32 is not a tear");
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
            h.dec.stats.epoch_fixes == 1 && h.dur(0) == 100 && h.start(0) == 3 * M - 116,
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
            h.dec.stats.epoch_fixes == 0 && h.dec.stats.order_regressions == 1 && h.start(0) == 4 * M - 24,
            "a ZONE_L behind a later stall zone is ordering, not a tear");
    }
    {
        Harness h;
        h.feed({sticky(3), word(PP_ZONE_ATOMIC, 1), kNear, 8, word(PP_ZONE_L), kNear + 8, 2, 100, 0});
        check(
            h.dec.stats.epoch_fixes == 1 && h.dec.stats.order_regressions == 0 && h.start(0) == 3 * M - 24,
            "a ZONE_L behind a torn ordinary zone repairs it");
    }
    {
        Harness h;
        h.feed({sticky(2), word(PP_ZONE_ATOMIC), kNear, 0xffffffffu, word(PP_ZONE_ATOMIC), 4, 8});
        check(
            h.dec.stats.epoch_fixes == 1 && h.start(0) == M - 15 && h.dur(0) == 0xffffffffu,
            "saturated stall, torn end");
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
        check(h.dec.stats.epoch_fixes == 1 && h.start(7) == 3 * M - 16, "in-block tear across the wrap repairs");
    }
    std::puts("PASS");
}
