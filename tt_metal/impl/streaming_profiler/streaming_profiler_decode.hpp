// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// A per-stream decoder that walks BULK_SPAN frames through the vector kernels into public records and repairs the
// wall-clock latch race.

#include <algorithm>
#include <cstdint>
#include <cstring>

#include <tt_stl/assert.hpp>

#include "impl/streaming_profiler/spsc_marker_decode.hpp"
#include "impl/streaming_profiler/streaming_profiler_consumer.hpp"

namespace tt::tt_metal::streaming_profiler {

// Every packet that becomes a record is at least two words, so a frame's payload decodes to at most half as many.
inline constexpr size_t kMaxFrameRecs = profiler::kSpscMaxPayloadWords / 2;
static_assert(std::ranges::all_of(profiler::kFormats, [](const profiler::PacketFormat& f) {
    return f.kind == profiler::Kind::Sticky || f.words >= 2;
}));

inline constexpr uint64_t kEpoch = 1ull << 32;

// A lane's timestamp went backwards. Lanes emit in end order and a wall-clock read is either right or exactly 2^32
// high (kernel_profiler_streaming.hpp read_wall_clock), so the previous record borrowed the next epoch and this one
// proves it. An S/ATOMIC start is derived from its end and a point is its timestamp, so those move down whole; a
// ZONE_L's start was read separately, so only its inflated duration moves. `rec_start` is the record to repair,
// nullptr when it is not at hand; `zone` says the record is one (a point's second qword is its value count, not a
// duration). False is an order regression the decoder only counts.
// noinline: inlined at its sites, the repair body inflated the walk's register pressure (+6% on the point path).
__attribute__((noinline)) inline bool repair_prev_record(uint8_t* rec_start, bool zone, uint64_t prev, uint64_t ts) {
    if (prev < kEpoch || prev - ts > kEpoch || rec_start == nullptr) {
        return false;
    }
    uint64_t* rec = reinterpret_cast<uint64_t*>(rec_start);
    if (zone && rec[1] >= kEpoch) {
        rec[1] -= kEpoch;
    } else if (rec[0] >= kEpoch) {
        rec[0] -= kEpoch;
    } else {
        return false;
    }
    return true;
}

struct Repairs {
    uint32_t fixes = 0, regressions = 0;
};

// A format whose duration is read in two halves (dur_hi present) reads its start separately from its end, so its
// duration is two reads apart: one that borrowed the next epoch shows as an elapsed time in [-2^32, -1] and moves
// down whole. Its end may legitimately precede the previous record's when that record is the stall zone raised by
// this zone's own ring reservation. `lane_ts` / `lane_rec` are the lane's last record before this run of `n`, whose
// first record sits at `first`; `lane_rec_zone` says whether that record is a zone.
template <profiler::PacketFormat F>
__attribute__((noinline)) inline Repairs zone_repairs64(
    const uint32_t* src,
    uint32_t n,
    uint8_t* first,
    bool wrapped,
    bool back,
    uint64_t lane_ts,
    uint8_t* lane_rec,
    bool lane_rec_zone) {
    static_assert(F.has_dur_hi());
    constexpr uint32_t kRec = profiler::kSpscRecBytes;
    Repairs out;
    const auto rec_at = [](uint8_t* at) { return reinterpret_cast<uint64_t*>(at); };
    const auto end_of = [src](uint32_t k) { return profiler::spsc_ts_at<F>(src, k, 0); };
    if (wrapped) {
        for (uint32_t k = 0; k < n; k++) {
            const uint32_t* r = src + F.words * k;
            if (r[F.dur_hi] != 0xFFFFFFFFu) {
                continue;
            }
            const uint64_t end = end_of(k);
            const uint64_t dur = (static_cast<uint64_t>(r[F.dur_hi]) << 32) | r[F.dur_lo];
            if (end >= dur + kEpoch) {
                out.fixes++;
                uint64_t* rec = rec_at(first + kRec * k);
                rec[1] = dur + kEpoch;
                rec[0] = end - rec[1];
            } else {
                out.regressions++;
            }
        }
    }
    if (back) {
        for (uint32_t k = 0; k < n; k++) {
            const uint64_t end = end_of(k);
            const uint64_t prev_ts = k == 0 ? lane_ts : end_of(k - 1);
            if (!(end < prev_ts)) {
                continue;
            }
            uint8_t* const prev_rec = k == 0 ? lane_rec : first + kRec * (k - 1);
            bool after_stall = true;
            if (prev_rec != nullptr) {
                const uint64_t* prev = rec_at(prev_rec);
                after_stall = (static_cast<uint32_t>(prev[2]) & 0x07FFFFFFu) == profiler::kSpscStallZoneId;
            }
            if (after_stall) {
                out.regressions++;
            } else {
                const bool fixed = repair_prev_record(prev_rec, k == 0 ? lane_rec_zone : true, prev_ts, end);
                out.fixes += fixed;
                out.regressions += !fixed;
            }
        }
    }
    return out;
}

// What one frame can decode to, and so what a caller reserves per kind before decoding it: kMaxFrameRecs records
// plus the whole quads the block kernels write past their last record, and for Data the values that follow the
// records (kSpscMaxPayloadWords words, two per value) plus the 32 bytes a point kernel stores past a record before
// it knows they count.
inline constexpr size_t kFrameRecsReserve = kMaxFrameRecs + profiler::kSpscSinkSlackRecs;
inline constexpr size_t kFrameDataBytesReserve =
    kFrameRecsReserve * profiler::kSpscRecBytes + profiler::kSpscMaxPayloadWords * 4 + 32;

// One stream's decode, owned by one thread; the wire-integrity totals are its owner's to report.
// A decoded local-refclk sample from the idle-eth clock tracker (PP_CLOCK): chip, producing lane, clock kind
// (CLOCK_LOCAL_REFCLK today), the 24-bit refclk value, and the full wall timestamp. The host fit unwraps value24
// against ts. Routed to StreamDecoder::clock_fn at decode; never becomes a record.
struct ClockSample {
    uint32_t dev;
    uint32_t lane;
    uint32_t kind;
    uint32_t value24;
    uint64_t ts;
};

struct StreamDecoder {
    profiler::SpanDecodeState* st = nullptr;
    uint64_t batch_seq = 0;  // a lane's last-record pointer is only meaningful within its own batch
    const profiler::SpscRecConsts* lanes = nullptr;  // per lane: what its records carry besides the packet's words
    StreamStats stats;
    uint64_t stall_zones = 0;
    // Idle-eth PP_CLOCK samples are routed here at decode, never delivered as records. null clock_fn = drop.
    uint32_t dev = 0;  // index into capture_context().devices, stamped on each ClockSample
    void* clock_ctx = nullptr;
    void (*clock_fn)(void*, const ClockSample&) = nullptr;

    // Where a frame's records go: kFrameRecsReserve records of room for zones and for events, kFrameDataBytesReserve
    // bytes for timestamped data.
    struct Out {
        uint8_t* zones;
        uint8_t* events;
        uint8_t* data;
    };
    struct Produced {
        uint32_t zones, events;
        uint32_t data_bytes;
    };
    Produced decode_frame(const uint32_t* frame, uint32_t frame_words, Out out);

    // After the records so far are published to readers: a lane's last record may no longer be repaired in place.
    void commit() { batch_seq++; }
};

// Decodes one packed BULK_SPAN frame in place. Every packet is decoded through its kFormats row: a run of fixed-size
// records through spsc_block, a lone one through spsc_one, a point through spsc_point, a sticky into the lane's
// state. Decode starts at the larger of the head mirror and the extent's start: the mirror runs behind after an
// upstream loss (adopt), the extent after a lagging head write-back (skip the overlap). The walk bounds
// every read by the frame and every packet by its run; a word it cannot decode is a fault.
inline StreamDecoder::Produced StreamDecoder::decode_frame(const uint32_t* frame, uint32_t frame_words, Out out) {
    namespace kp = kernel_profiler;
    using namespace profiler;
    SpanDecodeState& S = *st;
    const uint32_t* ctrl = frame + kp::SPSC_SPAN_PREFIX_WORDS;
    const uint32_t core = S.core_of_xy.find(frame[kp::SPSC_PREFIX_XY]);
    TT_FATAL(
        core != CoreTable::kNone,
        "streaming profiler: frame from NoC core {:#x}, which the capture did not seed",
        frame[kp::SPSC_PREFIX_XY]);
    // Raw locals for everything the hot walk touches: the kernels store through byte pointers, and a member reached
    // through `this` would be reloaded after every such store.
    const SpscRecConsts* const lane_consts = lanes;
    const uint64_t seq = batch_seq;
    void (*const clock_fn)(void*, const ClockSample&) = this->clock_fn;
    void* const clock_ctx = this->clock_ctx;
    const uint32_t clock_dev = this->dev;
    uint8_t* const zb = out.zones;
    uint8_t* const eb = out.events;
    uint8_t* const db = out.data;
    uint64_t zoff = 0;
    // The two point offsets share one register, data's in the high half: an array indexed by the packet kind would
    // live in memory and put every update on a store-to-load chain.
    uint64_t pt_off = 0;
    uint64_t zm = 0, sz = 0, oreg = 0, rc = 0, fixes = 0;
    uint64_t lane_ts = 0;
    uint8_t* lane_rec = nullptr;
    bool lane_rec_zone = false;
    // A run's first record against the lane's last, then the run's last becomes the lane's; a single record passes
    // its timestamp twice.
    const auto order = [&](uint64_t ts_first, uint64_t ts_last) {
        if (__builtin_expect(ts_first < lane_ts, 0)) {
            const bool fixed = repair_prev_record(lane_rec, lane_rec_zone, lane_ts, ts_first);
            fixes += fixed;
            oreg += !fixed;
        }
        lane_ts = ts_last;
    };
    const auto zones_emitted = [&](uint32_t n) {
        zm += n;
        rc += n;
        zoff += kSpscRecBytes * n;
        lane_rec = zb + zoff - kSpscRecBytes;
        lane_rec_zone = true;
    };

    uint32_t off = kp::SPSC_SPAN_PREFIX_WORDS + kp::SPSC_SPAN_WIRE_CTRL_WORDS;
    SpscLaneConsts lc;
    for (uint32_t r = 0; r < kSpscNRiscDecode; r++) {
        const uint32_t lane = core * kSpscNRiscDecode + r;
        SpscLane& L = S.lanes[lane];
        const uint32_t tail = ctrl[kp::SPSC_WIRE_TAIL_0 + r];
        const uint32_t start = frame[kp::SPSC_PREFIX_HEAD_0 + r];
        const uint32_t extent = tail - start;
        const uint32_t* p = nullptr;
        // A near-full wrapping run arrives as the whole ring image (predicate shared with the device,
        // spsc_span_wrap_image), the pad is phased for ring offset 0, and the payload advance is the full ring.
        const bool ring_ordered = extent != 0 && kp::spsc_span_wrap_image(start, extent, kSpscRingCap);
        if (extent != 0) {
            off += kp::spsc_span_pack_pad(ring_ordered ? 0u : start, off);
            p = frame + off;
            off += ring_ordered ? kSpscRingCap : extent;
            TT_FATAL(
                off <= frame_words,
                "streaming profiler: frame control block places lane {} {} words past the frame's {}",
                lane,
                off - frame_words,
                frame_words);
        }
        uint32_t head;
        if (L.seeded == 0) {
            L.seeded = 1;
            head = start;
        } else {
            head = L.head;
            if (static_cast<int32_t>(start - head) > 0) {
                head = start;
                L.need_state = 1;
            }
        }
        L.head = tail;
        const uint32_t run = tail - head;
        if (run == 0) {
            continue;
        }
        uint32_t th = L.timer_hi;
        uint32_t pg = L.prog;
        uint64_t cur = L.cursor;
        uint32_t lin[kSpscRingCap];
        if (ring_ordered) {
            const uint32_t hm = head & kSpscRingMask;
            const uint32_t first = kSpscRingCap - hm < run ? kSpscRingCap - hm : run;
            std::memcpy(lin, p + hm, first * sizeof(uint32_t));
            if (first < run) {
                std::memcpy(lin + first, p, (run - first) * sizeof(uint32_t));
            }
            p = lin;
        } else {
            p += extent - run;
        }
        // Loads may run to here (never emits): the frame's end, or the run's when the run was linearised (`lin` is
        // not comparable with the frame pointer).
        const uint32_t* const rd_end = ring_ordered ? p + run : frame + frame_words;
        uint32_t i = 0;
        uint32_t na = L.need_anchor;
        if (L.need_state) {
            // The slots hold the state at the frame's tail; a sticky inside the run means the words before it ran on
            // an earlier value nothing here records, so the run is taken up from its last sticky.
            th = ctrl[kp::SPSC_WIRE_TIMER_0 + r];
            pg = ctrl[kp::spsc_wire_prog_word(r)];
            uint32_t k = 0;
            while (k < run) {
                const uint32_t t = pp_type(p[k]);
                uint32_t w = kSpscWordsOfType[t];
                if (w == 0) {
                    break;
                }
                if (kSpscDataMaskOfType[t] != 0) {
                    if (k + kSpscDataFormat.size_word >= run) {
                        break;
                    }
                    w += (p[k + kSpscDataFormat.size_word] >> kSpscDataFormat.size_shift) & kSpscDataFormat.size_mask;
                }
                if (t == PP_STICKY_TIMER) {
                    th = pp_low27(p[k]);
                    i = k + w;
                } else if (t == PP_STICKY_PROG) {
                    pg = pp_low27(p[k]);
                    i = k + w;
                } else if (t == PP_STICKY_PROG_EXT) {
                    if (k + 1 < run) {
                        pg = p[k + 1];
                    }
                    i = k + w;
                }
                k += w;
            }
            L.need_state = 0;
            na = 1;
        }
        spsc_lane_consts(lc, lane_consts[lane], th, pg);
        lane_ts = L.last_ts;
        lane_rec = L.last_rec_seq == seq ? L.last_rec : nullptr;
        lane_rec_zone = L.last_rec_zone != 0;
        while (i < run) {
            const uint32_t* const src = p + i;
            const uint32_t t = pp_type(src[0]);
            const uint32_t readable = static_cast<uint32_t>(rd_end - src);
            const uint32_t left = run - i;
            uint32_t got = 0;
            // A block kernel flagged a step back somewhere in its run: find each one, record k against record k-1.
            const auto block_regress = [&]<PacketFormat F>(uint32_t n, uint8_t* first) __attribute__((always_inline)) {
                for (uint32_t k = 1; k < n; k++) {
                    const uint64_t a = spsc_ts_at<F>(src, k - 1u, lc.th_hi), b = spsc_ts_at<F>(src, k, lc.th_hi);
                    if (b < a) {
                        const bool fixed =
                            repair_prev_record(first + kSpscRecBytes * (k - 1), F.kind == Kind::Zone, a, b);
                        fixes += fixed;
                        oreg += !fixed;
                    }
                }
            };
            const auto repairs64 = [&]<PacketFormat F>(uint32_t n, uint8_t* first, bool wrapped, bool back)
                                       __attribute__((always_inline)) {
                                           const Repairs x = zone_repairs64<F>(
                                               src, n, first, wrapped, back, lane_ts, lane_rec, lane_rec_zone);
                                           fixes += x.fixes;
                                           oreg += x.regressions;
                                       };
            if (t == PP_CLOCK) {
                // 2-word local-refclk sample: word0 = type|kind<<shift|value24, word1 = wall_lo; full wall =
                // this lane s sticky-timer hi | wall_lo. Route to the clock sink, produce no record, advance 2.
                if (left >= 2u) {
                    if (clock_fn) {
                        const uint32_t low27 = pp_low27(src[0]);
                        clock_fn(
                            clock_ctx,
                            ClockSample{
                                clock_dev,
                                lane,
                                (low27 >> PP_CLOCK_KIND_SHIFT) & 0x7u,
                                low27 & PP_CLOCK_VALUE_MASK,
                                lc.th_hi | src[1]});
                    }
                    got = 2;
                }
            } else if ((kSpscPointTypes >> t) & 1u) {
                // Points: the same head record for every point kind, Data with a payload behind its size word. One
                // branch for them all keeps random alternation from mispredicting, so the run gate below tests the
                // words for a run of four of each fixed-size point kind without branching on t first.
                bool blocked = false;
                spsc_for_each_format<spsc_is_point>([&]<PacketFormat F>() __attribute__((always_inline)) {
                    if (!blocked && left >= 4u * F.words && readable >= 8u && spsc_run4<F>(src)) {
                        blocked = true;
                        uint8_t* const first = eb + static_cast<uint32_t>(pt_off);
                        const auto a = spsc_block<F>(src, readable, left / F.words, cur, lc, first);
                        pt_off += kSpscRecBytes * a.n;
                        if (a.n != 0) {
                            order(spsc_ts_at<F>(src, 0, lc.th_hi), a.ts_last);
                            if (__builtin_expect(a.regress != 0, 0)) {
                                block_regress.template operator()<F>(a.n, first);
                            }
                            rc += a.n;
                            lane_rec = eb + static_cast<uint32_t>(pt_off) - kSpscRecBytes;
                            lane_rec_zone = false;
                            got = F.words * a.n;
                        }
                    }
                });
                if (!blocked) {
                    // No branch follows the type from here: the Data-only quantities are masked, not selected.
                    const uint32_t dm = kSpscDataMaskOfType[t];
                    const uint32_t size_word = std::min<uint32_t>(kSpscDataFormat.size_word, readable - 1u);
                    const uint32_t n =
                        ((src[size_word] >> kSpscDataFormat.size_shift) & kSpscDataFormat.size_mask) & dm;
                    const uint32_t words = kSpscWordsOfType[t] + n;
                    if (left >= words) {
                        const uint64_t ts = lc.th_hi | src[kSpscDataFormat.ts_lo];
                        order(ts, ts);
                        const uint32_t sh = dm & 32u;  // 0 for a Point, 32 for Data
                        uint8_t* const dst = (dm ? db : eb) + static_cast<uint32_t>(pt_off >> sh);
                        const uint32_t values = spsc_point(src, readable, n, lc, dst);
                        pt_off += static_cast<uint64_t>(kSpscRecBytes + ((values * 8u) & dm)) << sh;
                        lane_rec = dst;
                        lane_rec_zone = false;
                        rc += 1;
                        got = words;
                    }
                }
            } else {
                spsc_for_format<spsc_is_zone_or_sticky>(t, [&]<PacketFormat F>() __attribute__((always_inline)) {
                    if (left < F.words) {
                        return;
                    }
                    if constexpr (F.kind == Kind::Sticky) {
                        const uint32_t value = F.value_word == 0 ? pp_low27(src[0]) : src[F.value_word];
                        if constexpr (F.sets == PacketFormat::Sets::TimerHi) {
                            th = value;
                            spsc_lane_consts_th(lc, th);
                        } else {
                            pg = value;
                            spsc_lane_consts_prog(lc, pg);
                        }
                        got = F.words;
                    } else if constexpr (F.delta16) {
                        // The block kernel takes every run length; its ends are exact, not sampled: in-block ends
                        // are cursor + positive deltas.
                        const auto a = spsc_block<F>(src, readable, left / F.words, cur, lc, zb + zoff);
                        if (a.n != 0) {
                            // Without a cursor the run decodes into the scratch it would have taken and is not
                            // counted.
                            if (!na) {
                                order(cur + (src[1] >> 16), a.ts_last);
                                zones_emitted(a.n);
                            }
                            cur = a.ts_last;
                            got = F.words * a.n;
                        }
                    } else {
                        uint8_t* const first = zb + zoff;
                        const bool run = left > F.words && pp_type(src[F.words]) == F.type;
                        const SpscBlockResult a = run ? spsc_block<F>(src, readable, left / F.words, cur, lc, first)
                                                      : spsc_one<F>(src, readable, lc, first);
                        if (a.n == 0) {
                            return;
                        }
                        sz += a.stalls;
                        if constexpr (F.has_dur_hi()) {
                            const bool back = spsc_ts_at<F>(src, 0, lc.th_hi) < lane_ts || a.regress != 0;
                            if (__builtin_expect(a.wrapped != 0 || back, 0)) {
                                repairs64.template operator()<F>(a.n, first, a.wrapped != 0, back);
                            }
                            lane_ts = a.ts_last;
                        } else {
                            order(spsc_ts_at<F>(src, 0, lc.th_hi), a.ts_last);
                            if (__builtin_expect(a.regress != 0, 0)) {
                                block_regress.template operator()<F>(a.n, first);
                            }
                        }
                        zones_emitted(a.n);
                        if constexpr (F.reanchor) {
                            cur = a.ts_last;
                            na = 0;
                        }
                        got = F.words * a.n;
                    }
                });
            }
            TT_FATAL(
                got != 0,
                "streaming profiler: undecodable word {:#010x} at offset {} of lane {}'s run of {}",
                src[0],
                i,
                lane,
                run);
            i += got;
        }
        L.last_ts = lane_ts;
        L.last_rec = lane_rec;
        L.last_rec_zone = lane_rec_zone ? 1u : 0u;
        L.last_rec_seq = seq;
        L.timer_hi = th;
        L.prog = pg;
        L.cursor = cur;
        L.need_anchor = na;
    }
    stats.zones += zm;
    stats.records += rc;
    stats.order_regressions += oreg;
    stats.epoch_fixes += fixes;
    stall_zones += sz;
    return Produced{
        static_cast<uint32_t>(zoff / kSpscRecBytes),
        static_cast<uint32_t>(static_cast<uint32_t>(pt_off) / kSpscRecBytes),
        static_cast<uint32_t>(pt_off >> 32)};
}

}  // namespace tt::tt_metal::streaming_profiler
