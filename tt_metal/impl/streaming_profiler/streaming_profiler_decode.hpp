// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// The consumer side's decode: a per-stream decoder that walks BULK_SPAN frames through the vector kernels into
// public Recs and repairs the wall-clock latch race, and the walk that reassembles whole frames out of ring lines.

#include <chrono>
#include <cstdint>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include <tt_stl/tt_pause.hpp>

#include "tt_metal/common/broadcast_ring.hpp"
#include "impl/streaming_profiler/spsc_marker_decode.hpp"
#include "impl/streaming_profiler/streaming_profiler_consumer.hpp"

namespace tt::tt_metal::streaming_profiler {

// One 64 B line of wire words. Frames are line-multiples stored back to back, so every frame starts on a line
// and the BULK_SPAN prefix does the framing.
struct alignas(64) RingLine {
    uint32_t w[16];
};
static_assert(sizeof(RingLine) == 64);

inline constexpr size_t kConsumerScratchRecs = 1 << 16;
// Ring lines per peek (256 KB): bounds what one lapped commit can lose. kMaxFrameRecs is the most records one frame
// can decode to (payload <= 2640 words, 2 words per record, plus DATA expansion slack).
inline constexpr size_t kConsumerLineBatch = 1 << 12;
inline constexpr size_t kMaxFrameRecs = 2048;
inline constexpr uint32_t kEmptyPollsBeforeSleep = 1000;

void set_os_thread_name(const std::string& name);

struct IdleBackoff {
    uint32_t cap_us;
    uint32_t empty_polls = 0;
    uint32_t sleep_us = 1;
    explicit IdleBackoff(uint32_t cap) : cap_us(cap) {}
    void idle() {
        if (++empty_polls < kEmptyPollsBeforeSleep) {
            ttsl::pause();
        } else {
            std::this_thread::sleep_for(std::chrono::microseconds(sleep_us));
            sleep_us = std::min(sleep_us + sleep_us / 4 + 1, cap_us);
        }
    }
    // The spin phase alone: true while the caller should poll again, false once it should park.
    bool spin() {
        if (++empty_polls < kEmptyPollsBeforeSleep) {
            ttsl::pause();
            return true;
        }
        return false;
    }
    void reset() {
        empty_polls = 0;
        sleep_us = 1;
    }
};

inline constexpr uint64_t kEpoch = 1ull << 32;

// A lane's timestamp went backwards. Lanes emit in end order and a wall-clock read is either right or exactly 2^32
// high (kernel_profiler_streaming.hpp read_wall_clock), so the previous record borrowed the next epoch and this one
// proves it. An S/ATOMIC start is derived from its end and a point is its timestamp, so those move down whole; a
// ZONE_L's start was read separately, so only its inflated duration moves. `rec_off` is the sink offset just past
// the record to repair, 0 when it is not at hand. False is an order regression the decoder only counts.
// noinline: inlined at its sites, the repair body inflated the walk's register pressure (+6% on the point path).
__attribute__((noinline)) inline bool repair_prev_record(uint8_t* buf, uint64_t prev, uint64_t ts, uint64_t rec_off) {
    if (prev < kEpoch || prev - ts > kEpoch || rec_off == 0) {
        return false;
    }
    uint64_t* rec = reinterpret_cast<uint64_t*>(buf + rec_off - profiler::kSpscRecBytes);
    if (rec[1] >= kEpoch) {
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

// A ZONE_L's start was read separately from its end, so its duration is two reads apart: one that borrowed the next
// epoch shows as an elapsed time in [-2^32, -1] and moves down whole. Its end may legitimately precede the previous
// record's when that record is the stall zone raised by this zone's own ring reservation. `lane_ts` / `lane_rec` are
// the lane's last record before this run of `n`, whose first record sits at sink offset `first_rec`.
__attribute__((noinline)) inline Repairs zone_l_repairs(
    uint8_t* buf,
    const uint32_t* src,
    uint32_t n,
    uint64_t first_rec,
    bool wrapped,
    bool back,
    uint64_t lane_ts,
    uint64_t lane_rec) {
    constexpr uint32_t kRec = profiler::kSpscRecBytes;
    Repairs out;
    const auto rec_at = [buf](uint64_t off) { return reinterpret_cast<uint64_t*>(buf + off); };
    const auto end_of = [src](uint32_t k) {
        return (static_cast<uint64_t>(src[5u * k + 2u]) << 32) | src[5u * k + 1u];
    };
    if (wrapped) {
        for (uint32_t k = 0; k < n; k++) {
            if (src[5u * k + 4u] != 0xFFFFFFFFu) {
                continue;
            }
            const uint64_t end = end_of(k);
            const uint64_t dur = (static_cast<uint64_t>(src[5u * k + 4u]) << 32) | src[5u * k + 3u];
            if (end >= dur + kEpoch) {
                out.fixes++;
                uint64_t* rec = rec_at(first_rec + kRec * k);
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
            const uint64_t prev_off = k == 0 ? lane_rec : first_rec + kRec * k;
            bool after_stall = true;
            if (prev_off != 0) {
                const uint64_t* prev = rec_at(prev_off - kRec);
                after_stall = (prev[2] >> 61) == static_cast<uint32_t>(RecType::Zone) &&
                              (static_cast<uint32_t>(prev[2]) & 0x07FFFFFFu) == profiler::kSpscStallZoneId;
            }
            if (after_stall) {
                out.regressions++;
            } else {
                const bool fixed = repair_prev_record(buf, prev_ts, end, prev_off);
                out.fixes += fixed;
                out.regressions += !fixed;
            }
        }
    }
    return out;
}

// One stream's decode, owned by one thread: composes public Recs into the sink's scratch, repairs the wall-clock
// latch race and keeps the wire-integrity totals its owner reports.
struct StreamDecoder {
    profiler::SpanDecodeState* st = nullptr;
    uint64_t batch_seq = 0;  // a lane's last-record offset is only meaningful within its own batch
    uint32_t dev = 0;
    profiler::SpscRecSink sink{};
    StreamStats stats;
    uint64_t stall_zones = 0, stall_mark = 0;

    uint32_t decode_frame(const uint32_t* frame, uint32_t frame_words);

    // Closes the batch in the sink: its record count and the stalls since the previous close; the sink restarts.
    struct BatchEnd {
        size_t records;
        uint64_t stalls;
    };
    BatchEnd end_batch() {
        const BatchEnd b{sink.off / profiler::kSpscRecBytes, stall_zones - stall_mark};
        stall_mark = stall_zones;
        sink.off = 0;
        batch_seq++;
        return b;
    }
};

// Decodes one packed BULK_SPAN frame in place. Every record composes through a vector kernel: a run of two or more
// same-type records through its block kernel, a lone zone through its one-record composer, a point packet (EVENT or
// DATA) through spsc_point. Returns the payload words the control vector implies, which the caller checks against
// the frame's length field (a pack-rule disagreement desynchronizes every later lane), or 0 for an unknown core.
// Decode starts at the larger of the head mirror and the extent's start: the mirror runs behind after an upstream
// loss (adopt and count), the extent after a lagging head write-back (skip the overlap).
inline uint32_t StreamDecoder::decode_frame(const uint32_t* frame, uint32_t frame_words) {
    namespace kp = kernel_profiler;
    using namespace profiler;
    static_assert((kConsumerScratchRecs + kMaxFrameRecs) * kSpscRecBytes < kEpoch);
    SpanDecodeState& S = *st;
    const uint32_t* ctrl = frame + kp::SPSC_SPAN_PREFIX_WORDS;
    const uint32_t core = S.core_of_xy.find(ctrl[kp::SPSC_WIRE_XY]);
    if (core == CoreTable::kNone) {
        stats.unknown_core_frames++;
        return 0;
    }
    // Raw locals for everything the hot walk touches: the kernels store through the sink's byte pointer, and a
    // member reached through `this` would be reloaded after every such store.
    const uint32_t d = dev;
    const uint64_t seq = batch_seq;
    SpscRecSink sk = sink;
    uint64_t zm = 0, sz = 0, oreg = 0, rc = 0, fixes = 0;
    uint64_t lane_ts = 0, lane_rec = 0;  // the lane's last record: its timestamp and its sink slot
    // A run's first record against the lane's last, then the run's last becomes the lane's; a single record passes
    // its timestamp twice.
    const auto order = [&](uint64_t ts_first, uint64_t ts_last) {
        if (__builtin_expect(ts_first < lane_ts, 0)) {
            const bool fixed = repair_prev_record(sk.buf, lane_ts, ts_first, lane_rec);
            fixes += fixed;
            oreg += !fixed;
        }
        lane_ts = ts_last;
    };
    // A kernel flagged a step back somewhere in its run: find each one, record k against record k-1.
    const auto block_regress = [&](const uint32_t* src, uint32_t stride, uint32_t n, uint64_t th_hi, uint64_t first) {
        for (uint32_t k = 1; k < n; k++) {
            if (src[stride * k + 1u] < src[stride * (k - 1u) + 1u]) {
                const bool fixed = repair_prev_record(
                    sk.buf,
                    th_hi | src[stride * (k - 1u) + 1u],
                    th_hi | src[stride * k + 1u],
                    first + kSpscRecBytes * k);
                fixes += fixed;
                oreg += !fixed;
            }
        }
    };
    const auto zone_l_fix = [&](const uint32_t* src, uint32_t n, uint64_t first, bool wrapped, bool back) {
        const Repairs r = zone_l_repairs(sk.buf, src, n, first, wrapped, back, lane_ts, lane_rec);
        fixes += r.fixes;
        oreg += r.regressions;
    };
    const auto zones_emitted = [&](uint32_t n) {
        zm += n;
        rc += n;
        lane_rec = sk.off;
    };

    uint32_t off = kp::SPSC_SPAN_PREFIX_WORDS + kp::SPSC_SPAN_WIRE_CTRL_WORDS;
    SpscLaneConsts lc;
    for (uint32_t r = 0; r < kSpscNRiscDecode; r++) {
        const uint32_t lane = core * kSpscNRiscDecode + r;
        SpscLane& L = S.lanes[lane];
        const uint32_t tail = ctrl[kp::SPSC_WIRE_TAIL_0 + r];
        const uint32_t frame_head = ctrl[kp::SPSC_WIRE_HEAD_0 + r];
        const uint32_t extent = kp::spsc_span_live(frame_head, tail, kSpscRingCap);
        if (extent != tail - frame_head) {
            stats.anomalies++;  // torn snapshot; the clamped geometry still frames consistently on both sides
        }
        const uint32_t start = tail - extent;
        const uint32_t* p = nullptr;
        // A near-full wrapping run arrives as the whole ring image (predicate shared with the device,
        // spsc_span_wrap_image), the pad is phased for ring offset 0, and the payload advance is the full ring.
        const bool ring_ordered = extent != 0 && kp::spsc_span_wrap_image(start, extent, kSpscRingCap);
        if (extent != 0) {
            off += kp::spsc_span_pack_pad(ring_ordered ? 0u : start, off);
            p = frame + off;
            off += ring_ordered ? kSpscRingCap : extent;
            // The frame is read in place from memory the device may be rewriting, so a torn control vector must
            // not send the walk past the frame's own length.
            if (off > frame_words) {
                stats.anomalies++;
                break;
            }
        }
        uint32_t head;
        if (L.seeded == 0) {
            L.seeded = 1;
            head = start;
        } else {
            head = L.head;
            const int32_t behind = static_cast<int32_t>(start - head);
            if (behind > 0) {
                stats.resync_words += static_cast<uint32_t>(behind);
                head = start;
            }
        }
        L.head = tail;
        const uint32_t run = tail - head;
        if (run == 0) {
            continue;
        }
        if (run > extent) {
            stats.anomalies++;
            continue;
        }
        uint32_t th = L.timer_hi;
        uint32_t pg = L.prog;
        uint64_t cur = L.cursor;
        // A wrap image is the whole ring in ring order, so its run is linearised before the walk.
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
        spsc_lane_consts(lc, lane, d, th, pg);
        lane_ts = L.last_ts;
        lane_rec = (L.last_rec >> 32) == seq ? static_cast<uint32_t>(L.last_rec) : 0;
        uint32_t i = 0;
        while (i < run) {
            const uint32_t* const src = p + i;
            const uint32_t t = pp_type(src[0]);
            const uint32_t readable = static_cast<uint32_t>(rd_end - src);
            const uint32_t left = run - i;
            uint32_t got = 0;
            if (t == PP_ZONE_S) {
                const auto z = spsc_zone_s16(src, readable, left / 2u, cur, lc, sk);
                if (z.n != 0) {
                    // Exact, not sampled: in-block ends are cursor + positive deltas.
                    order(cur + (src[1] >> 16), z.ts_last);
                    zones_emitted(z.n);
                    cur = z.ts_last;
                    got = 2u * z.n;
                }
            } else if (t == PP_ZONE_ATOMIC) {
                if (left >= 3u) {
                    if (left > 3u && pp_type(src[3]) == PP_ZONE_ATOMIC) {
                        const uint64_t first = sk.off;
                        const auto a = spsc_atomic8(src, readable, left / 3u, lc, sk);
                        if (a.n != 0) {
                            sz += a.stalls;
                            order(lc.th_hi | src[1], a.ts_last);
                            if (__builtin_expect(a.regress != 0, 0)) {
                                block_regress(src, 3, a.n, lc.th_hi, first);
                            }
                            zones_emitted(a.n);
                            // A block is atomics only (a sticky ends it), so th is constant across it and the last
                            // end re-anchors the lane cursor.
                            cur = a.ts_last;
                            got = 3u * a.n;
                        }
                    } else {
                        const uint64_t end = lc.th_hi | src[1];
                        sz += (src[0] & 0x07FFFFFFu) == kSpscStallZoneId;
                        order(end, end);
                        spsc_atomic1(src, readable, lc, sk);
                        zones_emitted(1);
                        cur = end;
                        got = 3;
                    }
                }
            } else if (t == PP_EVENT || t == PP_DATA) {
                // Both are points: the same head record, DATA with a payload behind a size word. One branch for the
                // pair keeps random alternation from mispredicting; only an EVENT run of four takes the block kernel.
                // Tested for both types so the branch does not follow the type: false for any DATA.
                if (left >= 8u && readable >= 8u && spsc_event_run4(src)) {
                    const uint64_t first = sk.off;
                    const auto a = spsc_event16(src, readable, left / 2u, lc, sk);
                    if (a.n != 0) {
                        order(lc.th_hi | src[1], a.ts_last);
                        if (__builtin_expect(a.regress != 0, 0)) {
                            block_regress(src, 2, a.n, lc.th_hi, first);
                        }
                        rc += a.n;
                        lane_rec = sk.off;
                        got = 2u * a.n;
                    }
                } else {
                    // No branch follows the type from here: the DATA-only quantities are masked, not selected.
                    const uint32_t dm = 0u - static_cast<uint32_t>(t == PP_DATA);
                    const uint32_t w2 = src[readable >= 3u ? 2u : 1u];
                    const uint32_t n = pp_data_size(w2) & dm;
                    const uint32_t words = 2u + (dm & (1u + n));
                    if (left >= words) {
                        const uint64_t ts = lc.th_hi | src[1];
                        order(ts, ts);
                        const uint64_t head_off = sk.off;
                        rc += spsc_point(src, readable, dm, n, lc, sk);
                        lane_rec = head_off + kSpscRecBytes;  // the head, not a DATA's Ext/Cont
                        got = words;
                    }
                }
            } else if (t == PP_ZONE_L) {
                if (left >= 5u) {
                    const uint64_t first = sk.off;
                    if (left > 5u && pp_type(src[5]) == PP_ZONE_L) {
                        const auto a = spsc_zone_l8(src, readable, left / 5u, lc, sk);
                        if (a.n != 0) {
                            const uint64_t ts_first = (static_cast<uint64_t>(src[2]) << 32) | src[1];
                            sz += a.stalls;
                            const bool back = ts_first < lane_ts || a.regress != 0;
                            if (__builtin_expect(a.wrapped != 0 || back, 0)) {
                                zone_l_fix(src, a.n, first, a.wrapped != 0, back);
                            }
                            lane_ts = a.ts_last;
                            zones_emitted(a.n);
                            got = 5u * a.n;  // the cursor is untouched
                        }
                    } else {
                        const uint64_t end = (static_cast<uint64_t>(src[2]) << 32) | src[1];
                        sz += (src[0] & 0x07FFFFFFu) == kSpscStallZoneId;
                        spsc_zone_l1(src, readable, lc, sk);
                        const bool wrapped = src[4] == 0xFFFFFFFFu;
                        const bool back = end < lane_ts;
                        if (__builtin_expect(wrapped || back, 0)) {
                            zone_l_fix(src, 1, first, wrapped, back);
                        }
                        lane_ts = end;
                        zones_emitted(1);
                        got = 5;
                    }
                }
            } else if (t == PP_STICKY_TIMER) {
                th = pp_timer_hi(src[0]);
                spsc_lane_consts_th(lc, th);
                got = 1;
            } else if (t == PP_STICKY_PROG) {
                pg = pp_low27(src[0]);
                spsc_lane_consts_prog(lc, pg);
                got = 1;
            } else if (t == PP_STICKY_PROG_EXT) {
                if (left >= 2u) {
                    pg = src[1];
                    spsc_lane_consts_prog(lc, pg);
                    got = 2;
                }
            }
            if (got == 0) {
                stats.anomalies++;  // undecodable word, or a record cut by the run's end
                break;
            }
            i += got;
        }
        L.last_ts = lane_ts;
        L.last_rec = (seq << 32) | lane_rec;
        L.timer_hi = th;
        L.prog = pg;
        L.cursor = cur;
    }
    sink = sk;
    stats.zones += zm;
    stats.records += rc;
    stats.order_regressions += oreg;
    stats.epoch_fixes += fixes;
    stall_zones += sz;
    return off - kp::SPSC_SPAN_PREFIX_WORDS;
}

// Reassembles whole frames from one ring's lines and feeds them to a decoder. After a drop the position is
// arbitrary, so it scans line by line for the next plausible head; the decoder's head adoption then counts the gap
// as a resync.
class FrameWalker {
public:
    // One peek of the ring: returns false when nothing moved. Frames decode in place from the ring's memory, which
    // the device may be rewriting, so nothing leaves the decoder until the reader commits the frames it consumed;
    // a failed commit rolls the decoder back to the last delivery and the reader resyncs. Only a frame cut by the
    // peek's end (the ring's wrap, or pages still landing) is copied, into pending_. `deliver(dropped_delta)` runs
    // after a successful commit at the end of the pass and whenever the scratch is within one frame of full; it
    // must close the decoder's batch.
    template <typename Deliver>
    bool pass(BroadcastRing<RingLine>::Reader& reader, StreamDecoder& dec, Deliver&& deliver) {
        const auto view = reader.peek(kConsumerLineBatch);
        // Drops are reported at delivery, so a failed commit's are simply carried into the next pass.
        auto deliver_now = [&] {
            const uint64_t dd = reader.dropped() - last_dropped_;
            last_dropped_ = reader.dropped();
            deliver(dd);
        };
        if (reader.dropped() != last_dropped_) {
            pending_.clear();
            scanning_ = true;
        }
        if (view.empty()) {
            if (reader.dropped() == last_dropped_) {
                return false;
            }
            deliver_now();
            return true;
        }
        // The decoder as of the last delivery: its sink offset and counters are the whole rollback state.
        StreamDecoder saved = dec;
        size_t consumed = 0;  // lines decoded since the last commit
        // False when the reader was lapped: what was decoded since the last delivery is discarded.
        auto commit = [&] {
            if (reader.commit(consumed)) {
                consumed = 0;
                return true;
            }
            dec = saved;
            pending_.clear();
            scanning_ = true;
            return false;
        };
        auto decode = [&](const uint32_t* frame, uint32_t fw) {
            if (dec.sink.off / sizeof(Rec) + kMaxFrameRecs > kConsumerScratchRecs) {
                if (!commit()) {
                    return false;
                }
                deliver_now();
                saved = dec;
            }
            const uint32_t payload = dec.decode_frame(frame, fw);
            if (payload != 0 && payload != frame[1]) {
                dec.stats.anomalies++;
            }
            return true;
        };
        const uint32_t* const w = reinterpret_cast<const uint32_t*>(view.data());
        const size_t words = view.size() * kernel_profiler::SPSC_SPAN_PAGE_WORDS;
        size_t off = 0;
        if (!pending_.empty()) {
            // Frames start on a line and the prefix is one line, so a waiting head always carries its length.
            const uint32_t fw = kernel_profiler::spsc_span_frame_words(pending_[1]);
            const size_t take = std::min<size_t>(fw - pending_.size(), words);
            pending_.insert(pending_.end(), w, w + take);
            off = take;
            consumed = take / kernel_profiler::SPSC_SPAN_PAGE_WORDS;
            if (pending_.size() < fw) {
                if (commit()) {
                    deliver_now();
                }
                return true;
            }
            if (!decode(pending_.data(), fw)) {
                return true;
            }
            pending_.clear();
        }
        while (words - off >= kernel_profiler::SPSC_SPAN_PREFIX_WORDS) {
            const uint32_t w0 = w[off];
            const uint32_t w1 = w[off + 1];
            if (!pp_is_bulkspan(w0) || w1 < kernel_profiler::SPSC_SPAN_WIRE_CTRL_WORDS ||
                w1 > profiler::kSpscMaxPayloadWords) {
                if (!scanning_) {
                    dec.stats.bad_frames++;
                }
                off += kernel_profiler::SPSC_SPAN_PAGE_WORDS;
                consumed++;
                continue;
            }
            scanning_ = false;
            const uint32_t fw = kernel_profiler::spsc_span_frame_words(w1);
            if (words - off < fw) {
                pending_.assign(w + off, w + words);
                consumed += (words - off) / kernel_profiler::SPSC_SPAN_PAGE_WORDS;
                break;
            }
            if (!decode(w + off, fw)) {
                return true;
            }
            off += fw;
            consumed += fw / kernel_profiler::SPSC_SPAN_PAGE_WORDS;
        }
        if (commit()) {
            deliver_now();
        }
        return true;
    }

private:
    std::vector<uint32_t> pending_;
    bool scanning_ = false;
    uint64_t last_dropped_ = 0;
};

}  // namespace tt::tt_metal::streaming_profiler
