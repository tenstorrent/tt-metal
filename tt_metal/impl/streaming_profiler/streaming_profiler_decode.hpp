// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Frame decode shared by the receiver's wire audit and the service's consumer threads: a per-stream decoder over
// the SPSC wire format, and the walk that reassembles whole frames out of ring lines.

#include <chrono>
#include <cstdint>
#include <cstring>
#include <span>
#include <thread>
#include <vector>

#include <x86intrin.h>

#include <tt_stl/tt_pause.hpp>

#include "tt_metal/common/broadcast_ring.hpp"
#include "impl/streaming_profiler/spsc_marker_decode.hpp"
#include "impl/streaming_profiler/streaming_profiler_receiver.hpp"

namespace tt::tt_metal::streaming_profiler {

inline constexpr size_t kConsumerScratchRecs = 1 << 16;
// Ring lines per peek (256 KB): bounds what one lapped commit can lose. kMaxFrameRecs is the most records one frame
// can decode to (payload <= 2640 words, 2 words per record, plus DATA expansion slack).
inline constexpr size_t kConsumerLineBatch = 1 << 12;
inline constexpr size_t kMaxFrameRecs = 2048;
inline constexpr uint32_t kEmptyPollsBeforeSleep = 1000;

inline uint64_t tsc_now() { return __rdtsc(); }
double tsc_ns_per_tick();
inline double ticks_to_ms(uint64_t ticks) { return ticks * tsc_ns_per_tick() / 1e6; }
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

// One stream's decode, owned by one thread. The Sink decides what a record becomes: SpscRecSink composes public
// Recs into the consumer's scratch, SpscNullRecSink composes nothing (the audit, which wants only wire-integrity
// accounting). `st`/`last_ts` are externally owned so the audit decodes straight into the Stream's report fields.
template <typename Sink>
struct StreamDecoder {
    using SinkT = Sink;
    profiler::SpanDecodeState* st = nullptr;
    uint64_t* last_ts = nullptr;
    // Per lane: (batch_seq << 32) | byte offset just past the lane's last timestamped record in the sink (a Data
    // head, not its Ext/Cont); 0 = none. A regression repairs that record while it is still in the scratch, so the
    // offset is only meaningful within its own batch.
    uint64_t* last_rec = nullptr;
    uint64_t batch_seq = 0;  // the owner bumps it whenever the sink restarts at offset 0
    uint32_t dev = 0;
    Sink sink{};
    uint64_t recs = 0, zones = 0, stall_zones = 0, order_regressions = 0, bad_frames = 0;
    uint64_t epoch_fixes = 0;  // timestamps repaired for the wall-clock latch race
    uint64_t stall_mark = 0;  // stall_zones at the last delivery; the batch delta is the difference
    uint64_t min_ts = 0, max_ts = 0;

    uint32_t decode_frame(const uint32_t* frame, uint32_t fw);
};

template <typename Sink>
uint32_t StreamDecoder<Sink>::decode_frame(const uint32_t* frame, uint32_t fw) {
    // Raw locals: the emitters store through casted pointers, so anything reached via `this` would be reloaded
    // after every store.
    const uint32_t d = dev;
    Sink sk = sink;  // a copy: through a reference, every store via sk.buf would force sk.off to reload
    uint64_t zm = 0, sz = 0, oreg = 0, rc = 0, fixes = 0;
    uint64_t mn = min_ts, mx = max_ts;
    uint64_t* const lts = last_ts;
    uint64_t* const lrec = last_rec;
    const uint64_t seq = batch_seq;
    constexpr uint64_t kEpoch = 1ull << 32;
    static_assert((kConsumerScratchRecs + kMaxFrameRecs) * profiler::kSpscRecBytes < kEpoch);
    // The lane's last record: its timestamp and its slot, register-resident across the lane's run.
    uint64_t lane_ts = 0, lane_rec = 0;
    auto enter_lane = [&](uint32_t lane) {
        lane_ts = lts[lane];
        if constexpr (Sink::kStores) {
            const uint64_t ref = lrec[lane];
            lane_rec = (ref >> 32) == seq ? static_cast<uint32_t>(ref) : 0;
        }
    };
    auto leave_lane = [&](uint32_t lane) {
        lts[lane] = lane_ts;
        if constexpr (Sink::kStores) {
            lrec[lane] = (seq << 32) | lane_rec;
        }
    };
    auto sk_off_or_zero = [&]() -> uint64_t {
        if constexpr (Sink::kStores) {
            return sk.off;
        } else {
            return 0;
        }
    };
    auto rec_at = [&](uint64_t byte_off) -> uint64_t* {
        if constexpr (Sink::kStores) {
            return reinterpret_cast<uint64_t*>(sk.buf + byte_off);
        } else {
            return nullptr;
        }
    };
    // A lane's timestamp went backwards. Lanes emit in end order and a wall-clock read is either right or exactly
    // 2^32 high (kernel_profiler_streaming.hpp read_wall_clock), so the previous record borrowed the next epoch
    // and this one proves it. An S/ATOMIC start is derived from its end and a point is its timestamp, so those
    // move down whole; a ZONE_L's start was read separately, so only its inflated duration moves.
    // noinline: inlined at its five sites, the repair body inflated the walk's register pressure (+6% on the point
    // path); out of line it is the cold call it should be.
    // `rec_off` is the sink offset just past the record to repair, 0 when it is not at hand.
    auto repair_prev = [&](uint64_t prev, uint64_t ts, uint64_t rec_off) __attribute__((noinline)) {
        if (prev < kEpoch || prev - ts > kEpoch) {
            oreg++;
            return;
        }
        if constexpr (Sink::kStores) {
            if (rec_off == 0) {
                oreg++;
                return;
            }
            uint64_t* rec = rec_at(rec_off - profiler::kSpscRecBytes);
            if (rec[1] >= kEpoch) {
                rec[1] -= kEpoch;
            } else if (rec[0] >= kEpoch) {
                rec[0] -= kEpoch;
            } else {
                oreg++;
                return;
            }
        }
        fixes++;
    };
    auto regressed = [&](uint64_t prev, uint64_t ts) {
        repair_prev(prev, ts, lane_rec);
        lane_ts = ts;
    };
    // A block's first record against the lane's last, then each in-block step back against the record before it
    // (the kernel's regress mask), each repairing that record when the epoch rule allows.
    auto block_ts = [&](uint64_t ts_first, uint64_t ts_last) {
        if (ts_first < lane_ts) {
            regressed(lane_ts, ts_first);
        }
        lane_ts = ts_last;
    };
    // A kernel flagged a step back somewhere in its run: find each one, record k against record k-1.
    auto block_regress = [&](const uint32_t* src, uint32_t stride, uint32_t n, uint64_t th_hi, uint64_t first_rec) {
        for (uint32_t k = 1; k < n; k++) {
            if (src[stride * k + 1u] < src[stride * (k - 1u) + 1u]) {
                repair_prev(
                    th_hi | src[stride * (k - 1u) + 1u],
                    th_hi | src[stride * k + 1u],
                    first_rec + profiler::kSpscRecBytes * k);
            }
        }
    };
    // Timestamp order for one record against the lane's last, then it becomes the last.
    auto one_ts = [&](uint64_t ts) {
        if (__builtin_expect(ts < lane_ts, 0)) {
            regressed(lane_ts, ts);
        } else {
            lane_ts = ts;
        }
    };
    struct Emitters {
        decltype(block_ts)& block_ts_;
        decltype(block_regress)& block_regress_;
        decltype(one_ts)& one_ts_;
        decltype(repair_prev)& repair_prev_;
        decltype(rec_at)& rec_at_;
        decltype(sk_off_or_zero)& sk_off_or_zero_;
        Sink& sk;
        uint64_t& zm;
        uint64_t& sz;
        uint64_t& rc;
        uint64_t& mn;
        uint64_t& mx;
        uint64_t& oreg;
        uint64_t& fixes;
        uint64_t& lane_ts;
        uint64_t& lane_rec;

        uint32_t atomic8(
            uint32_t, const profiler::SpscLaneConsts& c, const uint32_t* src, uint32_t avail, uint32_t max_recs) {
            const uint64_t first_rec = sk_off_or_zero_();
            const auto a = profiler::spsc_atomic8(src, avail, max_recs, c, sk);
            if (a.n == 0) {
                return 0;
            }
            const uint64_t ts_first = c.th_hi | src[1];
            zm += a.n;
            sz += a.stalls;
            block_ts_(ts_first, a.ts_last);
            if (__builtin_expect(a.regress != 0, 0)) {
                block_regress_(src, 3, a.n, c.th_hi, first_rec);
            }
            if (mn == 0) {
                mn = ts_first;
            }
            mx = a.ts_last;
            rc += a.n;
            if constexpr (Sink::kStores) {
                lane_rec = sk.off;
            }
            return a.n;
        }
        uint64_t atomic1(uint32_t, const profiler::SpscLaneConsts& c, const uint32_t* src, uint32_t readable) {
            const uint64_t end = c.th_hi | src[1];
            zm++;
            if constexpr (Sink::kStores) {
                sz += (src[0] & 0x07FFFFFFu) == profiler::kSpscStallZoneId;
            }
            if (mn == 0) {
                mn = end;
            }
            mx = end;
            one_ts_(end);
            rc++;
            profiler::spsc_atomic1(src, readable, c, sk);
            if constexpr (Sink::kStores) {
                lane_rec = sk.off;
            }
            return end;
        }
        uint32_t event16(
            uint32_t, const profiler::SpscLaneConsts& c, const uint32_t* src, uint32_t avail, uint32_t max_recs) {
            const uint64_t first_rec = sk_off_or_zero_();
            const auto a = profiler::spsc_event16(src, avail, max_recs, c, sk);
            if (a.n == 0) {
                return 0;
            }
            block_ts_(c.th_hi | src[1], a.ts_last);
            if (__builtin_expect(a.regress != 0, 0)) {
                block_regress_(src, 2, a.n, c.th_hi, first_rec);
            }
            rc += a.n;
            if constexpr (Sink::kStores) {
                lane_rec = sk.off;
            }
            return a.n;
        }
        void point(
            uint32_t,
            const profiler::SpscLaneConsts& c,
            const uint32_t* src,
            uint32_t readable,
            uint32_t dm,
            uint32_t n) {
            one_ts_(c.th_hi | src[1]);
            const uint64_t head_off = sk_off_or_zero_();
            rc += profiler::spsc_point(src, readable, dm, n, c, sk);
            if constexpr (Sink::kStores) {
                lane_rec = head_off + profiler::kSpscRecBytes;  // the head, not a DATA's Ext/Cont
            }
        }
        // A ZONE_L's start was read separately from its end, so its duration is two reads apart: one that borrowed
        // the next epoch shows as an elapsed time in [-2^32, -1] and moves down whole. Its end may legitimately
        // precede the previous record's when that record is the stall zone raised by this zone's own ring
        // reservation.
        void zone_l_checks(const uint32_t* src, uint32_t n, uint64_t first_rec, bool wrapped, bool back) {
            if (wrapped) {
                for (uint32_t k = 0; k < n; k++) {
                    if (src[5u * k + 4u] != 0xFFFFFFFFu) {
                        continue;
                    }
                    const uint64_t end = (static_cast<uint64_t>(src[5u * k + 2u]) << 32) | src[5u * k + 1u];
                    const uint64_t dur = (static_cast<uint64_t>(src[5u * k + 4u]) << 32) | src[5u * k + 3u];
                    if (end >= dur + (1ull << 32)) {
                        fixes++;
                        if constexpr (Sink::kStores) {
                            uint64_t* rec = rec_at_(first_rec + profiler::kSpscRecBytes * k);
                            rec[1] = dur + (1ull << 32);
                            rec[0] = end - rec[1];
                        }
                    } else {
                        oreg++;
                    }
                }
            }
            if (back) {
                for (uint32_t k = 0; k < n; k++) {
                    const uint64_t end = (static_cast<uint64_t>(src[5u * k + 2u]) << 32) | src[5u * k + 1u];
                    const uint64_t prev_ts =
                        k == 0 ? lane_ts : ((static_cast<uint64_t>(src[5u * k - 3u]) << 32) | src[5u * k - 4u]);
                    if (!(end < prev_ts)) {
                        continue;
                    }
                    const uint64_t prev_off = k == 0 ? lane_rec : first_rec + profiler::kSpscRecBytes * k;
                    bool after_stall = true;
                    if constexpr (Sink::kStores) {
                        if (prev_off != 0) {
                            const uint64_t* prev = rec_at_(prev_off - profiler::kSpscRecBytes);
                            after_stall = (prev[2] >> 61) == static_cast<uint32_t>(RecType::Zone) &&
                                          (static_cast<uint32_t>(prev[2]) & 0x07FFFFFFu) == profiler::kSpscStallZoneId;
                        }
                    }
                    if (after_stall) {
                        oreg++;
                    } else {
                        repair_prev_(prev_ts, end, prev_off);
                    }
                }
            }
        }
        uint32_t zone_l8(
            uint32_t, const profiler::SpscLaneConsts& c, const uint32_t* src, uint32_t avail, uint32_t max_recs) {
            const uint64_t first_rec = sk_off_or_zero_();
            const auto a = profiler::spsc_zone_l8(src, avail, max_recs, c, sk);
            if (a.n == 0) {
                return 0;
            }
            const uint64_t ts_first = (static_cast<uint64_t>(src[2]) << 32) | src[1];
            zm += a.n;
            sz += a.stalls;
            const bool back = ts_first < lane_ts || a.regress != 0;
            if (__builtin_expect(a.wrapped != 0 || back, 0)) {
                zone_l_checks(src, a.n, first_rec, a.wrapped != 0, back);
            }
            lane_ts = a.ts_last;
            if (mn == 0) {
                mn = ts_first;
            }
            mx = a.ts_last;
            rc += a.n;
            if constexpr (Sink::kStores) {
                lane_rec = sk.off;
            }
            return a.n;
        }
        void zone_l1(uint32_t, const profiler::SpscLaneConsts& c, const uint32_t* src, uint32_t readable) {
            const uint64_t first_rec = sk_off_or_zero_();
            const uint64_t end = (static_cast<uint64_t>(src[2]) << 32) | src[1];
            zm++;
            if constexpr (Sink::kStores) {
                sz += (src[0] & 0x07FFFFFFu) == profiler::kSpscStallZoneId;
            }
            profiler::spsc_zone_l1(src, readable, c, sk);
            const bool wrapped = src[4] == 0xFFFFFFFFu;
            const bool back = end < lane_ts;
            if (__builtin_expect(wrapped || back, 0)) {
                zone_l_checks(src, 1, first_rec, wrapped, back);
            }
            lane_ts = end;
            if (mn == 0) {
                mn = end;
            }
            mx = end;
            rc++;
            if constexpr (Sink::kStores) {
                lane_rec = sk.off;
            }
        }
        profiler::SpscZoneS16Result zone_s16(
            uint32_t,
            uint64_t cursor,
            const profiler::SpscLaneConsts& c,
            const uint32_t* src,
            uint32_t avail,
            uint32_t max_recs) {
            const auto z = profiler::spsc_zone_s16(src, avail, max_recs, cursor, c, sk);
            if (z.n == 0) {
                return z;
            }
            const uint64_t ts_first = cursor + (src[1] >> 16);
            zm += z.n;
            // Exact, not sampled: in-block ends are cursor + positive deltas.
            block_ts_(ts_first, z.ts_last);
            if (mn == 0) {
                mn = ts_first;
            }
            mx = z.ts_last;
            rc += z.n;
            if constexpr (Sink::kStores) {
                lane_rec = sk.off;
            }
            return z;
        }
    } em{
        block_ts,
        block_regress,
        one_ts,
        repair_prev,
        rec_at,
        sk_off_or_zero,
        sk,
        zm,
        sz,
        rc,
        mn,
        mx,
        oreg,
        fixes,
        lane_ts,
        lane_rec};

    const uint32_t payload = profiler::spsc_decode_frame(*st, frame, d, em, enter_lane, leave_lane, fw);
    sink = sk;
    zones += zm;
    stall_zones += sz;
    order_regressions += oreg;
    epoch_fixes += fixes;
    recs += rc;
    min_ts = mn;
    max_ts = mx;
    return payload;
}

// Reassembles whole frames from one ring's lines and feeds them to a decoder. After a drop the position is
// arbitrary, so it scans line by line for the next plausible head; the decoder's head adoption then counts the gap
// as a resync.
class FrameWalker {
public:
    // One peek of the ring: returns false when nothing moved. Frames decode in place from the ring's memory, which
    // the device may be rewriting, so nothing leaves the decoder until the reader commits the frames it consumed;
    // a failed commit rolls the sink and the decoder's counters back to the last delivery and the reader resyncs.
    // Only a frame cut by the peek's end (the ring's wrap, or pages still landing) is copied, into pending_.
    // `deliver(dropped_delta)` runs after a successful commit at the end of the pass and, for a storing sink,
    // whenever the scratch is within one frame of full; it must reset the sink.
    template <typename Dec, typename Deliver>
    bool pass(BroadcastRing<RingLine>::Reader& reader, Dec& dec, Deliver&& deliver) {
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
        struct Snapshot {
            uint64_t off, recs, zones, stall_zones, order_regressions, bad_frames, epoch_fixes, min_ts, max_ts;
        };
        auto snapshot = [&] {
            uint64_t off = 0;
            if constexpr (Dec::SinkT::kStores) {
                off = dec.sink.off;
            }
            return Snapshot{
                off,
                dec.recs,
                dec.zones,
                dec.stall_zones,
                dec.order_regressions,
                dec.bad_frames,
                dec.epoch_fixes,
                dec.min_ts,
                dec.max_ts};
        };
        auto restore = [&](const Snapshot& sn) {
            if constexpr (Dec::SinkT::kStores) {
                dec.sink.off = sn.off;
            }
            dec.recs = sn.recs;
            dec.zones = sn.zones;
            dec.stall_zones = sn.stall_zones;
            dec.order_regressions = sn.order_regressions;
            dec.bad_frames = sn.bad_frames;
            dec.epoch_fixes = sn.epoch_fixes;
            dec.min_ts = sn.min_ts;
            dec.max_ts = sn.max_ts;
        };
        Snapshot sn = snapshot();
        size_t consumed = 0;  // lines decoded since the last commit
        // False when the reader was lapped: what was decoded since the last delivery is discarded.
        auto commit = [&] {
            if (reader.commit(consumed)) {
                consumed = 0;
                return true;
            }
            restore(sn);
            pending_.clear();
            scanning_ = true;
            return false;
        };
        auto decode = [&](const uint32_t* frame, uint32_t fw) {
            if constexpr (Dec::SinkT::kStores) {
                if (dec.sink.off / sizeof(Rec) + kMaxFrameRecs > kConsumerScratchRecs) {
                    if (!commit()) {
                        return false;
                    }
                    deliver_now();
                    sn = snapshot();
                }
            }
            const uint32_t payload = dec.decode_frame(frame, fw);
            if (payload != 0 && payload != frame[1]) {
                dec.st->anomalies++;
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
                    dec.bad_frames++;
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
