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
#include "tools/profiler/spsc_marker_decode.hpp"
#include "tools/profiler/streaming_profiler_receiver.hpp"

namespace tt::tt_metal::streaming_profiler {

inline constexpr size_t kConsumerScratchRecs = 1 << 16;
// Ring lines per consumer read (256 KB scratch), and the most records one frame can decode to (payload <= 2640
// words, 2 words per record, plus DATA expansion slack).
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
    // Per lane: (batch_seq << 32) | 1 + index of the lane's last zone record in the sink, bit 31 set when the
    // repairable field is the duration rather than the start; 0 = none. Lets a regression repair the record it
    // follows while that record is still in the scratch.
    uint64_t* last_rec = nullptr;
    uint64_t batch_seq = 0;  // the owner bumps it whenever the sink restarts at offset 0
    StallIdMirror stall_ids;
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
    stall_ids.refresh();
    // Raw locals: the emitters store through casted pointers, so anything reached via `this` would be reloaded
    // after every store.
    const uint32_t* const stall_tab = stall_ids.table.empty() ? nullptr : stall_ids.table.data();
    const uint32_t stall_mask = stall_ids.mask;
    auto is_stall = [stall_tab, stall_mask](uint32_t id) -> bool {
        if (stall_tab == nullptr) {
            return false;
        }
        uint32_t slot = (id * 0x9E3779B9u) & stall_mask;
        while (true) {
            const uint32_t v = stall_tab[slot];
            if (v == id) {
                return true;
            }
            if (v == 0xFFFFFFFFu) {
                return false;
            }
            slot = (slot + 1) & stall_mask;
        }
    };
    const uint32_t d = dev;
    const bool k512 = profiler::spsc_host_avx512();
    Sink& sk = sink;
    uint64_t zm = 0, sz = 0, oreg = 0, rc = 0, fixes = 0;
    uint64_t mn = min_ts, mx = max_ts;
    uint64_t* const lts = last_ts;
    uint64_t* const lrec = last_rec;
    const uint64_t seq = batch_seq;
    auto rec_count = [&]() -> uint64_t {
        if constexpr (Sink::kStores) {
            return sk.off / profiler::kSpscRecBytes;
        } else {
            return 0;
        }
    };
    // qword 0 is the start (or point timestamp), qword 1 the duration.
    auto fix_rec = [&](uint64_t idx, uint32_t qword) {
        if constexpr (Sink::kStores) {
            reinterpret_cast<uint64_t*>(sk.buf + profiler::kSpscRecBytes * idx)[qword] -= 1ull << 32;
        }
    };
    // A lane's timestamp went backwards. Same high word and the previous low word inside the latch window is the
    // device race (the previous read borrowed the next epoch); anything else is a genuine order regression. A
    // storing sink repairs the previous record if it is still in the scratch and otherwise leaves the regression
    // on the books; the audit counts the race itself, which is what the consumers repair.
    auto regressed = [&](uint32_t lane, uint64_t prev, uint64_t ts) {
        if ((prev >> 32) != (ts >> 32) || static_cast<uint32_t>(prev) < profiler::kLatchWindow) {
            oreg++;
            return;
        }
        if constexpr (Sink::kStores) {
            if (lrec[lane] == 0 || (lrec[lane] >> 32) != seq) {
                oreg++;
                return;
            }
            fix_rec((lrec[lane] & 0x7FFFFFFFu) - 1u, (lrec[lane] >> 31) & 1u);
        }
        fixes++;
        lts[lane] = ts;
    };
    auto note_last = [&](uint32_t lane, bool fix_duration) {
        if constexpr (Sink::kStores) {
            lrec[lane] = (seq << 32) | (fix_duration ? 1ull << 31 : 0) | rec_count();
        }
    };
    const auto meta_of = [d](uint32_t lane, RecType t) {
        return static_cast<uint64_t>((lane << 16) | (d << 26) | (static_cast<uint32_t>(t) << 29)) << 32;
    };

    // A ZONE_L's duration is the difference of two wall-clock reads: a borrowed epoch on the start read leaves it
    // negative, one on the end read inflates it and leaves the start right, so its repairable field is the
    // duration. An S or ATOMIC duration is an exact count, and the whole zone moves with a borrowed end.
    auto emit = [&](uint32_t lane, uint32_t zone_id, uint64_t end, uint32_t prog, uint64_t dur, bool two_reads) {
        zm++;
        sz += is_stall(zone_id) ? 1 : 0;
        if (mn == 0) {
            mn = end;
        }
        mx = end;
        if (two_reads && static_cast<int64_t>(dur) < 0) {
            dur += 1ull << 32;
            fixes++;
        }
        if (end < lts[lane]) {
            regressed(lane, lts[lane], end);
        } else {
            lts[lane] = end;
        }
        rc++;
        if constexpr (Sink::kStores) {
            sk.put4(end - dur, dur, meta_of(lane, RecType::Zone) | zone_id, prog);
        }
        note_last(lane, two_reads);
    };
    auto emit_data = [&](uint32_t lane,
                         uint32_t type,
                         uint32_t id,
                         uint64_t ts,
                         uint32_t prog,
                         const uint32_t* payload,
                         uint32_t n) {
        const uint64_t pg = prog;
        // PP_EVENT is payload-less: one record is the whole packet, no Ext or Cont follows.
        if (type != PP_DATA) {
            rc++;
            if constexpr (Sink::kStores) {
                sk.put4(ts, 0, meta_of(lane, RecType::Event) | id, pg);
            }
            return;
        }
        rc += 2 + (n > 2 ? (n - 1) / 2 : 0);
        if constexpr (Sink::kStores) {
            // Ext carries the payload count in its id and payload words 1-2 in its ts, so a short DATA is two records.
            const uint64_t hi0 = n >= 1 ? payload[0] : 0;
            const uint64_t lo0 = n >= 2 ? payload[1] : 0;
            sk.put4(ts, 0, meta_of(lane, RecType::Data) | id, pg);
            sk.put4((hi0 << 32) | lo0, 0, meta_of(lane, RecType::Ext) | n, pg);
            for (uint32_t k = 2; k < n; k += 2) {
                const uint64_t hi = payload[k];
                const uint64_t lo = (k + 1 < n) ? payload[k + 1] : 0;
                sk.put4((hi << 32) | lo, 0, meta_of(lane, RecType::Cont), pg);
            }
        } else {
            (void)payload;
        }
    };

    auto emit_atomic16 =
        [&](uint32_t lane, uint32_t th, uint32_t prog, const uint32_t* src, uint32_t avail, uint32_t max_recs)
        -> uint32_t {
        const auto a = k512 ? profiler::spsc_atomic16_avx512(src, avail, max_recs, th, prog, lane, d, sk)
                            : profiler::spsc_atomic8_avx2(src, avail, max_recs, th, prog, lane, d, sk);
        if (a.n == 0) {
            return 0;
        }
        zm += a.n;
        // Stall zones ride the atomic wire, so the block pays the id probe: one L1 load per record on the miss path.
        if (stall_tab != nullptr) {
            for (uint32_t k = 0; k < a.n; k++) {
                sz += is_stall(src[3u * k] & 0x07FFFFFFu) ? 1 : 0;
            }
        }
        if (a.ts_first < lts[lane]) {
            regressed(lane, lts[lane], a.ts_first);
        }
        lts[lane] = a.ts_last;
        if (__builtin_expect(a.near_wrap, 0)) {
            const uint64_t block_base = rec_count() - a.n;
            for (uint32_t k = 1; k < a.n; k++) {
                if (src[3u * k + 1u] < src[3u * k - 2u] && src[3u * k - 2u] >= profiler::kLatchWindow) {
                    fixes++;
                    fix_rec(block_base + k - 1, 0);
                }
            }
        }
        if (mn == 0) {
            mn = a.ts_first;
        }
        mx = a.ts_last;
        rc += a.n;
        note_last(lane, false);
        return a.n;
    };
    auto emit_zone_s16 =
        [&](uint32_t lane, uint64_t cursor, uint32_t prog, const uint32_t* src, uint32_t avail, uint32_t max_recs)
        -> profiler::SpscZoneS16Result {
        const auto z = k512 ? profiler::spsc_zone_s16_avx512(src, avail, max_recs, cursor, prog, lane, d, sk)
                            : profiler::spsc_zone_s16_avx2(src, avail, max_recs, cursor, prog, lane, d, sk);
        if (z.n == 0) {
            return z;
        }
        zm += z.n;
        // Exact, not sampled: in-block ends are cursor + positive deltas.
        if (z.ts_first < lts[lane]) {
            regressed(lane, lts[lane], z.ts_first);
        }
        lts[lane] = z.ts_last;
        if (mn == 0) {
            mn = z.ts_first;
        }
        mx = z.ts_last;
        rc += z.n;
        note_last(lane, false);
        return z;
    };

    const uint32_t payload = profiler::spsc_decode_frame(*st, frame, emit, emit_data, emit_atomic16, emit_zone_s16, fw);
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
    // One read of the ring: returns false when nothing moved. Whole frames decode in place from the batch; only a
    // frame cut by the batch end waits in pending_ for the next pass. `deliver(dropped_delta)` runs at the end of
    // the pass and, for a storing sink, whenever the scratch is within one frame of full; it must reset the sink.
    template <typename Dec, typename Deliver>
    bool pass(BroadcastRing<RingLine>::Reader& reader, std::span<RingLine> lines, Dec& dec, Deliver&& deliver) {
        const auto got = reader.read_batch(lines);
        const uint64_t dropped_total = reader.dropped();
        uint64_t dd = dropped_total - last_dropped_;
        last_dropped_ = dropped_total;
        if (got.empty() && dd == 0) {
            return false;
        }
        if (dd != 0) {
            pending_.clear();
            scanning_ = true;
        }
        auto decode = [&](const uint32_t* frame, uint32_t fw) {
            if constexpr (Dec::SinkT::kStores) {
                if (dec.sink.off / sizeof(Rec) + kMaxFrameRecs > kConsumerScratchRecs) {
                    deliver(dd);
                    dd = 0;
                }
            }
            const uint32_t payload = dec.decode_frame(frame, fw);
            if (payload != 0 && payload != frame[1]) {
                dec.st->anomalies++;
            }
        };
        const uint32_t* const w = reinterpret_cast<const uint32_t*>(got.data());
        const size_t words = got.size() * kernel_profiler::SPSC_SPAN_PAGE_WORDS;
        size_t off = 0;
        if (!pending_.empty()) {
            // Frames start on a line and the prefix is one line, so a waiting head always carries its length.
            const uint32_t fw = kernel_profiler::spsc_span_frame_words(pending_[1]);
            const size_t take = std::min<size_t>(fw - pending_.size(), words);
            pending_.insert(pending_.end(), w, w + take);
            off = take;
            if (pending_.size() < fw) {
                deliver(dd);
                return true;
            }
            decode(pending_.data(), fw);
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
                continue;
            }
            scanning_ = false;
            const uint32_t fw = kernel_profiler::spsc_span_frame_words(w1);
            if (words - off < fw) {
                pending_.assign(w + off, w + words);
                break;
            }
            decode(w + off, fw);
            off += fw;
        }
        deliver(dd);
        return true;
    }

private:
    std::vector<uint32_t> pending_;
    bool scanning_ = false;
    uint64_t last_dropped_ = 0;
};

}  // namespace tt::tt_metal::streaming_profiler
