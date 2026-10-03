// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/streaming_profiler/decode.hpp"

#include <algorithm>
#include <bit>
#include <cstring>
#include <limits>

#include <tt_stl/assert.hpp>

namespace tt::tt_metal::streaming_profiler {

namespace {

profiler::SpscRecConsts record_consts(const experimental::streaming_profiler::Core& core, int64_t offset) {
    return profiler::SpscRecConsts{
        .coords =
            {static_cast<uint32_t>(core.logical.x) | (static_cast<uint32_t>(core.logical.y) << 8) |
                 (static_cast<uint32_t>(core.physical.x) << 16) | (static_cast<uint32_t>(core.physical.y) << 24),
             core.chip_id | (static_cast<uint32_t>(core.processor) << 16)},
        .offset = static_cast<uint64_t>(offset)};
}

constexpr uint64_t kEpoch = 1ull << 32;

// A wall-clock read is either right or exactly 2^32 high (kernel_profiler_streaming.hpp read_wall_clock), so a step
// back means the previous record borrowed the next epoch. A ZONE_L, the only zone with a duration of 2^32 or more,
// read its start separately, so only its duration moves; any other zone derives its start from its end and a point
// is its timestamp, so those move down whole.
// noinline: inlined at its sites, the repair body raises the walk's register pressure.
__attribute__((noinline)) bool repair_prev_record(uint8_t* rec_start, bool zone, uint64_t prev_ts, uint64_t next_ts) {
    if (prev_ts < kEpoch || prev_ts - next_ts > kEpoch || rec_start == nullptr) {
        return false;
    }
    uint64_t* rec = reinterpret_cast<uint64_t*>(rec_start);
    if (zone && rec[profiler::kSpscQwDuration] >= kEpoch) {
        rec[profiler::kSpscQwDuration] -= kEpoch;
    } else if (rec[profiler::kSpscQwTimestamp] >= kEpoch) {
        rec[profiler::kSpscQwTimestamp] -= kEpoch;
    } else {
        return false;
    }
    return true;
}

// Repairs the borrowed epochs in a run of n F records at `first`, each against the record before it and the first
// against the lane's last; returns the regressions it could not repair.
template <profiler::PacketFormat F>
__attribute__((noinline)) uint64_t
repair_run(const uint32_t* src, uint32_t n, uint8_t* first, uint64_t th_hi, uint64_t ts0, const SpscLane& lane) {
    constexpr uint32_t kRecBytes = profiler::spsc_rec_bytes<F>;
    uint64_t order_regressions = 0;
    if constexpr (F.has_dur_hi()) {
        for (uint32_t k = 0; k < n; k++) {
            const uint32_t* packet = src + F.words * k;
            if (packet[F.dur_hi] != profiler::kSpscBorrowedDurHi) {
                continue;
            }
            const uint64_t end = profiler::spsc_ts_at<F>(src, k, th_hi);
            const uint64_t duration = (static_cast<uint64_t>(packet[F.dur_hi]) << 32) | packet[F.dur_lo];
            if (end >= duration + kEpoch) {
                uint64_t* rec = reinterpret_cast<uint64_t*>(first + kRecBytes * k);
                rec[profiler::kSpscQwDuration] = duration + kEpoch;
                rec[profiler::kSpscQwTimestamp] = end - rec[profiler::kSpscQwDuration];
            } else {
                order_regressions++;
            }
        }
    }
    const auto step = [&](uint64_t prev_ts, uint64_t next_ts, uint8_t* prev_rec, bool prev_zone) {
        if (!(next_ts < prev_ts)) {
            return;
        }
        const bool fixed = repair_prev_record(prev_rec, prev_zone, prev_ts, next_ts);
        order_regressions += !fixed;
    };
    step(lane.last_ts, ts0, lane.last_rec, lane.last_rec_zone);
    if constexpr (!F.delta16) {  // delta16 ends are cursor + positive deltas
        for (uint32_t k = 1; k < n; k++) {
            step(
                profiler::spsc_ts_at<F>(src, k - 1, th_hi),
                profiler::spsc_ts_at<F>(src, k, th_hi),
                first + kRecBytes * (k - 1),
                F.kind == profiler::Kind::Zone);
        }
    }
    return order_regressions;
}

template <profiler::PacketFormat F>
constexpr uint32_t sticky_value(const uint32_t* src) {
    static_assert(F.kind == profiler::Kind::Sticky);
    return F.value_word == 0 ? src[0] & PP_LOW27_MASK : src[F.value_word];
}

}  // namespace

StreamDecoder::StreamDecoder(const CaptureContext::Device& dev) :
    lanes_(dev.lanes.size()),
    consts_(dev.lanes.size()),
    heads_(dev.lanes.size() / profiler::kSpscNRiscDecode),
    core_of_xy_(dev.core_of_xy) {
    rec_consts_.reserve(dev.lanes.size());
    for (size_t lane = 0; lane < dev.lanes.size(); lane++) {
        rec_consts_.push_back(
            record_consts(dev.lanes[lane], dev.tiles[lane / profiler::kSpscNRiscDecode].clock_offset));
    }
}

// Decoding starts at whichever is later, the head mirror or the start of the extent. The mirror falls behind after an
// upstream loss, which the decoder accepts, and the extent falls behind after a late head write-back, whose overlap it
// skips.
StreamDecoder::Produced StreamDecoder::decode_frames(
    const uint32_t* frames, std::span<const uint32_t> frame_words, Out out) {
    using namespace profiler;
    // Locals, not members: the kernels store through byte pointers, so anything reached through `this` reloads after
    // every store.
    const SpscRecConsts* const rec_consts = rec_consts_.data();
    const uint64_t seq = ++decoded_seq_;
    const uint64_t delivered_seq = delivered_seq_;
    uint8_t* const zones_out = out.zones;
    uint8_t* const events_out = out.events;
    uint8_t* const data_out = out.data;
    uint64_t* const values_out = reinterpret_cast<uint64_t*>(out.values);
    uint64_t zone_bytes = 0;
    uint64_t value_count_out = 0;
    // Both point offsets share one register, data's in the high half. An array indexed by packet kind would live in
    // memory and put every update on a store-to-load chain.
    uint64_t point_offsets = 0;
    uint64_t stalls = 0;
    uint64_t unrepaired = 0;
    int64_t newest_ticks = std::numeric_limits<int64_t>::min();

    const uint32_t* frame = frames;
    for (const uint32_t frame_len : frame_words) {
        const uint32_t* ctrl = frame + kernel_profiler::SPSC_SPAN_PREFIX_WORDS;
        const uint32_t core = core_of_xy_.find(frame[kernel_profiler::SPSC_PREFIX_XY]);
        TT_FATAL(
            core != CoreTable::kNone,
            "streaming profiler: frame from NoC core {:#x}, which the capture did not seed",
            frame[kernel_profiler::SPSC_PREFIX_XY]);
        uint32_t* const heads = heads_[core].data();
        SpscLane* const core_lanes = lanes_.data() + size_t{core} * kSpscNRiscDecode;
        SpscLaneConsts* const core_consts = consts_.data() + size_t{core} * kSpscNRiscDecode;
        const simde__m256i tail_v =
            simde_mm256_loadu_si256(reinterpret_cast<const simde__m256i*>(ctrl + kernel_profiler::SPSC_WIRE_TAIL_0));
        const simde__m256i idle = simde_mm256_and_si256(
            simde_mm256_cmpeq_epi32(
                tail_v,
                simde_mm256_loadu_si256(
                    reinterpret_cast<const simde__m256i*>(frame + kernel_profiler::SPSC_PREFIX_HEAD_0))),
            simde_mm256_cmpeq_epi32(tail_v, simde_mm256_loadu_si256(reinterpret_cast<const simde__m256i*>(heads))));
        uint32_t busy = ~static_cast<uint32_t>(simde_mm256_movemask_ps(simde_mm256_castsi256_ps(idle))) &
                        ((1u << kSpscNRiscDecode) - 1u);
        uint64_t frame_newest_ts = 0;
        uint32_t word_offset = kernel_profiler::SPSC_SPAN_PREFIX_WORDS + kernel_profiler::SPSC_SPAN_WIRE_CTRL_WORDS;
        for (; busy != 0; busy &= busy - 1) {
            const uint32_t risc = static_cast<uint32_t>(std::countr_zero(busy));
            const uint32_t lane_index = core * kSpscNRiscDecode + risc;
            SpscLane& lane = core_lanes[risc];
            const uint32_t tail = ctrl[kernel_profiler::SPSC_WIRE_TAIL_0 + risc];
            const uint32_t start = frame[kernel_profiler::SPSC_PREFIX_HEAD_0 + risc];
            const uint32_t extent = tail - start;
            const uint32_t* words = nullptr;
            // A nearly full run that wraps arrives as the whole ring image, so its pad is phased for ring offset 0 and
            // its payload advances by the full ring.
            const bool ring_ordered = extent != 0 && kernel_profiler::spsc_span_wrap_image(start, extent, kSpscRingCap);
            if (extent != 0) {
                word_offset += kernel_profiler::spsc_span_pack_pad(ring_ordered ? 0u : start, word_offset);
                words = frame + word_offset;
                word_offset += ring_ordered ? kSpscRingCap : extent;
                TT_FATAL(
                    word_offset <= frame_len,
                    "streaming profiler: frame control block places lane {} {} words past the frame's {}",
                    lane_index,
                    word_offset - frame_len,
                    frame_len);
            }
            uint32_t head;
            if (!lane.seeded) {
                lane.seeded = true;
                head = start;
            } else {
                head = heads[risc];
                if (static_cast<int32_t>(start - head) > 0) {
                    head = start;
                    lane.need_state = true;
                }
            }
            heads[risc] = tail;
            const uint32_t run = tail - head;
            if (run == 0) {
                continue;
            }
            SpscLaneConsts& consts = core_consts[risc];
            // The lane's state stays in the lane: copied into locals and written back, sparse streams decode 7% slower.
            uint32_t& timer_hi = lane.timer_hi;
            uint32_t& prog = lane.prog;
            uint64_t& cursor = lane.cursor;
            uint32_t linear[kSpscRingCap];
            if (ring_ordered) {
                const uint32_t ring_head = head & kSpscRingMask;
                const uint32_t first = kSpscRingCap - ring_head < run ? kSpscRingCap - ring_head : run;
                std::memcpy(linear, words + ring_head, first * sizeof(uint32_t));
                if (first < run) {
                    std::memcpy(linear + first, words, (run - first) * sizeof(uint32_t));
                }
                words = linear;
            } else {
                words += extent - run;
            }
            const uint32_t* const readable_end = ring_ordered ? words + run : frame + frame_len;
            uint32_t pos = 0;
            if (lane.need_state) {
                // The slots hold the state at the frame's tail. A sticky packet inside the run means the words before
                // it ran on an earlier value that isn't recorded here, so the run is picked up from its last sticky.
                timer_hi = ctrl[kernel_profiler::SPSC_WIRE_TIMER_0 + risc];
                prog = ctrl[kernel_profiler::spsc_wire_prog_word(risc)];
                uint32_t k = 0;
                while (k < run) {
                    const uint32_t type = pp_type(words[k]);
                    uint32_t packet_words = kSpscWordsOfType[type];
                    if (packet_words == 0) {
                        break;
                    }
                    if (kSpscDataMaskOfType[type] != 0) {
                        if (k + kSpscDataFormat.size_word >= run) {
                            break;
                        }
                        packet_words += (words[k + kSpscDataFormat.size_word] >> kSpscDataFormat.size_shift) &
                                        kSpscDataFormat.size_mask;
                    }
                    spsc_for_format<spsc_is_sticky>(type, [&]<PacketFormat F>() __attribute__((always_inline)) {
                        if (k + F.words <= run) {
                            (F.sets == PacketFormat::Sets::TimerHi ? timer_hi : prog) = sticky_value<F>(words + k);
                        }
                        pos = k + packet_words;
                    });
                    k += packet_words;
                }
                lane.need_state = false;
                lane.need_anchor = true;
                spsc_lane_consts(consts, rec_consts[lane_index], timer_hi, prog);
            }
            if (lane.last_rec_seq <= delivered_seq) {
                lane.last_rec = nullptr;
            }
            uint8_t* const entry_rec = lane.last_rec;
            const uint64_t entry_ts = lane.last_ts;
            const auto zones_emitted = [&](uint32_t n) {
                zone_bytes += kSpscZoneBytes * n;
                lane.last_rec = zones_out + zone_bytes - kSpscZoneBytes;
                lane.last_rec_zone = true;
            };
            while (pos < run) {
                const uint32_t* const src = words + pos;
                const uint32_t type = pp_type(src[0]);
                const uint32_t readable = static_cast<uint32_t>(readable_end - src);
                const uint32_t left = run - pos;
                uint32_t consumed = 0;
                if ((kSpscPointTypes >> type) & 1u) {
                    // One branch for all of them, so random alternation between them doesn't mispredict.
                    bool blocked = false;
                    spsc_for_each_format<spsc_is_point>([&]<PacketFormat F>() __attribute__((always_inline)) {
                        if (!blocked && left >= 4u * F.words && readable >= 8u && spsc_run4<F>(src)) {
                            blocked = true;
                            uint8_t* const first = events_out + static_cast<uint32_t>(point_offsets);
                            const auto block = spsc_block<F>(src, readable, left / F.words, cursor, consts, first);
                            point_offsets += kSpscEventBytes * block.n;
                            if (block.n != 0) {
                                const uint64_t ts0 = spsc_ts_at<F>(src, 0, consts.th_hi);
                                if (__builtin_expect(ts0 < lane.last_ts || block.regress != 0, 0)) {
                                    unrepaired += repair_run<F>(src, block.n, first, consts.th_hi, ts0, lane);
                                }
                                lane.last_ts = block.ts_last;
                                lane.last_rec = events_out + static_cast<uint32_t>(point_offsets) - kSpscEventBytes;
                                lane.last_rec_zone = false;
                                consumed = F.words * block.n;
                            }
                        }
                    });
                    if (!blocked) {
                        const uint32_t data_mask = kSpscDataMaskOfType[type];
                        const uint32_t size_word = std::min<uint32_t>(kSpscDataFormat.size_word, readable - 1u);
                        const uint32_t payload_words =
                            ((src[size_word] >> kSpscDataFormat.size_shift) & kSpscDataFormat.size_mask) & data_mask;
                        const uint32_t packet_words = kSpscWordsOfType[type] + payload_words;
                        if (left >= packet_words) {
                            const uint64_t timestamp = consts.th_hi | src[kSpscDataFormat.ts_lo];
                            const uint32_t half_shift = data_mask & 32u;
                            uint8_t* const dst = (data_mask ? data_out : events_out) +
                                                 static_cast<uint32_t>(point_offsets >> half_shift);
                            if (__builtin_expect(timestamp < lane.last_ts, 0)) {
                                unrepaired +=
                                    !repair_prev_record(lane.last_rec, lane.last_rec_zone, lane.last_ts, timestamp);
                            }
                            lane.last_ts = timestamp;
                            value_count_out +=
                                spsc_point(src, readable, payload_words, consts, dst, values_out + value_count_out);
                            point_offsets += static_cast<uint64_t>(
                                                 kSpscEventBytes + ((kSpscDataBytes - kSpscEventBytes) & data_mask))
                                             << half_shift;
                            lane.last_rec = dst;
                            lane.last_rec_zone = false;
                            consumed = packet_words;
                        }
                    }
                } else {
                    spsc_for_format<spsc_is_zone_or_sticky>(type, [&]<PacketFormat F>() __attribute__((always_inline)) {
                        if (left < F.words) {
                            return;
                        }
                        if constexpr (F.kind == Kind::Sticky) {
                            const uint32_t value = sticky_value<F>(src);
                            if constexpr (F.sets == PacketFormat::Sets::TimerHi) {
                                timer_hi = value;
                                spsc_lane_consts_th(consts, timer_hi);
                            } else {
                                prog = value;
                                spsc_lane_consts_prog(consts, prog);
                            }
                            consumed = F.words;
                        } else if constexpr (F.delta16) {
                            const bool long_run = left >= kSpscDelta16BlockRun * F.words &&
                                                  pp_type(src[(kSpscDelta16BlockRun - 1) * F.words]) == F.type;
                            const auto block =
                                long_run ? spsc_block<F>(
                                               src, readable, left / F.words, cursor, consts, zones_out + zone_bytes)
                                         : spsc_delta16_short<F>(
                                               src, left / F.words, cursor, consts, zones_out + zone_bytes);
                            if (block.n != 0) {
                                // Without a cursor the run decodes into the scratch space it would have used and isn't
                                // counted.
                                if (!lane.need_anchor) {
                                    const uint64_t ts0 = cursor + (src[1] >> 16);
                                    if (__builtin_expect(ts0 < lane.last_ts, 0)) {
                                        unrepaired += repair_run<F>(
                                            src, block.n, zones_out + zone_bytes, consts.th_hi, ts0, lane);
                                    }
                                    lane.last_ts = block.ts_last;
                                    zones_emitted(block.n);
                                }
                                cursor = block.ts_last;
                                consumed = F.words * block.n;
                            }
                        } else {
                            uint8_t* const first = zones_out + zone_bytes;
                            const bool same_type_follows = left > F.words && pp_type(src[F.words]) == F.type;
                            const SpscBlockResult block =
                                same_type_follows ? spsc_block<F>(src, readable, left / F.words, cursor, consts, first)
                                                  : spsc_one<F>(src, readable, consts, first);
                            if (block.n == 0) {
                                return;
                            }
                            stalls += block.stalls;
                            const uint64_t ts0 = spsc_ts_at<F>(src, 0, consts.th_hi);
                            if (__builtin_expect(
                                    (F.has_dur_hi() && block.wrapped != 0) || ts0 < lane.last_ts || block.regress != 0,
                                    0)) {
                                unrepaired += repair_run<F>(src, block.n, first, consts.th_hi, ts0, lane);
                            }
                            lane.last_ts = block.ts_last;
                            zones_emitted(block.n);
                            if constexpr (F.reanchor) {
                                cursor = block.ts_last;
                                lane.need_anchor = false;
                            }
                            consumed = F.words * block.n;
                        }
                    });
                }
                TT_FATAL(
                    consumed != 0,
                    "streaming profiler: undecodable word {:#010x} at offset {} of lane {}'s run of {}",
                    src[0],
                    pos,
                    lane_index,
                    run);
                pos += consumed;
            }
            if (lane.last_rec != entry_rec) {
                lane.last_rec_seq = seq;
            }
            // A last record 2^32 or more past the lane's previous one is taken for a torn read until a later record
            // repairs it, so the batch waits for the cover to reach the time it would repair to.
            const bool torn_last = entry_ts != 0 && lane.last_ts >= entry_ts + kEpoch;
            frame_newest_ts = std::max(frame_newest_ts, torn_last ? lane.last_ts - kEpoch : lane.last_ts);
        }
        if (frame_newest_ts != 0) {
            newest_ticks = std::max(
                newest_ticks,
                static_cast<int64_t>(frame_newest_ts) +
                    static_cast<int64_t>(rec_consts[core * kSpscNRiscDecode].offset));
        }
        frame += frame_len;
    }
    return Produced{
        .zones = static_cast<uint32_t>(zone_bytes / kSpscZoneBytes),
        .events = static_cast<uint32_t>(static_cast<uint32_t>(point_offsets) / kSpscEventBytes),
        .data = static_cast<uint32_t>((point_offsets >> 32) / kSpscDataBytes),
        .values = static_cast<uint32_t>(value_count_out),
        .newest_ticks = newest_ticks,
        .stalls = stalls,
        .order_regressions = unrepaired};
}

}  // namespace tt::tt_metal::streaming_profiler
