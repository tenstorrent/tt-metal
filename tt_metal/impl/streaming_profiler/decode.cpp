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

profiler::SpscLaneConsts lane_consts(const experimental::streaming_profiler::Core& core, int64_t clock_offset) {
    const uint64_t coords_lo = static_cast<uint32_t>(core.logical.x) | (static_cast<uint32_t>(core.logical.y) << 8) |
                               (static_cast<uint32_t>(core.physical.x) << 16) |
                               (static_cast<uint32_t>(core.physical.y) << 24);
    const uint64_t coords_hi = core.chip_id | (static_cast<uint32_t>(core.processor) << 16);
    const auto coords = static_cast<long long>(coords_lo | (coords_hi << 32));
    const auto offset = static_cast<long long>(clock_offset);
    profiler::SpscLaneConsts consts{};
    consts.tail = simde_mm256_setr_epi64x(coords, offset, coords, offset);
    consts.shared = simde_mm256_setr_epi64x(0, 0, coords, offset);
    return consts;
}

constexpr uint64_t kEpoch = 1ull << 32;
// How close below the low word's wrap, in ticks, a lane's last timestamp must be to count as possibly torn. A torn read
// takes its low word just before the low word wraps, and this window is far wider than that, so a lane resuming after
// an idle gap of 2^32 ticks or more is mistaken for torn once in 65,536.
constexpr uint64_t kTornLowWindow = 1ull << 16;

// Repairs the previous record when the next record's timestamp is lower, and returns whether it did. A wall-clock read
// is either correct or exactly 2^32 too high, because another RISC's read of the low word can land between the
// producer's low and high reads as the low word wraps. A step back therefore means the previous record's high word was
// one too high. A ZONE_L, the only zone with a duration of 2^32 or more, reads its start separately, so only its
// duration is corrected. Any other zone derives its start from its end, and a point is just its timestamp, so those are
// moved down by kEpoch whole. It is noinline because inlining the repair code into the decode loop raises the loop's
// register pressure.
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

// Repairs torn wall-clock reads in a run of n F records starting at `first`. Each record is checked against the one
// before it, and the first against the lane's last record. Returns the number of records it could not repair.
template <profiler::PacketFormat F>
__attribute__((noinline)) uint64_t
repair_run(const uint32_t* src, uint32_t n, uint8_t* first, uint64_t th_hi, uint64_t first_ts, const SpscLane& lane) {
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
        if (next_ts < prev_ts) {
            order_regressions += !repair_prev_record(prev_rec, prev_zone, prev_ts, next_ts);
        }
    };
    step(lane.last_ts, first_ts, lane.last_rec, lane.last_rec_zone);
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

StreamDecoder::StreamDecoder(const CaptureContext::Device& dev, experimental::streaming_profiler::RecordType types) :
    lanes_(dev.lanes.size()), heads_(dev.lanes.size() / profiler::kSpscNRiscDecode), dev_(dev) {
    using experimental::streaming_profiler::RecordType;
    using experimental::streaming_profiler::detail::has;
    using profiler::Kind;
    for (const profiler::PacketFormat& format : profiler::kFormats) {
        const bool skipped = (format.kind == Kind::Zone && !has(types, RecordType::Zones)) ||
                             (format.kind == Kind::Point && !has(types, RecordType::Events)) ||
                             (format.kind == Kind::Data && !has(types, RecordType::TimestampedData));
        skipped_types_ |= skipped ? 1u << format.type : 0u;
    }
    for (size_t lane = 0; lane < dev.lanes.size(); lane++) {
        lanes_[lane].consts = lane_consts(dev.lanes[lane], dev.clock_offsets[lane / profiler::kSpscNRiscDecode]);
    }
}

StreamDecoder::Produced StreamDecoder::decode_frames(
    size_t undelivered, const uint32_t* frames, std::span<const uint32_t> frame_words, Out out) {
    using namespace profiler;
    // Locals rather than members, because the kernels store through byte pointers, so anything reached through `this`
    // would be reloaded after every store.
    const uint64_t delivered_seq = decoded_seq_ - undelivered;
    const uint64_t seq = ++decoded_seq_;
    uint8_t* const zones_out = out.zones;
    uint8_t* const events_out = out.events;
    uint8_t* const data_out = out.data;
    uint64_t* const values_out = out.values;
    const uint32_t skipped_types = skipped_types_;
    uint64_t zone_bytes = 0;
    uint64_t value_count_out = 0;
    // The event and data output offsets share one 64-bit variable, with data's in the high half, so both stay in one
    // register. An array indexed by packet kind would live in memory and put every update on a store-to-load chain.
    uint64_t point_offsets = 0;
    uint64_t stalls = 0;
    uint64_t order_regressions = 0;
    int64_t newest_ticks = std::numeric_limits<int64_t>::min();

    const uint32_t* frame = frames;
    for (const uint32_t frame_len : frame_words) {
        const uint32_t* ctrl = frame + kernel_profiler::SPSC_SPAN_PREFIX_WORDS;
        const uint32_t core = dev_.core_of_xy.find(frame[kernel_profiler::SPSC_PREFIX_XY]);
        TT_FATAL(
            core != CoreTable::kNone,
            "streaming profiler: frame from NoC core {:#x}, which the capture did not seed",
            frame[kernel_profiler::SPSC_PREFIX_XY]);
        uint32_t* const heads = heads_[core].data();
        SpscLane* const core_lanes = lanes_.data() + size_t{core} * kSpscNRiscDecode;
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
            // A nearly full run that wraps arrives as the whole ring image, which starts at ring offset 0, so its pad
            // is computed for offset 0 and it takes a full ring of words in the frame.
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
            // Decoding starts at the later of heads[risc], where the lane's previous frame ended, and `start`, where
            // this frame's run begins. A later `start` means words were lost on the way, and the gap is accepted. A
            // later heads[risc] means the relay resent words after a late head write-back, and those are skipped.
            uint32_t head = heads[risc];
            if (!lane.seeded || static_cast<int32_t>(start - head) > 0) {
                lane.seeded = true;
                lane.need_state = true;
                head = start;
            }
            heads[risc] = tail;
            const uint32_t run = tail - head;
            if (run == 0) {
                continue;
            }
            SpscLaneConsts& consts = lane.consts;
            // Uses the lane's state in place, because copying it into locals and back costs sparse streams 7%.
            uint64_t& cursor = lane.cursor;
            uint32_t linear[kSpscRingCap];
            if (ring_ordered) {
                const uint32_t ring_head = head & kSpscRingMask;
                const uint32_t first = std::min(kSpscRingCap - ring_head, run);
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
                // The control block holds the lane's timer high word and runtime id as of the run's end. Words before a
                // sticky packet in the run used an older value that the frame doesn't carry, so decoding starts just
                // past the run's last sticky packet.
                uint32_t timer_hi = ctrl[kernel_profiler::SPSC_WIRE_TIMER_0 + risc];
                uint32_t runtime_id = ctrl[kernel_profiler::spsc_wire_prog_word(risc)];
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
                            (F.sets == PacketFormat::Sets::TimerHi ? timer_hi : runtime_id) =
                                sticky_value<F>(words + k);
                        }
                        pos = k + packet_words;
                    });
                    k += packet_words;
                }
                lane.need_state = false;
                lane.need_anchor = true;
                spsc_lane_consts_th(consts, timer_hi);
                spsc_lane_consts_prog(consts, runtime_id);
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
                if (((skipped_types >> type) & 1u) != 0) {
                    const uint32_t type_words = kSpscWordsOfType[type];
                    if (kSpscDataMaskOfType[type] != 0) {
                        const uint32_t size_word = std::min<uint32_t>(kSpscDataFormat.size_word, readable - 1u);
                        const uint32_t packet_words =
                            type_words + ((src[size_word] >> kSpscDataFormat.size_shift) & kSpscDataFormat.size_mask);
                        if (left >= packet_words) {
                            pos += packet_words;
                            continue;
                        }
                    } else if (left >= type_words) {
                        // The packets of a fixed-length type sit at a fixed stride, so the run's loads don't wait on
                        // one another.
                        uint32_t run_words = type_words;
                        uint32_t stall_zones = (src[0] & PP_LOW27_MASK) == kSpscStallZoneId ? 1u : 0u;
                        while (run_words + type_words <= left && pp_type(src[run_words]) == type) {
                            stall_zones += (src[run_words] & PP_LOW27_MASK) == kSpscStallZoneId ? 1u : 0u;
                            run_words += type_words;
                        }
                        stalls += ((kSpscZoneTypes >> type) & 1u) != 0 ? stall_zones : 0u;
                        pos += run_words;
                        continue;
                    }
                }
                uint32_t consumed = 0;
                if ((kSpscPointTypes >> type) & 1u) {
                    // Events and data records go through one code path, so a random mix of them doesn't cause branch
                    // mispredictions.
                    bool blocked = false;
                    spsc_for_each_format<spsc_is_point>([&]<PacketFormat F>() __attribute__((always_inline)) {
                        if (!blocked && left >= 4u * F.words && spsc_run4<F>(src)) {
                            blocked = true;
                            uint8_t* const first = events_out + static_cast<uint32_t>(point_offsets);
                            const auto block = spsc_block<F>(src, readable, left / F.words, cursor, consts, first);
                            point_offsets += kSpscEventBytes * block.n;
                            if (block.n != 0) {
                                const uint64_t first_ts = spsc_ts_at<F>(src, 0, consts.th_hi);
                                if (__builtin_expect(first_ts < lane.last_ts || block.regress != 0, 0)) {
                                    order_regressions +=
                                        repair_run<F>(src, block.n, first, consts.th_hi, first_ts, lane);
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
                                order_regressions +=
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
                                spsc_lane_consts_th(consts, value);
                            } else {
                                spsc_lane_consts_prog(consts, value);
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
                                // Until the lane has a ZONE_ATOMIC to count from, the run's ends are unknown. It is
                                // still decoded, into output space the next record overwrites, but not counted.
                                if (!lane.need_anchor) {
                                    const uint64_t first_ts = cursor + (src[1] >> 16);
                                    if (__builtin_expect(first_ts < lane.last_ts, 0)) {
                                        order_regressions += repair_run<F>(
                                            src, block.n, zones_out + zone_bytes, consts.th_hi, first_ts, lane);
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
                            const uint64_t first_ts = spsc_ts_at<F>(src, 0, consts.th_hi);
                            if (__builtin_expect(
                                    (F.has_dur_hi() && block.wrapped != 0) || first_ts < lane.last_ts ||
                                        block.regress != 0,
                                    0)) {
                                order_regressions += repair_run<F>(src, block.n, first, consts.th_hi, first_ts, lane);
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
            // A last record 2^32 or more ticks past the lane's previous last record, with a low word just below the
            // wrap, may be torn, so it counts toward newest_ticks 2^32 ticks earlier, where its repair would put it.
            const bool torn_last =
                entry_ts != 0 && lane.last_ts >= entry_ts + kEpoch && lane.last_ts % kEpoch >= kEpoch - kTornLowWindow;
            frame_newest_ts = std::max(frame_newest_ts, torn_last ? lane.last_ts - kEpoch : lane.last_ts);
        }
        if (frame_newest_ts != 0) {
            newest_ticks = std::max(newest_ticks, static_cast<int64_t>(frame_newest_ts) + dev_.clock_offsets[core]);
        }
        frame += frame_len;
    }
    return Produced{
        .zones = static_cast<uint32_t>(zone_bytes / kSpscZoneBytes),
        .events = static_cast<uint32_t>(point_offsets) / kSpscEventBytes,
        .data = static_cast<uint32_t>((point_offsets >> 32) / kSpscDataBytes),
        .values = static_cast<uint32_t>(value_count_out),
        .newest_ticks = newest_ticks,
        .stalls = stalls,
        .order_regressions = order_regressions};
}

}  // namespace tt::tt_metal::streaming_profiler
