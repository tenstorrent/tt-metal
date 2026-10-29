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

using experimental::device_observability::detail::code_bytes;
using experimental::device_observability::detail::TimestampWidth;
static_assert(
    uint32_t(TimestampWidth::Bits16) == 0 && uint32_t(TimestampWidth::Bits32) == 1 &&
        uint32_t(TimestampWidth::Bits64) == 2,
    "a width's value is its packets' count of time words");

// The lane's constant words of a record's head: its address, which the record points at, and the tile's clock offset.
profiler::SpscLaneConsts lane_consts(
    const experimental::device_observability::detail::RecordLane& lane, int64_t clock_offset) {
    const auto address = static_cast<long long>(reinterpret_cast<uintptr_t>(&lane));
    const auto offset = static_cast<long long>(clock_offset);
    profiler::SpscLaneConsts consts{};
    consts.tail = simde_mm256_setr_epi64x(address, offset, address, offset);
    consts.shared = simde_mm256_setr_epi64x(address, 0, 0, offset);
    return consts;
}

// How close below the low word's wrap, in ticks, a lane's last timestamp must be to count as possibly torn. A torn read
// takes its low word just before the low word wraps, and this window is far wider than that, so a lane resuming after
// an idle gap of 2^32 ticks or more is mistaken for torn once in 65,536.
constexpr uint64_t kTornLowWindow = 1ull << 16;

}  // namespace

StreamDecoder::StreamDecoder(
    const CaptureContext::Device& dev,
    std::span<const experimental::device_observability::detail::RecordLane> lanes,
    RecordRoutes& routes,
    const ClockMap* clocks,
    uint32_t dev_index) :
    lanes_(dev.lanes.size()),
    copies_(dev.lanes.size()),
    heads_(dev.lanes.size() / profiler::kSpscNRiscDecode),
    dev_(dev),
    clocks_(clocks),
    dev_index_(dev_index),
    routes_(routes),
    site_routes_(routes.routes_.data()),
    short_run_types_(routes.short_run_types_.data()),
    direct_types_(routes.types().direct_types_),
    strides_(routes.types().strides_),
    writes_order_(routes.types().type_count() > 1) {
    for (size_t lane = 0; lane < dev.lanes.size(); lane++) {
        lanes_[lane].consts = lane_consts(lanes[lane], dev.clock_offsets[lane / profiler::kSpscNRiscDecode]);
        profiler::spsc_lane_consts_th(lanes_[lane].consts, dev.start_timer_hi);
    }
}

namespace {

void write_direct(
    const PayloadToRecord& payload_to_record,
    const uint32_t* src,
    uint32_t payload_at,
    uint64_t time,
    const profiler::SpscLaneConsts& consts,
    uint8_t* dst) {
    simde_mm256_storeu_si256(
        reinterpret_cast<simde__m256i*>(dst), profiler::spsc_record_head(profiler::spsc_load_words(src, 2), consts));
    uint8_t* const record_time = dst + profiler::kSpscTimeAt;
    std::memcpy(record_time, &time, sizeof(time));
    const auto* payload = reinterpret_cast<const std::byte*>(src + payload_at);
    for (const PayloadToRecord::Field& field : payload_to_record.fields) {
        uint8_t* const out = dst + field.to;
        if (field.op == PayloadToRecord::Op::Copy) {
            const uint64_t value = field_value(payload, field.bit, field.width, field.code);
            std::memcpy(out, &value, code_bytes(field.code));
            continue;
        }
        const uint64_t sent =
            field.op == PayloadToRecord::Op::Published ? 0 : field_value(payload, field.bit, field.width, field.code);
        const uint64_t ticks = timestamp_ticks(field.op, time, sent);
        std::memcpy(out, &ticks, sizeof(ticks));
        std::memcpy(out + sizeof(ticks), record_time + sizeof(ticks), sizeof(uint64_t));
    }
}

}  // namespace

[[gnu::always_inline]] inline StreamDecoder::Written StreamDecoder::write_record_to(
    uint32_t type, const Packet& packet, const profiler::SpscLaneConsts& consts, uint64_t* values) {
    uint8_t* const dst = types_out_[type] + type_bytes_[type];
    type_bytes_[type] += strides_[type];
    Written written{.time_slot = dst + profiler::kSpscTimeAt, .value_count = 0};
    const uint32_t* const src = packet.src;
    const uint32_t payload_words = profiler::ppfmt::packet_words(src[0]) - packet.payload_at;
    if (!has_type(direct_types_, type)) {
        written.value_count = profiler::spsc_published(src, packet.payload_at, payload_words, consts, dst, values);
        std::memcpy(written.time_slot, &packet.time, sizeof(packet.time));
    } else if (packet.payload_to_record != nullptr) {
        write_direct(*packet.payload_to_record, src, packet.payload_at, packet.time, consts, dst);
    } else {
        profiler::spsc_record(src, payload_words, consts, dst);
    }
    return written;
}

// The first type of each kind, direct or built, is written from the packet and the others copy it: a call site's
// payload struct can be a type of its own and also one argument of a built type. Direct types go first, so the record's
// time slot is a direct record's when there is one. `copies` notes the other copies' time slots, the direct ones first.
[[gnu::noinline]] StreamDecoder::Written StreamDecoder::write_record_routed(
    uint64_t types, const Packet& packet, const profiler::SpscLaneConsts& consts, uint64_t* values, SpscLane& lane) {
    Written written{.time_slot = nullptr, .value_count = 0};
    const uint8_t* first[2] = {nullptr, nullptr};  // by whether the type is direct
    RecordCopies& copies = copies_[&lane - lanes_.data()];
    copies.count = 0;
    copies.direct_count = 0;
    for (const uint64_t group : {types & direct_types_, types & ~direct_types_}) {
        for (uint64_t rest = group; rest != 0; rest &= rest - 1) {
            const uint32_t type = static_cast<uint32_t>(std::countr_zero(rest));
            const bool direct = has_type(direct_types_, type);
            uint8_t* const record = types_out_[type] + type_bytes_[type];
            if (first[direct] == nullptr) {
                written.value_count += write_record_to(type, packet, consts, values).value_count;
                first[direct] = record;
            } else {
                std::memcpy(record, first[direct], strides_[type]);
                type_bytes_[type] += strides_[type];
            }
            uint8_t* const time_slot = record + profiler::kSpscTimeAt;
            if (written.time_slot == nullptr) {
                written.time_slot = time_slot;
            } else {
                copies.direct_count += direct ? 1 : 0;
                copies.time_slots[copies.count++] = time_slot;
            }
        }
    }
    copies.copied_time = written.time_slot;
    copies.seq = decoded_seq_;
    return written;
}

// A wall-clock read is either correct or exactly 2^32 too high, because another RISC's read of the low word can land
// between the producer's low and high reads as the low word wraps. A step back therefore means the last record's high
// word was one too high, so its time moves down by kEpoch, and so do a direct record's Timestamps computed from its
// time, in each direct copy too. noinline, because the repair code inlined into the decode loop raises the loop's
// register pressure.
[[gnu::noinline]] bool StreamDecoder::repair_last(const SpscLane& lane, uint64_t next_ts) {
    if (lane.last_ts < kEpoch || lane.last_ts - next_ts > kEpoch || lane.last_time == nullptr) {
        return false;
    }
    reinterpret_cast<uint64_t*>(lane.last_time)[0] -= kEpoch;
    uint8_t* const record = lane.last_time - profiler::kSpscTimeAt;
    const std::span<const uint32_t> follows_time = lane.last_payload_to_record != nullptr
                                                       ? std::span(lane.last_payload_to_record->follows_time)
                                                       : std::span<const uint32_t>();
    for (const uint32_t at : follows_time) {
        reinterpret_cast<uint64_t*>(record + at)[0] -= kEpoch;
    }
    const RecordCopies& copies = copies_[&lane - lanes_.data()];
    if (copies.copied_time == lane.last_time && copies.seq > delivered_seq_) {
        for (uint32_t k = 0; k < copies.count; k++) {
            uint8_t* const copy_time = copies.time_slots[k];
            std::memcpy(copy_time, lane.last_time, sizeof(uint64_t));
            if (k < copies.direct_count) {
                for (const uint32_t at : follows_time) {
                    std::memcpy(copy_time - profiler::kSpscTimeAt + at, record + at, sizeof(uint64_t));
                }
            }
        }
    }
    return true;
}

// The high word a restarted lane's 16-bit timer slot is widened around: the chip's clock sync's, which stays within
// seconds of the lane's for however long the capture runs, or the lane's last before the sync has a node.
uint32_t StreamDecoder::timer_reference(const SpscLane& lane) const {
    if (clocks_ != nullptr) {
        const int64_t wall = clocks_->covered(dev_index_);
        if (wall > 0) {
            return static_cast<uint32_t>(static_cast<uint64_t>(wall) >> 32);
        }
    }
    return static_cast<uint32_t>(lane.consts.th_hi >> 32);
}

StreamDecoder::Produced StreamDecoder::decode_frames(
    size_t undelivered, const uint32_t* frames, std::span<const uint32_t> frame_words, Out out) {
    return writes_order_ ? decode<true>(undelivered, frames, frame_words, out)
                         : decode<false>(undelivered, frames, frame_words, out);
}

namespace {

// The control block holds the low 16 bits of the lane's timer high word, and its runtime id, as of the run's end. The
// high word is the one with those bits nearest `near`. Words before a sticky packet in the run used an older value that
// the frame doesn't carry, so decoding starts just past the run's last sticky packet, whose end this returns.
[[gnu::always_inline]] inline uint32_t resume(
    SpscLane& lane, const uint32_t* ctrl, uint32_t risc, const uint32_t* words, uint32_t run, uint32_t near) {
    using namespace profiler;
    uint16_t timer_low;
    std::memcpy(&timer_low, reinterpret_cast<const uint16_t*>(ctrl + kernel_profiler::SPSC_WIRE_TIMER_0) + risc, 2);
    uint32_t timer_hi = near + static_cast<int16_t>(timer_low - static_cast<uint16_t>(near));
    uint32_t runtime_id = ctrl[kernel_profiler::spsc_wire_prog_word(risc)];
    uint32_t pos = 0;
    for (uint32_t k = 0; k < run;) {
        const uint32_t packet_words = spsc_packet_words(words[k]);
        if (packet_words == 0) {
            break;
        }
        if (ppfmt::type(words[k]) != PacketType::Record) {
            (ppfmt::type(words[k]) == PacketType::TimerHigh ? timer_hi : runtime_id) = spsc_sticky_value(words + k);
            pos = k + packet_words;
        }
        k += packet_words;
    }
    lane.need_state = false;
    lane.cursor = SpscLane::kNoCursor;
    spsc_lane_consts_th(lane.consts, timer_hi);
    spsc_lane_consts_prog(lane.consts, runtime_id);
    return pos;
}

}  // namespace

template <bool kWritesOrder>
StreamDecoder::Produced StreamDecoder::decode(
    size_t undelivered, const uint32_t* frames, std::span<const uint32_t> frame_words, Out out) {
    using namespace profiler;
    // Locals rather than members, because the kernels store through byte pointers, so anything reached through `this`
    // would be reloaded after every store.
    const uint64_t delivered_seq = decoded_seq_ - undelivered;
    delivered_seq_ = delivered_seq;
    const uint64_t seq = ++decoded_seq_;
    uint64_t* const values_out = out.values;
    uint64_t value_count_out = 0;
    std::ranges::copy(out.types, types_out_.begin());
    type_bytes_ = {};
    uint32_t* order_out = out.order;
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
            uint32_t pos = lane.need_state ? resume(lane, ctrl, risc, words, run, timer_reference(lane)) : 0;
            if (lane.last_time_seq <= delivered_seq) {
                lane.last_time = nullptr;
            }
            uint8_t* const entry_time = lane.last_time;
            const uint64_t entry_ts = lane.last_ts;
            while (pos < run) {
                const uint32_t* const src = words + pos;
                // The producer publishes its tail only on packet boundaries, so a run holds whole packets. A record,
                // the common packet, has its length in word 0, which spares it spsc_packet_words' switch.
                const bool is_record = ppfmt::type(src[0]) == PacketType::Record;
                const uint32_t packet_words = is_record ? ppfmt::packet_words(src[0]) : spsc_packet_words(src[0]);
                TT_FATAL(
                    packet_words != 0 && packet_words <= run - pos,
                    "streaming profiler: undecodable word {:#010x} at offset {} of lane {}'s run of {}",
                    src[0],
                    pos,
                    lane_index,
                    run);
                uint32_t consumed = packet_words;
                if (is_record) {
                    const uint32_t id = src[0] & ppfmt::CALL_SITE_ID_MASK;
                    RecordRoutes::Route route = site_routes_[id];
                    if (!route.resolved) [[unlikely]] {
                        route = routes_.resolve(id);
                    }
                    // Until a medium or full record sets the cursor, a short record has no time and makes no record.
                    const bool bits16 = packet_words == route.bits16_packet_words;
                    const bool bits64 = packet_words == route.bits64_packet_words;
                    const uint64_t short_run_type = bits16 ? short_run_types_[id] : 0;
                    if (short_run_type != 0 && lane.cursor != SpscLane::kNoCursor) {
                        const uint32_t type = static_cast<uint32_t>(std::countr_zero(short_run_type));
                        const RecordRoutes::ShortRun& short_run = *routes_.short_runs_[type];
                        uint8_t* const dst = types_out_[type] + type_bytes_[type];
                        const SpscBlockResult block = spsc_short_records(
                            src,
                            static_cast<uint32_t>(readable_end - src),
                            (run - pos) / 2,
                            lane.cursor,
                            consts,
                            short_run.layout,
                            short_run_types_,
                            short_run_type,
                            dst);
                        consumed = 2 * block.count;
                        const uint64_t first = lane.cursor + (src[1] >> 16);
                        if (first < lane.last_ts) [[unlikely]] {
                            order_regressions += !repair_last(lane, first);
                        }
                        type_bytes_[type] += block.count * short_run.layout.stride;
                        lane.cursor = block.last_time;
                        lane.last_ts = block.last_time;
                        lane.last_time = dst + (block.count - 1) * short_run.layout.stride + kSpscTimeAt;
                        lane.last_payload_to_record = short_run.payload_to_record;
                        if constexpr (kWritesOrder) {
                            order_out = std::fill_n(order_out, block.count, type);
                        }
                    } else if (!bits16 || lane.cursor != SpscLane::kNoCursor) {
                        const uint64_t timestamp = bits16   ? lane.cursor + (src[1] >> 16)
                                                   : bits64 ? uint64_t{src[2]} << 32 | src[1]
                                                            : consts.th_hi | src[1];
                        const TimestampWidth width = bits16   ? TimestampWidth::Bits16
                                                     : bits64 ? TimestampWidth::Bits64
                                                              : TimestampWidth::Bits32;
                        if (route.bits16_packet_words != 0) {
                            lane.cursor = timestamp;
                        }
                        stalls += id == TT_STALL_CALL_SITE_ID ? 1u : 0u;
                        if (timestamp < lane.last_ts) [[unlikely]] {
                            order_regressions += !repair_last(lane, timestamp);
                        }
                        lane.last_ts = timestamp;
                        const Packet packet{
                            .src = src,
                            .payload_at = 1 + uint32_t(width),  // word 0, then the form's time words
                            .time = timestamp,
                            .payload_to_record = route.payload_to_record != nullptr
                                                     ? &(*route.payload_to_record)[uint32_t(width)]
                                                     : nullptr};
                        uint64_t* const values = values_out + value_count_out;
                        Written written{.time_slot = reinterpret_cast<uint8_t*>(&lane.unrouted_time), .value_count = 0};
                        if (std::has_single_bit(route.types)) {
                            written = write_record_to(std::countr_zero(route.types), packet, consts, values);
                        } else if (route.types != 0) {
                            written = write_record_routed(route.types, packet, consts, values, lane);
                        }
                        value_count_out += written.value_count;
                        lane.last_time = written.time_slot;
                        lane.last_payload_to_record =
                            (route.types & direct_types_) != 0 ? packet.payload_to_record : nullptr;
                        if constexpr (kWritesOrder) {
                            for (uint64_t types = route.types; types != 0; types &= types - 1) {
                                *order_out++ = std::countr_zero(types);
                            }
                        }
                    }
                } else if (ppfmt::type(src[0]) == PacketType::TimerHigh) {
                    spsc_lane_consts_th(consts, spsc_sticky_value(src));
                } else {
                    spsc_lane_consts_prog(consts, spsc_sticky_value(src));
                }
                pos += consumed;
            }
            if (lane.last_time != entry_time) {
                lane.last_time_seq = seq;
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
        .values = static_cast<uint32_t>(value_count_out),
        .type_bytes = type_bytes_,
        .order_entries = static_cast<uint32_t>(order_out - out.order),
        .newest_ticks = newest_ticks,
        .stalls = stalls,
        .order_regressions = order_regressions};
}

namespace {

constexpr size_t kArenaBytes = size_t{256} << 10;

// A batch decodes into one buffer holding its values, then a region per DevicePublish record type of its consumer's,
// then, for a consumer of more than one type, the order of its records across types, each sized for a full batch. Each
// value comes from two payload words, plus 32 bytes per frame for what spsc_published() stores past its last value. A
// DevicePublish record takes at most 32 bytes per word of its packet, which a short record reaches with a 16-byte
// Timestamp per 16-bit distance. A packet that becomes a record is at least two words, and each record of each type is
// at most one entry of the order.
constexpr size_t kBatchWords = profiler::kSpscBatchBytes / sizeof(uint32_t);
constexpr size_t kTypesAt = (kBatchWords * 4 + size_t{32} * profiler::kSpscBatchFrames + 63) / 64 * 64;
constexpr size_t kTypeBytes = (kBatchWords * 32 + profiler::kSpscRecordSlackBytes + 63) / 64 * 64;
constexpr size_t kOrderBytes = kBatchWords / 2 * sizeof(uint32_t);

}  // namespace

ConsumerBatches::ConsumerBatches(
    std::vector<experimental::device_observability::detail::RecordTypeDescriptor> record_types) :
    types_(std::move(record_types)),
    records_(types_.type_count()),
    counts_(types_.type_count()),
    arena_buf_(std::make_unique_for_overwrite<std::byte[]>(kArenaBytes)),
    arena_(arena_buf_.get(), kArenaBytes) {
    batch_.records = records_;
    batch_.counts = counts_;
}

DecodedFrames ConsumerBatches::decode(
    StreamDecoder& decoder, size_t undelivered, const uint32_t* frames, std::span<const uint32_t> frame_words) {
    DecodedFrames decoded;
    if (free_buffers_.empty()) {
        decoded.buffer =
            std::make_unique_for_overwrite<uint8_t[]>(kTypesAt + types_.type_count() * (kTypeBytes + kOrderBytes));
    } else {
        decoded.buffer = std::move(free_buffers_.back());
        free_buffers_.pop_back();
    }
    decoded.produced = decoder.decode_frames(undelivered, frames, frame_words, out_in(decoded.buffer.get()));
    return decoded;
}

const experimental::device_observability::detail::BatchData& ConsumerBatches::build_batch(
    DecodedFrames& frames,
    const RecordRoutes& routes,
    const ClockMap& map,
    ClockMap::Reader& reader,
    uint32_t dev,
    uint64_t dropped_bytes) {
    const StreamDecoder::Out out = out_in(frames.buffer.get());
    const StreamDecoder::Produced& produced = frames.produced;
    place(out, produced, map, reader, dev);
    arena_.release();
    for (uint32_t type = 0; type < types_.type_count(); type++) {
        counts_[type] = produced.type_bytes[type] / types_.stride(type);
        records_[type] = reinterpret_cast<const std::byte*>(out.types[type]);
        if (types_.built(type)) {
            const auto* published = std::launder(reinterpret_cast<const PublishedRecord*>(out.types[type]));
            records_[type] = build_records(std::span(published, counts_[type]), type, routes, arena_, map, reader, dev);
        }
    }
    batch_.order = std::span(out.order, produced.order_entries);
    batch_.dropped_bytes = dropped_bytes;
    batch_.stall_count = produced.stalls;
    return batch_;
}

namespace {

// Puts the time at byte `at` of four records on host time: a time is a device timestamp and then its tile's clock
// offset, which becomes its host time.
[[gnu::always_inline]] inline void place_four(
    const ClockMap& map, ClockMap::Reader& reader, uint32_t dev, const std::array<uint8_t*, 4>& records, uint32_t at) {
    const auto slot = [&](uint32_t k) { return reinterpret_cast<uint64_t*>(records[k] + at); };
    alignas(32) int64_t hosts[4];
    simde_mm256_store_si256(
        reinterpret_cast<simde__m256i*>(hosts),
        map.place_host4(
            reader,
            dev,
            simde_mm256_setr_epi64x(
                static_cast<int64_t>(slot(0)[0] + slot(0)[1]),
                static_cast<int64_t>(slot(1)[0] + slot(1)[1]),
                static_cast<int64_t>(slot(2)[0] + slot(2)[1]),
                static_cast<int64_t>(slot(3)[0] + slot(3)[1]))));
    for (uint32_t k = 0; k < 4; k++) {
        slot(k)[1] = static_cast<uint64_t>(hosts[k]);
    }
}

// Whether the Timestamp at byte `field` of each of four records is its record's own time, from the same tile, so it
// takes the time's host time.
[[gnu::always_inline]] inline bool at_record_time(const std::array<uint8_t*, 4>& records, uint32_t field) {
    return std::ranges::all_of(records, [&](const uint8_t* record) {
        return reinterpret_cast<const uint64_t*>(record + field)[0] ==
               reinterpret_cast<const uint64_t*>(record + profiler::kSpscTimeAt)[0];
    });
}

}  // namespace

void ConsumerBatches::place(
    const StreamDecoder::Out& out,
    const StreamDecoder::Produced& produced,
    const ClockMap& map,
    ClockMap::Reader& reader,
    uint32_t dev) {
    using namespace profiler;
    // A memmove onto itself compiles to nothing and implicitly creates the DevicePublish records and payload values
    // already in these bytes.
    std::memmove(out.values, out.values, size_t{produced.values} * sizeof(uint64_t));
    // Each place a record holds a time, the record's own time first, is placed four records at a time; the last four
    // repeat the type's last record as needed, which places it more than once to the same result.
    for (uint32_t type = 0; type < types_.type_count(); type++) {
        std::memmove(out.types[type], out.types[type], produced.type_bytes[type]);
        if (types_.built(type)) {
            continue;
        }
        const std::span<const uint32_t> times = types_.time_offsets(type);
        const uint32_t stride = types_.stride(type), count = produced.type_bytes[type] / stride;
        for (uint32_t first = 0; first < count; first += 4) {
            std::array<uint8_t*, 4> records;
            for (uint32_t k = 0; k < 4; k++) {
                records[k] = out.types[type] + std::min(first + k, count - 1) * stride;
            }
            place_four(map, reader, dev, records, kSpscTimeAt);
            for (const uint32_t field : times) {
                if (at_record_time(records, field)) {
                    for (uint8_t* const record : records) {
                        reinterpret_cast<uint64_t*>(record + field)[1] =
                            reinterpret_cast<const uint64_t*>(record + kSpscTimeAt)[1];
                    }
                } else {
                    place_four(map, reader, dev, records, field);
                }
            }
        }
    }
}

StreamDecoder::Out ConsumerBatches::out_in(uint8_t* buffer) {
    for (size_t type = 0; type < types_.type_count(); type++) {
        type_out_[type] = buffer + kTypesAt + type * kTypeBytes;
    }
    return {
        .values = reinterpret_cast<uint64_t*>(buffer),
        .types = std::span(type_out_.data(), types_.type_count()),
        .order = types_.type_count() > 1
                     ? reinterpret_cast<uint32_t*>(buffer + kTypesAt + types_.type_count() * kTypeBytes)
                     : nullptr};
}

}  // namespace tt::tt_metal::streaming_profiler
