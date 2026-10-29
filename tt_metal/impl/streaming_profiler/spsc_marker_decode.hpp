// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// The vector kernels of the host-side decode, and the per-lane constants they compose records from.
// StreamDecoder::decode_frames walks the frames and runs these kernels on them. A frame is the SPSC_SPAN_PREFIX_WORDS
// prefix, whose word 1 is the payload length, then the SPSC_SPAN_WIRE_CTRL_WORDS control block, then each RISC's new
// ring words in turn, with alignment pads between them. The producer publishes its tail only on packet boundaries, so a
// RISC's words never end mid-packet.
#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>

// SIMDe uses native AVX2 where available and supported host instructions otherwise.
#include <simde/x86/avx2.h>

#include "hostdev/streaming_profiler_common.h"

namespace tt::tt_metal::profiler {

using kernel_profiler::PacketType;
using kernel_profiler::ppfmt;

inline constexpr uint32_t kSpscRingCap = kernel_profiler::PROFILER_L1_VECTOR_SIZE;
inline constexpr uint32_t kSpscRingMask = kSpscRingCap - 1;
inline constexpr uint32_t kSpscNRiscDecode = kernel_profiler::PROFILER_SPSC_TENSIX_RISC;

// Worst case: five full rings behind maximal pads. Bounds the bounce buffer and frame validation, not any
// device layout.
inline constexpr uint32_t kSpscMaxPayloadWords =
    kernel_profiler::PROFILER_L1_CONTROL_VECTOR_SIZE +
    kSpscNRiscDecode * (kSpscRingCap + kernel_profiler::SPSC_SPAN_PACK_ALIGN_WORDS - 1);
inline constexpr uint32_t kSpscMaxFrameWords = kernel_profiler::spsc_span_frame_words(kSpscMaxPayloadWords);
inline constexpr uint32_t kSpscMaxFramePages = kSpscMaxFrameWords / kernel_profiler::SPSC_SPAN_PAGE_WORDS;
static_assert(kSpscMaxFrameWords == 2656 && kSpscMaxFramePages == 166);
// The most frames one pass takes from a stream, and the most bytes they span.
inline constexpr uint32_t kSpscBatchFrames = 64;
inline constexpr size_t kSpscBatchBytes = size_t{kSpscBatchFrames} * kSpscMaxFrameWords * sizeof(uint32_t);

// Every record a consumer sees is a DevicePublish Record, written straight into its type's region of the batch buffer.
// Each starts with its origin, then its own time, {origin, time} in 8-byte words, then its payload as its type lays it
// out, or for a type its batch builds {count, pointer} with its arguments in the values region. An origin is the
// call-site id | its runtime id << 32, then coordinates | chip | processor. A time is a timestamp and then the tile's
// clock offset, which the consumer replaces with the time's host time; that saves the consumer a per-record lookup of
// the offset. Stores are cached rather than non-temporal because the consumer reads the buffers immediately.
inline constexpr uint32_t kSpscHeadBytes = 32;
inline constexpr uint32_t kSpscPublishedBytes = 48;
inline constexpr uint32_t kSpscTimeAt = 16;  // a record's time's byte offset

struct SpscBlockResult {
    uint64_t last_time;
    uint32_t count;
};

// The words of the packet that starts with `w0`, or 0 when no packet starts with it.
inline uint32_t spsc_packet_words(uint32_t w0) {
    switch (ppfmt::type(w0)) {
        case PacketType::Record: return ppfmt::packet_words(w0);
        case PacketType::TimerHigh:
        case PacketType::RuntimeId: return 1;
        case PacketType::RuntimeIdWide: return 2;
        default: return 0;
    }
}
// The value of a timer or runtime-id packet.
inline uint32_t spsc_sticky_value(const uint32_t* p) {
    return ppfmt::type(p[0]) == PacketType::RuntimeIdWide ? p[1] : p[0] & ppfmt::VALUE_MASK;
}

// The values all of a lane's records share, where a lane is the record stream of one RISC on one core. `shared` is the
// constant part of a record's origin and time: the lane's address (its RecordLane), then 32-bit words {0, runtime id},
// then {0, timer high word}, then the clock offset. The kernel that decodes 2-word packets works on several records at
// once, so it needs values repeated in every slot of a SIMD register: prog_hi_v repeats the runtime id shifted up 32
// bits, and tail the lane's address and clock offset.
struct SpscLaneConsts {
    uint64_t th_hi;
    simde__m256i prog_hi_v;
    simde__m256i tail;  // {the lane's address, offset} in each 128-bit half
    simde__m256i shared;
};
// The 32-bit words of a record's origin and time that come from the packet: the call site's id and the time's low word.
// Every other word, the lane's address, the runtime id, the time's high word and the clock offset, comes from the lane.
inline constexpr int kSpscIdDword = 2, kSpscRuntimeIdDword = 3, kSpscTimeLowDword = 4, kSpscTimeHighDword = 5;
inline constexpr int kSpscBlendShared = 0xFF & ~(1 << kSpscIdDword | 1 << kSpscTimeLowDword);

inline void spsc_lane_consts_th(SpscLaneConsts& consts, uint32_t timer_hi) {
    consts.th_hi = static_cast<uint64_t>(timer_hi) << 32;
    consts.shared = simde_mm256_blend_epi32(
        consts.shared, simde_mm256_set1_epi32(static_cast<int>(timer_hi)), 1 << kSpscTimeHighDword);
}
inline void spsc_lane_consts_prog(SpscLaneConsts& consts, uint32_t runtime_id) {
    consts.prog_hi_v = simde_mm256_set1_epi64x(static_cast<long long>(static_cast<uint64_t>(runtime_id) << 32));
    consts.shared = simde_mm256_blend_epi32(
        consts.shared, simde_mm256_set1_epi32(static_cast<int>(runtime_id)), 1 << kSpscRuntimeIdDword);
}

inline simde__m256i spsc_lane_idx() { return simde_mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7); }

// Records of the type in a quad's compare result, counted from lane 0 to the first miss. A full quad takes one test,
// and only a partial one extracts its mask.
inline uint32_t spsc_quad_count(simde__m256i matches) {
    if (simde_mm256_testc_si256(matches, simde_mm256_cmpeq_epi64(matches, matches))) {
        return 4;
    }
    return std::countr_zero(~static_cast<uint32_t>(simde_mm256_movemask_pd(simde_mm256_castsi256_pd(matches))) & 0x1Fu);
}

// How spsc_short_records() writes a short DevicePublish record of 2 words as a record of a type the decoder writes in
// place: its origin and time, then 32 bytes whose qwords are each the record's time, its payload's 16 bits (a distance
// back from that time, or bytes the type keeps), or the tile's clock offset.
struct SpscShortLayout {
    // The qwords a record's payload qwords are picked from.
    enum Qword : int32_t { kTime, kPayload, kSource, kOffset };
    // The shuffle index that writes a zero byte.
    static constexpr uint8_t kZeroByte = 0x80;
    // Where a record's payload starts in its qword of the run: word 1's low half.
    static constexpr uint32_t kPayloadByte = 4;
    // The record's head and the 32 bytes of qwords `payload_dwords` fills.
    static constexpr uint32_t kMaxStride = kSpscHeadBytes + 32;

    // The dword of the Qwords each of the record's payload dwords comes from.
    std::array<int32_t, 8> payload_dwords{};
    // Where the payload qword's bytes come from in the record's qword of the run, kZeroByte for none. It is a byte
    // shuffle within each 128-bit half, which holds two records.
    std::array<uint8_t, 32> payload_shuffle{};
    // All ones when the payload is a distance, which the payload qword holds as the time less it.
    int64_t distance_mask = 0;
    uint32_t stride = 0;
};

// A record is one qword of the run (word 0 low, word 1 high), four to a 256-bit load. Its time is the lane's cursor
// plus the inclusive prefix sum of the cursor distances in its payload's bytes 2 and 3. The four quads' in-lane
// prefixes are independent and their totals carry as a tree, so no dependency chain spans a block of 16. A record
// belongs to the run when its call site's bit in `short_run_types` is `run_type`. The kernel stops at the first record
// that doesn't, or after `max_records`. `readable` authorizes loads, never records, past the run. Records are written
// in order, so the 32 bytes past a record shorter than 64 bytes land in the next record or in its region's slack.
[[gnu::noinline]] inline SpscBlockResult spsc_short_records(
    const uint32_t* words,
    uint32_t readable,
    uint32_t max_records,
    uint64_t cursor,
    const SpscLaneConsts& consts,
    const SpscShortLayout& layout,
    const uint64_t* short_run_types,
    uint64_t run_type,
    uint8_t* dst) {
    // A 2-word record's type and packet words, in word 0's bits 20 to 31. Word 0's low 16 bits are the record's id,
    // word 1's high half its delta from the cursor and its low half its payload.
    const simde__m256i key_mask = simde_mm256_set1_epi64x(0xFFF00000);
    const simde__m256i key = simde_mm256_set1_epi64x(ppfmt::record_w0(2));
    const simde__m256i low16 = simde_mm256_set1_epi64x(0xFFFF);
    const simde__m256i zero = simde_mm256_setzero_si256();
    const simde__m256i offset = simde_mm256_unpackhi_epi64(consts.tail, consts.tail);
    const simde__m256i lane = simde_mm256_unpacklo_epi64(consts.tail, consts.tail);
    const simde__m256i payload_dwords =
        simde_mm256_loadu_si256(reinterpret_cast<const simde__m256i*>(layout.payload_dwords.data()));
    const simde__m256i payload_shuffle =
        simde_mm256_loadu_si256(reinterpret_cast<const simde__m256i*>(layout.payload_shuffle.data()));
    const simde__m256i distance_mask = simde_mm256_set1_epi64x(layout.distance_mask);
    const uint32_t stride = layout.stride;
    simde__m256i cursor_v = simde_mm256_set1_epi64x(static_cast<long long>(cursor));
    const auto write = [&](simde__m256i record) {
        // {the lane's address, id and runtime id, time, offset}, the address blended into qword 0.
        simde_mm256_storeu_si256(
            reinterpret_cast<simde__m256i*>(dst),
            simde_mm256_blend_epi32(simde_mm256_permute4x64_epi64(record, 0xC8), lane, 0x03));
        simde_mm256_storeu_si256(
            reinterpret_cast<simde__m256i*>(dst + 32), simde_mm256_permutevar8x32_epi32(record, payload_dwords));
        dst += stride;
    };
    const auto write_four = [&](simde__m256i quad, simde__m256i prefix, uint32_t count) {
        const simde__m256i time = simde_mm256_add_epi64(cursor_v, prefix);
        const simde__m256i payload = simde_mm256_or_si256(
            simde_mm256_and_si256(
                simde_mm256_sub_epi64(time, simde_mm256_and_si256(simde_mm256_srli_epi64(quad, 32), low16)),
                distance_mask),
            simde_mm256_shuffle_epi8(quad, payload_shuffle));
        const simde__m256i origin = simde_mm256_or_si256(simde_mm256_and_si256(quad, low16), consts.prog_hi_v);
        const simde__m256i tp_lo = simde_mm256_unpacklo_epi64(time, payload);
        const simde__m256i tp_hi = simde_mm256_unpackhi_epi64(time, payload);
        const simde__m256i oo_lo = simde_mm256_unpacklo_epi64(origin, offset);
        const simde__m256i oo_hi = simde_mm256_unpackhi_epi64(origin, offset);
        write(simde_mm256_permute2x128_si256(tp_lo, oo_lo, 0x20));
        if (count > 1) {
            write(simde_mm256_permute2x128_si256(tp_hi, oo_hi, 0x20));
        }
        if (count > 2) {
            write(simde_mm256_permute2x128_si256(tp_lo, oo_lo, 0x31));
        }
        if (count > 3) {
            write(simde_mm256_permute2x128_si256(tp_hi, oo_hi, 0x31));
        }
    };
    uint32_t total = 0;
    while (max_records != 0 && readable >= 2u) {
        simde__m256i quads[4];
        if (readable >= 32u) {
            for (int i = 0; i < 4; i++) {
                quads[i] = simde_mm256_loadu_si256(reinterpret_cast<const simde__m256i*>(words + 8 * i));
            }
        } else {
            // Whole records only: a trailing odd word is never a record's, so its lane reads as zero and ends the run.
            const int records = static_cast<int>(readable / 2u);
            for (int i = 0; i < 4; i++) {
                const simde__m256i mask = simde_mm256_cmpgt_epi64(
                    simde_mm256_set1_epi64x(records - 4 * i), simde_mm256_setr_epi64x(0, 1, 2, 3));
                quads[i] = simde_mm256_maskload_epi64(reinterpret_cast<const int64_t*>(words + 8 * i), mask);
            }
        }
        // Both lines of the block four blocks ahead.
        simde_mm_prefetch(reinterpret_cast<const char*>(words + 128), SIMDE_MM_HINT_T0);
        simde_mm_prefetch(reinterpret_cast<const char*>(words + 144), SIMDE_MM_HINT_T0);
        uint32_t count = 0;
        for (const auto& quad : quads) {
            const uint32_t in_quad =
                spsc_quad_count(simde_mm256_cmpeq_epi64(simde_mm256_and_si256(quad, key_mask), key));
            count += in_quad;
            if (in_quad < 4u) {
                break;
            }
        }
        count = std::min(count, max_records);
        uint64_t types = ~uint64_t{0};
// Clang's gather of these loads is slower than the loads one by one.
#if defined(__clang__)
#pragma clang loop vectorize(disable)
#endif
        for (uint32_t k = 0; k < count; k++) {
            types &= short_run_types[static_cast<uint16_t>(words[2 * k])];
        }
        if ((types & run_type) == 0) [[unlikely]] {
            count = 0;
            while ((short_run_types[static_cast<uint16_t>(words[2 * count])] & run_type) != 0) {
                count++;
            }
        }
        if (count == 0) {
            break;
        }
        simde__m256i prefixes[4];
        for (int i = 0; i < 4; i++) {
            simde__m256i distances = simde_mm256_srli_epi64(quads[i], 48);
            distances = simde_mm256_add_epi64(distances, simde_mm256_slli_si256(distances, 8));
            prefixes[i] = simde_mm256_add_epi64(
                distances, simde_mm256_blend_epi32(simde_mm256_permute4x64_epi64(distances, 0x55), zero, 0x0F));
        }
        const simde__m256i carry1 = simde_mm256_permute4x64_epi64(prefixes[0], 0xFF);
        const simde__m256i carry2 = simde_mm256_add_epi64(carry1, simde_mm256_permute4x64_epi64(prefixes[1], 0xFF));
        const simde__m256i carry3 = simde_mm256_add_epi64(carry2, simde_mm256_permute4x64_epi64(prefixes[2], 0xFF));
        prefixes[1] = simde_mm256_add_epi64(prefixes[1], carry1);
        prefixes[2] = simde_mm256_add_epi64(prefixes[2], carry2);
        prefixes[3] = simde_mm256_add_epi64(prefixes[3], carry3);
        if (count == 16u) [[likely]] {
            for (uint32_t i = 0; i < 4; i++) {
                write_four(quads[i], prefixes[i], 4);
            }
        } else {
            for (uint32_t i = 0; i < 4; i++) {
                if (4 * i < count) {
                    write_four(quads[i], prefixes[i], std::min(count - 4 * i, 4u));
                }
            }
        }
        total += count;
        if (count < 16u) {
            alignas(32) uint64_t prefix_words[16];
            for (int i = 0; i < 4; i++) {
                simde_mm256_store_si256(reinterpret_cast<simde__m256i*>(prefix_words + 4 * i), prefixes[i]);
            }
            return {
                static_cast<uint64_t>(simde_mm_cvtsi128_si64(simde_mm256_castsi256_si128(cursor_v))) +
                    prefix_words[count - 1],
                total};
        }
        cursor_v = simde_mm256_add_epi64(cursor_v, simde_mm256_permute4x64_epi64(prefixes[3], 0xFF));
        words += 32;
        readable -= 32;
        max_records -= 16;
    }
    return {static_cast<uint64_t>(simde_mm_cvtsi128_si64(simde_mm256_castsi256_si128(cursor_v))), total};
}

// Loads up to 8 words from `words`, and zeros from `count` on.
inline simde__m256i spsc_load_words(const uint32_t* words, uint32_t count) {
    return simde_mm256_maskload_epi32(
        reinterpret_cast<const int*>(words),
        simde_mm256_cmpgt_epi32(simde_mm256_set1_epi32(static_cast<int>(count)), spsc_lane_idx()));
}
// A record packet's origin and time from its words and the lane's constants: its id, and word 1 as its time's low word.
inline simde__m256i spsc_record_head(simde__m256i words, const SpscLaneConsts& consts) {
    return simde_mm256_blend_epi32(
        simde_mm256_permutevar8x32_epi32(
            simde_mm256_and_si256(words, simde_mm256_setr_epi32(ppfmt::CALL_SITE_ID_MASK, -1, -1, -1, -1, -1, -1, -1)),
            simde_mm256_setr_epi32(0, 0, 0, 0, 1, 0, 0, 0)),
        consts.shared,
        kSpscBlendShared);
}

// The last store's lanes past the payload land in slack that the next record overwrites, so a region needs
// kSpscRecordSlackBytes past its records.
inline constexpr uint32_t kSpscRecordSlackBytes = 32;
inline void spsc_record(const uint32_t* words, uint32_t payload_words, const SpscLaneConsts& consts, uint8_t* dst) {
    simde_mm256_storeu_si256(reinterpret_cast<simde__m256i*>(dst), spsc_record_head(spsc_load_words(words, 2), consts));
    for (uint32_t k = 0; k < payload_words; k += 8) {
        simde_mm256_storeu_si256(
            reinterpret_cast<simde__m256i*>(dst + kSpscHeadBytes + 4 * k),
            spsc_load_words(words + 2 + k, payload_words - k));
    }
}

// The caller writes the record's time, which a short or full record's head doesn't hold. The count, the length, the
// pointer and the first four payload words are always stored, and only a longer payload goes through the loop.
inline uint32_t spsc_published(
    const uint32_t* words,
    uint32_t payload_at,
    uint32_t payload_words,
    const SpscLaneConsts& consts,
    uint8_t* dst,
    uint64_t* values) {
    const uint32_t value_count = (payload_words + 1u) >> 1;
    const uint64_t bytes_and_length = uint64_t{payload_words} * 4u | uint64_t{ppfmt::packet_words(words[0])} << 32;
    const simde__m256i head_words = spsc_load_words(words, payload_at + payload_words);
    simde_mm256_storeu_si256(reinterpret_cast<simde__m256i*>(dst), spsc_record_head(head_words, consts));
    simde_mm_storeu_si128(
        reinterpret_cast<simde__m128i*>(dst + kSpscHeadBytes),
        simde_mm_set_epi64x(reinterpret_cast<long long>(values), static_cast<long long>(bytes_and_length)));
    simde_mm_storeu_si128(
        reinterpret_cast<simde__m128i*>(values),
        simde_mm256_castsi256_si128(simde_mm256_permutevar8x32_epi32(
            head_words, simde_mm256_add_epi32(simde_mm256_set1_epi32(static_cast<int>(payload_at)), spsc_lane_idx()))));
    if (__builtin_expect(payload_words > 4u, 0)) {
        for (uint32_t k = 4; k < payload_words; k += 8) {
            simde_mm256_storeu_si256(
                reinterpret_cast<simde__m256i*>(values + k / 2),
                spsc_load_words(words + payload_at + k, payload_words - k));
        }
    }
    return value_count;
}

}  // namespace tt::tt_metal::profiler
