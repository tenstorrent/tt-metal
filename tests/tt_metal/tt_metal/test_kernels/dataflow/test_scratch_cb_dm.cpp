// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// DM side of the scratch CB test (experimental/scratch_cb_api.h). `role` selects producer (channel 0, BRISC) or
// consumer (channel 1, NCRISC); `pattern` selects:
//
//   A  datacopy: in[i] -> ring A slot (i % capacity) -> compute -> ring B slot -> out[i]. The producer stamps
//      word 0 of every transfer with a sequence tag, and the consumer checks every word of every transfer.
//      `nosync`=1 drops reserve/wait (negative control: must corrupt). `real_cbs`=1 also passes one entry per
//      transfer through real CB (i % 62), so every ID the CB API hands out is live alongside scratch.
//   B  bounded producer: the producer pushes num_iters pages in batches of `batch` against a slow UNPACK
//      consumer and records the most pages ever in flight. `nosync`=1 drops reserve (negative control).
//   C  ping-pong: the producer alone owns both channels at capacity 1. Round i sends ring B's tile from round
//      i - 1 back to compute through ring A, so every round depends on the previous one, and it checks every
//      word that comes back.
//
// Report: A consumer {scratch errors, real CB errors}; B producer {pages produced, in-flight high-water mark};
// C producer {scratch errors}.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#include "experimental/scratch_cb_api.h"

namespace {

constexpr std::uint32_t PATTERN_A = 0;
constexpr std::uint32_t PATTERN_B = 1;
constexpr std::uint32_t PATTERN_C = 2;
constexpr std::uint32_t ROLE_PRODUCER = 0;
constexpr std::uint32_t ROLE_CONSUMER = 1;
constexpr std::uint32_t kRealCbs = 62;

// Mirrored by the host: two finite bf16 values encoding the transfer number, including across counter wrap.
inline std::uint32_t sequence_tag(std::uint32_t sequence) {
    return ((0x4000u | ((sequence >> 8) & 0x1fffu)) << 16) | (0x3f00u | (sequence & 0xffu));
}

inline std::uint32_t real_cb_word(std::uint32_t sequence, std::uint32_t word) {
    return 0xCB000000u ^ (sequence << 4) ^ word;
}

inline volatile tt_l1_ptr std::uint32_t* l1(std::uint32_t addr) {
    return reinterpret_cast<volatile tt_l1_ptr std::uint32_t*>(addr);
}

}  // namespace

void kernel_main() {
    constexpr std::uint32_t pattern = get_arg(args::pattern);
    constexpr std::uint32_t role = get_arg(args::role);
    constexpr std::uint32_t capacity = get_arg(args::capacity);
    constexpr std::uint32_t nosync = get_arg(args::nosync);
    constexpr std::uint32_t batch = get_arg(args::batch);
    constexpr std::uint32_t real_cbs = get_arg(args::real_cbs);
    const std::uint32_t num_iters = get_arg(args::num_iters);
    const std::uint32_t num_tiles = get_arg(args::num_tiles);
    const std::uint32_t tile_bytes = get_arg(args::tile_bytes);
    const std::uint32_t in_addr = get_arg(args::in_addr);
    const std::uint32_t ring_addr = get_arg(args::ring_addr);
    const std::uint32_t ring_b_addr = get_arg(args::ring_b_addr);
    const std::uint32_t out_addr = get_arg(args::out_addr);
    const std::uint32_t report_addr = get_arg(args::report_addr);

    if constexpr (pattern == PATTERN_A && role == ROLE_PRODUCER) {
        for (std::uint32_t i = 0; i < num_iters; ++i) {
            const std::uint32_t slot = ring_addr + (i % capacity) * tile_bytes;
            if constexpr (!nosync) {
                experimental::scratch_reserve_back<0, capacity>(1, slot);
            }
            noc_async_read(get_noc_addr(in_addr + (i % num_tiles) * tile_bytes), slot, tile_bytes);
            noc_async_read_barrier();
            // Every transfer has a new tag: stale data from an earlier round must not pass.
            l1(slot)[0] = sequence_tag(i);
            if constexpr (real_cbs) {
                DataflowBuffer real_cb(static_cast<std::uint16_t>(i % kRealCbs));
                real_cb.reserve_back(1);
                auto* entry = l1(real_cb.get_write_ptr());
                for (std::uint32_t word = 0; word < real_cb.get_entry_size() / sizeof(std::uint32_t); ++word) {
                    entry[word] = real_cb_word(i, word);
                }
                real_cb.push_back(1);
            }
            experimental::scratch_push_back<0, capacity>(1, slot);
        }
    }

    if constexpr (pattern == PATTERN_A && role == ROLE_CONSUMER) {
        std::uint32_t errors = 0;
        std::uint32_t real_cb_errors = 0;
        for (std::uint32_t i = 0; i < num_iters; ++i) {
            const std::uint32_t slot = ring_addr + (i % capacity) * tile_bytes;
            if constexpr (!nosync) {
                experimental::scratch_wait_front<1, capacity>(1, slot);
            }
            invalidate_l1_cache();
            const auto* original = l1(in_addr + (i % num_tiles) * tile_bytes);
            for (std::uint32_t word = 0; word < tile_bytes / sizeof(std::uint32_t); ++word) {
                if (l1(slot)[word] != (word == 0 ? sequence_tag(i) : original[word])) {
                    ++errors;
                    break;
                }
            }
            noc_async_write(slot, get_noc_addr(out_addr + (i % num_tiles) * tile_bytes), tile_bytes);
            noc_async_write_barrier();
            if constexpr (real_cbs) {
                DataflowBuffer real_cb(static_cast<std::uint16_t>(i % kRealCbs));
                real_cb.wait_front(1);
                invalidate_l1_cache();
                const auto* entry = l1(real_cb.get_read_ptr());
                for (std::uint32_t word = 0; word < real_cb.get_entry_size() / sizeof(std::uint32_t); ++word) {
                    if (entry[word] != real_cb_word(i, word)) {
                        ++real_cb_errors;
                        break;
                    }
                }
                real_cb.pop_front(1);
            }
            experimental::scratch_pop_front<1, capacity>(1, slot);
        }
        l1(report_addr)[0] = errors;
        l1(report_addr)[1] = real_cb_errors;
    }

    if constexpr (pattern == PATTERN_B && role == ROLE_PRODUCER) {
        const std::uintptr_t acked_ptr =
            reinterpret_cast<std::uintptr_t>(get_cb_tiles_acked_ptr(experimental::scratch_cb_detail::cb_id<0>()));
        std::uint32_t high_water = 0;
        for (std::uint32_t i = 0; i < num_iters; i += batch) {
            if constexpr (!nosync) {
                experimental::scratch_reserve_back<0, capacity>(batch);
            }
            experimental::scratch_push_back<0, capacity>(batch);
            const std::uint16_t in_flight = static_cast<std::uint16_t>(i + batch - reg_read(acked_ptr));
            high_water = in_flight > high_water ? in_flight : high_water;
        }
        l1(report_addr)[0] = num_iters;
        l1(report_addr)[1] = high_water;
    }

    static_assert(
        pattern != PATTERN_C || capacity == 1,
        "ping-pong holds ring B while it refills ring A, so each ring has one slot");
    if constexpr (pattern == PATTERN_C && role == ROLE_PRODUCER) {
        std::uint32_t errors = 0;
        const auto* original = l1(in_addr);
        for (std::uint32_t i = 0; i < num_iters; ++i) {
            experimental::scratch_reserve_back<0, capacity>(1, ring_addr);
            noc_async_read(get_noc_addr(i == 0 ? in_addr : ring_b_addr), ring_addr, tile_bytes);
            noc_async_read_barrier();
            if (i > 0) {
                experimental::scratch_pop_front<1, capacity>(1, ring_b_addr);
            }
            l1(ring_addr)[0] = sequence_tag(i);
            experimental::scratch_push_back<0, capacity>(1, ring_addr);

            experimental::scratch_wait_front<1, capacity>(1, ring_b_addr);
            invalidate_l1_cache();
            for (std::uint32_t word = 0; word < tile_bytes / sizeof(std::uint32_t); ++word) {
                if (l1(ring_b_addr)[word] != (word == 0 ? sequence_tag(i) : original[word])) {
                    ++errors;
                    break;
                }
            }
        }
        noc_async_write(ring_b_addr, get_noc_addr(out_addr), tile_bytes);
        noc_async_write_barrier();
        experimental::scratch_pop_front<1, capacity>(1, ring_b_addr);
        l1(report_addr)[0] = errors;
    }
}
