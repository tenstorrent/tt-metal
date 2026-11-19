// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// PMK phase form of kernels/prefill/moe_plan_mc_count.cpp (tt/pmk_moe.py; kernels/pmk/moe_common.hpp): the ORIGINAL
// kernel's arithmetic, CB roles and argument order, with its compile-time args at CT0 + i, its runtime args at RT0 + i
// (the including composite defines both before this header, inside the phase's namespace) and its CBs at pmk_moe's
// indices. Generated from the original by doc/ttnn/16-prefill-production/pmk/moe/gen_phases.py -- edit the original,
// regenerate.
// ---- original header follows ----
// Stage 16 multi-core routing plan (tt/prefill_tp.py EpMoe, GLM53_PF_MOE_PLAN=mc), program 1 of 3: each core counts
// the local experts of its pair chunk [i CHUNK, min((i + 1) CHUNK, N)) into row i of hist ([P, EP] uint32 ROW_MAJOR,
// one page per core). Same pair / expert conventions as moe_plan_v2.cpp (pair p = j C + t, e = ids[p] - lo, local iff
// e < E). Compile-time: N, E, EP, CHUNK (a multiple of 16: 64-B aligned DRAM reads). Runtime: ids, lo, hist addresses,
// core index i. Scratch CB CB_PLAN: CHUNK + EP + 32 words.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

FORCE_INLINE void phase_main() {
    constexpr uint32_t N = get_compile_time_arg_val(CT0 + 0);
    constexpr uint32_t E = get_compile_time_arg_val(CT0 + 1);
    constexpr uint32_t EP = get_compile_time_arg_val(CT0 + 2);
    constexpr uint32_t CHUNK = get_compile_time_arg_val(CT0 + 3);
    static_assert(CHUNK % 16 == 0 && EP % 16 == 0, "64-B aligned chunks / rows");
    const uint32_t ids_addr = get_arg_val<uint32_t>(RT0 + 0);
    const uint32_t lo_addr = get_arg_val<uint32_t>(RT0 + 1);
    const uint32_t hist_addr = get_arg_val<uint32_t>(RT0 + 2);
    const uint32_t i = get_arg_val<uint32_t>(RT0 + 3);

    const uint32_t base = get_write_ptr(CB_PLAN);
    uint32_t* ids = reinterpret_cast<uint32_t*>(base);
    uint32_t* cnt = ids + CHUNK;
    uint32_t* lo_w = cnt + EP;  // 32 words
    const uint32_t p0 = i * CHUNK;
    const uint32_t n = p0 >= N ? 0 : (N - p0 < CHUNK ? N - p0 : CHUNK);

    const InterleavedAddrGen<true> lo_g{.bank_base_address = lo_addr, .page_size = 32 * 4};
    noc_async_read(lo_g.get_noc_addr(0), reinterpret_cast<uint32_t>(lo_w), 32 * 4);
    if (n > 0) {
#ifdef PMK_IDS_U16
        // v2 (lever 5): ids from the family's router tail = [C, K] uint16 ROW_MAJOR (16 B per token row); this core's
        // slot-major pairs [p0, p0 + n) are rows t0 .. of slot j0 (and the next slot when the chunk wraps): the rows
        // are read as whole 64-B groups (CHUNK is a multiple of 16: p0 % 16 == 0 -> t0 % 16 == 0) and expanded
        {
            constexpr uint32_t K_ = PMK_IDS_K;
            constexpr uint32_t C_ = N / K_;
            const InterleavedAddrGen<true> ids_g{.bank_base_address = ids_addr, .page_size = C_ * 2 * K_};
            uint32_t* raw = lo_w + 32;  // scratch past every table of this phase: up to CHUNK rows x 16 B (+64)
            uint32_t p = 0;
            while (p < n) {
                const uint32_t j = (p0 + p) / C_, t0 = (p0 + p) % C_;
                const uint32_t cnt_ = (C_ - t0 < n - p) ? C_ - t0 : n - p;
                noc_async_read(
                    ids_g.get_noc_addr(0, t0 * 2 * K_),
                    reinterpret_cast<uint32_t>(raw),
                    (cnt_ * 2 * K_ + 63) / 64 * 64);
                noc_async_read_barrier();
                volatile tt_l1_ptr uint16_t* r16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(raw);
                for (uint32_t q = 0; q < cnt_; ++q) {
                    ids[p + q] = r16[q * K_ + j];
                }
                p += cnt_;
            }
        }
#else
        const InterleavedAddrGen<true> ids_g{.bank_base_address = ids_addr, .page_size = N * 4};
        noc_async_read(ids_g.get_noc_addr(0, p0 * 4), reinterpret_cast<uint32_t>(ids), (n * 4 + 63) / 64 * 64);
#endif
    }
    noc_async_read_barrier();
    const uint32_t lo = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(lo_w);
    for (uint32_t e = 0; e < EP; ++e) {
        cnt[e] = 0;
    }
    for (uint32_t p = 0; p < n; ++p) {
        const uint32_t e = ids[p] - lo;  // wraps above E for another chip's expert below lo
        if (e < E) {
            cnt[e] += 1;
        }
    }
    asm volatile("fence" ::: "memory");  // the counts are in L1 before the NoC reads them
    const InterleavedAddrGen<true> hist_g{.bank_base_address = hist_addr, .page_size = EP * 4};
    noc_async_write(reinterpret_cast<uint32_t>(cnt), hist_g.get_noc_addr(i), EP * 4);
    noc_async_write_barrier();
}
