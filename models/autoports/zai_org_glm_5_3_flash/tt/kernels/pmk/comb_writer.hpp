// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// PMK phase form of kernels/prefill/moe_combine_rm_writer.cpp (tt/pmk_moe.py; kernels/pmk/moe_common.hpp): the ORIGINAL
// kernel's arithmetic, CB roles and argument order, with its compile-time args at CT0 + i, its runtime args at RT0 + i
// (the including composite defines both before this header, inside the phase's namespace) and its CBs at pmk_moe's
// indices. Generated from the original by doc/ttnn/16-prefill-production/pmk/moe/gen_phases.py -- edit the original,
// regenerate.
// ---- original header follows ----
// Stage 16 combine "rm" -- writer (moe_combine_rm_reader.cpp): unit u's 32 PT bf16 tiles = tile row u / H_SPLIT,
// tile columns (u % H_SPLIT) 32 PT .. of y [C, H] TILE. Compile-time: PT, HT (= H / 32), H_SPLIT. Runtime: y address,
// first unit, unit count.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

FORCE_INLINE void phase_main() {
    constexpr uint32_t PT = get_compile_time_arg_val(CT0 + 0);
    constexpr uint32_t HT = get_compile_time_arg_val(CT0 + 1);
    constexpr uint32_t H_SPLIT = get_compile_time_arg_val(CT0 + 2);
    constexpr uint32_t TILE_BYTES = 2048;
#ifdef PMK_OUT_BFP8
    constexpr uint32_t NT = PMK_OUT_NT;  // the part's column tiles (the combine's output is TILE already)
#else
    constexpr uint32_t NT = 32 * PT;
#endif
    const uint32_t y_addr = get_arg_val<uint32_t>(RT0 + 0);
    const uint32_t u0 = get_arg_val<uint32_t>(RT0 + 1);
    const uint32_t nu = get_arg_val<uint32_t>(RT0 + 2);
    if (nu == 0) {
        return;
    }
    const InterleavedAddrGen<true> yg{.bank_base_address = y_addr, .page_size = TILE_BYTES};
#if !defined(PMK_COMB_STRIDE)
#define PMK_COMB_STRIDE \
    1  // units u0, u0 + 1, .. (contiguous); PMK_COMB_STRIDE = P: u0 + k P (the block halves land in rounds)
#endif
    for (uint32_t k_u = 0; k_u < nu; ++k_u) {
#if defined(PMK_COMB_UNITS)  // a per-rank start (the half this chip SENDS first in a streamed P15) wraps around
        const uint32_t u = (u0 + k_u * PMK_COMB_STRIDE) % PMK_COMB_UNITS;
#else
        const uint32_t u = u0 + k_u * PMK_COMB_STRIDE;
#endif
        const uint32_t chunk = u / H_SPLIT;
        const uint32_t base = chunk * HT + (u - chunk * H_SPLIT) * NT;
        cb_wait_front(CB_COUT, NT);
#if defined(PMK_BW_STAGE) && PMK_BW_STAGE == 5  // the M-B1 combine bisect: COUT consumed, y not written (no H9)
        cb_pop_front(CB_COUT, NT);
        continue;
#endif
        const uint32_t rp = get_read_ptr(CB_COUT);
        for (uint32_t x = 0; x < NT; ++x) {
            noc_async_write(rp + x * TILE_BYTES, yg.get_noc_addr(base + x), TILE_BYTES);
        }
        noc_async_write_barrier();
        PMK_H9_SIGNAL(chunk);  // H9: this unit of row block `chunk` landed (pmk_moe.py h9=)
        cb_pop_front(CB_COUT, NT);
    }
}
