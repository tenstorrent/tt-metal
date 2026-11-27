// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// PMK assembly: the mHC POST's coefficient tiles from DRAM (ruling 2026-10-22 ~20:30Z, closure 1: the layer runs as
// several programs, so the K / L tiles the root unicast into CB_PCOEF during the PRE (another program) are rebuilt
// here from the DRAM post [T, 4] / comb [T, 16] pages the root wrote with F_PC_DRAM -- the 7a mhc_post_pf7a reader
// pattern, in the mhc_sync.cpp root layout: per stream s, tile K_s: column 0 = post_s, 1 = C[0][s], 16 = C[1][s],
// 17 = C[2][s]; tile L_s: column 0 = C[3][s]; every other element 0 (the broadcast masks multiply them).
// Member NCRISC, before the mhc P10 / P16 sync part: reserves NRB x 8 pages of CB_PCOEF (the ring is exactly that:
// the post pops 8 per block), fills block k = rb0 + k x step, pushes. Scratch: the front of CB_PSTAGE (free before
// the post). CT at PH_CT0: [cb_pcoef, cb_stage] + TensorAccessorArgs(post) + TensorAccessorArgs(comb).
// RT at ph_rt0: [post address, comb address, rb0, step, nrb].
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace pcd {
constexpr uint32_t CB_PC = get_compile_time_arg_val(PH_CT0 + 0);
constexpr uint32_t CB_ST = get_compile_time_arg_val(PH_CT0 + 1);
constexpr uint32_t NS = 4;     // streams
constexpr uint32_t TB = 4096;  // fp32 tile
constexpr uint32_t TILE = 32;
constexpr uint32_t tile_off(uint32_t r, uint32_t c) {
    return (((r >> 4) << 1) + (c >> 4)) * 256u + (r & 15u) * 16u + (c & 15u);
}
}  // namespace pcd

void kernel_main() {
    using namespace pcd;
    constexpr auto post_args = TensorAccessorArgs<PH_CT0 + 2>();
    constexpr auto comb_args = TensorAccessorArgs<post_args.next_compile_time_args_offset()>();
    const uint32_t a = ph_rt0;
    const auto post = TensorAccessor(post_args, get_arg_val<uint32_t>(a + 0));
    const auto comb = TensorAccessor(comb_args, get_arg_val<uint32_t>(a + 1));
    const uint32_t rb0 = get_arg_val<uint32_t>(a + 2);
    const uint32_t step = get_arg_val<uint32_t>(a + 3);
    const uint32_t nrb = get_arg_val<uint32_t>(a + 4);
    cb_reserve_back(CB_PC, nrb * 2 * NS);
    const uint32_t pc0 = get_write_ptr(CB_PC);
    cb_reserve_back(CB_ST, 2);
    const uint32_t scr = get_write_ptr(CB_ST);
    volatile tt_l1_ptr uint32_t* po = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr);
    volatile tt_l1_ptr uint32_t* co = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr + TB);
    for (uint32_t k = 0; k < nrb; ++k) {
        const uint32_t rb = rb0 + k * step;
        noc_async_read_page(rb, post, scr);
        noc_async_read_page(rb, comb, scr + TB);
        noc_async_read_barrier();
        invalidate_l1_cache();
        const uint32_t pc = pc0 + k * 2 * NS * TB;
        volatile tt_l1_ptr uint32_t* z = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pc);
        for (uint32_t i = 0; i < 2 * NS * TB / 4; ++i) {
            z[i] = 0;
        }
        for (uint32_t s = 0; s < NS; ++s) {
            volatile tt_l1_ptr uint32_t* k_t = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pc + 2 * s * TB);
            volatile tt_l1_ptr uint32_t* l_t = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pc + (2 * s + 1) * TB);
            for (uint32_t t = 0; t < TILE; ++t) {
                k_t[tile_off(t, 0)] = po[tile_off(t, s)];
                k_t[tile_off(t, 1)] = co[tile_off(t, 0 * NS + s)];
                k_t[tile_off(t, 16)] = co[tile_off(t, 1 * NS + s)];
                k_t[tile_off(t, 17)] = co[tile_off(t, 2 * NS + s)];
                l_t[tile_off(t, 0)] = co[tile_off(t, 3 * NS + s)];
            }
        }
    }
    cb_push_back(CB_PC, nrb * 2 * NS);
}
