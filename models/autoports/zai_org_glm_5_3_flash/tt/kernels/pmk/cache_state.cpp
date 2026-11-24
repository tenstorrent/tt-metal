// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// PMK phase P9 (prefill_megakernel.md §3; tt/pmk_cache.py state_parts): every segment's final KDA state -> its slot
// in the decode stack's state store, inside the layer program, under P8. The same bytes and layout as the op path's
// kernels/prefill/state_io.cpp STATE_OUT (Stage 16's StateIO), per segment and spread over cores:
//   S : the segment's final state [1, H_r, 128, 128] fp32 TILE (N_S = 16 H_r pages, DRAM or L1 interleaved) -> stack
//       S of chunk slot / CHUNK [CHUNK, H_r, 128, 128] fp32, pages [rd N_S, (rd + 1) N_S), rd = slot % CHUNK;
//       this core copies pages [first_s, first_s + count_s) of every segment (two in flight);
//   C : the segment's conv carry [1, 3, 1, W_r] bf16 TILE (N_C = 3 W_r / 32 tiles; the carry is row 0 of each tile)
//       -> row rd of the stack's conv tile (C_c [1, 3, CHUNK (32 rows), W_r]: the other users' rows kept) --
//       read-modify-write of this core's tiles [first_c, first_c + count_c) per segment (a core owns its tiles,
//       so two segments of one chunk patch the same tile in program order on one core: no cross-core hazard).
// A fresh segment (F_FRESH) still writes: its final state is the state after its rows.
// Compile-time (PH_CT0 +): CB_SCR, SEM_WAIT, WAIT_MODE, WAIT_COUNT, NSEG_MAX, N_S, N_C, CHUNK, N_CHUNKS, SCR_DRAM
//   (1: the segments' scratches are DRAM interleaved, 0: L1 interleaved), then the TensorAccessorArgs of segtab.
// Runtime (ph_rt0 +): segtab address, NSEG_MAX x (scratch S, scratch C) addresses, N_CHUNKS x (stack S, stack C)
//   addresses, first_s, count_s, first_c, count_c.
// Scratch CB_SCR: 2 x 4096 + 2 x 2048 + align64(table).
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/phase_args.hpp"
#include "cache_common.hpp"

namespace {
FORCE_INLINE void copy32(uint32_t src, uint32_t dst) {
    volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src);
    volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst);
    for (uint32_t w = 0; w < 8; ++w) {
        d[w] = s[w];
    }
}
}  // namespace

void kernel_main() {
    constexpr uint32_t CB_SCR = get_compile_time_arg_val(PH_CT0 + 0);
    constexpr uint32_t SEM_WAIT = get_compile_time_arg_val(PH_CT0 + 1);
    constexpr uint32_t WAIT_MODE = get_compile_time_arg_val(PH_CT0 + 2);
    constexpr uint32_t WAIT_COUNT = get_compile_time_arg_val(PH_CT0 + 3);
    constexpr uint32_t NSEG_MAX = get_compile_time_arg_val(PH_CT0 + 4);
    constexpr uint32_t N_S = get_compile_time_arg_val(PH_CT0 + 5);
    constexpr uint32_t N_C = get_compile_time_arg_val(PH_CT0 + 6);
    constexpr uint32_t CHUNK = get_compile_time_arg_val(PH_CT0 + 7);
    constexpr uint32_t N_CHUNKS = get_compile_time_arg_val(PH_CT0 + 8);
    constexpr bool SCR_DRAM = get_compile_time_arg_val(PH_CT0 + 9) != 0;
    constexpr auto t_args = TensorAccessorArgs<PH_CT0 + 10>();
    const auto tab = TensorAccessor(t_args, get_arg_val<uint32_t>(ph_rt0 + 0));
    constexpr uint32_t RT_SCR = ph_rt0 + 1, RT_STK = RT_SCR + 2 * NSEG_MAX, RT_RANGE = RT_STK + 2 * N_CHUNKS;
    const uint32_t first_s = get_arg_val<uint32_t>(RT_RANGE + 0);
    const uint32_t count_s = get_arg_val<uint32_t>(RT_RANGE + 1);
    const uint32_t first_c = get_arg_val<uint32_t>(RT_RANGE + 2);
    const uint32_t count_c = get_arg_val<uint32_t>(RT_RANGE + 3);
    constexpr uint32_t SB = pmk::TILE_FP32, CB_ = pmk::TILE_BF16;
    if (count_s == 0 && count_c == 0) {
        return;
    }

    const uint32_t a = get_write_ptr(CB_SCR), b = a + SB, ca = b + SB, cb = ca + CB_;
    const uint32_t tab_l1 = cb + CB_;
    if constexpr (WAIT_MODE != pmk::WAIT_NONE) {
        pmk::wait_min(SEM_WAIT, WAIT_COUNT);  // the scan chains' final states have landed
    }
    const pmk::SegTable st{pmk::load_segtab(tab, tab_l1)};
    const uint32_t n = st.n();
    for (uint32_t k = 0; k < n; ++k) {
        const uint32_t slot = st.seg(k, pmk::S_SLOT);
        if (slot == pmk::SEG_NONE) {
            continue;
        }
        const uint32_t cidx = slot / CHUNK, rd = slot % CHUNK;
        const InterleavedAddrGen<SCR_DRAM> s_scr{
            .bank_base_address = get_arg_val<uint32_t>(RT_SCR + 2 * k), .page_size = SB};
        const InterleavedAddrGen<SCR_DRAM> c_scr{
            .bank_base_address = get_arg_val<uint32_t>(RT_SCR + 2 * k + 1), .page_size = CB_};
        const InterleavedAddrGen<true> s_stk{
            .bank_base_address = get_arg_val<uint32_t>(RT_STK + 2 * cidx), .page_size = SB};
        const InterleavedAddrGen<true> c_stk{
            .bank_base_address = get_arg_val<uint32_t>(RT_STK + 2 * cidx + 1), .page_size = CB_};
        // S: scratch pages -> the slot's pages, two in flight
        for (uint32_t p = first_s; p < first_s + count_s; p += 2) {
            const bool two = p + 1 < first_s + count_s;
            noc_async_read(s_scr.get_noc_addr(p), a, SB);
            if (two) {
                noc_async_read(s_scr.get_noc_addr(p + 1), b, SB);
            }
            noc_async_read_barrier();
            noc_async_write(a, s_stk.get_noc_addr(rd * N_S + p), SB);
            if (two) {
                noc_async_write(b, s_stk.get_noc_addr(rd * N_S + p + 1), SB);
            }
            noc_async_write_barrier();
        }
        // C: row 0 of the scratch tile -> row rd of the stack tile (both 32-B face rows), the other rows kept
        for (uint32_t p = first_c; p < first_c + count_c; ++p) {
            noc_async_read(c_scr.get_noc_addr(p), ca, CB_);
            noc_async_read(c_stk.get_noc_addr(p), cb, CB_);
            noc_async_read_barrier();
            invalidate_l1_cache();
            copy32(ca + pmk::row_off_bf16(0), cb + pmk::row_off_bf16(rd));
            copy32(ca + pmk::row_off_bf16(0) + 512, cb + pmk::row_off_bf16(rd) + 512);
            invalidate_l1_cache();
            noc_async_write(cb, c_stk.get_noc_addr(p), CB_);
            noc_async_write_barrier();
        }
    }
}
