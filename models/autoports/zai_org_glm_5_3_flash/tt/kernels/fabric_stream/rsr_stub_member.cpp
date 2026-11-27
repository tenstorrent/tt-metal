// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Stage 7a FabricStream RS -> members' POST 21 (FabricStreamRSR members=...), the unit's stand-in for PMK member (k,
// j)'s H6 consumer (the mHC post phase). Launch start: write this member's POST (CB 21) and FS_REDUCED addresses into
// its reducer's POSTADDR / RSEMADDR words, then +1 POSTFREE (the slot is free for block 0). Block i of the group's
// list: wait FS_REDUCED >= i + 1 (a program semaphore: re-initialized at launch), copy POST's 16 fp32 tiles to the y
// rows of the block, column tiles CPC j .. CPC j + CPC - 1, and free the slot (+1 POSTFREE) for block i + 1 (not after
// the last: the next launch's start credit frees it).
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "fs_common.hpp"

#if !defined(MK_PHASED)  // standalone; the composite (tt/pf_stream.composite_source) defines both per phase namespace
constexpr uint32_t PH_CT0 = 0;
uint32_t ph_rt0 = 0;
#endif

namespace rsr_m {
constexpr uint32_t TPB = get_compile_time_arg_val(PH_CT0 + 0);
constexpr uint32_t CPC = get_compile_time_arg_val(PH_CT0 + 1);
constexpr uint32_t CB_POST = get_compile_time_arg_val(PH_CT0 + 2);
constexpr uint32_t SEM_REDUCED = get_compile_time_arg_val(PH_CT0 + 3);
constexpr uint32_t TB32 = 4096;
}  // namespace rsr_m

void kernel_main() {
    using namespace rsr_m;
    constexpr auto y_args = TensorAccessorArgs<PH_CT0 + 4>();
    const uint32_t a0 = ph_rt0;
    const uint32_t red_sync = get_arg_val<uint32_t>(a0 + 0);
    const uint32_t rx = get_arg_val<uint32_t>(a0 + 1), ry = get_arg_val<uint32_t>(a0 + 2);
    const uint32_t y_addr = get_arg_val<uint32_t>(a0 + 3);
    const uint32_t j = get_arg_val<uint32_t>(a0 + 4);
    const uint32_t n = get_arg_val<uint32_t>(a0 + 5);  // + n block row tiles (own y rows)
    const auto yy = TensorAccessor(y_args, y_addr);
    const uint32_t post = get_write_ptr(CB_POST);
    const uint32_t sem = get_semaphore(SEM_REDUCED);
    const uint64_t free_w = get_noc_addr(rx, ry, red_sync + fs::POSTFREE * fs::WORD);
    noc_inline_dw_write(get_noc_addr(rx, ry, red_sync + fs::POSTADDR * fs::WORD), post);
    noc_inline_dw_write(get_noc_addr(rx, ry, red_sync + fs::RSEMADDR * fs::WORD), sem);
    noc_async_write_barrier();
    noc_semaphore_inc(free_w, 1);
    volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem);
    for (uint32_t i = 0; i < n; i++) {
        noc_semaphore_wait_min(s, i + 1);
        const uint32_t row = get_arg_val<uint32_t>(a0 + 6 + i);
        for (uint32_t t = 0; t < CPC; t++) {
            noc_async_write_page(row * TPB + CPC * j + t, yy, post + t * TB32);
        }
        noc_async_write_barrier();  // POST read out before the slot is freed
        if (i + 1 < n) {
            noc_semaphore_inc(free_w, 1);
        }
    }
    noc_async_atomic_barrier();
}
