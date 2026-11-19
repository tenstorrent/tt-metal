// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Stage 7a FabricStream RS, REDUCER core BRISC: the fp32 y tiles of its blocks (CB_Y, from rs_reduce.cpp) ->
//   default: y (DRAM fp32 rows, the unit's form);
//   FS_POST (PMK H6, prefill_megakernel.md §4b; the family's post_layout): reducer k serves member group k. Block i of
//     its list: wait POSTFREE >= ((g - 1) n + i + 1) x MPG (every member of the group freed its POST slot), then batch
//     j (= column tiles 16 j .. 16 j + 15 of the block's [32, 4096] y, NB = 16 fp32 tiles = 64 KB) -> member (k, j)'s
//     POST 21 base (address in POSTADDR) + NoC atomic +1 of its FS_REDUCED (address in RSEMADDR).
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "fs_common.hpp"

#if !defined(MK_PHASED)  // standalone; a composite (tt/pf_stream.composite_source) defines both per phase namespace
constexpr uint32_t PH_CT0 = 0;
uint32_t ph_rt0 = 0;
#endif

constexpr uint32_t TPB = get_compile_time_arg_val(PH_CT0 + 0);
constexpr uint32_t NB = get_compile_time_arg_val(PH_CT0 + 1);
constexpr uint32_t CB_Y = get_compile_time_arg_val(PH_CT0 + 2);
constexpr uint32_t TB32 = 4096;
#if defined(FS_Y_BF16)  // GLM53_PMK_RS8_S2_BF16: TP8 stage 1's r1 in bf16 tiles (the DRAM path; FS_POST stays fp32)
constexpr uint32_t TBY = 2048;
#else
constexpr uint32_t TBY = TB32;
#endif

void kernel_main() {
#ifdef FS_POST
    const uint32_t sync = get_arg_val<uint32_t>(ph_rt0 + 0);
    const uint32_t n = get_arg_val<uint32_t>(ph_rt0 + 1);
    const uint32_t mpg = get_arg_val<uint32_t>(ph_rt0 + 2);  // members per group (= TPB / NB column slices)
    constexpr uint32_t MXY = 3;                              // mpg x member (k, j) NoC (x, y)
    volatile tt_l1_ptr uint32_t* gen = fs::word(sync, fs::GEN_B);
    const uint32_t g = *gen + 1;
    for (uint32_t i = 0; i < n; i++) {
        fs::wait_ge(fs::word(sync, fs::POSTFREE), ((g - 1) * n + i + 1) * mpg);
        // THE RING-ADDRESS RULE (fs_common.hpp): the members' FS_REDUCED = this reducer's own; RSEMADDR is the
        // freshness token of POSTADDR (rsr_stub_member publishes POSTADDR, then RSEMADDR, then a write barrier)
        const uint32_t sem = get_semaphore(fs::SEM_REDUCED_ID);
        fs::wait_eq(fs::word(sync, fs::RSEMADDR), sem);
        const uint32_t post = *fs::word(sync, fs::POSTADDR);
        for (uint32_t j = 0; j < TPB / NB; j++) {
            const uint32_t mx = get_arg_val<uint32_t>(ph_rt0 + MXY + 2 * j),
                           my = get_arg_val<uint32_t>(ph_rt0 + MXY + 2 * j + 1);
            cb_wait_front(CB_Y, NB);
            noc_async_write(get_read_ptr(CB_Y), get_noc_addr(mx, my, post), NB * TB32);
            noc_async_write_barrier();  // the slice landed before its FS_REDUCED +1
            noc_semaphore_inc(get_noc_addr(mx, my, sem), 1);
            cb_pop_front(CB_Y, NB);
        }
    }
    noc_async_atomic_barrier();
    *gen = g;
#else
    constexpr auto y_args = TensorAccessorArgs<PH_CT0 + 3>();
    const uint32_t y_addr = get_arg_val<uint32_t>(ph_rt0 + 0);
    const uint32_t n = get_arg_val<uint32_t>(ph_rt0 + 1);
    const auto yy = TensorAccessor(y_args, y_addr);
    for (uint32_t i = 0; i < n; i++) {
        const uint32_t row = get_arg_val<uint32_t>(ph_rt0 + 2 + i);
        for (uint32_t b = 0; b < TPB / NB; b++) {
            cb_wait_front(CB_Y, NB);
            const uint32_t buf = get_read_ptr(CB_Y);
            for (uint32_t t = 0; t < NB; t++) {
                noc_async_write_page(row * TPB + b * NB + t, yy, buf + t * TBY);
            }
            noc_async_writes_flushed();
            cb_pop_front(CB_Y, NB);
        }
#if defined(FS_RED_SIG)
        // TP8 stage 1 -> stage 2: this block of r1 is written; +1 on its consumer's word (the stage-2 lane's R1RDY + k
        // or the stage-2 reducer's R1RDY), args after the rows: n x (NoC x, NoC y, L1 address)
        noc_async_write_barrier();
        const uint32_t sg = ph_rt0 + 2 + n + 3 * i;
        noc_semaphore_inc(
            get_noc_addr(get_arg_val<uint32_t>(sg), get_arg_val<uint32_t>(sg + 1), get_arg_val<uint32_t>(sg + 2)), 1);
#endif
    }
    noc_async_write_barrier();
#if defined(FS_RED_SIG)
    noc_async_atomic_barrier();
#endif
#endif
}
