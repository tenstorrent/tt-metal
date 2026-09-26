// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// End-to-end flat expert: x relay reader (BRISC). Extracts every expert's tokens from the row-major bf16 dispatch
// buffer (one row per page, interleaved) and stages them for tilizing: for each virtual expert (a sub-block of MT row
// tiles of expert e), each super-block of 32 K tiles (1024 columns) and each row tile m, one chunk of 32 rows x 2 KB
// goes into the row-major CB (32 pages of 2 KB = 32 rows of 1024 bf16). Rows at or past the expert's token count are
// not read (their tiles are padding: their outputs are never written back).
// CT: 0 RM_CB, 1 ROW_BYTES, 2 NUM_EXPERTS, 3 MT, 4 NSB (super-blocks per row), 5 MAX_SUB (sub-blocks per expert
//     at most), 6 BATCH (chunks per read barrier)
// RT: 0 dispatch buffer address, 1 STRIDE, 2 OFF (this relay takes super-blocks OFF, OFF + STRIDE, ... in stream
// order),
//     then per expert: region row offset, token count
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#ifdef SE_DYN
#include "se_dyn.hpp"
#endif
#ifdef SE_ZONES
#include "tools/profiler/kernel_profiler.hpp"
#define ZW(name) DeviceZoneScopedN(name)
#else
#define ZW(name)
#endif

void kernel_main() {
    constexpr uint32_t rm_cb = get_compile_time_arg_val(0);
    constexpr uint32_t row_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t num_e = get_compile_time_arg_val(2);
    constexpr uint32_t mt = get_compile_time_arg_val(3);
    constexpr uint32_t nsb = get_compile_time_arg_val(4);
    constexpr uint32_t batch = get_compile_time_arg_val(6);
    constexpr uint32_t seg = 2048;
    const InterleavedAddrGen<true> xg = {.bank_base_address = get_arg_val<uint32_t>(0), .page_size = row_bytes};
    uint32_t in_batch = 0, l1 = 0;
    auto flush = [&]() {
        {
            ZW("XRD_DRAM");
            noc_async_read_barrier();
        }
        cb_push_back(rm_cb, 32 * in_batch);
        in_batch = 0;
    };
    const uint32_t stride = get_arg_val<uint32_t>(1), soff = get_arg_val<uint32_t>(2);
    uint32_t g = 0;  // super-block index in stream order
#ifdef SE_DYN
    // Dynamic counts: the active experts' regions (RT 3.. are the se_dyn.hpp args, CB 7 this RISC's scratch); the
    // tilizer gets this relay's super-block count in CB 6.
    SeDyn dyn;
    se_dyn_load<num_e>(dyn, 3, get_write_ptr(tt::CBIndex::c_7), mt * 32);
    {
        const uint32_t tot = dyn.num_v * nsb;
        cb_reserve_back(tt::CBIndex::c_6, 1);
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(tt::CBIndex::c_6)) =
            tot > soff ? (tot - soff + stride - 1) / stride : 0;
        cb_push_back(tt::CBIndex::c_6, 1);
    }
    for (uint32_t a = 0; a < dyn.n_act; ++a) {
        const uint32_t off = dyn.off[a], count = dyn.cnt[a];
#else
    for (uint32_t e = 0; e < num_e; ++e) {
        const uint32_t off = get_arg_val<uint32_t>(3 + 2 * e), count = get_arg_val<uint32_t>(4 + 2 * e);
#endif
        const uint32_t subs = (count + mt * 32 - 1) / (mt * 32);
        for (uint32_t s = 0; s < subs; ++s) {
            for (uint32_t j = 0; j < nsb; ++j, ++g) {
                if (g % stride != soff) {
                    continue;
                }
                for (uint32_t m = 0; m < mt; ++m) {
                    if (in_batch == 0) {
                        ZW("XRD_FULL");
                        cb_reserve_back(rm_cb, 32 * batch);
                        l1 = get_write_ptr(rm_cb);
                    }
                    const uint32_t dst = l1 + in_batch * 32 * seg;
                    const uint32_t r0 = s * mt * 32 + m * 32;
                    for (uint32_t r = 0; r < 32 && r0 + r < count; ++r) {
                        noc_async_read(get_noc_addr(off + r0 + r, xg, j * seg), dst + r * seg, seg);
                    }
                    if (++in_batch == batch) {
                        flush();
                    }
                }
            }
        }
    }
    if (in_batch) {
        flush();
    }
}
