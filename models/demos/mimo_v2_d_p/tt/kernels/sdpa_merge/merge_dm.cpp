// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// K-split SDPA merge, data movement lane (one per RISC: BRISC on NOC0 lane 0, NCRISC on NOC1 lane 1). Lane l owns
// this core's work items k = l, l + 2, ... (items me + k P; item = head h, 32-row tile r): it reads their S partitions'
// max tiles, running-sum tiles and unnormalized output tiles into its CBs and writes the merged output tiles back, so
// both RISCs drive DRAM reads. Lane 0 also writes the all-ones tile (c_3) once.
// Lane CBs (lane l): max c_(0+4l), sum c_(1+4l), O c_(2+4l), out c_(16+l).
// Tensors (DRAM interleaved bf16 tiles): o [1, S * NH, St, DVt], stats [1, S * NH, 2 * HALFt, 1], out [1, NH, St, DVt].
// CT: 0 S, 1 NH, 2 St, 3 DVt, 4 HALFt, 5 P (cores), 6 lane   RT: 0 o addr, 1 stats addr, 2 out addr, 3 me
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#ifdef MERGE_ZONES
#include "tools/profiler/kernel_profiler.hpp"
#define MZ(name) DeviceZoneScopedN(name)
#else
#define MZ(name)
#endif

void kernel_main() {
    constexpr uint32_t S = get_compile_time_arg_val(0), NH = get_compile_time_arg_val(1);
    constexpr uint32_t St = get_compile_time_arg_val(2), DVt = get_compile_time_arg_val(3);
    constexpr uint32_t HALFt = get_compile_time_arg_val(4), P = get_compile_time_arg_val(5);
    constexpr uint32_t lane = get_compile_time_arg_val(6);
    constexpr uint32_t TB = 2048;  // bf16 tile
    constexpr uint32_t cb_m = 0 + 4 * lane, cb_l = 1 + 4 * lane, cb_o = 2 + 4 * lane, cb_out = 16 + lane;
    const InterleavedAddrGen<true> og = {.bank_base_address = get_arg_val<uint32_t>(0), .page_size = TB};
    const InterleavedAddrGen<true> sg = {.bank_base_address = get_arg_val<uint32_t>(1), .page_size = TB};
    const InterleavedAddrGen<true> wg = {.bank_base_address = get_arg_val<uint32_t>(2), .page_size = TB};
    const uint32_t me = get_arg_val<uint32_t>(3);

    if constexpr (lane == 0) {
        cb_reserve_back(tt::CBIndex::c_3, 1);
        volatile tt_l1_ptr uint32_t* ones =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(tt::CBIndex::c_3));
        for (uint32_t i = 0; i < TB / 4; ++i) {
            ones[i] = 0x3F803F80;  // bf16 1.0, 1.0
        }
        cb_push_back(tt::CBIndex::c_3, 1);
    }

    constexpr uint32_t items = NH * St;
    const uint32_t mine = me < items ? (items - me + P - 1) / P : 0;
    // outputs are written as compute finishes them, never blocking the reads: after each read batch the ready
    // outputs are drained (non-blocking), the rest at the end
    uint32_t next_out = lane;  // this lane's next output (item index k)
    auto write_out = [&](uint32_t item) {
        cb_wait_front(cb_out, DVt);
        uint32_t src = get_read_ptr(cb_out);
        for (uint32_t j = 0; j < DVt; ++j, src += TB) {
#ifndef MERGE_NO_DRAM
            noc_async_write_tile(item * DVt + j, wg, src);
#endif
        }
        noc_async_writes_flushed();
        cb_pop_front(cb_out, DVt);
    };
    for (uint32_t k = lane; k < mine; k += 2) {
        const uint32_t item = me + k * P;
        const uint32_t h = item / St, r = item % St;
        {
            MZ("MG_DM_SPACE");
            cb_reserve_back(cb_m, S);
            cb_reserve_back(cb_l, S);
            cb_reserve_back(cb_o, S * DVt);
        }
        uint32_t mp = get_write_ptr(cb_m), lp = get_write_ptr(cb_l), op = get_write_ptr(cb_o);
        for (uint32_t p = 0; p < S; ++p) {
            const uint32_t vh = p * NH + h;
#ifndef MERGE_NO_DRAM
            noc_async_read_tile(vh * 2 * HALFt + r, sg, mp);
            noc_async_read_tile(vh * 2 * HALFt + HALFt + r, sg, lp);
            for (uint32_t j = 0; j < DVt; ++j) {
                noc_async_read_tile((vh * St + r) * DVt + j, og, op + j * TB);
            }
#endif
            mp += TB;
            lp += TB;
            op += DVt * TB;
        }
        {
            MZ("MG_DM_READ");
            noc_async_read_barrier();
        }
        cb_push_back(cb_m, S);
        cb_push_back(cb_l, S);
        cb_push_back(cb_o, S * DVt);
        while (next_out < k && cb_pages_available_at_front(cb_out, DVt)) {
            write_out(me + next_out * P);
            next_out += 2;
        }
    }
    for (; next_out < mine; next_out += 2) {
        write_out(me + next_out * P);
    }
    noc_async_write_barrier();
}
