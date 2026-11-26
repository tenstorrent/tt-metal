// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Stage 7a FabricStream line RS of fp32 partials (tt/fabric_stream.FabricStreamRSL), REDUCER NCRISC. For each of its
// blocks b (owned by this chip): the NC terms in FIXED column-position order c = 0 .. NC - 1 — own r1 rows (c = me) or
// the landed copy in r2 (row = the source's slot x B2 + b) after polling the receiving lane's in-order counter ARR +
// slot x LANES + h (h = b mod LANES) >= (g - 1) PER + b / LANES + 1 over the NoC — streamed as matched NB-tile batches
// into CB_T + c (c = 0 .. 3) for rsl_reduce.cpp, which sums ((t0 + t1) + (t2 + t3)) in fp32: the same association on
// every chip, whichever chip owns the block.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "fs_common.hpp"

#if !defined(MK_PHASED)  // standalone; a composite (tt/pf_stream.composite_source) defines both per phase namespace
constexpr uint32_t PH_CT0 = 0;
uint32_t ph_rt0 = 0;
#endif

constexpr uint32_t TPB = get_compile_time_arg_val(PH_CT0 + 0);
constexpr uint32_t NB = get_compile_time_arg_val(PH_CT0 + 1);
constexpr uint32_t CB_T = get_compile_time_arg_val(PH_CT0 + 2);  // CB_T + c, c = 0 .. NC - 1
constexpr uint32_t NC = get_compile_time_arg_val(PH_CT0 + 3);
#if defined(FS_R1_BF16)  // GLM53_PMK_RS8_S2_BF16 (tt/fabric_stream.FabricStreamRSL r1_dtype): r1 / r2 in bf16 tiles
constexpr uint32_t TB32 = 2048;
#else
constexpr uint32_t TB32 = 4096;
#endif
static_assert(TPB % NB == 0 && NC == 4, "4 positions on the row");

void kernel_main() {
    constexpr auto r1_args = TensorAccessorArgs<PH_CT0 + 4>();
    constexpr auto r2_args = TensorAccessorArgs<r1_args.next_compile_time_args_offset()>();
    size_t a = ph_rt0;
    const uint32_t own_sync = get_arg_val<uint32_t>(a++);   // this reducer's shard (its launch counter, poll word)
    const uint32_t lane_sync = get_arg_val<uint32_t>(a++);  // the lanes' shard address (the same on every lane)
    const uint32_t per = get_arg_val<uint32_t>(a++);        // blocks per (source, lane) per launch
    const uint32_t me = get_arg_val<uint32_t>(a++);
    const uint32_t r1_addr = get_arg_val<uint32_t>(a++);
    const uint32_t r2_addr = get_arg_val<uint32_t>(a++);
    const uint32_t b2 = get_arg_val<uint32_t>(a++);     // blocks per owner
    const uint32_t ready = get_arg_val<uint32_t>(a++);  // 1: own r1 block i written (R1RDY + i >= g, stage 1)
    const uint32_t n = get_arg_val<uint32_t>(a++);
    const uint32_t blk = a;  // n x (b, receiving lane NoC x, y)
    const auto r1 = TensorAccessor(r1_args, r1_addr);
    const auto r2 = TensorAccessor(r2_args, r2_addr);
    volatile tt_l1_ptr uint32_t* gen = fs::word(own_sync, fs::GEN_NC);
    const uint32_t g = *gen + 1;
    volatile tt_l1_ptr uint32_t* probe = fs::word(own_sync, fs::GEN_C);
    for (uint32_t i = 0; i < n; i++) {
        const uint32_t b = get_arg_val<uint32_t>(blk + 3 * i);
        const uint32_t lx = get_arg_val<uint32_t>(blk + 3 * i + 1), ly = get_arg_val<uint32_t>(blk + 3 * i + 2);
        const uint32_t h = b % 2;
        if (ready) {
            fs::wait_ge(fs::word(own_sync, fs::R1RDY + i), g);
        }
        uint32_t rows[NC];
        bool landed[NC];
        for (uint32_t c = 0; c < NC; c++) {
            landed[c] = c != me;
            const uint32_t slot = c < me ? c : c - 1;
            rows[c] = c == me ? me * b2 + b : slot * b2 + b;
            if (landed[c]) {
                const uint64_t cnt = get_noc_addr(lx, ly, lane_sync + (fs::ARR + slot * 2 + h) * fs::WORD);
                const uint32_t need = (g - 1) * per + b / 2 + 1;
                do {
                    noc_async_read(cnt, reinterpret_cast<uint32_t>(probe), fs::WORD);
                    noc_async_read_barrier();
                    invalidate_l1_cache();
                } while (static_cast<int32_t>(*probe - need) < 0);
            }
        }
        for (uint32_t bb = 0; bb < TPB / NB; bb++) {
            for (uint32_t c = 0; c < NC; c++) {
                cb_reserve_back(CB_T + c, NB);
                const uint32_t w = get_write_ptr(CB_T + c);
                for (uint32_t t = 0; t < NB; t++) {
                    const uint32_t pg = rows[c] * TPB + bb * NB + t;
                    if (landed[c]) {
                        noc_async_read_page(pg, r2, w + t * TB32);
                    } else {
                        noc_async_read_page(pg, r1, w + t * TB32);
                    }
                }
            }
            noc_async_read_barrier();
            for (uint32_t c = 0; c < NC; c++) {
                cb_push_back(CB_T + c, NB);
            }
        }
    }
    *gen = g;
}
