// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Slot streaming of the ND-sharded routed-expert weights (fused_decode/nd.py, kernels/nd_reader.cpp), shared by both
// data-movement RISCs: with `streams` = 2 the reader (NOC0) streams the even K blocks into cb_w and the writer (NOC1)
// the odd K blocks into cb_w2, and the compute alternates between them, so both NOCs carry weight traffic.
//
// Weight layout: [1, E, kt_all, nt] tiles ND-sharded in [1, 1, kt_all, g] shards (one per expert and g-column group),
// so the K rows of one (expert, group) are one contiguous run of one DRAM bank; the TensorAccessor resolves the bank.
// sets = 2 (gate|up): set 0 reads the gate tensor (s_w), set 1 the up tensor (s_u) at the same column; the bias tiles
// come from the packed [1, E, 32, 2 nt] bias tensor (up columns at col + up_off).

#pragma once

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

// ND_BIAS_PACK > 1 (fused_decode/nd.py): bias tiles hold ND_BIAS_PACK columns each (column c in tile c / ND_BIAS_PACK)
#ifndef ND_BIAS_PACK
#define ND_BIAS_PACK 1
#endif

namespace fused_nd {

template <
    uint32_t tile_bytes,
    uint32_t kt,
    uint32_t kt_all,
    uint32_t nt,
    uint32_t bias_nt,
    uint32_t kb,
    uint32_t g,
    uint32_t num_sel,
    uint32_t sets,
    uint32_t up_off,
    uint32_t depth,
    uint32_t pkt_tiles,
    uint32_t streams,
    typename WAcc,
    typename UAcc,
    typename BAcc>
FORCE_INLINE void nd_stream_slots(
    uint32_t cb_w,
    const WAcc& s_w,
    const UAcc& s_u,
    const BAcc& s_b,
    const uint32_t* experts,
    uint32_t n_groups,
    uint32_t col_arg0,
    uint32_t parity,
    uint32_t vc) {
    static_assert(kt % kb == 0, "weight-row blocks divide K");
    constexpr uint32_t blocks = kt / kb + 1;  // + the bias block
    constexpr uint32_t slot_tiles = kb * g;
    constexpr uint32_t slot_bytes = slot_tiles * tile_bytes;
    constexpr uint32_t ahead = depth - 2;
    static_assert(depth >= 3 && depth <= 15, "slots in flight use transaction ids 1..depth");
    static_assert(pkt_tiles * tile_bytes <= NOC_MAX_BURST_SIZE, "packet");

    cb_reserve_back(cb_w, slot_tiles);
    const uint32_t cb_base = get_write_ptr(cb_w);
    reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, noc_index);

    uint32_t issued = 0;
    uint32_t pushed = 0;
    auto read_run = [&](uint64_t src, uint32_t dst, uint32_t tiles, uint32_t trid) {
        while (tiles > 0) {
            const uint32_t n = tiles < pkt_tiles ? tiles : pkt_tiles;
            noc_async_read_one_packet_set_state<true>(src, n * tile_bytes, vc);
            noc_async_read_one_packet_with_state_with_trid(static_cast<uint32_t>(src), 0, dst, trid);
            src += n * tile_bytes;
            dst += n * tile_bytes;
            tiles -= n;
        }
    };

    for (uint32_t gi = 0; gi < n_groups; ++gi) {
        const uint32_t col0 = get_arg_val<uint32_t>(col_arg0 + gi);
        for (uint32_t e = 0; e < num_sel; ++e) {
            for (uint32_t blk = 0; blk < blocks; ++blk) {
                if (blk % streams != parity) {
                    continue;
                }
                for (uint32_t s = 0; s < sets; ++s) {
                    cb_reserve_back(cb_w, slot_tiles * (issued - pushed + 1));
                    const uint32_t slot = issued % depth;
                    const uint32_t trid = slot + 1;
                    const uint32_t dst = cb_base + slot * slot_bytes;
                    noc_async_read_set_trid(trid);  // tags the reads below (the _with_trid issue only checks its count)
                    if (blk + 1 < blocks) {
                        const uint32_t page = (experts[e] * kt_all + blk * kb) * nt + col0;
                        read_run(s == 0 ? s_w.get_noc_addr(page) : s_u.get_noc_addr(page), dst, kb * g, trid);
                    } else {
                        // the bias tile of each column (one K row, times the activation's bias multiplier tile)
                        static_assert(ND_BIAS_PACK == 1 || g == 1, "packed bias tiles: one column per group");
                        read_run(
                            s_b.get_noc_addr(experts[e] * bias_nt + col0 / ND_BIAS_PACK + s * up_off), dst, g, trid);
                    }
                    ++issued;
                    if (issued - pushed > ahead) {
                        noc_async_read_barrier_with_trid((pushed % depth) + 1);
                        cb_push_back(cb_w, slot_tiles);
                        ++pushed;
                    }
                }
            }
        }
    }
    while (pushed < issued) {
        noc_async_read_barrier_with_trid((pushed % depth) + 1);
        cb_push_back(cb_w, slot_tiles);
        ++pushed;
    }
    noc_async_read_set_trid(0);
    noc_async_read_barrier();
}

}  // namespace fused_nd
