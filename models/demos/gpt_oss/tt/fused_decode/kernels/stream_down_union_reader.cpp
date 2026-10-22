// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Union-of-experts down-projection weight streamer (NCRISC, NOC0) for the multi-row decode MoE (fused_decode/union.py
// ExpertDownUnion). Per DRAM bank the down weights are expert-major, then column, then K tile, so this core's `cols`
// columns of one expert are one contiguous range (cols segments of seg_tiles tiles): it is read as one block per
// expert (page_bytes packets), and compute accumulates every column's output tile across the experts in DST
// (kernels/stream_down_union_compute.cpp).
//
// runtime args: [w_addr, union_addr, bank_id, vc, reader_offset_bytes]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#ifdef MOE_PRED
#include "moe_order.hpp"
#endif

void kernel_main() {
    const uint32_t w_addr = get_arg_val<uint32_t>(0);
    const uint32_t uni_addr = get_arg_val<uint32_t>(1);
    const uint32_t bank_id = get_arg_val<uint32_t>(2);
    const uint32_t vc = get_arg_val<uint32_t>(3);
    const uint32_t reader_offset = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_w = get_compile_time_arg_val(0);
    constexpr uint32_t cb_idx = get_compile_time_arg_val(1);
    constexpr uint32_t seg_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t cols = get_compile_time_arg_val(3);
    constexpr uint32_t expert_stride = get_compile_time_arg_val(4);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(5);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(6);   // NOC packet size, divides seg_bytes
    constexpr uint32_t num_buffers = get_compile_time_arg_val(7);  // weight blocks in the ring (2 or 3)
    constexpr auto uni_args = TensorAccessorArgs<8>();

    constexpr uint32_t seg_bytes = seg_tiles * tile_bytes;
    constexpr uint32_t block_tiles = cols * seg_tiles;
    constexpr uint32_t block_bytes = cols * seg_bytes;
    static_assert(seg_bytes % page_bytes == 0 && page_bytes <= NOC_MAX_BURST_SIZE, "packet size");
#ifdef D_NCOLS_ARG
    // balanced down layout (union.py bank_cols): this core's column count (<= cols); its blocks keep the ring's
    // cols-column stride, only the first ncols segments are read
    const uint32_t ncols = get_arg_val<uint32_t>(D_NCOLS_ARG);
#else
    constexpr uint32_t ncols = cols;
#endif
    const uint32_t read_bytes = ncols * seg_bytes;
    constexpr uint32_t pkt = page_bytes;
    const uint32_t pages = read_bytes / page_bytes;

    cb_reserve_back(cb_idx, 1);
    const uint32_t idx_l1 = get_write_ptr(cb_idx);
    const uint32_t cb_start = get_write_ptr(cb_w);  // the ring's first block (nothing reserved yet)
#ifdef MOE_PRED
    // runtime args 5..7: hub noc x, y, this layer's prediction address; prefetch the predicted experts' blocks into
    // MOE_PRED_CB before the union is known (the boundary and the router are still running)
    uint32_t pred[MOE_PRED];
    moe_order::read_pred<MOE_PRED>(
        get_arg_val<uint32_t>(5), get_arg_val<uint32_t>(6), get_arg_val<uint32_t>(7), idx_l1 + 128, MOE_PRED_SEM, pred);
    cb_reserve_back(MOE_PRED_CB, MOE_PRED * block_tiles);
    const uint32_t pre = get_write_ptr(MOE_PRED_CB);
    for (uint32_t k = 0; k < MOE_PRED; ++k) {
        const uint64_t src = get_noc_addr_from_bank_id<true>(bank_id, w_addr + pred[k] * expert_stride + reader_offset);
        noc_async_read(src, pre + k * block_bytes, read_bytes);
    }
#endif
#ifndef MOE_NO_ROUTER
    // fused MoE program (dflash/moe_fused.py): the router's union is written (program semaphore 1)
    noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(1)), 1);
#endif
    const auto s_uni = TensorAccessor(uni_args, uni_addr, 128);
    noc_async_read(s_uni.get_noc_addr(0), idx_l1, 128);
    noc_async_read_barrier();
    invalidate_l1_cache();
    volatile tt_l1_ptr uint16_t* uni = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(idx_l1);
    const uint32_t num_sel = uni[0];
    uint32_t order[32];
    uint32_t e0 = 0;  // first slot (in `order`) streamed through the ring
#ifdef MOE_PRED
    {
        constexpr uint32_t P = MOE_PRED;
        uint32_t kept_pred[P];
        e0 = moe_order::make_order<P>(pred, uni, order, kept_pred);
        noc_async_read_barrier();                           // the prefetch
        cb_push_back(MOE_PRED_CB, MOE_PRED * block_tiles);  // compute takes the routed ones, skips the others
    }
#else
    for (uint32_t e = 0; e < num_sel; ++e) {
        order[e] = e;
    }
#endif

    const uint64_t w_base = get_noc_addr_from_bank_id<true>(bank_id, w_addr);
    reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, noc_index);
    noc_async_read_one_packet_set_state<true>(w_base, pkt, vc);

    constexpr uint32_t extra_in_flight = 1;
    cb_reserve_back(cb_w, block_tiles * (extra_in_flight + 1));
    uint32_t l1_write = get_write_ptr(cb_w);
    const uint32_t cb_base = cb_start;
    const uint32_t cb_end = cb_base + num_buffers * block_bytes;
    uint32_t free_blocks = num_buffers;
    uint32_t trid = 1;
    uint32_t trid_to_wait = 1;

    for (uint32_t e = e0; e < num_sel; ++e) {
        const uint32_t src = static_cast<uint32_t>(uni[1 + order[e]]) * expert_stride + reader_offset;
        noc_async_read_set_trid(trid);
        for (uint32_t pg = 0; pg < pages; ++pg) {
            noc_async_read_one_packet_with_state_with_trid(w_base, src + pg * pkt, l1_write + pg * pkt, trid);
        }
        l1_write += block_bytes;
        if (free_blocks == num_buffers - extra_in_flight) {
            noc_async_read_barrier_with_trid(trid_to_wait);
            cb_push_back(cb_w, block_tiles);
            trid_to_wait = trid_to_wait == num_buffers ? 1 : trid_to_wait + 1;
            cb_reserve_back(cb_w, block_tiles * (extra_in_flight + 1));
        } else {
            free_blocks -= 1;
        }
        trid = trid == num_buffers ? 1 : trid + 1;
        if (l1_write >= cb_end) {
            l1_write = cb_base;
        }
    }
    const uint32_t pending = num_sel - e0 < extra_in_flight ? num_sel - e0 : extra_in_flight;
    for (uint32_t i = 0; i < pending; ++i) {
        noc_async_read_barrier_with_trid(trid_to_wait);
        cb_push_back(cb_w, block_tiles);
        trid_to_wait = trid_to_wait == num_buffers ? 1 : trid_to_wait + 1;
    }
#ifdef MOE_PRED
    noc_async_atomic_barrier();  // read_pred's notification to the hub (issued early: overlaps the weight stream)
#endif
}
