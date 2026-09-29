// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// sparse_sdpa_tiled writer (BRISC, NOC 1).  Once: the reduce identity scaler and the col-identity tile for
// compute, and an all-FLOOR band template.  Per tile: wait for the reader's tile control (U, chunk count, t0, the
// seed count), locate the tile's own incomplete blocks in the union, then per chunk build the chunk's additive
// bf16 mask band (Skt tiles, one per key-tile column; row i = query i % TQ; 0 where the query attends the key,
// MASK_FLOOR otherwise) BEFORE gathering the writer's share of the chunk's K/V rows (so the band never waits
// behind a NoC barrier), then drain the tile's untilized output rows to the [1, H, S, v_dim] ROW_MAJOR output.
//
// Band rule per query q and key column kc of the chunk (slot s = slot0 + kc / BT, token j = kc % BT, block
// b = union[s]):  0 if b is q's own incomplete block (b == complete_q) and j < tail_q;  0 if bit q of member[s];
// MASK_FLOOR otherwise (every padding slot s >= U included).  Built as a NoC copy of the FLOOR template followed by
// the member cells only (the set bits of every slot's membership word) and the at most TQ diagonal cells (located
// once per tile by a binary search of the union's two ascending ranges): the cell-by-cell first form wrote every
// cell of every band and cost half the layer (2.75 of 5.45 ms at 28k on an 11x10 die, 2026-09-25).
//
// Named compile-time args: TQ H S CB BT SKT SQT VDHT K_ROW_BYTES OUT_ROW_BYTES RING_DEPTH DEBUG_STAGE and the CB ids
// CB_OUT_RM CB_SCALE CB_COL_IDENTITY CB_MASK_BAND CB_BAND_TEMPLATE CB_POS CB_UNION CB_MEMBER CB_KREQ CB_KACK
// CB_TILECTL.  Positional compile-time args: TensorAccessorArgs of out, kv.  Runtime args: 0 out, 1 kv addresses,
// 2 tile_start, 3 tile_count.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel/dataflow/generate_bcast_scalar.hpp"  // generate_bcast_col_scalar
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>
#include "common.h"
#include "../../kernels/zones.h"

namespace {

constexpr uint32_t one_bf16_packed = 0x3F803F80u;  // bf16(1.0) double-packed; generate_bcast_col_scalar uses >>16

template <uint32_t D, uint32_t BT, typename Accessor>
FORCE_INLINE void union_gather(
    Noc& noc,
    const Accessor& kv,
    uint32_t dst_l1,
    volatile tt_l1_ptr uint32_t* union_ptr,
    uint32_t slot0,
    uint32_t lo,
    uint32_t hi,
    uint32_t row_bytes) {
    const UnicastEndpoint local_l1;
    const uint32_t cnt = hi - lo;
    for (uint32_t i = 0; i < cnt; ++i) {
        const uint32_t p = lo + i;
        const uint32_t trid = (i % D) + 1;
        if (i >= D) {
            experimental::async_read_barrier_with_trid(noc, trid);
        }
        experimental::set_read_trid(noc, trid);
        const uint32_t page = union_ptr[slot0 + p / BT] * BT + p % BT;
        noc.async_read(kv, local_l1, row_bytes, {.page_id = page}, {.addr = dst_l1 + p * row_bytes});
    }
    const uint32_t to_drain = (cnt < D) ? cnt : D;
    for (uint32_t d = 0; d < to_drain; ++d) {
        experimental::async_read_barrier_with_trid(noc, ((cnt - to_drain + d) % D) + 1);
    }
    experimental::set_read_trid(noc, 0);
}

// The union slot of block b, or NO_BLOCK: the seed prefix [0, n_seeds) and the rest [n_seeds, U) are each ascending.
FORCE_INLINE uint32_t union_slot_of(volatile tt_l1_ptr uint32_t* uni, uint32_t b, uint32_t n_seeds, uint32_t U) {
    uint32_t lo = 0, hi = n_seeds;
    for (uint32_t pass = 0; pass < 2; ++pass) {
        while (lo < hi) {
            const uint32_t mid = (lo + hi) >> 1;
            const uint32_t v = uni[mid];
            if (v == b) {
                return mid;
            }
            if (v < b) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        lo = n_seeds;
        hi = U;
    }
    return sst::NO_BLOCK;
}

}  // namespace

void kernel_main() {
    constexpr uint32_t TQ = get_named_compile_time_arg_val("TQ");
    constexpr uint32_t H = get_named_compile_time_arg_val("H");
    constexpr uint32_t S = get_named_compile_time_arg_val("S");
    constexpr uint32_t CB = get_named_compile_time_arg_val("CB");
    constexpr uint32_t BT = get_named_compile_time_arg_val("BT");
    constexpr uint32_t SKT = get_named_compile_time_arg_val("SKT");
    constexpr uint32_t SQT = get_named_compile_time_arg_val("SQT");
    constexpr uint32_t VDHT = get_named_compile_time_arg_val("VDHT");
    constexpr uint32_t K_ROW_BYTES = get_named_compile_time_arg_val("K_ROW_BYTES");
    constexpr uint32_t OUT_ROW_BYTES = get_named_compile_time_arg_val("OUT_ROW_BYTES");
    constexpr uint32_t RING_DEPTH = get_named_compile_time_arg_val("RING_DEPTH");
    constexpr uint32_t DEBUG_STAGE =
        get_named_compile_time_arg_val("DEBUG_STAGE");  // 5+: study, no K/V reads; 8/9: no band
    constexpr uint32_t cb_out_rm = get_named_compile_time_arg_val("CB_OUT_RM");
    constexpr uint32_t cb_scale = get_named_compile_time_arg_val("CB_SCALE");
    constexpr uint32_t cb_col_identity = get_named_compile_time_arg_val("CB_COL_IDENTITY");
    constexpr uint32_t cb_mask_band = get_named_compile_time_arg_val("CB_MASK_BAND");
    constexpr uint32_t cb_band_template = get_named_compile_time_arg_val("CB_BAND_TEMPLATE");
    constexpr uint32_t cb_pos = get_named_compile_time_arg_val("CB_POS");
    constexpr uint32_t cb_union = get_named_compile_time_arg_val("CB_UNION");
    constexpr uint32_t cb_member = get_named_compile_time_arg_val("CB_MEMBER");
    constexpr uint32_t cb_kreq = get_named_compile_time_arg_val("CB_KREQ");
    constexpr uint32_t cb_kack = get_named_compile_time_arg_val("CB_KACK");
    constexpr uint32_t cb_tilectl = get_named_compile_time_arg_val("CB_TILECTL");

    constexpr uint32_t R = TQ * H;
    static_assert(BT == 4, "the band builder writes one block as two bf16 pairs (4 tokens)");
    static_assert(32 % TQ == 0, "a band tile's 32 rows hold whole copies of the TQ query patterns");
    constexpr uint32_t ROW_COPIES = 32 / TQ;  // rows i = q + TQ * r share query q
    static_assert(SKT * 32 == CB * BT, "the band spans the chunk's keys");
    constexpr uint32_t band_bytes = SKT * sst::BF16_TILE_BYTES;

    constexpr auto out_args = TensorAccessorArgs<0>();
    constexpr auto kv_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();

    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t kv_addr = get_arg_val<uint32_t>(1);
    const uint32_t tile_start = get_arg_val<uint32_t>(2);
    const uint32_t tile_count = get_arg_val<uint32_t>(3);

    Noc noc;
    experimental::CB out_cb(cb_out_rm), band_cb(cb_mask_band), template_cb(cb_band_template), pos_cb(cb_pos);
    experimental::CB union_cb(cb_union), member_cb(cb_member), kreq_cb(cb_kreq), kack_cb(cb_kack),
        tilectl_cb(cb_tilectl);
    const auto out = TensorAccessor(out_args, out_addr);
    const auto kv = TensorAccessor(kv_args, kv_addr);
    // The reader's scratch (never pushed: the same fifo base on every RISC).
    volatile tt_l1_ptr uint32_t* pos = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pos_cb.get_write_ptr());
    volatile tt_l1_ptr uint32_t* uni = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(union_cb.get_write_ptr());
    volatile tt_l1_ptr uint32_t* member = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(member_cb.get_write_ptr());

    // Persistent compute inputs: the reduce identity scaler (1.0; the softmax scale is applied in the exp) and the
    // col-identity (column 0 = 1.0) that finalizes the row sum in the normalization.
    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
        cb_scale,
        ckernel::PoolType::MAX,
        ckernel::ReduceDim::REDUCE_ROW,
        /*reduce_factor=*/1>();
    generate_bcast_col_scalar(CircularBuffer(cb_col_identity), one_bf16_packed);

    // The all-FLOOR band template (SKT tiles), filled once; every chunk's band starts as a NoC copy of it.
    template_cb.reserve_back(1);
    const uint32_t template_l1 = template_cb.get_write_ptr();
    {
        volatile tt_l1_ptr uint32_t* t = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(template_l1);
        for (uint32_t w = 0; w < band_bytes / sizeof(uint32_t); ++w) {
            t[w] = sst::MASK_FLOOR_PAIR;
        }
    }
    const UnicastEndpoint local_src;

    uint32_t diag_slot_q[32];  // the union slot of the row's own incomplete block, NO_BLOCK when the row has no tail
    uint32_t tail_q[32];

    for (uint32_t tile = tile_start; tile < tile_start + tile_count; ++tile) {
        FUSED_ZONE("fz_ss_w_tile");
        tilectl_cb.wait_front(1);
        uint32_t U, n_chunks, t0, n_seeds;
        {
            volatile tt_l1_ptr uint32_t* m = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tilectl_cb.get_read_ptr());
            U = m[sst::tilectl::U];
            n_chunks = m[sst::tilectl::N_CHUNKS];
            t0 = m[sst::tilectl::T0];
            n_seeds = m[sst::tilectl::N_SEEDS];
        }
        tilectl_cb.pop_front(1);
        for (uint32_t qi = 0; qi < TQ; ++qi) {
            const uint32_t p = pos[qi];
            tail_q[qi] = (p + 1) & 3;
            // the reader marked the diagonal block of every row with a tail inside the cache; NO_BLOCK otherwise
            diag_slot_q[qi] = tail_q[qi] ? union_slot_of(uni, (p + 1) >> 2, n_seeds, U) : sst::NO_BLOCK;
        }

        for (uint32_t c = 0; c < n_chunks; ++c) {
            const uint32_t slot0 = c * CB;
            const uint32_t slots = (U - slot0) < CB ? (U - slot0) : CB;
            // The band first (its inputs are complete for the whole tile), then the gather share.
            band_cb.reserve_back(SKT);
            if constexpr (DEBUG_STAGE != 8 && DEBUG_STAGE != 9) {
                FUSED_ZONE("fz_ss_w_band");
                const uint32_t base = band_cb.get_write_ptr();
                // 1. every cell FLOOR: the template copied over the band (L1 -> L1 on this NoC)
                noc.async_read(
                    local_src, band_cb, band_bytes, experimental::local_addr(template_l1, noc.get_noc_id()), {});
                noc.async_read_barrier();
                // 2. the member cells: two zero words per query row copy at the block's four columns
                for (uint32_t s = 0; s < slots; ++s) {
                    uint32_t m = member[slot0 + s];
                    if (m == 0) {
                        continue;
                    }
                    const uint32_t kc0 = s * BT;
                    volatile tt_l1_ptr uint32_t* tile_words =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + (kc0 >> 5) * sst::BF16_TILE_BYTES);
                    const uint32_t col = kc0 & 31;
                    while (m) {
                        const uint32_t qi = sst::ctz32(m);
                        m &= m - 1;
                        for (uint32_t r = 0; r < ROW_COPIES; ++r) {
                            const uint32_t word = sst::tile_elem(qi + TQ * r, col) >> 1;
                            tile_words[word] = 0;
                            tile_words[word + 1] = 0;
                        }
                    }
                }
                // 3. the diagonal cells in this chunk: the row's own incomplete block, its first tail_q tokens visible
                //    (a member bit never sits on a row's own diagonal block: the reader marks members below complete)
                for (uint32_t qi = 0; qi < TQ; ++qi) {
                    const uint32_t slot = diag_slot_q[qi];
                    if (slot < slot0 || slot >= slot0 + slots) {
                        continue;
                    }
                    const uint32_t tail = tail_q[qi];  // 1..3
                    const uint32_t w0 = (tail > 1 ? 0u : sst::MASK_FLOOR_BF16)
                                        << 16;  // token 0 visible; token 1 by tail
                    const uint32_t w1 = (tail > 2 ? 0u : sst::MASK_FLOOR_BF16) | (sst::MASK_FLOOR_BF16 << 16);
                    const uint32_t kc0 = (slot - slot0) * BT;
                    volatile tt_l1_ptr uint32_t* tile_words =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + (kc0 >> 5) * sst::BF16_TILE_BYTES);
                    const uint32_t col = kc0 & 31;
                    for (uint32_t r = 0; r < ROW_COPIES; ++r) {
                        const uint32_t word = sst::tile_elem(qi + TQ * r, col) >> 1;
                        tile_words[word] = w0;
                        tile_words[word + 1] = w1;
                    }
                }
            }
            band_cb.push_back(SKT);

            kreq_cb.wait_front(1);
            uint32_t req_slot0, split, dst;
            {
                volatile tt_l1_ptr uint32_t* m = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kreq_cb.get_read_ptr());
                req_slot0 = m[sst::kreq::SLOT0];
                split = m[sst::kreq::SPLIT];
                dst = m[sst::kreq::DST_L1];
            }
            kreq_cb.pop_front(1);
            if constexpr (DEBUG_STAGE != 5 && DEBUG_STAGE < 6) {
                FUSED_ZONE("fz_ss_w_gather");
                union_gather<RING_DEPTH, BT>(noc, kv, dst, uni, req_slot0, 0, split, K_ROW_BYTES);
            }
            kack_cb.reserve_back(1);
            kack_cb.push_back(1);
        }

        // The tile's output rows: row h * TQ + q -> page h * S + t0 + q.
        out_cb.wait_front(SQT * VDHT);
        for (uint32_t r = 0; r < R; ++r) {
            const uint32_t h = r / TQ;
            const uint32_t qi = r % TQ;
            noc.async_write(
                out_cb, out, OUT_ROW_BYTES, {.offset_bytes = r * OUT_ROW_BYTES}, {.page_id = h * S + t0 + qi});
        }
        noc.async_write_barrier();
        out_cb.pop_front(SQT * VDHT);
    }
}
