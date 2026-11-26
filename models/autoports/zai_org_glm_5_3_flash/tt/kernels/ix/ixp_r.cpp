// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Indexer phase member BRISC (ixp_common.hpp): Q (NQ queries x 4 tiles, query-major [R 32, 128]) and WS (NQ bf16
// tiles: row r of tile r = query r's 32 head weights, the rest zero) into L1 once; then the CB_K receiver of the
// streamer's blocks, interleaved with the running top-512 of rows [NQA, NQ) over CB_SC2 (v3: the two DM RISCs split
// the filter). Compile-time: NQ, KC, CAP, NQA, DEPTH, then the TensorAccessorArgs of q, w and the selection (w:
// ROW_MAJOR [R, 32] bf16, 64-B pages, staged then copied into the tile rows). Runtime: q address, w address, first
// query q0, valid queries nq
// (<= NQ), blocks, streamer NoC x, y, publish (1 on member 0: its CB_K base -> the streamer's ADDR word), selection
// address, CS, pools [, IXP_CS_TENSOR: arg 9 = the start tensor's address, arg 11 = the row base's; their
// TensorAccessorArgs follow the selection's].
#include "api/dataflow/dataflow_api.h"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/ix/ixp_filter.hpp"

using namespace ixp;

void kernel_main() {
    constexpr uint32_t NQ = get_compile_time_arg_val(0);
    constexpr uint32_t KC = get_compile_time_arg_val(1);
    constexpr uint32_t CAP = get_compile_time_arg_val(2);
    constexpr uint32_t NQA = get_compile_time_arg_val(3);
    constexpr uint32_t DEPTH = get_compile_time_arg_val(4);
    constexpr auto q_args = TensorAccessorArgs<5>();
    constexpr auto w_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto s_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    const uint32_t q_addr = get_arg_val<uint32_t>(0), w_addr = get_arg_val<uint32_t>(1);
    const uint32_t q0 = get_arg_val<uint32_t>(2), nq = get_arg_val<uint32_t>(3);
    const uint32_t nblk = get_arg_val<uint32_t>(4);
    const uint32_t sx = get_arg_val<uint32_t>(5), sy = get_arg_val<uint32_t>(6);
    const uint32_t publish = get_arg_val<uint32_t>(7);
    const uint32_t s_addr = get_arg_val<uint32_t>(8), pools = get_arg_val<uint32_t>(10);
    const auto ss = TensorAccessor(s_args, s_addr);
    const auto qq = TensorAccessor(q_args, q_addr);
    const auto ww = TensorAccessor(w_args, w_addr);

    // Q: query r's 4 dim tiles (rows = its 32 heads); padding queries (r >= nq) repeat the last valid query
    cb_reserve_back(CB_Q, 4 * NQ);
    const uint32_t qb = get_write_ptr(CB_Q);
    for (uint32_t r = 0; r < NQ; ++r) {
        const uint32_t q = q0 + (r < nq ? r : nq - 1);
        for (uint32_t d = 0; d < 4; ++d) {
            noc_async_read_page(q * 4 + d, qq, qb + (4 * r + d) * T16);
        }
    }
    // WS: zero the NQ bf16 tiles, stage the nq 64-B weight rows (64-B aligned, behind the tiles' space in the K ring:
    // CB_K is not live yet), then row r of tile r <- w[q0 + r] (two 32-B face-row halves, 8 words each)
    cb_reserve_back(CB_WS, NQ);
    const uint32_t wb = get_write_ptr(CB_WS);
    volatile tt_l1_ptr uint32_t* wz = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(wb);
    for (uint32_t i = 0; i < NQ * T16 / 4; ++i) {
        wz[i] = 0;
    }
    cb_reserve_back(CB_K, 4 * KC);
    const uint32_t stg = (get_write_ptr(CB_K) + 63) & ~63u;  // NQ x 64 B (64-B aligned: the DRAM residue rule)
    for (uint32_t r = 0; r < nq; ++r) {
        noc_async_read(ww.get_noc_addr(q0 + r), stg + r * 64, 64);
    }
    noc_async_read_barrier();
    for (uint32_t r = 0; r < nq; ++r) {
        volatile tt_l1_ptr uint32_t* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stg + r * 64);
        volatile tt_l1_ptr uint32_t* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(wb + r * T16);
        const uint32_t o0 = tile_off(r, 0) / 2, o1 = tile_off(r, 16) / 2;  // uint32 word offsets (2 bf16 per word)
        for (uint32_t i = 0; i < 8; ++i) {
            dst[o0 + i] = src[i];
            dst[o1 + i] = src[8 + i];
        }
    }
    cb_push_back(CB_Q, 4 * NQ);
    cb_push_back(CB_WS, NQ);
    {  // v8: the reduce scaler, a bf16 tile of 1.0
        cb_reserve_back(CB_ONE, 1);
        volatile tt_l1_ptr uint32_t* o = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(CB_ONE));
        for (uint32_t i = 0; i < T16 / 4; ++i) {
            o[i] = 0x3F803F80u;
        }
        cb_push_back(CB_ONE, 1);
    }

    // the CB_K receiver
    volatile tt_l1_ptr uint32_t* valid = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_VALID));
    const uint64_t ready = get_noc_addr(sx, sy, get_semaphore(SEM_READY));
    if (publish) {
        cb_reserve_back(CB_K, 4 * KC);  // (the first slot: its write pointer = the base)
        noc_inline_dw_write(get_noc_addr(sx, sy, get_semaphore(SEM_ADDR)), get_write_ptr(CB_K));
        noc_async_write_barrier();
    }
    // receive block b + DEPTH - 1 ahead of filtering block b: the compute holds K blocks b .. b + DEPTH - 1 while this
    // RISC waits for block b's scores (it pops K block b before it packs them)
    auto receive = [&](uint32_t bb) {
        cb_reserve_back(CB_K, 4 * KC);
        noc_semaphore_inc(ready, 1);
        noc_semaphore_wait_min(valid, bb + 1);
        cb_push_back(CB_K, 4 * KC);
    };
    uint32_t got = 0;
    for (; got < DEPTH && got < nblk; ++got) {
        receive(got);
    }
    cb_reserve_back(CB_CAND2, 1);  // scratch (never pushed)
#ifdef IXP_CS_TENSOR  // arg 9 = the start tensor's address, arg 11 = the row base tensor's (their accessors after
                      // s_args)
    constexpr auto c_args = TensorAccessorArgs<s_args.next_compile_time_args_offset()>();
    constexpr auto b_args = TensorAccessorArgs<c_args.next_compile_time_args_offset()>();
    const uint32_t cs = read_cs(
        TensorAccessor(c_args, get_arg_val<uint32_t>(9)),
        TensorAccessor(b_args, get_arg_val<uint32_t>(11)),
        (get_write_ptr(CB_CAND2) + 63) & ~63u);  // a 64-B aligned slot inside the candidate scratch
#else
    const uint32_t cs = get_arg_val<uint32_t>(9);
#endif
    Filter<CAP> f;
    f.init(get_write_ptr(CB_CAND2), NQA, NQ, cs, q0, nq, pools);
    for (uint32_t b = 0; b < nblk; ++b) {
        cb_wait_front(CB_SC2, 2 * KC);  // KC score tiles + KC row-max tiles (v8)
        f.template block<KC>(get_read_ptr(CB_SC2), b);
        cb_pop_front(CB_SC2, 2 * KC);
        if (got < nblk) {
            receive(got++);
        }
    }
    noc_async_atomic_barrier();
    f.finish(ss, q0, nq);
}
