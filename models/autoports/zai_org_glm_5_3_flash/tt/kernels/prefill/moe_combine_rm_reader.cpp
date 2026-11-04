// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Stage 16 combine "rm" (tt/prefill_tp.py EpMoe._combine_rm, GLM53_PF_MOE_COMBINE=rm) -- reader: the gather of the
// weighted top-k sum y[t] = sum_j w[t, j] * out[dest[j C + t]], fused (no [C k, H] gathered copy, no separate multiply
// / reduce passes). out = the expert outputs untilized to ROW_MAJOR [CAP, H] bf16 (one page per dispatch row).
//
// Unit u = (token chunk u / H_SPLIT of 32 tokens, column part u % H_SPLIT of PART_BYTES). Per token t of the chunk the
// K rows' parts land in CB_IN (K x PT pages of 2 KB: each 1,024 bf16 of one row, the compute's "row tiles") and K
// weight tiles in CB_W (fp32, the weight at datum 0: SCALAR broadcast).
// Compile-time: C, K, ROW_BYTES (= H x 2, the out page), PART_BYTES, PT (= PART_BYTES / 2,048), H_SPLIT.
// Runtime: out, dest ([1, K C] uint32, pair p = j C + t), w ([1, 1, C, K] fp32 TILE: the chunk's 32 x K weights are
// ONE 4-KB tile, (t, j) in face 0 (t < 16) or face 2), first unit, unit count. CB_IDX scratch: K x 32 dest words,
// then the weight tile (both at 64-B multiples: BH DRAM reads land at 64-B aligned L1 offsets).
// RM_SKIP_ZERO_W (GLM53_PF_MOE_RM_SKIP, W8 unit 1): a pair whose weight is 0 (EP: another chip's expert, w x local) is
// not read from DRAM -- its slot is a local copy of the zeroed CB_ZERO part, so 0 x 0 adds the same +0 as the zero row
// it used to read. EP's other-chip pairs all point at dispatch row 0 (moe_plan_mc_dest.cpp), so 7/8 of the reads at
// TP8 hit one 8-KB page in one bank (W6: 2.03 ms vs 0.42 ms for the same combine on XSL), and no unwritten row is read.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t C = get_compile_time_arg_val(0);
    constexpr uint32_t K = get_compile_time_arg_val(1);
    constexpr uint32_t ROW_BYTES = get_compile_time_arg_val(2);
    constexpr uint32_t PART_BYTES = get_compile_time_arg_val(3);
    constexpr uint32_t PT = get_compile_time_arg_val(4);
    constexpr uint32_t H_SPLIT = get_compile_time_arg_val(5);
    constexpr uint32_t CB_IN = 0;
    constexpr uint32_t CB_W = 1;
    constexpr uint32_t CB_IDX = 2;
    constexpr uint32_t TILE_BYTES = 2048;
    constexpr uint32_t W_TILE_BYTES = 4096;  // fp32
    static_assert(K <= 16, "the K weights of a token sit in one face row");
    static_assert(PT * TILE_BYTES == PART_BYTES, "a part = whole 1,024-datum row tiles");
    static_assert((C * 4) % 64 == 0, "dest rows of the K slots start 64-B aligned");

    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t dest_addr = get_arg_val<uint32_t>(1);
    const uint32_t wt_addr = get_arg_val<uint32_t>(2);
    const uint32_t u0 = get_arg_val<uint32_t>(3);
    const uint32_t nu = get_arg_val<uint32_t>(4);
    if (nu == 0) {
        return;
    }

#if defined(RM_SKIP_ZERO_W)
    constexpr uint32_t CB_ZERO = 3;
    const uint32_t zero_l1 = get_write_ptr(CB_ZERO);
    {  // one part of +0, zeroed once (the slot source of every weight-0 pair)
        volatile tt_l1_ptr uint32_t* z = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(zero_l1);
        for (uint32_t i = 0; i < PART_BYTES / 4; ++i) {
            z[i] = 0;
        }
    }
    const uint64_t zero_noc = get_noc_addr(zero_l1);
#endif
    // the weight tiles: datum 0 is written per pair; the rest stays 0 (zeroed once: every page of the CB)
    {
        cb_reserve_back(CB_W, 2 * K);
        volatile tt_l1_ptr uint32_t* z = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(CB_W));
        for (uint32_t i = 0; i < 2 * K * W_TILE_BYTES / 4; ++i) {
            z[i] = 0;
        }
    }

    const uint32_t idx_l1 = get_write_ptr(CB_IDX);
    const uint32_t w_l1 = idx_l1 + K * 32 * 4;
    volatile tt_l1_ptr uint32_t* idx = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(idx_l1);
    volatile tt_l1_ptr uint32_t* wl = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(w_l1);
    const InterleavedAddrGen<true> out_g{.bank_base_address = out_addr, .page_size = ROW_BYTES};
    const InterleavedAddrGen<true> dest_g{.bank_base_address = dest_addr, .page_size = K * C * 4};
    const InterleavedAddrGen<true> wt_g{.bank_base_address = wt_addr, .page_size = W_TILE_BYTES};

    for (uint32_t u = u0; u < u0 + nu; ++u) {
        const uint32_t chunk = u / H_SPLIT;
        const uint32_t part = u - chunk * H_SPLIT;
        const uint32_t t0 = chunk * 32;
        for (uint32_t j = 0; j < K; ++j) {
            noc_async_read(dest_g.get_noc_addr(0, (j * C + t0) * 4), idx_l1 + j * 128, 128);
        }
        noc_async_read(wt_g.get_noc_addr(chunk), w_l1, W_TILE_BYTES);
        noc_async_read_barrier();
        for (uint32_t t = 0; t < 32; ++t) {
            cb_reserve_back(CB_IN, K * PT);
            const uint32_t wp = get_write_ptr(CB_IN);
            const uint32_t wrow = t < 16 ? t * 16 : 512 + (t - 16) * 16;  // face 0 / face 2, row t % 16
            for (uint32_t j = 0; j < K; ++j) {
#if defined(RM_SKIP_ZERO_W)
                if ((wl[wrow + j] & 0x7fffffffu) == 0) {  // weight +-0: the zero part, no DRAM read
                    noc_async_read(zero_noc, wp + j * PART_BYTES, PART_BYTES);
                    continue;
                }
#endif
                noc_async_read(out_g.get_noc_addr(idx[j * 32 + t], part * PART_BYTES), wp + j * PART_BYTES, PART_BYTES);
            }
            cb_reserve_back(CB_W, K);
            const uint32_t ww = get_write_ptr(CB_W);
            for (uint32_t j = 0; j < K; ++j) {
                *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ww + j * W_TILE_BYTES) = wl[wrow + j];
            }
            noc_async_read_barrier();
            cb_push_back(CB_IN, K * PT);
            cb_push_back(CB_W, K);
        }
    }
}
