// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// MiMo MoE decode routing for one user (tt/moe_route.py): top-k of the biased sigmoid scores, the unbiased
// scores of the selected experts normalized to sum 1 (eps 1e-20), times the user's token mask.
//
// Inputs (DRAM interleaved, TILE): scores and biased [1, 1, 32, E] fp32, token mask [1, 1, 32, 1] bf16.
// Outputs (L1 interleaved, one row-major page each): expert ids uint16 [users * k] and weights fp32
// [users * k]; this core writes entries [user * k, user * k + k). Optionally also the per-pair row weights
// [1, users * k, 32, 1] bf16 TILE (zero-initialised, persistent): pair p's tile gets its weight at (user, 0).
//
// The biased scores are positive (sigmoid in (0, 1) plus a small correction), so their fp32 bit patterns
// order like the values: selection is integer compares on the exact fp32 inputs, ties keep the lower expert
// id. The normalization runs in IEEE fp32 on the RISC-V.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace {

constexpr uint32_t kFaceRow = 16;          // fp32 values per face row
constexpr uint32_t kFaceBytesFp32 = 1024;  // 16 x 16 fp32
constexpr uint32_t kTileBytesFp32 = 4096;  // 32 x 32 fp32
constexpr uint32_t kTileBytesBf16 = 2048;  // 32 x 32 bf16

// Read row `row` (< 16) of every fp32 tile of a [32, E] TILE tensor into `dst` as E contiguous fp32.
FORCE_INLINE void read_row(uint32_t base, uint32_t num_tiles, uint32_t row, uint32_t dst) {
    const InterleavedAddrGen<true> gen = {.bank_base_address = base, .page_size = kTileBytesFp32};
    for (uint32_t t = 0; t < num_tiles; ++t) {
        uint64_t tile = get_noc_addr(t, gen);
        noc_async_read(tile + row * kFaceRow * 4, dst + t * 128, kFaceRow * 4);                        // face 0
        noc_async_read(tile + kFaceBytesFp32 + row * kFaceRow * 4, dst + t * 128 + 64, kFaceRow * 4);  // face 1
    }
}

}  // namespace

void kernel_main() {
    const uint32_t scores_addr = get_arg_val<uint32_t>(0);
    const uint32_t biased_addr = get_arg_val<uint32_t>(1);
    const uint32_t mask_addr = get_arg_val<uint32_t>(2);
    const uint32_t ids_addr = get_arg_val<uint32_t>(3);
    const uint32_t weights_addr = get_arg_val<uint32_t>(4);
    const uint32_t user = get_arg_val<uint32_t>(5);
    const uint32_t row_weight_addr = get_arg_val<uint32_t>(6);  // 0: not requested

    constexpr uint32_t num_experts = get_compile_time_arg_val(0);
    constexpr uint32_t k = get_compile_time_arg_val(1);
    constexpr uint32_t cb_scratch = get_compile_time_arg_val(2);
    constexpr uint32_t ids_page_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t weights_page_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t row_weight_page_bytes = get_compile_time_arg_val(5);
    constexpr uint32_t num_tiles = num_experts / 32;
    static_assert(num_experts % 32 == 0 && k <= 16, "E a multiple of 32, k <= 16");

    // Scratch: scores [E] fp32 | biased [E] fp32 | mask face 0 (512 B) | ids out (32 B) | weights out (64 B)
    //          | row-weight rows (k x 16 B)
    cb_reserve_back(cb_scratch, 1);
    const uint32_t scratch = get_write_ptr(cb_scratch);
    const uint32_t scores_l1 = scratch;
    const uint32_t biased_l1 = scores_l1 + num_experts * 4;
    const uint32_t mask_l1 = biased_l1 + num_experts * 4;
    const uint32_t ids_l1 = mask_l1 + 512;
    const uint32_t weights_l1 = ids_l1 + 32;
    const uint32_t rows_l1 = weights_l1 + 64;

    read_row(scores_addr, num_tiles, user, scores_l1);
    read_row(biased_addr, num_tiles, user, biased_l1);
    const InterleavedAddrGen<true> mask_gen = {.bank_base_address = mask_addr, .page_size = kTileBytesBf16};
    noc_async_read(get_noc_addr(0, mask_gen), mask_l1, 512);  // face 0: rows 0-15, column 0 of each row at row*32 B
    noc_async_read_barrier();

    // Top-k by exact fp32 order (positive values compare as their bit patterns), descending.
    volatile tt_l1_ptr uint32_t* biased = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(biased_l1);
    uint32_t top_val[k];
    uint32_t top_idx[k];
    uint32_t filled = 0;
    for (uint32_t e = 0; e < num_experts; ++e) {
        const uint32_t v = biased[e];
        if (filled == k && v <= top_val[k - 1]) {
            continue;  // not better than the current k-th (ties keep the lower id)
        }
        uint32_t pos = filled < k ? filled++ : k - 1;
        while (pos > 0 && top_val[pos - 1] < v) {
            top_val[pos] = top_val[pos - 1];
            top_idx[pos] = top_idx[pos - 1];
            --pos;
        }
        top_val[pos] = v;
        top_idx[pos] = e;
    }

    // Unbiased scores of the selected experts, normalized; then the token mask.
    volatile tt_l1_ptr float* scores = reinterpret_cast<volatile tt_l1_ptr float*>(scores_l1);
    float w[k];
    float sum = 0.0f;
    for (uint32_t i = 0; i < k; ++i) {
        w[i] = scores[top_idx[i]];
        sum += w[i];
    }
    const float denom = sum + 1e-20f;
    volatile tt_l1_ptr uint16_t* mask_row = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(mask_l1 + user * 32);
    union {
        uint32_t u;
        float f;
    } mask_bits;
    mask_bits.u = static_cast<uint32_t>(mask_row[0]) << 16;
    const float mask = mask_bits.f;

    volatile tt_l1_ptr uint16_t* ids_out = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(ids_l1);
    volatile tt_l1_ptr float* weights_out = reinterpret_cast<volatile tt_l1_ptr float*>(weights_l1);
    for (uint32_t i = 0; i < k; ++i) {
        ids_out[i] = static_cast<uint16_t>(top_idx[i]);
        weights_out[i] = (w[i] / denom) * mask;
    }

    const InterleavedAddrGen<false> ids_gen = {.bank_base_address = ids_addr, .page_size = ids_page_bytes};
    const InterleavedAddrGen<false> weights_gen = {.bank_base_address = weights_addr, .page_size = weights_page_bytes};
    noc_async_write(ids_l1, get_noc_addr(0, ids_gen, user * k * 2), k * 2);
    noc_async_write(weights_l1, get_noc_addr(0, weights_gen, user * k * 4), k * 4);
    if (row_weight_addr != 0) {
        // Row `user` of pair p's bf16 32x32 tile starts at user * 32 B in face 0 (user < 16); its first 16 B hold
        // columns 0-7: the weight (round to nearest even, as typecast) and seven zeros.
        const InterleavedAddrGen<false> rw_gen = {
            .bank_base_address = row_weight_addr, .page_size = row_weight_page_bytes};
        volatile tt_l1_ptr uint32_t* rows = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rows_l1);
        for (uint32_t i = 0; i < k; ++i) {
            union {
                float f;
                uint32_t u;
            } bits;
            bits.f = weights_out[i];
            const uint32_t bf16 = (bits.u + 0x7FFFu + ((bits.u >> 16) & 1u)) >> 16;
            rows[i * 4 + 0] = bf16;  // little endian: column 0 is the low half
            rows[i * 4 + 1] = 0;
            rows[i * 4 + 2] = 0;
            rows[i * 4 + 3] = 0;
            noc_async_write(rows_l1 + i * 16, get_noc_addr(user * k + i, rw_gen, user * 32), 16);
        }
    }
    noc_async_write_barrier();
}
