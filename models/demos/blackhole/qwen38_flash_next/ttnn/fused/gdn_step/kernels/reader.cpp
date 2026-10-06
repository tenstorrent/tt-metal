// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Per (lane, head) item: the projection's q/k/v/z/a/b tiles, the three older conv ring slots' q/k/v tiles, the four
// taps' q/k/v tiles (tile-major: CB_S holds tile t's three slots at 3t.., CB_T its four taps at 4t.., pushed per
// tile), the head's dt_bias / neg_exp_A tiles, the lane's state tiles, and the lane's row mask; writes
// the projection's q/k/v tiles into the newest ring slot.  Tile ids: q 4(h/3)+c, k 16+4(h/3)+c, v 32+4h+c,
// z 80+4h+c, a 128, b 129 (one tile row of the [1,1,rows,4160] projection); state ((lane*12+h)*16 + t).
// Compile-time args: 0 rows; then TensorAccessorArgs of projected, slot0, slot1, slot2, tap0..tap3, dtna, norm,
// state, newest.  Runtime args: 0 projected, 1-3 slots, 4-7 taps, 8 dtna, 9 norm, 10 state, 11 newest addresses,
// 12 items, then (lane, head) pairs.

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/noc_traits.h"
#include "../../kernels/zones.h"

namespace {
constexpr uint32_t CB_P = 0, CB_S = 1, CB_T = 2, CB_Z = 3, CB_AB = 4, CB_DTNA = 5, CB_W = 6, CB_STATE = 7;
constexpr uint32_t CB_MASK = 8, CB_SCALER = 9;
constexpr uint32_t HEADS = 12, HT = 4, QKV_TILES = 12, STATE_TILES = 16;
constexpr uint32_t K_TILE0 = 16, V_TILE0 = 32, Z_TILE0 = 80, A_TILE = 128, B_TILE = 129;
constexpr uint32_t BF16_TILE = 2048, FP32_TILE = 4096, FACE_ELEMS = 256;

constexpr uint32_t face_element(uint32_t row, uint32_t col) {
    return ((row >> 4) * 2 + (col >> 4)) * FACE_ELEMS + (row & 15) * 16 + (col & 15);
}

template <typename Acc>
void read_tiles_at(uint32_t l1, const Acc& acc, const uint32_t* ids, uint32_t count, uint32_t tile_bytes) {
    for (uint32_t t = 0; t < count; ++t) {
        noc_async_read_page(ids[t], acc, l1 + t * tile_bytes);
    }
}

template <typename Acc>
uint32_t read_tiles(uint32_t cb, const Acc& acc, const uint32_t* ids, uint32_t count, uint32_t tile_bytes) {
    cb_reserve_back(cb, count);
    const uint32_t l1 = get_write_ptr(cb);
    read_tiles_at(l1, acc, ids, count, tile_bytes);
    return l1;
}

void fill_words(uint32_t l1, uint32_t words, uint32_t value) {
    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1);
    for (uint32_t k = 0; k < words; ++k) {
        p[k] = value;
    }
}
}  // namespace

void kernel_main() {
    uint32_t arg = 0;
    const uint32_t p_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t s0_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t s1_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t s2_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t t0_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t t1_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t t2_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t t3_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t dtna_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t w_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t state_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t newest_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t items = get_arg_val<uint32_t>(arg++);

    constexpr auto p_args = TensorAccessorArgs<1>();
    constexpr auto s0_args = TensorAccessorArgs<p_args.next_compile_time_args_offset()>();
    constexpr auto s1_args = TensorAccessorArgs<s0_args.next_compile_time_args_offset()>();
    constexpr auto s2_args = TensorAccessorArgs<s1_args.next_compile_time_args_offset()>();
    constexpr auto t0_args = TensorAccessorArgs<s2_args.next_compile_time_args_offset()>();
    constexpr auto t1_args = TensorAccessorArgs<t0_args.next_compile_time_args_offset()>();
    constexpr auto t2_args = TensorAccessorArgs<t1_args.next_compile_time_args_offset()>();
    constexpr auto t3_args = TensorAccessorArgs<t2_args.next_compile_time_args_offset()>();
    constexpr auto dtna_args = TensorAccessorArgs<t3_args.next_compile_time_args_offset()>();
    constexpr auto w_args = TensorAccessorArgs<dtna_args.next_compile_time_args_offset()>();
    constexpr auto state_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    constexpr auto newest_args = TensorAccessorArgs<state_args.next_compile_time_args_offset()>();

    const auto p = TensorAccessor(p_args, p_addr);
    const auto s0 = TensorAccessor(s0_args, s0_addr);
    const auto s1 = TensorAccessor(s1_args, s1_addr);
    const auto s2 = TensorAccessor(s2_args, s2_addr);
    const auto t0 = TensorAccessor(t0_args, t0_addr);
    const auto t1 = TensorAccessor(t1_args, t1_addr);
    const auto t2 = TensorAccessor(t2_args, t2_addr);
    const auto t3 = TensorAccessor(t3_args, t3_addr);
    const auto dtna = TensorAccessor(dtna_args, dtna_addr);
    const auto w = TensorAccessor(w_args, w_addr);
    const auto state = TensorAccessor(state_args, state_addr);
    const auto newest = TensorAccessor(newest_args, newest_addr);

    {
        FUSED_ZONE("fz_gs_r_setup");
        uint32_t ids[HT] = {0, 1, 2, 3};
        read_tiles(CB_W, w, ids, HT, BF16_TILE);
        noc_async_read_barrier();
        cb_push_back(CB_W, HT);
        // reduce scalers (fp32, whole tiles): 1.0 for sums, 1/128 for means
        cb_reserve_back(CB_SCALER, 2);
        const uint32_t l1 = get_write_ptr(CB_SCALER);
        fill_words(l1, FP32_TILE / 4, 0x3F800000u);
        fill_words(l1 + FP32_TILE, FP32_TILE / 4, 0x3C000000u);
        cb_push_back(CB_SCALER, 2);
    }

    for (uint32_t item = 0; item < items; ++item) {
        FUSED_ZONE("fz_gs_r_item");
        const uint32_t lane = get_arg_val<uint32_t>(arg++);
        const uint32_t head = get_arg_val<uint32_t>(arg++);
        const uint32_t qk_head = head / 3;

        uint32_t ids[QKV_TILES];
        for (uint32_t c = 0; c < HT; ++c) {
            ids[c] = qk_head * HT + c;
            ids[HT + c] = K_TILE0 + qk_head * HT + c;
            ids[2 * HT + c] = V_TILE0 + head * HT + c;
        }
        noc_async_write_barrier();  // the previous item's newest-slot writes have left the CB_P slot
        // The conv inputs go out per output tile t (P[t], the three slots' tile t at CB_S 3t.., the four taps' tile t
        // at CB_T 4t..) with one group of reads in flight ahead of each barrier, so the conv starts after the first
        // group instead of after all 96 tiles (the same tiles, the same per-element operands: bitwise).
        cb_reserve_back(CB_P, QKV_TILES);
        cb_reserve_back(CB_S, 3 * QKV_TILES);
        cb_reserve_back(CB_T, 4 * QKV_TILES);
        const uint32_t p_l1 = get_write_ptr(CB_P);
        const uint32_t s_l1 = get_write_ptr(CB_S);
        const uint32_t t_l1 = get_write_ptr(CB_T);
        auto issue_group = [&](uint32_t t) {
            noc_async_read_page(ids[t], p, p_l1 + t * BF16_TILE);
            noc_async_read_page(ids[t], s0, s_l1 + (3 * t + 0) * BF16_TILE);
            noc_async_read_page(ids[t], s1, s_l1 + (3 * t + 1) * BF16_TILE);
            noc_async_read_page(ids[t], s2, s_l1 + (3 * t + 2) * BF16_TILE);
            noc_async_read_page(ids[t], t0, t_l1 + (4 * t + 0) * BF16_TILE);
            noc_async_read_page(ids[t], t1, t_l1 + (4 * t + 1) * BF16_TILE);
            noc_async_read_page(ids[t], t2, t_l1 + (4 * t + 2) * BF16_TILE);
            noc_async_read_page(ids[t], t3, t_l1 + (4 * t + 3) * BF16_TILE);
        };
        issue_group(0);
        for (uint32_t t = 0; t < QKV_TILES; ++t) {
            if (t + 1 < QKV_TILES) {
                issue_group(t + 1);
            }
            noc_async_read_barrier();  // groups <= t + 1 landed; group t goes to the compute, P[t] to the newest slot
            noc_async_write_page(ids[t], newest, p_l1 + t * BF16_TILE);
            cb_push_back(CB_P, 1);
            cb_push_back(CB_S, 3);
            cb_push_back(CB_T, 4);
        }

        uint32_t z_ids[HT];
        for (uint32_t c = 0; c < HT; ++c) {
            z_ids[c] = Z_TILE0 + head * HT + c;
        }
        read_tiles(CB_Z, p, z_ids, HT, BF16_TILE);
        const uint32_t ab_ids[2] = {A_TILE, B_TILE};
        const uint32_t ab_l1 = read_tiles(CB_AB, p, ab_ids, 2, BF16_TILE);
        const uint32_t dtna_ids[2] = {head, HEADS + head};
        read_tiles(CB_DTNA, dtna, dtna_ids, 2, FP32_TILE);
        uint32_t state_ids[STATE_TILES];
        for (uint32_t t = 0; t < STATE_TILES; ++t) {
            state_ids[t] = (lane * HEADS + head) * STATE_TILES + t;
        }
        read_tiles(CB_STATE, state, state_ids, STATE_TILES, FP32_TILE);
        noc_async_read_barrier();
        {
            // element (lane, head) of the a and b tiles -> element (0, 0): the compute broadcasts [0, 0]
            volatile tt_l1_ptr uint16_t* a = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(ab_l1);
            volatile tt_l1_ptr uint16_t* b = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(ab_l1 + BF16_TILE);
            const uint32_t src = face_element(lane, head);
            a[0] = a[src];
            b[0] = b[src];
        }
        cb_push_back(CB_Z, HT);
        cb_push_back(CB_AB, 2);
        cb_push_back(CB_DTNA, 2);
        cb_push_back(CB_STATE, STATE_TILES);

        cb_reserve_back(CB_MASK, 1);
        {
            const uint32_t l1 = get_write_ptr(CB_MASK);
            fill_words(l1, BF16_TILE / 4, 0);
            volatile tt_l1_ptr uint16_t* elems = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1);
            const uint32_t left = face_element(lane, 0);
            for (uint32_t c = 0; c < 16; ++c) {
                elems[left + c] = 0x3F80;
                elems[left + FACE_ELEMS + c] = 0x3F80;
            }
        }
        cb_push_back(CB_MASK, 1);
    }
    noc_async_write_barrier();
}
