// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Per (lane, head) item: the projection's q/k/v/z/a/b tiles, the three older conv ring slots' q/k/v tiles, the four
// taps' q/k/v tiles (tile-major: cb_s holds tile t's three slots at 3t.., cb_t its four taps at 4t.., pushed per
// tile), the head's dt_bias / neg_exp_A tiles, the lane's state tiles, and the lane's row mask; writes
// the projection's q/k/v tiles into the newest ring slot.  Tile ids: q 4(h/3)+c, k 16+4(h/3)+c, v 32+4h+c,
// z 80+4h+c, a 128, b 129 (one tile row of the [1,1,rows,4160] projection); state ((lane*12+h)*16 + t).
// Compile-time args: 0 rows; then TensorAccessorArgs of projected, slot0, slot1, slot2, tap0..tap3, dtna, norm,
// state, newest.  Runtime args: 0 projected, 1-3 slots, 4-7 taps, 8 dtna, 9 norm, 10 state, 11 newest addresses,
// 12 items, then (lane, head) pairs.

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/noc_traits.h"

namespace {
constexpr uint32_t cb_p = 0, cb_s = 1, cb_t = 2, cb_z = 3, cb_ab = 4, cb_dtna = 5, cb_w = 6, cb_state = 7;
constexpr uint32_t cb_mask = 8, cb_scaler = 9;
constexpr uint32_t value_heads = 12, head_tiles = 4, qkv_tiles = 12, head_state_tiles = 16;
constexpr uint32_t k_tile0 = 16, v_tile0 = 32, z_tile0 = 80, a_tile = 128, b_tile = 129;
constexpr uint32_t bf16_tile_bytes = 2048, fp32_tile_bytes = 4096, face_elements = 256;

constexpr uint32_t face_element(uint32_t row, uint32_t col) {
    return ((row >> 4) * 2 + (col >> 4)) * face_elements + (row & 15) * 16 + (col & 15);
}

template <typename Accessor>
void read_tiles_at(uint32_t l1, const Accessor& tensor, const uint32_t* page_ids, uint32_t count, uint32_t tile_bytes) {
    for (uint32_t t = 0; t < count; ++t) {
        noc_async_read_page(page_ids[t], tensor, l1 + t * tile_bytes);
    }
}

template <typename Accessor>
uint32_t read_tiles(
    uint32_t cb, const Accessor& tensor, const uint32_t* page_ids, uint32_t count, uint32_t tile_bytes) {
    cb_reserve_back(cb, count);
    const uint32_t l1 = get_write_ptr(cb);
    read_tiles_at(l1, tensor, page_ids, count, tile_bytes);
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
    const uint32_t projected_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t slot0_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t slot1_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t slot2_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t tap0_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t tap1_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t tap2_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t tap3_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t dtna_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t norm_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t state_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t newest_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t items = get_arg_val<uint32_t>(arg++);

    constexpr auto projected_args = TensorAccessorArgs<1>();
    constexpr auto slot0_args = TensorAccessorArgs<projected_args.next_compile_time_args_offset()>();
    constexpr auto slot1_args = TensorAccessorArgs<slot0_args.next_compile_time_args_offset()>();
    constexpr auto slot2_args = TensorAccessorArgs<slot1_args.next_compile_time_args_offset()>();
    constexpr auto tap0_args = TensorAccessorArgs<slot2_args.next_compile_time_args_offset()>();
    constexpr auto tap1_args = TensorAccessorArgs<tap0_args.next_compile_time_args_offset()>();
    constexpr auto tap2_args = TensorAccessorArgs<tap1_args.next_compile_time_args_offset()>();
    constexpr auto tap3_args = TensorAccessorArgs<tap2_args.next_compile_time_args_offset()>();
    constexpr auto dtna_args = TensorAccessorArgs<tap3_args.next_compile_time_args_offset()>();
    constexpr auto norm_args = TensorAccessorArgs<dtna_args.next_compile_time_args_offset()>();
    constexpr auto state_args = TensorAccessorArgs<norm_args.next_compile_time_args_offset()>();
    constexpr auto newest_args = TensorAccessorArgs<state_args.next_compile_time_args_offset()>();

    const auto p = TensorAccessor(projected_args, projected_addr);
    const auto slot0 = TensorAccessor(slot0_args, slot0_addr);
    const auto slot1 = TensorAccessor(slot1_args, slot1_addr);
    const auto slot2 = TensorAccessor(slot2_args, slot2_addr);
    const auto tap0 = TensorAccessor(tap0_args, tap0_addr);
    const auto tap1 = TensorAccessor(tap1_args, tap1_addr);
    const auto tap2 = TensorAccessor(tap2_args, tap2_addr);
    const auto tap3 = TensorAccessor(tap3_args, tap3_addr);
    const auto dtna = TensorAccessor(dtna_args, dtna_addr);
    const auto norm = TensorAccessor(norm_args, norm_addr);
    const auto state = TensorAccessor(state_args, state_addr);
    const auto newest = TensorAccessor(newest_args, newest_addr);

    {
        uint32_t page_ids[head_tiles] = {0, 1, 2, 3};
        read_tiles(cb_w, norm, page_ids, head_tiles, bf16_tile_bytes);
        noc_async_read_barrier();
        cb_push_back(cb_w, head_tiles);
        // reduce scalers (fp32, whole tiles): 1.0 for sums, 1/128 for means
        cb_reserve_back(cb_scaler, 2);
        const uint32_t l1 = get_write_ptr(cb_scaler);
        fill_words(l1, fp32_tile_bytes / 4, 0x3F800000u);
        fill_words(l1 + fp32_tile_bytes, fp32_tile_bytes / 4, 0x3C000000u);
        cb_push_back(cb_scaler, 2);
    }

    for (uint32_t item = 0; item < items; ++item) {
        const uint32_t lane = get_arg_val<uint32_t>(arg++);
        const uint32_t head = get_arg_val<uint32_t>(arg++);
        const uint32_t qk_head = head / 3;

        uint32_t page_ids[qkv_tiles];
        for (uint32_t c = 0; c < head_tiles; ++c) {
            page_ids[c] = qk_head * head_tiles + c;
            page_ids[head_tiles + c] = k_tile0 + qk_head * head_tiles + c;
            page_ids[2 * head_tiles + c] = v_tile0 + head * head_tiles + c;
        }
        noc_async_write_barrier();  // the previous item's newest-slot writes have left the cb_p slot
        // The conv inputs go out per output tile t (P[t], the three slots' tile t at cb_s 3t.., the four taps' tile t
        // at cb_t 4t..) with one group of reads in flight ahead of each barrier, so the conv starts after one group.
        cb_reserve_back(cb_p, qkv_tiles);
        cb_reserve_back(cb_s, 3 * qkv_tiles);
        cb_reserve_back(cb_t, 4 * qkv_tiles);
        const uint32_t p_l1 = get_write_ptr(cb_p);
        const uint32_t s_l1 = get_write_ptr(cb_s);
        const uint32_t t_l1 = get_write_ptr(cb_t);
        auto issue_group = [&](uint32_t t) {
            noc_async_read_page(page_ids[t], p, p_l1 + t * bf16_tile_bytes);
            noc_async_read_page(page_ids[t], slot0, s_l1 + (3 * t + 0) * bf16_tile_bytes);
            noc_async_read_page(page_ids[t], slot1, s_l1 + (3 * t + 1) * bf16_tile_bytes);
            noc_async_read_page(page_ids[t], slot2, s_l1 + (3 * t + 2) * bf16_tile_bytes);
            noc_async_read_page(page_ids[t], tap0, t_l1 + (4 * t + 0) * bf16_tile_bytes);
            noc_async_read_page(page_ids[t], tap1, t_l1 + (4 * t + 1) * bf16_tile_bytes);
            noc_async_read_page(page_ids[t], tap2, t_l1 + (4 * t + 2) * bf16_tile_bytes);
            noc_async_read_page(page_ids[t], tap3, t_l1 + (4 * t + 3) * bf16_tile_bytes);
        };
        issue_group(0);
        for (uint32_t t = 0; t < qkv_tiles; ++t) {
            if (t + 1 < qkv_tiles) {
                issue_group(t + 1);
            }
            noc_async_read_barrier();  // groups <= t + 1 landed; group t goes to the compute, P[t] to the newest slot
            noc_async_write_page(page_ids[t], newest, p_l1 + t * bf16_tile_bytes);
            cb_push_back(cb_p, 1);
            cb_push_back(cb_s, 3);
            cb_push_back(cb_t, 4);
        }

        uint32_t z_ids[head_tiles];
        for (uint32_t c = 0; c < head_tiles; ++c) {
            z_ids[c] = z_tile0 + head * head_tiles + c;
        }
        read_tiles(cb_z, p, z_ids, head_tiles, bf16_tile_bytes);
        const uint32_t ab_ids[2] = {a_tile, b_tile};
        const uint32_t ab_l1 = read_tiles(cb_ab, p, ab_ids, 2, bf16_tile_bytes);
        const uint32_t dtna_ids[2] = {head, value_heads + head};
        read_tiles(cb_dtna, dtna, dtna_ids, 2, fp32_tile_bytes);
        uint32_t state_ids[head_state_tiles];
        for (uint32_t t = 0; t < head_state_tiles; ++t) {
            state_ids[t] = (lane * value_heads + head) * head_state_tiles + t;
        }
        read_tiles(cb_state, state, state_ids, head_state_tiles, fp32_tile_bytes);
        noc_async_read_barrier();
        {
            // element (lane, head) of the a and b tiles -> element (0, 0): the compute broadcasts [0, 0]
            volatile tt_l1_ptr uint16_t* a = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(ab_l1);
            volatile tt_l1_ptr uint16_t* b = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(ab_l1 + bf16_tile_bytes);
            const uint32_t src = face_element(lane, head);
            a[0] = a[src];
            b[0] = b[src];
        }
        cb_push_back(cb_z, head_tiles);
        cb_push_back(cb_ab, 2);
        cb_push_back(cb_dtna, 2);
        cb_push_back(cb_state, head_state_tiles);

        cb_reserve_back(cb_mask, 1);
        {
            const uint32_t l1 = get_write_ptr(cb_mask);
            fill_words(l1, bf16_tile_bytes / 4, 0);
            volatile tt_l1_ptr uint16_t* elems = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1);
            const uint32_t left = face_element(lane, 0);
            for (uint32_t c = 0; c < 16; ++c) {
                elems[left + c] = 0x3F80;
                elems[left + face_elements + c] = 0x3F80;
            }
        }
        cb_push_back(cb_mask, 1);
    }
    noc_async_write_barrier();
}
