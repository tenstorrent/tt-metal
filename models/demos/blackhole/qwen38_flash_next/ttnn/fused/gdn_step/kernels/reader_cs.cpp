// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Head core of the split GDN step (QWEN38_GDN_CONV_SPLIT): the (lane, head) item's z, a/b, dt/A, state tiles and row
// mask as reader.cpp, but no conv inputs and no newest-slot write (the conv cores do both).  The conv tiles arrive as
// bf16 unicasts into CB_QKV (q slots 0-3, k 4-7, v 8-11; same-address CB over head and conv cores) counted by two
// semaphores: 0 = q/k (8 increments), 1 = v (4); the reader publishes them to the compute after the state reads and
// the mask.  Compile-time args: 0 rows, 1 HEADS, 2 K_TILE0, 3 V_TILE0, 4 Z_TILE0, 5 A_TILE, 6 B_TILE; then
// TensorAccessorArgs of projected, dtna, norm, state.  Runtime args: 0 projected, 1 dtna, 2 norm, 3 state addresses, 4
// items, then (lane, head) pairs.

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/noc_traits.h"
#include "../../kernels/zones.h"

namespace {
constexpr uint32_t CB_Z = 3, CB_AB = 4, CB_DTNA = 5, CB_W = 6, CB_STATE = 7;
constexpr uint32_t CB_MASK = 8, CB_SCALER = 9, CB_QKV = 11;
constexpr uint32_t SEM_QK = 0, SEM_V = 1, QK_TILES = 8, V_TILES = 4;
constexpr uint32_t HT = 4, QKV_TILES = 12, STATE_TILES = 16;
constexpr uint32_t HEADS = get_compile_time_arg_val(1);
constexpr uint32_t K_TILE0 = get_compile_time_arg_val(2), V_TILE0 = get_compile_time_arg_val(3);
constexpr uint32_t Z_TILE0 = get_compile_time_arg_val(4), A_TILE = get_compile_time_arg_val(5);
constexpr uint32_t B_TILE = get_compile_time_arg_val(6);
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
    const uint32_t dtna_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t w_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t state_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t items = get_arg_val<uint32_t>(arg++);

    constexpr auto p_args = TensorAccessorArgs<7>();
    constexpr auto dtna_args = TensorAccessorArgs<p_args.next_compile_time_args_offset()>();
    constexpr auto w_args = TensorAccessorArgs<dtna_args.next_compile_time_args_offset()>();
    constexpr auto state_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();

    const auto p = TensorAccessor(p_args, p_addr);
    const auto dtna = TensorAccessor(dtna_args, dtna_addr);
    const auto w = TensorAccessor(w_args, w_addr);
    const auto state = TensorAccessor(state_args, state_addr);
    volatile tt_l1_ptr uint32_t* sem_qk = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_QK));
    volatile tt_l1_ptr uint32_t* sem_v = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_V));

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
        cb_reserve_back(CB_QKV, QK_TILES + V_TILES);  // the conv cores write all 12 slots

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

        {
            FUSED_ZONE("fz_gs_r_convwait");
            noc_semaphore_wait_min(sem_qk, QK_TILES);  // after the state reads: the conv tiles land meanwhile
            cb_push_back(CB_QKV, QK_TILES);
            noc_semaphore_wait_min(sem_v, V_TILES);
            cb_push_back(CB_QKV, V_TILES);
            noc_semaphore_set(sem_qk, 0);
            noc_semaphore_set(sem_v, 0);
        }
    }
}
