// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Head core of the lane-split GDN step (QWEN38_GDN_LANE_SPLIT, rows 8 / 16 / 32): reader_vb3.cpp for COUNT items of one
// (head, column block s): the lanes lane0 .. lane0+COUNT-1, one after the other.  The conv tiles of the (head, s) block
// hold every lane's row, so they arrive ONCE (q 0-3, k 4-7, this block's VT v tiles from slot 8 of CB_QKV, counted by
// SEM_QK (8) and SEM_V (VT)) and stay in the CB for all items (the compute pops them after the last item).  Once per
// core: the two fp32 reduce scaler tiles and the VT norm weight tiles (never popped by the compute).  Per item: a/b,
// dtna, this lane's state tiles (page (lane*HEADS+head)*16 + i*4 + s*VT + c), the lane mask tile, z; every DRAM read of
// an item in one batch, one barrier, pushed in the order the compute needs them (reader_vb3.cpp).  The next item's
// reads wait for CB space only (the state CB holds two items when COUNT > 1). Compile-time args: 0 rows, 1 HEADS, 2
// K_TILE0, 3 V_TILE0, 4 Z_TILE0, 5 A_TILE, 6 B_TILE, 7 VT; then TensorAccessorArgs of projected, dtna, norm, state,
// scalers, masks.  Runtime args: 0 projected, 1 dtna, 2 norm, 3 state addresses, 4 COUNT, 5 lane0, 6 head, 7 s, 8
// scalers, 9 masks addresses.

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/noc_traits.h"
#include "../../kernels/zones.h"

namespace {
constexpr uint32_t CB_Z = 3, CB_AB = 4, CB_DTNA = 5, CB_W = 6, CB_STATE = 7;
constexpr uint32_t CB_MASK = 8, CB_SCALER = 9, CB_QKV = 11;
constexpr uint32_t SEM_QK = 0, SEM_V = 1, QK_TILES = 8;
constexpr uint32_t HT = 4, STATE_TILES = 16;
constexpr uint32_t VT = get_compile_time_arg_val(7);
constexpr uint32_t ST = HT * VT;
constexpr uint32_t HEADS = get_compile_time_arg_val(1);
constexpr uint32_t Z_TILE0 = get_compile_time_arg_val(4), A_TILE = get_compile_time_arg_val(5);
constexpr uint32_t B_TILE = get_compile_time_arg_val(6);
constexpr uint32_t BF16_TILE = 2048, FP32_TILE = 4096, FACE_ELEMS = 256;

constexpr uint32_t face_element(uint32_t row, uint32_t col) {
    return ((row >> 4) * 2 + (col >> 4)) * FACE_ELEMS + (row & 15) * 16 + (col & 15);
}

template <typename Acc>
uint32_t read_tiles(uint32_t cb, const Acc& acc, const uint32_t* ids, uint32_t count, uint32_t tile_bytes) {
    cb_reserve_back(cb, count);
    const uint32_t l1 = get_write_ptr(cb);
    for (uint32_t t = 0; t < count; ++t) {
        noc_async_read_page(ids[t], acc, l1 + t * tile_bytes);
    }
    return l1;
}
}  // namespace

void kernel_main() {
    uint32_t arg = 0;
    const uint32_t p_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t dtna_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t w_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t state_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t count = get_arg_val<uint32_t>(arg++);
    const uint32_t lane0 = get_arg_val<uint32_t>(arg++);
    const uint32_t head = get_arg_val<uint32_t>(arg++);
    const uint32_t s = get_arg_val<uint32_t>(arg++);
    const uint32_t scalers_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t masks_addr = get_arg_val<uint32_t>(arg++);

    constexpr auto p_args = TensorAccessorArgs<8>();
    constexpr auto dtna_args = TensorAccessorArgs<p_args.next_compile_time_args_offset()>();
    constexpr auto w_args = TensorAccessorArgs<dtna_args.next_compile_time_args_offset()>();
    constexpr auto state_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    constexpr auto scalers_args = TensorAccessorArgs<state_args.next_compile_time_args_offset()>();
    constexpr auto masks_args = TensorAccessorArgs<scalers_args.next_compile_time_args_offset()>();

    const auto p = TensorAccessor(p_args, p_addr);
    const auto dtna = TensorAccessor(dtna_args, dtna_addr);
    const auto w = TensorAccessor(w_args, w_addr);
    const auto state = TensorAccessor(state_args, state_addr);
    const auto scalers = TensorAccessor(scalers_args, scalers_addr);
    const auto masks = TensorAccessor(masks_args, masks_addr);
    volatile tt_l1_ptr uint32_t* sem_qk = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_QK));
    volatile tt_l1_ptr uint32_t* sem_v = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_V));

    FUSED_ZONE("fz_gls_r_core");
    cb_reserve_back(CB_QKV, QK_TILES + VT);  // the conv cores write all slots

    for (uint32_t item = 0; item < count; ++item) {
        const uint32_t lane = lane0 + item;
        // every read of the item, one batch, one barrier; the order is the order the compute needs the data
        if (item == 0) {
            const uint32_t scaler_ids[2] = {0, 1};
            read_tiles(CB_SCALER, scalers, scaler_ids, 2, FP32_TILE);
        }
        const uint32_t ab_ids[2] = {A_TILE, B_TILE};
        const uint32_t ab_l1 = read_tiles(CB_AB, p, ab_ids, 2, BF16_TILE);
        const uint32_t dtna_ids[2] = {head, HEADS + head};
        read_tiles(CB_DTNA, dtna, dtna_ids, 2, FP32_TILE);
        uint32_t state_ids[ST];
        for (uint32_t i = 0; i < HT; ++i) {
            for (uint32_t c = 0; c < VT; ++c) {
                state_ids[i * VT + c] = (lane * HEADS + head) * STATE_TILES + i * HT + s * VT + c;
            }
        }
        read_tiles(CB_STATE, state, state_ids, ST, FP32_TILE);
        const uint32_t mask_ids[1] = {lane};
        read_tiles(CB_MASK, masks, mask_ids, 1, BF16_TILE);
        if (item == 0) {
            uint32_t w_ids[VT];
            for (uint32_t c = 0; c < VT; ++c) {
                w_ids[c] = s * VT + c;
            }
            read_tiles(CB_W, w, w_ids, VT, BF16_TILE);
        }
        uint32_t z_ids[VT];
        for (uint32_t c = 0; c < VT; ++c) {
            z_ids[c] = Z_TILE0 + head * HT + s * VT + c;
        }
        read_tiles(CB_Z, p, z_ids, VT, BF16_TILE);
        noc_async_read_barrier();

        if (item == 0) {
            cb_push_back(CB_SCALER, 2);
        }
        {
            // element (lane, head) of the a and b tiles -> element (0, 0): the compute broadcasts [0, 0]
            volatile tt_l1_ptr uint16_t* a = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(ab_l1);
            volatile tt_l1_ptr uint16_t* b = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(ab_l1 + BF16_TILE);
            const uint32_t src = face_element(lane, head);
            a[0] = a[src];
            b[0] = b[src];
        }
        cb_push_back(CB_AB, 2);
        cb_push_back(CB_DTNA, 2);
        cb_push_back(CB_STATE, ST);
        cb_push_back(CB_MASK, 1);
        if (item == 0) {
            cb_push_back(CB_W, VT);
        }
        cb_push_back(CB_Z, VT);

        if (item == 0) {
            FUSED_ZONE("fz_gls_r_convwait");
            noc_semaphore_wait_min(sem_qk, QK_TILES);
            cb_push_back(CB_QKV, QK_TILES);
            noc_semaphore_wait_min(sem_v, VT);
            cb_push_back(CB_QKV, VT);
            noc_semaphore_set(sem_qk, 0);
            noc_semaphore_set(sem_v, 0);
        }
    }
}
