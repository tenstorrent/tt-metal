// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Head core of the V-split GDN step, early-delivery variant (QWEN38_GDN_CONV_SPLIT=3): reader_vb.cpp with the same CB
// contents but delivered earlier: no word loops (the two fp32 reduce scaler tiles and the lane mask tile are read from
// the constant tensors scalers and masks, whose bytes equal what reader_vb.cpp's loops wrote) and every DRAM read (w,
// z, a/b, dtna, state, scalers, mask) is issued before ONE barrier; then the pushes in the order the compute needs them
// (scaler, a/b, dtna, state, mask, w, z).  Original description of reader_vb.cpp: reader_cs.cpp for one column block s
// of the item's head: w, z and state tiles of the VT value columns s*VT .. s*VT+VT-1 only (state tile (i, j) of the 16:
// page (lane*HEADS+head)*16 + i*4 + j, local index i*VT + c for j = s*VT + c).  The conv tiles arrive by unicast into
// CB_QKV (q 0-3, k 4-7, this block's VT v tiles from slot 8) counted by SEM_QK (8) and SEM_V (VT). Compile-time args: 0
// rows, 1 HEADS, 2 K_TILE0, 3 V_TILE0, 4 Z_TILE0, 5 A_TILE, 6 B_TILE, 7 VT; then TensorAccessorArgs of projected, dtna,
// norm, state.  Runtime args: 0 projected, 1 dtna, 2 norm, 3 state addresses, 4 items, then (lane, head, s), then the
// addresses of the scalers and masks tensors (8, 9).  Single item only (items == 1). Compile-time args after state:
// TensorAccessorArgs of scalers (fp32 tiles 0 = 1.0, 1 = 1/128) and masks (bf16 tile per lane).

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

}  // namespace

void kernel_main() {
    uint32_t arg = 0;
    const uint32_t p_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t dtna_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t w_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t state_addr = get_arg_val<uint32_t>(arg++);
    arg++;  // items: always 1 (one (lane, head, block) per core)
    const uint32_t lane = get_arg_val<uint32_t>(arg++);
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

    FUSED_ZONE("fz_gs_r_item");
    cb_reserve_back(CB_QKV, QK_TILES + VT);  // the conv cores write all 12 slots

    // every read of the step, one batch, one barrier; the order is the order the compute needs the data
    const uint32_t scaler_ids[2] = {0, 1};
    read_tiles(CB_SCALER, scalers, scaler_ids, 2, FP32_TILE);
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
    uint32_t w_ids[VT];
    uint32_t z_ids[VT];
    for (uint32_t c = 0; c < VT; ++c) {
        w_ids[c] = s * VT + c;
        z_ids[c] = Z_TILE0 + head * HT + s * VT + c;
    }
    read_tiles(CB_W, w, w_ids, VT, BF16_TILE);
    read_tiles(CB_Z, p, z_ids, VT, BF16_TILE);
    noc_async_read_barrier();

    cb_push_back(CB_SCALER, 2);
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
    cb_push_back(CB_W, VT);
    cb_push_back(CB_Z, VT);

    {
        FUSED_ZONE("fz_gs_r_convwait");
        noc_semaphore_wait_min(sem_qk, QK_TILES);
        cb_push_back(CB_QKV, QK_TILES);
        noc_semaphore_wait_min(sem_v, VT);
        cb_push_back(CB_QKV, VT);
        noc_semaphore_set(sem_qk, 0);
        noc_semaphore_set(sem_v, 0);
    }
}
