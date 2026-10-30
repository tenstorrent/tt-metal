// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Head core of the GDN step split 4 (QWEN38_GDN_CONV_SPLIT=4), reader: reader_vb3.cpp without the a/b and dtna reads
// (the gate core of the head computes beta and decay) and without the q/k conv tiles (the norm cores of the key head
// l2 norm them).  One batch of reads (scalers, state, mask, w, z) with one barrier, then the pushes: scaler, state,
// mask, w, z; then the waits: SEM_GATE (1 increment of the gate core) -> CB_BETA, CB_DECAY; SEM_V (VT v tiles from the
// conv cores, CB_QKV slots 0 .. VT-1) -> CB_QKV; SEM_NORM (2 increments, the q and the k norm core) -> CB_KROW,
// CB_QROW. Compile-time args: 0 rows, 1 HEADS, 2 K_TILE0, 3 V_TILE0, 4 Z_TILE0, 5 A_TILE, 6 B_TILE, 7 VT; then
// TensorAccessorArgs of projected, norm (w), state, scalers, masks.  Runtime args: 0 projected, 1 norm, 2 state, 3
// scalers, 4 masks addresses, 5 lane, 6 head, 7 s.  Single item only.

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/noc_traits.h"
#include "../../kernels/zones.h"

namespace {
constexpr uint32_t CB_Z = 3, CB_W = 6, CB_STATE = 7;
constexpr uint32_t CB_MASK = 8, CB_SCALER = 9, CB_QKV = 11;
constexpr uint32_t CB_QROW = 15, CB_KROW = 16, CB_BETA = 18, CB_DECAY = 19;
constexpr uint32_t SEM_V = 1, SEM_NORM = 3, SEM_GATE = 4, HT = 4, STATE_TILES = 16;
constexpr uint32_t VT = get_compile_time_arg_val(7);
constexpr uint32_t ST = HT * VT;
constexpr uint32_t HEADS = get_compile_time_arg_val(1);
constexpr uint32_t Z_TILE0 = get_compile_time_arg_val(4);
constexpr uint32_t BF16_TILE = 2048, FP32_TILE = 4096;

template <typename Acc>
void read_tiles(uint32_t cb, const Acc& acc, const uint32_t* ids, uint32_t count, uint32_t tile_bytes) {
    cb_reserve_back(cb, count);
    const uint32_t l1 = get_write_ptr(cb);
    for (uint32_t t = 0; t < count; ++t) {
        noc_async_read_page(ids[t], acc, l1 + t * tile_bytes);
    }
}

}  // namespace

void kernel_main() {
    uint32_t arg = 0;
    const uint32_t p_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t w_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t state_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t scalers_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t masks_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t lane = get_arg_val<uint32_t>(arg++);
    const uint32_t head = get_arg_val<uint32_t>(arg++);
    const uint32_t s = get_arg_val<uint32_t>(arg++);

    constexpr auto p_args = TensorAccessorArgs<8>();
    constexpr auto w_args = TensorAccessorArgs<p_args.next_compile_time_args_offset()>();
    constexpr auto state_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    constexpr auto scalers_args = TensorAccessorArgs<state_args.next_compile_time_args_offset()>();
    constexpr auto masks_args = TensorAccessorArgs<scalers_args.next_compile_time_args_offset()>();

    const auto p = TensorAccessor(p_args, p_addr);
    const auto w = TensorAccessor(w_args, w_addr);
    const auto state = TensorAccessor(state_args, state_addr);
    const auto scalers = TensorAccessor(scalers_args, scalers_addr);
    const auto masks = TensorAccessor(masks_args, masks_addr);
    volatile tt_l1_ptr uint32_t* sem_v = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_V));
    volatile tt_l1_ptr uint32_t* sem_norm = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_NORM));
    volatile tt_l1_ptr uint32_t* sem_gate = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_GATE));

    FUSED_ZONE("fz_gs_r_item");
    cb_reserve_back(CB_QKV, VT);  // the conv cores write the v tiles
    cb_reserve_back(CB_BETA, 1);  // the gate core writes beta and decay
    cb_reserve_back(CB_DECAY, 1);
    cb_reserve_back(CB_QROW, HT);  // the norm cores write the normed q and k
    cb_reserve_back(CB_KROW, HT);

    const uint32_t scaler_ids[2] = {0, 1};
    read_tiles(CB_SCALER, scalers, scaler_ids, 2, FP32_TILE);
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
    cb_push_back(CB_STATE, ST);
    cb_push_back(CB_MASK, 1);
    cb_push_back(CB_W, VT);
    cb_push_back(CB_Z, VT);

    {
        FUSED_ZONE("fz_gs_r_convwait");
        noc_semaphore_wait_min(sem_gate, 1);
        cb_push_back(CB_DECAY, 1);
        cb_push_back(CB_BETA, 1);
        noc_semaphore_wait_min(sem_v, VT);
        cb_push_back(CB_QKV, VT);
        noc_semaphore_wait_min(sem_norm, 2);
        cb_push_back(CB_KROW, HT);
        cb_push_back(CB_QROW, HT);
        noc_semaphore_set(sem_gate, 0);
        noc_semaphore_set(sem_v, 0);
        noc_semaphore_set(sem_norm, 0);
    }
}
