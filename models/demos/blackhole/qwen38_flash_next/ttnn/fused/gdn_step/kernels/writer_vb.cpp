// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Head core of the V-split GDN step (QWEN38_GDN_CONV_SPLIT=2), block s of VB: (1) the o exchange: this core's bf16 o
// tiles (CB_OMINE, VT tiles) go to tile slots s*VT.. of CB_OBF on every block core of the head (self included), data
// first, write barrier, then the SEM_X increment on each; wait for VB increments, reset, publish CB_OBF (4 tiles) to
// the compute (CB_OBF is one CBDescriptor over the head cores: the same L1 address on all of them); (2) the state
// drain, ST tiles (tile i*VT + c -> page (lane*HEADS+head)*16 + i*4 + s*VT + c); (3) the output: the VT gated tiles
// (page head*4 + s*VT + c; whole tile when rows == 1, else only the lane's half rows). Compile-time args: 0 rows, 1
// HEADS, 2 VT, 3 VB; then TensorAccessorArgs of state, out.  Runtime args: 0 state, 1 out addresses, 2 items, then per
// item (lane, head, s) and the VB peers' NoC (x, y) of the head's block cores in block order.

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/noc_traits.h"
#include "../../kernels/zones.h"

namespace {
constexpr uint32_t CB_OBF = 10, CB_OUTS = 25, CB_OUTG = 26, CB_OMINE = 29, SEM_X = 2;
constexpr uint32_t HEADS = get_compile_time_arg_val(1);
constexpr uint32_t VT = get_compile_time_arg_val(2), VB = get_compile_time_arg_val(3);
constexpr uint32_t HT = 4, STATE_TILES = 16, ST = HT * VT;
constexpr uint32_t BF16_TILE = 2048, FP32_TILE = 4096, FACE_BYTES = 512, HALF_ROW_BYTES = 32;
}  // namespace

void kernel_main() {
    constexpr uint32_t ROWS = get_compile_time_arg_val(0);
    constexpr auto state_args = TensorAccessorArgs<4>();
    constexpr auto out_args = TensorAccessorArgs<state_args.next_compile_time_args_offset()>();
    uint32_t arg = 0;
    const uint32_t state_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t out_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t items = get_arg_val<uint32_t>(arg++);
    const auto state = TensorAccessor(state_args, state_addr);
    const auto out = TensorAccessor(out_args, out_addr);
    const uint32_t sem_addr = get_semaphore(SEM_X);
    volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr);

    for (uint32_t item = 0; item < items; ++item) {
        FUSED_ZONE("fz_gv_w_item");
        const uint32_t lane = get_arg_val<uint32_t>(arg++);
        const uint32_t head = get_arg_val<uint32_t>(arg++);
        const uint32_t s = get_arg_val<uint32_t>(arg++);
        uint32_t peer_x[VB], peer_y[VB];
        for (uint32_t p = 0; p < VB; ++p) {
            peer_x[p] = get_arg_val<uint32_t>(arg++);
            peer_y[p] = get_arg_val<uint32_t>(arg++);
        }

        {
            FUSED_ZONE("fz_gv_w_exchange");
            cb_reserve_back(CB_OBF, HT);
            const uint32_t obf = get_write_ptr(CB_OBF);
            cb_wait_front(CB_OMINE, VT);
            const uint32_t src = get_read_ptr(CB_OMINE);
            for (uint32_t p = 0; p < VB; ++p) {
                noc_async_write(src, get_noc_addr(peer_x[p], peer_y[p], obf + s * VT * BF16_TILE), VT * BF16_TILE);
            }
            noc_async_write_barrier();
            for (uint32_t p = 0; p < VB; ++p) {
                noc_semaphore_inc(get_noc_addr(peer_x[p], peer_y[p], sem_addr), 1);
            }
            noc_semaphore_wait_min(sem, VB);
            noc_semaphore_set(sem, 0);
            noc_async_atomic_barrier();
            cb_pop_front(CB_OMINE, VT);
            cb_push_back(CB_OBF, HT);
        }

        cb_wait_front(CB_OUTS, ST);
        {
            const uint32_t l1 = get_read_ptr(CB_OUTS);
            for (uint32_t i = 0; i < HT; ++i) {
                for (uint32_t c = 0; c < VT; ++c) {
                    const uint32_t page = (lane * HEADS + head) * STATE_TILES + i * HT + s * VT + c;
                    noc_async_write_page(page, state, l1 + (i * VT + c) * FP32_TILE);
                }
            }
        }
        noc_async_write_barrier();
        cb_pop_front(CB_OUTS, ST);

        cb_wait_front(CB_OUTG, VT);
        {
            const uint32_t l1 = get_read_ptr(CB_OUTG);
            for (uint32_t c = 0; c < VT; ++c) {
                const uint32_t page = head * HT + s * VT + c;
                const uint32_t src = l1 + c * BF16_TILE;
                if constexpr (ROWS == 1) {
                    noc_async_write_page(page, out, src);
                } else {
                    const uint32_t row = ((lane >> 4) * 2) * FACE_BYTES + (lane & 15) * HALF_ROW_BYTES;
                    noc_async_write(src + row, out.get_noc_addr(page, row), HALF_ROW_BYTES);
                    noc_async_write(src + row + FACE_BYTES, out.get_noc_addr(page, row + FACE_BYTES), HALF_ROW_BYTES);
                }
            }
        }
        noc_async_write_barrier();
        cb_pop_front(CB_OUTG, VT);
    }
}
