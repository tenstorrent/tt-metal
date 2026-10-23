// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// Writer for FusedNormDecode: packs column 0 of this core's partial row-sum tile(s) (32 fp32 per input) into one
// contiguous block, sends it to slot `me` of every core's cb_stage (itself included) and bumps their semaphores;
// then writes the normalized tiles of every output to pages c0..c0+per-1.
// compile-time args: [cb_part, cb_send, cb_stage, cb_out0, cb_out1, cb_out2, per, N, NIN, NOUT, sem_id]
//                    + TensorAccessorArgs(out0, out1, out2)
// runtime args: [out0_addr, out1_addr, out2_addr, c0, me, (noc_x, noc_y) x N]
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t out0_addr = get_arg_val<uint32_t>(0);
    const uint32_t out1_addr = get_arg_val<uint32_t>(1);
    const uint32_t out2_addr = get_arg_val<uint32_t>(2);
    const uint32_t c0 = get_arg_val<uint32_t>(3);
    const uint32_t me = get_arg_val<uint32_t>(4);
    constexpr uint32_t cb_part = get_compile_time_arg_val(0);
    constexpr uint32_t cb_send = get_compile_time_arg_val(1);
    constexpr uint32_t cb_stage = get_compile_time_arg_val(2);
    constexpr uint32_t cb_out0 = get_compile_time_arg_val(3);
    constexpr uint32_t cb_out1 = get_compile_time_arg_val(4);
    constexpr uint32_t cb_out2 = get_compile_time_arg_val(5);
    constexpr uint32_t per = get_compile_time_arg_val(6);
    constexpr uint32_t N = get_compile_time_arg_val(7);
    constexpr uint32_t NIN = get_compile_time_arg_val(8);
    constexpr uint32_t NOUT = get_compile_time_arg_val(9);
    constexpr uint32_t sem_id = get_compile_time_arg_val(10);
    constexpr auto out0_args = TensorAccessorArgs<11>();
    constexpr auto out1_args = TensorAccessorArgs<out0_args.next_compile_time_args_offset()>();
    constexpr auto out2_args = TensorAccessorArgs<out1_args.next_compile_time_args_offset()>();
    constexpr uint32_t tile_bytes = 2048;
    constexpr uint32_t f32_tile_words = 1024;
    constexpr uint32_t block_bytes = NIN * 128;

    CircularBuffer cp(cb_part);
    cp.wait_front(NIN);
    volatile tt_l1_ptr uint32_t* part = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_part));
    const uint32_t send_addr = get_write_ptr(cb_send);
    volatile tt_l1_ptr uint32_t* send = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(send_addr);
    for (uint32_t i = 0; i < NIN; ++i) {  // column 0 of row r: face (r / 16) * 2, element (r % 16) * 16
        for (uint32_t r = 0; r < 32; ++r) {
            send[i * 32 + r] = part[i * f32_tile_words + ((r >> 4) * 2) * 256 + (r & 15) * 16];
        }
    }
    const uint32_t stage_base = get_write_ptr(cb_stage);  // same L1 address on every core
    const uint32_t sem_addr = get_semaphore(sem_id);
    for (uint32_t j = 0; j < N; ++j) {
        const uint32_t nx = get_arg_val<uint32_t>(5 + 2 * j), ny = get_arg_val<uint32_t>(6 + 2 * j);
        noc_async_write(send_addr, get_noc_addr(nx, ny, stage_base + me * block_bytes), block_bytes);
    }
    noc_async_write_barrier();
    for (uint32_t j = 0; j < N; ++j) {
        const uint32_t nx = get_arg_val<uint32_t>(5 + 2 * j), ny = get_arg_val<uint32_t>(6 + 2 * j);
        noc_semaphore_inc(get_noc_addr(nx, ny, sem_addr), 1);
    }
    noc_async_atomic_barrier();
    cp.pop_front(NIN);
    // the outputs
    const uint32_t cbs[3] = {cb_out0, cb_out1, cb_out2};
    const uint32_t addrs[3] = {out0_addr, out1_addr, out2_addr};
    for (uint32_t o = 0; o < NOUT; ++o) {
        CircularBuffer co(cbs[o]);
        co.wait_front(per);
        const uint32_t src = get_read_ptr(cbs[o]);
        if (o == 0) {
            const auto s = TensorAccessor(out0_args, addrs[0], tile_bytes);
            for (uint32_t c = 0; c < per; ++c) {
                noc_async_write(src + c * tile_bytes, s.get_noc_addr(c0 + c), tile_bytes);
            }
        } else if (o == 1) {
            const auto s = TensorAccessor(out1_args, addrs[1], tile_bytes);
            for (uint32_t c = 0; c < per; ++c) {
                noc_async_write(src + c * tile_bytes, s.get_noc_addr(c0 + c), tile_bytes);
            }
        } else {
            const auto s = TensorAccessor(out2_args, addrs[2], tile_bytes);
            for (uint32_t c = 0; c < per; ++c) {
                noc_async_write(src + c * tile_bytes, s.get_noc_addr(c0 + c), tile_bytes);
            }
        }
        noc_async_write_barrier();
        co.pop_front(per);
    }
}
