// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Fused MoE expert step, collector (one data-movement core outside the 72 workers).
// Waits until all 36 gate cores wrote their 512 B hidden slot into cb_hid (sem_g == 36), then multicasts the whole
// gathered hidden (36 x 8 tiny tiles) into every worker's cb_hid (same L1 address) and sets their sem_r to 1.
// Runtime args: NOC rectangle x0 y0 x1 y1 of the workers (top-left / bottom-right in NOC0 terms), num_dests.
// X_MCAST (tt/fused/moe_bank.py) first: reads row 0 of x's 72 tiles (as 1x32 tiles, as moe_bank_reader.cpp does) into
//   its cb_x, multicasts them into every core's cb_x and sets sem_x there; runtime arg 5: x address, compile args:
//   x's TensorAccessorArgs.
// GUESS_OUT (tt/fused/moe_bank.py, after the multicast; with PF_RING, also after sem_pf reached n_ring: every ring
//   worker has its down entries, so the prefetcher has read this call's mask for both tensors): writes the mask of this
//   step's n_guess highest-weight routed experts
//   ([n_groups] uint16, 1 = marked) for a later step's prefetch. Runtime args 6..8: ids, routing weights (row 0 of
//   their TILE tensors), mask out; compile args after x's: their TensorAccessorArgs.
// PF_SIGNAL: then raises tensor prefetcher op signal (runtime arg 9: its address), releasing the next step's
//   prefetch request (queue_tensor_prefetcher_wait_for_signal ahead of it), so that request's DRAM reads start
//   after this step's instead of during it.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#ifdef PF_SIGNAL
#include "api/dataflow/tensor_prefetcher_signal.h"
#endif

constexpr uint32_t cb_hid = get_named_compile_time_arg_val("cb_hid");
constexpr uint32_t sem_g_id = get_named_compile_time_arg_val("sem_g");
constexpr uint32_t sem_r_id = get_named_compile_time_arg_val("sem_r");
constexpr uint32_t n_senders = get_named_compile_time_arg_val("n_senders");  // 36
constexpr uint32_t hid_bytes = get_named_compile_time_arg_val("hid_bytes");  // 18432
#ifdef X_MCAST
constexpr uint32_t cb_x = get_named_compile_time_arg_val("cb_x");
constexpr uint32_t x_tiles = get_named_compile_time_arg_val("x_tiles");
constexpr uint32_t sem_x_id = get_named_compile_time_arg_val("sem_x");
#endif

void kernel_main() {
    const uint32_t x0 = get_arg_val<uint32_t>(0);
    const uint32_t y0 = get_arg_val<uint32_t>(1);
    const uint32_t x1 = get_arg_val<uint32_t>(2);
    const uint32_t y1 = get_arg_val<uint32_t>(3);
    const uint32_t num_dests = get_arg_val<uint32_t>(4);
    const bool far = noc_index != 0;  // NOC1 multicasts start at the bottom-right corner
#ifdef X_MCAST
    {
        constexpr auto x_args = TensorAccessorArgs<0>();
        const auto xs = TensorAccessor(x_args, get_arg_val<uint32_t>(5));
        const uint32_t xl = get_write_ptr(cb_x);
#ifdef X_ROWS  // multi-row expert step (moe_bank_mr_*.cpp): rows 0 .. X_ROWS - 1, row r's tile t at r * x_tiles + t
        constexpr uint32_t x_rows = X_ROWS;
#else
        constexpr uint32_t x_rows = 1;
#endif
        for (uint32_t r = 0; r < x_rows; ++r) {
            for (uint32_t t = 0; t < x_tiles; ++t) {  // row r = 16 values of face 0 + 16 of face 1
                noc_async_read(xs.get_noc_addr(t, r * 32), xl + (r * x_tiles + t) * 64, 32);
                noc_async_read(xs.get_noc_addr(t, 512 + r * 32), xl + (r * x_tiles + t) * 64 + 32, 32);
            }
        }
        noc_async_read_barrier();
        const uint64_t x_mc =
            far ? get_noc_multicast_addr(x1, y1, x0, y0, xl) : get_noc_multicast_addr(x0, y0, x1, y1, xl);
        noc_async_write_multicast(xl, x_mc, x_rows * x_tiles * 64, num_dests);
        noc_async_write_barrier();
        const uint32_t sem_x_addr = get_semaphore(sem_x_id);
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_x_addr) = 1;
        const uint64_t sx_mc = far ? get_noc_multicast_addr(x1, y1, x0, y0, sem_x_addr)
                                   : get_noc_multicast_addr(x0, y0, x1, y1, sem_x_addr);
        noc_semaphore_set_multicast(sem_x_addr, sx_mc, num_dests);
        noc_async_write_barrier();
    }
#endif

    const uint32_t hid_base = get_write_ptr(cb_hid);
    volatile tt_l1_ptr uint32_t* sem_g = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_g_id));
    noc_semaphore_wait(sem_g, n_senders);

    const uint64_t data_mc =
        far ? get_noc_multicast_addr(x1, y1, x0, y0, hid_base) : get_noc_multicast_addr(x0, y0, x1, y1, hid_base);
    noc_async_write_multicast(hid_base, data_mc, hid_bytes, num_dests);
    noc_async_write_barrier();

    const uint32_t sem_r_addr = get_semaphore(sem_r_id);
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_r_addr) = 1;
    const uint64_t sem_mc =
        far ? get_noc_multicast_addr(x1, y1, x0, y0, sem_r_addr) : get_noc_multicast_addr(x0, y0, x1, y1, sem_r_addr);
    noc_semaphore_set_multicast(sem_r_addr, sem_mc, num_dests);
    noc_async_write_barrier();

#ifdef GUESS_OUT
    {
        constexpr uint32_t topk = get_named_compile_time_arg_val("topk");
        constexpr uint32_t n_guess = get_named_compile_time_arg_val("n_guess");
        constexpr uint32_t n_groups = get_named_compile_time_arg_val("n_groups");
#ifdef X_MCAST
        constexpr uint32_t g0 = TensorAccessorArgs<0>().next_compile_time_args_offset();
#else
        constexpr uint32_t g0 = 0;
#endif
        constexpr auto ids_args = TensorAccessorArgs<g0>();
        constexpr auto wts_args = TensorAccessorArgs<ids_args.next_compile_time_args_offset()>();
        constexpr auto msk_args = TensorAccessorArgs<wts_args.next_compile_time_args_offset()>();
        const auto ids = TensorAccessor(ids_args, get_arg_val<uint32_t>(6));
        const auto wts = TensorAccessor(wts_args, get_arg_val<uint32_t>(7));
        const auto msk = TensorAccessor(msk_args, get_arg_val<uint32_t>(8));
#ifdef PF_RING
        noc_semaphore_wait(
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(get_named_compile_time_arg_val("sem_pf"))),
            get_named_compile_time_arg_val("n_ring"));
#endif
        const uint32_t buf = get_write_ptr(cb_hid) + hid_bytes;  // cb_hid's spare 512 B past the hidden (moe_bank.py)
        noc_async_read(ids.get_noc_addr(0), buf, 32);
        noc_async_read(wts.get_noc_addr(0), buf + 32, 32);
        noc_async_read_barrier();
        volatile tt_l1_ptr uint16_t* id16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(buf);
        volatile tt_l1_ptr uint16_t* w16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(buf + 32);
        volatile tt_l1_ptr uint16_t* mk = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(buf + 64);
        for (uint32_t g = 0; g < n_groups; ++g) {
            mk[g] = 0;
        }
        uint32_t taken = 0;  // bit e: routed expert e already picked
        for (uint32_t k = 0; k < n_guess; ++k) {
            uint32_t best = topk;
            for (uint32_t e = 0; e < topk; ++e) {  // positive bf16: larger value = larger bit pattern
                if (!(taken & (1u << e)) && (best == topk || w16[e] > w16[best])) {
                    best = e;
                }
            }
            taken |= 1u << best;
            mk[id16[best]] = 1;
        }
        noc_async_write(buf + 64, msk.get_noc_addr(0), n_groups * 2);
        noc_async_write_barrier();
    }
#endif
#ifdef PF_SIGNAL
    experimental::tensor_prefetcher_signal(get_arg_val<uint32_t>(9));
    noc_async_atomic_barrier();
#endif
}
