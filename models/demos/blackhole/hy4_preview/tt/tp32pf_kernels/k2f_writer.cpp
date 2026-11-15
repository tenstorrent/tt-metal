// SPDX-License-Identifier: Apache-2.0
// tp32pf K2f writer (NCRISC1), ALL passes S in one launch. Per pass: wait until every owner released the slot buffer of
// the previous pass (free semaphore >= NOWN * S), send the untilized rows of cb_rm (2 KB = 1024 columns of pass S) by
// NoC to the owner core of their token (slot + ((k * 32 + token % 32) * 2048) in the owner's L1), one semaphore
// increment per owner with the number of rows sent; owners (me < NOWN) wait for 256 * (S + 1) rows, push cb_slot to
// compute, write the 32 tilized output tiles to out tile (o * 192 + 24 b + 4 S + n), vc = 4 b + n, then wait until
// compute popped cb_slot (reserve) and increment the free semaphore of every core. CT: 0 cb_rm, 1 cb_ws, 2 cb_slot, 3
// cb_til, 4 NOWN, 5 NPASS, 6 sem id (rows), 7 NC, 8 free sem id, TensorAccessorArgs(plan), (dest), (out). RT: 0 me.
// Common: 0 plan, 1 dest, 2 out, then NC x (x, y) physical coords of the cores (the first NOWN are the owners)
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
    constexpr uint32_t cb_rm = get_compile_time_arg_val(0);
    constexpr uint32_t cb_ws = get_compile_time_arg_val(1);
    constexpr uint32_t cb_slot = get_compile_time_arg_val(2);
    constexpr uint32_t cb_til = get_compile_time_arg_val(3);
    constexpr uint32_t NOWN = get_compile_time_arg_val(4);
    constexpr uint32_t NPASS = get_compile_time_arg_val(5);
    constexpr uint32_t sem_id = get_compile_time_arg_val(6);
    constexpr uint32_t NC = get_compile_time_arg_val(7);
    constexpr uint32_t free_id = get_compile_time_arg_val(8);
    constexpr auto p_args = TensorAccessorArgs<9>();
    constexpr auto d_args = TensorAccessorArgs<p_args.next_compile_time_args_offset()>();
    constexpr auto o_args = TensorAccessorArgs<d_args.next_compile_time_args_offset()>();
    const uint32_t me = get_arg_val<uint32_t>(0);
    const auto pa = TensorAccessor(p_args, get_common_arg_val<uint32_t>(0), 2048);
    const auto da = TensorAccessor(d_args, get_common_arg_val<uint32_t>(1), 2048);
    const auto oa = TensorAccessor(o_args, get_common_arg_val<uint32_t>(2), 2048);
    cb_reserve_back(cb_ws, 4);
    const uint32_t ml = get_write_ptr(cb_ws);
    const uint32_t dl = ml + 2048;
    noc_async_read(pa.get_noc_addr(me), ml, 2048);
    noc_async_read_barrier();
    volatile tt_l1_ptr uint32_t* plan = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ml);
    volatile tt_l1_ptr uint32_t* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dl);
    const uint32_t nu = plan[0];
    cb_reserve_back(cb_slot, 256);
    const uint32_t slot = get_write_ptr(cb_slot);
    const uint32_t sem_l1 = get_semaphore(sem_id);
    const uint32_t free_l1 = get_semaphore(free_id);
    volatile tt_l1_ptr uint32_t* sp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_l1);
    volatile tt_l1_ptr uint32_t* fp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(free_l1);
    uint32_t cnt[64];
    for (uint32_t S = 0; S < NPASS; ++S) {
        if (S > 0) {
            noc_semaphore_wait_min(fp, NOWN * S);  // every owner is done with the slot buffers of the previous pass
        }
        for (uint32_t o = 0; o < NOWN; ++o) {
            cnt[o] = 0;
        }
        for (uint32_t u = 0; u < nu; ++u) {
            const uint32_t mt0 = plan[2 + 3 * u], mtc = plan[3 + 3 * u];
            for (uint32_t pr = 0; pr < mtc / 2; ++pr) {
                noc_async_read(da.get_noc_addr((mt0 + 2 * pr) >> 4, ((mt0 + 2 * pr) & 15) * 128), dl + pr * 256, 256);
            }
            noc_async_read_barrier();
            for (uint32_t pr = 0; pr < mtc / 2; ++pr) {
                volatile tt_l1_ptr uint32_t* dp = dst + pr * 64;
                cb_wait_front(cb_rm, 64);
                const uint32_t rd = get_read_ptr(cb_rm);
                for (uint32_t i = 0; i < 64; ++i) {
                    const uint32_t d = dp[i];
                    if (d != 0xFFFFFFFFu) {
                        const uint32_t o = d >> 16;
                        const uint32_t k = (d >> 8) & 7;
                        const uint32_t tl = d & 31;
                        const uint32_t ox = get_common_arg_val<uint32_t>(3 + 2 * o);
                        const uint32_t oy = get_common_arg_val<uint32_t>(4 + 2 * o);
                        noc_async_write(rd + i * 2048, get_noc_addr(ox, oy, slot + (k * 32 + tl) * 2048), 2048);
                        cnt[o] += 1;
                    }
                }
                noc_async_writes_flushed();
                cb_pop_front(cb_rm, 64);
            }
        }
        noc_async_write_barrier();
        for (uint32_t o = 0; o < NOWN; ++o) {
            if (cnt[o] != 0) {
                noc_semaphore_inc(
                    get_noc_addr(
                        get_common_arg_val<uint32_t>(3 + 2 * o), get_common_arg_val<uint32_t>(4 + 2 * o), sem_l1),
                    cnt[o]);
            }
        }
        noc_async_atomic_barrier();
        if (me < NOWN) {
            noc_semaphore_wait_min(sp, 256 * (S + 1));
            cb_push_back(cb_slot, 256);
            cb_wait_front(cb_til, 32);
            const uint32_t tr = get_read_ptr(cb_til);
            for (uint32_t vc = 0; vc < 32; ++vc) {
                noc_async_write(tr + vc * 2048, oa.get_noc_addr(me * 192 + 24 * (vc >> 2) + 4 * S + (vc & 3)), 2048);
            }
            noc_async_write_barrier();
            cb_pop_front(cb_til, 32);
            cb_reserve_back(cb_slot, 256);  // compute popped the slot buffer: it may be overwritten by the next pass
            for (uint32_t q = 0; q < NC; ++q) {
                noc_semaphore_inc(
                    get_noc_addr(
                        get_common_arg_val<uint32_t>(3 + 2 * q), get_common_arg_val<uint32_t>(4 + 2 * q), free_l1),
                    1);
            }
            noc_async_atomic_barrier();
        }
    }
}
