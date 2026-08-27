// SPDX-License-Identifier: Apache-2.0
// FUSED writer, MULTI M_block (integration step 3). Per block: drain cb_out N-slice
// to DRAM, then increment the RELEASE semaphore on every column peer. Because cb_out
// is produced only after compute consumed BOTH external gather buffers for the block,
// this release proves consumption -> gates peers' next-block external writes.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t cb_out = tt::CBIndex::c_2;

void kernel_main() {
    uint32_t a = 0;
    const uint32_t out_addr = get_arg_val<uint32_t>(a++);
    const uint32_t n_start = get_arg_val<uint32_t>(a++);
    const uint32_t my_slot = get_arg_val<uint32_t>(a++);
    const uint32_t m_base = get_arg_val<uint32_t>(a++);

    constexpr uint32_t N_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t Ns = get_compile_time_arg_val(1);
    constexpr uint32_t M_t = get_compile_time_arg_val(2);
    constexpr uint32_t tb = get_compile_time_arg_val(3);
    constexpr uint32_t P = get_compile_time_arg_val(4);
    constexpr uint32_t sem_rel_id = get_compile_time_arg_val(5);
    constexpr uint32_t MBPC = get_compile_time_arg_val(6);
    constexpr uint32_t obn = M_t * Ns;

    uint32_t px[16], py[16];
    for (uint32_t j = 0; j < P; j++) {
        px[j] = get_arg_val<uint32_t>(a++);
        py[j] = get_arg_val<uint32_t>(a++);
    }

    constexpr auto o_args = TensorAccessorArgs<7>();
    const auto o_acc = TensorAccessor(o_args, out_addr, tb);
    const uint32_t sem_rel = get_semaphore(sem_rel_id);

    for (uint32_t mb = 0; mb < MBPC; mb++) {
        uint32_t m0 = m_base + mb * M_t;
        cb_wait_front(cb_out, obn);
        // EARLY release: cb_out is produced only AFTER compute consumed both external gather
        // buffers, so signal peers NOW (before draining) -> block mb+1's external write no longer
        // waits on this block's output drain. cb_out is double-buffered so compute mb+1 can
        // produce into the other slot while this drain runs -> drain overlaps next-block compute.
        for (uint32_t j = 0; j < P; j++) {
            noc_semaphore_inc(get_noc_addr(px[j], py[j], sem_rel), 1);
        }
        uint32_t rp = get_read_ptr(cb_out);
        for (uint32_t m = 0; m < M_t; m++) {
            for (uint32_t n = 0; n < Ns; n++) {
                noc_async_write_page((m0 + m) * N_tiles + n_start + n, o_acc, rp);
                rp += tb;
            }
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out, obn);
    }
}
