// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Message-gate prep + down_proj (E10c), writer of split s. Prep cores first run the gate-prep writer (args 0..2:
// x row into DRAM), then bump sem_x on every split core. s > 0: send the kz partial tiles to slot s - 1 of the
// root's cb_recv (same L1 address on every core) and bump the root's sem_z. Root: wait for all partials, hand them
// to compute, write z (kz fp32 tiles) to DRAM.
#define kernel_main prep_main
#include "gate_prep_writer.cpp"
#undef kernel_main

void kernel_main() {
    constexpr uint32_t kz = get_named_compile_time_arg_val("kz");
    constexpr uint32_t n_split = get_named_compile_time_arg_val("n_split");
    constexpr uint32_t cb_recv = 7, cb_z = 18;
    constexpr uint32_t A = 3;  // first z arg

    const uint32_t z_addr = get_arg_val<uint32_t>(A + 0);
    const uint32_t s = get_arg_val<uint32_t>(A + 1);
    const uint32_t sem_x = get_semaphore(get_named_compile_time_arg_val("sem_x"));
    const uint32_t sem_z = get_semaphore(get_named_compile_time_arg_val("sem_z"));
    // A + 2: prep flag; A + 3 ..: virtual (x, y) of the n_split cores, root first
    if (get_arg_val<uint32_t>(A + 2)) {
        prep_main();  // ends with a write barrier: the x row is in DRAM
        for (uint32_t c = 0; c < n_split; ++c) {
            noc_semaphore_inc(
                get_noc_addr(get_arg_val<uint32_t>(A + 3 + 2 * c), get_arg_val<uint32_t>(A + 4 + 2 * c), sem_x), 1);
        }
        noc_async_atomic_barrier();
    }
    const uint32_t root_x = get_arg_val<uint32_t>(A + 3), root_y = get_arg_val<uint32_t>(A + 4);

    if (s == 0) {
        volatile tt_l1_ptr uint32_t* sp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_z);
        noc_semaphore_wait(sp, n_split - 1);
        noc_semaphore_set(sp, 0);
        cb_reserve_back(cb_recv, (n_split - 1) * kz);
        cb_push_back(cb_recv, (n_split - 1) * kz);
        cb_wait_front(cb_z, kz);
        rowio::write_pages(z_addr, 4096, 0, kz, get_read_ptr(cb_z));
        noc_async_write_barrier();
        return;
    }
    cb_wait_front(cb_z, kz);
    const uint32_t dst = get_write_ptr(cb_recv) + (s - 1) * kz * 4096;
    noc_async_write(get_read_ptr(cb_z), get_noc_addr(root_x, root_y, dst), kz * 4096);
    noc_async_write_barrier();
    noc_semaphore_inc(get_noc_addr(root_x, root_y, sem_z), 1);
    noc_async_atomic_barrier();
}
