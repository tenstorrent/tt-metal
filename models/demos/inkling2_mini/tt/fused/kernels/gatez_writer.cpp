// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Message-gate prep + down_proj (E10c), writer of K part s. Prep cores wait for their rms row (cb_np), then bump
// sem_x on every core (args 0..2 unused). s > 0: send rows 0..R-1 of the kz partial tiles into packed rows of the
// root's cb_recv (same L1 address on every core) and bump the root's sem_z. Root: wait for all partials, hand them
// to compute, write z (kz fp32 tiles) to DRAM.
#include "rowio.hpp"
#include "api/debug/kernel_profiler.hpp"

void kernel_main() {
    constexpr uint32_t kz = get_named_compile_time_arg_val("kz");
    constexpr uint32_t n_split = get_named_compile_time_arg_val("n_split");
    constexpr uint32_t rows = get_named_compile_time_arg_val("rows");
    constexpr uint32_t nb = get_named_compile_time_arg_val("nb");  // packed partial-row tiles per z tile
    constexpr uint32_t cb_recv = 7, cb_z = 18;
    constexpr uint32_t A = 3;  // first z arg

    const uint32_t z_addr = get_arg_val<uint32_t>(A + 0);
    const uint32_t s = get_arg_val<uint32_t>(A + 1);
    const uint32_t sem_x = get_semaphore(get_named_compile_time_arg_val("sem_x"));
    const uint32_t sem_z = get_semaphore(get_named_compile_time_arg_val("sem_z"));
    // A + 2: prep flag; A + 3 ..: virtual (x, y) of the n_split cores, root first
    if (get_arg_val<uint32_t>(A + 2)) {
        constexpr uint32_t T = get_named_compile_time_arg_val("pages");
        constexpr uint32_t ntc = get_named_compile_time_arg_val("ntc");
        {  // second half of the prep input row (args after the core list: the prep reader's input args)
            DeviceZoneScopedN("W_gather");
            const uint32_t b = A + 3 + 2 * n_split;
            const uint32_t p_addr = get_arg_val<uint32_t>(b), p_tile = get_arg_val<uint32_t>(b + 1);
            const uint32_t p_es = get_arg_val<uint32_t>(b + 2), p_row = get_arg_val<uint32_t>(b + 3);
            if (!get_arg_val<uint32_t>(b + 4)) {  // not a dense twin
                cb_reserve_back(29, 1);
                const uint32_t stage = align_power_of_2(get_write_ptr(30), rowio::DRAM_RD_ALIGN);
                rowio::gather_row(
                    p_addr, p_tile, p_es, ntc, p_row, get_write_ptr(0), T * 1024 * p_es, stage, ntc / 2, ntc);
                cb_push_back(29, 1);
            }
        }
        cb_wait_front(16, T);  // the rms row is in cb_np (the split cores read it from here)
        for (uint32_t c = 0; c < n_split; ++c) {
            noc_semaphore_inc(
                get_noc_addr(get_arg_val<uint32_t>(A + 3 + 2 * c), get_arg_val<uint32_t>(A + 4 + 2 * c), sem_x), 1);
        }
        noc_async_atomic_barrier();
    }
    const uint32_t root_x = get_arg_val<uint32_t>(A + 3), root_y = get_arg_val<uint32_t>(A + 4);

    if (s == 0) {
        DeviceZoneScopedN("W_root");
        volatile tt_l1_ptr uint32_t* sp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_z);
        noc_semaphore_wait(sp, n_split - 1);
        noc_semaphore_set(sp, 0);
        // rows no sender owns get zeros (S is 0 there, but stale NaN * 0 would still be NaN)
        cb_wait_front(9, 1);  // cb_zero (reader)
        const uint64_t zero = get_noc_addr(get_read_ptr(9));
        const uint32_t recv = get_write_ptr(cb_recv);
        for (uint32_t g = (n_split - 1) * rows; g < nb * 32; ++g) {
            for (uint32_t t = 0; t < kz; ++t) {
                const uint32_t off = (t * nb + g / 32) * 4096 + rowio::row_offset(g % 32, 4);
                noc_async_read(zero, recv + off, 64);
                noc_async_read(zero, recv + off + 1024, 64);
            }
        }
        noc_async_read_barrier();
        cb_reserve_back(cb_recv, nb * kz);
        cb_push_back(cb_recv, nb * kz);
        cb_wait_front(cb_z, kz);
        rowio::write_pages(z_addr, 4096, 0, kz, get_read_ptr(cb_z));
        noc_async_write_barrier();
        return;
    }
    cb_wait_front(cb_z, kz);
    // only rows 0..rows-1 of the partials matter (the combine reads only those rows of z). Row r of sender s goes
    // to row g = (s - 1) * rows + r of the root's packed tiles: recv tile t * nb + g / 32, row g % 32.
    const uint32_t recv = get_write_ptr(cb_recv);
    const uint32_t src = get_read_ptr(cb_z);
    for (uint32_t t = 0; t < kz; ++t) {
        for (uint32_t r = 0; r < rows; ++r) {
            const uint32_t g = (s - 1) * rows + r;
            const uint32_t so = t * 4096 + rowio::row_offset(r, 4);
            const uint32_t d = recv + (t * nb + g / 32) * 4096 + rowio::row_offset(g % 32, 4);
            noc_async_write(src + so, get_noc_addr(root_x, root_y, d), 64);
            noc_async_write(src + so + 1024, get_noc_addr(root_x, root_y, d + 1024), 64);
        }
    }
    noc_async_write_barrier();
    noc_semaphore_inc(get_noc_addr(root_x, root_y, sem_z), 1);
    noc_async_atomic_barrier();
}
