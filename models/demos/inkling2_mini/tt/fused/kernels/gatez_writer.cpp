// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Message-gate prep + down_proj (E10c), writer of K part s. Prep cores wait for their rms row (cb_np), then bump
// sem_x on every core (args 0..2 unused). s > 0: send rows 0..R-1 of the kz partial tiles into packed rows of the
// root's cb_recv (same L1 address on every core) and bump the root's sem_z. Root: wait for all partials, hand them
// to compute, write z rows (compact, several copies) to DRAM.
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
        if (get_arg_val<uint32_t>(A + 2) == 2) {  // prep only (fused gate)
            return;
        }
    }
    const uint32_t root_x = get_arg_val<uint32_t>(A + 3), root_y = get_arg_val<uint32_t>(A + 4);

    if (s == 0) {
        DeviceZoneScopedN("W_root");
        volatile tt_l1_ptr uint32_t* sp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_z);
        // rows no sender owns get zeros (S is 0 there, but stale NaN * 0 would still be NaN); senders write other
        // rows only, so this runs while they compute
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
        {
            DeviceZoneScopedN("W_semz");
            noc_semaphore_wait(sp, n_split - 1);
        }
        noc_semaphore_set(sp, 0);
        cb_reserve_back(cb_recv, nb * kz);
        cb_push_back(cb_recv, nb * kz);
        {
            DeviceZoneScopedN("W_zwait");
            cb_wait_front(cb_z, kz);
        }
        if constexpr (get_named_compile_time_arg_val("fused")) {
            DeviceZoneScopedN("W_mc");
            // z + combine in one program: z in compact form (rows 0..rows-1 only, block (t, r) = 128 B at
            // (t * rows + r) * 128) into the combine cores' cb 31 (same L1 address there), then set their sem_zl.
            // Args after the prep input args: the number of combine rectangles, then per rectangle its virtual
            // start x, y, end x, y, core count.
            const uint32_t zt = get_read_ptr(cb_z), zl = get_write_ptr(31), land = zl;
            for (uint32_t t = 0; t < kz; ++t) {
                for (uint32_t r = 0; r < rows; ++r) {
                    const uint32_t so = zt + t * 4096 + rowio::row_offset(r, 4), d = zl + (t * rows + r) * 128;
                    noc_async_read(get_noc_addr(so), d, 64);
                    noc_async_read(get_noc_addr(so + 1024), d + 64, 64);
                }
            }
            noc_async_read_barrier();
            const uint32_t b = A + 3 + 2 * n_split + 5;
            const uint32_t n_rect = get_arg_val<uint32_t>(b);
            const uint32_t sem_zl = get_semaphore(get_named_compile_time_arg_val("sem_zl"));
            const bool far = noc_index != 0;  // NOC1 multicasts start at the bottom-right corner
            auto mc = [far](uint32_t x0, uint32_t y0, uint32_t x1, uint32_t y1, uint32_t a) {
                return far ? get_noc_multicast_addr(x1, y1, x0, y0, a) : get_noc_multicast_addr(x0, y0, x1, y1, a);
            };
            for (uint32_t r = 0; r < n_rect; ++r) {
                const uint32_t q = b + 1 + 5 * r;
                const uint32_t x0 = get_arg_val<uint32_t>(q), y0 = get_arg_val<uint32_t>(q + 1);
                const uint32_t x1 = get_arg_val<uint32_t>(q + 2), y1 = get_arg_val<uint32_t>(q + 3);
                noc_async_write_multicast(zl, mc(x0, y0, x1, y1, land), kz * rows * 128, get_arg_val<uint32_t>(q + 4));
            }
            {
                DeviceZoneScopedN("W_mcbar");
                noc_async_write_barrier();
            }
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_zl) = 1;
            for (uint32_t r = 0; r < n_rect; ++r) {
                const uint32_t q = b + 1 + 5 * r;
                const uint32_t x0 = get_arg_val<uint32_t>(q), y0 = get_arg_val<uint32_t>(q + 1);
                const uint32_t x1 = get_arg_val<uint32_t>(q + 2), y1 = get_arg_val<uint32_t>(q + 3);
                noc_semaphore_set_multicast(sem_zl, mc(x0, y0, x1, y1, sem_zl), get_arg_val<uint32_t>(q + 4));
            }
            noc_async_write_barrier();
            return;
        }
        // z out in compact form: rows 0..rows-1 only, block (t, r) = 128 B at (t * rows + r) * 128 of one page,
        // written as n_copies pages (on different DRAM banks: the 72 combine cores each read one copy)
        constexpr uint32_t n_copies = get_named_compile_time_arg_val("z_copies");
        const uint32_t zl = get_read_ptr(cb_z), zc = get_write_ptr(31);
        for (uint32_t t = 0; t < kz; ++t) {
            for (uint32_t r = 0; r < rows; ++r) {
                const uint32_t so = zl + t * 4096 + rowio::row_offset(r, 4), d = zc + (t * rows + r) * 128;
                noc_async_read(get_noc_addr(so), d, 64);
                noc_async_read(get_noc_addr(so + 1024), d + 64, 64);
            }
        }
        noc_async_read_barrier();
        const InterleavedAddrGen<true> zo = {.bank_base_address = z_addr, .page_size = 4096};
        for (uint32_t c = 0; c < n_copies; ++c) {
            noc_async_write(zc, zo.get_noc_addr(c), kz * rows * 128);
        }
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
