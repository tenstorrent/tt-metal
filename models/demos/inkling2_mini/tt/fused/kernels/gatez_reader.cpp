// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Message-gate prep + down_proj (E10c), reader of K part s (one core per part).
//   prep cores (prep = 1, one per gated vector): the gate-prep reader (attnres_reader, args 0..8)
//   then: down_proj tiles of K part s (prefetched), wait until every prep core has its rms row, x tiles of part s
//   from their L1. A part is kt consecutive K tiles from x_first; down_proj pages w_first .. (kt * kz).
//   Args A + 5 ..: virtual (x, y) of the cores, prep cores first.
#define kernel_main prep_main
#define ATTNRES_SPLIT_GATHER
#include "attnres_reader.cpp"
#undef kernel_main
#include "api/debug/kernel_profiler.hpp"

void kernel_main() {
    constexpr uint32_t kt_n = get_named_compile_time_arg_val("kt");  // K tiles per split
    constexpr uint32_t kz = get_named_compile_time_arg_val("kz");    // output tiles (64 g / 32)
    constexpr uint32_t n_prep = get_named_compile_time_arg_val("n_prep");
    constexpr uint32_t cb_x = 5, cb_w = 6, cb_sel = 8, cb_zero = 9;
    constexpr uint32_t A = 9;  // first z arg

    if (get_arg_val<uint32_t>(A + 4)) {
        prep_main();
    }
    const uint32_t w_addr = get_arg_val<uint32_t>(A + 1);
    const uint32_t x_first = get_arg_val<uint32_t>(A + 2);
    const uint32_t w_first = get_arg_val<uint32_t>(A + 3);

    cb_reserve_back(cb_zero, 1);  // DST is not zeroed between programs: matmuls start from a copied zero tile
    rowio::zero_words(get_write_ptr(cb_zero), 4096);
    cb_push_back(cb_zero, 1);
    if (x_first == 0) {  // root: the row-selection tiles that sum the packed partial rows
        constexpr uint32_t nb = get_named_compile_time_arg_val("nb");
        cb_reserve_back(cb_sel, nb);
        rowio::read_pages(get_arg_val<uint32_t>(A + 0), 4096, 0, nb, get_write_ptr(cb_sel));
        noc_async_read_barrier();
        cb_push_back(cb_sel, nb);
    }
    cb_reserve_back(cb_w, kt_n * kz);
    rowio::read_pages(w_addr, 2048, w_first, kt_n * kz, get_write_ptr(cb_w));
    {
        DeviceZoneScopedN("R_wait_x");
        volatile tt_l1_ptr uint32_t* sp =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(get_named_compile_time_arg_val("sem_x")));
        noc_semaphore_wait(sp, n_prep);
        noc_semaphore_set(sp, 0);
    }
    DeviceZoneScopedN("R_x");
    // x tiles straight from the prep cores' L1: K tile t of row r is elements 32 t' .. of the bf16 rms row in
    // cb_np of prep core v (t < ntc: h, v = r, t' = t; else m, v = rows + r, t' = t - ntc). Rows >= rows of the
    // tiles stay stale: they only reach rows of z nobody reads.
    constexpr uint32_t ntc = get_named_compile_time_arg_val("ntc");
    constexpr uint32_t rows = get_named_compile_time_arg_val("rows");
    cb_reserve_back(cb_x, kt_n);
    const uint32_t xl = get_write_ptr(cb_x);
    const uint32_t np = get_read_ptr(16);  // cb_np: same L1 address on every core
    for (uint32_t i = 0; i < kt_n; ++i) {
        const uint32_t t = x_first + i;
        for (uint32_t r = 0; r < rows; ++r) {
            const uint32_t v = t < ntc ? r : rows + r;
            const uint32_t tt = t < ntc ? t : t - ntc;
            const uint64_t a =
                get_noc_addr(get_arg_val<uint32_t>(A + 5 + 2 * v), get_arg_val<uint32_t>(A + 6 + 2 * v), np + tt * 64);
            const uint32_t d = xl + i * 2048 + rowio::row_offset(r, 2);
            noc_async_read(a, d, 32);
            noc_async_read(a + 32, d + 512, 32);
        }
    }
    noc_async_read_barrier();
    cb_push_back(cb_x, kt_n);
    cb_push_back(cb_w, kt_n * kz);
}
