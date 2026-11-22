// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Message-gate prep + down_proj (E10c), reader of split s (one core per split).
//   prep cores (prep = 1, one per gated vector): the gate-prep reader (attnres_reader, args 0..8)
//   then: down_proj tiles of split s (prefetched), wait until every prep core has written its x row, x tiles of
//   split s; the root (s = 0) also reads the identity tile it uses to add the other splits' partial sums.
#define kernel_main prep_main
#include "attnres_reader.cpp"
#undef kernel_main
#include "api/debug/kernel_profiler.hpp"

void kernel_main() {
    constexpr uint32_t kt_n = get_named_compile_time_arg_val("kt");  // K tiles per split
    constexpr uint32_t kz = get_named_compile_time_arg_val("kz");    // output tiles (64 g / 32)
    constexpr uint32_t n_prep = get_named_compile_time_arg_val("n_prep");
    constexpr uint32_t cb_x = 5, cb_w = 6, cb_eye = 8, cb_zero = 9;
    constexpr uint32_t A = 9;  // first z arg

    if (get_arg_val<uint32_t>(A + 4)) {
        prep_main();
    }
    const uint32_t x_addr = get_arg_val<uint32_t>(A + 0);
    const uint32_t w_addr = get_arg_val<uint32_t>(A + 1);
    const uint32_t s = get_arg_val<uint32_t>(A + 2);
    const uint32_t eye_addr = get_arg_val<uint32_t>(A + 3);

    cb_reserve_back(cb_zero, 1);  // DST is not zeroed between programs: matmuls start from a copied zero tile
    rowio::zero_words(get_write_ptr(cb_zero), 4096);
    cb_push_back(cb_zero, 1);
    cb_reserve_back(cb_w, kt_n * kz);
    rowio::read_pages(w_addr, 2048, s * kt_n * kz, kt_n * kz, get_write_ptr(cb_w));
    if (s == 0) {
        cb_reserve_back(cb_eye, 1);
        rowio::read_pages(eye_addr, 4096, 0, 1, get_write_ptr(cb_eye));
    }
    {
        DeviceZoneScopedN("R_wait_x");
        volatile tt_l1_ptr uint32_t* sp =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(get_named_compile_time_arg_val("sem_x")));
        noc_semaphore_wait(sp, n_prep);
        noc_semaphore_set(sp, 0);
    }
    DeviceZoneScopedN("R_x");
    cb_reserve_back(cb_x, kt_n);
    rowio::read_pages(x_addr, 2048, s * kt_n, kt_n, get_write_ptr(cb_x));
    noc_async_read_barrier();
    cb_push_back(cb_x, kt_n);
    cb_push_back(cb_w, kt_n * kz);
    if (s == 0) {
        cb_push_back(cb_eye, 1);
    }
}
