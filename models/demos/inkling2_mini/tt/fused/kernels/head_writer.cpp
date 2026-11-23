// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Logits tail, step 2 writer: the core's softcapped values -> their place in the row-major fp32 logits row,
// and the core's (max, index) -> its 16 B slot in the master core's scratch (slot = core order = column
// order), then a bump of the master's semaphore. The master reduces the slots in order and writes the token
// (uint32). Comparison and tie rule as ttnn.argmax on fp32: strictly greater wins (first index on ties),
// +0 == -0, the first NaN wins.
#include "rowio.hpp"

namespace {
FORCE_INLINE bool is_nan(uint32_t a) { return (a & 0x7fffffffu) > 0x7f800000u; }
FORCE_INLINE uint32_t key(uint32_t a) {  // ordered key, -0 folded into +0
    a = (a == 0x80000000u) ? 0 : a;
    return a ^ (static_cast<uint32_t>(static_cast<int32_t>(a) >> 31) | 0x80000000u);
}
FORCE_INLINE bool greater(uint32_t a, uint32_t b) {
    const bool an = is_nan(a), bn = is_nan(b);
    if (an || bn) {
        return an && !bn;
    }
    return key(a) > key(b);
}
}  // namespace

void kernel_main() {
    constexpr uint32_t T = get_named_compile_time_arg_val("pages");
    constexpr uint32_t logits_page = get_named_compile_time_arg_val("logits_page");
    constexpr uint32_t sem_id = get_named_compile_time_arg_val("sem");
    constexpr uint32_t cb_out = 16, cb_scr = 17;

    const uint32_t logits = get_arg_val<uint32_t>(0);
    const uint32_t col0 = get_arg_val<uint32_t>(1);     // first column of this core
    const uint32_t n_write = get_arg_val<uint32_t>(2);  // columns to write (< logits width)
    const uint32_t n_arg = get_arg_val<uint32_t>(3);    // columns that take part in the argmax (< real vocab)
    const uint32_t slot = get_arg_val<uint32_t>(4);
    const uint32_t mx = get_arg_val<uint32_t>(5);
    const uint32_t my = get_arg_val<uint32_t>(6);
    const uint32_t n_cores = get_arg_val<uint32_t>(7);
    const uint32_t token = get_arg_val<uint32_t>(8);

    // R verify rows (rows > 1): every row's logits and (max, index) slot; args 9..: rows 1.. logits, token
    constexpr uint32_t R = get_named_compile_time_arg_val("rows");
    cb_wait_front(cb_out, T * R);
    const uint32_t scr = get_write_ptr(cb_scr);  // same address on every core; slots of row r at 16 * r * n_cores
    for (uint32_t r = 0; r < R; ++r) {
        const uint32_t src = get_read_ptr(cb_out) + r * T * 4096;
        const InterleavedAddrGen<true> lg = {
            .bank_base_address = r == 0 ? logits : get_arg_val<uint32_t>(7 + 2 * r), .page_size = logits_page};
        if (n_write) {
            noc_async_write(src, lg.get_noc_addr(0, col0 * 4), n_write * 4);
        }

        // a strict-compare scan over ordered keys; the first NaN, if any, wins outright
        const uint32_t* v = reinterpret_cast<const uint32_t*>(src);
        uint32_t bk = 0, bi = 0;
        for (uint32_t i = 0; i < n_arg; ++i) {
            const uint32_t a = v[i];
            if (is_nan(a)) {
                bi = i;
                break;
            }
            const uint32_t kv = key(a);
            if (kv > bk || i == 0) {
                bk = kv;
                bi = i;
            }
        }
        const uint32_t own = scr + 16 * (R * n_cores + r);
        volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(own);
        s[0] = n_arg ? v[bi] : 0;
        s[1] = col0 + bi;
        s[2] = n_arg ? 1 : 0;
        s[3] = 0;
        noc_async_write(own, get_noc_addr(mx, my, scr + 16 * (r * n_cores + slot)), 16);
    }
    noc_async_write_barrier();
    cb_pop_front(cb_out, T * R);
    if (slot != 0) {
        noc_semaphore_inc(get_noc_addr(mx, my, get_semaphore(sem_id)), 1);
        noc_async_atomic_barrier();
        return;
    }
    volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_id));
    noc_semaphore_wait(sem, n_cores - 1);
    for (uint32_t r = 0; r < R; ++r) {
        volatile tt_l1_ptr uint32_t* a = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr + 16 * r * n_cores);
        uint32_t gb = 0, gi = 0;
        bool ghave = false;
        for (uint32_t k = 0; k < n_cores; ++k) {
            volatile tt_l1_ptr uint32_t* e = a + k * 4;
            if (e[2] && (!ghave || greater(e[0], gb))) {
                gb = e[0];
                gi = e[1];
                ghave = true;
            }
        }
        const uint32_t own = scr + 16 * (R * n_cores + r);
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(own) = gi;
        const InterleavedAddrGen<true> tk = {
            .bank_base_address = r == 0 ? token : get_arg_val<uint32_t>(8 + 2 * r), .page_size = 4};
        noc_async_write(own, tk.get_noc_addr(0), 4);
    }
    noc_async_write_barrier();
}
