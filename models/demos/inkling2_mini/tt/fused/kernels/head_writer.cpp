// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Logits tail, writer: the core's softcapped values -> their place in the row-major fp32 logits row, and the
// core's (max, index) -> its 64 B slot of the partials buffer (slot = chunk * cores + core, i.e. column order).
// Last chunk: the other cores bump the master's semaphore once their slot is written; the master then reduces
// all slots in order and writes the token (uint32). Comparison and tie rule as ttnn.argmax on fp32: strictly
// greater wins (first index on ties), +0 == -0, the first NaN wins.
#include "rowio.hpp"

namespace {
FORCE_INLINE bool is_nan(uint32_t a) { return (a & 0x7f800000u) == 0x7f800000u && (a & 0x007fffffu) != 0; }
FORCE_INLINE uint32_t key(uint32_t a) { return (a & 0x80000000u) ? ~a : (a | 0x80000000u); }
FORCE_INLINE bool greater(uint32_t a, uint32_t b) {
    const bool an = is_nan(a), bn = is_nan(b);
    if (an || bn) {
        return an && !bn;
    }
    if ((a & 0x7fffffffu) == 0 && (b & 0x7fffffffu) == 0) {
        return false;
    }
    return key(a) > key(b);
}
}  // namespace

void kernel_main() {
    constexpr uint32_t T = get_named_compile_time_arg_val("pages");
    constexpr uint32_t n_slots = get_named_compile_time_arg_val("n_slots");
    constexpr uint32_t logits_page = get_named_compile_time_arg_val("logits_page");
    constexpr uint32_t sem_id = get_named_compile_time_arg_val("sem");
    constexpr uint32_t cb_out = 16, cb_scr = 17;

    const uint32_t logits = get_arg_val<uint32_t>(0);
    const uint32_t col0 = get_arg_val<uint32_t>(1);     // first column of this core
    const uint32_t n_write = get_arg_val<uint32_t>(2);  // columns to write (< logits width)
    const uint32_t n_arg = get_arg_val<uint32_t>(3);    // columns that take part in the argmax (< real vocab)
    const uint32_t parts = get_arg_val<uint32_t>(4);
    const uint32_t slot = get_arg_val<uint32_t>(5);
    const uint32_t last = get_arg_val<uint32_t>(6);
    const uint32_t master = get_arg_val<uint32_t>(7);
    const uint32_t mx = get_arg_val<uint32_t>(8);
    const uint32_t my = get_arg_val<uint32_t>(9);
    const uint32_t n_peers = get_arg_val<uint32_t>(10);
    const uint32_t token = get_arg_val<uint32_t>(11);

    cb_wait_front(cb_out, T);
    const uint32_t src = get_read_ptr(cb_out);
    const InterleavedAddrGen<true> lg = {.bank_base_address = logits, .page_size = logits_page};
    if (n_write) {
        noc_async_write(src, lg.get_noc_addr(0, col0 * 4), n_write * 4);
    }

    // Same result as a scan with greater(): ordered keys (-0 folded into +0) and a strict compare; the first NaN,
    // if any, wins outright.
    const uint32_t* v = reinterpret_cast<const uint32_t*>(src);
    uint32_t bk = 0, bi = 0, nan_i = n_arg;
    for (uint32_t i = 0; i < n_arg; ++i) {
        uint32_t a = v[i];
        if ((a & 0x7fffffffu) > 0x7f800000u) {
            nan_i = i;
            break;
        }
        a = (a == 0x80000000u) ? 0 : a;
        const uint32_t kv = a ^ (static_cast<uint32_t>(static_cast<int32_t>(a) >> 31) | 0x80000000u);
        if (kv > bk || i == 0) {
            bk = kv;
            bi = i;
        }
    }
    const bool have = n_arg > 0;
    uint32_t best;
    if (nan_i < n_arg) {
        best = v[nan_i];
        bi = nan_i;
    } else {
        best = v[bi];
    }
    const uint32_t scr = get_write_ptr(cb_scr);
    volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr);
    s[0] = best;
    s[1] = col0 + bi;
    s[2] = have ? 1 : 0;
    s[3] = 0;
    const InterleavedAddrGen<true> pg = {.bank_base_address = parts, .page_size = n_slots * 64};
    noc_async_write(scr, pg.get_noc_addr(0, slot * 64), 16);
    noc_async_write_barrier();
    cb_pop_front(cb_out, T);
    if (!last) {
        return;
    }
    if (!master) {
        noc_semaphore_inc(get_noc_addr(mx, my, get_semaphore(sem_id)), 1);
        noc_async_atomic_barrier();
        return;
    }
    volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_id));
    noc_semaphore_wait(sem, n_peers);
    const uint32_t all = scr + 64;
    noc_async_read(pg.get_noc_addr(0, 0), all, n_slots * 64);
    noc_async_read_barrier();
    volatile tt_l1_ptr uint32_t* a = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(all);
    uint32_t gb = 0, gi = 0;
    bool ghave = false;
    for (uint32_t k = 0; k < n_slots; ++k) {
        volatile tt_l1_ptr uint32_t* e = a + k * 16;
        if (e[2] && (!ghave || greater(e[0], gb))) {
            gb = e[0];
            gi = e[1];
            ghave = true;
        }
    }
    s[0] = gi;
    const InterleavedAddrGen<true> tk = {.bank_base_address = token, .page_size = 4};
    noc_async_write(scr, tk.get_noc_addr(0), 4);
    noc_async_write_barrier();
}
