// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Fused near partial, writer of core p = g * nsplit + i.
// 1. Row max: once compute has packed this core's score tiles S_j (fp32, rows 0..2 = the group's 3 heads), scan
//    rows 0..2 over them (and the sink block on the core that holds the sink) and push the max tile c (cb_c, fp32:
//    rows 0..2 = m_k in every column, rest 0) for compute's exp. The max is exact; c also goes to mp page p.
// 2. Partial z tile -> zp page p, the 4 O tiles -> op pages 4p .., all fp32.
#include "attn_io.hpp"

namespace {
// fp32 bits -> signed key with the same order as the floats (no NaNs; -0 just below +0): flip the magnitude bits of
// negatives. Its own inverse. Branch-free, so the scan keeps up with the score tiles compute packs.
FORCE_INLINE int32_t key(uint32_t x) {
    return static_cast<int32_t>(x ^ (static_cast<uint32_t>(static_cast<int32_t>(x) >> 31) >> 1));
}
FORCE_INLINE uint32_t unkey(int32_t k) { return static_cast<uint32_t>(k) ^ (static_cast<uint32_t>(k >> 31) >> 1); }

// max key over rows 0..2 of an fp32 tile at L1 t (faces 0/1: 16 values each per row)
FORCE_INLINE void row_max(uint32_t t, int32_t* m) {
    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(t);
    for (uint32_t k = 0; k < 3; ++k) {
        int32_t best = m[k];
        for (uint32_t f = 0; f < 2; ++f) {
            volatile tt_l1_ptr uint32_t* r = p + f * 256 + k * 16;
#pragma GCC unroll 16
            for (uint32_t c = 0; c < 16; ++c) {
                const int32_t kk = key(r[c]);
                best = kk > best ? kk : best;
            }
        }
        m[k] = best;
    }
}
}  // namespace

void kernel_main() {
    constexpr uint32_t nj = get_named_compile_time_arg_val("nj");
    constexpr uint32_t has_sink = get_named_compile_time_arg_val("has_sink");
    constexpr uint32_t cb_sink = 4, cb_c = 8, cb_s = 16, cb_z = 22, cb_o = 23, dt = 4;
    const uint32_t zp = get_arg_val<uint32_t>(0);
    const uint32_t op = get_arg_val<uint32_t>(1);
    const uint32_t mp = get_arg_val<uint32_t>(2);
    const uint32_t p = get_arg_val<uint32_t>(3);

    int32_t m[3] = {INT32_MIN, INT32_MIN, INT32_MIN};  // below every float
    if constexpr (has_sink) {   // the max is exact, so the scan order does not matter: sink first, then each S_j
        cb_wait_front(cb_sink, 1);  // as soon as compute packs it (the scan hides behind the next q.k)
        row_max(get_read_ptr(cb_sink), m);
    }
    const uint32_t s = get_read_ptr(cb_s);
    for (uint32_t j = 0; j < nj; ++j) {
        cb_wait_front(cb_s, j + 1);
        row_max(s + j * 4096, m);
    }
    cb_reserve_back(cb_c, 1);
    const uint32_t c = get_write_ptr(cb_c);
    attnio::zero_l1(c, 4096);
    noc_async_read_barrier();
    volatile tt_l1_ptr uint32_t* cp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(c);
    for (uint32_t k = 0; k < 3; ++k) {
        const uint32_t bits = unkey(m[k]);
        for (uint32_t f = 0; f < 2; ++f) {
            for (uint32_t x = 0; x < 16; ++x) {
                cp[f * 256 + k * 16 + x] = bits;
            }
        }
    }
    cb_push_back(cb_c, 1);
    const InterleavedAddrGen<true> md = {.bank_base_address = mp, .page_size = 4096};
    noc_async_write(c, md.get_noc_addr(p), 4096);

    const InterleavedAddrGen<true> zd = {.bank_base_address = zp, .page_size = 4096};
    const InterleavedAddrGen<true> od = {.bank_base_address = op, .page_size = 4096};
    cb_wait_front(cb_z, 1);
    noc_async_write(get_read_ptr(cb_z), zd.get_noc_addr(p), 4096);
    cb_wait_front(cb_o, dt);
    for (uint32_t t = 0; t < dt; ++t) {
        noc_async_write(get_read_ptr(cb_o) + t * 4096, od.get_noc_addr(p * dt + t), 4096);
    }
    noc_async_write_barrier();
    cb_pop_front(cb_z, 1);
    cb_pop_front(cb_o, dt);
}
