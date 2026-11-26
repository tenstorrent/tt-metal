// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Fused attention glue, prep input fetch: runs inside the w_all projection's program on the prep cores (idle there)
// and reads this app's prep DRAM inputs into the prep scratch, an L1 shard on each prep core (attn.py PREP_PRE).
// The prep's readers then take them from L1 (attn_head_reader.cpp / attn_bias_reader.cpp `use`); the slim bytes,
// in the scratch layouts those readers describe.
//   role 0 (q / k / v): taps of item i lag l (512 B of page first_i + l) at 512 (i nlag + l); has_tau: tau's first
//     64 B at 2048. Runtime args: scratch, taps, tau, has_tau, first page per item.
//   role 1 (bias): rel_scale face 0 at 0 and face 2 rows 0..7 at 1024, near rows at 1536, ring slot at s_ring,
//     tau at s_ring + 64, then (ring slot known) the profile row blocks at s_ring + 128 + 1088 j. Runtime args:
//     scratch, rel_scale, near, ring, tau, has_tau, table, j0.
#include "attn_io.hpp"

void kernel_main() {
    constexpr uint32_t role = get_named_compile_time_arg_val("role");
    constexpr uint32_t nitems = get_named_compile_time_arg_val("nitems");
    constexpr uint32_t nlag = get_named_compile_time_arg_val("nlag");
    constexpr uint32_t nj = get_named_compile_time_arg_val("nj");
    uint32_t a = 0;
    const uint32_t scratch = get_arg_val<uint32_t>(a++);
    if constexpr (role == 0) {
        const uint32_t taps = get_arg_val<uint32_t>(a++);
        const uint32_t tau = get_arg_val<uint32_t>(a++);
        const uint32_t has_tau = get_arg_val<uint32_t>(a++);
        const InterleavedAddrGen<true> ta = {.bank_base_address = taps, .page_size = 4096};
        for (uint32_t i = 0; i < nitems; ++i) {
            const uint32_t page = get_arg_val<uint32_t>(a++);
            for (uint32_t l = 0; l < nlag; ++l) {
                noc_async_read(ta.get_noc_addr(page + l, 0), scratch + (i * nlag + l) * 512, 512);
            }
        }
        if (has_tau) {
            const InterleavedAddrGen<true> tau_a = {.bank_base_address = tau, .page_size = 4096};
            noc_async_read(tau_a.get_noc_addr(0, 0), scratch + 2048, 64);
        }
        noc_async_read_barrier();
    } else {
        constexpr uint32_t s_near = 1536, s_ring = s_near + 128 * nj, s_tau = s_ring + 64, s_prof = s_ring + 128;
        const uint32_t rs = get_arg_val<uint32_t>(a++);
        const uint32_t near = get_arg_val<uint32_t>(a++);
        const uint32_t ring = get_arg_val<uint32_t>(a++);
        const uint32_t tau = get_arg_val<uint32_t>(a++);
        const uint32_t has_tau = get_arg_val<uint32_t>(a++);
        const uint32_t table = get_arg_val<uint32_t>(a++);
        const uint32_t j0 = get_arg_val<uint32_t>(a++);
        const InterleavedAddrGen<true> rs_a = {.bank_base_address = rs, .page_size = 4096};
        const InterleavedAddrGen<true> near_a = {.bank_base_address = near, .page_size = 4096};
        const InterleavedAddrGen<true> ring_a = {.bank_base_address = ring, .page_size = 4};
        noc_async_read(rs_a.get_noc_addr(0, 0), scratch, 1024);
        noc_async_read(rs_a.get_noc_addr(0, 2048), scratch + 1024, 512);
        for (uint32_t j = 0; j < nj; ++j) {
            noc_async_read(near_a.get_noc_addr(j0 + j, 0), scratch + s_near + j * 128, 64);
            noc_async_read(near_a.get_noc_addr(j0 + j, 1024), scratch + s_near + j * 128 + 64, 64);
        }
        noc_async_read(ring_a.get_noc_addr(0, 0), scratch + s_ring, 64);
        if (has_tau) {
            const InterleavedAddrGen<true> tau_a = {.bank_base_address = tau, .page_size = 4096};
            noc_async_read(tau_a.get_noc_addr(0, 0), scratch + s_tau, 64);
        }
        noc_async_read_barrier();
        const uint32_t rpos = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch + s_ring);
        const InterleavedAddrGen<true> tab = {.bank_base_address = table, .page_size = 1};
        for (uint32_t j = 0; j < nj; ++j) {
            const uint32_t off = ((32 * (j0 + j) + 1024 - rpos) & 1023) * 32;
            noc_async_read(tab.get_noc_addr(0, off & ~(attnio::ALIGN - 1)), scratch + s_prof + j * 1088, 1088);
        }
        noc_async_read_barrier();
    }
}
