// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// moe_decode gate|up, bank-local form (RISCV_1, NOC_0). Two roles (runtime arg):
//   1 bank reader (bank b, reader r; on the bank's row): streams, for each of the k routed experts, its nr gate|up
//     columns over K slice b (one contiguous chunk of nr * ks tiles in bank b, layout [expert][reader][n][k]); all
//     chunks in flight at once (one transaction id each), each pushed as it lands.
//   2 reducer (one or more (routing slot, pair) units): waits until the banks' readers have written all its units'
//     partial (g, u) tiles into its cb_recv (semaphore = banks x units), then pushes them.
// compile-time args: [cb_w, cb_route, cb_recv, (unused), nr, readers, k, tile_bytes, route_page_bytes, sem_id, banks]
//                    + TensorAccessorArgs(route)
// runtime args: [w_addr, route_addr, role, bank, reader, ks (this bank's K tiles), units (reducer: units on core), vc]
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

static inline uint32_t bf16_to_uint(uint16_t bits) {
    union {
        uint32_t u;
        float f;
    } v;
    v.u = static_cast<uint32_t>(bits) << 16;
    return static_cast<uint32_t>(v.f);
}

void kernel_main() {
    const uint32_t w_addr = get_arg_val<uint32_t>(0);
    const uint32_t route_addr = get_arg_val<uint32_t>(1);
    const uint32_t role = get_arg_val<uint32_t>(2);
    const uint32_t bank = get_arg_val<uint32_t>(3);
    const uint32_t reader = get_arg_val<uint32_t>(4);
    const uint32_t ks = get_arg_val<uint32_t>(5);
    const uint32_t units = get_arg_val<uint32_t>(6);
    const uint32_t vc = get_arg_val<uint32_t>(7);
    constexpr uint32_t cb_w = get_compile_time_arg_val(0);
    constexpr uint32_t cb_route = get_compile_time_arg_val(1);
    constexpr uint32_t cb_recv = get_compile_time_arg_val(2);
    constexpr uint32_t nr = get_compile_time_arg_val(4);
    constexpr uint32_t readers = get_compile_time_arg_val(5);
    constexpr uint32_t k = get_compile_time_arg_val(6);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(7);
    constexpr uint32_t route_page_bytes = get_compile_time_arg_val(8);
    constexpr uint32_t sem_id = get_compile_time_arg_val(9);
    constexpr uint32_t banks = get_compile_time_arg_val(10);
    constexpr auto r_args = TensorAccessorArgs<11>();
    if (role == 1) {
        const auto s_r = TensorAccessor(r_args, route_addr, route_page_bytes);
        cb_reserve_back(cb_route, 1);
        const uint32_t route_l1 = get_write_ptr(cb_route);
        noc_async_read(s_r.get_noc_addr(0), route_l1, route_page_bytes);
        noc_async_read_barrier();
        volatile tt_l1_ptr uint16_t* row = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(route_l1);
        const uint32_t chunk_tiles = nr * ks;
        const uint32_t chunk_bytes = chunk_tiles * tile_bytes;
        cb_reserve_back(cb_w, k * chunk_tiles);
        const uint32_t l1 = get_write_ptr(cb_w);
        reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, noc_index);
        // gpt-oss stream pattern: the bank's base address and this reader's NOC virtual channel set once, then
        // <= 16 KB packets per transaction id
        const uint64_t base = get_noc_addr_from_bank_id<true>(bank, w_addr);
        const uint32_t page_tiles = ks;  // one K slice of one column per packet (ks * tile_bytes <= 16 KB)
        const uint32_t page_bytes = page_tiles * tile_bytes;
        noc_async_read_one_packet_set_state<true>(base, page_bytes, vc);
        for (uint32_t s = 0; s < k; ++s) {
            uint32_t src = (bf16_to_uint(row[k + s]) * readers + reader) * chunk_bytes;  // offset from base
            uint32_t dst = l1 + s * chunk_bytes;
            noc_async_read_set_trid(s + 1);
            for (uint32_t p = 0; p < nr; ++p) {
                noc_async_read_one_packet_with_state_with_trid(base, src, dst, s + 1);
                src += page_bytes;
                dst += page_bytes;
            }
        }
        for (uint32_t s = 0; s < k; ++s) {
            noc_async_read_barrier_with_trid(s + 1);
            cb_push_back(cb_w, chunk_tiles);
        }
        noc_async_read_set_trid(0);
    } else if (role == 2) {
        volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_id));
        cb_reserve_back(cb_recv, 2 * banks * units);
        noc_semaphore_wait(sem, banks * units);
        cb_push_back(cb_recv, 2 * banks * units);
    }
}
