// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Resident DFlash2 drafter ring, slot zeroing (BRISC; M7 ingest v5, C9: M7_BUILD_PREREG_v1_ADDENDUM_1): a ring slot handed
// to a user without a seed is zeroed before its first draft, so the drafter never reads a previous occupant's K / V (the
// drafter window clamps at 0). The DrafterRing's K and V are DRAM TILE BF8 tensors whose slot s is the tile range
// [s n, (s + 1) n) (n = its rows / 32 x 8 tiles); this core writes a zero tile over its share of both. An all-zero BF8
// tile (exponents and mantissas 0) is the value 0. Writes only the named ring pages.
// Runtime args: 0 ring K address, 1 ring V address, 2 first tile, 3 tiles (this core's share of the slot's range)

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t kBf8Tile = 1088;
constexpr uint32_t cb_zero = 0;

void kernel_main() {
    const uint32_t k_addr = get_arg_val<uint32_t>(0);
    const uint32_t v_addr = get_arg_val<uint32_t>(1);
    const uint32_t first = get_arg_val<uint32_t>(2);
    const uint32_t tiles = get_arg_val<uint32_t>(3);
    auto& iface = get_local_cb_interface(cb_zero);
    const uint32_t zero = iface.fifo_limit - iface.fifo_size;
    volatile tt_l1_ptr uint32_t* z = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(zero);
    for (uint32_t i = 0; i < kBf8Tile / 4; i++) {
        z[i] = 0;
    }
    const InterleavedAddrGen<true> kg{.bank_base_address = k_addr, .page_size = kBf8Tile};
    const InterleavedAddrGen<true> vg{.bank_base_address = v_addr, .page_size = kBf8Tile};
    for (uint32_t t = first; t < first + tiles; t++) {
        noc_async_write(zero, kg.get_noc_addr(t), kBf8Tile);
        noc_async_write(zero, vg.get_noc_addr(t), kBf8Tile);
    }
    noc_async_write_barrier();
}
