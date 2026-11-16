// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// K2a standalone harness (k2a_harness.py; not part of the engine): one core stands in for the 32 streamers. At launch
// its o_in address to every head core's H_OADDR; per layer: each head core's rx tiles (q at [0, Dt), k / v at
// [16, 16 + Dt), 32x32 tiles of the 32 launch rows) from DRAM into that head's rx, + 1 on its A_RX; once the 8 heads'
// O tiles are in o_in (sem_o), o_in to DRAM (the layer's dump).
//
// Runtime args: 0 rx source (DRAM bf16 tiles: (l * 8 + h) * 32 + idx), 1 O dump (DRAM bf16 tiles: l * 128 + i),
// 2 sem_o id, 3 sem_rxaddr id, 4 layers, then 8 head core xy (x | y << 16)

#include "api/dataflow/dataflow_api.h"

namespace {
constexpr uint32_t kT = 2048;
FORCE_INLINE uint32_t cb_base(uint32_t cb) {
    auto& iface = get_local_cb_interface(cb);
    return iface.fifo_limit - iface.fifo_size;
}
}  // namespace

void kernel_main() {
    const uint32_t rx_src = get_arg_val<uint32_t>(0), dump = get_arg_val<uint32_t>(1);
    volatile tt_l1_ptr uint32_t* o_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(get_arg_val<uint32_t>(2)));
    volatile tt_l1_ptr uint32_t* rxbox = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(get_arg_val<uint32_t>(3)));
    const uint32_t layers = get_arg_val<uint32_t>(4);
    constexpr uint32_t A_RX = 0, A_OADDR = 5;
    const uint32_t o_in = cb_base(0), scratch = cb_base(1);
    auto head = [&](uint32_t h, uint32_t local) {
        const uint32_t xy = get_arg_val<uint32_t>(5 + h);
        return get_noc_addr(xy & 0xffff, xy >> 16, local);
    };
    for (uint32_t h = 0; h < 8; h++) {
        noc_inline_dw_write(head(h, get_semaphore(A_OADDR)), o_in);
    }
    uint32_t rx = 0;
    while (rx == 0) {
        invalidate_l1_cache();
        rx = *rxbox;
    }
    const InterleavedAddrGenFast<true> src{.bank_base_address = rx_src, .page_size = kT, .data_format = DataFormat::Float16_b};
    const InterleavedAddrGenFast<true> dst{.bank_base_address = dump, .page_size = kT, .data_format = DataFormat::Float16_b};
    for (uint32_t l = 0; l < layers; l++) {
        for (uint32_t h = 0; h < 8; h++) {
            for (uint32_t i = 0; i < 32; i++) {
                noc_async_read_page((l * 8 + h) * 32 + i, src, scratch + i * kT);
            }
            noc_async_read_barrier();
            noc_async_write(scratch, head(h, rx), 32 * kT);
            noc_async_write_barrier();
            noc_semaphore_inc(head(h, get_semaphore(A_RX)), 1);
            noc_async_atomic_barrier();
        }
        noc_semaphore_wait_min(o_sem, 8 * (l + 1));
        for (uint32_t i = 0; i < 128; i++) {
            noc_async_write_page(l * 128 + i, dst, o_in + i * kT);
        }
        noc_async_write_barrier();
    }
}
