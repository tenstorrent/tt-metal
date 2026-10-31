// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// DeepSeek-V4.1-Flash device-resident block-paged APC (DSV41_APC_PAGED, tt/apc_paged_store.py): copy the 128-row window
// halos of NL layers between the persistent per-layer halo tensors and a halo STORE slot, device local (no CCL), tensor
// driven (usable inside a captured trace, SPMD over the mesh).
//
//   halos NL tensors, bf16 TILE [U, 1, 128, 512] (page of (user u, row block rb, tile column t) = u * 64 + rb * 16 +
//   t), addresses in the common args 1 .. NL store bf16 TILE [K * NL, 1, 128, 512]        the halo store: layer l of
//   slot k = pages ((k * NL) + l) * 64 + [0, 64) tab   uint32 ROW_MAJOR [U, 32]               per device: row u,
//   element 0 = store slot of user u, 0xFFFFFFFF = SKIP (user untouched by this launch)
//
//   DIR = 0 (SAVE):    store slot tab[u], layer l  <-  halo_l[u]
//   DIR = 1 (RESTORE): halo_l[u]  <-  store slot tab[u], layer l
//
// ``NC`` cores; core c moves the pages [c * PPC, (c + 1) * PPC) of every (user, layer) (PPC = 64 / NC). The halos are
// written IN PLACE (a restore must not move them: a captured trace keeps their address).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

void kernel_main() {
    constexpr uint32_t NL = get_compile_time_arg_val(0);
    constexpr uint32_t U = get_compile_time_arg_val(1);
    constexpr uint32_t PPC = get_compile_time_arg_val(2);
    constexpr uint32_t DIR = get_compile_time_arg_val(3);
    constexpr uint32_t TILE_B = get_compile_time_arg_val(4);
    constexpr uint32_t cb_id = get_compile_time_arg_val(5);
    constexpr uint32_t K = get_compile_time_arg_val(6);
    constexpr auto tab_args = TensorAccessorArgs<7>();
    constexpr auto store_args = TensorAccessorArgs<tab_args.next_compile_time_args_offset()>();
    constexpr auto halo_args = TensorAccessorArgs<store_args.next_compile_time_args_offset()>();

    constexpr uint32_t HP = 64;  // pages per (user, layer) halo
    constexpr uint32_t TAB_B = 128;
    constexpr uint32_t SKIP = 0xFFFFFFFFu;

    const uint32_t a_tab = get_common_arg_val<uint32_t>(0), a_store = get_common_arg_val<uint32_t>(1);
    const uint32_t c = get_arg_val<uint32_t>(0);

    Noc noc;
    const auto tab_acc = TensorAccessor(tab_args, a_tab, TAB_B);
    const auto store_acc = TensorAccessor(store_args, a_store, TILE_B);
    experimental::CB cb(cb_id);
    cb.reserve_back(1);
    const uint32_t base = cb.get_write_ptr();
    volatile tt_l1_ptr uint32_t* tab = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base);
    const uint32_t B_T = TAB_B;

    for (uint32_t u = 0; u < U; ++u) {
        noc.async_read(tab_acc, cb, TAB_B, {.page_id = u, .offset_bytes = 0}, {.offset_bytes = 0});
        noc.async_read_barrier();
        invalidate_l1_cache();
        const uint32_t slot = tab[0];
        if (slot == SKIP || slot >= K) {
            continue;
        }
        for (uint32_t l = 0; l < NL; ++l) {
            const auto halo_acc = TensorAccessor(halo_args, get_common_arg_val<uint32_t>(2 + l), TILE_B);
            const uint32_t sp0 = (slot * NL + l) * HP + c * PPC;
            const uint32_t hp0 = u * HP + c * PPC;
            for (uint32_t p = 0; p < PPC; ++p) {
                if constexpr (DIR == 0) {
                    noc.async_read(
                        halo_acc,
                        cb,
                        TILE_B,
                        {.page_id = hp0 + p, .offset_bytes = 0},
                        {.offset_bytes = B_T + p * TILE_B});
                } else {
                    noc.async_read(
                        store_acc,
                        cb,
                        TILE_B,
                        {.page_id = sp0 + p, .offset_bytes = 0},
                        {.offset_bytes = B_T + p * TILE_B});
                }
            }
            noc.async_read_barrier();
            for (uint32_t p = 0; p < PPC; ++p) {
                if constexpr (DIR == 0) {
                    noc.async_write(
                        cb,
                        store_acc,
                        TILE_B,
                        {.offset_bytes = B_T + p * TILE_B},
                        {.page_id = sp0 + p, .offset_bytes = 0});
                } else {
                    noc.async_write(
                        cb,
                        halo_acc,
                        TILE_B,
                        {.offset_bytes = B_T + p * TILE_B},
                        {.page_id = hp0 + p, .offset_bytes = 0});
                }
            }
            noc.async_write_barrier();
        }
    }
}
