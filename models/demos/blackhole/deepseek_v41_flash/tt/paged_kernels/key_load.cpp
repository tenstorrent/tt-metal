// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// DeepSeek-V4.1-Flash device-resident block-paged APC (DSV41_APC_PAGED, tt/apc_paged_store.py): LOAD cached index-key
// blocks from the key STORE into the HISTORY rows of the single-trace key slab (the inverse of key_save.cpp; tensor
// driven: usable inside a trace, one launch per index-key owner).
//
//   store bf16 TILE [K, 1, NRB * 32, 128] key store (read only): page of (slot k, row block rb, tile column t) = k * NP
//   + rb * 4 + t, NP = NRB * 4 ltab  uint32 ROW_MAJOR [U, NBMAX]      per device: row u = store slots of the chunk
//   blocks 0, 1, 2, ... of user u's cached prefix, the list ends at the first
//                                          0xFFFFFFFF (an all-SKIP row: nothing is loaded for the user)
//   slab  bf16 TILE [U, 1, T, 128]         key slab of the owner (written IN PLACE): page of (user u, row block rb,
//   tile column t) = u * DP + rb * 4 + t
//
// Chunk block i goes to the history row blocks [NRB * (1 + i), NRB * (2 + i)) of the slab (history entry j at slot Cc +
// j; the scratch rows [0, NRB) come first), i.e. slab pages u * DP + NP * (1 + i) + [0, NP). One core per user (runtime
// arg u). Blocks that would not fit the slab (i >= MAXI) are not loaded.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

void kernel_main() {
    constexpr uint32_t NP = get_compile_time_arg_val(0);
    constexpr uint32_t DP = get_compile_time_arg_val(1);
    constexpr uint32_t TILE_B = get_compile_time_arg_val(2);
    constexpr uint32_t cb_id = get_compile_time_arg_val(3);
    constexpr uint32_t K = get_compile_time_arg_val(4);
    constexpr uint32_t NBMAX = get_compile_time_arg_val(5);
    constexpr uint32_t MAXI = get_compile_time_arg_val(6);
    constexpr auto src_args = TensorAccessorArgs<7>();
    constexpr auto tab_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();
    constexpr auto dst_args = TensorAccessorArgs<tab_args.next_compile_time_args_offset()>();

    constexpr uint32_t TAB_B = NBMAX * 4;
    constexpr uint32_t B_TAB = 0, B_T = TAB_B;
    constexpr uint32_t SKIP = 0xFFFFFFFFu;

    const uint32_t a_src = get_common_arg_val<uint32_t>(0), a_tab = get_common_arg_val<uint32_t>(1),
                   a_dst = get_common_arg_val<uint32_t>(2);
    const uint32_t u = get_arg_val<uint32_t>(0);

    Noc noc;
    const auto src_acc = TensorAccessor(src_args, a_src, TILE_B);
    const auto tab_acc = TensorAccessor(tab_args, a_tab, TAB_B);
    const auto dst_acc = TensorAccessor(dst_args, a_dst, TILE_B);
    experimental::CB cb(cb_id);
    cb.reserve_back(1);
    const uint32_t base = cb.get_write_ptr();
    volatile tt_l1_ptr uint32_t* tab = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + B_TAB);

    noc.async_read(tab_acc, cb, TAB_B, {.page_id = u, .offset_bytes = 0}, {.offset_bytes = B_TAB});
    noc.async_read_barrier();
    invalidate_l1_cache();
    for (uint32_t i = 0; i < NBMAX && i < MAXI; ++i) {
        const uint32_t slot = tab[i];
        if (slot == SKIP || slot >= K) {
            break;
        }
        for (uint32_t p = 0; p < NP; ++p) {
            noc.async_read(
                src_acc, cb, TILE_B, {.page_id = slot * NP + p, .offset_bytes = 0}, {.offset_bytes = B_T + p * TILE_B});
        }
        noc.async_read_barrier();
        for (uint32_t p = 0; p < NP; ++p) {
            noc.async_write(
                cb,
                dst_acc,
                TILE_B,
                {.offset_bytes = B_T + p * TILE_B},
                {.page_id = u * DP + NP * (1 + i) + p, .offset_bytes = 0});
        }
        noc.async_write_barrier();
    }
}
