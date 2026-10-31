// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// DeepSeek-V4.1-Flash device-resident block-paged APC (DSV41_APC_PAGED, tt/apc_paged_store.py): SAVE the chunk's index
// keys (the SCRATCH rows of the single-trace key slab, i.e. what ``key_append`` writes with offset 0) into the key
// STORE at a DEVICE-TENSOR driven slot. Twin of key_append.cpp (tensor driven, no host offsets: usable inside a
// captured trace).
//
//   slab  bf16 TILE [U, 1, T, 128]       the key slab of an index-key owner (read only); page of (user u, row block rb,
//   tile column t) = u * DP + rb * 4 + t sid   uint32 ROW_MAJOR [U, 32]       per device: row u, element 0 = store slot
//   of user u, 0xFFFFFFFF = SKIP (user not saved by this launch) store bf16 TILE [K, 1, NRB * 32, 128] the key store
//   (written IN PLACE): page of (slot k, row block rb, tile column t) = k * NP + rb * 4 + t, NP = NRB * 4
//
// Copies the NP scratch pages of user u (slab pages u * DP + [0, NP)) to store pages sid[u] * NP + [0, NP). One core
// per user (runtime arg u).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

void kernel_main() {
    constexpr uint32_t NP = get_compile_time_arg_val(0);
    constexpr uint32_t DP = get_compile_time_arg_val(1);
    constexpr uint32_t TILE_B = get_compile_time_arg_val(2);
    constexpr uint32_t cb_id = get_compile_time_arg_val(3);
    constexpr uint32_t K = get_compile_time_arg_val(4);
    constexpr auto src_args = TensorAccessorArgs<5>();
    constexpr auto sid_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();
    constexpr auto dst_args = TensorAccessorArgs<sid_args.next_compile_time_args_offset()>();

    constexpr uint32_t SID_B = 128;  // one [32] uint32 row
    constexpr uint32_t B_SID = 0, B_T = SID_B;
    constexpr uint32_t SKIP = 0xFFFFFFFFu;

    const uint32_t a_src = get_common_arg_val<uint32_t>(0), a_sid = get_common_arg_val<uint32_t>(1),
                   a_dst = get_common_arg_val<uint32_t>(2);
    const uint32_t u = get_arg_val<uint32_t>(0);

    Noc noc;
    const auto src_acc = TensorAccessor(src_args, a_src, TILE_B);
    const auto sid_acc = TensorAccessor(sid_args, a_sid, SID_B);
    const auto dst_acc = TensorAccessor(dst_args, a_dst, TILE_B);
    experimental::CB cb(cb_id);
    cb.reserve_back(1);
    const uint32_t base = cb.get_write_ptr();
    volatile tt_l1_ptr uint32_t* sidb = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + B_SID);

    noc.async_read(sid_acc, cb, SID_B, {.page_id = u, .offset_bytes = 0}, {.offset_bytes = B_SID});
    noc.async_read_barrier();
    invalidate_l1_cache();
    const uint32_t slot = sidb[0];
    if (slot == SKIP || slot >= K) {
        return;
    }
    for (uint32_t p = 0; p < NP; ++p) {
        noc.async_read(
            src_acc, cb, TILE_B, {.page_id = u * DP + p, .offset_bytes = 0}, {.offset_bytes = B_T + p * TILE_B});
    }
    noc.async_read_barrier();
    for (uint32_t p = 0; p < NP; ++p) {
        noc.async_write(
            cb, dst_acc, TILE_B, {.offset_bytes = B_T + p * TILE_B}, {.page_id = slot * NP + p, .offset_bytes = 0});
    }
    noc.async_write_barrier();
}
