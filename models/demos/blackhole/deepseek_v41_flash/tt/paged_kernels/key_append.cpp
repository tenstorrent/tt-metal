// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// DeepSeek-V4.1-Flash SINGLE-TRACE prefill (DSV41_PF_SINGLE=1, tt/prefill_single.py): append the chunk's index keys to
// the key slab of the indexer at a DEVICE-TENSOR driven row-block offset (a captured trace cannot change a
// host-computed ``slice_write`` offset).
//
//   src   bf16 TILE [U, 1, NRB * 32, 128]   the chunk's keys of this device (tiles of one user are contiguous: NP = NRB
//   * 4 pages) kofs  uint32 ROW_MAJOR [U, 32]          row u, element 0: first 32-row block of the destination of user
//   u (0 = the scratch region: the user is not part of this replay) dst   bf16 TILE [U, 1, T, 128]          the key
//   slab (written IN PLACE): page of (user u, row block rb, tile column t) = u * DP + rb * 4 + t
//
// Copies the NP pages of user u to dst pages u * DP + kofs[u] * 4 + [0, NP). One core.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

void kernel_main() {
    constexpr uint32_t U = get_compile_time_arg_val(0);
    constexpr uint32_t NP = get_compile_time_arg_val(1);
    constexpr uint32_t DP = get_compile_time_arg_val(2);
    constexpr uint32_t TILE_B = get_compile_time_arg_val(3);
    constexpr uint32_t cb_id = get_compile_time_arg_val(4);
    constexpr auto src_args = TensorAccessorArgs<5>();
    constexpr auto ko_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();
    constexpr auto dst_args = TensorAccessorArgs<ko_args.next_compile_time_args_offset()>();

    constexpr uint32_t KO_B = 128;  // one [32] uint32 row
    constexpr uint32_t B_KO = 0, B_T = KO_B;

    const uint32_t a_src = get_common_arg_val<uint32_t>(0), a_ko = get_common_arg_val<uint32_t>(1),
                   a_dst = get_common_arg_val<uint32_t>(2);

    Noc noc;
    const auto src_acc = TensorAccessor(src_args, a_src, TILE_B);
    const auto ko_acc = TensorAccessor(ko_args, a_ko, KO_B);
    const auto dst_acc = TensorAccessor(dst_args, a_dst, TILE_B);
    experimental::CB cb(cb_id);
    cb.reserve_back(1);
    const uint32_t base = cb.get_write_ptr();
    volatile tt_l1_ptr uint32_t* kob = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + B_KO);

    for (uint32_t u = 0; u < U; ++u) {
        noc.async_read(ko_acc, cb, KO_B, {.page_id = u, .offset_bytes = 0}, {.offset_bytes = B_KO});
        noc.async_read_barrier();
        invalidate_l1_cache();  // (the L1 line of the previous user's row is reused)
        const uint32_t rb0 = kob[0];
        for (uint32_t p = 0; p < NP; ++p) {
            noc.async_read(src_acc, cb, TILE_B, {.page_id = u * NP + p, .offset_bytes = 0}, {.offset_bytes = B_T});
            noc.async_read_barrier();
            noc.async_write(
                cb, dst_acc, TILE_B, {.offset_bytes = B_T}, {.page_id = u * DP + rb0 * 4 + p, .offset_bytes = 0});
            noc.async_write_barrier();
        }
    }
}
