// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// DeepSeek-V4.1-Flash cross-row copy (DSV41_XROW_OPS, tt/xrow_ops.py): MULTI-SLOT block copy, ONE launch for N blocks.
//   for i in [0, N): if tab[i] != SKIP and tab[NT + i] != SKIP:  dst block tab[NT + i]  <-  src block tab[i]
// A block = NP consecutive tile pages (page of (block k, page p) = k * NP + p). src / dst bf16 TILE tensors ([K, 1, R,
// 128]). tab: uint32 ROW_MAJOR, per device one row of 2 * NT entries (src ids then dst ids, 0xFFFFFFFF = skip), sharded
// over the mesh rows, so one launch on the whole mesh touches only the rows that have valid entries. Entry i is handled
// by core i % NCORES.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

void kernel_main() {
    constexpr uint32_t NP = get_compile_time_arg_val(0);
    constexpr uint32_t TILE_B = get_compile_time_arg_val(1);
    constexpr uint32_t cb_id = get_compile_time_arg_val(2);
    constexpr uint32_t NT = get_compile_time_arg_val(3);
    constexpr uint32_t N = get_compile_time_arg_val(4);
    constexpr uint32_t KS = get_compile_time_arg_val(5);
    constexpr uint32_t KD = get_compile_time_arg_val(6);
    constexpr uint32_t NCORES = get_compile_time_arg_val(7);
    constexpr auto tab_args = TensorAccessorArgs<8>();
    constexpr auto src_args = TensorAccessorArgs<tab_args.next_compile_time_args_offset()>();
    constexpr auto dst_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();
    constexpr uint32_t TAB_B = 2 * NT * 4;
    constexpr uint32_t SKIP = 0xFFFFFFFFu;

    const uint32_t a_tab = get_common_arg_val<uint32_t>(0), a_src = get_common_arg_val<uint32_t>(1),
                   a_dst = get_common_arg_val<uint32_t>(2);
    const uint32_t core = get_arg_val<uint32_t>(0);

    Noc noc;
    const auto tab_acc = TensorAccessor(tab_args, a_tab, TAB_B);
    const auto src_acc = TensorAccessor(src_args, a_src, TILE_B);
    const auto dst_acc = TensorAccessor(dst_args, a_dst, TILE_B);
    experimental::CB cb(cb_id);
    cb.reserve_back(1);
    const uint32_t base = cb.get_write_ptr();
    volatile tt_l1_ptr uint32_t* tab = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base);

    noc.async_read(tab_acc, cb, TAB_B, {.page_id = 0, .offset_bytes = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    invalidate_l1_cache();
    for (uint32_t i = core; i < N; i += NCORES) {
        const uint32_t s = tab[i], d = tab[NT + i];
        if (s == SKIP || d == SKIP || s >= KS || d >= KD) {
            continue;
        }
        for (uint32_t p = 0; p < NP; ++p) {
            noc.async_read(
                src_acc, cb, TILE_B, {.page_id = s * NP + p, .offset_bytes = 0}, {.offset_bytes = TAB_B + p * TILE_B});
        }
        noc.async_read_barrier();
        for (uint32_t p = 0; p < NP; ++p) {
            noc.async_write(
                cb, dst_acc, TILE_B, {.offset_bytes = TAB_B + p * TILE_B}, {.page_id = d * NP + p, .offset_bytes = 0});
        }
        noc.async_write_barrier();
    }
}
