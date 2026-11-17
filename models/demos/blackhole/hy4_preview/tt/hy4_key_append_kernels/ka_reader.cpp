// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Hy4 B32 key append (HY4_B32_KAPPEND=1), reader. ONE core appends the S (<= 4) verify key rows of a user into the bfp8
// key pool, one row after the other, each as a read-modify-write of the 4-tile (128-wide) tile row that holds the
// position (the stock paged_update_cache does: untilize cache tiles -> patch the row -> tilize back), so the result is
// bit-identical to S stock calls. Per row j: wait for the writer's "done" of row j-1 (same tile may be read again) ->
// read idx_j (uint32 per device, 0xFFFFFFFF = skip) -> page table entry -> read the 4 cache tiles (cb_cache) -> compute
// untilizes them (cb_ut) -> patch the 256 B row j of the input (untilized once into cb_kin_rm) into the block -> hand
// the block back as cb_ut2 (aliases cb_ut) -> compute tilizes (cb_out) -> writer. CT: 0 cb_kin, 1 cb_cache, 2 cb_ut, 3
// cb_ut2, 4 cb_kin_rm, 5 cb_done, 6 cb_meta, 7 cb_misc, 8 S, 9 TB_IN (bf16 tile bytes), 10 TB_C (bfp8 tile bytes), then
// TensorAccessorArgs(k), (pool), (pt), (idx). Common: k addr, pool addr, pt addr, idx addr x S.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t cb_kin = get_compile_time_arg_val(0);
    constexpr uint32_t cb_cache = get_compile_time_arg_val(1);
    constexpr uint32_t cb_ut = get_compile_time_arg_val(2);
    constexpr uint32_t cb_ut2 = get_compile_time_arg_val(3);
    constexpr uint32_t cb_kin_rm = get_compile_time_arg_val(4);
    constexpr uint32_t cb_done = get_compile_time_arg_val(5);
    constexpr uint32_t cb_meta = get_compile_time_arg_val(6);
    constexpr uint32_t cb_misc = get_compile_time_arg_val(7);
    constexpr uint32_t S = get_compile_time_arg_val(8);
    constexpr uint32_t TB_IN = get_compile_time_arg_val(9);
    constexpr uint32_t TB_C = get_compile_time_arg_val(10);
    constexpr auto k_args = TensorAccessorArgs<11>();
    constexpr auto pool_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    constexpr auto pt_args = TensorAccessorArgs<pool_args.next_compile_time_args_offset()>();
    constexpr auto idx_args = TensorAccessorArgs<pt_args.next_compile_time_args_offset()>();
    constexpr uint32_t WT = 4, WBYTES = 256, BLOCK = 64;
    const auto k_acc = TensorAccessor(k_args, get_common_arg_val<uint32_t>(0), TB_IN);
    const auto pool_acc = TensorAccessor(pool_args, get_common_arg_val<uint32_t>(1), TB_C);
    const auto pt_acc = TensorAccessor(pt_args, get_common_arg_val<uint32_t>(2), 64);
    Noc noc;
    CircularBuffer kin(cb_kin), cache(cb_cache), ut(cb_ut), ut2(cb_ut2), kin_rm(cb_kin_rm), done(cb_done),
        meta(cb_meta), misc(cb_misc);
    kin.reserve_back(WT);
    for (uint32_t t = 0; t < WT; ++t) {
        noc.async_read(k_acc, kin, TB_IN, {.page_id = t, .offset_bytes = 0}, {.offset_bytes = t * TB_IN});
    }
    noc.async_read_barrier();
    kin.push_back(WT);
    misc.reserve_back(1);
    const uint32_t mb = misc.get_write_ptr();
    volatile tt_l1_ptr uint32_t* sc = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mb);
    kin_rm.wait_front(WT);
    const uint32_t in_rm = kin_rm.get_read_ptr();
    for (uint32_t j = 0; j < S; ++j) {
        if (j > 0) {
            done.wait_front(1);
            done.pop_front(1);
        }
        const auto idx_acc = TensorAccessor(idx_args, get_common_arg_val<uint32_t>(3 + j), 64);
        noc.async_read(idx_acc, misc, 64, {.page_id = 0, .offset_bytes = 0}, {.offset_bytes = 0});
        noc.async_read_barrier();
        invalidate_l1_cache();
        const uint32_t v = sc[0];
        const bool skip = (v == 0xFFFFFFFFu);
        uint32_t tile_base = 0, off = 0;
        if (!skip) {
            const uint32_t vb = v / BLOCK;
            noc.async_read(pt_acc, misc, 64, {.page_id = 0, .offset_bytes = (vb * 4) & ~63u}, {.offset_bytes = 64});
            noc.async_read_barrier();
            invalidate_l1_cache();
            const uint32_t phys = sc[16 + (vb & 15)];
            tile_base = phys * (2 * WT) + ((v % BLOCK) / 32) * WT;
            off = (v % 32) * WBYTES;
        }
        cache.reserve_back(WT);
        for (uint32_t t = 0; t < WT; ++t) {
            noc.async_read(
                pool_acc, cache, TB_C, {.page_id = tile_base + t, .offset_bytes = 0}, {.offset_bytes = t * TB_C});
        }
        noc.async_read_barrier();
        cache.push_back(WT);
        meta.reserve_back(1);
        volatile tt_l1_ptr uint32_t* mp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(meta.get_write_ptr());
        mp[0] = skip ? 1u : 0u;
        mp[1] = tile_base;
        meta.push_back(1);
        ut.wait_front(WT);
        if (!skip) {
            volatile tt_l1_ptr uint32_t* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ut.get_read_ptr() + off);
            volatile tt_l1_ptr uint32_t* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in_rm + j * WBYTES);
            for (uint32_t w = 0; w < WBYTES / 4; ++w) {
                dst[w] = src[w];
            }
        }
        ut2.reserve_back(WT);
        ut2.push_back(WT);
        ut.pop_front(WT);
    }
}
