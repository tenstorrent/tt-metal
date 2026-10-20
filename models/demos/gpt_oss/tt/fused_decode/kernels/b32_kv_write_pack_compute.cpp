// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Batch-32 DFlash verify cache write, compute (b32_kv_write_pack.cpp): per kv (K, then V) and cache tile q (group g =
// q / head_tiles): out = old * mask_g + new, packed to BFP8_b. old: the cache tile (BFP8_b, exact in DEST); mask_g:
// 1.0 / 0 rows (BF16); new: the new rows, zero elsewhere (BF16). Products by 1.0 / 0 and sums with 0 are exact, so the
// packer sees the kept rows' values and the new rows' BF16 values. The group count arrives in the mailbox.
//
// compile-time args: [cb_old, cb_new, cb_mask, cb_out, head_tiles]  (K's cb_old / cb_new / cb_out; V's the next index)

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"

void kernel_main() {
    constexpr uint32_t cb_old0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_new0 = get_compile_time_arg_val(1);
    constexpr uint32_t cb_mask = get_compile_time_arg_val(2);
    constexpr uint32_t cb_out0 = get_compile_time_arg_val(3);
    constexpr uint32_t head_tiles = get_compile_time_arg_val(4);

    uint32_t groups = 0;
    UNPACK(groups = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId));
    MATH(groups = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId));
    PACK(groups = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId));
    if (groups == 0) {
        return;
    }
    const uint32_t n = groups * head_tiles;
    compute_kernel_hw_startup(cb_old0, cb_mask, cb_out0);
    cb_wait_front(cb_mask, groups);
    for (uint32_t kv = 0; kv < 2; ++kv) {
        const uint32_t cb_old = cb_old0 + kv, cb_new = cb_new0 + kv, cb_out = cb_out0 + kv;
        cb_wait_front(cb_old, n);
        cb_wait_front(cb_new, n);
        for (uint32_t q = 0; q < n; ++q) {
            tile_regs_acquire();
            // (the inits do not switch the unpacker formats: state_configure is a no-op outside the sentinel build)
            reconfig_data_format(cb_old, cb_mask);
            mul_init(cb_old, cb_mask);
            mul_tiles(cb_old, cb_mask, q, q / head_tiles, 0);
            reconfig_data_format(cb_new, cb_new);  // (DEST -> SrcA, cb_new -> SrcB)
            add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_new);
            add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_new, q, 0);
            tile_regs_commit();
            // one tile per push (several per push corrupted one landing on the CB's last page, gmoe_gu_compute.cpp)
            cb_reserve_back(cb_out, 1);
            tile_regs_wait();
            pack_reconfig_data_format(cb_out);
            pack_tile(0, cb_out);
            tile_regs_release();
            cb_push_back(cb_out, 1);
        }
        cb_pop_front(cb_old, n);
        cb_pop_front(cb_new, n);
    }
    cb_pop_front(cb_mask, groups);
}
