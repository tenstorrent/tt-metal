// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Kernel: stream BFP2 ternary tiles and apply one fp16 scale per K-group.
//
// Structure from the DeepSeek compressed matmul (A0: 162 GB/s, 31.7% of peak on
// the Bonsai projection shapes), simplified to BFP2-only so that all per-tile
// metadata becomes compile-time constant.
//
// Per output column, K is walked group by group. Each group contributes
//
//     prod = (sum over the group's tiles of x * w) * scale_row
//     acc += prod
//
// Both steps involving the scale go through one scratch CB, cb_p, so the data
// flow is a single well-defined cycle per group:
//
//     1. MVMUL the group's tiles into Dst        -> pack to cb_p  (partial)
//     2. unpack cb_p, multiply by the scale row  -> Dst           (product)
//     3. pack Dst to cb_p
//     4. unpack cb_p, add into the accumulator (from Dst, reuse) -> Dst
//
// i.e. cb_p is pushed and popped exactly twice per group. Step 2 uses the bcast
// multiply whose left operand is the scale CB and whose right operand is the
// partial; step 4 is the only place a Dst operand is read back.
//
// Why the scale cannot be folded into the MVMUL: it depends on the output column
// as well as the group, so it is not shared by the activation, and the codes are
// +-1 so there is nothing on the weight side to fold it into. DESIGN.md records
// the alternatives and why this one is the cheapest correct arrangement.

#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
#include "api/dataflow/dataflow_api.h"
#endif

// The compute-side headers need DST_ACCUM_MODE and the SFPU register definitions,
// which only exist on a TRISC build, so they must not be seen by NCRISC/BRISC.
#if defined(COMPILE_FOR_TRISC)
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/cb_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#endif

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);    // activation [M,32] x Kt
    constexpr uint32_t cb_w = get_compile_time_arg_val(1);    // streamed BFP2 weights
    constexpr uint32_t cb_s = get_compile_time_arg_val(2);    // per-group fp16 scale rows
    constexpr uint32_t cb_p = get_compile_time_arg_val(3);    // scratch: partial/product
    constexpr uint32_t cb_out = get_compile_time_arg_val(4);
    constexpr uint32_t cols_per_core = get_compile_time_arg_val(5);
    constexpr uint32_t subblock_k = get_compile_time_arg_val(6);
    constexpr uint32_t num_subblocks = get_compile_time_arg_val(7);
    constexpr uint32_t tiles_per_group = get_compile_time_arg_val(8);
    constexpr uint32_t num_tiles_k = subblock_k * num_subblocks;
    constexpr uint32_t groups_per_subblock = subblock_k / tiles_per_group;
    constexpr uint32_t weight_subblock_bytes = subblock_k * 320;
    constexpr uint32_t scale_subblock_bytes = groups_per_subblock * cols_per_core * 2;

#if defined(COMPILE_FOR_NCRISC)
    // ==================================================================
    // Reader: this core's column range, subblock by subblock, double buffered.
    // ==================================================================
    const uint32_t weight_addr = get_arg_val<uint32_t>(0);
    const uint32_t scale_addr = get_arg_val<uint32_t>(1);
    const uint32_t col_offset = get_arg_val<uint32_t>(2);  // in tiles
    const uint32_t bank_id = get_arg_val<uint32_t>(3);

    const uint64_t w_noc = get_noc_addr_from_bank_id<true>(bank_id, weight_addr);
    const uint64_t s_noc = get_noc_addr_from_bank_id<true>(bank_id, scale_addr);

    uint32_t w_off = col_offset * num_tiles_k * 320;
    uint32_t s_off = col_offset * (num_tiles_k / tiles_per_group) * 2;

    const uint32_t total_subblocks = cols_per_core * num_subblocks;
    for (uint32_t i = 0; i < total_subblocks; ++i) {
        cb_reserve_back(cb_w, subblock_k);
        noc_async_read(w_noc + w_off, get_write_ptr(cb_w), weight_subblock_bytes);
        cb_push_back(cb_w, subblock_k);
        w_off += weight_subblock_bytes;

        cb_reserve_back(cb_s, 1);
        noc_async_read(s_noc + s_off, get_write_ptr(cb_s), scale_subblock_bytes);
        cb_push_back(cb_s, 1);
        s_off += scale_subblock_bytes;
    }
    noc_async_read_barrier();

#elif defined(COMPILE_FOR_BRISC)
    // Nothing to do: the compute side packs into cb_out itself.

#elif defined(COMPILE_FOR_TRISC)
    // ==================================================================
    // Compute.
    // ==================================================================
    reconfig_data_format_srca(cb_x);
    pack_reconfig_data_format(cb_out);
    matmul_init(cb_x, cb_w, cb_out);
    add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_p);

    constexpr uint32_t dst_scratch = 0;
    constexpr uint32_t dst_acc = 1;

    for (uint32_t col = 0; col < cols_per_core; ++col) {
        cb_wait_front(cb_x, num_tiles_k);
        bool have_acc = false;

        for (uint32_t sb = 0; sb < num_subblocks; ++sb) {
            cb_wait_front(cb_w, subblock_k);
            cb_wait_front(cb_s, 1);

            for (uint32_t g = 0; g < groups_per_subblock; ++g) {
                const uint32_t k0 = sb * subblock_k + g * tiles_per_group;

                // 1. partial = sum of this group's K tiles -> cb_p
                tile_regs_acquire();
                for (uint32_t t = 0; t < tiles_per_group; ++t) {
                    matmul_tiles(cb_x, cb_w, k0 + t, t, dst_scratch);
                }
                tile_regs_commit();
                tile_regs_wait();
                cb_reserve_back(cb_p, 1);
                pack_tile(dst_scratch, cb_p);
                cb_push_back(cb_p, 1);
                tile_regs_release();

                // 2. product = partial * scale_row -> Dst
                cb_wait_front(cb_p, 1);
                tile_regs_acquire();
                mul_tiles_bcast_rows(cb_s, cb_p, g, 0, dst_scratch);
                tile_regs_commit();
                tile_regs_wait();
                cb_pop_front(cb_p, 1);

                // 3. repack the product so step 4 can read it back
                cb_reserve_back(cb_p, 1);
                pack_tile(dst_scratch, cb_p);
                tile_regs_release();
                cb_push_back(cb_p, 1);

                // 4. acc += product
                cb_wait_front(cb_p, 1);
                tile_regs_acquire();
                if (!have_acc) {
                    copy_tile(cb_p, 0, dst_acc);
                    have_acc = true;
                } else {
                    add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_p, 0, dst_acc);
                }
                tile_regs_commit();
                tile_regs_wait();
                tile_regs_release();
                cb_pop_front(cb_p, 1);
            }

            cb_pop_front(cb_w, subblock_k);
            cb_pop_front(cb_s, 1);
        }

        // Emit the finished column. cb_out has one slot, so the consumer side
        // (the host readback, via the generic op's completion) drains it; the
        // kernel only needs to push.
        cb_reserve_back(cb_out, 1);
        tile_regs_acquire();
        pack_tile(dst_acc, cb_out);
        tile_regs_commit();
        tile_regs_wait();
        cb_push_back(cb_out, 1);
        tile_regs_release();
        // The activation is reused by the next column, so it is re-pushed here
        // rather than popped.
    }
    cb_pop_front(cb_x, num_tiles_k);
#endif
}
