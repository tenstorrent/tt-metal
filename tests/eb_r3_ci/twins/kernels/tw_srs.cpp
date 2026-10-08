// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// Round 3 eltwise binary twin of the strided reduce-scatter's reduction compute with tt_dit's fused addcmul
// (ttnn/cpp/ttnn/operations/experimental/ccl/strided_reduce_scatter_async/device/kernels/minimal_ring_reduction.cpp): its startup (:76-77) and, for every ring step i = 1 .. ring_size - 1 of a chunk, the per-step body
// (:128-246) verbatim on tile_granularity tiles, the final step fusing the addcmul; run TWIN_ITERS times.
// Compile args: the op's indices 0-4 (input_cb, intermediate_cb, output_cb, tile_granularity, ring_size), 16-18 (addcmul
// temp, a, b CBs) at 5-7, iterations at 8; runtime arg 0 the scalar bits.
#include <cstdint>
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t input_cb = get_compile_time_arg_val(0);
    constexpr uint32_t intermediate_cb = get_compile_time_arg_val(1);
    constexpr uint32_t output_cb = get_compile_time_arg_val(2);
    constexpr uint32_t tile_granularity = get_compile_time_arg_val(3);
    constexpr uint32_t ring_size = get_compile_time_arg_val(4);
#ifdef FUSE_RS_ADDCMUL
    constexpr uint32_t addcmul_temp_cb = get_compile_time_arg_val(5);
    constexpr uint32_t addcmul_a_cb = get_compile_time_arg_val(6);
    constexpr uint32_t addcmul_b_cb = get_compile_time_arg_val(7);
#endif
    constexpr uint32_t twin_iters = get_compile_time_arg_val(8);
#ifdef FUSE_RS_ADDCMUL
    const uint32_t fused_ternary_scalar_uint = get_arg_val<uint32_t>(0);
#endif

    CircularBuffer cb_in(input_cb);
    CircularBuffer cb_intermediate(intermediate_cb);
    CircularBuffer cb_out(output_cb);
#ifdef FUSE_RS_ADDCMUL
    CircularBuffer cb_addcmul_temp(addcmul_temp_cb);
    CircularBuffer cb_addcmul_a(addcmul_a_cb);
    CircularBuffer cb_addcmul_b(addcmul_b_cb);
#endif

    compute_kernel_hw_startup(input_cb, intermediate_cb, output_cb);
    add_init(input_cb, intermediate_cb, false);

    for (uint32_t it = 0; it < twin_iters; ++it) {
        for (uint32_t i = 1; i < ring_size; i++) {
#ifdef FUSE_RS_ADDCMUL
            const bool is_final_ring_step = (i == ring_size - 1);
#endif
            const uint32_t tiles_to_read_in_this_step = tile_granularity;
#ifdef FUSE_RS_ADDCMUL
            if (is_final_ring_step) {
                // -------------------------------------------------------
                // Fused addcmul at the final ring step:
                //   output = a + scalar * (input + intermediate) * b
                //
                // Step 1: acc = input + intermediate -> addcmul_temp_cb
                // -------------------------------------------------------
                cb_in.wait_front(tile_granularity);
                cb_intermediate.wait_front(tile_granularity);

                add_init(input_cb, intermediate_cb, false);
                reconfig_data_format(input_cb, intermediate_cb);

                tile_regs_acquire();
                for (uint32_t tile_id = 0; tile_id < tiles_to_read_in_this_step; tile_id++) {
                    add_tiles(input_cb, intermediate_cb, tile_id, tile_id, tile_id);
                }
                tile_regs_commit();

                cb_in.pop_front(tile_granularity);
                cb_intermediate.pop_front(tile_granularity);

                cb_addcmul_temp.reserve_back(tile_granularity);
                tile_regs_wait();
                pack_reconfig_data_format(addcmul_temp_cb);
                for (uint32_t tile_id = 0; tile_id < tiles_to_read_in_this_step; tile_id++) {
                    pack_tile(tile_id, addcmul_temp_cb);
                }
                tile_regs_release();
                cb_addcmul_temp.push_back(tile_granularity);

                // -------------------------------------------------------
                // Step 2: scalar * acc * b -> pack back to addcmul_temp_cb
                // When ADDCMUL_B_BROADCAST: b has 1 row per tile, broadcast across acc's rows.
                // Otherwise: b has full rows (per-token), element-wise multiply.
                // -------------------------------------------------------
                cb_addcmul_temp.wait_front(tile_granularity);
                cb_addcmul_b.wait_front(tile_granularity);

#ifdef ADDCMUL_B_BROADCAST
                mul_bcast_rows_init(addcmul_temp_cb, addcmul_b_cb);
#else
                mul_init(addcmul_temp_cb, addcmul_b_cb, 0, __builtin_LINE());
#endif
                reconfig_data_format(addcmul_temp_cb, addcmul_b_cb);
                pack_reconfig_data_format(addcmul_temp_cb);
                binop_with_scalar_tile_init();

                for (uint32_t tile_id = 0; tile_id < tiles_to_read_in_this_step; tile_id++) {
                    tile_regs_acquire();
#ifdef ADDCMUL_B_BROADCAST
                    mul_tiles_bcast<BroadcastType::ROW>(
                        addcmul_temp_cb, addcmul_b_cb, tile_id, tile_id, 0);
#else
                    mul_tiles(addcmul_temp_cb, addcmul_b_cb, tile_id, tile_id, 0);
#endif
                    mul_unary_tile(0, fused_ternary_scalar_uint);
                    tile_regs_commit();
                    tile_regs_wait();
                    pack_tile(0, addcmul_temp_cb);
                    tile_regs_release();
                }
                cb_addcmul_b.pop_front(tile_granularity);
                cb_addcmul_temp.pop_front(tile_granularity);
                cb_addcmul_temp.reserve_back(tile_granularity);
                cb_addcmul_temp.push_back(tile_granularity);

                // -------------------------------------------------------
                // Step 3: a + scalar*acc*b -> output_cb
                // -------------------------------------------------------
                cb_addcmul_temp.wait_front(tile_granularity);
                cb_addcmul_a.wait_front(tile_granularity);

                add_init(addcmul_temp_cb, addcmul_a_cb, false);
                reconfig_data_format(addcmul_temp_cb, addcmul_a_cb);

                tile_regs_acquire();
                for (uint32_t tile_id = 0; tile_id < tiles_to_read_in_this_step; tile_id++) {
                    add_tiles(addcmul_temp_cb, addcmul_a_cb, tile_id, tile_id, tile_id);
                }
                tile_regs_commit();

                cb_addcmul_temp.pop_front(tile_granularity);
                cb_addcmul_a.pop_front(tile_granularity);

                cb_out.reserve_back(tile_granularity);
                tile_regs_wait();
                pack_reconfig_data_format(output_cb);
                for (uint32_t tile_id = 0; tile_id < tiles_to_read_in_this_step; tile_id++) {
                    pack_tile(tile_id, output_cb);
                }
                tile_regs_release();
                cb_out.push_back(tile_granularity);
            } else {
#endif
                // Normal ring accumulation step: acc = input + intermediate
                cb_in.wait_front(tile_granularity);
                cb_intermediate.wait_front(tile_granularity);

                tile_regs_acquire();
                for (uint32_t tile_id = 0; tile_id < tiles_to_read_in_this_step; tile_id++) {
                    add_tiles(input_cb, intermediate_cb, tile_id, tile_id, tile_id);
                }
                tile_regs_commit();

                cb_in.pop_front(tile_granularity);
                cb_intermediate.pop_front(tile_granularity);

                cb_out.reserve_back(tile_granularity);
                tile_regs_wait();
                for (uint32_t tile_id = 0; tile_id < tiles_to_read_in_this_step; tile_id++) {
                    pack_tile(tile_id, output_cb);
                }
                tile_regs_release();
                cb_out.push_back(tile_granularity);
#ifdef FUSE_RS_ADDCMUL
            }  // end else (non-final ring step)
#endif
        }
    }
}
