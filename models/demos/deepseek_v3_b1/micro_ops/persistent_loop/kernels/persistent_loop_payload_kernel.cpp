// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// PersistentLoop with a real Tensix payload: every iteration copies num_tiles tiles from the input
// shard to the output shard through unpack, math and pack, then the pack thread records the number of
// completed iterations in the status shard. The host terminates the loop part-way through its run and
// checks that every completed iteration left a correct copy behind, across repeated launches.

#include "../../../unified_kernels/kernel_op_api.hpp"
#include "../../../unified_kernels/kernel_utils.hpp"
#include "../../../unified_kernels/persistent_loop.hpp"

#if defined(COMPILE_FOR_TRISC)
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/experimental/deepseek_compute_kernel_hw_startup.h"
#endif

void kernel_main() {
    constexpr uint32_t persistent_mode = get_named_compile_time_arg_val("persistent_mode");
    constexpr uint32_t termination_semaphore_addr = get_named_compile_time_arg_val("termination_semaphore_addr");
    constexpr uint32_t max_iterations = get_named_compile_time_arg_val("max_iterations");
    constexpr uint32_t input_cb = get_named_compile_time_arg_val("input_cb");
    constexpr uint32_t output_cb = get_named_compile_time_arg_val("output_cb");
    constexpr uint32_t status_addr = get_named_compile_time_arg_val("status_addr");
    constexpr uint32_t num_tiles = get_named_compile_time_arg_val("num_tiles");

    deepseek_b1_ops::PersistentLoop<persistent_mode == 1> loop(termination_semaphore_addr, max_iterations);

#if defined(COMPILE_FOR_NCRISC)
    // The input shard is tensor-backed: mark it available once; compute never pops it.
    unified_kernels::setup_sharded_buffer(input_cb, num_tiles);
    while (loop.next()) {
    }
#elif defined(COMPILE_FOR_BRISC)
    while (loop.next()) {
    }
#elif defined(COMPILE_FOR_TRISC)
    deepseek_compute_kernel_init();
    copy_init(input_cb);
    pack_reconfig_data_format<true>(output_cb);

    // Status shard words: [0] completed iterations (pack), [1] last decided iteration (unpack),
    // [2] stop flag (unpack). Every thread calls loop.next() each iteration, as the fused kernels do,
    // but the three threads must take the same termination decision for the same iteration (one thread
    // exiting while another has started the next iteration waits on it forever), so only the unpack
    // thread's answer counts: it publishes the verdict and math and pack follow it.
    volatile tt_l1_ptr uint32_t* status = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(status_addr);
    uint32_t iter = 0;
    uint32_t completed = 0;
    while (true) {
        ++iter;
        bool go = loop.next();
        UNPACK(({
            status[2] = go ? 0u : 1u;
            status[1] = iter;
        }));
        MATH(({
            do {
                invalidate_l1_cache();
            } while (status[1] < iter);
            go = status[2] == 0u;
        }));
        PACK(({
            do {
                invalidate_l1_cache();
            } while (status[1] < iter);
            go = status[2] == 0u;
        }));
        if (!go) {
            break;
        }

        cb_wait_front(input_cb, num_tiles);
        cb_reserve_back(output_cb, num_tiles);
        tile_regs_acquire();
        for (uint32_t i = 0; i < num_tiles; i++) {
            copy_tile(input_cb, i, i);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < num_tiles; i++) {
            pack_tile(i, output_cb);
        }
        tile_regs_release();
        cb_push_back(output_cb, num_tiles);
        // The output shard is exactly num_tiles pages: wait for this iteration's pack to land, then pop so
        // the next iteration writes the shard again from the start.
        cb_wait_front(output_cb, num_tiles);
        cb_pop_front(output_cb, num_tiles);
        ++completed;
        PACK((status[0] = completed));
    }
#endif
}
