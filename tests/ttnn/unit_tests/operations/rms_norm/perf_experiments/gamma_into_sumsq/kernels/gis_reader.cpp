// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ISOLATED BENCH — "where does apply_gamma belong?", reader (NoC0) half.
//
// IDENTICAL in every variant.  Reconstructs the shipped rms_norm reader for the
// TILE + BLOCK-sharded geometry and nothing else:
//   * publish the resident input shard once (load_block is bookkeeping there);
//   * `load_gamma_once` — the row-0-only two-face-segment DRAM read of this core's
//     S gamma tiles, ISSUED before the block loop with its barrier DEFERRED to the
//     top of block 0, exactly as the op does (the deferral is a measured property:
//     gamma is the only DRAM tensor on this path);
//   * the reduce-scatter combine's reader half — owners wait the gather incast,
//     funnel their finalized rows to the root, root multicasts them back.
//
// The gamma barrier placement is what makes the bake-off fair: `gamma_first` and
// `fused` need gamma EARLIER (before the combine instead of after it), so if the
// deferral only worked because the combine hid the DRAM round trip, this kernel is
// where that would show up — as a `cp_gamma_wait` on the compute side.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/tensor/noc_traits.h"
#include "hostdevcommon/common_values.hpp"

#include "ttnn/cpp/ttnn/kernel_lib/mcast_pipe.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

using namespace dataflow_kernel_lib;

constexpr uint32_t cb_input_tiles = 0;
constexpr uint32_t cb_gamma_tiles = 1;
constexpr uint32_t cb_slice_stat = 3;
constexpr uint32_t cb_gathered_partials = 4;
constexpr uint32_t cb_rms_bcast = 5;
constexpr uint32_t cb_rms_recip = 6;
constexpr uint32_t cb_scaler = 7;

constexpr uint32_t FACE_DIM_L = 16;

void kernel_main() {
    constexpr auto mc = McastArgs</*CT=*/0, /*RT=*/0>();
    constexpr uint32_t CT = mc.next_compile_time_args_offset();

    constexpr uint32_t SLICE_HIDDEN_TILES = get_compile_time_arg_val(CT + 0);  // S
    constexpr uint32_t BLOCK_ROWS = get_compile_time_arg_val(CT + 1);          // B
    constexpr uint32_t NUM_HIDDEN_SLICES = get_compile_time_arg_val(CT + 2);   // s
    constexpr uint32_t HAS_GAMMA = get_compile_time_arg_val(CT + 3);
    constexpr uint32_t GAMMA_TILE_BYTES = get_compile_time_arg_val(CT + 4);
    constexpr uint32_t GAMMA_ELEM_BYTES = get_compile_time_arg_val(CT + 5);
    constexpr uint32_t GATHER_SEM_ID = get_compile_time_arg_val(CT + 6);
    constexpr uint32_t STAT_READY_SEM_ID = get_compile_time_arg_val(CT + 7);
    constexpr uint32_t STAT_TILE_BYTES = get_compile_time_arg_val(CT + 8);
    constexpr uint32_t IN_WAIT_TILES = get_compile_time_arg_val(CT + 9);
    constexpr uint32_t NUM_OWNERS = get_compile_time_arg_val(CT + 10);
    constexpr uint32_t OWN_ROWS = get_compile_time_arg_val(CT + 11);
    constexpr auto gamma_args = TensorAccessorArgs<CT + 12>();

    constexpr uint32_t BLOCK_TILES = BLOCK_ROWS * SLICE_HIDDEN_TILES;

    constexpr uint32_t RT = mc.next_runtime_args_offset();
    const uint32_t num_blocks = get_arg_val<uint32_t>(RT + 0);
    const uint32_t is_root = get_arg_val<uint32_t>(RT + 1);
    const uint32_t is_owner = get_arg_val<uint32_t>(RT + 2);
    const uint32_t my_first_row = get_arg_val<uint32_t>(RT + 3);
    const uint32_t gamma_slice_base = get_arg_val<uint32_t>(RT + 4);
    const uint32_t gamma_addr = get_arg_val<uint32_t>(RT + 5);
    const uint32_t root_noc_x = get_arg_val<uint32_t>(RT + 6);
    const uint32_t root_noc_y = get_arg_val<uint32_t>(RT + 7);

    Noc noc;
    CircularBuffer cb_gamma_obj(cb_gamma_tiles);

    {
        MaybeDeviceZoneScope("rd_stat_consts");
        calculate_and_prepare_reduce_scaler<cb_scaler, PoolType::SUM, ReduceDim::REDUCE_ROW>();
    }

    // Publish the whole resident input shard once — keeping cb_input_tiles FULL at
    // every block boundary is what makes get_write_ptr() == get_read_ptr(), which
    // the in-place rewrite of x depends on.
    {
        MaybeDeviceZoneScope("rd_boot_zero_publish");
        cb_reserve_back(cb_input_tiles, IN_WAIT_TILES);
        cb_push_back(cb_input_tiles, IN_WAIT_TILES);
    }

    // ---- load_gamma_once: ISSUE only; the barrier is deferred into the loop ----
    bool gamma_pending = false;
    if constexpr (HAS_GAMMA) {
        MaybeDeviceZoneScope("rd_gamma_issue");
        gamma_pending = true;
        const auto gamma_accessor = TensorAccessor(gamma_args, gamma_addr);
        cb_reserve_back(cb_gamma_tiles, SLICE_HIDDEN_TILES);
        const uint32_t gamma_l1 = get_write_ptr(cb_gamma_tiles);
        noc.async_write_zeros(cb_gamma_obj, SLICE_HIDDEN_TILES * GAMMA_TILE_BYTES);
        noc.write_zeros_l1_barrier();
        // gamma is [W] -> a TILE-layout (1,..,1,W) tensor: real data in ROW 0 only,
        // straddling two faces (byte 0 and byte 16*16*elem).
        constexpr uint32_t FACE_ROW_BYTES = FACE_DIM_L * GAMMA_ELEM_BYTES;
        constexpr uint32_t FACE1_OFFSET = FACE_DIM_L * FACE_DIM_L * GAMMA_ELEM_BYTES;
        for (uint32_t t = 0; t < SLICE_HIDDEN_TILES; ++t) {
            const uint32_t dst = gamma_l1 + t * GAMMA_TILE_BYTES;
            noc_async_read(gamma_accessor.get_noc_addr(gamma_slice_base + t, 0), dst, FACE_ROW_BYTES);
            noc_async_read(
                gamma_accessor.get_noc_addr(gamma_slice_base + t, FACE1_OFFSET), dst + FACE1_OFFSET, FACE_ROW_BYTES);
        }
    }

    auto sender_pipe = mc.sender(noc);
    auto receiver_pipe = mc.receiver(noc);
    Semaphore<> gather_progress(GATHER_SEM_ID);
    Semaphore<> stat_ready(STAT_READY_SEM_ID);
    uint32_t rms_bcast_base = 0;
    if constexpr (NUM_HIDDEN_SLICES > 1 && NUM_OWNERS > 1) {
        rms_bcast_base = get_read_ptr(cb_rms_bcast);
    }

    for (uint32_t block = 0; block < num_blocks; ++block) {
        {
            MaybeDeviceZoneScope("rd_load_total");
            if (block > 0) {
                cb_reserve_back(cb_input_tiles, BLOCK_TILES);
                cb_push_back(cb_input_tiles, BLOCK_TILES);
            }
        }

        if (gamma_pending) {
            MaybeDeviceZoneScope("rd_gamma_barrier");
            noc_async_read_barrier();
            cb_push_back(cb_gamma_tiles, SLICE_HIDDEN_TILES);
            gamma_pending = false;
        }

        if constexpr (NUM_HIDDEN_SLICES > 1) {
            if (is_owner) {
                cb_reserve_back(cb_gathered_partials, NUM_HIDDEN_SLICES * OWN_ROWS);
                {
                    MaybeDeviceZoneScope("rd_gather_wait");
                    gather_progress.wait_min((block + 1) * NUM_HIDDEN_SLICES);
                }
                cb_push_back(cb_gathered_partials, NUM_HIDDEN_SLICES * OWN_ROWS);
            }

            if constexpr (NUM_OWNERS > 1) {
                if (is_owner) {
                    {
                        MaybeDeviceZoneScope("rd_stat_funnel_wait");
                        cb_wait_front(cb_slice_stat, OWN_ROWS);
                    }
                    {
                        MaybeDeviceZoneScope("rd_stat_funnel");
                        noc_async_write(
                            get_read_ptr(cb_slice_stat),
                            get_noc_addr(root_noc_x, root_noc_y, rms_bcast_base + my_first_row * STAT_TILE_BYTES),
                            OWN_ROWS * STAT_TILE_BYTES);
                        noc_async_write_barrier();
                        stat_ready.up(noc, root_noc_x, root_noc_y, 1);
                    }
                    cb_pop_front(cb_slice_stat, OWN_ROWS);
                }
            }

            if (is_root) {
                uint32_t bcast_src;
                if constexpr (NUM_OWNERS > 1) {
                    {
                        MaybeDeviceZoneScope("rd_bcast_wait_stat");
                        stat_ready.wait_min((block + 1) * NUM_OWNERS);
                    }
                    bcast_src = rms_bcast_base;
                } else {
                    {
                        MaybeDeviceZoneScope("rd_bcast_wait_stat");
                        cb_wait_front(cb_rms_bcast, BLOCK_ROWS);
                    }
                    bcast_src = get_read_ptr(cb_rms_bcast);
                }
                cb_reserve_back(cb_rms_recip, BLOCK_ROWS);
                {
                    MaybeDeviceZoneScope("rd_bcast_send");
                    sender_pipe.send(bcast_src, get_write_ptr(cb_rms_recip), BLOCK_ROWS * STAT_TILE_BYTES);
                }
                cb_push_back(cb_rms_recip, BLOCK_ROWS);
                if constexpr (NUM_OWNERS == 1) {
                    cb_pop_front(cb_rms_bcast, BLOCK_ROWS);
                }
            } else {
                cb_reserve_back(cb_rms_recip, BLOCK_ROWS);
                {
                    MaybeDeviceZoneScope("rd_bcast_recv");
                    receiver_pipe.receive();
                }
                cb_push_back(cb_rms_recip, BLOCK_ROWS);
            }
        }
    }

    if (gamma_pending) {
        noc_async_read_barrier();
        cb_push_back(cb_gamma_tiles, SLICE_HIDDEN_TILES);
    }
}
