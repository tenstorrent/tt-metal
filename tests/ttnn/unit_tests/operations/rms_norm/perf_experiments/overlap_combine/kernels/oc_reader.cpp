// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// overlap_combine bench — reader (NoC0).
//
// The reader half of the cross-core combine ONLY.  Everything else about the op
// is held constant / trivial per the isolation table:
//   * x and out are RESIDENT L1 shards (their CBs are bound to the caller's
//     buffers), so there is no `load_block` at all — the reader publishes the
//     shard once and never moves a byte of x.
//   * no gamma, no W-mask, no tilize (TILE layout only).
// What remains is exactly the part under test: the gather landing + the stat
// broadcast, and the SCHEDULE they impose on the compute kernel.
//
// This kernel is IDENTICAL for every variant in the bake-off (baseline,
// pipelined, coarse-stat).  The only things a variant changes are the compute
// kernel's loop ORDER and the host CB DEPTHS — which is the point: the schedule
// lives in compute, not in the dataflow.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"

#include "ttnn/cpp/ttnn/kernel_lib/mcast_pipe.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

using namespace dataflow_kernel_lib;

constexpr uint32_t cb_input_tiles = 0;
constexpr uint32_t cb_gathered_partials = 4;
constexpr uint32_t cb_rms_bcast = 5;
constexpr uint32_t cb_rms_recip = 6;
constexpr uint32_t cb_scaler = 7;

void kernel_main() {
    constexpr auto mc = McastArgs</*CT=*/0, /*RT=*/0>();
    constexpr uint32_t CT = mc.next_compile_time_args_offset();

    constexpr uint32_t STAT_ROWS = get_compile_time_arg_val(CT + 0);   // SB
    constexpr uint32_t NUM_SLICES = get_compile_time_arg_val(CT + 1);  // s
    constexpr uint32_t STAT_TILE_BYTES = get_compile_time_arg_val(CT + 2);
    constexpr uint32_t GATHER_SEM_ID = get_compile_time_arg_val(CT + 3);
    constexpr uint32_t SHARD_TILES = get_compile_time_arg_val(CT + 4);

    constexpr uint32_t RT = mc.next_runtime_args_offset();
    const uint32_t num_stat_blocks = get_arg_val<uint32_t>(RT + 0);
    const uint32_t is_root = get_arg_val<uint32_t>(RT + 1);

    Noc noc;

    {
        MaybeDeviceZoneScope("rd_stat_consts");
        calculate_and_prepare_reduce_scaler<cb_scaler, PoolType::SUM, ReduceDim::REDUCE_ROW>();
    }

    // Publish the resident input shard ONCE.  cb_input_tiles is bound to the
    // caller's L1 buffer, so this is pure bookkeeping (zero NoC reads); compute
    // holds the whole window for the kernel's life and addresses every block by
    // an absolute tile base, so nothing is ever popped or re-published.
    {
        MaybeDeviceZoneScope("rd_publish_shard");
        cb_reserve_back(cb_input_tiles, SHARD_TILES);
        cb_push_back(cb_input_tiles, SHARD_TILES);
    }

    auto sender_pipe = mc.sender(noc);
    auto receiver_pipe = mc.receiver(noc);
    Semaphore<> gather_progress(GATHER_SEM_ID);

    for (uint32_t sb = 0; sb < num_stat_blocks; ++sb) {
        if (is_root) {
            // Gather landing.  With a depth-2 landing buffer this reserve is what
            // lets contributors run one stat block AHEAD of the root's reduce.
            cb_reserve_back(cb_gathered_partials, NUM_SLICES * STAT_ROWS);
            {
                MaybeDeviceZoneScope("rd_gather_wait");
                gather_progress.wait_min((sb + 1) * NUM_SLICES);
            }
            cb_push_back(cb_gathered_partials, NUM_SLICES * STAT_ROWS);

            {
                MaybeDeviceZoneScope("rd_bcast_wait_stat");
                cb_wait_front(cb_rms_bcast, STAT_ROWS);
            }
            cb_reserve_back(cb_rms_recip, STAT_ROWS);
            {
                MaybeDeviceZoneScope("rd_bcast_send");
                sender_pipe.send(get_read_ptr(cb_rms_bcast), get_write_ptr(cb_rms_recip), STAT_ROWS * STAT_TILE_BYTES);
            }
            cb_push_back(cb_rms_recip, STAT_ROWS);
            cb_pop_front(cb_rms_bcast, STAT_ROWS);
        } else {
            cb_reserve_back(cb_rms_recip, STAT_ROWS);
            {
                // The contributor's exposed combine latency.  This is THE number
                // the whole idea exists to cut.
                MaybeDeviceZoneScope("rd_bcast_recv");
                receiver_pipe.receive();
            }
            cb_push_back(cb_rms_recip, STAT_ROWS);
        }
    }
}
