// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ISOLATED BENCH — rms_norm's cross-core combine, reader (NoC0) half.
//
// ONE topology, identical in every variant: the shipped op's reduce-scatter
// combine.  NUM_OWNERS cores of a row-group each gather NSLICE * OWN_ROWS stat
// tiles, reduce their OWN rows, funnel the finalized rows to the root, and the
// root multicasts the assembled B tiles back to the group (mcast_pipe).
//
// Nothing here is under test — the knob is the writer's landing-page map and the
// compute's matching read order.  This file only has to be the SAME for every
// variant so the measured delta is attributable to the gather.
//
// At NUM_OWNERS == 1 the owner IS the root and the funnel becomes a loopback
// write; the code stays one path (this costs the flat regime one local hop that
// the shipped flat root does not pay, identically in baseline and candidate).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "hostdevcommon/common_values.hpp"

#include "ttnn/cpp/ttnn/kernel_lib/mcast_pipe.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

using namespace dataflow_kernel_lib;

constexpr uint32_t cb_in = 0;
constexpr uint32_t cb_gathered = 4;
constexpr uint32_t cb_stat_out = 5;
constexpr uint32_t cb_rms_recip = 6;
constexpr uint32_t cb_scaler = 7;
constexpr uint32_t cb_bcast_stage = 8;

void kernel_main() {
    constexpr auto mc = McastArgs</*CT=*/0, /*RT=*/0>();
    constexpr uint32_t CT = mc.next_compile_time_args_offset();

    constexpr uint32_t S = get_compile_time_arg_val(CT + 0);
    constexpr uint32_t B = get_compile_time_arg_val(CT + 1);
    constexpr uint32_t NSLICE = get_compile_time_arg_val(CT + 2);
    constexpr uint32_t OWN_ROWS = get_compile_time_arg_val(CT + 3);
    constexpr uint32_t NUM_OWNERS = get_compile_time_arg_val(CT + 4);
    constexpr uint32_t STAT_BYTES = get_compile_time_arg_val(CT + 5);
    constexpr uint32_t GATHER_SEM = get_compile_time_arg_val(CT + 6);
    constexpr uint32_t STAT_READY_SEM = get_compile_time_arg_val(CT + 7);
    constexpr uint32_t IN_WAIT_TILES = get_compile_time_arg_val(CT + 8);

    constexpr uint32_t BLOCK_TILES = B * S;

    constexpr uint32_t RT = mc.next_runtime_args_offset();
    const uint32_t num_blocks = get_arg_val<uint32_t>(RT + 0);
    const uint32_t is_root = get_arg_val<uint32_t>(RT + 1);
    const uint32_t is_owner = get_arg_val<uint32_t>(RT + 2);
    const uint32_t my_first_row = get_arg_val<uint32_t>(RT + 3);
    const uint32_t root_x = get_arg_val<uint32_t>(RT + 4);
    const uint32_t root_y = get_arg_val<uint32_t>(RT + 5);

    Noc noc;

    {
        MaybeDeviceZoneScope("rd_stat_consts");
        calculate_and_prepare_reduce_scaler<cb_scaler, PoolType::SUM, ReduceDim::REDUCE_ROW>();
    }
    // cb_rms_recip IS the caller's resident output shard: the broadcast lands
    // straight into it, so the bench pays no copy for "the stat reached this core".
    const uint32_t recip_base = get_write_ptr(cb_rms_recip);
    const uint32_t bcast_stage_base = get_write_ptr(cb_bcast_stage);
    {
        MaybeDeviceZoneScope("rd_publish");
        cb_reserve_back(cb_in, IN_WAIT_TILES);
        cb_push_back(cb_in, IN_WAIT_TILES);
    }

    Semaphore<> gather_progress(GATHER_SEM);
    Semaphore<> stat_ready(STAT_READY_SEM);

    auto sender_pipe = mc.sender(noc);
    auto receiver_pipe = mc.receiver(noc);

    for (uint32_t block = 0; block < num_blocks; ++block) {
        if (block > 0) {
            cb_reserve_back(cb_in, BLOCK_TILES);
            cb_push_back(cb_in, BLOCK_TILES);
        }
        if (is_owner) {
            cb_reserve_back(cb_gathered, NSLICE * OWN_ROWS);
            {
                // The owner's view of the gather: how long it sits waiting for the
                // group's partials to land.  This is the zone the candidate is
                // trying to shorten.
                MaybeDeviceZoneScope("rd_gather_wait");
                gather_progress.wait_min((block + 1) * NSLICE);
            }
            cb_push_back(cb_gathered, NSLICE * OWN_ROWS);
            {
                MaybeDeviceZoneScope("rd_bcast_wait_stat");
                cb_wait_front(cb_stat_out, OWN_ROWS);
            }
            {
                MaybeDeviceZoneScope("rd_stat_to_root");
                const uint32_t dst = bcast_stage_base + (block * B + my_first_row) * STAT_BYTES;
                noc_async_write(get_read_ptr(cb_stat_out), get_noc_addr(root_x, root_y, dst), OWN_ROWS * STAT_BYTES);
                noc_async_write_barrier();
                stat_ready.up(noc, root_x, root_y, 1);
            }
            cb_pop_front(cb_stat_out, OWN_ROWS);
        }
        if (is_root) {
            {
                MaybeDeviceZoneScope("rd_bcast_wait_stat_root");
                stat_ready.wait_min((block + 1) * NUM_OWNERS);
            }
            cb_reserve_back(cb_rms_recip, B);
            {
                MaybeDeviceZoneScope("rd_bcast_send");
                sender_pipe.send(
                    bcast_stage_base + block * B * STAT_BYTES, recip_base + block * B * STAT_BYTES, B * STAT_BYTES);
            }
            cb_push_back(cb_rms_recip, B);
        } else {
            cb_reserve_back(cb_rms_recip, B);
            {
                MaybeDeviceZoneScope("rd_bcast_recv");
                receiver_pipe.receive();
            }
            cb_push_back(cb_rms_recip, B);
        }
    }
}
