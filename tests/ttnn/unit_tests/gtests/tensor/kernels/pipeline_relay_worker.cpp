// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Generic per-stage relay worker for the single-host streaming-pipeline test
// (Host -> H2D -> [D2D]* -> output -> Host). One kernel serves every stage of an
// N-stage pipeline because the two handshake halves are uniform:
//
//   * the CONSUMER (upstream) half is identical whether the upstream is an
//     H2DStreamService or a D2DStreamServiceReceiver — both expose a worker-grid
//     data_ready GlobalSemaphore + a per-coord consumed_counter on a service core.
//   * the PRODUCER (downstream) half is always the D2DStreamServiceSender inverted
//     handshake.
//
// Per stage, per iteration:
//   1. wait on the upstream data_ready_sem (the upstream service mcasts it after a
//      transfer lands), reset it,
//   2. relay this worker's assigned page range upstream_backing -> downstream_dest
//      (both are local DRAM on this stage's device),
//   3. atomic-inc the upstream consumed_counter (release the upstream to refill),
//   4. if this stage produces downstream (every stage except the last): atomic-inc
//      the downstream D2D sender's data_ready_counter (signal there's data to
//      forward), then RETURN — it does NOT wait for the D2D to drain.
//
//   stage 0      : [H2D consumer] -> [D2D-sender producer]
//   middle stage : [D2D consumer] -> [D2D-sender producer]   (produce_downstream==1)
//   last stage   : [D2D consumer] -> write output_tensor      (produce_downstream==0)
//
// DECOUPLED from the downstream D2D on purpose: this models a stage's compute op,
// which runs to completion BEFORE the D2D forwards its output. The host runs the
// per-stage cadence wait_for_fabric_links -> launch -> Finish -> release_fabric_links,
// so the D2D only forwards after this program Finishes (release is post-Finish).
// Waiting on the downstream consumed_sem here would deadlock that order. Flow control
// for downstream_dest is the host's wait_for_fabric_links() on the boundary before
// the stage's next launch.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/dataflow/endpoints.h"
#include "api/tensor/noc_traits.h"

// CT layout (must stay in sync with build_relay_workload in test_stream_pipeline.cpp).
constexpr uint32_t upstream_data_ready_sem_addr = get_compile_time_arg_val(0);  // local worker L1
constexpr uint32_t upstream_backing_addr = get_compile_time_arg_val(1);         // read from (local DRAM)
constexpr uint32_t downstream_dest_addr = get_compile_time_arg_val(2);          // write to (local DRAM)
constexpr uint32_t page_size = get_compile_time_arg_val(3);
constexpr uint32_t num_iters = get_compile_time_arg_val(4);
constexpr uint32_t scratch_cb_index = get_compile_time_arg_val(5);
constexpr uint32_t produce_downstream = get_compile_time_arg_val(6);
constexpr uint32_t metadata_enabled = get_compile_time_arg_val(7);
constexpr uint32_t metadata_size_bytes = get_compile_time_arg_val(8);
// L1 address (uniform on every worker core) where the UPSTREAM service multicast
// this iter's metadata blob; the designated worker forwards it downstream.
constexpr uint32_t upstream_metadata_l1_addr = get_compile_time_arg_val(9);
// Upstream backing and downstream dest share the same per-shard spec, so a single
// TensorAccessorArgs set is reused with the two distinct base addresses.
constexpr auto acc_args = TensorAccessorArgs<10>();

void kernel_main() {
    const uint32_t start_page = get_arg_val<uint32_t>(0);
    const uint32_t end_page = get_arg_val<uint32_t>(1);
    const uint32_t upstream_consumed_counter_addr = get_arg_val<uint32_t>(2);
    const uint32_t upstream_service_noc_x = get_arg_val<uint32_t>(3);
    const uint32_t upstream_service_noc_y = get_arg_val<uint32_t>(4);
    // Downstream service-core fields are only meaningful when produce_downstream == 1.
    const uint32_t downstream_data_ready_counter_addr = get_arg_val<uint32_t>(5);
    const uint32_t downstream_service_noc_x = get_arg_val<uint32_t>(6);
    const uint32_t downstream_service_noc_y = get_arg_val<uint32_t>(7);
    // Metadata fields: only the designated writer (is_metadata_writer == 1) forwards,
    // and only when this stage produces downstream. downstream_metadata_l1_addr is the
    // D2D sender service core's metadata L1 buffer.
    const uint32_t is_metadata_writer = get_arg_val<uint32_t>(8);
    const uint32_t downstream_metadata_l1_addr = get_arg_val<uint32_t>(9);

    auto upstream = TensorAccessor(acc_args, upstream_backing_addr);
    auto downstream = TensorAccessor(acc_args, downstream_dest_addr);

    // 2.0 NoC interface for the bulk DRAM<->L1 relay and the unicast metadata forward.
    Noc noc;
    CircularBuffer scratch_cb(scratch_cb_index);
    // Fixed scratch L1 staging slot (the relay never push/pop_front's the CB; it reuses
    // this single entry's address as both read-dest and write-src, as the legacy code did).
    CoreLocalMem<uint32_t> scratch(scratch_cb.get_write_ptr());

    volatile tt_l1_ptr uint32_t* up_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(upstream_data_ready_sem_addr);
    const uint64_t up_consumed_noc =
        get_noc_addr(upstream_service_noc_x, upstream_service_noc_y, upstream_consumed_counter_addr);

    // Downstream producer state — dead-code-eliminated when produce_downstream == 0.
    uint64_t down_data_ready_noc = 0;
    if constexpr (produce_downstream) {
        down_data_ready_noc =
            get_noc_addr(downstream_service_noc_x, downstream_service_noc_y, downstream_data_ready_counter_addr);
    }

    for (uint32_t iter = 0; iter < num_iters; ++iter) {
        // 1. Wait for the upstream service to signal this iter's data landed.
        while (true) {
            invalidate_l1_cache();
            if (*up_sem > 0) {
                *up_sem = 0;
                break;
            }
        }

        // 2. Relay this worker's assigned pages upstream_backing -> downstream_dest
        //    (local DRAM -> scratch -> local DRAM), page by page.
        for (uint32_t p = start_page; p < end_page; ++p) {
            noc.async_read(upstream, scratch, page_size, {.page_id = p}, {});
            noc.async_read_barrier();
            noc.async_write<NocOptions::DEFAULT, page_size>(scratch, downstream, page_size, {}, {.page_id = p});
        }
        noc.async_write_barrier();

        // 2b. (designated producer worker only) forward the metadata the upstream
        //     service multicast into this core's L1 to the downstream D2D sender
        //     service core, so the sender ships it with this iter's transfer. Done
        //     before acking the upstream (which may refill the metadata L1) and
        //     before signaling downstream data_ready (so it is in place to forward).
        //     The terminal stage produces nothing: its metadata already landed in the
        //     worker L1 from the upstream receiver's multicast (verified by the host).
        if constexpr (produce_downstream && metadata_enabled) {
            if (is_metadata_writer != 0) {
                noc.async_write(
                    CoreLocalMem<uint32_t>(upstream_metadata_l1_addr),
                    UnicastEndpoint{},
                    metadata_size_bytes,
                    {},
                    {.noc_x = downstream_service_noc_x,
                     .noc_y = downstream_service_noc_y,
                     .addr = downstream_metadata_l1_addr});
                noc.async_write_barrier();
            }
        }

        // 3. Ack the upstream: it may now refill upstream_backing with the next iter.
        noc_semaphore_inc(up_consumed_noc, 1);
        noc.async_atomic_barrier();

        // 4. Producer half (every stage except the last): signal the downstream D2D
        //    sender there's data to forward, then RETURN — do NOT wait for the D2D to
        //    drain. The host releases the fabric links (release_fabric_links) only
        //    AFTER this program has Finished, mirroring the production path where a
        //    stage's compute op runs to completion and the D2D forwards afterward.
        //    Blocking on the D2D here would deadlock that finish-before-release order.
        //    Flow control (not overwriting downstream_dest until the prior transfer
        //    drained) is the host's job: it wait_for_fabric_links() on this boundary
        //    before re-launching this stage next iter.
        if constexpr (produce_downstream) {
            noc_semaphore_inc(down_data_ready_noc, 1);
            noc.async_atomic_barrier();
        }
    }
}
