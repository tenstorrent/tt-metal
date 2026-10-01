// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Terminal-stage (D2H) relay worker for the single-host streaming-pipeline test
// (Host -> H2D -> [D2D]* -> D2H -> Host). A variant of pipeline_relay_worker.cpp: the
// CONSUMER (upstream) half is byte-for-byte identical, but the PRODUCER half drives a
// D2HStreamService instead of a D2DStreamServiceSender, so the last stage streams its
// result to a host consumer over a PCIe socket rather than writing a standalone output
// tensor the host reads back via ReadShard.
//
//   * the CONSUMER (upstream) half is identical whether the upstream is an
//     H2DStreamService or a D2DStreamServiceReceiver — both expose a worker-grid
//     data_ready GlobalSemaphore + a per-coord consumed_counter on a service core.
//   * the PRODUCER (D2H) half stages this iter's pages into the D2H backing tensor and
//     hands off to the persistent D2H sender kernel via the inverted
//     write_ack / transfer_done handshake (the D2H analog of the D2D sender's
//     data_ready / consumed handshake).
//
// Per iteration:
//   0. overwrite gate (skipped on iter 0): wait on transfer_done_sem — the persistent
//      D2H sender mcasts it once it has streamed the PREVIOUS iter's backing to host —
//      then reset it, so this iter does not clobber the backing mid-drain. This is the
//      D2H analog of waiting on the D2D sender's consumed_sem.
//   1. wait on the upstream data_ready_sem (the upstream service mcasts it after a
//      transfer lands), reset it,
//   2. relay this worker's assigned page range upstream_backing -> d2h_backing (both
//      local DRAM on this stage's device); the designated metadata writer also forwards
//      the upstream's metadata blob to the D2H service core so the sender ships it inline,
//   3. atomic-inc the upstream consumed_counter (release the upstream to refill),
//   4. atomic-inc the D2H sender's write_ack_counter (signal this iter's data is staged
//      in the backing). The sender drains the backing to the host socket once write_ack
//      has been bumped once per worker; the host pulls it via read_from_tensor().
//
// Unlike pipeline_relay_worker.cpp there is no produce_downstream branch: every stage
// running this kernel is the terminal stage, so the D2H producer half always executes.
// Cross-iter flow control is the step-0 overwrite gate, NOT a host-side
// wait_for_fabric_links() boundary (which is what gated the D2D producer path).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/dataflow/endpoints.h"
#include "api/tensor/noc_traits.h"

// CT layout (must stay in sync with make_d2h_relay_workload in test_cross_process_d2d_stream_service.cpp).
constexpr uint32_t upstream_data_ready_sem_addr = get_compile_time_arg_val(0);  // local worker L1
constexpr uint32_t upstream_backing_addr = get_compile_time_arg_val(1);         // read from (local DRAM)
constexpr uint32_t d2h_backing_addr = get_compile_time_arg_val(2);              // write to (D2H backing, local DRAM)
constexpr uint32_t page_size = get_compile_time_arg_val(3);
constexpr uint32_t num_iters = get_compile_time_arg_val(4);
constexpr uint32_t scratch_cb_index = get_compile_time_arg_val(5);
constexpr uint32_t metadata_enabled = get_compile_time_arg_val(6);
constexpr uint32_t metadata_size_bytes = get_compile_time_arg_val(7);
// L1 address (uniform on every worker core) where the UPSTREAM service multicast
// this iter's metadata blob; the designated worker forwards it to the D2H service core.
constexpr uint32_t upstream_metadata_l1_addr = get_compile_time_arg_val(8);

constexpr uint32_t d2h_transfer_done_sem_addr = get_compile_time_arg_val(9);
// Upstream backing and D2H backing share the same per-shard spec, so a single
// TensorAccessorArgs set is reused with the two distinct base addresses.
constexpr auto acc_args = TensorAccessorArgs<10>();

void kernel_main() {
    const uint32_t start_page = get_arg_val<uint32_t>(0);
    const uint32_t end_page = get_arg_val<uint32_t>(1);
    const uint32_t upstream_consumed_counter_addr = get_arg_val<uint32_t>(2);
    const uint32_t upstream_service_noc_x = get_arg_val<uint32_t>(3);
    const uint32_t upstream_service_noc_y = get_arg_val<uint32_t>(4);
    // D2H service-core fields: write_ack_counter the worker incs to tell the persistent
    // D2H sender this iter's data is staged, plus the NoC coords that locate the core.
    const uint32_t d2h_write_ack_counter_addr = get_arg_val<uint32_t>(5);
    const uint32_t d2h_service_noc_x = get_arg_val<uint32_t>(6);
    const uint32_t d2h_service_noc_y = get_arg_val<uint32_t>(7);
    // Metadata fields: only the designated writer (is_metadata_writer == 1) forwards.
    // d2h_metadata_input_addr is the metadata L1 buffer on the D2H service core.
    const uint32_t is_metadata_writer = get_arg_val<uint32_t>(8);
    const uint32_t d2h_metadata_input_addr = get_arg_val<uint32_t>(9);

    auto upstream = TensorAccessor(acc_args, upstream_backing_addr);
    auto d2h_backing = TensorAccessor(acc_args, d2h_backing_addr);

    // 2.0 NoC interface for the bulk DRAM<->L1 relay and the unicast metadata forward.
    Noc noc;
    CircularBuffer scratch_cb(scratch_cb_index);
    // Fixed scratch L1 staging slot (the relay never push/pop_front's the CB; it reuses
    // this single entry's address as both read-dest and write-src, as the legacy code did).
    CoreLocalMem<uint32_t> scratch(scratch_cb.get_write_ptr());

    volatile tt_l1_ptr uint32_t* up_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(upstream_data_ready_sem_addr);
    const uint64_t up_consumed_noc =
        get_noc_addr(upstream_service_noc_x, upstream_service_noc_y, upstream_consumed_counter_addr);

    // D2H producer state — always live (this kernel only runs on the terminal stage).
    volatile tt_l1_ptr uint32_t* transfer_done_sem =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(d2h_transfer_done_sem_addr);
    const uint64_t d2h_write_ack_noc = get_noc_addr(d2h_service_noc_x, d2h_service_noc_y, d2h_write_ack_counter_addr);

    for (uint32_t iter = 0; iter < num_iters; ++iter) {
        // 0. D2H overwrite gate: wait for the persistent D2H sender to signal it has
        //    streamed the previous iter to host before reusing the backing. (Skipped on
        //    iter 0; no-op in this test since num_iters == 1.)
        if (iter > 0) {
            while (*transfer_done_sem == 0) {
                invalidate_l1_cache();
            }
            *transfer_done_sem = 0;
        }
        // 1. Wait for the upstream service to signal this iter's data landed.
        while (true) {
            invalidate_l1_cache();
            if (*up_sem > 0) {
                *up_sem = 0;
                break;
            }
        }

        // 2. Relay this worker's assigned pages upstream_backing -> D2H backing
        //    (local DRAM -> scratch -> local DRAM), page by page.
        for (uint32_t p = start_page; p < end_page; ++p) {
            noc.async_read(upstream, scratch, page_size, {.page_id = p}, {});
            noc.async_read_barrier();
            noc.async_write<NocOptions::DEFAULT, page_size>(scratch, d2h_backing, page_size, {}, {.page_id = p});
        }
        noc.async_write_barrier();

        // 2b. (designated metadata writer only) forward the metadata the upstream
        //     service multicast into this core's L1 to the D2H service core, so the
        //     persistent D2H sender ships it to host with this iter's transfer. Done
        //     before acking the upstream (which may refill the metadata L1) and before
        //     signaling write_ack (so it is in place to forward).
        if constexpr (metadata_enabled) {
            if (is_metadata_writer != 0) {
                noc.async_write(
                    CoreLocalMem<uint32_t>(upstream_metadata_l1_addr),
                    UnicastEndpoint{},
                    metadata_size_bytes,
                    {},
                    {.noc_x = d2h_service_noc_x, .noc_y = d2h_service_noc_y, .addr = d2h_metadata_input_addr});
                noc.async_write_barrier();
            }
        }

        // 3. Ack the upstream: it may now refill upstream_backing with the next iter.
        noc_semaphore_inc(up_consumed_noc, 1);
        noc.async_atomic_barrier();

        // 4. D2H producer half: signal the persistent D2H sender this iter's data is in
        //    the backing by inc'ing its write_ack_counter. The sender drains the backing
        //    to host once write_ack has been bumped num_workers times; the overwrite gate
        //    (step 0) keeps the next iter from clobbering the backing before that drain.
        noc_semaphore_inc(d2h_write_ack_noc, 1);
        noc.async_atomic_barrier();
    }
}
