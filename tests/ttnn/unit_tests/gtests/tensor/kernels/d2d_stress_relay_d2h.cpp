// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Terminal-stage (D2H) COMPUTE op for the ForwardChainStress test. A variant of
// d2d_stress_relay.cpp for the LAST stage, which streams its result to a host consumer
// over a PCIe socket (D2HStreamService) instead of forwarding it over fabric to a
// downstream D2DStreamServiceSender. The consumer half (overwrite gate, wait inbound
// data_ready, per-stage +delta mutation, ack inbound consumed) mirrors the D2D stress op;
// only the producer half differs — it drives the D2H-inverted handshake.
//
// Only the COMPUTE build (STRESS_MODE=0) is ever instantiated for D2H: there is no SIGNAL
// pass. The D2D op deferred its data_ready inc to a separate SIGNAL workload run after the
// lease release; D2H instead bumps the persistent sender's write_ack_counter inline at the
// end of COMPUTE (the D2H analog of data_ready), so the host-side read_from_tensor drains
// this iter. The STRESS_MODE=1 branch below is an empty stub, kept only so this file stays
// a structural mirror of d2d_stress_relay.cpp.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/dataflow/endpoints.h"
#include "api/tensor/noc_traits.h"

#define STRESS_MODE_COMPUTE 0
#define STRESS_MODE_SIGNAL 1

#ifndef STRESS_MODE
#error "STRESS_MODE must be defined (0 = COMPUTE, 1 = SIGNAL) via DataMovementConfig.defines"
#endif

#if STRESS_MODE == STRESS_MODE_COMPUTE
void kernel_main() {
    // CT: [0]=inbound data_ready_sem_addr (local L1), [1]=upstream_backing_addr,
    //     [2]=d2h_backing_addr (dest), [3]=page_size, [4]=scratch_cb_index,
    //     [5]=fill_delta (fallback), [6]=metadata_enabled, [7]=metadata_size_bytes,
    //     [8]=inbound_metadata_l1_addr, [9]=d2h_transfer_done_sem_addr (local L1; the D2H
    //     overwrite gate — D2H-specific, appended), [10..]=TensorAccessorArgs (upstream &
    //     D2H backing share the per-shard spec).
    constexpr uint32_t data_ready_sem_addr = get_compile_time_arg_val(0);
    constexpr uint32_t upstream_backing_addr = get_compile_time_arg_val(1);
    constexpr uint32_t dest_addr = get_compile_time_arg_val(2);
    constexpr uint32_t page_size = get_compile_time_arg_val(3);
    constexpr uint32_t scratch_cb_index = get_compile_time_arg_val(4);
    constexpr uint32_t fill_delta = get_compile_time_arg_val(5);
    constexpr uint32_t metadata_enabled = get_compile_time_arg_val(6);
    constexpr uint32_t metadata_size_bytes = get_compile_time_arg_val(7);
    constexpr uint32_t inbound_metadata_l1_addr = get_compile_time_arg_val(8);
    constexpr uint32_t transfer_done_sem_addr = get_compile_time_arg_val(9);
    constexpr auto acc_args = TensorAccessorArgs<10>();

    // RT: [0]=start_page, [1]=end_page, [2]=inbound consumed_counter_addr,
    //     [3]=inbound service NoC x, [4]=inbound service NoC y, [5]=skip_gate,
    //     [6]=is_metadata_writer, [7]=d2h_metadata_input_addr, [8]=d2h service NoC x,
    //     [9]=d2h service NoC y, [10]=d2h_write_ack_counter_addr (D2H-specific, appended).
    const uint32_t start_page = get_arg_val<uint32_t>(0);
    const uint32_t end_page = get_arg_val<uint32_t>(1);
    const uint32_t consumed_counter_addr = get_arg_val<uint32_t>(2);
    const uint32_t service_noc_x = get_arg_val<uint32_t>(3);
    const uint32_t service_noc_y = get_arg_val<uint32_t>(4);
    const uint32_t skip_gate = get_arg_val<uint32_t>(5);
    const uint32_t is_metadata_writer = get_arg_val<uint32_t>(6);
    const uint32_t d2h_metadata_input_addr = get_arg_val<uint32_t>(7);
    const uint32_t d2h_service_noc_x = get_arg_val<uint32_t>(8);
    const uint32_t d2h_service_noc_y = get_arg_val<uint32_t>(9);
    const uint32_t d2h_write_ack_counter_addr = get_arg_val<uint32_t>(10);

    // (a) D2H overwrite gate: wait the persistent D2H sender to drain the PREVIOUS iter
    //     (transfer_done_sem), then reset. Skipped on iter 0 via skip_gate. This is the
    //     D2H analog of the D2D consumed_sem gate.
    if (skip_gate == 0) {
        volatile tt_l1_ptr uint32_t* transfer_done_sem =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(transfer_done_sem_addr);
        while (true) {
            invalidate_l1_cache();
            if (*transfer_done_sem > 0) {
                *transfer_done_sem = 0;
                break;
            }
        }
    }

    // (b) Wait the inbound service to signal this iter's data landed, then reset.
    volatile tt_l1_ptr uint32_t* data_ready_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(data_ready_sem_addr);
    while (true) {
        invalidate_l1_cache();
        if (*data_ready_sem > 0) {
            *data_ready_sem = 0;
            break;
        }
    }

    // (b2) Per-stage increment from the inbound metadata blob (else compile-time fallback).
    uint32_t delta = fill_delta;
    if constexpr (metadata_enabled) {
        invalidate_l1_cache();
        delta = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(inbound_metadata_l1_addr);
    }

    // (c) Relay this worker's page range upstream -> D2H backing, mutating every element
    //     by delta (+1 per stage) so the end value is a function of every hop.
    auto upstream = TensorAccessor(acc_args, upstream_backing_addr);
    auto d2h_backing = TensorAccessor(acc_args, dest_addr);
    Noc noc;
    CircularBuffer scratch_cb(scratch_cb_index);
    const uint32_t cb_l1 = scratch_cb.get_write_ptr();
    volatile tt_l1_ptr uint32_t* scratch = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_l1);
    CoreLocalMem<uint32_t> scratch_mem(cb_l1);
    constexpr uint32_t page_elems = page_size / sizeof(uint32_t);
    for (uint32_t p = start_page; p < end_page; ++p) {
        noc.async_read(upstream, scratch_mem, page_size, {.page_id = p}, {});
        noc.async_read_barrier();
        for (uint32_t e = 0; e < page_elems; ++e) {
            scratch[e] += delta;
        }
        noc.async_write<NocOptions::DEFAULT, page_size>(scratch_mem, d2h_backing, page_size, {}, {.page_id = p});
    }
    noc.async_write_barrier();

    // (c2) Designated writer forwards the metadata blob to the D2H service core, so the
    //      persistent D2H sender ships it to host inline with this iter's transfer.
    if constexpr (metadata_enabled) {
        if (is_metadata_writer != 0) {
            noc.async_write(
                CoreLocalMem<uint32_t>(inbound_metadata_l1_addr),
                UnicastEndpoint{},
                metadata_size_bytes,
                {},
                {.noc_x = d2h_service_noc_x, .noc_y = d2h_service_noc_y, .addr = d2h_metadata_input_addr});
            noc.async_write_barrier();
        }
    }

    // (d) Ack the inbound: it may refill its backing with the next iter.
    const uint64_t consumed_noc = get_noc_addr(service_noc_x, service_noc_y, consumed_counter_addr);
    noc_semaphore_inc(consumed_noc, 1);
    noc.async_atomic_barrier();

    // (e) D2H producer half: signal the persistent D2H sender this iter's data is staged
    //     by inc'ing its write_ack_counter (replaces the D2D SIGNAL pass).
    const uint64_t d2h_write_ack_noc = get_noc_addr(d2h_service_noc_x, d2h_service_noc_y, d2h_write_ack_counter_addr);
    noc_semaphore_inc(d2h_write_ack_noc, 1);
    noc.async_atomic_barrier();
}
#else  // STRESS_MODE == STRESS_MODE_SIGNAL — UNUSED for D2H (write_ack replaces SIGNAL)
void kernel_main() {}
#endif
