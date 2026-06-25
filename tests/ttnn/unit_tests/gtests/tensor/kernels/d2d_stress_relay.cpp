// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Mode-switched stage op for the ForwardChainStress test. One source file, two builds
// selected by the STRESS_MODE kernel define (set via DataMovementConfig.defines), split so
// the per-iter sequence is:
//
//   COMPUTE -> wait both fabric links -> (competing fabric op) -> release both links -> SIGNAL
//
//   STRESS_MODE == COMPUTE: the producing op. (a) fused overwrite-gate — wait the outbound
//     consumed_sem (skipped iter 0 via skip_gate; absent on the last stage, which has no
//     outbound) so the prior forward has drained before we overwrite; (b) wait the inbound
//     data_ready_sem (the H2D feed on stage 0, or the D2D receiver elsewhere); (c) for this
//     worker's page range, read the upstream backing, ADD fill_delta to every element (the
//     per-stage mutation), write the dest; (d) atomic-inc the inbound consumed_counter.
//     Does NOT signal data_ready — that's the SIGNAL pass, run after the lease release so
//     the forward fires last.
//
//   STRESS_MODE == SIGNAL: atomic-inc the outbound data_ready_counter (one inc per worker
//     core -> num_workers acks), triggering the sender service to forward.
//
// WHY A PREPROCESSOR DEFINE, not a compile-time CT arg + if constexpr: the COMPUTE body
// reads CT[0..7] and a TensorAccessorArgs<8>, which only exist for the COMPUTE build; the
// SIGNAL build passes no CT args. An `if constexpr (MODE == ...)` dispatch — even inside a
// template with the discarded branch made dependent — was still observed to instantiate the
// COMPUTE handler for the SIGNAL kernel (the JIT compiler did not honor the discard),
// failing with "Index out of range" on the COMPUTE CT reads. #if physically removes the
// dead branch before compilation, so the SIGNAL build never sees the COMPUTE CT reads.

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
    // CT: [0]=has_gate, [1]=consumed_sem_addr (local L1; 0 if no gate),
    //     [2]=inbound data_ready_sem_addr (local L1), [3]=upstream_backing_addr,
    //     [4]=dest_addr, [5]=page_size, [6]=scratch_cb_index, [7]=fill_delta (fallback),
    //     [8]=metadata_enabled, [9]=metadata_size_bytes,
    //     [10]=inbound_metadata_l1_addr (worker L1 the inbound service mcast the blob into),
    //     [11..]=TensorAccessorArgs (upstream backing & dest share the per-shard spec).
    constexpr uint32_t has_gate = get_compile_time_arg_val(0);
    constexpr uint32_t consumed_sem_addr = get_compile_time_arg_val(1);
    constexpr uint32_t data_ready_sem_addr = get_compile_time_arg_val(2);
    constexpr uint32_t upstream_backing_addr = get_compile_time_arg_val(3);
    constexpr uint32_t dest_addr = get_compile_time_arg_val(4);
    constexpr uint32_t page_size = get_compile_time_arg_val(5);
    constexpr uint32_t scratch_cb_index = get_compile_time_arg_val(6);
    constexpr uint32_t fill_delta = get_compile_time_arg_val(7);
    constexpr uint32_t metadata_enabled = get_compile_time_arg_val(8);
    constexpr uint32_t metadata_size_bytes = get_compile_time_arg_val(9);
    constexpr uint32_t inbound_metadata_l1_addr = get_compile_time_arg_val(10);
    constexpr auto acc_args = TensorAccessorArgs<11>();

    // RT: [0]=start_page, [1]=end_page, [2]=inbound consumed_counter_addr,
    //     [3]=inbound service NoC x, [4]=inbound service NoC y, [5]=skip_gate,
    //     [6]=is_metadata_writer (1 on the designated producing core, else 0),
    //     [7]=outbound D2D sender metadata L1 addr, [8]=outbound service NoC x, [9]=y.
    const uint32_t start_page = get_arg_val<uint32_t>(0);
    const uint32_t end_page = get_arg_val<uint32_t>(1);
    const uint32_t consumed_counter_addr = get_arg_val<uint32_t>(2);
    const uint32_t service_noc_x = get_arg_val<uint32_t>(3);
    const uint32_t service_noc_y = get_arg_val<uint32_t>(4);
    const uint32_t skip_gate = get_arg_val<uint32_t>(5);
    const uint32_t is_metadata_writer = get_arg_val<uint32_t>(6);
    const uint32_t outbound_metadata_l1_addr = get_arg_val<uint32_t>(7);
    const uint32_t outbound_service_noc_x = get_arg_val<uint32_t>(8);
    const uint32_t outbound_service_noc_y = get_arg_val<uint32_t>(9);

    // (a) Fused overwrite-gate: wait the prior forward to drain (sender consumed_sem), then
    //     reset. Compiled away when has_gate == 0 (last stage). Skipped on iter 0.
    if constexpr (has_gate) {
        if (skip_gate == 0) {
            volatile tt_l1_ptr uint32_t* consumed_sem =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(consumed_sem_addr);
            while (true) {
                invalidate_l1_cache();
                if (*consumed_sem > 0) {
                    *consumed_sem = 0;
                    break;
                }
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

    // (b2) The per-stage increment: read it from the inbound metadata the upstream
    //      service multicast into this core's L1 (present once data_ready fired above),
    //      instead of the compile-time fill_delta. Exercises the metadata path: a
    //      dropped/garbled blob shifts the end-stage iota and fails the test.
    uint32_t delta = fill_delta;
    if constexpr (metadata_enabled) {
        invalidate_l1_cache();
        delta = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(inbound_metadata_l1_addr);
    }

    // (c) Relay this worker's page range upstream -> dest, mutating every element by
    //     delta (+1 per stage) so the end value is a function of every hop.
    auto upstream = TensorAccessor(acc_args, upstream_backing_addr);
    auto downstream = TensorAccessor(acc_args, dest_addr);
    Noc noc;
    CircularBuffer scratch_cb(scratch_cb_index);
    const uint32_t cb_l1 = scratch_cb.get_write_ptr();
    // Raw volatile view for the in-place += delta (an L1 read-modify-write, not a NoC
    // op); CoreLocalMem view is the NoC read-dest / write-src.
    volatile tt_l1_ptr uint32_t* scratch = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_l1);
    CoreLocalMem<uint32_t> scratch_mem(cb_l1);
    constexpr uint32_t page_elems = page_size / sizeof(uint32_t);
    for (uint32_t p = start_page; p < end_page; ++p) {
        noc.async_read(upstream, scratch_mem, page_size, {.page_id = p}, {});
        noc.async_read_barrier();
        for (uint32_t e = 0; e < page_elems; ++e) {
            scratch[e] += delta;
        }
        noc.async_write<NocOptions::DEFAULT, page_size>(scratch_mem, downstream, page_size, {}, {.page_id = p});
    }
    noc.async_write_barrier();

    // (c2) Propagate the metadata downstream: the designated producing core copies the
    //      blob it received (this core's L1) into the outbound D2D sender's service-core
    //      metadata L1, so the sender ships it to the next stage. Mirrors the H2D->D2D
    //      bridge worker. Only the producing designated core does this (is_metadata_writer).
    if constexpr (metadata_enabled) {
        if (is_metadata_writer != 0) {
            noc.async_write(
                CoreLocalMem<uint32_t>(inbound_metadata_l1_addr),
                UnicastEndpoint{},
                metadata_size_bytes,
                {},
                {.noc_x = outbound_service_noc_x, .noc_y = outbound_service_noc_y, .addr = outbound_metadata_l1_addr});
            noc.async_write_barrier();
        }
    }

    // (d) Ack the inbound: it may refill its backing with the next iter.
    const uint64_t consumed_noc = get_noc_addr(service_noc_x, service_noc_y, consumed_counter_addr);
    noc_semaphore_inc(consumed_noc, 1);
    noc.async_atomic_barrier();
}

#else  // STRESS_MODE == STRESS_MODE_SIGNAL

void kernel_main() {
    // RT: [0]=outbound data_ready_counter_addr, [1]=outbound service NoC x, [2]=y.
    const uint32_t data_ready_counter_addr = get_arg_val<uint32_t>(0);
    const uint32_t service_noc_x = get_arg_val<uint32_t>(1);
    const uint32_t service_noc_y = get_arg_val<uint32_t>(2);
    const uint64_t data_ready_noc = get_noc_addr(service_noc_x, service_noc_y, data_ready_counter_addr);
    Noc noc;
    noc_semaphore_inc(data_ready_noc, 1);
    noc.async_atomic_barrier();
}

#endif
