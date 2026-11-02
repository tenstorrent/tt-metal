// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// The line transport of the fused GR read: one transport core per fabric link all-gathers `tiles` local tiles over the
// TP4 line's 1D fabric from inside the program, into page t * page_tile_stride + rank * page_rank_stride of the
// gathered tensor (the chain's all-gather pages); this core carries tiles t = first_tile + i * tile_step.  `source` 0:
// a local tensor this kernel reads (NCRISC also copies it into the device's own pages); `source` 1: producer cores of
// the program write the tiles into this core's scratch CB and the device's own pages and raise the scratch semaphore
// once per tile.  A fabric link has one worker sender channel per direction and the open handshake does not queue, so
// a program keeps at most one open sender per link per direction (transport2.cpp runs two phases on one connection).
//
// NCRISC (NOC_0) sends backward (the ranks below); BRISC (NOC_1) sends forward and owns the waits.  Per phase per call:
//   1. barrier: every device raises the phase's barrier semaphore on every peer once (FACT 1: exactly
//      line_devices - 1 arrivals per counter per call); BRISC waits for line_devices - 1, resets it with a plain L1
//      store and only then releases NCRISC through `go` (FACT 2: the reset precedes every data send of either RISC).
//   2. data: each tile is one fused write + atomic-increment packet; BRISC waits for (line_devices - 1) x this core's
//      tiles (FACT 3: no device leaves a phase with partial data) and resets.
// No launch resets a semaphore, so each is reset by its owner before the program ends (BRISC: barrier, data and, with
// `source` 1, `done` and the producers' scratch semaphore; NCRISC: `go`).  A packet header is rewritten only after
// the NoC's writes are flushed.  Two phases reuse `go` and `done`: BRISC's second-phase `go.set(1)` is ordered after
// NCRISC's first-phase `go.set(0)` through `done`, so a release is never mistaken for the previous phase's; both
// arrives go out at kernel start, and the barrier argument holds per counter.  After the data wait BRISC raises each
// listed consumer's semaphore once.  The two delay arguments (spin cycles) are for skew testing; both 0 in the model.
//
// Runtime block of a phase at `first_arg`: 0 gathered tensor address, 1 local tensor address, 2 barrier semaphore
// address, 3 data semaphore address, 4 consumer count N, 5 consumer semaphore id, 6..6+2N the consumers' NoC (x, y).

#pragma once

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/semaphore.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/linear/api.h"

using namespace tt::tt_fabric::linear::experimental;

constexpr uint32_t phase_runtime_args = 6;  // before the consumers' coordinates

// This RISC's connection along the line: BRISC forward (the ranks above), NCRISC backward (the ranks below).
// The 1D fabric sender (WorkerToFabricEdmSender, PacketHeaderPool, fabric_multicast_noc_*) has no Device 2.0
// interface; its packets take raw 64-bit NoC addresses.
struct LineSender {
    tt::tt_fabric::WorkerToFabricEdmSender connection;
    volatile PACKET_HEADER_TYPE* header;
    uint32_t rank;
    uint32_t range;
    bool has_peers;

    template <uint32_t line_devices>
    FORCE_INLINE void open(uint32_t my_rank, size_t& connection_arg) {
        rank = my_rank;
#if defined(COMPILE_FOR_BRISC)
        range = line_devices - 1 - rank;  // forward: the ranks above
#else
        range = rank;  // backward: the ranks below
#endif
        has_peers = range != 0;
        if (has_peers) {
            connection =
                tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(connection_arg);
            connection.open();
        }
        PacketHeaderPool::reset();
        header = PacketHeaderPool::allocate_header();
    }

    FORCE_INLINE void close() {
        if (has_peers) {
            connection.close();
        }
    }

    // FACT 1: one packet per direction per counter per call, one increment on every peer's copy of the counter
    // (the peers' copies of this core hold the semaphore at the same address and the same virtual coordinates).
    FORCE_INLINE void arrive(uint32_t barrier_addr) {
        if (has_peers) {
            const Noc noc;
            noc.async_writes_flushed();
            fabric_multicast_noc_unicast_atomic_inc(
                &connection,
                header,
                tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
                    safe_get_noc_addr(my_x[0], my_y[0], barrier_addr, 0), 1, true},
                1,
                range);
        }
    }
};

// One gather phase after its arrive: the barrier wait and reset, the release, this core's tiles out, the data wait.
// close_connection: close this RISC's connection after its sends (the last phase of the kernel).
template <
    uint32_t scratch_cb,
    uint32_t tiles,
    uint32_t first_tile,
    uint32_t tile_step,
    uint32_t page_tile_stride,
    uint32_t page_rank_stride,
    uint32_t source,
    uint32_t line_devices,
    uint32_t go_semaphore_id,
    uint32_t scratch_semaphore_id,
    uint32_t done_semaphore_id,
    bool close_connection,
    typename OutArgs,
    typename LocalArgs>
FORCE_INLINE void transport_phase(
    LineSender& line,
    const OutArgs& out_args,
    const LocalArgs& local_args,
    uint32_t first_arg,
    uint32_t delay_after_reset) {
    constexpr uint32_t carried_tiles = first_tile < tiles ? (tiles - first_tile + tile_step - 1) / tile_step : 0;
    static_assert(carried_tiles > 0, "a transport core carries at least one tile");
    const uint32_t out_addr = get_arg_val<uint32_t>(first_arg + 0);
    const uint32_t local_addr = get_arg_val<uint32_t>(first_arg + 1);
    const uint32_t barrier_addr = get_arg_val<uint32_t>(first_arg + 2);
    const uint32_t data_addr = get_arg_val<uint32_t>(first_arg + 3);
    const uint32_t consumers = get_arg_val<uint32_t>(first_arg + 4);
    const uint32_t consumer_semaphore_id = get_arg_val<uint32_t>(first_arg + 5);
    const auto out = TensorAccessor(out_args, out_addr);
    const uint32_t tile_bytes = get_tile_size(scratch_cb);
    const Noc noc;
    DataflowBuffer scratch(scratch_cb);
    Semaphore<> go(go_semaphore_id);
    Semaphore<> scratch_ready(scratch_semaphore_id);
    Semaphore<> done(done_semaphore_id);
    // barrier_addr and data_addr are GlobalSemaphore addresses and Semaphore<> binds a program semaphore id, so their
    // waits and resets are noc_semaphore_wait_min / set, which take these volatile L1 pointers.
    volatile tt_l1_ptr uint32_t* barrier = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_addr);
    volatile tt_l1_ptr uint32_t* data = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(data_addr);
    const uint64_t data_noc_addr = safe_get_noc_addr(my_x[0], my_y[0], data_addr, 0);
    const uint32_t rank = line.rank;

#if defined(COMPILE_FOR_BRISC)
    // `barrier` is a GlobalSemaphore word (see above).
    noc_semaphore_wait_min(barrier, line_devices - 1);
    noc_semaphore_set(barrier, 0);  // FACT 2: reset before the release below; the sends follow the release
    if (delay_after_reset != 0) {
        riscv_wait(delay_after_reset);
    }
    go.set(1);
    uint32_t scratch_addr;
    if constexpr (source == 0) {
        scratch.wait_front(carried_tiles);
        scratch_addr = scratch.get_read_ptr();
    } else {
        scratch_ready.wait_min(carried_tiles);
        scratch_addr = scratch.get_write_ptr();
    }
#else
    uint32_t scratch_addr;
    if constexpr (source == 0) {
        const auto local = TensorAccessor(local_args, local_addr);
        scratch.reserve_back(carried_tiles);
        for (uint32_t i = 0; i < carried_tiles; ++i) {
            noc.async_read(
                local, scratch, tile_bytes, {.page_id = first_tile + i * tile_step}, {.offset_bytes = i * tile_bytes});
        }
        noc.async_read_barrier();
        scratch_addr = scratch.get_write_ptr();
        scratch.push_back(carried_tiles);
        for (uint32_t i = 0; i < carried_tiles; ++i) {
            noc.async_write(
                CoreLocalMem<uint32_t>(scratch_addr + i * tile_bytes),
                out,
                tile_bytes,
                {},
                {.page_id = (first_tile + i * tile_step) * page_tile_stride + rank * page_rank_stride});
        }
        noc.async_write_barrier();
    } else {
        scratch_addr = scratch.get_write_ptr();
    }
    go.wait(1);
    go.set(0);
    if constexpr (source != 0) {
        scratch_ready.wait_min(carried_tiles);
    }
#endif

    if (line.has_peers) {
        for (uint32_t i = 0; i < carried_tiles; ++i) {
            noc.async_writes_flushed();
            // The fabric write takes the page's raw NoC address.
            fabric_multicast_noc_fused_unicast_with_atomic_inc(
                &line.connection,
                line.header,
                scratch_addr + i * tile_bytes,
                tile_bytes,
                tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{
                    out.get_noc_addr((first_tile + i * tile_step) * page_tile_stride + rank * page_rank_stride, 0, 0),
                    data_noc_addr,
                    1,
                    true},
                1,
                line.range);
        }
    }
    if constexpr (close_connection) {
        line.close();
    }
#if defined(COMPILE_FOR_BRISC)
    // `data` is a GlobalSemaphore word (see above).
    noc_semaphore_wait_min(data, (line_devices - 1) * carried_tiles);  // FACT 3: every peer's every tile
    noc_semaphore_set(data, 0);
    for (uint32_t c = 0; c < consumers; ++c) {  // the gathered pages are complete: release this program's consumers
        Semaphore<> ready(consumer_semaphore_id);
        ready.up(
            noc,
            get_arg_val<uint32_t>(first_arg + phase_runtime_args + 2 * c),
            get_arg_val<uint32_t>(first_arg + phase_runtime_args + 1 + 2 * c),
            1);
    }
    noc.async_atomic_barrier();
    if constexpr (source == 0) {
        scratch.pop_front(carried_tiles);
    } else {
        done.wait(1);
        done.set(0);
        scratch_ready.set(0);
    }
#else
    if constexpr (source != 0) {
        done.set(1);
    }
#endif
}
