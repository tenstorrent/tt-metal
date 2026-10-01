// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// The line transport of the fused GR read: a transport core per fabric link moves this device's tiles to the other
// devices of the TP4 line over the 1D fabric from inside the program, so the chain's collective is not a program of
// its own.  A phase is the all-gather of TILES local tiles (the stats: one per branch; the partials: one per down
// worker) into the gathered tensor at page t * PAGE_TILE_STRIDE + rank * PAGE_RANK_STRIDE, exactly the pages the
// chain's all-gather writes; this core carries the tiles t = TILE_FIRST + i * TILE_STEP (one core per link, the tiles
// dealt round robin).  SOURCE selects where they come from: 0 = a local tensor this kernel reads (the standalone
// gather of the smoke tool; NCRISC also copies them into the device's own pages), 1 = producer cores of the same
// program write them into this core's scratch CB and into the device's own pages themselves and raise the scratch
// semaphore once per tile.  transport.cpp runs one phase; transport2.cpp two phases in sequence on the same
// connection (the fused read's front: the stats phase, then the partials phase), because a fabric link's worker
// sender channel is one per direction (fabric.cpp: sender_channel 0; the open handshake does not queue), so one
// program keeps at most one open sender per link per direction: the design note's link budget.
//
// Both data-movement RISCs run these bodies.  NCRISC (reader config, NOC_0) sends backward (the ranks below); BRISC
// (writer config, NOC_1) sends forward (the ranks above) and owns the waits.  Both connections are opened at kernel
// start (LineSender::open) and closed after this RISC's last send.  Cross-device protocol per phase per call (the
// design note walks the interleavings; the three facts it stands on are marked FACT below):
//   1. barrier (LineSender::arrive): every device raises the phase's barrier semaphore of every peer by one (one line
//      multicast per direction: FACT 1, exactly RING - 1 arrivals per counter per call, every peer once); BRISC waits
//      for RING - 1, resets the counter with a plain L1 store and only then releases NCRISC through the program-local
//      `go` semaphore (FACT 2: the reset is program-ordered on BRISC before the release, and every data send of either
//      RISC is program-ordered after it: BRISC's by program order, NCRISC's by its `go` wait).
//   2. data: each tile goes out as one fused write + atomic-increment packet (the payload lands in the peer's tensor
//      page, then the peer's data semaphore steps by one); BRISC waits for (RING - 1) x this core's tiles (FACT 3: the
//      full count from all peers, so no device leaves a phase with partial data) and resets.
// The barrier and data semaphores are global semaphores (one address on every device), a pair per phase.  No launch
// resets a semaphore, so every semaphore raised here is reset by its owner before the program ends: BRISC resets the
// barrier and data counters; NCRISC resets `go`; with SOURCE 1, NCRISC raises `done` once its sends are issued and
// BRISC, last, resets `done` and the producers' scratch semaphore.  A packet header is rewritten only after
// noc_async_writes_flushed().  Two phases on one core reuse `go` and `done`: BRISC's phase-B `go.set(1)` follows its
// phase-A `done.wait(1)`, which follows NCRISC's phase-A `done.set(1)`, which follows NCRISC's phase-A `go.set(0)` in
// NCRISC program order, so a phase's release is never mistaken for the previous phase's.  Both arrives of a two-phase
// kernel go out at kernel start (the phase-B wait comes after phase A); the barrier argument holds per counter.
// Consumers: once a phase's gathered pages are complete (after the data wait and its reset) BRISC raises the
// consumer semaphore of every listed consumer core by one, so consumers of the same program may read the pages.
// The two delay arguments (cycles) exist for the skew soak: a spin before the barrier arrive, a spin between the reset
// and the release of the sends; both 0 in the model.
//
// Runtime block of a phase at `rt`: 0 gathered tensor address, 1 local tensor address, 2 barrier semaphore address,
// 3 data semaphore address, 4 consumer count N, 5 consumer semaphore id, 6..6+2N the consumers' NoC (x, y).

#pragma once

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/linear/api.h"

using namespace tt::tt_fabric::linear::experimental;

constexpr uint32_t PHASE_RT_ARGS = 6;  // before the consumers' coordinates

// This RISC's connection along the line: BRISC forward (the ranks above), NCRISC backward (the ranks below).
struct LineSender {
    tt::tt_fabric::WorkerToFabricEdmSender connection;
    volatile PACKET_HEADER_TYPE* header;
    uint32_t rank;
    uint32_t range;
    bool has_peers;

    template <uint32_t RING>
    FORCE_INLINE void open(uint32_t my_rank, size_t& arg_idx) {
        rank = my_rank;
#if defined(COMPILE_FOR_BRISC)
        range = RING - 1 - rank;  // forward: the ranks above
#else
        range = rank;  // backward: the ranks below
#endif
        has_peers = range != 0;
        if (has_peers) {
            connection = tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(arg_idx);
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
            noc_async_writes_flushed();
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
// CLOSE: close this RISC's connection after its sends (the last phase of the kernel).
template <
    uint32_t SCRATCH_CB,
    uint32_t TILES,
    uint32_t TILE_FIRST,
    uint32_t TILE_STEP,
    uint32_t PAGE_TILE_STRIDE,
    uint32_t PAGE_RANK_STRIDE,
    uint32_t SOURCE,
    uint32_t RING,
    uint32_t SEM_GO,
    uint32_t SEM_SCRATCH,
    uint32_t SEM_DONE,
    bool CLOSE,
    typename OutArgs,
    typename LocalArgs>
FORCE_INLINE void transport_phase(
    LineSender& line, const OutArgs& out_args, const LocalArgs& local_args, uint32_t rt, uint32_t delay_after_reset) {
    constexpr uint32_t MY_TILES = TILE_FIRST < TILES ? (TILES - TILE_FIRST + TILE_STEP - 1) / TILE_STEP : 0;
    static_assert(MY_TILES > 0, "a transport core carries at least one tile");
    const uint32_t out_addr = get_arg_val<uint32_t>(rt + 0);
    const uint32_t local_addr = get_arg_val<uint32_t>(rt + 1);
    const uint32_t barrier_addr = get_arg_val<uint32_t>(rt + 2);
    const uint32_t data_addr = get_arg_val<uint32_t>(rt + 3);
    const uint32_t consumers = get_arg_val<uint32_t>(rt + 4);
    const uint32_t consumer_sem = get_arg_val<uint32_t>(rt + 5);
    const auto out = TensorAccessor(out_args, out_addr);
    const uint32_t tile_bytes = get_tile_size(SCRATCH_CB);
    Noc noc;
    DataflowBuffer scratch(SCRATCH_CB);
    Semaphore<> go(SEM_GO);
    Semaphore<> scratch_ready(SEM_SCRATCH);
    Semaphore<> done(SEM_DONE);
    volatile tt_l1_ptr uint32_t* barrier = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_addr);
    volatile tt_l1_ptr uint32_t* data = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(data_addr);
    const uint64_t data_noc = safe_get_noc_addr(my_x[0], my_y[0], data_addr, 0);
    const uint32_t rank = line.rank;

#if defined(COMPILE_FOR_BRISC)
    noc_semaphore_wait_min(barrier, RING - 1);
    noc_semaphore_set(barrier, 0);  // FACT 2: reset before the release below; the sends follow the release
    if (delay_after_reset != 0) {
        riscv_wait(delay_after_reset);
    }
    go.set(1);
    uint32_t scratch_addr;
    if constexpr (SOURCE == 0) {
        scratch.wait_front(MY_TILES);
        scratch_addr = scratch.get_read_ptr();
    } else {
        scratch_ready.wait_min(MY_TILES);
        scratch_addr = scratch.get_write_ptr();
    }
#else
    uint32_t scratch_addr;
    if constexpr (SOURCE == 0) {
        const auto local = TensorAccessor(local_args, local_addr);
        scratch.reserve_back(MY_TILES);
        for (uint32_t i = 0; i < MY_TILES; ++i) {
            noc.async_read(
                local, scratch, tile_bytes, {.page_id = TILE_FIRST + i * TILE_STEP}, {.offset_bytes = i * tile_bytes});
        }
        noc.async_read_barrier();
        scratch_addr = scratch.get_write_ptr();
        scratch.push_back(MY_TILES);
        for (uint32_t i = 0; i < MY_TILES; ++i) {
            noc.async_write(
                CoreLocalMem<uint32_t>(scratch_addr + i * tile_bytes),
                out,
                tile_bytes,
                {},
                {.page_id = (TILE_FIRST + i * TILE_STEP) * PAGE_TILE_STRIDE + rank * PAGE_RANK_STRIDE});
        }
        noc.async_write_barrier();
    } else {
        scratch_addr = scratch.get_write_ptr();
    }
    go.wait(1);
    go.set(0);
    if constexpr (SOURCE != 0) {
        scratch_ready.wait_min(MY_TILES);
    }
#endif

    if (line.has_peers) {
        for (uint32_t i = 0; i < MY_TILES; ++i) {
            noc_async_writes_flushed();
            fabric_multicast_noc_fused_unicast_with_atomic_inc(
                &line.connection,
                line.header,
                scratch_addr + i * tile_bytes,
                tile_bytes,
                tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{
                    out.get_noc_addr((TILE_FIRST + i * TILE_STEP) * PAGE_TILE_STRIDE + rank * PAGE_RANK_STRIDE, 0, 0),
                    data_noc,
                    1,
                    true},
                1,
                line.range);
        }
    }
    if constexpr (CLOSE) {
        line.close();
    }
#if defined(COMPILE_FOR_BRISC)
    noc_semaphore_wait_min(data, (RING - 1) * MY_TILES);  // FACT 3: every peer's every tile
    noc_semaphore_set(data, 0);
    for (uint32_t c = 0; c < consumers; ++c) {  // the gathered pages are complete: release this program's consumers
        Semaphore<> ready(consumer_sem);
        ready.up(
            noc,
            get_arg_val<uint32_t>(rt + PHASE_RT_ARGS + 2 * c),
            get_arg_val<uint32_t>(rt + PHASE_RT_ARGS + 1 + 2 * c),
            1);
    }
    noc.async_atomic_barrier();
    if constexpr (SOURCE == 0) {
        scratch.pop_front(MY_TILES);
    } else {
        done.wait(1);
        done.set(0);
        scratch_ready.set(0);
    }
#else
    if constexpr (SOURCE != 0) {
        done.set(1);
    }
#endif
}
