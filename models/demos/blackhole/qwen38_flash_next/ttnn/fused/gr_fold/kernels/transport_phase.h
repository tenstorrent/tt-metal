// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// The line transport of the fused GR read: a transport core per fabric link moves this device's tiles to the other
// devices of the TP_SIZE line over the 1D fabric from inside the program, so the chain's collective is not a program of
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
// ROW_MODE 4 (packed, no barrier; QWEN38_GR_NOBARRIER, NOBARRIER below): the packed mode without the barrier.  The
// landing slots are two sets (set = call parity) of a persistent L1 buffer (LAND_ADDR: a call counter word, then set 0,
// then set 1, RING slots each) and each parity has its own data semaphore (the runtime block's `data` = parity 0,
// DATA1_ADDR = parity 1).  The counter is read by both RISCs at the start of the phase and stepped by BRISC at the very
// end (after `done`, i.e. after NCRISC's read); it is never reset, so a trace replay continues the parity.  Safety: a
// device sends call n + 2 (parity of n) into a peer only after it received that peer's call n + 1 data, which the peer
// sends in its call n + 1 program, after its call n program (unpack, write barrier, consumers) ended; so a slot set and
// a data semaphore are reused only after the owner is done with them.  The destination pages are written only by their
// own device's unpack, in its own program; the persistent buffer and the global semaphores exist whatever program the
// peer runs when a packet lands.
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
    uint32_t ROWS = 32,
    uint32_t ROW_MODE = 0,
    uint32_t PACK_CB = 0,
    uint32_t SLOT_BYTES = 0,
    uint32_t LAND_ADDR = 0,
    uint32_t DATA1_ADDR = 0,
    typename OutArgs,
    typename LocalArgs>
FORCE_INLINE void transport_phase(
    LineSender& line, const OutArgs& out_args, const LocalArgs& local_args, uint32_t rt, uint32_t delay_after_reset) {
    constexpr uint32_t MY_TILES = TILE_FIRST < TILES ? (TILES - TILE_FIRST + TILE_STEP - 1) / TILE_STEP : 0;
    static_assert(MY_TILES > 0, "a transport core carries at least one tile");
    const uint32_t out_addr = get_arg_val<uint32_t>(rt + 0);
    const uint32_t local_addr = get_arg_val<uint32_t>(rt + 1);
    const uint32_t barrier_addr = get_arg_val<uint32_t>(rt + 2);
    constexpr bool NOBARRIER = ROW_MODE == 4;
    // NOBARRIER: the call parity of this phase (the counter is stepped after both RISCs read it, see the end)
    volatile tt_l1_ptr uint32_t* call_counter = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(LAND_ADDR);
    const uint32_t parity = NOBARRIER ? (*call_counter & 1) : 0;
    const uint32_t data_addr = (NOBARRIER && parity) ? DATA1_ADDR : get_arg_val<uint32_t>(rt + 3);
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

    // Row-trimmed send (ROW_MODE 1 span, 2 chunks; 0 or ROWS >= 32 = the whole tile): only rows [0, ROWS) of the tile
    // carry data.  Tile = 4 faces of 16x16 stored face after face, each face row-major.  The payload goes to the same
    // byte offset of the destination page as the full-tile send, so a consumer reads the same bytes; rows >= ROWS of
    // the destination are not rewritten.  Every chunk is a fused write + atomic inc, so the data wait counts packets.
    // Alignment: NOC_L1_WRITE_ALIGNMENT_BYTES / NOC_DRAM_WRITE_ALIGNMENT_BYTES = 16 on Blackhole
    // (tt_metal/hw/inc/internal/tt-1xx/blackhole/noc/noc_parameters.h:376,380); a row is 32 B (bf16) or 64 B (fp32), so
    // every offset and size below is a multiple of 16.
    constexpr bool TRIM = ROW_MODE != 0 && ROWS < 32;
    constexpr uint32_t FACE_ROWS = 16;
    // ROW_MODE 3 (packed): the valid chunks of all this core's tiles are copied into one staging buffer and go out as
    // ONE packet per direction into the peers' transport core L1 (landing slot rank * SLOT_BYTES of the PACK_CB
    // buffer: RING landing slots, then one staging slot per RISC); the peer's BRISC waits for one packet per source
    // device and unpacks every slot into the destination pages at the byte offsets of the full-tile send.
    constexpr bool PACKED = TRIM && (ROW_MODE == 3 || ROW_MODE == 4);
    constexpr uint32_t NCH = !TRIM ? 1 : (ROW_MODE == 1 ? 1 : (ROWS <= FACE_ROWS ? 2 : 3));
    constexpr uint32_t PKTS = PACKED ? 1 : NCH;
    const uint32_t face_bytes = tile_bytes / 4;
    const uint32_t row_bytes = face_bytes / FACE_ROWS;  // one row of one face
    uint32_t chunk_off[3] = {0, 0, 0};
    uint32_t chunk_len[3] = {tile_bytes, 0, 0};
    if constexpr (TRIM) {
        if constexpr (ROW_MODE == 1) {
            chunk_len[0] =
                ROWS <= FACE_ROWS ? face_bytes + ROWS * row_bytes : 3 * face_bytes + (ROWS - FACE_ROWS) * row_bytes;
        } else if constexpr (ROWS <= FACE_ROWS) {  // modes 2 and 3
            chunk_len[0] = ROWS * row_bytes;       // face 0 rows [0, ROWS)
            chunk_off[1] = face_bytes;
            chunk_len[1] = ROWS * row_bytes;  // face 1 rows [0, ROWS)
        } else {
            chunk_len[0] = 2 * face_bytes;  // faces 0 and 1, contiguous
            chunk_off[1] = 2 * face_bytes;
            chunk_len[1] = (ROWS - FACE_ROWS) * row_bytes;  // face 2 rows [0, ROWS - 16)
            chunk_off[2] = 3 * face_bytes;
            chunk_len[2] = (ROWS - FACE_ROWS) * row_bytes;  // face 3 rows [0, ROWS - 16)
        }
    }

    uint32_t pack_off[3] = {0, 0, 0};  // chunk k's offset inside one packed tile
    uint32_t pack_tile_bytes = tile_bytes;
    uint32_t pack_base = 0;
    if constexpr (PACKED) {
        pack_tile_bytes = 0;
        for (uint32_t k = 0; k < NCH; ++k) {
            pack_off[k] = pack_tile_bytes;
            pack_tile_bytes += chunk_len[k];
        }
        DataflowBuffer pack(PACK_CB);
        pack_base = pack.get_write_ptr();
    }
    // where the landing slots of this call are: mode 3 the CB buffer's first RING slots, mode 4 the persistent buffer's
    // slot set of this call's parity
    const uint32_t land_base = NOBARRIER ? LAND_ADDR + 16 + parity * RING * SLOT_BYTES : pack_base;

#if defined(COMPILE_FOR_BRISC)
    if constexpr (!NOBARRIER) {
        noc_semaphore_wait_min(barrier, RING - 1);
        noc_semaphore_set(barrier, 0);  // FACT 2: reset before the release below; the sends follow the release
    }
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

    if constexpr (PACKED) {
        if (line.has_peers) {
#if defined(COMPILE_FOR_BRISC)
            const uint32_t stage = pack_base + (NOBARRIER ? 0 : RING) * SLOT_BYTES;
#else
            const uint32_t stage = pack_base + (NOBARRIER ? 1 : RING + 1) * SLOT_BYTES;
#endif
            for (uint32_t i = 0; i < MY_TILES; ++i) {
                for (uint32_t k = 0; k < NCH; ++k) {
                    noc_async_write(
                        scratch_addr + i * tile_bytes + chunk_off[k],
                        safe_get_noc_addr(
                            my_x[noc_index], my_y[noc_index], stage + i * pack_tile_bytes + pack_off[k], noc_index),
                        chunk_len[k]);
                }
            }
            noc_async_write_barrier();
            noc_async_writes_flushed();
            fabric_multicast_noc_fused_unicast_with_atomic_inc(
                &line.connection,
                line.header,
                stage,
                MY_TILES * pack_tile_bytes,
                tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{
                    safe_get_noc_addr(my_x[0], my_y[0], land_base + rank * SLOT_BYTES, 0), data_noc, 1, true},
                1,
                line.range);
        }
    } else if (line.has_peers) {
        for (uint32_t i = 0; i < MY_TILES; ++i) {
            const uint32_t page = (TILE_FIRST + i * TILE_STEP) * PAGE_TILE_STRIDE + rank * PAGE_RANK_STRIDE;
            for (uint32_t k = 0; k < PKTS; ++k) {
                noc_async_writes_flushed();
                fabric_multicast_noc_fused_unicast_with_atomic_inc(
                    &line.connection,
                    line.header,
                    scratch_addr + i * tile_bytes + chunk_off[k],
                    chunk_len[k],
                    tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{
                        out.get_noc_addr(page, chunk_off[k], 0), data_noc, 1, true},
                    1,
                    line.range);
            }
        }
    }
    if constexpr (CLOSE) {
        line.close();
    }
#if defined(COMPILE_FOR_BRISC)
    noc_semaphore_wait_min(data, (RING - 1) * (PACKED ? 1 : MY_TILES * PKTS));  // FACT 3: every peer's every packet
    noc_semaphore_set(data, 0);
    if constexpr (PACKED) {  // unpack every source slot into the destination pages, then release the consumers
        for (uint32_t r = 0; r < RING; ++r) {
            if (r == rank) {
                continue;
            }
            for (uint32_t i = 0; i < MY_TILES; ++i) {
                const uint32_t page = (TILE_FIRST + i * TILE_STEP) * PAGE_TILE_STRIDE + r * PAGE_RANK_STRIDE;
                for (uint32_t k = 0; k < NCH; ++k) {
                    noc_async_write(
                        land_base + r * SLOT_BYTES + i * pack_tile_bytes + pack_off[k],
                        out.get_noc_addr(page, chunk_off[k]),
                        chunk_len[k]);
                }
            }
        }
        noc_async_write_barrier();
    }
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
    if constexpr (NOBARRIER) {
        *call_counter = *call_counter + 1;  // after NCRISC's read (its done.set(1) follows it): the next call's parity
    }
#else
    if constexpr (SOURCE != 0) {
        done.set(1);
    }
#endif
}
