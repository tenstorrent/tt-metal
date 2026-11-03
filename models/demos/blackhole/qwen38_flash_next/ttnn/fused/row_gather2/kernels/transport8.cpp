// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// The packed row all-gather of the per-block hidden for 3..8 valid rows (row_gather2, rows mode).  The one persistent
// landing buffer of transport.cpp holds the packed rows of at most 2 rows, so this mode lands its packets in the
// program's own CB (transient, like row_gather v1) and keeps v1's per-call barrier for that reason.  Per call, per
// transport core (one per link, the tiles dealt round robin: tile = TILE_FIRST + i * TILE_STEP), per RISC:
//   1. reads the valid rows [0, ROWS) of faces 0 and 1 of the core's tiles into a staging slot (one chunk per face,
//   each
//      chunk on a 64 B pitch: DRAM-read aligned), and clears its zero block,
//   2. raises the barrier semaphore of every peer (LineSender::arrive, as transport_phase.h FACT 1),
//   3. while the barrier is in flight, writes the rows >= ROWS of every output page this RISC owns as zeros (face 0
//      rows [ROWS, 16), then face 1 rows [ROWS, 16) and faces 2 and 3 as one write): the gathered tile has exact
//      zeros past the live rows whatever the input held there,
//   4. BRISC waits for RING - 1 arrivals, resets the counter and releases NCRISC (program-local `go`),
//   5. sends the staging slot as ONE fused write + atomic increment per direction into slot `rank` of the peers' CB
//      landing area (the peers' barrier is complete, so their program runs and the CB is theirs),
//   6. waits for one packet from every peer and writes the live rows of every source into the gathered tensor's tile
//      pages at the byte offsets a full-tile write uses; the two RISCs share the sources (BRISC also takes this
//      device's own chunks), and BRISC resets the data semaphore after NCRISC's `done`.
// The live rows and the zero rows of a page are disjoint byte ranges written by the same RISC.
// CB layout (PACK_CB, one raw buffer): RING landing slots | BRISC staging slot | NCRISC staging slot | BRISC zero block
// | NCRISC zero block (ZERO_BYTES each). Compile-time args: 0 PACK_CB, 1 TILES, 2 go semaphore id, 3 RING, 6 done
// semaphore id, 7 TILE_FIRST, 8 TILE_STEP,
//   11 ROWS, 12 TILE_BYTES, 13.. TensorAccessorArgs of the gathered tensor, then of the shard.
// Runtime args as gr_fold/kernels/transport.cpp (phase block at 3: out address, shard address, barrier address, data
// address, consumer count).

#include "../../gr_fold/kernels/transport_phase.h"
#include "../../kernels/zones.h"

constexpr uint32_t PACK_CB = get_compile_time_arg_val(0);
constexpr uint32_t TILES = get_compile_time_arg_val(1);
constexpr uint32_t SEM_GO = get_compile_time_arg_val(2);
constexpr uint32_t RING = get_compile_time_arg_val(3);
constexpr uint32_t SEM_DONE = get_compile_time_arg_val(6);
constexpr uint32_t TILE_FIRST = get_compile_time_arg_val(7);
constexpr uint32_t TILE_STEP = get_compile_time_arg_val(8);
constexpr uint32_t ROWS = get_compile_time_arg_val(11);
constexpr uint32_t TILE_BYTES = get_compile_time_arg_val(12);
constexpr uint32_t ACCESSOR_BASE = 13;
constexpr uint32_t PHASE_RT = 3;
constexpr uint32_t MY_TILES = (TILES - TILE_FIRST + TILE_STEP - 1) / TILE_STEP;
constexpr uint32_t FACE_BYTES = TILE_BYTES / 4;
constexpr uint32_t ROW_BYTES = FACE_BYTES / 16;
constexpr uint32_t CHUNK_LEN = ROWS * ROW_BYTES;                // rows [0, ROWS) of a face
constexpr uint32_t CHUNK_PITCH = ((CHUNK_LEN + 63) / 64) * 64;  // DRAM-read aligned staging pitch
constexpr uint32_t SLOT_BYTES = MY_TILES * 2 * CHUNK_PITCH;     // 2 chunks per tile
constexpr uint32_t ZERO_BYTES = TILE_BYTES - FACE_BYTES;        // the longest zero write (face 1 tail + faces 2, 3)
constexpr uint32_t ZERO_A_LEN = FACE_BYTES - CHUNK_LEN;         // face 0 rows [ROWS, 16)
constexpr uint32_t ZERO_B_OFF = FACE_BYTES + CHUNK_LEN;         // face 1 rows [ROWS, 16) .. end of the tile
constexpr uint32_t ZERO_B_LEN = TILE_BYTES - ZERO_B_OFF;
static_assert(ROWS >= 3 && ROWS <= 8, "rows 3..8 (bf16)");
static_assert(MY_TILES > 0, "a transport core carries at least one tile");
static_assert(ZERO_A_LEN <= ZERO_BYTES && ZERO_B_LEN <= ZERO_BYTES, "zero block covers the tails");

void kernel_main() {
    const uint32_t rank = get_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_arg_val<uint32_t>(PHASE_RT + 0);
    const uint32_t local_addr = get_arg_val<uint32_t>(PHASE_RT + 1);
    const uint32_t barrier_addr = get_arg_val<uint32_t>(PHASE_RT + 2);
    const uint32_t data_addr = get_arg_val<uint32_t>(PHASE_RT + 3);
    size_t arg_idx = PHASE_RT + PHASE_RT_ARGS + 2 * get_arg_val<uint32_t>(PHASE_RT + 4);
    constexpr auto out_args = TensorAccessorArgs<ACCESSOR_BASE>();
    constexpr auto local_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    const auto out = TensorAccessor(out_args, out_addr);
    const auto local = TensorAccessor(local_args, local_addr);
    LineSender line;
    line.open<RING>(rank, arg_idx);
    {
        FUSED_ZONE("fz_rg2_r8_main");
        Semaphore<> go(SEM_GO);
        Semaphore<> done(SEM_DONE);
        DataflowBuffer pack(PACK_CB);
        volatile tt_l1_ptr uint32_t* barrier = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_addr);
        volatile tt_l1_ptr uint32_t* data = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(data_addr);
        const uint64_t data_noc = safe_get_noc_addr(my_x[0], my_y[0], data_addr, 0);
        const uint32_t land = pack.get_write_ptr();
#if defined(COMPILE_FOR_BRISC)
        const uint32_t stage = land + RING * SLOT_BYTES;
        const uint32_t zero = land + (RING + 2) * SLOT_BYTES;
        const bool stage_needed = true;  // the device's own chunks are unpacked from it
#else
        const uint32_t stage = land + (RING + 1) * SLOT_BYTES;
        const uint32_t zero = land + (RING + 2) * SLOT_BYTES + ZERO_BYTES;
        const bool stage_needed = line.has_peers;
#endif
        // the sources this RISC unpacks, in order, every other peer for NCRISC; BRISC also takes the device's own
        uint32_t mine_mask = 0;
        {
            uint32_t k = 0;
            for (uint32_t s = 0; s < RING; ++s) {
                const bool own = s == rank;
#if defined(COMPILE_FOR_BRISC)
                const bool mine = own || (k++ & 1);
#else
                const bool mine = !own && !(k++ & 1);
#endif
                if (mine) {
                    mine_mask |= 1u << s;
                }
            }
        }
        {
            FUSED_ZONE("fz_rg2_r8_stage");
            if (stage_needed) {
                for (uint32_t i = 0; i < MY_TILES; ++i) {
                    const uint32_t t = TILE_FIRST + i * TILE_STEP;
                    noc_async_read(local.get_noc_addr(t, 0), stage + (2 * i) * CHUNK_PITCH, CHUNK_LEN);
                    noc_async_read(local.get_noc_addr(t, FACE_BYTES), stage + (2 * i + 1) * CHUNK_PITCH, CHUNK_LEN);
                }
            }
            volatile tt_l1_ptr uint32_t* z = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(zero);
            for (uint32_t w = 0; w < ZERO_BYTES / 4; ++w) {
                z[w] = 0;
            }
            if (stage_needed) {
                noc_async_read_barrier();
            }
        }
        line.arrive(barrier_addr);
        {
            FUSED_ZONE("fz_rg2_r8_zero");
            for (uint32_t s = 0; s < RING; ++s) {
                if (!((mine_mask >> s) & 1u)) {
                    continue;
                }
                for (uint32_t i = 0; i < MY_TILES; ++i) {
                    const uint32_t page = TILE_FIRST + i * TILE_STEP + TILES * s;
                    noc_async_write(zero, out.get_noc_addr(page, CHUNK_LEN), ZERO_A_LEN);
                    noc_async_write(zero, out.get_noc_addr(page, ZERO_B_OFF), ZERO_B_LEN);
                }
            }
        }
#if defined(COMPILE_FOR_BRISC)
        noc_semaphore_wait_min(barrier, RING - 1);
        noc_semaphore_set(barrier, 0);  // before the release: every send of either RISC follows it
        go.set(1);
#else
        go.wait(1);
        go.set(0);
#endif
        if (line.has_peers) {
            noc_async_writes_flushed();
            fabric_multicast_noc_fused_unicast_with_atomic_inc(
                &line.connection,
                line.header,
                stage,
                SLOT_BYTES,
                tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{
                    safe_get_noc_addr(my_x[0], my_y[0], land + rank * SLOT_BYTES, 0), data_noc, 1, true},
                1,
                line.range);
        }
        line.close();
        {
            FUSED_ZONE("fz_rg2_r8_land");
            noc_semaphore_wait_min(data, RING - 1);  // one packet from every peer
        }
        {
            FUSED_ZONE("fz_rg2_r8_unpack");
            for (uint32_t s = 0; s < RING; ++s) {
                if (!((mine_mask >> s) & 1u)) {
                    continue;
                }
                const uint32_t slot = s == rank ? stage : land + s * SLOT_BYTES;
                for (uint32_t i = 0; i < MY_TILES; ++i) {
                    const uint32_t page = TILE_FIRST + i * TILE_STEP + TILES * s;
                    noc_async_write(slot + (2 * i) * CHUNK_PITCH, out.get_noc_addr(page, 0), CHUNK_LEN);
                    noc_async_write(slot + (2 * i + 1) * CHUNK_PITCH, out.get_noc_addr(page, FACE_BYTES), CHUNK_LEN);
                }
            }
            noc_async_write_barrier();
        }
#if defined(COMPILE_FOR_BRISC)
        done.wait(1);
        done.set(0);
        noc_semaphore_set(data, 0);
#else
        done.set(1);
#endif
    }
    noc_async_full_barrier();
}
