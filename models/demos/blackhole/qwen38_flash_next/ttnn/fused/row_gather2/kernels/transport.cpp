// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// The barrier-free packed row all-gather of the per-block hidden (row_gather2).  Per call every device
//   1. reads the valid rows of its shard's TILES tiles (rows [0, ROWS) of faces 0 and 1, one 64 B-pitch chunk each:
//      aligned for a DRAM source, no bounce) into a staging slot (both RISCs, each its own copy),
//   2. multicasts the staging slot as ONE fused write + atomic increment per direction (BRISC forward, NCRISC backward)
//      into its slot of the peers' persistent landing buffer (the set of this call's parity) and the peers' data
//      semaphore of that parity (no barrier: the landing buffer and the two parity semaphores outlive every program,
//      the safety argument of gr_fold/kernels/transport_phase.h ROW_MODE 4: a device sends call n + 2 into a peer only
//      after it received that peer's call n + 1 data, which the peer sends after its call n program ended),
//   3. waits for one packet from every peer and writes every source's chunks into the gathered tensor's tile pages
//      (page = tile + TILES * source) at the byte offsets a full-tile write uses; the two RISCs share the sources.
// Rows >= ROWS of the gathered tiles are not written.  The single call counter word of the landing buffer is shared by
// every call of this op (one buffer, one counter, one semaphore pair per mesh), stepped by BRISC at the very end.
// Compile-time args: 0 PACK_CB, 1 TILES, 2 go sem (unused), 3 RING, 4-5 unused, 6 done semaphore id, 7-10 unused, 11
// ROWS,
//   12 LAND_ADDR, 13 DATA1_ADDR, 14 TILE_BYTES, 15.. TensorAccessorArgs of the gathered tensor, then of the shard.
// Runtime args as gr_fold/kernels/transport.cpp (phase block at 3: out address, shard address, unused, data0 address).

#include "../../gr_fold/kernels/transport_phase.h"
#include "../../kernels/zones.h"

constexpr uint32_t PACK_CB = get_compile_time_arg_val(0);
constexpr uint32_t TILES = get_compile_time_arg_val(1);
constexpr uint32_t RING = get_compile_time_arg_val(3);
constexpr uint32_t SEM_DONE = get_compile_time_arg_val(6);
constexpr uint32_t ROWS = get_compile_time_arg_val(11);
constexpr uint32_t LAND_ADDR = get_compile_time_arg_val(12);
constexpr uint32_t DATA1_ADDR = get_compile_time_arg_val(13);
constexpr uint32_t TILE_BYTES = get_compile_time_arg_val(14);
constexpr uint32_t ACCESSOR_BASE = 15;
constexpr uint32_t PHASE_RT = 3;
constexpr uint32_t CHUNK_PITCH = 64;                      // one chunk (<= 64 B) per 64 B: DRAM-read aligned
constexpr uint32_t SLOT_BYTES = TILES * 2 * CHUNK_PITCH;  // 2 chunks per tile
constexpr uint32_t FACE_BYTES = TILE_BYTES / 4;
constexpr uint32_t CHUNK_LEN = ROWS * (FACE_BYTES / 16);  // rows [0, ROWS) of a face
static_assert(CHUNK_LEN <= CHUNK_PITCH, "rows <= 2 (bf16)");

void kernel_main() {
    const uint32_t rank = get_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_arg_val<uint32_t>(PHASE_RT + 0);
    const uint32_t local_addr = get_arg_val<uint32_t>(PHASE_RT + 1);
    const uint32_t data0_addr = get_arg_val<uint32_t>(PHASE_RT + 3);
    size_t arg_idx = PHASE_RT + PHASE_RT_ARGS + 2 * get_arg_val<uint32_t>(PHASE_RT + 4);
    constexpr auto out_args = TensorAccessorArgs<ACCESSOR_BASE>();
    constexpr auto local_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    const auto out = TensorAccessor(out_args, out_addr);
    const auto local = TensorAccessor(local_args, local_addr);
    LineSender line;
    line.open<RING>(rank, arg_idx);
    {
        FUSED_ZONE("fz_rg2_main");
        Semaphore<> done(SEM_DONE);
        DataflowBuffer pack(PACK_CB);
        volatile tt_l1_ptr uint32_t* counter = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(LAND_ADDR);
        const uint32_t parity = *counter & 1;
        const uint32_t data_addr = parity ? DATA1_ADDR : data0_addr;
        const uint32_t land = LAND_ADDR + 16 + parity * RING * SLOT_BYTES;
        volatile tt_l1_ptr uint32_t* data = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(data_addr);
        const uint64_t data_noc = safe_get_noc_addr(my_x[0], my_y[0], data_addr, 0);
        const uint32_t pack_base = pack.get_write_ptr();
#if defined(COMPILE_FOR_BRISC)
        const uint32_t stage = pack_base;
        const bool stage_needed = true;  // the device's own chunks are unpacked from it
#else
        const uint32_t stage = pack_base + SLOT_BYTES;
        const bool stage_needed = line.has_peers;
#endif
        {
            FUSED_ZONE("fz_rg2_stage");
            if (stage_needed) {
                for (uint32_t t = 0; t < TILES; ++t) {
                    noc_async_read(local.get_noc_addr(t, 0), stage + (2 * t) * CHUNK_PITCH, CHUNK_LEN);
                    noc_async_read(local.get_noc_addr(t, FACE_BYTES), stage + (2 * t + 1) * CHUNK_PITCH, CHUNK_LEN);
                }
                noc_async_read_barrier();
            }
        }
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
            FUSED_ZONE("fz_rg2_land");
            noc_semaphore_wait_min(data, RING - 1);  // one packet from every peer
        }
        {
            FUSED_ZONE("fz_rg2_unpack");
            // the sources in order, every other one for NCRISC; BRISC also takes the device's own chunks
            uint32_t k = 0;
            for (uint32_t s = 0; s < RING; ++s) {
                const bool own = s == rank;
#if defined(COMPILE_FOR_BRISC)
                const bool mine = own || (k++ & 1);
#else
                const bool mine = !own && !(k++ & 1);
#endif
                if (!mine) {
                    continue;
                }
                const uint32_t slot = own ? stage : land + s * SLOT_BYTES;
                for (uint32_t t = 0; t < TILES; ++t) {
                    const uint32_t page = t + TILES * s;
                    noc_async_write(slot + (2 * t) * CHUNK_PITCH, out.get_noc_addr(page, 0), CHUNK_LEN);
                    noc_async_write(slot + (2 * t + 1) * CHUNK_PITCH, out.get_noc_addr(page, FACE_BYTES), CHUNK_LEN);
                }
            }
            noc_async_write_barrier();
        }
#if defined(COMPILE_FOR_BRISC)
        done.wait(1);
        done.set(0);
        noc_semaphore_set(data, 0);
        *counter = *counter + 1;  // after NCRISC's read of it (its done.set follows): the next call's parity
#else
        done.set(1);
#endif
    }
    noc_async_full_barrier();
}
