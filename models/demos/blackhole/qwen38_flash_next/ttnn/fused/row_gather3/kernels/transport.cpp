// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// row_gather2's barrier-free packed gather transport (row_gather2/kernels/transport.cpp: same landing buffer, parity
// slots, one packet per direction, split unpack) as the extra kernels of the GR gate program (QWEN38_GATE_GATHER=1):
// the valid-row chunks are not read from DRAM but written into a persistent staging slot (STAGE_ADDR, chunk 2 j + k at
// (2 j + k) * 64) by the TILES gate worker cores, each stepping the arrival semaphore SEM_ARRIVE on this core when its
// chunks have landed; both RISCs wait for TILES arrivals, then send / unpack exactly as row_gather2.  BRISC resets the
// arrival semaphore after NCRISC's `done`. Compile-time args: 1 TILES, 3 RING, 6 done semaphore id, 11 ROWS, 12
// LAND_ADDR, 13 DATA1_ADDR, 14 TILE_BYTES, 15 STAGE_ADDR,
//   16 SEM_ARRIVE, 17.. TensorAccessorArgs of the gathered tensor, then of the (unused) local tensor.
// Runtime args as row_gather2 (phase block at 3: out address, unused, unused, data parity-0 address).

#include "../../gr_fold/kernels/transport_phase.h"
#include "../../kernels/zones.h"

constexpr uint32_t TILES = get_compile_time_arg_val(1);
constexpr uint32_t RING = get_compile_time_arg_val(3);
constexpr uint32_t SEM_DONE = get_compile_time_arg_val(6);
constexpr uint32_t ROWS = get_compile_time_arg_val(11);
constexpr uint32_t LAND_ADDR = get_compile_time_arg_val(12);
constexpr uint32_t DATA1_ADDR = get_compile_time_arg_val(13);
constexpr uint32_t TILE_BYTES = get_compile_time_arg_val(14);
constexpr uint32_t STAGE_ADDR = get_compile_time_arg_val(15);
constexpr uint32_t SEM_ARRIVE = get_compile_time_arg_val(16);
constexpr uint32_t ACCESSOR_BASE = 17;
constexpr uint32_t PHASE_RT = 3;
constexpr uint32_t CHUNK_PITCH = 64;
constexpr uint32_t SLOT_BYTES = TILES * 2 * CHUNK_PITCH;
constexpr uint32_t FACE_BYTES = TILE_BYTES / 4;
constexpr uint32_t CHUNK_LEN = ROWS * (FACE_BYTES / 16);
static_assert(CHUNK_LEN <= CHUNK_PITCH, "rows <= 2 (bf16)");

void kernel_main() {
    const uint32_t rank = get_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_arg_val<uint32_t>(PHASE_RT + 0);
    const uint32_t data0_addr = get_arg_val<uint32_t>(PHASE_RT + 3);
    size_t arg_idx = PHASE_RT + PHASE_RT_ARGS + 2 * get_arg_val<uint32_t>(PHASE_RT + 4);
    constexpr auto out_args = TensorAccessorArgs<ACCESSOR_BASE>();
    const auto out = TensorAccessor(out_args, out_addr);
    LineSender line;
    line.open<RING>(rank, arg_idx);
    {
        FUSED_ZONE("fz_rg3_main");
        Semaphore<> done(SEM_DONE);
        Semaphore<> arrive(SEM_ARRIVE);
        volatile tt_l1_ptr uint32_t* counter = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(LAND_ADDR);
        const uint32_t parity = *counter & 1;
        const uint32_t data_addr = parity ? DATA1_ADDR : data0_addr;
        const uint32_t land = LAND_ADDR + 16 + parity * RING * SLOT_BYTES;
        volatile tt_l1_ptr uint32_t* data = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(data_addr);
        const uint64_t data_noc = safe_get_noc_addr(my_x[0], my_y[0], data_addr, 0);
        const uint32_t stage = STAGE_ADDR;
        {
            FUSED_ZONE("fz_rg3_wait");
            arrive.wait_min(TILES);  // every gate worker's chunks are in the staging slot
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
            FUSED_ZONE("fz_rg3_land");
            noc_semaphore_wait_min(data, RING - 1);  // one packet from every peer
        }
        {
            FUSED_ZONE("fz_rg3_unpack");
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
        arrive.set(0);
        *counter = *counter + 1;  // the next call's parity
#else
        done.set(1);
#endif
    }
    noc_async_full_barrier();
}
