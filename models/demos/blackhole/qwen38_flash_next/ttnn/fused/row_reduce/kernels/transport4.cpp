// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// row_reduce v2, "unicast slice" mode (rows above the packed-multicast limits: bf16 3..8, fp32 2..8).  As
// transport2.cpp (same summation tree, same compute.cpp, same persistent landing buffer, counter and parity scheme)
// except:
//   * every sender sends each receiver r != rank ONLY r's slice block (the live rows of r's W / RING columns, both
//   faces,
//     2 * OUT_TILES chunks of PITCH bytes, contiguous in the staged row) with start hop = distance to r and range 1:
//     the forward connection (BRISC) serves the ranks above, the backward one (NCRISC) the ranks below.  A block larger
//     than a packet goes in ceil(BLOCK / PKT) packets of whole chunks; each packet raises the receiver's data semaphore
//     by one;
//   * the landing slot of a sender holds only that one block (the slot stride SLOT_BYTES is the persistent buffer's);
//   * the slice of a source does not fit one tile: D dense tiles (CPT = tile bytes / PITCH chunks each, the last one
//     shorter) are fed, summed (compute.cpp with OUT_TILES = D) and written one after the other.
// Compile-time args as transport2.cpp (0 CB_IN, 1 TILES, 3 RING, 5 nc-read semaphore, 6 done semaphore, 11 ROWS, 12
// PACK_CB,
//   13 PITCH, 14 CB_FS, 15 CB_BS, 19 CB_OUT, 20 OUT_TILES, 21 MIRROR, 22 LAND_ADDR, 23 DATA1_ADDR, 24 PAYLOAD, 25
//   SLOT_BYTES), 26 D (dense tiles), 27.. TensorAccessorArgs of the output, then the input.

#include "../../gr_fold/kernels/transport_phase.h"
#include "../../kernels/zones.h"

constexpr uint32_t CB_IN = get_compile_time_arg_val(0);
constexpr uint32_t TILES = get_compile_time_arg_val(1);
constexpr uint32_t SEM_GO = get_compile_time_arg_val(2);
constexpr uint32_t RING = get_compile_time_arg_val(3);
constexpr uint32_t SEM_NC_READ = get_compile_time_arg_val(5);
constexpr uint32_t SEM_DONE = get_compile_time_arg_val(6);
constexpr uint32_t ROWS = get_compile_time_arg_val(11);
constexpr uint32_t PACK_CB = get_compile_time_arg_val(12);
constexpr uint32_t PITCH = get_compile_time_arg_val(13);
constexpr uint32_t CB_FS = get_compile_time_arg_val(14);
constexpr uint32_t CB_BS = get_compile_time_arg_val(15);
constexpr uint32_t CB_OUT = get_compile_time_arg_val(19);
constexpr uint32_t OUT_TILES = get_compile_time_arg_val(20);
constexpr uint32_t MIRROR = get_compile_time_arg_val(21);
constexpr uint32_t LAND_ADDR = get_compile_time_arg_val(22);
constexpr uint32_t DATA1_ADDR = get_compile_time_arg_val(23);
constexpr uint32_t PAYLOAD = get_compile_time_arg_val(24);
constexpr uint32_t SLOT_BYTES = get_compile_time_arg_val(25);
constexpr uint32_t DENSE = get_compile_time_arg_val(26);
constexpr uint32_t ACCESSOR_BASE = 27;
constexpr uint32_t PHASE_RT = 3;
constexpr uint32_t CHUNKS = 2 * OUT_TILES;           // chunks of one receiver's slice
constexpr uint32_t BLOCK_BYTES = CHUNKS * PITCH;     // one receiver's block
constexpr uint32_t TOTAL_BYTES = TILES * 2 * PITCH;  // the staged row
constexpr uint32_t CPP = PAYLOAD / PITCH;            // whole chunks per packet
constexpr uint32_t PKT_BYTES = CPP * PITCH;
constexpr uint32_t PACKETS = (BLOCK_BYTES + PKT_BYTES - 1) / PKT_BYTES;  // packets of one block
static_assert(CPP >= 1, "a packet holds a chunk");
static_assert(TOTAL_BYTES == RING * BLOCK_BYTES, "the row is RING slices");
static_assert(BLOCK_BYTES <= SLOT_BYTES, "the landing slot holds one block");

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
        FUSED_ZONE("fz_rr_v4_main");
        Semaphore<> done(SEM_DONE);
        DataflowBuffer pack(PACK_CB);
        const uint32_t tile_bytes = get_tile_size(CB_IN);
        const uint32_t face_bytes = tile_bytes / 4;
        const uint32_t chunk_len = ROWS * (face_bytes / 16);  // rows [0, ROWS) of one face
        volatile tt_l1_ptr uint32_t* counter = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(LAND_ADDR);
        const uint32_t parity = *counter & 1;
        const uint32_t data_addr = parity ? DATA1_ADDR : data0_addr;
        const uint32_t land = LAND_ADDR + 16 + parity * RING * SLOT_BYTES;
        volatile tt_l1_ptr uint32_t* data = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(data_addr);
        const uint64_t data_noc = safe_get_noc_addr(my_x[0], my_y[0], data_addr, 0);
        const uint32_t stage = pack.get_write_ptr();  // the whole row, chunk c of tile t at (2 t + k) * PITCH
        {
            FUSED_ZONE("fz_rr_v4_stage");
#if defined(COMPILE_FOR_BRISC)
            const uint32_t first = TILES / 2, last = TILES;
#else
            const uint32_t first = 0, last = TILES / 2;
#endif
            for (uint32_t t = first; t < last; ++t) {
                noc_async_read(local.get_noc_addr(t, 0), stage + (2 * t) * PITCH, chunk_len);
                noc_async_read(local.get_noc_addr(t, face_bytes), stage + (2 * t + 1) * PITCH, chunk_len);
            }
            noc_async_read_barrier();
            Semaphore<> go(SEM_GO), nc_read(SEM_NC_READ);
#if defined(COMPILE_FOR_BRISC)
            go.set(1);  // BRISC's half is in; wait for NCRISC's
            nc_read.wait(1);
            nc_read.set(0);
#else
            nc_read.set(1);
            go.wait(1);
            go.set(0);
#endif
        }
        if (line.has_peers) {
            for (uint32_t h = 1; h <= line.range; ++h) {
#if defined(COMPILE_FOR_BRISC)
                const uint32_t q = (rank + h) % RING;
#else
                const uint32_t q = (rank + RING - h) % RING;
#endif
                for (uint32_t p = 0; p < PACKETS; ++p) {
                    const uint32_t offset = p * PKT_BYTES;
                    const uint32_t bytes = BLOCK_BYTES - offset < PKT_BYTES ? BLOCK_BYTES - offset : PKT_BYTES;
                    noc_async_writes_flushed();  // the header is rewritten only after the previous packet left
                    fabric_multicast_noc_fused_unicast_with_atomic_inc(
                        &line.connection,
                        line.header,
                        stage + q * BLOCK_BYTES + offset,
                        bytes,
                        tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{
                            safe_get_noc_addr(my_x[0], my_y[0], land + rank * SLOT_BYTES + offset, 0),
                            data_noc,
                            1,
                            true},
                        h,
                        1);
                }
            }
        }
        line.close();
#if defined(COMPILE_FOR_BRISC)
        {
            FUSED_ZONE("fz_rr_v4_land");
            noc_semaphore_wait_min(data, (RING - 1) * PACKETS);  // every packet from every peer
            noc_semaphore_set(data, 0);
        }
        DataflowBuffer cin(CB_IN), cfs(CB_FS), cbs(CB_BS), cout(CB_OUT);
        const uint32_t L = MIRROR ? RING - 1 - rank : rank;
        const uint32_t cpt = tile_bytes / PITCH;  // chunks of a dense tile
        const uint32_t cin_base = cin.get_write_ptr();
        uint32_t cin_page = 0;  // page of CB_IN the next push starts at (the CB wraps between dense tiles)
        for (uint32_t d = 0; d < DENSE; ++d) {
            const uint32_t first_chunk = d * cpt;
            const uint32_t n_chunks = CHUNKS - first_chunk < cpt ? CHUNKS - first_chunk : cpt;
            const uint32_t span = n_chunks * PITCH;
            {
                FUSED_ZONE("fz_rr_v4_feed");
                // dense tile d of a source: its n_chunks chunks of my slice, contiguous in the source's block
                auto fill = [&](uint32_t dst, uint32_t logical) {
                    const uint32_t phys = MIRROR ? RING - 1 - logical : logical;
                    noc_async_write(
                        (phys == rank ? stage + rank * BLOCK_BYTES : land + phys * SLOT_BYTES) + first_chunk * PITCH,
                        safe_get_noc_addr(my_x[noc_index], my_y[noc_index], dst, noc_index),
                        span);
                };
                cfs.reserve_back(1);
                fill(cfs.get_write_ptr(), 0);  // L >= 1: p_0 (the F chain's seed); L == 0: p_0 is the device's own term
                if (L < RING - 1) {
                    cbs.reserve_back(1);
                    fill(cbs.get_write_ptr(), RING - 1);
                }
                const uint32_t n_in =
                    (L >= 1 ? L : 0) + (L < RING - 1 ? RING - 2 - L : 0);  // p_1 .. p_L, p_{R-2} .. p_{L+1}
                cin.reserve_back(n_in);
                uint32_t page = cin_page;
                for (uint32_t s = 1; s <= L; ++s) {  // the F chain's p_1 .. p_{L-1}, then the own term p_L
                    fill(cin_base + page * tile_bytes, s);
                    page = page + 1 == RING ? 0 : page + 1;
                }
                for (uint32_t s = RING - 2; s > L && L < RING - 1; --s) {  // the B chain's p_{R-2} .. p_{L+1}
                    fill(cin_base + page * tile_bytes, s);
                    page = page + 1 == RING ? 0 : page + 1;
                }
                noc_async_write_barrier();
                cfs.push_back(1);
                if (L < RING - 1) {
                    cbs.push_back(1);
                }
                cin.push_back(n_in);
                cin_page = page;
            }
            {
                FUSED_ZONE("fz_rr_v4_write");
                cout.wait_front(1);
                const uint32_t dense =
                    cout.get_read_ptr();  // chunk first_chunk + i = rows [0, ROWS) of face (c & 1) of tile c / 2
                for (uint32_t i = 0; i < n_chunks; ++i) {
                    const uint32_t c = first_chunk + i;
                    noc_async_write(dense + i * PITCH, out.get_noc_addr(c >> 1, (c & 1) * face_bytes), chunk_len);
                }
                noc_async_write_barrier();
                cout.pop_front(1);
            }
        }
        done.wait(1);  // NCRISC has read the counter and closed its connection
        done.set(0);
        *counter = *counter + 1;  // the next call's parity
#else
        done.set(1);
#endif
    }
    noc_async_full_barrier();
}
