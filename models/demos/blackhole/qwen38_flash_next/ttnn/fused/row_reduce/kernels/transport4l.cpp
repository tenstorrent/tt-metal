// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// row_reduce v2, "unicast slice" mode over two links (see transport4.cpp for the mode: rows above the packed-multicast
// limits, one block per receiver sent with start hop = distance and range 1, D dense tiles).  As transport2l.cpp, LINK
// 0 = the reduce core (stages, sends, lands, sums, writes), LINK 1 = the second transport core (stages and sends, then
// tells the reduce core it is done).  Link i carries the i-th half of every receiver's block (OUT_TILES chunks) into
// the reduce core's landing slot of the sender at offset i * HALF_BYTES; LINK 0 also stages the whole own block.
// Compile-time args as transport2l.cpp (7 LINK, 8 LINKS, 26 SEM_L1, 27 scratch offset, 28 unused) and 29 D (dense
// tiles),
//   30.. TensorAccessorArgs of the output, then the input.

#include "../../gr_fold/kernels/transport_phase.h"
#include "../../kernels/zones.h"

constexpr uint32_t CB_IN = get_compile_time_arg_val(0);
constexpr uint32_t TILES = get_compile_time_arg_val(1);
constexpr uint32_t SEM_GO = get_compile_time_arg_val(2);
constexpr uint32_t RING = get_compile_time_arg_val(3);
constexpr uint32_t SEM_NC_READ = get_compile_time_arg_val(5);
constexpr uint32_t SEM_DONE = get_compile_time_arg_val(6);
constexpr uint32_t LINK = get_compile_time_arg_val(7);
constexpr uint32_t LINKS = get_compile_time_arg_val(8);
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
constexpr uint32_t SEM_L1 = get_compile_time_arg_val(26);
constexpr uint32_t SCRATCH_OFF =
    get_compile_time_arg_val(27);  // 2 x 16 B scratch (counter read of LINK 1) after the staging
constexpr uint32_t DENSE = get_compile_time_arg_val(29);
constexpr uint32_t ACCESSOR_BASE = 30;
constexpr uint32_t PHASE_RT = 3;
constexpr uint32_t RX_OFF = 0;  // the reduce core's NoC (x, y) in the runtime args after the phase block
constexpr uint32_t CHUNKS = 2 * OUT_TILES;
constexpr uint32_t BLOCK_BYTES = CHUNKS * PITCH;
constexpr uint32_t SEG_CHUNKS = CHUNKS / LINKS;  // chunks of one receiver's block this link carries
constexpr uint32_t SEG_BYTES = SEG_CHUNKS * PITCH;
constexpr uint32_t SEG_OFF = LINK * SEG_BYTES;
constexpr uint32_t TOTAL_BYTES = TILES * 2 * PITCH;
constexpr uint32_t CPP = PAYLOAD / PITCH;
constexpr uint32_t PKT_BYTES = CPP * PITCH;
constexpr uint32_t PACKETS =
    (SEG_BYTES + PKT_BYTES - 1) / PKT_BYTES;  // packets of one segment (the same on every link)
static_assert(LINKS == 2 && CHUNKS % LINKS == 0, "two equal halves of a block");
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
    const uint32_t rx = get_arg_val<uint32_t>(PHASE_RT + PHASE_RT_ARGS + RX_OFF);  // NoC (x, y) of the reduce core
    const uint32_t ry = get_arg_val<uint32_t>(PHASE_RT + PHASE_RT_ARGS + RX_OFF + 1);
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
        const uint32_t tx =
            LINK == 0 ? my_x[0] : rx;  // NoC 0 coordinates of the reduce core (landing + data semaphore)
        const uint32_t ty = LINK == 0 ? my_y[0] : ry;
        uint32_t parity;
        if constexpr (LINK == 0) {
            parity = *counter & 1;
        } else {  // the call counter lives in the reduce core's L1; the reduce core steps it only after this core is
                  // done
#if defined(COMPILE_FOR_BRISC)
            const uint32_t cslot = pack.get_write_ptr() + SCRATCH_OFF + 16;
#else
            const uint32_t cslot = pack.get_write_ptr() + SCRATCH_OFF;
#endif
            noc_async_read(safe_get_noc_addr(rx, ry, LAND_ADDR, noc_index), cslot, 16);
            noc_async_read_barrier();
            parity = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cslot) & 1;
        }
        const uint32_t data_addr = parity ? DATA1_ADDR : data0_addr;
        const uint32_t land = LAND_ADDR + 16 + parity * RING * SLOT_BYTES;
        volatile tt_l1_ptr uint32_t* data = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(data_addr);
        const uint64_t data_noc = safe_get_noc_addr(tx, ty, data_addr, 0);
        const uint32_t stage = pack.get_write_ptr();  // the row's chunks of this link (LINK 0: plus the own block)
        {
            FUSED_ZONE("fz_rr_v4_stage");
#if defined(COMPILE_FOR_BRISC)
            const uint32_t first = 1;
#else
            const uint32_t first = 0;
#endif
            for (uint32_t t = first; t < TILES; t += 2) {  // the two RISCs take alternate tiles
                for (uint32_t k = 0; k < 2; ++k) {
                    const uint32_t c = 2 * t + k;
                    const uint32_t q = c / CHUNKS;
                    const uint32_t half = (c % CHUNKS) / SEG_CHUNKS;
                    const bool mine = (q != rank && half == LINK) || (LINK == 0 && q == rank);
                    if (mine) {
                        noc_async_read(local.get_noc_addr(t, k * face_bytes), stage + c * PITCH, chunk_len);
                    }
                }
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
                    const uint32_t bytes = SEG_BYTES - offset < PKT_BYTES ? SEG_BYTES - offset : PKT_BYTES;
                    noc_async_writes_flushed();  // the header is rewritten only after the previous packet left
                    fabric_multicast_noc_fused_unicast_with_atomic_inc(
                        &line.connection,
                        line.header,
                        stage + q * BLOCK_BYTES + SEG_OFF + offset,
                        bytes,
                        tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{
                            safe_get_noc_addr(tx, ty, land + rank * SLOT_BYTES + SEG_OFF + offset, 0),
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
        if constexpr (LINK == 0) {
            {
                FUSED_ZONE("fz_rr_v4_land");
                noc_semaphore_wait_min(
                    data, (RING - 1) * LINKS * PACKETS);  // every packet of every link from every peer
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
                    auto fill = [&](uint32_t dst, uint32_t logical) {
                        const uint32_t phys = MIRROR ? RING - 1 - logical : logical;
                        noc_async_write(
                            (phys == rank ? stage + rank * BLOCK_BYTES : land + phys * SLOT_BYTES) +
                                first_chunk * PITCH,
                            safe_get_noc_addr(my_x[noc_index], my_y[noc_index], dst, noc_index),
                            span);
                    };
                    cfs.reserve_back(1);
                    fill(cfs.get_write_ptr(), 0);  // L >= 1: p_0 (the F chain's seed); L == 0: p_0 is the own term
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
                    const uint32_t dense = cout.get_read_ptr();  // chunk first_chunk + i = face (c & 1) of tile c / 2
                    for (uint32_t i = 0; i < n_chunks; ++i) {
                        const uint32_t c = first_chunk + i;
                        noc_async_write(dense + i * PITCH, out.get_noc_addr(c >> 1, (c & 1) * face_bytes), chunk_len);
                    }
                    noc_async_write_barrier();
                    cout.pop_front(1);
                }
            }
        }  // LINK 0
        done.wait(1);  // NCRISC has read the counter and closed its connection
        done.set(0);
        if constexpr (LINK == 0) {
            Semaphore<> l1(SEM_L1);  // the other link's core has read the counter and closed its connection
            l1.wait(LINKS - 1);
            l1.set(0);
            *counter = *counter + 1;  // the next call's parity
        } else {
            Noc noc;
            Semaphore<> l1(SEM_L1);
            l1.up(noc, rx, ry, 1);
            noc.async_atomic_barrier();
        }
#else
        done.set(1);
#endif
    }
    noc_async_full_barrier();
}
