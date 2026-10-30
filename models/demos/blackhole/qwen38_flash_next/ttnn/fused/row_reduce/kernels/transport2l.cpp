// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// row_reduce v2 (QWEN38_ROW_REDUCE=2): the row-packed output reduce without the per-call barrier.  As transport.cpp
// (v1) -- the same summation tree, the same compute.cpp, the same dense-tile build and output scatter -- except:
//   * no barrier: the packed rows of every device land in a PERSISTENT L1 landing buffer ([call counter 16 B][2 parity
//     sets x RING slots of SLOT_BYTES]) with the data semaphore of the call's parity (the gather's scheme, row_gather2
//     / transport_phase.h ROW_MODE 4: a device sends call n + 2 into a peer only after it received that peer's call n +
//     1 data, which the peer sends after its call n program ended); one buffer, one counter and one semaphore pair
//     serve every call of this op on the mesh; BRISC steps the counter at the very end after NCRISC's `done`;
//   * the valid rows are read straight from the partial into the staging slot, chunk k of tile t at (2 t + k) * PITCH
//     (PITCH = the chunk size for an L1 partial or a chunk that is a multiple of 64 B, else 64 B: a DRAM read of a 32 B
//     chunk into a 32 B-offset L1 address is corrupt): no bounce buffer, no copy;
//   * the packed rows go out in ceil(TOTAL / PAYLOAD) packets per direction (1 with a payload >= TOTAL).
// Compile-time args: 0 CB_IN, 1 TILES, 2 unused, 3 RING, 4-5 unused, 6 done semaphore id, 7-10 unused, 11 ROWS, 12
// PACK_CB,
//   13 PITCH, 14 CB_FS, 15 CB_BS, 16 CB_F, 17 CB_B, 18 CB_G, 19 CB_OUT, 20 OUT_TILES, 21 MIRROR, 22 LAND_ADDR, 23
//   DATA1_ADDR, 24 PAYLOAD, 25 SLOT_BYTES (landing slot), 26.. TensorAccessorArgs of the output, then the input.
// Runtime args as transport.cpp (phase block at 3: out address, input address, unused, data parity-0 address).

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
constexpr uint32_t ACCESSOR_BASE = 29;
constexpr uint32_t PHASE_RT = 3;
constexpr uint32_t TOTAL_BYTES = TILES * 2 * PITCH;
constexpr uint32_t PACKETS = (TOTAL_BYTES + PAYLOAD - 1) / PAYLOAD;
static_assert(TOTAL_BYTES <= SLOT_BYTES, "the landing slot holds the packed rows");
// two-link variant (QWEN38_ROW_REDUCE_LINKS=2): LINK 0 = the reduce core (stages its packets and its own slice, sends
// the packets p with p % LINKS == 0, lands + sums + writes), LINK 1 = the second transport core (stages and sends the
// packets p with p % LINKS == 1 into the reduce core's landing slots on every peer, then tells the reduce core it is
// done)
constexpr uint32_t LINK = get_compile_time_arg_val(7);
constexpr uint32_t LINKS = get_compile_time_arg_val(8);
constexpr uint32_t SEM_L1 = get_compile_time_arg_val(26);
constexpr uint32_t SCRATCH_OFF =
    get_compile_time_arg_val(27);  // 2 x 16 B scratch (counter read of LINK 1) after the staging slot
constexpr uint32_t RX_OFF = 0;     // the reduce core's NoC (x, y) in the runtime args after the phase block
// SPLIT 0: packet p goes over link p % LINKS; SPLIT 1: link i carries the i-th equal share of the packed bytes (cut at
// a chunk)
constexpr uint32_t SPLIT = get_compile_time_arg_val(28);
constexpr uint32_t SHARE = (TOTAL_BYTES / PITCH / LINKS) * PITCH;
constexpr uint32_t SHARE_BEGIN = LINK * SHARE;
constexpr uint32_t SHARE_END = LINK + 1 == LINKS ? TOTAL_BYTES : SHARE_BEGIN + SHARE;
// the packets of the whole call (each peer sends them over its links; the data semaphore counts them all)
constexpr uint32_t PACKETS_ALL =
    SPLIT == 0 ? PACKETS
               : (SHARE + PAYLOAD - 1) / PAYLOAD + (TOTAL_BYTES - SHARE * (LINKS - 1) + PAYLOAD - 1) / PAYLOAD;
constexpr uint32_t CHUNKS_PER_PACKET = PAYLOAD / PITCH;
static_assert(PAYLOAD % PITCH == 0, "a packet holds whole chunks");

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
        FUSED_ZONE("fz_rr_v2_main");
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
        const uint32_t pack_base = pack.get_write_ptr();
        const uint32_t stage =
            pack_base;  // ONE staging slot, read by both RISCs (NCRISC the first half of the tiles, BRISC the rest)
        {
            FUSED_ZONE("fz_rr_v2_stage");
#if defined(COMPILE_FOR_BRISC)
            const uint32_t first = 1;
#else
            const uint32_t first = 0;
#endif
            // only the chunks of this link's packets (LINK 0: plus the own slice the feed reads); the two RISCs take
            // alternate tiles
            for (uint32_t t = first; t < TILES; t += 2) {
                for (uint32_t k = 0; k < 2; ++k) {
                    const uint32_t c = 2 * t + k;
                    const bool mine = SPLIT == 0 ? (c / CHUNKS_PER_PACKET) % LINKS == LINK
                                                 : (c * PITCH >= SHARE_BEGIN && c * PITCH < SHARE_END);
                    const bool own = LINK == 0 && c >= rank * OUT_TILES * 2 && c < (rank + 1) * OUT_TILES * 2;
                    if (mine || own) {
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
            const uint32_t n_mine = SPLIT == 0 ? PACKETS : (SHARE_END - SHARE_BEGIN + PAYLOAD - 1) / PAYLOAD;
            for (uint32_t q = 0; q < n_mine; ++q) {
                const uint32_t p = SPLIT == 0 ? LINK + q * LINKS : q;
                if (SPLIT == 0 && p >= PACKETS) {
                    break;
                }
                const uint32_t offset = SPLIT == 0 ? p * PAYLOAD : SHARE_BEGIN + q * PAYLOAD;
                const uint32_t end = SPLIT == 0 ? TOTAL_BYTES : SHARE_END;
                const uint32_t bytes = end - offset < PAYLOAD ? end - offset : PAYLOAD;
                noc_async_writes_flushed();  // the header is rewritten only after the previous packet left
                fabric_multicast_noc_fused_unicast_with_atomic_inc(
                    &line.connection,
                    line.header,
                    stage + offset,
                    bytes,
                    tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{
                        safe_get_noc_addr(tx, ty, land + rank * SLOT_BYTES + offset, 0), data_noc, 1, true},
                    1,
                    line.range);
            }
        }
        line.close();
#if defined(COMPILE_FOR_BRISC)
        if constexpr (LINK == 0) {
            {
                FUSED_ZONE("fz_rr_v2_land");
                noc_semaphore_wait_min(data, (RING - 1) * PACKETS_ALL);  // every packet from every peer
                noc_semaphore_set(data, 0);
            }
            {
                FUSED_ZONE("fz_rr_v2_feed");
                DataflowBuffer cin(CB_IN), cfs(CB_FS), cbs(CB_BS);
                const uint32_t L = MIRROR ? RING - 1 - rank : rank;
                // the slice's OUT_TILES packed tiles (2 chunks each, PITCH apart) are contiguous in every source's
                // slot: ONE dense tile per source (the sum is elementwise, the layout inside the tile is irrelevant)
                const uint32_t slice = rank * OUT_TILES * 2 * PITCH;
                const uint32_t slice_bytes = OUT_TILES * 2 * PITCH;
                auto fill = [&](uint32_t dst, uint32_t logical) {
                    const uint32_t phys = MIRROR ? RING - 1 - logical : logical;
                    noc_async_write(
                        (phys == rank ? stage : land + phys * SLOT_BYTES) + slice,
                        safe_get_noc_addr(my_x[noc_index], my_y[noc_index], dst, noc_index),
                        slice_bytes);
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
                uint32_t at = cin.get_write_ptr();
                for (uint32_t s = 1; s <= L;
                     ++s, at += tile_bytes) {  // the F chain's p_1 .. p_{L-1}, then the own term p_L
                    fill(at, s);
                }
                for (uint32_t s = RING - 2; s > L && L < RING - 1;
                     --s, at += tile_bytes) {  // the B chain's p_{R-2} .. p_{L+1}
                    fill(at, s);
                }
                noc_async_write_barrier();
                cfs.push_back(1);
                if (L < RING - 1) {
                    cbs.push_back(1);
                }
                cin.push_back(n_in);
            }
            {
                FUSED_ZONE("fz_rr_v2_write");
                DataflowBuffer cout(CB_OUT);
                cout.wait_front(1);
                const uint32_t dense =
                    cout.get_read_ptr();  // chunk 2j / 2j + 1 = rows [0, ROWS) of faces 0 / 1 of output tile j
                for (uint32_t j = 0; j < OUT_TILES; ++j) {
                    noc_async_write(dense + (2 * j) * PITCH, out.get_noc_addr(j, 0), chunk_len);
                    noc_async_write(dense + (2 * j + 1) * PITCH, out.get_noc_addr(j, face_bytes), chunk_len);
                }
                noc_async_write_barrier();
                cout.pop_front(1);
            }
        }  // LINK 0
        done.wait(1);  // NCRISC has read the counter and closed its connection
        done.set(0);
        if constexpr (LINK == 0) {
            if constexpr (LINKS > 1) {  // the other links' cores have read the counter and closed their connections
                Semaphore<> l1(SEM_L1);
                l1.wait(LINKS - 1);
                l1.set(0);
            }
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
