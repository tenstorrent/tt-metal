// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// The row-packed output reduce of a decode row (reduce-scatter of a [1, 1, ROWS, W] partial over the TP line) as one
// program per device: transport (this file, one core, NCRISC backward / BRISC forward) + the sum (compute.cpp).
// Every device packs the valid rows of its whole partial (row [0, ROWS) of faces 0 and 1 of every tile: the packed
// layout of row_gather, ROW_MODE 3) into ONE packet per direction on the line (fused write + atomic increment into the
// peers' landing slots, one slot per source device, behind the per-call barrier of gr_fold/kernels/transport_phase.h).
// BRISC then waits for every packet of every peer.  The OUT_TILES tiles of THIS device's slice (tiles rank * OUT_TILES
// ..) are contiguous in every source's packed slot, so ONE dense tile per source (a raw copy of that range, by NoC, the
// layout inside the tile is irrelevant to an elementwise sum) goes to the compute kernel in the order of the stock line
// reduce-scatter's summation tree; BRISC then scatters the compute kernel's dense tile into the rows [0, ROWS) of the
// faces 0 and 1 of the output pages 0 .. OUT_TILES - 1 (the other rows of the output tiles are not written).
// The tree, in the LOGICAL rank L (MIRROR flips the line: L = RING - 1 - rank, source s -> RING - 1 - s):
//   out = (p_L + F) + B,  F = p_{L-1} + (p_{L-2} + ... + (p_1 + p_0)),  B = p_{L+1} + (p_{L+2} + ... + (p_{R-2} +
//   p_{R-1}))
// L = 0 has no F (out = p_0 + B) and L = RING - 1 no B (out = p_L + F).
// Feed (compute.cpp consumes in this order):  CB_FS <- p_0 (for L = 0 that is the device's own term); CB_BS <-
//   p_{R-1} (L < R-1); CB_IN <- p_1 .. p_{L-1}, then p_L (L >= 1), then p_{R-2} .. p_{L+1} (L <= R-3).
// Compile-time args: 0 CB_IN, 1 TILES (input tiles), 2 go semaphore id, 3 RING, 4 SOURCE (unused), 5 and 6 (unused), 7
//   TILE_FIRST (unused), 8 TILE_STEP (unused), 9 and 10 (unused), 11 ROWS, 12 PACK_CB, 13 SLOT_BYTES, 14 CB_FS, 15
//   CB_BS, 16 CB_F, 17 CB_B, 18 CB_G, 19 CB_OUT, 20 OUT_TILES, 21 MIRROR, 22 TOTAL_BYTES (the packed rows of the whole
//   partial), 23 PAYLOAD (fabric packet payload bytes), 24 L1_INPUT, 25.. TensorAccessorArgs of the output, then the
//   input.
// Runtime args as gr_fold/kernels/transport.cpp (phase block at 3: out address, input address, barrier address, data
// address, consumer count 0, consumer semaphore).
// PACK_CB raw L1 buffer: RING landing slots, then one staging slot per RISC, then one bounce buffer per RISC (32 B
// chunks only).

#include "../../gr_fold/kernels/transport_phase.h"
#include "../../kernels/zones.h"

constexpr uint32_t CB_IN = get_compile_time_arg_val(0);
constexpr uint32_t TILES = get_compile_time_arg_val(1);
constexpr uint32_t SEM_GO = get_compile_time_arg_val(2);
constexpr uint32_t RING = get_compile_time_arg_val(3);
constexpr uint32_t ROWS = get_compile_time_arg_val(11);
constexpr uint32_t PACK_CB = get_compile_time_arg_val(12);
constexpr uint32_t SLOT_BYTES = get_compile_time_arg_val(13);
constexpr uint32_t CB_FS = get_compile_time_arg_val(14);
constexpr uint32_t CB_BS = get_compile_time_arg_val(15);
constexpr uint32_t CB_F = get_compile_time_arg_val(16);
constexpr uint32_t CB_B = get_compile_time_arg_val(17);
constexpr uint32_t CB_G = get_compile_time_arg_val(18);
constexpr uint32_t CB_OUT = get_compile_time_arg_val(19);
constexpr uint32_t OUT_TILES = get_compile_time_arg_val(20);
constexpr uint32_t MIRROR = get_compile_time_arg_val(21);
constexpr uint32_t TOTAL_BYTES = get_compile_time_arg_val(22);
constexpr uint32_t PAYLOAD = get_compile_time_arg_val(23);  // the fabric packet payload, a multiple of 16
constexpr uint32_t PACKETS = (TOTAL_BYTES + PAYLOAD - 1) / PAYLOAD;
constexpr uint32_t L1_INPUT = get_compile_time_arg_val(24);  // the partial lives in L1 (16 B read alignment: no bounce)
constexpr uint32_t ACCESSOR_BASE = 25;
constexpr uint32_t PHASE_RT = 3;
constexpr uint32_t READ_ALIGN =
    64;  // NOC_DRAM_READ_ALIGNMENT_BYTES on Blackhole: a 32 B chunk is bounced through a 64 B aligned scratch

// L1 -> L1 by the RISC: eight words loaded back to back, then stored (a load-store alternation stalls on every load)
FORCE_INLINE void copy_words(uint32_t dst, uint32_t src, uint32_t bytes) {
    volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst);
    volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src);
    uint32_t i = 0;
    for (; i + 8 <= bytes / 4; i += 8) {
        const uint32_t a0 = s[i], a1 = s[i + 1], a2 = s[i + 2], a3 = s[i + 3];
        const uint32_t a4 = s[i + 4], a5 = s[i + 5], a6 = s[i + 6], a7 = s[i + 7];
        d[i] = a0;
        d[i + 1] = a1;
        d[i + 2] = a2;
        d[i + 3] = a3;
        d[i + 4] = a4;
        d[i + 5] = a5;
        d[i + 6] = a6;
        d[i + 7] = a7;
    }
    for (; i < bytes / 4; ++i) {
        d[i] = s[i];
    }
}

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
    Noc noc;
    Semaphore<> go(SEM_GO);
    LineSender line;
    line.open<RING>(rank, arg_idx);
    {
        FUSED_ZONE("fz_rr_main");
        line.arrive(barrier_addr);

        DataflowBuffer pack(PACK_CB);
        const uint32_t tile_bytes = get_tile_size(CB_IN);
        const uint32_t face_bytes = tile_bytes / 4;
        const uint32_t row_bytes = face_bytes / 16;
        const uint32_t chunk_len = ROWS * row_bytes;     // rows [0, ROWS) of one face
        const uint32_t pack_tile_bytes = 2 * chunk_len;  // faces 0 and 1
        const uint32_t pack_base = pack.get_write_ptr();
        const uint32_t scratch_bytes = TILES * 2 * READ_ALIGN;  // the bounce buffer of the 32 B chunk case
#if defined(COMPILE_FOR_BRISC)
        const uint32_t stage = pack_base + RING * SLOT_BYTES;
        const uint32_t scratch = pack_base + (RING + 2) * SLOT_BYTES;
#else
        const uint32_t stage = pack_base + (RING + 1) * SLOT_BYTES;
        const uint32_t scratch = pack_base + (RING + 2) * SLOT_BYTES + scratch_bytes;
#endif
        const uint64_t data_noc = safe_get_noc_addr(my_x[0], my_y[0], data_addr, 0);
        volatile tt_l1_ptr uint32_t* barrier = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_addr);
        volatile tt_l1_ptr uint32_t* data = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(data_addr);

        // this device's partial, rows [0, ROWS), packed (BRISC always: it is the device's own term in the sum)
#if defined(COMPILE_FOR_BRISC)
        constexpr bool STAGE = true;
#else
        const bool STAGE = line.has_peers;
#endif
        if (STAGE) {
            FUSED_ZONE("fz_rr_stage");
            if (L1_INPUT ||
                chunk_len % READ_ALIGN == 0) {  // L1 sources and aligned chunks go straight into the staging slot
                for (uint32_t t = 0; t < TILES; ++t) {
                    noc_async_read(local.get_noc_addr(t, 0), stage + t * pack_tile_bytes, chunk_len);
                    noc_async_read(
                        local.get_noc_addr(t, face_bytes), stage + t * pack_tile_bytes + chunk_len, chunk_len);
                }
                noc_async_read_barrier();
            } else {  // a 32 B chunk read into an L1 address that is not 64 B aligned is corrupt (measured): bounce it
                for (uint32_t t = 0; t < TILES; ++t) {
                    noc_async_read(local.get_noc_addr(t, 0), scratch + (2 * t) * READ_ALIGN, chunk_len);
                    noc_async_read(local.get_noc_addr(t, face_bytes), scratch + (2 * t + 1) * READ_ALIGN, chunk_len);
                }
                noc_async_read_barrier();
                for (uint32_t t = 0; t < TILES; ++t) {
                    copy_words(stage + t * pack_tile_bytes, scratch + (2 * t) * READ_ALIGN, chunk_len);
                    copy_words(stage + t * pack_tile_bytes + chunk_len, scratch + (2 * t + 1) * READ_ALIGN, chunk_len);
                }
            }
        }
        {
            FUSED_ZONE("fz_rr_sync");
#if defined(COMPILE_FOR_BRISC)
            noc_semaphore_wait_min(barrier, RING - 1);
            noc_semaphore_set(barrier, 0);  // reset before the release below; the sends follow the release
            go.set(1);
#else
            go.wait(1);
            go.set(0);
#endif
        }
        if (line.has_peers) {
            for (uint32_t p = 0; p < PACKETS; ++p) {  // one packet per direction when the packed rows fit the payload
                const uint32_t offset = p * PAYLOAD;
                const uint32_t bytes = TOTAL_BYTES - offset < PAYLOAD ? TOTAL_BYTES - offset : PAYLOAD;
                noc_async_writes_flushed();  // the header is rewritten only after the previous packet left
                fabric_multicast_noc_fused_unicast_with_atomic_inc(
                    &line.connection,
                    line.header,
                    stage + offset,
                    bytes,
                    tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{
                        safe_get_noc_addr(my_x[0], my_y[0], pack_base + rank * SLOT_BYTES + offset, 0),
                        data_noc,
                        1,
                        true},
                    1,
                    line.range);
            }
        }
        line.close();
#if defined(COMPILE_FOR_BRISC)
        {
            FUSED_ZONE("fz_rr_land");
            noc_semaphore_wait_min(data, (RING - 1) * PACKETS);  // every packet from every peer
            noc_semaphore_set(data, 0);
        }
        {
            FUSED_ZONE("fz_rr_feed");
            DataflowBuffer cin(CB_IN), cfs(CB_FS), cbs(CB_BS);
            const uint32_t L = MIRROR ? RING - 1 - rank : rank;
            // the slice's OUT_TILES packed tiles are contiguous in every source's slot: ONE dense tile per source holds
            // them (the sum is elementwise, so the layout inside the tile does not matter), 7 adds instead of 7 x
            // OUT_TILES
            const uint32_t slice = rank * OUT_TILES * pack_tile_bytes;
            const uint32_t slice_bytes = OUT_TILES * pack_tile_bytes;
            auto fill = [&](uint32_t dst, uint32_t logical) {
                const uint32_t phys = MIRROR ? RING - 1 - logical : logical;
                noc_async_write(
                    (phys == rank ? stage : pack_base + phys * SLOT_BYTES) + slice,
                    safe_get_noc_addr(my_x[noc_index], my_y[noc_index], dst, noc_index),
                    slice_bytes);
            };
            // all the dense tiles are built by NoC copies (one barrier), then pushed in the compute kernel's order; the
            // CBs hold them all (CB_IN: RING pages) and a call starts with empty CBs, so no page wraps
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
            FUSED_ZONE("fz_rr_write");
            DataflowBuffer cout(CB_OUT);
            cout.wait_front(1);
            const uint32_t dense =
                cout.get_read_ptr();  // chunk 2j / 2j + 1 = rows [0, ROWS) of faces 0 / 1 of output tile j
            for (uint32_t j = 0; j < OUT_TILES; ++j) {
                noc_async_write(dense + j * pack_tile_bytes, out.get_noc_addr(j, 0), chunk_len);
                noc_async_write(dense + j * pack_tile_bytes + chunk_len, out.get_noc_addr(j, face_bytes), chunk_len);
            }
            noc_async_write_barrier();
            cout.pop_front(1);
        }
#endif
    }
    noc_async_full_barrier();
}
