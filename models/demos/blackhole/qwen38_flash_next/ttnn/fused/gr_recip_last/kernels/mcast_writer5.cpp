// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// mcast_writer3.cpp for the norm core of the GR front v2 (QWEN38_GR_FRONT_V2=1).  Phase A hands the stats tile to the
// transport core as the two packed row chunks of the exchange (face 0 and face 1, ROWS rows of 32 B) written straight
// into the transport core's staging slot at this core's tile slot, then the scratch semaphore steps; nothing is written
// to the device's own stats page (the norm core's reader pulls the chunks from the transport core's L1).  The gate
// signal and phases B (u multicast) and C (recip multicast + normalized') are mcast_writer3's. Compile-time args as
// mcast_writer3.cpp except phase A's and phase B's (9 = the pack buffer cb, 7 = the workers' packed landing cb): 0 src
// cb, 1 pack cb (staging on the transport core), 5 scratch semaphore id; after the six accessor sets: 0 slot bytes, 1
// chunk bytes, 2 bf16 face bytes, 3 RING. Runtime args as mcast_writer3.cpp, phase A's: 0 tile slot on the transport
// core, 1-2 its NoC x, y (3-12 unused).

#include "../../gr_read/kernels/mcast_phase.h"
#include "../../kernels/zones.h"

// mcast_phase.h's mcast_phase for phase B (the u row into the down workers' CB), but only the valid rows travel: the
// two row chunks (face 0 and face 1, ROWS rows of 32 B) of the NUM_TILES u tiles are packed into one buffer
// ([tile][chunk][bytes]) and multicast ONCE into the workers' packed landing buffer at this core's block; the workers'
// reader unpacks them into the CB tiles (rows >= ROWS of those tiles stay as they were: a matmul output row depends on
// its own row of the left operand only).
template <uint32_t SRC_CB, uint32_t PACK_CB, uint32_t LAND_CB, uint32_t NUM_TILES, uint32_t SEM_ID>
FORCE_INLINE void u_phase(uint32_t rt, uint32_t chunk, uint32_t face) {
    const uint32_t block = get_arg_val<uint32_t>(rt + 0);  // this core's tile block of the workers' CB (tiles)
    const uint32_t x0 = get_arg_val<uint32_t>(rt + 1);
    const uint32_t y0 = get_arg_val<uint32_t>(rt + 2);
    const uint32_t x1 = get_arg_val<uint32_t>(rt + 3);
    const uint32_t y1 = get_arg_val<uint32_t>(rt + 4);
    const uint32_t num_dests = (x1 - x0 + 1) * (y1 - y0 + 1);
    const uint32_t tile_bytes = get_tile_size(SRC_CB);
    Noc noc;
    DataflowBuffer src(SRC_CB);
    DataflowBuffer pack(PACK_CB);
    DataflowBuffer land(LAND_CB);
    Semaphore<> sem(SEM_ID);
    src.wait_front(NUM_TILES);
    const uint32_t src_addr = src.get_read_ptr();
    const uint32_t pack_addr = pack.get_write_ptr();
    for (uint32_t t = 0; t < NUM_TILES; ++t) {
        for (uint32_t c = 0; c < 2; ++c) {
            volatile tt_l1_ptr uint32_t* from =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src_addr + t * tile_bytes + c * face);
            volatile tt_l1_ptr uint32_t* to =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pack_addr + (t * 2 + c) * chunk);
            for (uint32_t w = 0; w < chunk / 4;
                 w += 8) {  // 8 independent loads then 8 stores: a chunk is a multiple of 32 B
                const uint32_t a0 = from[w], a1 = from[w + 1], a2 = from[w + 2], a3 = from[w + 3];
                const uint32_t a4 = from[w + 4], a5 = from[w + 5], a6 = from[w + 6], a7 = from[w + 7];
                to[w] = a0;
                to[w + 1] = a1;
                to[w + 2] = a2;
                to[w + 3] = a3;
                to[w + 4] = a4;
                to[w + 5] = a5;
                to[w + 6] = a6;
                to[w + 7] = a7;
            }
        }
    }
    const uint32_t bytes = NUM_TILES * 2 * chunk;
    const uint32_t dst_addr = land.get_write_ptr() + (block / NUM_TILES) * bytes;  // block = branch * NUM_TILES tiles
    MulticastEndpoint mcast;
    constexpr bool from_far_corner = noc_index != 0;
    noc.async_write_multicast(
        CoreLocalMem<uint32_t>(pack_addr),
        mcast,
        bytes,
        num_dests,
        {},
        {.noc_x_start = from_far_corner ? x1 : x0,
         .noc_y_start = from_far_corner ? y1 : y0,
         .noc_x_end = from_far_corner ? x0 : x1,
         .noc_y_end = from_far_corner ? y0 : y1,
         .addr = dst_addr});
    noc.async_write_barrier();
    for (uint32_t x = x0; x <= x1; ++x) {
        for (uint32_t y = y0; y <= y1; ++y) {
            sem.up(noc, x, y, 1);
        }
    }
    src.pop_front(NUM_TILES);
    noc.async_atomic_barrier();
}

// mcast_phase.h's mcast_phase for phase C (recip tile multicast, no tiles write, normalized' as the extra stream) with
// zones
template <uint32_t SRC_CB, uint32_t DST_CB, uint32_t NUM_TILES, uint32_t EXTRA_CB, uint32_t SEM_ID, typename ExtraArgs>
FORCE_INLINE void recip_phase(const ExtraArgs& extra_args, uint32_t rt) {
    const uint32_t dst_tile_offset = get_arg_val<uint32_t>(rt + 0);
    const uint32_t x0 = get_arg_val<uint32_t>(rt + 1);
    const uint32_t y0 = get_arg_val<uint32_t>(rt + 2);
    const uint32_t x1 = get_arg_val<uint32_t>(rt + 3);
    const uint32_t y1 = get_arg_val<uint32_t>(rt + 4);
    const uint32_t num_dests = (x1 - x0 + 1) * (y1 - y0 + 1);
    const uint32_t tile_bytes = get_tile_size(SRC_CB);
    Noc noc;
    DataflowBuffer src(SRC_CB);
    DataflowBuffer dst(DST_CB);
    Semaphore<> sem(SEM_ID);
    {
        FUSED_ZONE("fz_gr2_w_recip_wait");
        src.wait_front(NUM_TILES);
    }
    const uint32_t src_addr = src.get_read_ptr();
    const uint32_t dst_addr = dst.get_write_ptr() + dst_tile_offset * tile_bytes;
    MulticastEndpoint mcast;
    constexpr bool from_far_corner = noc_index != 0;
    {
        FUSED_ZONE("fz_gr2_w_recip_mc");
        noc.async_write_multicast(
            CoreLocalMem<uint32_t>(src_addr),
            mcast,
            NUM_TILES * tile_bytes,
            num_dests,
            {},
            {.noc_x_start = from_far_corner ? x1 : x0,
             .noc_y_start = from_far_corner ? y1 : y0,
             .noc_x_end = from_far_corner ? x0 : x1,
             .noc_y_end = from_far_corner ? y0 : y1,
             .addr = dst_addr});
        noc.async_write_barrier();
    }
    {
        FUSED_ZONE("fz_gr2_w_recip_sem");
        for (uint32_t x = x0; x <= x1; ++x) {
            for (uint32_t y = y0; y <= y1; ++y) {
                sem.up(noc, x, y, 1);
            }
        }
    }
    src.pop_front(NUM_TILES);
    {
        FUSED_ZONE("fz_gr2_w_nout");
        const auto extra = TensorAccessor(extra_args, get_arg_val<uint32_t>(rt + 8));
        const uint32_t count = get_arg_val<uint32_t>(rt + 9);
        const uint32_t first = get_arg_val<uint32_t>(rt + 10);
        const uint32_t stride = get_arg_val<uint32_t>(rt + 11);
        const uint32_t batch = get_arg_val<uint32_t>(rt + 12);
        DataflowBuffer extra_cb(EXTRA_CB);
        const uint32_t extra_bytes = get_tile_size(EXTRA_CB);
        for (uint32_t done = 0; done < count; done += batch) {
            extra_cb.wait_front(batch);
            for (uint32_t t = 0; t < batch; ++t) {
                noc.async_write(
                    extra_cb,
                    extra,
                    extra_bytes,
                    {.offset_bytes = t * extra_bytes},
                    {.page_id = first + (done + t) * stride});
            }
            noc.async_write_barrier();
            extra_cb.pop_front(batch);
        }
    }
    noc.async_atomic_barrier();
}

constexpr uint32_t GATE_SEM = get_compile_time_arg_val(18);
constexpr uint32_t ACCESSOR_BASE = 19;

void kernel_main() {
    constexpr auto a_tiles = TensorAccessorArgs<ACCESSOR_BASE>();
    constexpr auto a_extra = TensorAccessorArgs<a_tiles.next_compile_time_args_offset()>();
    constexpr auto b_tiles = TensorAccessorArgs<a_extra.next_compile_time_args_offset()>();
    constexpr auto b_extra = TensorAccessorArgs<b_tiles.next_compile_time_args_offset()>();
    constexpr auto c_tiles = TensorAccessorArgs<b_extra.next_compile_time_args_offset()>();
    constexpr auto c_extra = TensorAccessorArgs<c_tiles.next_compile_time_args_offset()>();
    constexpr uint32_t EXTRA = c_extra.next_compile_time_args_offset();
    {
        FUSED_ZONE("fz_gr2_w_stats");
        constexpr uint32_t SRC_CB = get_compile_time_arg_val(0);
        constexpr uint32_t PACK_CB = get_compile_time_arg_val(1);
        constexpr uint32_t SEM_SCRATCH = get_compile_time_arg_val(5);
        constexpr uint32_t SLOT = get_compile_time_arg_val(EXTRA + 0);
        constexpr uint32_t CHUNK = get_compile_time_arg_val(EXTRA + 1);
        constexpr uint32_t FACE = get_compile_time_arg_val(EXTRA + 2);
        constexpr uint32_t RING = get_compile_time_arg_val(EXTRA + 3);
        const uint32_t k = get_arg_val<uint32_t>(0);
        const uint32_t tx = get_arg_val<uint32_t>(1);
        const uint32_t ty = get_arg_val<uint32_t>(2);
        DataflowBuffer src_cb(SRC_CB);
        DataflowBuffer pack(PACK_CB);
        src_cb.wait_front(1);
        const uint32_t src = src_cb.get_read_ptr();
        const uint32_t dst = pack.get_write_ptr() + RING * SLOT + k * 2 * CHUNK;  // the BRISC staging slot
        noc_async_write(src, get_noc_addr(tx, ty, dst), CHUNK);
        noc_async_write(src + FACE, get_noc_addr(tx, ty, dst + CHUNK), CHUNK);
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(tx, ty, get_semaphore(SEM_SCRATCH)), 1);
        noc_async_atomic_barrier();
        src_cb.pop_front(1);
    }
    if constexpr (GATE_SEM != 0xFF) {
        Noc noc;
        Semaphore<> gate(GATE_SEM);
        gate.up(noc, get_arg_val<uint32_t>(39), get_arg_val<uint32_t>(40), 1);
        noc.async_atomic_barrier();
    }
    {
        FUSED_ZONE("fz_gl_w_u");
        // phase B's compile args: 6 src cb (the u row), 7 landing cb on the workers, 8 tiles, 11 semaphore; 9 = the
        // pack cb here
        u_phase<
            get_compile_time_arg_val(6),
            get_compile_time_arg_val(9),
            get_compile_time_arg_val(7),
            get_compile_time_arg_val(8),
            get_compile_time_arg_val(11)>(13, get_compile_time_arg_val(EXTRA + 1), get_compile_time_arg_val(EXTRA + 2));
    }
    {
        FUSED_ZONE("fz_gl_w_recip");
        recip_phase<
            get_compile_time_arg_val(12),
            get_compile_time_arg_val(13),
            get_compile_time_arg_val(14),
            get_compile_time_arg_val(16),
            get_compile_time_arg_val(17)>(c_extra, 26);
    }
}
