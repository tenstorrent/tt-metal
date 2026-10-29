// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Two-phase line transport of the GR front v2 (QWEN38_GR_FRONT_V2=1): gr_fold/kernels/transport2.cpp with a new phase
// B. Phase A (the stats): packed mode 3 with the barrier, direct staging by the norm cores, no DRAM unpack
// (stats_phase).  Phase B (the partials):
//   - barrier-free parity landing (as gr_fold's ROW_MODE 4: persistent L1 landing buffer [call counter, 16 B][2 slot
//   sets of
//     RING slots], a data semaphore per parity, the counter stepped by BRISC at the end of the call);
//   - the packed bytes are already in the staging buffer (the down workers write their row chunks there directly,
//     raw_writer.cpp): one scratch-semaphore wait, no per-tile copy loop and no write barrier, then ONE fused packet
//     per direction straight from the staging buffer (both RISCs send the same bytes);
//   - the unpack of the RING - 1 landed slots into the gathered pages is split over both RISCs (BRISC: even sources
//     in order, NCRISC: odd ones) behind a program-local `unpack` semaphore BRISC raises once the data count is
//     complete.
// Phase B has no consumers inside the program (gr_read's low-rank program reads the pages later).
// Compile-time args: 0 go sem, 1 RING, 2 done sem, 3 TILE_FIRST, 4 TILE_STEP, 5-12 phase A (scratch cb, TILES, scratch
// sem,
//   page tile stride, page rank stride, ROWS, pack cb, slot bytes), 13-19 phase B (stage cb, TILES, scratch sem, page
//   tile stride, page rank stride, ROWS, SLOT bytes), 20 unpack sem, 21 odd data semaphore address, 22 landing buffer
//   address; then the TensorAccessorArgs of A out, A local, B out, B local.
// Runtime args: as transport2.cpp (0 rank, 1 delay before the arrives, 2 delay after phase A's reset, 3.. phase A's
// block,
//   phase B's block, the fabric connection).

#include "../../gr_fold/kernels/transport_phase.h"
#include "../../kernels/zones.h"

constexpr uint32_t SEM_GO = get_compile_time_arg_val(0);
constexpr uint32_t RING = get_compile_time_arg_val(1);
constexpr uint32_t SEM_DONE = get_compile_time_arg_val(2);
constexpr uint32_t TILE_FIRST = get_compile_time_arg_val(3);
constexpr uint32_t TILE_STEP = get_compile_time_arg_val(4);
constexpr uint32_t A_SCRATCH_CB = get_compile_time_arg_val(5);
constexpr uint32_t A_TILES = get_compile_time_arg_val(6);
constexpr uint32_t A_SEM_SCRATCH = get_compile_time_arg_val(7);
constexpr uint32_t A_PAGE_TILE_STRIDE = get_compile_time_arg_val(8);
constexpr uint32_t A_PAGE_RANK_STRIDE = get_compile_time_arg_val(9);
constexpr uint32_t A_ROWS = get_compile_time_arg_val(10);
constexpr uint32_t A_PACK_CB = get_compile_time_arg_val(11);
constexpr uint32_t A_SLOT_BYTES = get_compile_time_arg_val(12);
constexpr uint32_t B_STAGE_CB = get_compile_time_arg_val(13);
constexpr uint32_t B_TILES = get_compile_time_arg_val(14);
constexpr uint32_t B_SEM_SCRATCH = get_compile_time_arg_val(15);
constexpr uint32_t B_PAGE_TILE_STRIDE = get_compile_time_arg_val(16);
constexpr uint32_t B_PAGE_RANK_STRIDE = get_compile_time_arg_val(17);
constexpr uint32_t B_ROWS = get_compile_time_arg_val(18);
constexpr uint32_t B_SLOT_BYTES = get_compile_time_arg_val(19);
constexpr uint32_t SEM_UNPACK = get_compile_time_arg_val(20);
constexpr uint32_t B_DATA1 = get_compile_time_arg_val(21);
constexpr uint32_t B_LAND = get_compile_time_arg_val(22);
constexpr uint32_t ACCESSOR_BASE = 23;
constexpr uint32_t A_RT = 3;
static_assert(B_ROWS <= 16, "the packed partial rows are two chunks of face rows");

// Phase A (the stats), packed mode 3 with the barrier as gr_fold's transport_phase, but: the norm cores write their
// packed row chunks straight into the staging slot (no copy loop), both RISCs send from it, and nothing is unpacked
// into DRAM: the landing slots (program CB memory, the pack CB) are read by the norm cores' reader once the consumers
// are released.
template <typename OutArgs>
FORCE_INLINE void stats_phase(LineSender& line, const OutArgs&, uint32_t rt, uint32_t delay_after_reset) {
    constexpr uint32_t MY_TILES = TILE_FIRST < A_TILES ? (A_TILES - TILE_FIRST + TILE_STEP - 1) / TILE_STEP : 0;
    constexpr uint32_t PACK_TILE = 2 * A_ROWS * 32;  // bf16: face 0 and face 1, A_ROWS rows of 32 B each
    static_assert(MY_TILES * PACK_TILE <= A_SLOT_BYTES, "one slot holds this core's packed stats");
    const uint32_t barrier_addr = get_arg_val<uint32_t>(rt + 2);
    const uint32_t data_addr = get_arg_val<uint32_t>(rt + 3);
    const uint32_t consumers = get_arg_val<uint32_t>(rt + 4);
    const uint32_t consumer_sem = get_arg_val<uint32_t>(rt + 5);
    Noc noc;
    DataflowBuffer pack_cb(A_PACK_CB);
    Semaphore<> go(SEM_GO);
    Semaphore<> scratch_ready(A_SEM_SCRATCH);
    Semaphore<> done(SEM_DONE);
    volatile tt_l1_ptr uint32_t* barrier = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_addr);
    volatile tt_l1_ptr uint32_t* data = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(data_addr);
    const uint64_t data_noc = safe_get_noc_addr(my_x[0], my_y[0], data_addr, 0);
    const uint32_t rank = line.rank;
    const uint32_t pack_base = pack_cb.get_write_ptr();
    const uint32_t stage = pack_base + RING * A_SLOT_BYTES;
#if defined(COMPILE_FOR_BRISC)
    noc_semaphore_wait_min(barrier, RING - 1);
    noc_semaphore_set(barrier, 0);  // reset before the release: every data send follows it
    if (delay_after_reset != 0) {
        riscv_wait(delay_after_reset);
    }
    go.set(1);
#else
    go.wait(1);
    go.set(0);
#endif
    scratch_ready.wait_min(MY_TILES);  // the norm cores' chunks are in the staging slot
    if (line.has_peers) {
        noc_async_writes_flushed();
        fabric_multicast_noc_fused_unicast_with_atomic_inc(
            &line.connection,
            line.header,
            stage,
            MY_TILES * PACK_TILE,
            tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{
                safe_get_noc_addr(my_x[0], my_y[0], pack_base + rank * A_SLOT_BYTES, 0), data_noc, 1, true},
            1,
            line.range);
    }
#if defined(COMPILE_FOR_BRISC)
    noc_semaphore_wait_min(data, RING - 1);  // one packet from every peer
    noc_semaphore_set(data, 0);
    for (uint32_t c = 0; c < consumers; ++c) {  // the landing slots are complete: release the norm cores' readers
        Semaphore<> ready(consumer_sem);
        ready.up(
            noc,
            get_arg_val<uint32_t>(rt + PHASE_RT_ARGS + 2 * c),
            get_arg_val<uint32_t>(rt + PHASE_RT_ARGS + 1 + 2 * c),
            1);
    }
    noc.async_atomic_barrier();
    done.wait(1);
    done.set(0);
    scratch_ready.set(0);
#else
    done.set(1);
#endif
}

template <typename OutArgs>
FORCE_INLINE void partials_phase(LineSender& line, const OutArgs& out_args, uint32_t rt) {
    constexpr uint32_t MY_TILES = TILE_FIRST < B_TILES ? (B_TILES - TILE_FIRST + TILE_STEP - 1) / TILE_STEP : 0;
    constexpr uint32_t FACE_BYTES = 1024, ROW_BYTES = 64;
    constexpr uint32_t CHUNK = B_ROWS * ROW_BYTES;
    constexpr uint32_t PACK_TILE = 2 * CHUNK;
    static_assert(MY_TILES * PACK_TILE <= B_SLOT_BYTES, "one slot holds this core's packed tiles");
    const uint32_t out_addr = get_arg_val<uint32_t>(rt + 0);
    volatile tt_l1_ptr uint32_t* counter = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(B_LAND);
    const uint32_t parity = *counter & 1;
    const uint32_t data_addr = parity ? B_DATA1 : get_arg_val<uint32_t>(rt + 3);
    const uint32_t land_base = B_LAND + 16 + parity * RING * B_SLOT_BYTES;
    const auto out = TensorAccessor(out_args, out_addr);
    Noc noc;
    DataflowBuffer stage_cb(B_STAGE_CB);
    Semaphore<> scratch_ready(B_SEM_SCRATCH);
    Semaphore<> done(SEM_DONE);
    Semaphore<> unpack(SEM_UNPACK);
    volatile tt_l1_ptr uint32_t* data = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(data_addr);
    const uint64_t data_noc = safe_get_noc_addr(my_x[0], my_y[0], data_addr, 0);
    const uint32_t rank = line.rank;
    const uint32_t stage = stage_cb.get_write_ptr();

    {
        FUSED_ZONE("fz_gr2_tr_p_wait");
        scratch_ready.wait_min(MY_TILES);  // the workers' chunks are in the staging buffer
    }
    if (line.has_peers) {
        FUSED_ZONE("fz_gr2_tr_p_send");
        noc_async_writes_flushed();
        fabric_multicast_noc_fused_unicast_with_atomic_inc(
            &line.connection,
            line.header,
            stage,
            MY_TILES * PACK_TILE,
            tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{
                safe_get_noc_addr(my_x[0], my_y[0], land_base + rank * B_SLOT_BYTES, 0), data_noc, 1, true},
            1,
            line.range);
    }
    line.close();

    auto unpack_sources =
        [&](uint32_t first) {  // sources first, first + 2, ... (not this device: its page is the workers')
            for (uint32_t r = first; r < RING; r += 2) {
                if (r == rank) {
                    continue;
                }
                for (uint32_t i = 0; i < MY_TILES; ++i) {
                    const uint32_t page = (TILE_FIRST + i * TILE_STEP) * B_PAGE_TILE_STRIDE + r * B_PAGE_RANK_STRIDE;
                    for (uint32_t k = 0; k < 2; ++k) {
                        noc_async_write(
                            land_base + r * B_SLOT_BYTES + i * PACK_TILE + k * CHUNK,
                            out.get_noc_addr(page, k * FACE_BYTES),
                            CHUNK);
                    }
                }
            }
            noc_async_write_barrier();
        };
#if defined(COMPILE_FOR_BRISC)
    {
        FUSED_ZONE("fz_gr2_tr_p_data");
        noc_semaphore_wait_min(data, RING - 1);  // one packet from every peer
    }
    noc_semaphore_set(data, 0);
    unpack.set(1);
    {
        FUSED_ZONE("fz_gr2_tr_p_unpack");
        unpack_sources(0);
    }
    done.wait(1);
    done.set(0);
    scratch_ready.set(0);
    *counter = *counter + 1;  // after NCRISC's read at the top of the phase (its done.set(1) follows)
#else
    unpack.wait(1);
    unpack.set(0);
    unpack_sources(1);
    done.set(1);
#endif
}

void kernel_main() {
    const uint32_t rank = get_arg_val<uint32_t>(0);
    const uint32_t delay_before_arrive = get_arg_val<uint32_t>(1);
    const uint32_t delay_after_reset = get_arg_val<uint32_t>(2);
    const uint32_t b_rt = A_RT + PHASE_RT_ARGS + 2 * get_arg_val<uint32_t>(A_RT + 4);
    size_t arg_idx = b_rt + PHASE_RT_ARGS + 2 * get_arg_val<uint32_t>(b_rt + 4);
    constexpr auto a_out_args = TensorAccessorArgs<ACCESSOR_BASE>();
    constexpr auto a_local_args = TensorAccessorArgs<a_out_args.next_compile_time_args_offset()>();
    constexpr auto b_out_args = TensorAccessorArgs<a_local_args.next_compile_time_args_offset()>();
    LineSender line;
    line.open<RING>(rank, arg_idx);
    if (delay_before_arrive != 0) {
        riscv_wait(delay_before_arrive);
    }
    {
        FUSED_ZONE("fz_gr2_tr_stats");
        line.arrive(get_arg_val<uint32_t>(A_RT + 2));
        stats_phase(line, a_out_args, A_RT, delay_after_reset);
    }
    {
        FUSED_ZONE("fz_gr2_tr_partials");
        partials_phase(line, b_out_args, b_rt);
    }
    noc_async_full_barrier();
}
