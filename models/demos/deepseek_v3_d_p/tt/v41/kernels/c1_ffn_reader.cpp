// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader of the decode-shaped routed expert FFN of the DeepSeek-V4.1 c1 MoE (tt/v41/moe_c1.py C1FfnMatvec,
// ``V41_DECODE_C1_FFN=matvec``). One token row (row 0 of the bf16 tile-row ``x`` [32, hidden],
// the other rows zero) through the chip's selected experts (``counts[idx[l]] > 0``), serially per expert, with the
// whole core grid reading one expert's weights at once instead of 11 column senders multicasting them:
//
//   1. ``x``: X_SENDERS cores read a stripe of the row's K tiles each into core 0's CB, which multicasts the row
//      into the same CB of every core, then multicasts the go; under the first expert's first weight reads, which
//      every core issues before it.
//   2. gate/up: core c owns N column tiles [TPC c, TPC c + TPC) (TPC 1 or 2; adjacent tiles of a shard row read
//      as one request): it streams gate[kb, c] and up[kb, c] over the K tiles
//      (BLK_GU at a time, two blocks in flight by transaction id, a three-block ring) for the compute's two
//      one-tile dot products and the fused activation.
//   3. the activated tile of every column is written to slot c of core 0's staging CB (unicast); core 0 counts
//      the NT_GU producers, multicasts the staged row into slot (expert mod 3) of ``cb_actf`` on every core and
//      multicasts the go. A slot is reused three experts later, after every core has reported its compute done
//      with it (the ``done`` token, ``sem_done``). 72 producers each multicasting to the grid cost ~100 us an
//      expert (the first attempt); one staged multicast is the fix.
//   4. down: core c owns output tiles nd[j] (j < ND): it streams down[k, nd[j]] over the K tiles (the first ring
//      blocks issued before the activation gather so the DRAM does not idle through it); the writer puts the
//      output tiles into expert l's region of ``y``.
//
// Every multicast is on this RISC's NoC (NoC 0): a worker multicast on NoC 1 collides with the fabric's CCL
// traffic (unified_routed_expert_ffn's UP_WRITER_MCAST). The exit drains every read, non-posted write and atomic.
// Acks landing after the next kernel's NIU snapshot hang its first barrier.
//
// compile_time_args = [cb_x, cb_gate, cb_up, cb_down, cb_act, cb_actf, cb_scr, cb_done, cb_stage, EPC, E, KT,
//                      NT_GU, NGU_FULL, KD, NT_D, ND_FULL, BLK_GU, BLK_D, GRID_N, X_TILE, W_TILE, ACT_TILE,
//                      ACT_SLOTS, PROBE, TPC, SHARD_GU, SHARD_D, RING_GU, RING_D, X_SENDERS, X_PER_SENDER,
//                      TensorAccessorArgs(x)..., (gate)..., (up)..., (down)..., (counts)..., (idx)...]
// runtime args = [x_addr, counts_addr, idx_addr, my_ngu, nd0, nd1, is_core0, core0_nx, core0_ny, grid_x0,
//                      grid_y0, grid_x1, grid_y1, sem_xgo, sem_act, sem_actgo, sem_done, sem_xready, x_sender,
//                      x_first, gate_addr[EPC], up_addr[EPC], down_addr[EPC]]

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"

constexpr uint32_t NONE = 0xFFFFFFFFu;

void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t counts_addr = get_arg_val<uint32_t>(1);
    const uint32_t idx_addr = get_arg_val<uint32_t>(2);
    const uint32_t my_ngu = get_arg_val<uint32_t>(3);
    const uint32_t nd[2] = {get_arg_val<uint32_t>(4), get_arg_val<uint32_t>(5)};
    const uint32_t is_core0 = get_arg_val<uint32_t>(6);
    const uint32_t core0_nx = get_arg_val<uint32_t>(7);
    const uint32_t core0_ny = get_arg_val<uint32_t>(8);
    const uint32_t gx0 = get_arg_val<uint32_t>(9);
    const uint32_t gy0 = get_arg_val<uint32_t>(10);
    const uint32_t gx1 = get_arg_val<uint32_t>(11);
    const uint32_t gy1 = get_arg_val<uint32_t>(12);
    const uint32_t sem_xgo_id = get_arg_val<uint32_t>(13);
    const uint32_t sem_act_id = get_arg_val<uint32_t>(14);
    const uint32_t sem_actgo_id = get_arg_val<uint32_t>(15);
    const uint32_t sem_done_id = get_arg_val<uint32_t>(16);
    const uint32_t sem_xready_id = get_arg_val<uint32_t>(17);
    const uint32_t x_sender = get_arg_val<uint32_t>(18);
    const uint32_t x_first = get_arg_val<uint32_t>(19);
    constexpr uint32_t WADDR = 20;

    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_gate = get_compile_time_arg_val(1);
    constexpr uint32_t cb_up = get_compile_time_arg_val(2);
    constexpr uint32_t cb_down = get_compile_time_arg_val(3);
    constexpr uint32_t cb_act = get_compile_time_arg_val(4);
    constexpr uint32_t cb_actf = get_compile_time_arg_val(5);
    constexpr uint32_t cb_scr = get_compile_time_arg_val(6);
    constexpr uint32_t cb_done = get_compile_time_arg_val(7);
    constexpr uint32_t cb_stage = get_compile_time_arg_val(8);
    constexpr uint32_t EPC = get_compile_time_arg_val(9);
    constexpr uint32_t E = get_compile_time_arg_val(10);
    constexpr uint32_t KT = get_compile_time_arg_val(11);
    constexpr uint32_t NT_GU = get_compile_time_arg_val(12);
    constexpr uint32_t NGU_FULL = get_compile_time_arg_val(13);
    constexpr uint32_t KD = get_compile_time_arg_val(14);
    constexpr uint32_t NT_D = get_compile_time_arg_val(15);
    constexpr uint32_t ND_FULL = get_compile_time_arg_val(16);
    constexpr uint32_t BLK_GU = get_compile_time_arg_val(17);
    constexpr uint32_t BLK_D = get_compile_time_arg_val(18);
    constexpr uint32_t GRID_N = get_compile_time_arg_val(19);
    constexpr uint32_t X_TILE = get_compile_time_arg_val(20);
    constexpr uint32_t W_TILE = get_compile_time_arg_val(21);
    constexpr uint32_t ACT_TILE = get_compile_time_arg_val(22);
    constexpr uint32_t ACT_SLOTS = get_compile_time_arg_val(23);
    constexpr uint32_t PROBE = get_compile_time_arg_val(24);  // 1 / 3: no weight reads (a timing probe, wrong output)
    constexpr uint32_t TPC = get_compile_time_arg_val(25);    // gate/up column tiles a core (1 or 2)
    constexpr uint32_t SHARD_GU = get_compile_time_arg_val(26);  // tiles a DRAM shard row (0: interleaved)
    constexpr uint32_t SHARD_D = get_compile_time_arg_val(27);
    constexpr uint32_t RING_GU = get_compile_time_arg_val(28);  // blocks in the gate/up rings
    constexpr uint32_t RING_D = get_compile_time_arg_val(29);   // two-column blocks in the down ring
    constexpr uint32_t X_SENDERS = get_compile_time_arg_val(30);
    constexpr uint32_t X_PER_SENDER = get_compile_time_arg_val(31);
    constexpr auto x_args = TensorAccessorArgs<32>();
    constexpr auto gate_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto up_args = TensorAccessorArgs<gate_args.next_compile_time_args_offset()>();
    constexpr auto down_args = TensorAccessorArgs<up_args.next_compile_time_args_offset()>();
    constexpr auto counts_args = TensorAccessorArgs<down_args.next_compile_time_args_offset()>();
    constexpr auto idx_args = TensorAccessorArgs<counts_args.next_compile_time_args_offset()>();
    static_assert(KT % BLK_GU == 0 && KD % BLK_D == 0, "the K blocks must tile K");
    static_assert(RING_GU >= 3 && RING_D >= 3, "two blocks in flight beside the one the compute reads");
    constexpr uint32_t NB_GU = KT / BLK_GU;
    constexpr uint32_t NGU_CORES = (NT_GU + TPC - 1) / TPC;  // the gate/up producers
    constexpr uint32_t NB_D = KD / BLK_D;
    (void)NT_D;

    const auto x = TensorAccessor(x_args, x_addr, X_TILE);
    const auto counts = TensorAccessor(counts_args, counts_addr);
    const auto idx = TensorAccessor(idx_args, idx_addr);
    DataflowBuffer xb(cb_x), gate(cb_gate), up(cb_up), down(cb_down), act(cb_act), actf(cb_actf), scr(cb_scr);
    DataflowBuffer done(cb_done), stage(cb_stage);
    volatile tt_l1_ptr uint32_t* sem_xgo = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_xgo_id));
    volatile tt_l1_ptr uint32_t* sem_act = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_act_id));
    volatile tt_l1_ptr uint32_t* sem_actgo =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_actgo_id));
    volatile tt_l1_ptr uint32_t* sem_done = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_done_id));
    const uint64_t core0_act = get_noc_addr(core0_nx, core0_ny, get_semaphore(sem_act_id));
    const uint64_t core0_done = get_noc_addr(core0_nx, core0_ny, get_semaphore(sem_done_id));
    const uint64_t core0_xready = get_noc_addr(core0_nx, core0_ny, get_semaphore(sem_xready_id));
    volatile tt_l1_ptr uint32_t* sem_xready =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_xready_id));
    const uint64_t grid_actgo = get_noc_multicast_addr(gx0, gy0, gx1, gy1, get_semaphore(sem_actgo_id));
    const uint32_t nd_count = (nd[0] == NONE) ? 0 : ((nd[1] == NONE) ? 1 : 2);
    // adjacent column tiles of one shard row are contiguous in one DRAM bank: read them as one request
    const bool nd_pair = (nd_count == 2) && (nd[1] == nd[0] + 1) && (SHARD_D == 0 || (nd[0] % SHARD_D) + 1 < SHARD_D);
    const uint32_t d_blk_tiles = BLK_D * nd_count;
    // the down ring holds RING blocks of two columns; a one-column core's blocks are half the size, twice as many
    const uint32_t d_ring_blocks = (nd_count > 0) ? (RING_D * BLK_D * 2) / d_blk_tiles : 0;

    // the counts and the chip's expert table, for every RISC of this core (the scratch page: counts, then idx)
    scr.reserve_back(1);
    const uint32_t scr_l1 = (scr.get_write_ptr() + 63) & ~63u;  // the page is 64 bytes larger than its contents
    noc_async_read(counts.get_noc_addr(0, 0), scr_l1, E * 4);
    noc_async_read(idx.get_noc_addr(0, 0), scr_l1 + E * 4, EPC * 4);
    noc_async_read_barrier();
    scr.push_back(1);
    volatile tt_l1_ptr uint32_t* cnt = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr_l1);
    volatile tt_l1_ptr uint32_t* table = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr_l1 + E * 4);
    uint32_t selected = 0;
    for (uint32_t l = 0; l < EPC; ++l) {
        selected += (cnt[table[l]] != 0);
    }
    if (selected == 0) {
        return;  // every core of the chip reads the same page and skips the same way
    }

    xb.reserve_back(KT);
    const uint32_t x_l1 = xb.get_write_ptr();
    const uint64_t core0_x = get_noc_addr(core0_nx, core0_ny, x_l1);
    // the staging row on core 0 and the activated row's slots: the CBs' first addresses
    stage.reserve_back(NT_GU);
    const uint32_t stage_l1 = stage.get_write_ptr();
    actf.reserve_back(NT_GU);
    const uint32_t actf_l1 = actf.get_write_ptr();
    const uint32_t gate_l1 = gate.get_write_ptr();
    const uint32_t up_l1 = up.get_write_ptr();
    const uint32_t down_l1 = down.get_write_ptr();
    const uint64_t core0_stage =
        get_noc_addr(core0_nx, core0_ny, stage_l1 + (my_ngu == NONE ? 0 : my_ngu * TPC) * ACT_TILE);
    const uint32_t my_tiles = (my_ngu == NONE) ? 0 : ((my_ngu * TPC + TPC <= NT_GU) ? TPC : NT_GU - my_ngu * TPC);

    // the weight streams: block gb (counted over the experts) sits in ring slot gb % RING and carries transaction
    // id 1 + gb % 2, so two blocks are in flight while the compute consumes the one before them
    // 1. the row to every core, once: core 0 reads it and multicasts it, under the first expert's weight reads
    auto x_prologue = [&]() {
        if (x_sender) {
            // X_SENDERS cores read a stripe each (one core reading all 160 tiles took ~15 us) and put it into core
            // 0's row; transaction id 3 (1 and 2 are the weight blocks'): a barrier on id 0 never returned (the
            // sixth attempt)
            noc_async_read_set_trid(3);
            for (uint32_t t = x_first; t < x_first + X_PER_SENDER; ++t) {
                noc_async_read(x.get_noc_addr(t), x_l1 + t * X_TILE, X_TILE);
            }
            noc_async_read_barrier_with_trid(3);
            noc_async_write(x_l1 + x_first * X_TILE, core0_x + x_first * X_TILE, X_PER_SENDER * X_TILE);
            noc_async_write_barrier();
            noc_semaphore_inc(core0_xready, 1);
        }
        if (is_core0) {
            noc_semaphore_wait_min(sem_xready, X_SENDERS);
            noc_async_write_multicast(x_l1, get_noc_multicast_addr(gx0, gy0, gx1, gy1, x_l1), KT * X_TILE, GRID_N - 1);
            noc_async_write_barrier();
            *sem_xgo = 1;
            noc_semaphore_set_multicast(
                get_semaphore(sem_xgo_id),
                get_noc_multicast_addr(gx0, gy0, gx1, gy1, get_semaphore(sem_xgo_id)),
                GRID_N - 1);
        }
        noc_semaphore_wait_min(sem_xgo, 1);
        xb.push_back(KT);
    };
    uint32_t gb_gu = 0, gb_d = 0;
    uint32_t eidx = 0;
    for (uint32_t l = 0; l < EPC; ++l) {
        if (cnt[table[l]] == 0) {
            continue;
        }
        const auto gate_w = TensorAccessor(gate_args, get_arg_val<uint32_t>(WADDR + l), W_TILE);
        const auto up_w = TensorAccessor(up_args, get_arg_val<uint32_t>(WADDR + EPC + l), W_TILE);
        const auto down_w = TensorAccessor(down_args, get_arg_val<uint32_t>(WADDR + 2 * EPC + l), W_TILE);
        auto issue_gu = [&](uint32_t b) {
            const uint32_t slot = (gb_gu + b) % RING_GU;
            noc_async_read_set_trid(1 + ((gb_gu + b) & 1));
            for (uint32_t t = 0; t < BLK_GU && (PROBE & 1) == 0; ++t) {
                const uint32_t n0 = my_ngu * TPC;
                const uint32_t page = (b * BLK_GU + t) * NGU_FULL + n0;
                const uint32_t l1 = (slot * BLK_GU + t) * TPC * W_TILE;
                uint32_t j = 0;
                while (j < my_tiles) {
                    uint32_t run = my_tiles - j;
                    if (SHARD_GU > 0) {
                        const uint32_t to_edge = SHARD_GU - ((n0 + j) % SHARD_GU);
                        run = (to_edge < run) ? to_edge : run;
                    } else {
                        run = 1;
                    }
                    noc_async_read(gate_w.get_noc_addr(page + j), gate_l1 + l1 + j * W_TILE, run * W_TILE);
                    noc_async_read(up_w.get_noc_addr(page + j), up_l1 + l1 + j * W_TILE, run * W_TILE);
                    j += run;
                }
            }
        };
        auto issue_d = [&](uint32_t b) {
            const uint32_t slot = (gb_d + b) % d_ring_blocks;
            noc_async_read_set_trid(1 + ((gb_d + b) & 1));
            for (uint32_t t = 0; t < BLK_D && (PROBE & 1) == 0; ++t) {
                const uint32_t l1 = down_l1 + (slot * d_blk_tiles + t * nd_count) * W_TILE;
                if (nd_pair) {
                    noc_async_read(down_w.get_noc_addr((b * BLK_D + t) * ND_FULL + nd[0]), l1, 2 * W_TILE);
                } else {
                    for (uint32_t j = 0; j < nd_count; ++j) {
                        noc_async_read(down_w.get_noc_addr((b * BLK_D + t) * ND_FULL + nd[j]), l1 + j * W_TILE, W_TILE);
                    }
                }
            }
        };
        // 2. this column's gate and up weights over K
        if (my_ngu != NONE) {
            gate.reserve_back(2 * BLK_GU * TPC);
            up.reserve_back(2 * BLK_GU * TPC);
            issue_gu(0);
            if (NB_GU > 1) {
                issue_gu(1);
            }
        }
        if (eidx == 0) {
            x_prologue();
        }
        if (my_ngu != NONE) {
            for (uint32_t b = 0; b < NB_GU; ++b) {
                noc_async_read_barrier_with_trid(1 + ((gb_gu + b) & 1));
                gate.push_back(BLK_GU * TPC);
                up.push_back(BLK_GU * TPC);
                if (b + 2 < NB_GU) {
                    gate.reserve_back(2 * BLK_GU * TPC);
                    up.reserve_back(2 * BLK_GU * TPC);
                    issue_gu(b + 2);
                }
            }
            gb_gu += NB_GU;
        }
        // 4a. the first down blocks, issued before the gather so the DRAM stays busy through it
        if (nd_count > 0) {
            down.reserve_back(2 * d_blk_tiles);
            issue_d(0);
            if (NB_D > 1) {
                issue_d(1);
            }
        }
        // 3. the activated tile to core 0's staging row; core 0 gathers and multicasts the row
        if (my_ngu != NONE) {
            act.wait_front(TPC);
            noc_async_write(act.get_read_ptr(), core0_stage, my_tiles * ACT_TILE);
            noc_async_write_barrier();
            act.pop_front(TPC);
            noc_semaphore_inc(core0_act, 1);
        }
        if (eidx > 0) {
            // this core's compute has released the previous expert's slot (its down matmul is done)
            done.wait_front(1);
            done.pop_front(1);
            if (eidx >= ACT_SLOTS - 1) {
                noc_semaphore_inc(core0_done, 1);
            }
        }
        if (is_core0) {
            noc_semaphore_wait_min(sem_act, NGU_CORES * (eidx + 1));
            if (eidx >= ACT_SLOTS) {
                // slot eidx % ACT_SLOTS was expert eidx - ACT_SLOTS's: every core has reported that one done
                noc_semaphore_wait_min(sem_done, GRID_N * (eidx - ACT_SLOTS + 1));
            }
            const uint32_t slot = actf_l1 + (eidx % ACT_SLOTS) * NT_GU * ACT_TILE;
            noc_async_write_multicast_loopback_src(
                stage_l1, get_noc_multicast_addr(gx0, gy0, gx1, gy1, slot), NT_GU * ACT_TILE, GRID_N);
            noc_async_write_barrier();
            *sem_actgo = eidx + 1;
            noc_semaphore_set_multicast(get_semaphore(sem_actgo_id), grid_actgo, GRID_N - 1);
        }
        noc_semaphore_wait_min(sem_actgo, eidx + 1);
        if (eidx > 0) {
            actf.reserve_back(NT_GU);  // the first reserve was taken above
        }
        actf.push_back(NT_GU);
        // 4b. the rest of this core's down weights over K
        if (nd_count > 0) {
            for (uint32_t b = 0; b < NB_D; ++b) {
                noc_async_read_barrier_with_trid(1 + ((gb_d + b) & 1));
                down.push_back(d_blk_tiles);
                if (b + 2 < NB_D) {
                    down.reserve_back(2 * d_blk_tiles);
                    issue_d(b + 2);
                }
            }
            gb_d += NB_D;
        }
        ++eidx;
    }
    noc_async_read_set_trid(0);
    noc_async_read_barrier();
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
