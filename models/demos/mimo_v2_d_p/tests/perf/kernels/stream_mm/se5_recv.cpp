// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Chained M-split big-M receiver data movement (BRISC, NOC0) on a compute core of M-group G. Both activations travel
// down chains of the group's cores by unicast: x from a chain head's DRAM reader (se4_xhead.cpp); h is gathered
// directly into every chain head of the group (each core writes its slice into all of them) and forwarded from there.
// A non-blocking event loop that
//   * core A of a pair: runs the in1 ring (one credit per free slot to the forwarder, publishes blocks as they land,
//   and
//     tells partner B each one has landed: AVAIL); a slot is only credited again once compute has freed it and B has
//     copied it (BCOPY, written by B's se5_bnc.cpp). Core B: its NCRISC pulls the blocks from A, nothing to do here.
//     The ring holds one expert's weight slice and compute frees blocks after their last use,
//   * publishes x blocks as they arrive (XARR, incremented by the predecessor after the block's write is acknowledged),
//     forwards each block into the successor's x ring slot once the successor has freed it (SFREE, the successor's
//     freed count), then bumps the successor's XARR; a slot counts as freed here once compute has consumed it and it
//     has been forwarded, and that count goes to the predecessor's SFREE (or, on a chain head, to its own HFREE, which
//     its DRAM reader waits on),
//   * sends its h slice of each virtual expert v ([MT x NP] tiles, h K-tiles CG * NP + p) into the h_all of every chain
//     head of its group with a GATH increment there, once go >= v (every core done with down(v - 1), which means
//     every head has consumed and forwarded h(v - 1)); a head counts h(v) as fully arrived once all GROUP_NCC slices
//     are in,
//   * receives h in H_PIECES pieces (HARR counts pieces, incremented by the predecessor / relay after each piece's
//     write is acknowledged; single buffer), forwards each piece into the successor's h_all as soon as it has arrived
//     and the successor has freed its h_all (HSFREE, in whole h), publishes h_all(v) to compute once all its pieces
//     are in, and reports min(consumed, forwarded) h to the predecessor's free word (chain heads have none),
//   * drains each output into an S-deep ring (the last expert's S sub-blocks stay) and reports it to the coordinator,
//     which counts the virtual experts every core has finished ("go").
//
// CT: 0 X_CB, 1 X_BLK_TILES, 2 IN1_CB, 3 SLOT_TILES, 4 TOTAL_W_BLOCKS, 5 NUM_V, 6 W_SLOTS, 7 OUT_CB, 8 OUT_TILES,
//     9 H_LOCAL_CB, 10 H_ALL_CB, 11 MT, 12 H_TILE_BYTES, 13 NCC, 14 DATA_SEM, 15 HARR_SEM, 16 GO_SEM, 17 DONE_SEM,
//     18 KBLK, 19 HARR1_SEM, 20 SFREE_SEM, 21 HBUF, 22 XARR_SEM, 23 HFREE_SEM, 24 X_SLOTS, 25 X_BYTES, 26 S, 27 NP,
//     28 HSFREE_SEM, 29 H_PIECES, 30 X_PER_V (x blocks per virtual expert), 31-32 (SE_GU_ONLY) GATH ids of h buffers
//     1, 2, 33 X_PIECES (x blocks travel the chain cut-through in this many pieces; XARR counts pieces), 34 G (M-groups
//     sharing the down cores' h_all)
// SE_X_RELAY2 = run, SE_X_NRELAY = n (with SE_X_RELAY): n relays take turns, runs of `run` blocks each (relay k the
//     runs k, k + n, ...), each counting its own blocks in its own semaphore (XARR, 3, 2, 1 for relays 0..3); relay xy
//     in RT 11, 15, 16, 17 (every relay keeps a copy of this core's freed word at RT 12).
// SE_X_RELAY: x comes from a relay's multicast (XARR set by it, no chain); freed counts go to this core's word on the
//     relay (RT 11 relay xy, RT 12 word address). SE_NO_PARTNER: no core B (weights are not gated on its copies).
// SE_W_NCRISC: no weight handling here at all (the NCRISCs read / pull the weights).
// SE_GU_ONLY (spatially pipelined variant): gate/up only; h slices go into the down cores' chain heads (the h_all of
// both M-groups, group G at G * HALF), no h / output handling here, "go" comes from the down cores' coordinator.
// RT: 0 forwarder xy, 1 credit sem id on the forwarder, 2 column group CG, 3 NH (chain heads of the group), 4 unused,
//     5 coordinator xy, 6 group's GATH sem id, 7 x ring address, 8 is coordinator, 9 predecessor xy (0: chain
//     head), 10 successor xy (0: chain tail), 11 h predecessor xy (0: chain head), 12 sem id of this core's h-free word
//     on the h predecessor, 13 h_all address, 14 role (0: A, 1: B), 15 partner xy, 16 AVAIL sem id (on B), 17 BCOPY
//     sem id (on A), 18 second x successor xy (0: none; x travels down a tree), 19 sem id of this core's x-free word
//     on its predecessor (SFREE or SFREE2 there), 20 sem id of the second successor's free word here, 21.. NH head xy,
//     then NCC compute-core xy
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#ifdef SE_DYN
#include "se_dyn.hpp"
#endif
#ifdef SE_ZONES
#include "tools/profiler/kernel_profiler.hpp"
#define SE_MARK(name)            \
    {                            \
        DeviceZoneScopedN(name); \
    }
#else
#define SE_MARK(name)
#endif

void kernel_main() {
    constexpr uint32_t x_cb = get_compile_time_arg_val(0);
    constexpr uint32_t x_blk = get_compile_time_arg_val(1);
    constexpr uint32_t in1_cb = get_compile_time_arg_val(2);
    constexpr uint32_t slot = get_compile_time_arg_val(3);
    constexpr uint32_t total_w_ct = get_compile_time_arg_val(4);
    constexpr uint32_t num_v_ct = get_compile_time_arg_val(5);
    constexpr uint32_t w_slots = get_compile_time_arg_val(6);
    constexpr uint32_t out_cb = get_compile_time_arg_val(7);
    constexpr uint32_t out_tiles = get_compile_time_arg_val(8);
    constexpr uint32_t h_local_cb = get_compile_time_arg_val(9);
    constexpr uint32_t h_all_cb = get_compile_time_arg_val(10);
    constexpr uint32_t mt = get_compile_time_arg_val(11);
    constexpr uint32_t h_tile_bytes = get_compile_time_arg_val(12);
    constexpr uint32_t ncc = get_compile_time_arg_val(13);
    constexpr uint32_t data_sem_id = get_compile_time_arg_val(14);
    constexpr uint32_t harr_sem_id = get_compile_time_arg_val(15);
    constexpr uint32_t go_sem_id = get_compile_time_arg_val(16);
    constexpr uint32_t done_sem_id = get_compile_time_arg_val(17);
    constexpr uint32_t kblk = get_compile_time_arg_val(18);
    constexpr uint32_t harr1_sem_id = get_compile_time_arg_val(19);
    constexpr uint32_t hbuf = get_compile_time_arg_val(21);
    constexpr uint32_t xarr_sem_id = get_compile_time_arg_val(22);
    constexpr uint32_t x_slots = get_compile_time_arg_val(24);
    constexpr uint32_t total_x = get_compile_time_arg_val(25);
    constexpr uint32_t sfree_sem_id = get_compile_time_arg_val(20);
    constexpr uint32_t hfree_sem_id = get_compile_time_arg_val(23);
    constexpr uint32_t x_bytes = get_compile_time_arg_val(25);
    constexpr uint32_t sub = get_compile_time_arg_val(26);
    constexpr uint32_t np = get_compile_time_arg_val(27);
    constexpr uint32_t hsfree_sem_id = get_compile_time_arg_val(28);
    constexpr uint32_t h_pieces = get_compile_time_arg_val(29);
    constexpr uint32_t x_per_v = get_compile_time_arg_val(30);
    constexpr uint32_t xp = get_compile_time_arg_val(33);
    constexpr uint32_t groups = get_compile_time_arg_val(34);
    constexpr uint32_t h_all_tiles = (ncc / groups) * np * mt;  // the group's h: NCC / G cores x NP K-tiles, MT rows
    constexpr uint32_t half_bytes = h_all_tiles * h_tile_bytes;
    constexpr uint32_t h_piece_bytes = half_bytes / h_pieces;
#ifdef SE_GU_ONLY
    static_assert(hbuf >= 1 && hbuf <= 4);
#else
    static_assert(hbuf == 1 || hbuf == 2);
#endif

    const uint32_t fxy = get_arg_val<uint32_t>(0);
    const uint32_t credit_sem_id = get_arg_val<uint32_t>(1);
    const uint32_t cg = get_arg_val<uint32_t>(2);
    const uint32_t nh = get_arg_val<uint32_t>(3);
    const uint32_t cxy = get_arg_val<uint32_t>(5);
    const uint32_t gath_sem_id = get_arg_val<uint32_t>(6);
    const uint32_t x_ring = get_arg_val<uint32_t>(7);
    const uint32_t pxy = get_arg_val<uint32_t>(9), sxy = get_arg_val<uint32_t>(10);
    const bool is_coord = get_arg_val<uint32_t>(8) != 0;
    auto sem = [](uint32_t id) { return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(id)); };
    const uint64_t credit_noc = get_noc_addr(fxy >> 16, fxy & 0xFFFF, get_semaphore(credit_sem_id));
    const uint64_t done_noc = get_noc_addr(cxy >> 16, cxy & 0xFFFF, get_semaphore(done_sem_id));
    volatile tt_l1_ptr uint32_t* gath_sem = sem(gath_sem_id);
    const uint32_t peer0 = 21 + nh;
#ifdef SE_DYN
    // Dynamic counts: the active experts' weights and sub-blocks only (CT 4 is then blocks per expert x NUM_E, CT 5
    // unused); RT 21 + NH.. are the se_dyn.hpp args, CB 7 this RISC's scratch; the compute gets them in CB 6.
    constexpr uint32_t num_e_dyn = get_compile_time_arg_val(35);
    SeDyn dyn;
    se_dyn_load<num_e_dyn>(dyn, peer0, get_write_ptr(tt::CBIndex::c_7), mt * 32);
    se_dyn_publish(dyn, tt::CBIndex::c_6);
    const uint32_t total_w = dyn.n_act * (total_w_ct / num_e_dyn);
    const uint32_t num_v = dyn.num_v;
#else
    constexpr uint32_t total_w = total_w_ct;
    constexpr uint32_t num_v = num_v_ct;
#endif
#ifdef SE_W_NCRISC
    const bool is_b = true;  // every core's weights are handled by its NCRISC (se7_anc.cpp on A, se5_bnc.cpp on B)
#else
    const bool is_b = get_arg_val<uint32_t>(14) != 0;
#endif
    const uint32_t partner = get_arg_val<uint32_t>(15);
    const uint64_t partner_avail =
        get_noc_addr(partner >> 16, partner & 0xFFFF, get_semaphore(get_arg_val<uint32_t>(16)));
    volatile tt_l1_ptr uint32_t* bcopy = sem(get_arg_val<uint32_t>(17));
#ifdef SE_NO_PARTNER
    constexpr bool no_partner = true;
#else
    constexpr bool no_partner = false;
#endif
#ifdef SE_X_RELAY
    const uint32_t hpxy = 0;  // RT 11/12 name the x relay; the gate/up cores all send h straight to the down heads
#else
    const uint32_t hpxy = get_arg_val<uint32_t>(11);
#endif
    const uint64_t hpred_free = get_noc_addr(hpxy >> 16, hpxy & 0xFFFF, get_semaphore(get_arg_val<uint32_t>(12)));
    const uint32_t h_all_addr = get_arg_val<uint32_t>(13);
    volatile tt_l1_ptr uint32_t* hsfree = sem(hsfree_sem_id);
    const uint64_t succ_harr = get_noc_addr(sxy >> 16, sxy & 0xFFFF, get_semaphore(harr_sem_id));
    const uint64_t succ_hall = get_noc_addr(sxy >> 16, sxy & 0xFFFF, h_all_addr);
    uint32_t h_cons = 0, h_fwd = 0, h_iss = 0, h_rep = 0;
    const bool h_head = hpxy == 0;
    constexpr uint32_t group_ncc = ncc / 2;
    constexpr uint32_t link_depth = 3;
    auto write_trid = [](uint32_t src, uint64_t dst, uint32_t bytes, uint32_t trid) {
        for (uint32_t o = 0; o < bytes; o += NOC_MAX_BURST_SIZE) {
            const uint32_t n = bytes - o < NOC_MAX_BURST_SIZE ? bytes - o : NOC_MAX_BURST_SIZE;
            noc_async_write_one_packet_with_trid(src + o, dst + o, n, trid);
        }
    };
    volatile tt_l1_ptr uint32_t* sfree = sem(sfree_sem_id);
    volatile tt_l1_ptr uint32_t* hfree = sem(hfree_sem_id);
    const uint64_t pred_sfree = get_noc_addr(pxy >> 16, pxy & 0xFFFF, get_semaphore(get_arg_val<uint32_t>(19)));
    // Up to two x successors (a tree keeps the delivery skew across the group low); each has its own free word here.
    const uint32_t s_xy[2] = {sxy, get_arg_val<uint32_t>(18)};
    volatile tt_l1_ptr uint32_t* s_free[2] = {sem(sfree_sem_id), sem(get_arg_val<uint32_t>(20))};
    uint64_t s_ring[2], s_xarr[2];
    for (uint32_t k = 0; k < 2; ++k) {
        s_ring[k] = get_noc_addr(s_xy[k] >> 16, s_xy[k] & 0xFFFF, x_ring);
        s_xarr[k] = get_noc_addr(s_xy[k] >> 16, s_xy[k] & 0xFFFF, get_semaphore(xarr_sem_id));
    }
    const uint32_t nsucc = s_xy[1] ? 2 : (s_xy[0] ? 1 : 0);
    volatile tt_l1_ptr uint32_t* data_sem = sem(data_sem_id);
    volatile tt_l1_ptr uint32_t* done_sem = sem(done_sem_id);
    volatile tt_l1_ptr uint32_t* go_sem = sem(go_sem_id);
    volatile tt_l1_ptr uint32_t* xarr_sem = sem(xarr_sem_id);
    volatile tt_l1_ptr uint32_t* harr_sem[2] = {sem(harr_sem_id), sem(harr1_sem_id)};

    if (!is_b) {
        noc_semaphore_inc(credit_noc, w_slots);  // every landing slot starts free
    }
    uint32_t granted = w_slots, pushed = 0;
    uint32_t x_pub = 0, x_cons = 0, x_rep = 0;
    uint32_t x_iss[2] = {0, 0}, x_fwd[2] = {0, 0};
    uint32_t h_sent = 0, h_pending = 0, h_pub = 0;
    uint32_t out_done = 0, out_popped = 0, go_sent = 0;

    uint32_t iters = 0;
#ifdef SE_GU_ONLY
    const uint32_t my_group = get_arg_val<uint32_t>(14);
    while (h_sent < num_v || (nsucc > 0 && x_fwd[0] < num_v * x_per_v * xp) ||
           (nsucc > 1 && x_fwd[1] < num_v * x_per_v * xp) || (!is_b && pushed < total_w)) {
#else
    while (out_done < num_v) {
#endif
        invalidate_l1_cache();
        if (++iters % 2048 == 0) {
            SE_MARK("LOOP2K");
        }
        while (!is_b && granted < total_w && cb_pages_reservable_at_back(in1_cb, (granted - pushed + 1) * slot) &&
               (no_partner || *bcopy >= granted + 1 - w_slots)) {
            noc_semaphore_inc(credit_noc, 1);
            ++granted;
        }
        while (!is_b && *data_sem > pushed) {
            cb_push_back(in1_cb, slot);
            ++pushed;
            if (!no_partner) {
                noc_semaphore_inc(partner_avail, 1);  // the forwarder's write is acknowledged before DATA moves
            }
            if (pushed % 8 == 0) {
                SE_MARK("W_LANDED");
            }
        }
#ifdef SE_X_RELAY2
        // Two relays alternate runs of SE_X_RELAY2 blocks (a super-block; relay A the even runs, B the odd ones), each
        // counting its own blocks (A in XARR, B in semaphore 3): the contiguous prefix ends at the first block either
        // has not delivered.
        constexpr uint32_t run = SE_X_RELAY2, nrl = SE_X_NRELAY;
        constexpr uint32_t xsem[4] = {xarr_sem_id, 3, 2, 1};
        uint32_t arrived_p = 0xFFFFFFFF;
        for (uint32_t k = 0; k < nrl; ++k) {
            const uint32_t xk = *sem(xsem[k]);
            const uint32_t pos = (xk / run) * nrl * run + k * run + xk % run;
            arrived_p = pos < arrived_p ? pos : arrived_p;
        }
#else
        const uint32_t arrived_p = *xarr_sem;  // pieces
#endif
        const uint32_t arrived = arrived_p / xp;  // whole blocks
        while (arrived > x_pub) {
            cb_push_back(x_cb, x_blk);
            ++x_pub;
            if (x_pub % 8 == 0) {
                SE_MARK("X_ARR");
            }
        }
        if (x_cons < x_pub && cb_pages_reservable_at_back(x_cb, (x_slots - (x_pub - x_cons) + 1) * x_blk)) {
            ++x_cons;  // compute has consumed one more block
        }
        // Chain links keep up to LINK_DEPTH blocks in flight, each tagged with its own NoC transaction id and completed
        // in order (write acknowledged -> successor's counter); one block per round trip only reached ~13 GB/s.
        for (uint32_t k = 0; k < nsucc; ++k) {
            // x_iss / x_fwd count pieces: each piece is forwarded as soon as it has arrived (cut-through), once the
            // successor has freed the block's slot.
            const uint32_t trid0 = 1 + 8 * k;  // successor 0: ids 1..6, successor 1: ids 9..14
            constexpr uint32_t x_depth = 6;
            constexpr uint32_t piece = x_bytes / xp;
            while (x_iss[k] < arrived_p && x_iss[k] - x_fwd[k] < x_depth &&
                   (x_iss[k] / xp < x_slots || *s_free[k] >= x_iss[k] / xp + 1 - x_slots)) {
                const uint32_t off = ((x_iss[k] / xp) % x_slots) * x_bytes + (x_iss[k] % xp) * piece;
                write_trid(x_ring + off, s_ring[k] + off, piece, trid0 + x_iss[k] % x_depth);
                ++x_iss[k];
            }
            while (x_fwd[k] < x_iss[k] &&
                   ncrisc_noc_nonposted_write_with_transaction_id_flushed(noc_index, trid0 + x_fwd[k] % x_depth)) {
                noc_semaphore_inc(s_xarr[k], 1);
                ++x_fwd[k];
            }
        }
#ifndef SE_GU_ONLY
        // Pieces of h that have arrived here: a head has all of h(v) once every slice of the group is in.
        const uint32_t harr = h_head ? (*gath_sem / group_ncc) * h_pieces : *harr_sem[0];
        while (sxy && h_iss < harr && h_iss - h_fwd < link_depth && *hsfree >= h_iss / h_pieces) {
            const uint32_t off = (h_iss % h_pieces) * h_piece_bytes;  // the successor has freed its previous h
            write_trid(h_all_addr + off, succ_hall + off, h_piece_bytes, 1 + link_depth + h_iss % link_depth);
            ++h_iss;
        }
        while (h_fwd < h_iss &&
               ncrisc_noc_nonposted_write_with_transaction_id_flushed(noc_index, 1 + link_depth + h_fwd % link_depth)) {
            noc_semaphore_inc(succ_harr, 1);
            ++h_fwd;
        }
#endif
        uint32_t freed = x_cons;
        for (uint32_t k = 0; k < nsucc; ++k) {
            freed = x_fwd[k] / xp < freed ? x_fwd[k] / xp : freed;
        }
        // Counter reports are atomic increments of the delta: on Blackhole an inline write first waits for every
        // outstanding write to leave L1, which serialized the chain behind its own block writes.
#ifdef SE_X_RELAY
        if (freed > x_rep) {
            const uint32_t rxy = get_arg_val<uint32_t>(11);
            noc_semaphore_inc(get_noc_addr(rxy >> 16, rxy & 0xFFFF, get_arg_val<uint32_t>(12)), freed - x_rep);
#ifdef SE_X_RELAY2
            constexpr uint32_t rl_arg[3] = {15, 16, 17};  // relays 1..: their own copies of this core's word
            for (uint32_t k = 1; k < SE_X_NRELAY; ++k) {
                const uint32_t rxy2 = get_arg_val<uint32_t>(rl_arg[k - 1]);
                noc_semaphore_inc(get_noc_addr(rxy2 >> 16, rxy2 & 0xFFFF, get_arg_val<uint32_t>(12)), freed - x_rep);
            }
#endif
            x_rep = freed;
        }
        if (false) {
#else
        if (freed > x_rep) {
#endif
            if (pxy) {
                noc_semaphore_inc(pred_sfree, freed - x_rep);
            }
            x_rep = freed;
            if (!pxy) {
                *hfree = x_rep;  // the chain head's DRAM reader
            }
        }
        if (!h_pending && h_sent < num_v && *go_sem + hbuf >= h_sent + 1 &&
            cb_pages_available_at_front(h_local_cb, mt * np)) {
            const uint32_t src = get_read_ptr(h_local_cb);
#ifdef SE_GU_ONLY
            const uint32_t dst = ((h_sent % hbuf) * groups + my_group) * half_bytes;  // down h_all: [HBUF][group]
#else
            const uint32_t dst = (h_sent % hbuf) * half_bytes;
#endif
            for (uint32_t hd = 0; hd < nh; ++hd) {
                const uint32_t hxy = get_arg_val<uint32_t>(21 + hd);
                const uint64_t bc_h = get_noc_addr(hxy >> 16, hxy & 0xFFFF, h_all_addr);
#ifdef SE_GU_ONLY
                // Down cores keep h K-tile-major ([K][rows] per group): this core's NP K-tiles are one run.
                noc_async_write(src, bc_h + dst + cg * np * mt * h_tile_bytes, np * mt * h_tile_bytes);
#else
                for (uint32_t m = 0; m < mt; ++m) {
                    for (uint32_t p = 0; p < np; ++p) {  // h_all: row-major within K-blocks of KBLK tiles
                        const uint32_t k = cg * np + p;
                        const uint32_t off = ((k / kblk) * mt * kblk + m * kblk + k % kblk) * h_tile_bytes;
                        noc_async_write(src + (m * np + p) * h_tile_bytes, bc_h + dst + off, h_tile_bytes);
                    }
                }
#endif
            }
            h_pending = 1;
        }
        if (h_pending && ncrisc_noc_nonposted_writes_flushed(noc_index)) {
#ifdef SE_GU_ONLY
            // Slices of h(v) are counted per down-core buffer v % HBUF (CT 31.. give buffers 1, 2).
            const uint32_t gid = h_sent % hbuf == 0   ? gath_sem_id
                                 : h_sent % hbuf == 1 ? get_compile_time_arg_val(31)
                                                      : get_compile_time_arg_val(32);
#else
            const uint32_t gid = gath_sem_id;
#endif
            for (uint32_t hd = 0; hd < nh; ++hd) {
                const uint32_t hxy = get_arg_val<uint32_t>(21 + hd);
                noc_semaphore_inc(get_noc_addr(hxy >> 16, hxy & 0xFFFF, get_semaphore(gid)), 1);
            }
            cb_pop_front(h_local_cb, mt * np);
            ++h_sent;
            h_pending = 0;
        }
#ifndef SE_GU_ONLY
        if (h_pub < num_v && harr >= (h_pub + 1) * h_pieces && h_cons == h_pub) {
            cb_push_back(h_all_cb, h_all_tiles);
            ++h_pub;
        }
        if (h_cons < h_pub && cb_pages_reservable_at_back(h_all_cb, h_all_tiles)) {
            ++h_cons;  // compute is done with h(h_cons)
        }
        const uint32_t hfreed = sxy && h_fwd / h_pieces < h_cons ? h_fwd / h_pieces : h_cons;
        if (hfreed > h_rep) {
            if (!h_head) {
                noc_semaphore_inc(hpred_free, hfreed - h_rep);
            }
            h_rep = hfreed;
        }
        if (cb_pages_available_at_front(out_cb, (out_done - out_popped + 1) * out_tiles)) {
            if (out_done + sub < num_v) {
                cb_pop_front(out_cb, out_tiles);  // the ring keeps the last S outputs (the last expert)
                ++out_popped;
            }
            noc_semaphore_inc(done_noc, 1);
            ++out_done;
        }
        if (is_coord && go_sent + 1 < num_v && *done_sem >= ncc * (go_sent + 1)) {
            for (uint32_t p = 0; p < ncc; ++p) {
                const uint32_t xy = get_arg_val<uint32_t>(peer0 + p);
                noc_semaphore_inc(get_noc_addr(xy >> 16, xy & 0xFFFF, get_semaphore(go_sem_id)), 1);
            }
            ++go_sent;
        }
#endif
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
