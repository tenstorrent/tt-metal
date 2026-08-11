// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// INTRA self-loop that samples the mirrored DM-visible overlay tile counters (0-15) so the host can
// detect the Quasar T6→overlay alias bug. Pack only pushes, unpack only pops, no L1 payload
// modification. Firmware has already programmed the ClientL capacity and the remapper alias before
// this kernel runs.
//
// Pack issues every push before unpack is allowed to pop, so the ClientL counter is a deterministic
// function of the tiles pushed/popped so far and the host can tell a single count from a double
// count. Steps carry uneven tile counts for the same reason: a doubled event on a series of 1s is
// indistinguishable from the next step's push.
//
// Two handshakes through uncached L1, one writer per word:
//   ready       — pack has taken the baseline overlay sample; unpack may start
//   pushes_done — pack has issued the whole push series; unpack may start popping
//
// Scratch layout (uncached L1 words) is defined by host CTAs:
//   [0]                    scratch_magic
//   [1]                    ClientL capacity after DataflowBuffer ctor (get_local_num_entries)
//   [ready_idx]            handshake_ready once pack's baseline sample is in L1
//   [pushes_done_idx]      handshake_pushes_done once every push has been issued
//   [num_steps_idx]        step count decoded from the step CTAs
//   [posted_base ..)       pack: ClientL tiles-available after each push
//   [acked_base ..)        unpack: ClientL space-available after each pop
//   [baseline_base ..)     mirrored DM TC0..N-1 baseline {cap, posted, acked}
//   [final_base ..)        mirrored DM TC0..N-1 after the INTRA series {cap, posted, acked}
//   [done_idx]             scratch_done

#include "api/dataflow/dataflow_buffer.h"
#include "api/debug/dprint.h"
#include "ckernel_trisc_common.h"
#include "dev_mem_map.h"
#include "experimental/kernel_args.h"

namespace {
// Host may pass fewer steps; a zero tile count ends the series.
constexpr uint32_t max_steps = 4;

volatile uint32_t* scratch_ptr(uint32_t l1_address, uint32_t word_idx) {
    return reinterpret_cast<volatile uint32_t*>(l1_address + MEM_L1_UNCACHED_BASE + word_idx * sizeof(uint32_t));
}

// A counter reads back a derived view rather than the credit total that was written:
// tile_counters[].f.posted sits at the TILES_AVAILABLE offset and .f.acked at SPACE_AVAILABLE.
// Sampling right after a push/pop returns the pre-update value, so each sample re-reads until the
// value settles.
uint32_t settled_capacity(uint32_t tc, uint32_t settle_reads) {
    uint32_t value = 0;
    for (uint32_t i = 0; i < settle_reads; i++) {
        value = ckernel::trisc::tile_counters[tc].f.buf_capacity;
    }
    return value;
}

uint32_t settled_posted(uint32_t tc, uint32_t settle_reads) {
    uint32_t value = 0;
    for (uint32_t i = 0; i < settle_reads; i++) {
        value = ckernel::trisc::tile_counters[tc].f.posted;
    }
    return value;
}

uint32_t settled_acked(uint32_t tc, uint32_t settle_reads) {
    uint32_t value = 0;
    for (uint32_t i = 0; i < settle_reads; i++) {
        value = ckernel::trisc::tile_counters[tc].f.acked;
    }
    return value;
}

void write_tc_triple(uint32_t l1_address, uint32_t tc, uint32_t base_idx, uint32_t settle_reads) {
    *scratch_ptr(l1_address, base_idx + 0) = settled_capacity(tc, settle_reads);
    *scratch_ptr(l1_address, base_idx + 1) = settled_posted(tc, settle_reads);
    *scratch_ptr(l1_address, base_idx + 2) = settled_acked(tc, settle_reads);
}

void sample_mirrored_dm_tc_block(
    uint32_t l1_address, uint32_t base_idx, uint32_t num_overlay_tcs, uint32_t words_per_tc, uint32_t settle_reads) {
    for (uint32_t tc = 0; tc < num_overlay_tcs; tc++) {
        write_tc_triple(l1_address, tc, base_idx + tc * words_per_tc, settle_reads);
    }
}
}  // namespace

void kernel_main() {
    constexpr uint32_t scratch_l1_address = get_arg(args::scratch_l1_address);
    constexpr uint32_t scratch_magic = get_arg(args::scratch_magic);
    constexpr uint32_t scratch_done = get_arg(args::scratch_done);
    constexpr uint32_t handshake_ready = get_arg(args::handshake_ready);
    constexpr uint32_t handshake_pushes_done = get_arg(args::handshake_pushes_done);
    constexpr uint32_t settle_reads = get_arg(args::settle_reads);
    constexpr uint32_t client_l_tc = get_arg(args::client_l_tc);
    constexpr uint32_t num_overlay_tcs = get_arg(args::num_overlay_tcs);
    constexpr uint32_t words_per_tc = get_arg(args::words_per_tc);
    constexpr uint32_t ready_idx = get_arg(args::ready_idx);
    constexpr uint32_t pushes_done_idx = get_arg(args::pushes_done_idx);
    constexpr uint32_t num_steps_idx = get_arg(args::num_steps_idx);
    constexpr uint32_t posted_base = get_arg(args::posted_base);
    constexpr uint32_t acked_base = get_arg(args::acked_base);
    constexpr uint32_t baseline_base = get_arg(args::baseline_base);
    constexpr uint32_t final_base = get_arg(args::final_base);
    constexpr uint32_t done_idx = get_arg(args::done_idx);

    const uint32_t step_tiles[max_steps] = {
        get_arg(args::step0), get_arg(args::step1), get_arg(args::step2), get_arg(args::step3)};
    uint32_t num_steps = 0;
    while (num_steps < max_steps && step_tiles[num_steps] != 0) {
        num_steps++;
    }

    DataflowBuffer dfb(dfb::out);

#ifdef UCK_CHLKC_PACK
    // Firmware already programmed ClientL capacity + remapper alias. Snapshot mirrored DM TCs before
    // any INTRA push so a leaked capacity/credit write would already show up here.
    *scratch_ptr(scratch_l1_address, 0) = scratch_magic;
    *scratch_ptr(scratch_l1_address, 1) = dfb.get_local_num_entries();
    *scratch_ptr(scratch_l1_address, num_steps_idx) = num_steps;
    sample_mirrored_dm_tc_block(scratch_l1_address, baseline_base, num_overlay_tcs, words_per_tc, settle_reads);
    *scratch_ptr(scratch_l1_address, ready_idx) = handshake_ready;
    DPRINT(
        "PACK baseline: ClientL TC{} capacity={} mirrored_dm_tc0={}/{}/{}\n",
        client_l_tc,
        *scratch_ptr(scratch_l1_address, 1),
        *scratch_ptr(scratch_l1_address, baseline_base + 0),
        *scratch_ptr(scratch_l1_address, baseline_base + 1),
        *scratch_ptr(scratch_l1_address, baseline_base + 2));

    // Whole series is issued before unpack pops, so ClientL tiles-available after step s must equal
    // the tiles pushed through step s. Twice that means the update was counted twice.
    for (uint32_t step = 0; step < num_steps; step++) {
        const uint32_t num_tiles = step_tiles[step];
        WAYPOINT("PW");
        dfb.reserve_back(num_tiles);
        WAYPOINT("PWD");
        dfb.push_back(num_tiles);
        WAYPOINT("PPD");
        const uint32_t posted = settled_posted(client_l_tc, settle_reads);
        *scratch_ptr(scratch_l1_address, posted_base + step) = posted;
        DPRINT("PACK step{} push={} TC{} posted={}\n", step, num_tiles, client_l_tc, posted);
    }
    *scratch_ptr(scratch_l1_address, pushes_done_idx) = handshake_pushes_done;
#endif

#ifdef UCK_CHLKC_UNPACK
    WAYPOINT("URW");
    while (*scratch_ptr(scratch_l1_address, ready_idx) != handshake_ready) {
    }
    WAYPOINT("URD");

    // Wait for the full push series so the pop arithmetic is a deterministic function of tiles popped.
    WAYPOINT("UPW");
    while (*scratch_ptr(scratch_l1_address, pushes_done_idx) != handshake_pushes_done) {
    }
    WAYPOINT("UPR");

    for (uint32_t step = 0; step < num_steps; step++) {
        const uint32_t num_tiles = step_tiles[step];
        WAYPOINT("WT");
        dfb.wait_front(num_tiles);
        WAYPOINT("WTD");
        dfb.pop_front(num_tiles);
        WAYPOINT("UPD");
        const uint32_t acked = settled_acked(client_l_tc, settle_reads);
        *scratch_ptr(scratch_l1_address, acked_base + step) = acked;
        DPRINT("UNPACK step{} pop={} TC{} acked={}\n", step, num_tiles, client_l_tc, acked);
    }
#endif

    dfb.finish();

#ifdef UCK_CHLKC_PACK
    sample_mirrored_dm_tc_block(scratch_l1_address, final_base, num_overlay_tcs, words_per_tc, settle_reads);
    *scratch_ptr(scratch_l1_address, done_idx) = scratch_done;
    DPRINT(
        "PACK final: mirrored_dm_tc0={}/{}/{}\n",
        *scratch_ptr(scratch_l1_address, final_base + 0),
        *scratch_ptr(scratch_l1_address, final_base + 1),
        *scratch_ptr(scratch_l1_address, final_base + 2));
#endif
}
