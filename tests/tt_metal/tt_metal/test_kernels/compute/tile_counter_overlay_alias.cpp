// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/debug/dprint.h"
#include "ckernel_trisc_common.h"
#include "dev_mem_map.h"
#include "experimental/kernel_args.h"
#include "internal/tt-2xx/quasar/overlay/remapper_common.hpp"  // for pack-side remapper toggle below

namespace {
// Pack and unpack both use TC16 for the real credit protocol (push / wait / pop). The remapper also routes
// TC16 -> TC17 so the HW update copy lands on a sacrificial counter instead of aliasing into overlay 0-15.
// Neither RISC programs or pops TC17; it exists only as the remapper ClientR target and is not sampled.
constexpr std::uint32_t kProducerTc = 16;
constexpr std::uint32_t kRemapShadowTc = 17;
constexpr std::uint32_t kOverlayTc0 = 0;
constexpr std::uint32_t kOverlayTc1 = 1;
constexpr std::uint32_t kMaxSteps = 4;

// A counter register reads back a derived view rather than the credit total that was written:
// tile_counters[].f.posted sits at the TILES_AVAILABLE offset and tile_counters[].f.acked at
// SPACE_AVAILABLE. So a push raises .posted by the tiles pushed only while they are unconsumed, and a pop
// raises .acked toward buf_capacity. Sampling right after the push/pop instruction returns the pre-update
// value, so each sample is taken after a bounded series of re-reads.
constexpr std::uint32_t kSettleReads = 256;

// Bound on the software poll for credits to show up on TC16 before TT_WAIT_TILES. Without it, a missing
// push would park unpack in the hardware stall and hang the run instead of reporting.
constexpr std::uint32_t kCreditWaitSpins = 1u << 16;
constexpr std::uint32_t kTimedOut = 0xFFFFFFFEu;

// L1 scratch layout (uncached), one writer per word. Must match the host test.
//   [0]        pack: TC16 reset + capacity programmed  (handshake, host-initialized to 0)
//   [1 .. 4]   pack: TC16 posted after each push
//   [5 .. 8]   unpack: TC16 acked after each pop
//   [9]        pack: every push issued                          (handshake, host-initialized to 0)
//   [10]       pack: TC16 buf_capacity read back
//   [11]       pack: step count decoded from runtime args
// Overlay TC0 / TC1 isolation samples (must stay at the pre-TC16 baseline). Read through the NEO-local
// tile counter mirror, the only tile-counter aperture a TRISC can address:
//   TC0: [12..14] baseline, [15..17] after capacity, [18..21] after push posted,
//         [22..25] after pop acked, [26..28] final
//   TC1: [29..31] baseline, [32..34] after capacity, [35..38] after push posted,
//         [39..42] after pop acked, [43..45] final
constexpr std::uint32_t kReadyIdx = 0;
constexpr std::uint32_t kProducerPostedBaseIdx = 1;
constexpr std::uint32_t kProducerAckedBaseIdx = kProducerPostedBaseIdx + kMaxSteps;
constexpr std::uint32_t kPushesDoneIdx = kProducerAckedBaseIdx + kMaxSteps;
constexpr std::uint32_t kProducerCapacityIdx = kPushesDoneIdx + 1;
constexpr std::uint32_t kNumStepsIdx = kProducerCapacityIdx + 1;

constexpr std::uint32_t kOverlay0BaselineCapIdx = kNumStepsIdx + 1;
constexpr std::uint32_t kOverlay0BaselinePostedIdx = kOverlay0BaselineCapIdx + 1;
constexpr std::uint32_t kOverlay0BaselineAckedIdx = kOverlay0BaselinePostedIdx + 1;
constexpr std::uint32_t kOverlay0AfterCapCapIdx = kOverlay0BaselineAckedIdx + 1;
constexpr std::uint32_t kOverlay0AfterCapPostedIdx = kOverlay0AfterCapCapIdx + 1;
constexpr std::uint32_t kOverlay0AfterCapAckedIdx = kOverlay0AfterCapPostedIdx + 1;
constexpr std::uint32_t kOverlay0AfterPushPostedBaseIdx = kOverlay0AfterCapAckedIdx + 1;
constexpr std::uint32_t kOverlay0AfterPopAckedBaseIdx = kOverlay0AfterPushPostedBaseIdx + kMaxSteps;
constexpr std::uint32_t kOverlay0FinalCapIdx = kOverlay0AfterPopAckedBaseIdx + kMaxSteps;
constexpr std::uint32_t kOverlay0FinalPostedIdx = kOverlay0FinalCapIdx + 1;
constexpr std::uint32_t kOverlay0FinalAckedIdx = kOverlay0FinalPostedIdx + 1;

constexpr std::uint32_t kOverlay1BaselineCapIdx = kOverlay0FinalAckedIdx + 1;
constexpr std::uint32_t kOverlay1BaselinePostedIdx = kOverlay1BaselineCapIdx + 1;
constexpr std::uint32_t kOverlay1BaselineAckedIdx = kOverlay1BaselinePostedIdx + 1;
constexpr std::uint32_t kOverlay1AfterCapCapIdx = kOverlay1BaselineAckedIdx + 1;
constexpr std::uint32_t kOverlay1AfterCapPostedIdx = kOverlay1AfterCapCapIdx + 1;
constexpr std::uint32_t kOverlay1AfterCapAckedIdx = kOverlay1AfterCapPostedIdx + 1;
constexpr std::uint32_t kOverlay1AfterPushPostedBaseIdx = kOverlay1AfterCapAckedIdx + 1;
constexpr std::uint32_t kOverlay1AfterPopAckedBaseIdx = kOverlay1AfterPushPostedBaseIdx + kMaxSteps;
constexpr std::uint32_t kOverlay1FinalCapIdx = kOverlay1AfterPopAckedBaseIdx + kMaxSteps;
constexpr std::uint32_t kOverlay1FinalPostedIdx = kOverlay1FinalCapIdx + 1;
constexpr std::uint32_t kOverlay1FinalAckedIdx = kOverlay1FinalPostedIdx + 1;

volatile std::uint32_t* scratch_ptr(std::uint32_t l1_address, std::uint32_t word_idx) {
    return reinterpret_cast<volatile std::uint32_t*>(
        l1_address + MEM_L1_UNCACHED_BASE + word_idx * sizeof(std::uint32_t));
}

// The overlay::llk_intf_* helpers cannot serve these reads: they hardcode 0x03003000-based addresses, which
// is the DM/NOC view of the overlay, while a T6 thread reaches the overlay at OVERLAY_REGS_BASE
// (0x02000000) and its tile counters at TILE_COUNTERS_BASE (0x0080c000). From a TRISC they read an address
// outside the map and return 0 no matter what the counter holds.
std::uint32_t settled_posted(std::uint32_t tc) {
    std::uint32_t value = 0;
    for (std::uint32_t i = 0; i < kSettleReads; i++) {
        value = ckernel::trisc::tile_counters[tc].f.posted;
    }
    return value;
}

std::uint32_t settled_acked(std::uint32_t tc) {
    std::uint32_t value = 0;
    for (std::uint32_t i = 0; i < kSettleReads; i++) {
        value = ckernel::trisc::tile_counters[tc].f.acked;
    }
    return value;
}

std::uint32_t settled_capacity(std::uint32_t tc) {
    std::uint32_t value = 0;
    for (std::uint32_t i = 0; i < kSettleReads; i++) {
        value = ckernel::trisc::tile_counters[tc].f.buf_capacity;
    }
    return value;
}

void write_overlay_triple(
    std::uint32_t l1_address,
    std::uint32_t tc,
    std::uint32_t cap_idx,
    std::uint32_t posted_idx,
    std::uint32_t acked_idx) {
    *scratch_ptr(l1_address, cap_idx) = settled_capacity(tc);
    *scratch_ptr(l1_address, posted_idx) = settled_posted(tc);
    *scratch_ptr(l1_address, acked_idx) = settled_acked(tc);
}
}  // namespace

void kernel_main() {
    const std::uint32_t l1_address = get_arg(args::l1_address);
    const std::uint32_t capacity = get_arg(args::capacity);
    const std::uint32_t step_tiles[kMaxSteps] = {
        get_arg(args::step0), get_arg(args::step1), get_arg(args::step2), get_arg(args::step3)};

    // A zero-sized step ends the series, so a case can use fewer than kMaxSteps.
    std::uint32_t num_steps = 0;
    while (num_steps < kMaxSteps && step_tiles[num_steps] != 0) {
        num_steps++;
    }

#ifdef TRISC_PACK
    // Keep the trigger independent of DFB allocation. The bug is in the RTL
    // T6->overlay update path, so one direct legal T6 counter update is enough.
    //
    // Remapper toggle: DM installs its route when DFB_HACK_REMAP_TC16_IDENTITY=1 in dm.cc. To drive the
    // route from pack instead, set that define to 0 and uncomment the block below (do not leave both
    // enabled). This route is Neo0/tc16 -> Neo0/tc17 (sacrificial shadow); pack/unpack still use tc16.
    // #if 0
    constexpr std::uint32_t remap_pair_idx = 48;
    constexpr std::uint32_t neo0_client_id = static_cast<std::uint32_t>(overlay::NEO_0);
    // ClientR packs slot r as id at bit r*8 and cnt_sel at bit r*8+3; only slot 0 is used.
    constexpr std::uint32_t client_r_val = (neo0_client_id & 0x7) | ((kRemapShadowTc & 0x1F) << 3);
    // ClientL: id_L[2:0], cnt_sel_L[7:3], valid[11:8], clientl_is_producer[12], clientr_group[13],
    // distribute[14]. Only slot 0 is valid and Neo0 is the producer.
    constexpr std::uint32_t client_l_val =
        (neo0_client_id & 0x7) | ((kProducerTc & 0x1F) << 3) | (0x1u << 8) | (0x1u << 12);

    *reinterpret_cast<volatile std::uint32_t*>(REMAP_CLIENT_R_CONFIG_REG_ADDR32(remap_pair_idx)) = client_r_val;
    *reinterpret_cast<volatile std::uint32_t*>(REMAP_CLIENT_L_CONFIG_REG_ADDR32(remap_pair_idx)) = client_l_val;
    asm volatile("fence" ::: "memory");

    auto* remapper_control = reinterpret_cast<volatile std::uint32_t*>(REMAP_GLOBAL_CONTROL_REG_ADDR32);
    *remapper_control = 1;
    WAYPOINT("RCW");
    while ((*remapper_control & 1) == 0) {
    }
    WAYPOINT("RCD");
    // #endif

    // Pack owns TC16: reset + capacity. Overlay TC0/TC1 are reset before the baseline so isolation
    // samples compare a clean counter against itself after TC16 ops.
    ckernel::trisc::tile_counters[kProducerTc].f.reset = 1;
    ckernel::trisc::tile_counters[kOverlayTc0].f.reset = 1;
    ckernel::trisc::tile_counters[kOverlayTc1].f.reset = 1;

    write_overlay_triple(
        l1_address, kOverlayTc0, kOverlay0BaselineCapIdx, kOverlay0BaselinePostedIdx, kOverlay0BaselineAckedIdx);
    write_overlay_triple(
        l1_address, kOverlayTc1, kOverlay1BaselineCapIdx, kOverlay1BaselinePostedIdx, kOverlay1BaselineAckedIdx);

    ckernel::trisc::tile_counters[kProducerTc].f.buf_capacity = capacity;
    *scratch_ptr(l1_address, kProducerCapacityIdx) = settled_capacity(kProducerTc);
    *scratch_ptr(l1_address, kNumStepsIdx) = num_steps;
    write_overlay_triple(
        l1_address, kOverlayTc0, kOverlay0AfterCapCapIdx, kOverlay0AfterCapPostedIdx, kOverlay0AfterCapAckedIdx);
    write_overlay_triple(
        l1_address, kOverlayTc1, kOverlay1AfterCapCapIdx, kOverlay1AfterCapPostedIdx, kOverlay1AfterCapAckedIdx);
    *scratch_ptr(l1_address, kReadyIdx) = 1;

    DPRINT(
        "AFTER capacity={}: TC{} cap={}; TC0={}/{}/{} TC1={}/{}/{}\n",
        capacity,
        kProducerTc,
        *scratch_ptr(l1_address, kProducerCapacityIdx),
        *scratch_ptr(l1_address, kOverlay0AfterCapCapIdx),
        *scratch_ptr(l1_address, kOverlay0AfterCapPostedIdx),
        *scratch_ptr(l1_address, kOverlay0AfterCapAckedIdx),
        *scratch_ptr(l1_address, kOverlay1AfterCapCapIdx),
        *scratch_ptr(l1_address, kOverlay1AfterCapPostedIdx),
        *scratch_ptr(l1_address, kOverlay1AfterCapAckedIdx));

    // Every push is issued before unpack is allowed to pop, so the running total of pushed tiles is the
    // expected value on TC16; 2x means the event was counted twice.
    for (std::uint32_t step = 0; step < num_steps; step++) {
        const std::uint32_t num_tiles = step_tiles[step];
        WAYPOINT("PW");
        TT_WAIT_FREE(ckernel::p_stall::STALL_PACK, num_tiles, kProducerTc);
        WAYPOINT("PWD");
        TT_PUSH_TILES(/*PACK_SEL=*/0x1, num_tiles, kProducerTc);
        WAYPOINT("PPD");
        const std::uint32_t producer_posted = settled_posted(kProducerTc);
        *scratch_ptr(l1_address, kProducerPostedBaseIdx + step) = producer_posted;
        *scratch_ptr(l1_address, kOverlay0AfterPushPostedBaseIdx + step) = settled_posted(kOverlayTc0);
        *scratch_ptr(l1_address, kOverlay1AfterPushPostedBaseIdx + step) = settled_posted(kOverlayTc1);
        DPRINT(
            "PACK step{} push={} TC{} posted={} overlay_tc0_posted={} overlay_tc1_posted={}\n",
            step,
            num_tiles,
            kProducerTc,
            producer_posted,
            *scratch_ptr(l1_address, kOverlay0AfterPushPostedBaseIdx + step),
            *scratch_ptr(l1_address, kOverlay1AfterPushPostedBaseIdx + step));
    }
    *scratch_ptr(l1_address, kPushesDoneIdx) = 1;
#endif

#ifdef TRISC_UNPACK
    WAYPOINT("URW");
    while (*scratch_ptr(l1_address, kReadyIdx) == 0) {
    }
    WAYPOINT("URD");

    // Wait for the full push series so the pop arithmetic is a deterministic function of tiles popped.
    while (*scratch_ptr(l1_address, kPushesDoneIdx) == 0) {
    }
    WAYPOINT("UPD");

    for (std::uint32_t step = 0; step < num_steps; step++) {
        const std::uint32_t num_tiles = step_tiles[step];

        // Confirm credits on TC16 before the hardware stall so a missing push reports a timeout instead of
        // hanging.
        std::uint32_t spins = 0;
        WAYPOINT("SW");
        while (ckernel::trisc::tile_counters[kProducerTc].f.posted < num_tiles && spins < kCreditWaitSpins) {
            spins++;
        }
        WAYPOINT("SD");
        if (spins == kCreditWaitSpins) {
            *scratch_ptr(l1_address, kProducerAckedBaseIdx + step) = kTimedOut;
            DPRINT(
                "UNPACK step{} timed out waiting for {} tiles on TC{}; posted={}\n",
                step,
                num_tiles,
                kProducerTc,
                settled_posted(kProducerTc));
            break;
        }

        WAYPOINT("WT");
        TT_WAIT_TILES(ckernel::p_stall::STALL_UNPACK, num_tiles, kProducerTc);
        WAYPOINT("WTD");
        TT_POP_TILES(/*UNPACK_SEL=*/0x3, num_tiles, kProducerTc);
        WAYPOINT("UPD");
        const std::uint32_t producer_acked = settled_acked(kProducerTc);
        *scratch_ptr(l1_address, kProducerAckedBaseIdx + step) = producer_acked;
        *scratch_ptr(l1_address, kOverlay0AfterPopAckedBaseIdx + step) = settled_acked(kOverlayTc0);
        *scratch_ptr(l1_address, kOverlay1AfterPopAckedBaseIdx + step) = settled_acked(kOverlayTc1);
        DPRINT(
            "UNPACK step{} pop={} TC{} acked={} overlay_tc0_acked={} overlay_tc1_acked={}\n",
            step,
            num_tiles,
            kProducerTc,
            producer_acked,
            *scratch_ptr(l1_address, kOverlay0AfterPopAckedBaseIdx + step),
            *scratch_ptr(l1_address, kOverlay1AfterPopAckedBaseIdx + step));
    }
    write_overlay_triple(
        l1_address, kOverlayTc0, kOverlay0FinalCapIdx, kOverlay0FinalPostedIdx, kOverlay0FinalAckedIdx);
    write_overlay_triple(
        l1_address, kOverlayTc1, kOverlay1FinalCapIdx, kOverlay1FinalPostedIdx, kOverlay1FinalAckedIdx);
#endif
}
