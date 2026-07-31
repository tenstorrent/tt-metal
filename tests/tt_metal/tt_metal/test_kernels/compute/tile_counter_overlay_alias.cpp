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
// Pack posts to the producer counter and the remapper routes that credit to the consumer counter, which
// unpack waits on and pops. Two counters instead of a self-route: an identity route lands both the native
// T6 update and the routed copy on one counter (2x), while routing to a second counter keeps each at 1x and
// still stops the copy from aliasing down into the overlay's 0-15 range.
constexpr std::uint32_t kProducerTc = 16;
constexpr std::uint32_t kConsumerTc = 16;
constexpr std::uint32_t kOverlayTc = 0;
constexpr std::uint32_t kMaxSteps = 4;

// A counter register reads back a derived view rather than the credit total that was written:
// tile_counters[].f.posted sits at the TILES_AVAILABLE offset and tile_counters[].f.acked at
// SPACE_AVAILABLE. So a push raises .posted by the tiles pushed only while they are unconsumed, and a pop
// raises .acked toward buf_capacity. Sampling right after the push/pop instruction returns the pre-update
// value, so each sample is taken after a bounded series of re-reads.
constexpr std::uint32_t kSettleReads = 256;

// Bound on the software poll for a routed credit to reach the consumer counter. Without it, a route that
// never delivers would park unpack in TT_WAIT_TILES and hang the run instead of reporting.
constexpr std::uint32_t kCreditWaitSpins = 1u << 16;
constexpr std::uint32_t kTimedOut = 0xFFFFFFFEu;

// L1 scratch layout (uncached), one writer per word. Must match the host test.
//   [0]        pack: producer tc16 reset + capacity programmed  (handshake, host-initialized to 0)
//   [1 .. 4]   pack: producer tc16 posted after each push
//   [5 .. 8]   unpack: consumer tc17 acked after each pop
//   [9]        pack: every push issued                          (handshake, host-initialized to 0)
//   [10]       pack: producer tc16 buf_capacity read back
//   [11]       pack: step count decoded from runtime args
// Cross samples: each side also reads the counter it does not drive, so a credit that doubles (both
// counters advance per event) is distinguishable from one that moves (only the routed target advances).
//   [12..15]   pack: consumer tc17 posted after each push (remapper delivery)
//   [16..19]   unpack: producer tc16 acked after each pop
//   [20]       unpack: consumer tc17 buf_capacity (no RISC programs it; the route should carry tc16's write)
// TC0 isolation samples (must stay at the pre-TC16 baseline; host does not program TC0 capacity). Read
// through the NEO-local tile counter mirror, the only tile-counter aperture a TRISC can address:
//   [21..23]   pack: capacity / posted / acked before any TC16 op
//   [24..26]   pack: same triple after the capacities are programmed
//   [27..30]   pack: posted after each push
//   [31..34]   unpack: acked after each pop
//   [35..37]   unpack: capacity / posted / acked after the last pop
constexpr std::uint32_t kReadyIdx = 0;
constexpr std::uint32_t kProducerPostedBaseIdx = 1;
constexpr std::uint32_t kConsumerAckedBaseIdx = kProducerPostedBaseIdx + kMaxSteps;
constexpr std::uint32_t kPushesDoneIdx = kConsumerAckedBaseIdx + kMaxSteps;
constexpr std::uint32_t kProducerCapacityIdx = kPushesDoneIdx + 1;
constexpr std::uint32_t kNumStepsIdx = kProducerCapacityIdx + 1;
constexpr std::uint32_t kConsumerPostedBaseIdx = kNumStepsIdx + 1;
constexpr std::uint32_t kProducerAckedBaseIdx = kConsumerPostedBaseIdx + kMaxSteps;
constexpr std::uint32_t kConsumerCapacityIdx = kProducerAckedBaseIdx + kMaxSteps;
constexpr std::uint32_t kOverlayBaselineCapIdx = kConsumerCapacityIdx + 1;
constexpr std::uint32_t kOverlayBaselinePostedIdx = kOverlayBaselineCapIdx + 1;
constexpr std::uint32_t kOverlayBaselineAckedIdx = kOverlayBaselinePostedIdx + 1;
constexpr std::uint32_t kOverlayAfterCapCapIdx = kOverlayBaselineAckedIdx + 1;
constexpr std::uint32_t kOverlayAfterCapPostedIdx = kOverlayAfterCapCapIdx + 1;
constexpr std::uint32_t kOverlayAfterCapAckedIdx = kOverlayAfterCapPostedIdx + 1;
constexpr std::uint32_t kOverlayAfterPushPostedBaseIdx = kOverlayAfterCapAckedIdx + 1;
constexpr std::uint32_t kOverlayAfterPopAckedBaseIdx = kOverlayAfterPushPostedBaseIdx + kMaxSteps;
constexpr std::uint32_t kOverlayFinalCapIdx = kOverlayAfterPopAckedBaseIdx + kMaxSteps;
constexpr std::uint32_t kOverlayFinalPostedIdx = kOverlayFinalCapIdx + 1;
constexpr std::uint32_t kOverlayFinalAckedIdx = kOverlayFinalPostedIdx + 1;

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
    std::uint32_t l1_address, std::uint32_t cap_idx, std::uint32_t posted_idx, std::uint32_t acked_idx) {
    *scratch_ptr(l1_address, cap_idx) = settled_capacity(kOverlayTc);
    *scratch_ptr(l1_address, posted_idx) = settled_posted(kOverlayTc);
    *scratch_ptr(l1_address, acked_idx) = settled_acked(kOverlayTc);
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
    // enabled). This route is Neo0/tc16 -> Neo0/tc17, not the identity route dm.cc installs.
    // #if 0
    constexpr std::uint32_t remap_pair_idx = 48;
    constexpr std::uint32_t neo0_client_id = static_cast<std::uint32_t>(overlay::NEO_0);
    // ClientR packs slot r as id at bit r*8 and cnt_sel at bit r*8+3; only slot 0 is used.
    constexpr std::uint32_t client_r_val = (neo0_client_id & 0x7) | ((kConsumerTc & 0x1F) << 3);
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

    write_overlay_triple(l1_address, kOverlayBaselineCapIdx, kOverlayBaselinePostedIdx, kOverlayBaselineAckedIdx);

    // Pack owns the producer counter and nothing else: no RISC programs the consumer counter, so the only
    // thing that can put credits on it is the remapper route.
    ckernel::trisc::tile_counters[kProducerTc].f.reset = 1;
    ckernel::trisc::tile_counters[kOverlayTc].f.reset = 1;
    ckernel::trisc::tile_counters[kProducerTc].f.buf_capacity = capacity;
    *scratch_ptr(l1_address, kProducerCapacityIdx) = settled_capacity(kProducerTc);
    *scratch_ptr(l1_address, kNumStepsIdx) = num_steps;
    write_overlay_triple(l1_address, kOverlayAfterCapCapIdx, kOverlayAfterCapPostedIdx, kOverlayAfterCapAckedIdx);
    *scratch_ptr(l1_address, kReadyIdx) = 1;

    DPRINT(
        "AFTER capacity={}: TC{} cap={}; overlay TC0 cap/posted/acked={}/{}/{} (baseline {}/{}/{})\n",
        capacity,
        kProducerTc,
        *scratch_ptr(l1_address, kProducerCapacityIdx),
        *scratch_ptr(l1_address, kOverlayAfterCapCapIdx),
        *scratch_ptr(l1_address, kOverlayAfterCapPostedIdx),
        *scratch_ptr(l1_address, kOverlayAfterCapAckedIdx),
        *scratch_ptr(l1_address, kOverlayBaselineCapIdx),
        *scratch_ptr(l1_address, kOverlayBaselinePostedIdx),
        *scratch_ptr(l1_address, kOverlayBaselineAckedIdx));

    // Every push is issued before unpack is allowed to pop, so the running total of pushed tiles is the
    // expected value on whichever counter receives the credit; 2x means the event was counted twice.
    for (std::uint32_t step = 0; step < num_steps; step++) {
        const std::uint32_t num_tiles = step_tiles[step];
        WAYPOINT("PW");
        TT_WAIT_FREE(ckernel::p_stall::STALL_PACK, num_tiles, kProducerTc);
        WAYPOINT("PWD");
        TT_PUSH_TILES(/*PACK_SEL=*/0x1, num_tiles, kProducerTc);
        WAYPOINT("PPD");
        const std::uint32_t producer_posted = settled_posted(kProducerTc);
        const std::uint32_t consumer_posted = settled_posted(kConsumerTc);
        *scratch_ptr(l1_address, kProducerPostedBaseIdx + step) = producer_posted;
        *scratch_ptr(l1_address, kConsumerPostedBaseIdx + step) = consumer_posted;
        *scratch_ptr(l1_address, kOverlayAfterPushPostedBaseIdx + step) = settled_posted(kOverlayTc);
        DPRINT(
            "PACK step{} push={} TC{} posted={} TC{} posted={} overlay_tc0_posted={}\n",
            step,
            num_tiles,
            kProducerTc,
            producer_posted,
            kConsumerTc,
            consumer_posted,
            *scratch_ptr(l1_address, kOverlayAfterPushPostedBaseIdx + step));
    }
    *scratch_ptr(l1_address, kPushesDoneIdx) = 1;
#endif

#ifdef TRISC_UNPACK
    WAYPOINT("URW");
    while (*scratch_ptr(l1_address, kReadyIdx) == 0) {
    }
    WAYPOINT("URD");

    // No RISC programs the consumer counter: the route is expected to carry pack's TC16 capacity write onto
    // it, so this samples the capacity only after pack has published that its own write landed. A 0 here says
    // the route moved credits without the capacity that bounds them.
    // Re-enable these two lines to program TC17 directly instead of relying on the route to configure it.
    // ckernel::trisc::tile_counters[kConsumerTc].f.reset = 1;
    // ckernel::trisc::tile_counters[kConsumerTc].f.buf_capacity = capacity;
    *scratch_ptr(l1_address, kConsumerCapacityIdx) = settled_capacity(kConsumerTc);
    DPRINT("UNPACK observed TC{} capacity={}\n", kConsumerTc, *scratch_ptr(l1_address, kConsumerCapacityIdx));
    // Wait for the full push series so the pop arithmetic is a deterministic function of tiles popped.
    while (*scratch_ptr(l1_address, kPushesDoneIdx) == 0) {
    }
    WAYPOINT("UPD");

    for (std::uint32_t step = 0; step < num_steps; step++) {
        const std::uint32_t num_tiles = step_tiles[step];

        // Confirm in software that the routed credits arrived before committing to the hardware stall, so a
        // route that never delivers reports a timeout instead of hanging the run.
        std::uint32_t spins = 0;
        WAYPOINT("SW");
        while (ckernel::trisc::tile_counters[kConsumerTc].f.posted < num_tiles && spins < kCreditWaitSpins) {
            spins++;
        }
        WAYPOINT("SD");
        if (spins == kCreditWaitSpins) {
            *scratch_ptr(l1_address, kConsumerAckedBaseIdx + step) = kTimedOut;
            *scratch_ptr(l1_address, kProducerAckedBaseIdx + step) = kTimedOut;
            DPRINT(
                "UNPACK step{} timed out waiting for {} tiles on TC{}; TC{} posted={} TC{} posted={}\n",
                step,
                num_tiles,
                kConsumerTc,
                kConsumerTc,
                settled_posted(kConsumerTc),
                kProducerTc,
                settled_posted(kProducerTc));
            break;
        }

        WAYPOINT("WT");
        TT_WAIT_TILES(ckernel::p_stall::STALL_UNPACK, num_tiles, kConsumerTc);
        WAYPOINT("WTD");
        TT_POP_TILES(/*UNPACK_SEL=*/0x3, num_tiles, kConsumerTc);
        WAYPOINT("UPD");
        const std::uint32_t consumer_acked = settled_acked(kConsumerTc);
        const std::uint32_t producer_acked = settled_acked(kProducerTc);
        *scratch_ptr(l1_address, kConsumerAckedBaseIdx + step) = consumer_acked;
        *scratch_ptr(l1_address, kProducerAckedBaseIdx + step) = producer_acked;
        *scratch_ptr(l1_address, kOverlayAfterPopAckedBaseIdx + step) = settled_acked(kOverlayTc);
        DPRINT(
            "UNPACK step{} pop={} TC{} acked={} TC{} acked={} overlay_tc0_acked={}\n",
            step,
            num_tiles,
            kConsumerTc,
            consumer_acked,
            kProducerTc,
            producer_acked,
            *scratch_ptr(l1_address, kOverlayAfterPopAckedBaseIdx + step));
    }
    write_overlay_triple(l1_address, kOverlayFinalCapIdx, kOverlayFinalPostedIdx, kOverlayFinalAckedIdx);
#endif
}
