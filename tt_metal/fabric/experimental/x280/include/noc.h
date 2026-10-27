// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * noc.h - NOC TLB configuration and write helpers for X280 bare-metal
 *
 * Provides 2M TLB window programming and NOC write-verify primitives.
 *
 * Style: free inline functions, no global state, shared X280 CPU primitives.
 */
#ifndef NOC_H
#define NOC_H

#include <stdint.h>

#include "rt/x280_hw.h"
#include "x280_noc_coord.h"
#include "x280_noc_ordering.h"

/* ------------------------------------------------------------------
 * Address constants
 * ------------------------------------------------------------------ */

/* Peripheral Port: TLB configuration registers (224 entries × 16 bytes) */
#define NOC_TLB_2M_CONFIG_BASE 0x2FF00000UL

/* System Port: 2M window access base (uncached, no atomics per PMA) */
#define NOC_2M_WINDOW_BASE 0x430000000ULL /* = System Port + 0x400000000 */
#define NOC_2M_WINDOW_STRIDE 0x200000ULL  /* 2 MiB per window */
#define NOC_2M_WINDOW_COUNT 224

/* Memory Port: 2M window access base (cached, coherent, atomics+LR/SC per PMA) */
#define NOC_2M_WINDOW_BASE_MEMPORT 0x400430000000ULL /* = Memory Port + 0x400000000 */

/* 128G windows.
 *
 * NOC_128G_CONFIG_BASE: peripheral-port offset where the 128 GiB
 *   TLB cfg block starts (right after the 224 x 16 B 2 MiB block).
 *   Per-entry stride is 0xC bytes (3 x u32), NOT 0x10. See the
 *   noc_configure_tlb_128g helper below.
 * NOC_128G_WINDOW_BASE: X280 system-port address of window 0, per
 *   BlackholeA0/L2CPUTile/MemoryMap.md, "x280 physical address space".
 * NOC_128G_WINDOW_STRIDE: width of one 128 GiB window in X280 phys
 *   address space (= 1 << 37).
 * NOC_128G_WINDOW_COUNT: hardware-defined number of 128 GiB slots. */
#define NOC_128G_CONFIG_BASE (NOC_TLB_2M_CONFIG_BASE + 0x10 * 224) /* 0x2FF00E00 */
#define NOC_128G_WINDOW_BASE 0x80430000000ULL /* = System Port + 0x80400000000 */
#define NOC_128G_WINDOW_STRIDE (1ULL << 37)
#define NOC_128G_WINDOW_COUNT 32

/* ------------------------------------------------------------------
 * TLB register bitfield (128-bit, written as 4×u32)
 * Field names and positions: BlackholeA0/L2CPUTile/TLBWindows.md,
 * "Configuration" (local_offset, then noc_properties_lo and _hi).
 *
 * For unicast: x_end = x_start = noc_x, y_end = y_start = noc_y.
 * local_offset stores target_address >> 21 (only upper 43 bits).
 * Low 21 bits of the destination address are the window *offset*.
 * ------------------------------------------------------------------ */

typedef union {
    struct {
        uint64_t local_offset : 43; /* target NOC address >> 21 */
        uint64_t reserved0 : 21;
        uint64_t x_end : 6;    /* NOC X (unicast destination) */
        uint64_t y_end : 6;    /* NOC Y (unicast destination) */
        uint64_t x_start : 6;  /* = x_end for unicast */
        uint64_t y_start : 6;  /* = y_end for unicast */
        uint64_t mcast : 1;    /* 0 for unicast */
        uint64_t ordering : 2; /* noc_ordering_t; 3 = Counted Writes */
        uint64_t linked : 1;
        uint64_t static_vc : 1;
        uint64_t reserved1 : 2;
        uint64_t noc_sel : 1; /* 0=NOC0, 1=NOC1 */
        uint64_t static_vc_buddy : 1;
        uint64_t static_vc_class : 2;
        uint64_t x_keep : 2;
        uint64_t x_skip : 2;
        uint64_t y_keep : 2;
        uint64_t y_skip : 2;
        uint64_t x_exclude_coord : 5;
        uint64_t y_exclude_coord : 4;
        uint64_t x_exclude_direction : 1;
        uint64_t y_exclude_direction : 1;
        uint64_t apply_exclusion : 1;
        uint64_t optimize_routing_for_exclusion : 1;
        uint64_t num_destinations_override : 8;
    };
    uint32_t data[4];
} noc_tlb_2m_t;

/* The same positions as shifts, for code that builds the words without the
 * union (src/device/x280_noc_window.cpp): the address in words 0-1, the rest
 * in noc_properties_lo, word 2. tests/host/test_x280_noc_window.cpp checks
 * that the two agree. */
#define NOC_TLB_2M_LOCAL_OFFSET_SHIFT 21u /* the entry holds target_address >> 21 */
#define NOC_TLB_2M_LOCAL_OFFSET_BITS 43u
#define NOC_TLB_COORD_BITS 6u
#define NOC_TLB_X_END_SHIFT 0u
#define NOC_TLB_Y_END_SHIFT 6u
#define NOC_TLB_X_START_SHIFT 12u
#define NOC_TLB_Y_START_SHIFT 18u
#define NOC_TLB_MCAST_SHIFT 24u
#define NOC_TLB_ORDERING_SHIFT 25u
#define NOC_TLB_ORDERING_MASK 0x3u
#define NOC_TLB_NOC_SEL_SHIFT 31u

/* ------------------------------------------------------------------
 * API functions (free inline, no global state)
 * ------------------------------------------------------------------ */

/* ------------------------------------------------------------------
 * 128 GiB CPU NOC TLB bitfield (96-bit, written as 3 x u32).
 *
 * Used for DMA-driven NOC->NOC transfers. The 128 GiB TLB
 * region of the peripheral port follows the 224 x 16 B 2 MiB block
 * at offset 0xE00 (= 224 * 0x10); each entry is **0xC bytes** (3 x
 * u32), not 0x10. The address field is 27 bits (= addr >> 37), so a
 * single entry covers a full 128 GiB span of NOC address space at the
 * chosen (noc_x, noc_y) endpoint.
 * ------------------------------------------------------------------ */

typedef union {
    struct {
        uint64_t local_offset : 27; /* target NOC address >> 37 */
        uint64_t reserved0 : 5;
        uint64_t x_end : 6;
        uint64_t y_end : 6;
        uint64_t x_start : 6;
        uint64_t y_start : 6;
        uint64_t mcast : 1;
        uint64_t ordering : 2;
        uint64_t linked : 1;
        uint64_t static_vc : 1;
        uint64_t reserved1 : 2;
        uint64_t noc_sel : 1;
        uint64_t static_vc_buddy : 1;
        uint64_t static_vc_class : 2;
        uint64_t x_keep : 2;
        uint64_t x_skip : 2;
        uint64_t y_keep : 2;
        uint64_t y_skip : 2;
        uint64_t x_exclude_coord : 5;
        uint64_t y_exclude_coord : 4;
        uint64_t x_exclude_direction : 1;
        uint64_t y_exclude_direction : 1;
        uint64_t apply_exclusion : 1;
        uint64_t optimize_routing_for_exclusion : 1;
        uint64_t num_destinations_override : 8;
    };
    uint32_t data[3];
} noc_tlb_128g_t;

/**
 * Configure a 128 GiB NOC TLB window and return the X280 system-port
 * window base for that slot.
 *
 * Writes the 96-bit bitfield as 3 x u32 stores at
 * peripheral-port offset 0xE00 + tlb_index * 0xC, with full
 * peripheral-port fences before and after, and stores
 * `tlb.address = addr >> 37` (27-bit field, captures the high
 * bits of the NOC address only).
 *
 * IMPORTANT -- caller responsibility for the low 37 bits.
 *
 * The TLB stores only `addr >> 37`. The remaining 37 bits of
 * `addr` (i.e. the offset within the 128 GiB window) are NOT
 * encoded in the return value; the caller must split them across
 * two destinations:
 *
 *   phys[36:28] (9 bits)   -> bits [8:0] of the outbound DMA TLB
 *                              slot value the caller writes
 *                              before kicking the channel
 *   phys[27:0]  (28 bits)  -> the DMA SAR/DAR (passed through to
 *                              phys[27:0] by the outbound DMA TLB)
 *
 * The DMA-side base of the slot is `(window - SYSTEM_PORT) >> 28`
 * (e.g. 0x8040 for index 0, 0x8240 for index 1, before the phys
 * [36:28] addend is added).
 *
 * Per-kick byte ceiling for a transfer that uses this TLB is
 * 256 MiB (one outbound DMA TLB slot's DMA-address window). A
 * transfer that would straddle a 256 MiB phys[27:0] boundary
 * must be split by the caller.
 *
 * @param index        128 GiB window index 0..(NOC_128G_WINDOW_COUNT-1).
 * @param noc_x        Destination NOC X.
 * @param noc_y        Destination NOC Y.
 * @param addr         Destination NOC address. Only `addr >> 37`
 *                     is programmed into the TLB; see "caller
 *                     responsibility" above.
 * @param posted       1 = posted, 0 = non-posted.
 * @param strict_order 1 = enforce in-order delivery.
 * @return  X280 system-port window base
 *          (= NOC_128G_WINDOW_BASE + index * NOC_128G_WINDOW_STRIDE,
 *          no offset).
 *          The firmware does NOT dereference this directly -- the
 *          DMA engine is the consumer.
 */
static inline uint64_t noc_configure_tlb_128g(
    uint32_t index, uint32_t noc_x, uint32_t noc_y, uint64_t addr, int posted, int strict_order) {
    /* Per-entry stride in the 128 GiB block is 0xC (3 x u32), NOT
     * 0x10. The block itself starts at NOC_128G_CONFIG_BASE
     * (= 2 MiB block end). */
    volatile uint32_t* cfg = (volatile uint32_t*)(NOC_128G_CONFIG_BASE + (uint64_t)index * 0xCUL);

    /* Word by word, not `= {0}`: see the note in noc_configure_tlb_2m. */
    noc_tlb_128g_t tlb;
    tlb.data[0] = 0;
    tlb.data[1] = 0;
    tlb.data[2] = 0;
    tlb.local_offset = addr >> 37;
    tlb.x_end = noc_x;
    tlb.y_end = noc_y;
    tlb.x_start = noc_x;
    tlb.y_start = noc_y;
    tlb.ordering = (posted ? 2u : 0u) | (strict_order ? 1u : 0u);

    x280_fence_iorw();
    cfg[0] = tlb.data[0];
    cfg[1] = tlb.data[1];
    cfg[2] = tlb.data[2];
    x280_fence_iorw();

    /* The low 37 bits of `addr` are NOT added here; see the doc
     * comment above for the caller's responsibility. The firmware
     * does not dereference this pointer (the DMA does). */
    return NOC_128G_WINDOW_BASE + (uint64_t)index * NOC_128G_WINDOW_STRIDE;
}

/**
 * Configure a 2M NOC TLB window and return the window access pointer.
 *
 * Programs TLB entry `index` to route accesses to (noc_x, noc_y, addr).
 * Full fences (fence iorw, iorw) are issued before and after the 4-word
 * config register write to prevent store buffer reordering.
 *
 * @param index       Window index 0-223. Entries 0-8 may be overwritten by ARC;
 *                    use the reservations in rt/x280_noc_map.h rather than
 *                    treating low indices as generally safe.
 * @param noc_x       Destination NOC X coordinate.
 * @param noc_y       Destination NOC Y coordinate.
 * @param addr        Destination address. Low 21 bits become the window offset.
 * @param posted      1 = posted (fire-and-forget), 0 = non-posted (ACK required).
 * @param strict_order 1 = enforce in-order NOC delivery.
 * @return  Volatile pointer into the window (offset = low 21 bits of addr).
 */
static inline volatile void* noc_configure_tlb_2m(
    uint32_t index, uint32_t noc_x, uint32_t noc_y, uint64_t addr, int posted, int strict_order) {
    volatile uint32_t* cfg = (volatile uint32_t*)(NOC_TLB_2M_CONFIG_BASE + (uint64_t)index * 0x10UL);

    /* Zeroed word by word rather than with `= {0}`, which GCC is free to lower
     * to a memset call -- and these images build -nostdlib with no libc to
     * resolve it, so the choice shows up as a link failure in whichever image
     * inlines this into a cold path. */
    noc_tlb_2m_t tlb;
    tlb.data[0] = 0;
    tlb.data[1] = 0;
    tlb.data[2] = 0;
    tlb.data[3] = 0;
    tlb.local_offset = addr >> 21;
    tlb.x_end = noc_x;
    tlb.y_end = noc_y;
    tlb.x_start = noc_x;
    tlb.y_start = noc_y;
    tlb.ordering = (posted ? 2u : 0u) | (strict_order ? 1u : 0u);

    x280_fence_iorw();
    cfg[0] = tlb.data[0];
    cfg[1] = tlb.data[1];
    cfg[2] = tlb.data[2];
    cfg[3] = tlb.data[3];
    x280_fence_iorw();

    uint64_t offset = addr & (NOC_2M_WINDOW_STRIDE - 1ULL);
    return (volatile void*)(NOC_2M_WINDOW_BASE + (uint64_t)index * NOC_2M_WINDOW_STRIDE + offset);
}

/**
 * Extended TLB configuration: exposes all bitfields for experiments.
 *
 * The caller pre-fills a noc_tlb_2m_t with desired fields (linked,
 * noc_sel, the keep/skip pairs, static_vc and its buddy and class, etc.)
 * before calling. This function writes the config and returns the
 * System Port window pointer.
 *
 * @param index       Window index 0-223.
 * @param tlb         Fully populated TLB bitfield struct.
 * @param use_memport If nonzero, return Memory Port pointer instead of System Port.
 * @return  Volatile pointer into the window.
 */
static inline volatile void* noc_configure_tlb_2m_ext(uint32_t index, noc_tlb_2m_t* tlb, int use_memport) {
    volatile uint32_t* cfg = (volatile uint32_t*)(NOC_TLB_2M_CONFIG_BASE + (uint64_t)index * 0x10UL);

    x280_fence_iorw();
    cfg[0] = tlb->data[0];
    cfg[1] = tlb->data[1];
    cfg[2] = tlb->data[2];
    cfg[3] = tlb->data[3];
    x280_fence_iorw();

    uint64_t base = use_memport ? NOC_2M_WINDOW_BASE_MEMPORT : NOC_2M_WINDOW_BASE;
    uint64_t offset = ((uint64_t)tlb->local_offset << 21) & (NOC_2M_WINDOW_STRIDE - 1ULL);
    return (volatile void*)(base + (uint64_t)index * NOC_2M_WINDOW_STRIDE + offset);
}

/**
 * Helper: populate a noc_tlb_2m_t for a window aimed at addr.
 *
 * multicast_en 0 makes (x_end, y_end) the single target tile and leaves the
 * start pair unread by hardware; 1 makes the two pairs opposite corners of a
 * rectangle of Tensix tiles. TLBWindows.md, "Configuration": "mcast ... true
 * causes (x_start, y_start) through (x_end, y_end) to specify a rectangle of
 * target Tensix tiles, whereas false causes (x_end, y_end) to specify a single
 * tile target".
 *
 * num_destinations is the destination-count override. Zero is permitted only
 * when the window does no keep/skip masking and sets no exclusion, in which
 * case "hardware will substitute the correct value" (TLBWindows.md,
 * "Configuration"). Whether a non-zero override is honoured under posted writes
 * is not documented.
 *
 * Which corner is the start is the caller's to decide: under coordinate
 * translation "when performing broadcasts, StartX <-> EndX need to be swapped
 * by software, and likewise StartY <-> EndY ... because coordinate translation
 * does not change the direction of data flow"
 * (tt-isa-documentation, BlackholeA0/NoC/Coordinates.md:24). The fields are
 * written as given.
 *
 * Coordinates are not bounded here -- each is a six-bit field and an
 * out-of-range value carries into its neighbour. noc_configure_tlb_2m_mcast is
 * the entry point that refuses one.
 */
static inline void noc_fill_tlb_2m(
    noc_tlb_2m_t* tlb,
    uint32_t x_start,
    uint32_t y_start,
    uint32_t x_end,
    uint32_t y_end,
    uint64_t addr,
    noc_ordering_t ordering,
    int multicast_en,
    uint32_t num_destinations) {
    tlb->local_offset = addr >> 21;
    tlb->x_end = x_end;
    tlb->y_end = y_end;
    tlb->x_start = x_start;
    tlb->y_start = y_start;
    tlb->ordering = (uint32_t)ordering;
    tlb->mcast = multicast_en ? 1u : 0u;
    tlb->num_destinations_override = num_destinations;
}

/**
 * Verify TLB config register read-back matches expected values.
 *
 * Reads the 4 config words back from the Peripheral Port and compares
 * them to the expected union. Use after noc_configure_tlb_2m() to
 * confirm the Peripheral Port accepted the configuration.
 *
 * @param index     TLB window index (must match the configure call).
 * @param expected  Pointer to the noc_tlb_2m_t used during configure.
 * @return  0 on success, -1 on mismatch.
 */
static inline int noc_verify_tlb_2m(uint32_t index, const noc_tlb_2m_t* expected) {
    volatile uint32_t* cfg = (volatile uint32_t*)(NOC_TLB_2M_CONFIG_BASE + (uint64_t)index * 0x10UL);
    for (int i = 0; i < 4; i++) {
        if (cfg[i] != expected->data[i]) {
            return -1;
        }
    }
    return 0;
}

/* Widest coordinate the window configuration can hold. x and y each occupy a
 * six-bit field, so 64 does not fail -- it carries into the neighbouring field
 * and the window then aims at a real tile that is the wrong one. */
#define NOC_MAX_COORD X280_NOC_COORD_MAX

/**
 * Program a window aimed at the rectangle (x_start, y_start) through
 * (x_end, y_end), with every coordinate and the index refused rather than
 * truncated.
 *
 * Both corner pairs are bounded against NOC_MAX_COORD, not just the end pair:
 * under multicast the start pair aims the window as much as the end pair does,
 * and a masked start coordinate names a different real tile, so it would
 * silently widen or move the rectangle. Returns NULL and programs nothing when
 * any coordinate exceeds NOC_MAX_COORD, the ordering is not a noc_ordering_t
 * member, or the index is past NOC_2M_WINDOW_COUNT.
 *
 * multicast_en 0 targets the single tile (x_end, y_end) and ignores the start
 * pair, per TLBWindows.md; noc_configure_tlb_2m_checked is that case.
 *
 * num_destinations_override must be 0 unless the window does keep/skip masking
 * or sets an exclusion, neither of which this entry point programs; zero makes
 * hardware substitute the count (TLBWindows.md, "Configuration"). A non-zero
 * value is passed through as given and is unverified under posted writes, which
 * the documentation does not cover.
 *
 * Refuses, never masks. A bounded coordinate is still only a well-formed one --
 * reaching a harvested or router-only tile hangs the access rather than
 * faulting, so the caller remains responsible for aiming at tiles that
 * answer.
 *
 * `ordering` must be a noc_ordering_t member (noc_ordering_valid); Counted
 * Writes is reached only through the raw entry points.
 *
 * @param programmed  Receives the words this call intends the entry to hold, so
 *                    a caller can pass it straight to noc_verify_tlb_2m.
 *                    Untouched on refusal.
 * @return  As noc_configure_tlb_2m, or NULL when an argument was refused.
 */
static inline volatile void* noc_configure_tlb_2m_mcast(
    uint32_t index,
    uint32_t x_start,
    uint32_t y_start,
    uint32_t x_end,
    uint32_t y_end,
    uint64_t addr,
    noc_ordering_t ordering,
    int multicast_en,
    uint32_t num_destinations_override,
    noc_tlb_2m_t* programmed) {
    if (programmed == (noc_tlb_2m_t*)0) {
        return (volatile void*)0;
    }
    if (x_start > NOC_MAX_COORD || y_start > NOC_MAX_COORD || x_end > NOC_MAX_COORD || y_end > NOC_MAX_COORD ||
        num_destinations_override > 0xFFu || !noc_ordering_valid(ordering) || index >= NOC_2M_WINDOW_COUNT) {
        return (volatile void*)0;
    }
    /* Word by word, not `= {0}`: see the note in noc_configure_tlb_2m. */
    programmed->data[0] = 0;
    programmed->data[1] = 0;
    programmed->data[2] = 0;
    programmed->data[3] = 0;
    noc_fill_tlb_2m(
        programmed, x_start, y_start, x_end, y_end, addr, ordering, multicast_en, num_destinations_override);
    /* The _ext form because it takes the words already built. It returns the
     * window base rather than base plus offset, so the low 21 bits of addr are
     * added here to match noc_configure_tlb_2m's return. */
    (void)noc_configure_tlb_2m_ext(index, programmed, /*use_memport=*/0);
    return (volatile void*)(NOC_2M_WINDOW_BASE + (uint64_t)index * NOC_2M_WINDOW_STRIDE +
                            (addr & (NOC_2M_WINDOW_STRIDE - 1ULL)));
}

/**
 * noc_configure_tlb_2m with the coordinates and the index refused rather than
 * truncated.
 *
 * For a window aimed by a value the firmware did not choose -- a host-supplied
 * coordinate, a peer address read out of a mailbox. Returns NULL and programs
 * nothing when x or y exceeds NOC_MAX_COORD, the ordering is not a
 * noc_ordering_t member, or the index is past NOC_2M_WINDOW_COUNT.
 *
 * Refuses, never masks: masking a coordinate produces a different real tile,
 * which is the failure this exists to stop rather than a safe fallback. A
 * bounded coordinate is still only a well-formed one -- reaching a harvested or
 * router-only tile hangs the access rather than faulting, so the caller
 * remains responsible for aiming at a tile that answers.
 *
 * `ordering` must be a noc_ordering_t member (noc_ordering_valid); Counted
 * Writes is reached only through the raw entry points.
 *
 * @param programmed  Receives the words this call intends the entry to hold, so
 *                    a caller can pass it straight to noc_verify_tlb_2m.
 *                    Deriving them a second time at the call site is the
 *                    duplication this avoids: a field added to the fill would
 *                    then read as a hardware mismatch. Untouched on refusal,
 *                    and never read back from the register -- a read-back here
 *                    would make the verification compare the register against
 *                    itself.
 * @return  As noc_configure_tlb_2m, or NULL when an argument was refused.
 */
static inline volatile void* noc_configure_tlb_2m_checked(
    uint32_t index, uint32_t noc_x, uint32_t noc_y, uint64_t addr, noc_ordering_t ordering, noc_tlb_2m_t* programmed) {
    return noc_configure_tlb_2m_mcast(
        index,
        noc_x,
        noc_y,
        noc_x,
        noc_y,
        addr,
        ordering,
        /*multicast_en=*/0,
        /*num_destinations_override=*/0u,
        programmed);
}

/**
 * Zero one 2 MiB window's configuration and confirm the zeroing took.
 *
 * Zero is not a disabled entry -- the bitfield has no valid bit, so an all-zero
 * window still routes to NOC0 (0, 0) at offset 0. It is inert in the sense that
 * matters: it names no tile the caller was reaching, so an address the caller's
 * peer once owned is no longer reachable through this slot. An image clears its
 * window on every exit path for that reason, and a zeroing that did not land
 * must not read as one that did.
 *
 * An index past NOC_2M_WINDOW_COUNT writes nothing and returns -1, so an error
 * path holding an untrusted index cannot store past the TLB block.
 *
 * @return  0 when all four config words read back zero, -1 otherwise.
 */
static inline int noc_clear_tlb_2m(uint32_t index) {
    if (index >= NOC_2M_WINDOW_COUNT) {
        return -1;
    }
    volatile uint32_t* cfg = (volatile uint32_t*)(NOC_TLB_2M_CONFIG_BASE + (uint64_t)index * 0x10UL);
    x280_fence_iorw();
    cfg[0] = 0;
    cfg[1] = 0;
    cfg[2] = 0;
    cfg[3] = 0;
    x280_fence_iorw();
    if (cfg[0] != 0 || cfg[1] != 0 || cfg[2] != 0 || cfg[3] != 0) {
        return -1;
    }
    return 0;
}

/** How noc_open_window_2m ended. */
typedef enum {
    NOC_WINDOW_OPEN = 0,      /* programmed, and read back as programmed */
    NOC_WINDOW_REFUSED = 1,   /* an argument was refused; nothing was written */
    NOC_WINDOW_UNVERIFIED = 2 /* read back otherwise; the slot is left as it reads */
} noc_window_status_t;

/**
 * Program 2 MiB window `slot` aimed at `at` with `ordering`, and read it back.
 *
 * Never clears. Why a slot reads back otherwise than programmed is not
 * determined -- another writer to the tile's one config block, or this write
 * not latching -- so whether that mapping is the
 * caller's to release is the caller's policy, which it states where it calls
 * this.
 *
 * @param window  Receives the window pointer (offset = low 21 bits of `addr`)
 *                whenever this call wrote the slot: on NOC_WINDOW_OPEN and
 *                NOC_WINDOW_UNVERIFIED, so a caller's cleanup that keys on it
 *                still sees an unverified slot. NULL on NOC_WINDOW_REFUSED.
 * @return  NOC_WINDOW_REFUSED when noc_configure_tlb_2m_checked refuses the
 *          slot, `at` or `ordering`.
 */
static inline noc_window_status_t noc_open_window_2m(
    uint32_t slot, x280_noc_coord_t at, uint64_t addr, noc_ordering_t ordering, volatile void** window) {
    noc_tlb_2m_t programmed;
    *window = noc_configure_tlb_2m_checked(slot, at.x, at.y, addr, ordering, &programmed);
    if (*window == (volatile void*)0) {
        return NOC_WINDOW_REFUSED;
    }
    return noc_verify_tlb_2m(slot, &programmed) == 0 ? NOC_WINDOW_OPEN : NOC_WINDOW_UNVERIFIED;
}

/**
 * Write a 64-bit value through a NOC window as two 32-bit stores.
 *
 * No fence: callers batch these and fence once. Use noc_write_verify_u64 for a single fenced, confirmed
 * write. Two u32 stores rather than one u64 is the measured shape; memcpy()
 * does not work through a NOC window.
 *
 * @param window_ptr  Pointer returned by noc_configure_tlb_2m().
 * @param value       64-bit value to write (lo half first).
 */
static inline void noc_write_u64(volatile void* window_ptr, uint64_t value) {
    volatile uint32_t* lo = (volatile uint32_t*)window_ptr;
    lo[0] = (uint32_t)(value & 0xFFFFFFFFU);
    lo[1] = (uint32_t)(value >> 32);
}

/**
 * Write a 64-bit value through a NOC window and poll for read-back match.
 *
 * Writes the value as two 32-bit stores (lo then hi) followed by a store
 * fence (fence ow, ow). Then polls the same address until the read-back
 * matches or the iteration count is exhausted.
 *
 * For posted windows: nothing orders the read-back behind the stores, so an
 * early read can return the old value. The poll, not the ordering, is what
 * confirms the data landed: it ends only when the value read matches.
 *
 * For non-posted windows: each store gets an ACK before the next one
 * proceeds, and the read-back directly confirms the data landed.
 *
 * WARNING: Do not use memcpy() to write through NOC windows.
 * Always use explicit volatile pointer stores.
 *
 * @param window_ptr  Pointer returned by noc_configure_tlb_2m().
 * @param value       64-bit value to write and verify.
 * @param timeout     Max polling iterations (~500M = ~500ms at 1 GHz).
 * @return  0 on success (data confirmed at destination), -1 on timeout.
 */
static inline int noc_write_verify_u64(volatile void* window_ptr, uint64_t value, uint32_t timeout) {
    volatile uint32_t* lo = (volatile uint32_t*)window_ptr;
    volatile uint32_t* hi = lo + 1;

    *lo = (uint32_t)(value & 0xFFFFFFFFU);
    *hi = (uint32_t)(value >> 32);
    x280_fence_ow();

    for (uint32_t i = 0; i < timeout; i++) {
        uint32_t r_lo = *lo;
        uint32_t r_hi = *hi;
        uint64_t rb = ((uint64_t)r_hi << 32) | r_lo;
        if (rb == value) {
            return 0;
        }
    }
    return -1; /* timeout */
}

/* ------------------------------------------------------------------
 * NOC read helpers
 * Reads through System Port TLB windows are inherently non-posted:
 * each load stalls the pipeline until the NOC response arrives.
 * No fence needed — the load itself is the synchronization.
 * ------------------------------------------------------------------ */

/**
 * Read a 64-bit value from a NOC TLB window.
 * Two 32-bit loads (lo then hi). Each load stalls until NOC response arrives.
 *
 * @param window_ptr  Pointer returned by noc_configure_tlb_2m().
 * @return  64-bit value from remote address.
 */
static inline uint64_t noc_read_u64(volatile void* window_ptr) {
    volatile uint32_t* lo = (volatile uint32_t*)window_ptr;
    volatile uint32_t* hi = lo + 1;
    uint32_t r_lo = *lo;
    uint32_t r_hi = *hi;
    return ((uint64_t)r_hi << 32) | r_lo;
}

static inline uint32_t noc_read_u32(volatile void* window_ptr) { return *(volatile uint32_t*)window_ptr; }

static inline uint16_t noc_read_u16(volatile void* window_ptr) { return *(volatile uint16_t*)window_ptr; }

static inline uint8_t noc_read_u8(volatile void* window_ptr) { return *(volatile uint8_t*)window_ptr; }

#endif /* NOC_H */
