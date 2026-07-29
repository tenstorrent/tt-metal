/*
 * noc.h - NOC TLB configuration and write helpers for X280 bare-metal
 *
 * Provides 2M TLB window programming and NOC write-verify primitives.
 * Reused by Phase 5 (NOC reads) and Phase 6 (ordering) firmware.
 *
 * Style: free inline functions, no global state, single stdint.h dependency.
 * Source: NOC_DEEP_DIVE.md, junkcode/blackhole-thing/src/l2cpu_core.hpp:206-231,
 *         junkcode/x280-noc/driver/l2cpu_noc.c
 *
 * Build: make -C x280 noc-test
 */
#ifndef NOC_H
#define NOC_H

#include <stdint.h>

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
 * NOC_128G_WINDOW_STRIDE: width of one 128 GiB window in X280 phys
 *   address space (= 1 << 37 = NOC_BIT). The X280 system-port window
 *   base for slot i is computed in noc_configure_tlb_128g per the
 *   reference (BIG_BIT | NOC_BIT*(1+i) | SYSTEM_PORT).
 * NOC_128G_WINDOW_COUNT: hardware-defined number of 128 GiB slots. */
#define NOC_128G_CONFIG_BASE (NOC_TLB_2M_CONFIG_BASE + 0x10 * 224) /* 0x2FF00E00 */
#define NOC_128G_WINDOW_STRIDE (1ULL << 37)
#define NOC_128G_WINDOW_COUNT 32

/* ------------------------------------------------------------------
 * TLB register bitfield (128-bit, written as 4×u32)
 * Source: NOC_DEEP_DIVE.md §1.2, l2cpu_core.hpp configure_noc_tlb_2M()
 *
 * For unicast: x_end = x_start = noc_x, y_end = y_start = noc_y.
 * addr field stores target_address >> 21 (only upper 43 bits).
 * Low 21 bits of the destination address are the window *offset*.
 * ------------------------------------------------------------------ */

typedef union {
    struct {
        uint64_t addr : 43; /* target NOC address >> 21 */
        uint64_t reserved0 : 21;
        uint64_t x_end : 6;        /* NOC X (unicast destination) */
        uint64_t y_end : 6;        /* NOC Y (unicast destination) */
        uint64_t x_start : 6;      /* = x_end for unicast */
        uint64_t y_start : 6;      /* = y_end for unicast */
        uint64_t multicast_en : 1; /* 0 for unicast */
        uint64_t strict_order : 1; /* 1 = enforce in-order delivery */
        uint64_t posted : 1;       /* 1 = fire-and-forget writes */
        uint64_t linked : 1;
        uint64_t static_en : 1; /* 1 = use static_vc */
        uint64_t stream_header : 1;
        uint64_t reserved1 : 1;
        uint64_t noc_selector : 1; /* 0=NOC0, 1=NOC1 */
        uint64_t static_vc : 3;    /* virtual channel */
        uint64_t strided : 8;
        uint64_t exclude_coord_x : 5;
        uint64_t exclude_coord_y : 4;
        uint64_t exclude_dir_x : 1;
        uint64_t exclude_dir_y : 1;
        uint64_t exclude_enable : 1;
        uint64_t exclude_routing_option : 1;
        uint64_t num_destinations : 8;
    };
    uint32_t data[4];
} noc_tlb_2m_t;

/* ------------------------------------------------------------------
 * API functions (free inline, no global state)
 * ------------------------------------------------------------------ */

/* ------------------------------------------------------------------
 * 128 GiB CPU NOC TLB bitfield (96-bit, written as 3 x u32).
 *
 * Used for DMA-driven NOC->NOC transfers (see
 * x280/include/dma_engine.h::dma_engine_noc_to_noc). The 128 GiB TLB
 * region of the peripheral port follows the 224 x 16 B 2 MiB block
 * at offset 0xE00 (= 224 * 0x10); each entry is **0xC bytes** (3 x
 * u32), not 0x10. The address field is 27 bits (= addr >> 37), so a
 * single entry covers a full 128 GiB span of NOC address space at the
 * chosen (noc_x, noc_y) endpoint.
 *
 * Source: blackhole-bringup/blackhole-thing/x280/dma_benchmark.cpp
 *         Tlb128G + config_noc_tlb_128G_sysport (lines 73-105, 215-236).
 * ------------------------------------------------------------------ */

typedef union {
    struct {
        uint64_t addr : 27; /* target NOC address >> 37 */
        uint64_t reserved0 : 5;
        uint64_t x_end : 6;
        uint64_t y_end : 6;
        uint64_t x_start : 6;
        uint64_t y_start : 6;
        uint64_t multicast_en : 1;
        uint64_t strict_order : 1;
        uint64_t posted : 1;
        uint64_t linked : 1;
        uint64_t static_en : 1;
        uint64_t stream_header : 1;
        uint64_t reserved1 : 1;
        uint64_t noc_selector : 1;
        uint64_t static_vc : 3;
        uint64_t strided : 8;
        uint64_t exclude_coord_x : 5;
        uint64_t exclude_coord_y : 4;
        uint64_t exclude_dir_x : 1;
        uint64_t exclude_dir_y : 1;
        uint64_t exclude_enable : 1;
        uint64_t exclude_routing_option : 1;
        uint64_t num_destinations : 8;
    };
    uint32_t data[3];
} noc_tlb_128g_t;

/**
 * Configure a 128 GiB NOC TLB window and return the 128 GiB-aligned
 * X280 system-port window base for that slot.
 *
 * Mirrors the reference's `config_noc_tlb_128G_sysport`
 * (blackhole-bringup/blackhole-thing/x280/dma_benchmark.cpp lines
 * 215-236): writes the 96-bit bitfield as 3 x u32 stores at
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
 *                              slot value the caller writes into
 *                              X280_DMA_TLB[slot] before kicking
 *                              the channel
 *   phys[27:0]  (28 bits)  -> the DMA SAR/DAR (passed through to
 *                              phys[27:0] by the outbound DMA TLB)
 *
 * The DMA-side base of the slot is `(window - SYSTEM_PORT) >> 28`
 * (e.g. 0x8200 for index 0, 0x8400 for index 1, before the phys
 * [36:28] addend is OR'd in). See dma_engine_noc_to_noc for the
 * complete recipe.
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
 * @return  128 GiB-aligned X280 system-port window base
 *          (= BIG_BIT | NOC_BIT*(1+index) | SYSTEM_PORT, no offset).
 *          The firmware does NOT dereference this directly -- the
 *          DMA engine is the consumer.
 */
static inline uint64_t noc_configure_tlb_128g(
    uint32_t index, uint32_t noc_x, uint32_t noc_y, uint64_t addr, int posted, int strict_order) {
    /* Per-entry stride in the 128 GiB block is 0xC (3 x u32), NOT
     * 0x10. The block itself starts at NOC_128G_CONFIG_BASE
     * (= 2 MiB block end). */
    volatile uint32_t* cfg = (volatile uint32_t*)(NOC_128G_CONFIG_BASE + (uint64_t)index * 0xCUL);

    noc_tlb_128g_t tlb; /* explicit zero (avoid memset call under -nostdlib) */
    tlb.data[0] = 0;
    tlb.data[1] = 0;
    tlb.data[2] = 0;
    tlb.addr = addr >> 37;
    tlb.x_end = noc_x;
    tlb.y_end = noc_y;
    tlb.x_start = noc_x;
    tlb.y_start = noc_y;
    tlb.posted = posted ? 1u : 0u;
    tlb.strict_order = strict_order ? 1u : 0u;

    __asm__ volatile("fence iorw, iorw");
    cfg[0] = tlb.data[0];
    cfg[1] = tlb.data[1];
    cfg[2] = tlb.data[2];
    __asm__ volatile("fence iorw, iorw");

    /* X280 system-port window for 128 GiB slot `index`:
     *   BIG_BIT | (NOC_BIT * (1 + index)) | SYSTEM_PORT
     * with BIG_BIT = 1<<43, NOC_BIT = 1<<37, SYSTEM_PORT = 0x30000000.
     * (Mirrors blackhole-thing/x280/dma_benchmark.cpp line 235.)
     *
     * Index 0 -> 0x82030000000, index 1 -> 0x84030000000. The DMA-side
     * value for the outbound DMA TLB is then
     * `(window - SYSTEM_PORT) >> 28`, i.e. 0x8200 / 0x8400 / ...
     *
     * The low 37 bits of `addr` are NOT added here; see the doc
     * comment above for the caller's responsibility. The firmware
     * does not dereference this pointer (the DMA does). */
    (void)addr;
    const uint64_t k_big_bit = 1ULL << 43;
    const uint64_t k_noc_bit = 1ULL << 37;
    const uint64_t k_system_port = 0x30000000ULL;
    return k_big_bit | (k_noc_bit * (1ULL + index)) | k_system_port;
}

/**
 * Configure a 2M NOC TLB window and return the window access pointer.
 *
 * Programs TLB entry `index` to route accesses to (noc_x, noc_y, addr).
 * Full fences (fence iorw, iorw) are issued before and after the 4-word
 * config register write to prevent store buffer reordering (Pitfall 3).
 *
 * @param index       Window index 0-223. Low indices (0-7) safe for bare-metal.
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

    noc_tlb_2m_t tlb; /* explicit zero (avoid memset call under -nostdlib) */
    tlb.data[0] = 0;
    tlb.data[1] = 0;
    tlb.data[2] = 0;
    tlb.data[3] = 0;
    tlb.addr = addr >> 21;
    tlb.x_end = noc_x;
    tlb.y_end = noc_y;
    tlb.x_start = noc_x;
    tlb.y_start = noc_y;
    tlb.posted = posted ? 1u : 0u;
    tlb.strict_order = strict_order ? 1u : 0u;

    __asm__ volatile("fence iorw, iorw");
    cfg[0] = tlb.data[0];
    cfg[1] = tlb.data[1];
    cfg[2] = tlb.data[2];
    cfg[3] = tlb.data[3];
    __asm__ volatile("fence iorw, iorw");

    uint64_t offset = addr & (NOC_2M_WINDOW_STRIDE - 1ULL);
    return (volatile void*)(NOC_2M_WINDOW_BASE + (uint64_t)index * NOC_2M_WINDOW_STRIDE + offset);
}

/**
 * Extended TLB configuration: exposes all bitfields for experiments.
 *
 * The caller pre-fills a noc_tlb_2m_t with desired fields (linked,
 * noc_selector, stream_header, strided, static_en, static_vc, etc.)
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

    __asm__ volatile("fence iorw, iorw");
    cfg[0] = tlb->data[0];
    cfg[1] = tlb->data[1];
    cfg[2] = tlb->data[2];
    cfg[3] = tlb->data[3];
    __asm__ volatile("fence iorw, iorw");

    uint64_t base = use_memport ? NOC_2M_WINDOW_BASE_MEMPORT : NOC_2M_WINDOW_BASE;
    uint64_t offset = ((uint64_t)tlb->addr << 21) & (NOC_2M_WINDOW_STRIDE - 1ULL);
    return (volatile void*)(base + (uint64_t)index * NOC_2M_WINDOW_STRIDE + offset);
}

/**
 * Helper: populate a noc_tlb_2m_t for unicast to (noc_x, noc_y, addr).
 */
static inline void noc_fill_tlb_2m(
    noc_tlb_2m_t* tlb, uint32_t noc_x, uint32_t noc_y, uint64_t addr, int posted, int strict_order) {
    tlb->addr = addr >> 21;
    tlb->x_end = noc_x;
    tlb->y_end = noc_y;
    tlb->x_start = noc_x;
    tlb->y_start = noc_y;
    tlb->posted = posted ? 1u : 0u;
    tlb->strict_order = strict_order ? 1u : 0u;
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

/**
 * Write a 64-bit value through a NOC window and poll for read-back match.
 *
 * Writes the value as two 32-bit stores (lo then hi) followed by a store
 * fence (fence ow, ow). Then polls the same address until the read-back
 * matches or the iteration count is exhausted.
 *
 * For posted windows: the read-back issues a non-posted read. Per NOC
 * ordering rules, the read response arrives only after all prior posted
 * writes on the same VC to the same destination are complete.
 *
 * For non-posted windows: each store gets an ACK before the next one
 * proceeds, and the read-back directly confirms the data landed.
 *
 * WARNING: Do not use memcpy() to write through NOC windows (BUG-4).
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
    __asm__ volatile("fence ow, ow");

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
 * NOC read helpers (Phase 5)
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
