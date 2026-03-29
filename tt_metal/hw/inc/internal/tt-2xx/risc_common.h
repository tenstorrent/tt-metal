// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#ifndef _RISC_COMMON_H_
#define _RISC_COMMON_H_

#include <stdint.h>

#include <cstdint>

#include "eth_l1_address_map.h"
#include "limits.h"
#include "internal/mod_div_lib.h"
#include "noc_overlay_parameters.h"
#include "noc_parameters.h"
#include "stream_io_map.h"
#include "tensix.h"
#include "tensix_neo_reg.h"
#include "api/debug/assert.h"

#define NOC_X(x) NOC_0_X(noc_index, noc_size_x, (x))
#define NOC_Y(y) NOC_0_Y(noc_index, noc_size_y, (y))
#define DYNAMIC_NOC_X(noc, x) NOC_0_X(noc, noc_size_x, (x))
#define DYNAMIC_NOC_Y(noc, y) NOC_0_Y(noc, noc_size_y, (y))
#define NOC_X_PHYS_COORD(x) NOC_0_X_PHYS_COORD(noc_index, noc_size_x, x)
#define NOC_Y_PHYS_COORD(y) NOC_0_Y_PHYS_COORD(noc_index, noc_size_y, y)

#define TILE_WORD_2_BIT ((256 + 64 + 32) >> 4)
#define TILE_WORD_4_BIT ((512 + 64 + 32) >> 4)
#define TILE_WORD_8_BIT ((32 * 32 * 1 + 64 + 32) >> 4)
#define TILE_WORD_16_BIT ((32 * 32 * 2 + 32) >> 4)
#define TILE_WORD_32_BIT ((32 * 32 * 4 + 32) >> 4)

const uint32_t STREAM_RESTART_CHECK_MASK = (0x1 << 3) - 1;

const uint32_t MAX_TILES_PER_PHASE = 2048;

// These values are defined in each core type's FW .cc file

// Virtual X coordinate
extern uint8_t my_x[NUM_NOCS];

// Virtual Y coordinate
extern uint8_t my_y[NUM_NOCS];

inline void WRITE_REG(uintptr_t addr, uint32_t val) {
    volatile tt_reg_ptr uint32_t* ptr = (volatile tt_reg_ptr uint32_t*)addr;
    ptr[0] = val;
}

inline uint32_t READ_REG(uintptr_t addr) {
    volatile tt_reg_ptr uint32_t* ptr = (volatile tt_reg_ptr uint32_t*)addr;
    return ptr[0];
}

inline uint32_t dram_io_incr_ptr(uint32_t curr_ptr, uint32_t incr, uint32_t buf_size_q_slots) {
    uint32_t next_ptr = curr_ptr + incr;
    uint32_t double_buf_size_q_slots = 2 * buf_size_q_slots;
    if (next_ptr >= double_buf_size_q_slots) {
        next_ptr -= double_buf_size_q_slots;
    }
    return next_ptr;
}

inline __attribute__((always_inline)) uint32_t dram_io_empty(uint32_t rd_ptr, uint32_t wr_ptr) {
    return (rd_ptr == wr_ptr);
}

inline __attribute__((always_inline)) uint32_t
dram_io_local_empty(uint32_t local_rd_ptr, uint32_t rd_ptr, uint32_t wr_ptr) {
    if (rd_ptr == wr_ptr) {
        return true;
    }

    uint32_t case1 = rd_ptr < wr_ptr && (local_rd_ptr < rd_ptr || local_rd_ptr >= wr_ptr);
    uint32_t case2 = rd_ptr > wr_ptr && wr_ptr <= local_rd_ptr && local_rd_ptr < rd_ptr;

    return case1 || case2;
}

inline uint32_t dram_io_full(uint32_t rd_ptr, uint32_t wr_ptr, uint32_t buf_size_q_slots) {
    uint32_t wr_ptr_reduced_by_q_slots = wr_ptr - buf_size_q_slots;
    uint32_t rd_ptr_reduced_by_q_slots = rd_ptr - buf_size_q_slots;
    uint32_t case1 = (wr_ptr_reduced_by_q_slots == rd_ptr);
    uint32_t case2 = (rd_ptr_reduced_by_q_slots == wr_ptr);
    return case1 || case2;
}

inline __attribute__((always_inline)) uint32_t buf_ptr_inc_wrap(uint32_t buf_ptr, uint32_t inc, uint32_t buf_size) {
    uint32_t result = buf_ptr + inc;
    if (result >= buf_size) {
        result -= buf_size;
    }
    return result;
}

inline __attribute__((always_inline)) uint32_t buf_ptr_dec_wrap(uint32_t buf_ptr, uint32_t dec, uint32_t buf_size) {
    uint32_t result = buf_ptr;
    if (dec > result) {
        result += buf_size;
    }
    result -= dec;
    return result;
}

// This definition of reg_read conflicts with the one in
// tt_metal/third_party/tt_llk_wormhole_b0/common/inc/ckernel.h, which trisc
// kernels bring into the global namespace using "using namespace ckernel".
#if !defined(COMPILE_FOR_TRISC)  // BRISC, NCRISC, ERISC, IERISC
inline __attribute__((always_inline)) uint32_t reg_read(uintptr_t addr) {
    volatile tt_reg_ptr uint32_t* p_reg = reinterpret_cast<volatile tt_reg_ptr uint32_t*>(addr);
    return p_reg[0];
}
#endif

inline void assert_trisc_reset() {
    uint32_t soft_reset_0 = READ_REG(NEO_REGS_0__LOCAL_REGS_DEBUG_REGS_SOFT_RESET_0_REG_ADDR);
    uint32_t trisc_reset_mask = T6_DEBUG_REGS_SOFT_RESET_0_RISC_CONTROL_SOFT_RESET_MASK;
    WRITE_REG(NEO_REGS_0__LOCAL_REGS_DEBUG_REGS_SOFT_RESET_0_REG_ADDR, soft_reset_0 | trisc_reset_mask);
}

inline void deassert_trisc_reset() {
    uint32_t soft_reset_0 = READ_REG(NEO_REGS_0__LOCAL_REGS_DEBUG_REGS_SOFT_RESET_0_REG_ADDR);
    uint32_t trisc_reset_mask = T6_DEBUG_REGS_SOFT_RESET_0_RISC_CONTROL_SOFT_RESET_MASK;
    WRITE_REG(NEO_REGS_0__LOCAL_REGS_DEBUG_REGS_SOFT_RESET_0_REG_ADDR, soft_reset_0 & ~trisc_reset_mask);
    soft_reset_0 = READ_REG(NEO_REGS_1__LOCAL_REGS_DEBUG_REGS_SOFT_RESET_0_REG_ADDR);
    WRITE_REG(NEO_REGS_1__LOCAL_REGS_DEBUG_REGS_SOFT_RESET_0_REG_ADDR, soft_reset_0 & ~trisc_reset_mask);
    soft_reset_0 = READ_REG(NEO_REGS_2__LOCAL_REGS_DEBUG_REGS_SOFT_RESET_0_REG_ADDR);
    WRITE_REG(NEO_REGS_2__LOCAL_REGS_DEBUG_REGS_SOFT_RESET_0_REG_ADDR, soft_reset_0 & ~trisc_reset_mask);
    soft_reset_0 = READ_REG(NEO_REGS_3__LOCAL_REGS_DEBUG_REGS_SOFT_RESET_0_REG_ADDR);
    WRITE_REG(NEO_REGS_3__LOCAL_REGS_DEBUG_REGS_SOFT_RESET_0_REG_ADDR, soft_reset_0 & ~trisc_reset_mask);
}

inline uint32_t special_mult(uint32_t a, uint32_t special_b) {
    if (special_b == TILE_WORD_8_BIT) {
        return a * TILE_WORD_8_BIT;
    } else if (special_b == TILE_WORD_16_BIT) {
        return a * TILE_WORD_16_BIT;
    } else if (special_b == TILE_WORD_4_BIT) {
        return a * TILE_WORD_4_BIT;
    } else if (special_b == TILE_WORD_2_BIT) {
        return a * TILE_WORD_2_BIT;
    } else if (special_b == TILE_WORD_32_BIT) {
        return a * TILE_WORD_32_BIT;
    }

    while (true);
    return 0;
}

// Invalidates Blackhole's entire L1 cache
// Blackhole L1 cache is a small write-through cache (4x16B L1 lines). The cache covers all of L1 (no
// MMU or range registers).
//  Writing an address on one proc and reading it from another proc only requires the reader to invalidate.
//  Need to invalidate any address written by noc that may have been previously read by riscv
#if !defined(ARCH_QUASAR)
inline __attribute__((always_inline)) void invalidate_l1_cache() {
#if defined(ARCH_BLACKHOLE)
    asm("fence");
#endif
}
#endif  // !ARCH_QUASAR

#if defined(ARCH_QUASAR) && defined(COMPILE_FOR_DM)
// =============================================================================
// Quasar DM Core Cache Management
// =============================================================================
//
// Cache hierarchy (per DM core):
//   L1 D$ : 4KB write-back, 2-way, 64B lines, private per core
//   L1 I$ : 4KB, 2-way, private per core
//   L2    : 128KB write-back, 4-way, 64B lines, shared between DM cores
//   TL1   : Tensix L1 (node memory), visible to host and other cores
//
// Data path: Core -> L1 D$ -> L2 -> TL1
//
// Key points:
//   - Writes go through L1 D$ and L2 before reaching TL1
//   - Flush is required for data to be visible to other agents
//   - L1 and L2 are coherent: L2 flush probes L1 D$ for dirty data before writing to TL1
//
// References:
//   - SiFive X280 Core Manual, sections 3.4.2, 6.1.1, 6.1.2 (CFLUSH.D.L1, CDISCARD.D.L1)
//   - Chipyard rocket-chip (basis for Quasar DM cores)
//   - overlay/software/metal_lib/freedom-metal/src/cache.c (reference implementation)
//
// =============================================================================

#include "internal/tt-2xx/quasar/overlay/overlay_addresses.h"

// -----------------------------------------------------------------------------
// L1 Data Cache Operations
// -----------------------------------------------------------------------------
// Uses tt.cache.cflush.d.l1 and tt.cache.cdiscard.d.l1 instructions.
// If rs1=x0, operates on entire L1 D$.

// Flush L1 D$ line (or entire cache if addr=0) to L2.
// Writes back dirty data to L2 and invalidates the line.
inline __attribute__((always_inline)) void flush_l1_dcache(uintptr_t addr) {
    if (addr) {
        __asm__ __volatile__("tt.cache.cflush.d.l1 %0" :: "r"(addr) : "memory");
    } else {
        __asm__ __volatile__("tt.cache.cflush.d.l1 x0" ::: "memory");
    }
    __asm__ __volatile__("fence" ::: "memory");
}

// Invalidate L1 D$ line (or entire cache if addr=0) without writeback.
// Discards dirty data - use only when data is known to be stale.
inline __attribute__((always_inline)) void invalidate_l1_dcache(uintptr_t addr) {
    if (addr) {
        __asm__ __volatile__("tt.cache.cdiscard.d.l1 %0" :: "r"(addr) : "memory");
    } else {
        __asm__ __volatile__("tt.cache.cdiscard.d.l1 x0" ::: "memory");
    }
    __asm__ __volatile__("fence" ::: "memory");
}

// -----------------------------------------------------------------------------
// L1 Instruction Cache Operations
// -----------------------------------------------------------------------------

// Invalidate entire L1 I$ using FENCE.I instruction.
// Required after modifying instruction memory before jumping to new code.
inline __attribute__((always_inline)) void invalidate_l1_icache() {
    __asm__ __volatile__("fence.i" ::: "memory");
}

// -----------------------------------------------------------------------------
// L2 Cache Operations
// -----------------------------------------------------------------------------
// L2 is controlled via memory-mapped registers in the cache controller.
// See overlay_addresses.h for register definitions and geometry.

// Flush a single 64B cache line from L2 to TL1 (node memory).
// Probes L1 D$ for dirty data before flushing - no need to flush L1 first.
inline __attribute__((always_inline)) void flush_l2_cache_line(uintptr_t addr) {
    __asm__ __volatile__("fence" ::: "memory");
    volatile uint64_t* flush_reg = (volatile uint64_t*)L2_FLUSH_ADDR;
    *flush_reg = (uint64_t)addr;
    __asm__ __volatile__("fence" ::: "memory");
}

// Invalidate a single 64B cache line from L2 without writeback.
// Discards dirty data - use only when data is known to be stale.
inline __attribute__((always_inline)) void invalidate_l2_cache_line(uintptr_t addr) {
    __asm__ __volatile__("fence" ::: "memory");
    volatile uint64_t* inv_reg = (volatile uint64_t*)L2_INVALIDATE_ADDR;
    *inv_reg = (uint64_t)addr;
    __asm__ __volatile__("fence" ::: "memory");
}

// Flush a range of addresses from L2 to TL1.
// Flushes all cache lines covering [start_addr, start_addr + size).
inline __attribute__((always_inline)) void flush_l2_cache_range(uintptr_t start_addr, size_t size) {
    uintptr_t aligned_start = start_addr & ~(uintptr_t)63;  // align to 64B
    uintptr_t end_addr = start_addr + size;

    for (uintptr_t addr = aligned_start; addr < end_addr; addr += 64) {
        flush_l2_cache_line(addr);
    }
}

// Flush entire L2 cache to TL1.
// Iterates FLUSH64 over all cacheable TL1 addresses (4MB).
inline void flush_l2_cache_full() {
    __asm__ __volatile__("fence" ::: "memory");

    volatile uint64_t* flush_reg = (volatile uint64_t*)L2_FLUSH_ADDR;
    for (uint32_t addr = 0; addr < MEMORY_PORT_CACHEABLE_MEM_PORT_MEM_SIZE; addr += L2_CACHE_LINE_SIZE) {
        *flush_reg = (uint64_t)addr;
    }

    __asm__ __volatile__("fence" ::: "memory");
}

// Coordinated L2 invalidation across all DM cores.
// Each core signals ready by writing its bit, then polls until HW clears register.
// Call from all DM cores, or write other cores' bits if only one core is active.
inline void invalidate_l2_cache(uint32_t hartid) {
    volatile uint64_t* inv_reg = (volatile uint64_t*)L2_FULL_INVALIDATE_ADDR;
    *inv_reg = (uint64_t)(1 << hartid);
    while (*inv_reg != 0);  // Wait for HW to complete and clear
}

// -----------------------------------------------------------------------------
// Combined Cache Operations
// -----------------------------------------------------------------------------

// Invalidate entire L1 cache (D$ + I$) on this core.
// Provided for API compatibility with previous architectures.
// Uses flush (not invalidate) for D$ since older architectures had write-through caches.
inline void invalidate_l1_cache() {
    flush_l1_dcache(0);
    invalidate_l1_icache();
}

// Invalidate entire cache hierarchy: L2 + L1 D$ + L1 I$.
// Must be called from all DM cores for proper synchronization.
// After return, all caches are cold and will fetch fresh data from TL1.
inline void invalidate_cache_all(uint32_t hartid) {
    // 1. Coordinate L2 wipe across all cores
    invalidate_l2_cache(hartid);

    // 2. Invalidate local L1 (D$ + I$)
    invalidate_l1_cache();
}

#endif  // ARCH_QUASAR && COMPILE_FOR_DM

// Fallback for Quasar non-DM cores (TRISC) - no cache management needed
#if defined(ARCH_QUASAR) && !defined(COMPILE_FOR_DM)
inline __attribute__((always_inline)) void invalidate_l1_cache() {
    // No-op for non-DM cores on Quasar
}
#endif  // ARCH_QUASAR && !COMPILE_FOR_DM

template <bool enable = true>
inline __attribute__((always_inline)) void set_l1_data_cache() {
#if defined(ARCH_BLACKHOLE)
    if constexpr (enable) {
        asm(R"ASM(
            li t1, 0x8
            csrrc zero, 0x7c0, t1
             )ASM" ::
                : "t1");
#if !defined(ENABLE_HW_CACHE_INVALIDATION)
        // Disable gathering to stop HW from invalidating the data cache after 128 transactions by setting bit 24
        // This is default enabled
        asm(R"ASM(
            li   t1, 0x1
            slli t1, t1, 24
            fence
            csrrs zero, 0x7c0, t1
            )ASM" ::
                : "t1");
#endif
    } else {
        asm(R"ASM(
            fence
            li t1, 0x8
            csrrs zero, 0x7c0, t1
             )ASM" ::
                : "t1");
    }
#endif
}

// risc_init function isn't required for TRISCS
#if !defined(COMPILE_FOR_TRISC)  // BRISC, NCRISC, ERISC, IERISC
#include "noc_nonblocking_api.h"
#include "internal/tt-2xx/dataflow_buffer/dataflow_buffer_isr.h"

inline void risc_init() {
    for (uint32_t n = 0; n < NUM_NOCS; n++) {
        uint32_t noc_id_reg = MY_NOC_ENCODING(n);
        my_x[n] = noc_id_reg & NOC_NODE_ID_MASK;
        my_y[n] = (noc_id_reg >> NOC_ADDR_NODE_ID_BITS) & NOC_NODE_ID_MASK;
    }
}

inline __attribute__((interrupt, hot)) void handle_interrupt() {
    uint64_t mcause;
    asm volatile("csrr %0, mcause" : "=r"(mcause));
    if ((mcause & 0x8000000000000000) == 0) {  // this is HW exception
        ASSERT(0 == 1, debug_assert_type_t::DebugAssertHwFault);
#if !defined(WATCHER_ENABLED)  // hang anyway
        while (1) {
            ;
        }
#endif
    } else {  // otherwise it's DFB sync interrupt
        dfb_implicit_sync_handler();
    }
}

inline __attribute__((always_inline)) void setup_isr_csrs() {
    uint64_t isr_address = reinterpret_cast<uint64_t>(handle_interrupt);
    asm volatile("csrw mtvec, %0" : : "r"(isr_address));  // set the interrupt handler
}

inline __attribute__((always_inline)) void enable_dfb_tile_isr() {
    // Enable ROCC interrupt in mie
    uint64_t mie_val;
    asm volatile("csrr %0, mie" : "=r"(mie_val));
    mie_val |= (1 << 13);
    asm volatile("csrrs zero, mie, %0" : : "r"(mie_val));

    // Enable MIE in mstatus
    uint64_t mstatus_val;
    asm volatile("csrr %0, mstatus" : "=r"(mstatus_val));
    mstatus_val |= (1 << 3);
    asm volatile("csrrs zero, mstatus, %0" : : "r"(mstatus_val));
}

inline __attribute__((always_inline)) void disable_dfb_tile_isr() {
    // Disable ROCC interrupt in mie
    uint64_t mie_val;
    asm volatile("csrr %0, mie" : "=r"(mie_val));
    mie_val &= ~(1 << 13);
    asm volatile("csrrc zero, mie, %0" : : "r"(mie_val));

    // Disable MIE in mstatus
    uint64_t mstatus_val;
    asm volatile("csrr %0, mstatus" : "=r"(mstatus_val));
    mstatus_val &= ~(1 << 3);
    asm volatile("csrrc zero, mstatus, %0" : : "r"(mstatus_val));
}

#endif  // !defined(COMPILE_FOR_TRISC)

// Helper function to wait for a specified number of cycles, safe to call in erisc kernels.
#if defined(COMPILE_FOR_ERISC)
#include "internal/ethernet/erisc.h"
#endif
inline void riscv_wait(uint32_t cycles) {
    volatile uint tt_reg_ptr* clock_lo =
        reinterpret_cast<volatile uint tt_reg_ptr*>(NEO_REGS_0__LOCAL_REGS_DEBUG_REGS_WALL_CLOCK_0_REG_ADDR);
    volatile uint tt_reg_ptr* clock_hi =
        reinterpret_cast<volatile uint tt_reg_ptr*>(NEO_REGS_0__LOCAL_REGS_DEBUG_REGS_WALL_CLOCK_1_REG_ADDR);
    uint64_t wall_clock_timestamp = clock_lo[0] | ((uint64_t)clock_hi[0] << 32);
    uint64_t wall_clock = 0;
    do {
#if defined(COMPILE_FOR_ERISC)
        internal_::risc_context_switch();
#endif
        wall_clock = clock_lo[0] | ((uint64_t)clock_hi[0] << 32);
    } while (wall_clock < (wall_clock_timestamp + cycles));
}

// Flush i$ on ethernet riscs
inline __attribute__((always_inline)) void flush_erisc_icache() {
#ifdef ARCH_BLACKHOLE
#pragma GCC unroll 2048
    for (int i = 0; i < 2048; i++) {
        __asm__ volatile("nop");
    }
#else
#pragma GCC unroll 128
    for (int i = 0; i < 128; i++) {
        __asm__ volatile("nop");
    }
#endif
}

// Zero a buffer in L1 memory
void zero_l1_buf(tt_l1_ptr uint32_t* buf, uint32_t size_bytes) {
    for (uint32_t i = 0; i < size_bytes / 4; i++) {
        buf[i] = 0;
    }
}

// Get the wall clock timestamp. Reading RISCV_DEBUG_REG_WALL_CLOCK_L samples/freezes (for readback)
// upper 32 bits of the 64-bit timestamp. Upper 32 bits are read from RISCV_DEBUG_REG_WALL_CLOCK_H.
inline uint64_t get_timestamp() {
    volatile uint timestamp_low =
        *reinterpret_cast<volatile uint tt_reg_ptr*>(NEO_REGS_0__LOCAL_REGS_DEBUG_REGS_WALL_CLOCK_0_REG_ADDR);
    volatile uint timestamp_high =
        *reinterpret_cast<volatile uint tt_reg_ptr*>(NEO_REGS_0__LOCAL_REGS_DEBUG_REGS_WALL_CLOCK_1_REG_ADDR);
    return (((uint64_t)timestamp_high) << 32) | timestamp_low;
}

// Get only the lower 32 bits of the wall clock timestamp
inline uint32_t get_timestamp_32b() {
    return *reinterpret_cast<volatile uint tt_reg_ptr*>(NEO_REGS_0__LOCAL_REGS_DEBUG_REGS_WALL_CLOCK_0_REG_ADDR);
}

#endif
