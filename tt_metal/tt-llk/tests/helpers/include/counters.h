// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#ifndef MEASURE_PERF_COUNTERS
#define MEASURE_PERF_COUNTERS(zone_name)
#endif

#include <cstdint>

#include "ckernel.h"

// ============================================================================
// L1 Address Constants — ALWAYS compiled (shared by NC and WC TRISC builds).
// Both builds produce IDENTICAL TRISC ELFs. Only BRISC behavior differs.
// ============================================================================

#define PERF_COUNTERS_BASE_ADDR    0x169000
// L1 layout: shared config + per-zone data (bank cycles + counter counts).
//   WH = 130 counters (59 INSTRN + 3 FPU + 22 TDMA_UNPACK + 14 TDMA_PACK + 32 L1)
//   BH = 169 counters (59 INSTRN + 3 FPU + 22 TDMA_UNPACK + 5 TDMA_PACK + 80 L1)
// Source of truth: tt_metal/hw/inc/internal/tt-1xx/{wormhole,blackhole}/hw_counters.h
//
// Shared single config buffer (all zones use the same counter selection).
// Per-zone data = 5 bank cycles (one OUT_L per bank) + N counter counts (OUT_H
// per counter, only N is bank-bound). OUT_L is bank-wide so dedup saves space.
#define PERF_COUNTERS_CONFIG_WORDS      200
#define PERF_COUNTERS_DATA_WORDS        200 // counter counts (OUT_H) per zone
#define PERF_COUNTERS_BANK_CYCLES_WORDS 5   // OUT_L per bank (INSTRN, FPU, TDMA_U, L1, TDMA_P)

namespace llk_perf
{

// 8 zones × 860 B + 800 B shared config = 7680 B; fits below profiler at 0x16AFF4.
constexpr std::uint32_t PERF_COUNTERS_MAX_ZONES = 8;
constexpr std::uint32_t SYNC_ZONE_COMPLETE      = 0xFFu;

// Per-zone block = bank cycles + counter counts + sync (+ pad to 40-byte tail
// so layout matches existing sync/stop-flags expectations).
constexpr std::uint32_t PERF_COUNTERS_ZONE_DATA_BYTES = (PERF_COUNTERS_BANK_CYCLES_WORDS + PERF_COUNTERS_DATA_WORDS) * 4;
constexpr std::uint32_t PERF_COUNTERS_ZONE_SIZE       = PERF_COUNTERS_ZONE_DATA_BYTES + 40;

// Shared config lives at base; per-zone blocks follow immediately after.
constexpr std::uint32_t PERF_COUNTERS_SHARED_CONFIG_ADDR = PERF_COUNTERS_BASE_ADDR;
constexpr std::uint32_t PERF_COUNTERS_ZONES_BASE         = PERF_COUNTERS_BASE_ADDR + PERF_COUNTERS_CONFIG_WORDS * 4;

constexpr std::uint32_t perf_counters_zone_data_addr(std::uint32_t zone)
{
    return PERF_COUNTERS_ZONES_BASE + zone * PERF_COUNTERS_ZONE_SIZE;
}

constexpr std::uint32_t perf_counters_zone_counts_addr(std::uint32_t zone)
{
    return perf_counters_zone_data_addr(zone) + PERF_COUNTERS_BANK_CYCLES_WORDS * 4;
}

constexpr std::uint32_t perf_counters_sync_ctrl_addr(std::uint32_t zone)
{
    return perf_counters_zone_data_addr(zone) + PERF_COUNTERS_ZONE_DATA_BYTES;
}

constexpr std::uint32_t perf_counters_stop_flags_addr(std::uint32_t zone)
{
    return perf_counters_sync_ctrl_addr(zone) + 4;
}

// Compile-time guard: all zones must fit between base addr and profiler region
// (profiler barrier starts at 0x16AFF4 on WH/BH).
static_assert(
    PERF_COUNTERS_ZONES_BASE + PERF_COUNTERS_MAX_ZONES * PERF_COUNTERS_ZONE_SIZE <= 0x16AFF4u,
    "Perf counter L1 layout overflows profiler region — reduce MAX_ZONES or DATA_WORDS");

constexpr std::uint32_t PERF_COUNTERS_ENABLED_FLAG_ADDR = PERF_COUNTERS_ZONES_BASE + PERF_COUNTERS_MAX_ZONES * PERF_COUNTERS_ZONE_SIZE;

} // namespace llk_perf

// ============================================================================
// TRISC functions — ALWAYS compiled (identical binary in NC and WC).
// Runtime L1 enabled flag controls behavior. BRISC sets flag=1 in WC builds.
// ============================================================================

namespace llk_perf
{

// Start: no-op. BRISC arms counters before releasing TRISCs.
__attribute__((always_inline)) inline void start_perf_counters(std::uint32_t zone)
{
    asm volatile("" : : "r"(zone) : "memory");
}

// Stop: no-op. BRISC observes zone boundaries by scanning each TRISC's profiler
// buffer (ZONE_END entries written by zone_scoped). TRISCs do zero extra work.
__attribute__((always_inline)) inline void stop_perf_counters(std::uint32_t zone)
{
    asm volatile("" : : "r"(zone) : "memory");
}

} // namespace llk_perf

// ============================================================================
// BRISC-only code — only compiled when PERF_COUNTERS_COMPILED is defined.
// NC BRISC uses stubs (below). WC BRISC gets the full PerfCounterManager.
// ============================================================================

#ifndef PERF_COUNTERS_COMPILED

// NC BRISC stubs
namespace llk_perf
{
__attribute__((noinline, section(".text.zzz_perf_counters"))) inline void configure_and_arm_from_brisc()
{
    asm volatile("" ::: "memory");
}

inline void monitor_zones_from_brisc()
{
}
} // namespace llk_perf

#else

// WC-only: PerfCounterManager, BRISC functions, and counter hardware access.
// L1 address constants, hooks, and TRISC start/stop are defined above (always compiled).

namespace llk_perf
{

// Shared config (single buffer for all zones) and per-zone data addresses.
// Config selection (which counters to read) is identical across zones, so we
// keep one shared config and only replicate the result buffers per zone.
constexpr std::uint32_t perf_counters_config_addr(std::uint32_t /*zone*/)
{
    return PERF_COUNTERS_SHARED_CONFIG_ADDR;
}

// Returns address of zone's data area (bank cycles first, then counter counts).
constexpr std::uint32_t perf_counters_data_addr(std::uint32_t zone)
{
    return perf_counters_zone_data_addr(zone);
}

#if defined(ARCH_QUASAR)
#define PERF_COUNTERS_THREAD_COUNT 4
#else
#define PERF_COUNTERS_THREAD_COUNT 3
#endif

constexpr std::uint32_t perf_counters_start_counter_addr(std::uint32_t zone)
{
    return perf_counters_sync_ctrl_addr(zone) + 4;
}

constexpr std::uint32_t perf_counters_stop_counter_addr(std::uint32_t zone)
{
    return perf_counters_start_counter_addr(zone) + (PERF_COUNTERS_THREAD_COUNT * 4);
}

constexpr std::uint32_t perf_counters_stop_elect_addr(std::uint32_t zone)
{
    return perf_counters_stop_counter_addr(zone) + (PERF_COUNTERS_THREAD_COUNT * 4);
}

constexpr std::uint32_t perf_counters_start_barrier_addr(std::uint32_t zone)
{
    return perf_counters_stop_elect_addr(zone) + 4;
}

constexpr std::uint32_t perf_counters_stop_barrier_addr(std::uint32_t zone)
{
    return perf_counters_start_barrier_addr(zone) + 4;
}

constexpr std::uint32_t PERF_COUNTERS_BANK_MASK_ADDR   = PERF_COUNTERS_ENABLED_FLAG_ADDR + 4;
constexpr std::uint32_t PERF_COUNTERS_VALID_COUNT_ADDR = PERF_COUNTERS_BANK_MASK_ADDR + 4;

// ============================================================================
// Sync Control Word Bit Layout
// ============================================================================

// Sync control word bit layout (layout differs for 3 vs 4 TRISCs):
// 3 TRISCs: Bits 0-2 start, 3-5 stop, 6 started, 7 stopped, 8-9 starter, 10-11 stopper
// 4 TRISCs: Bits 0-3 start, 4-7 stop, 8 started, 9 stopped, 10-11 starter, 12-13 stopper

// SYNC_ZONE_COMPLETE defined above (shared section)
constexpr std::uint32_t SYNC_START_MASK     = (1u << PERF_COUNTERS_THREAD_COUNT) - 1u;
constexpr std::uint32_t SYNC_STOP_BIT_SHIFT = PERF_COUNTERS_THREAD_COUNT;
constexpr std::uint32_t SYNC_STOP_MASK      = SYNC_START_MASK << SYNC_STOP_BIT_SHIFT;
constexpr std::uint32_t SYNC_STARTED_FLAG   = 1u << (2u * PERF_COUNTERS_THREAD_COUNT);
constexpr std::uint32_t SYNC_STOPPED_FLAG   = 1u << (2u * PERF_COUNTERS_THREAD_COUNT + 1u);
constexpr std::uint32_t SYNC_STARTER_SHIFT  = 2u * PERF_COUNTERS_THREAD_COUNT + 2u;
constexpr std::uint32_t SYNC_STARTER_MASK   = 0x3u << SYNC_STARTER_SHIFT;
constexpr std::uint32_t SYNC_STOPPER_SHIFT  = SYNC_STARTER_SHIFT + 2u;
constexpr std::uint32_t SYNC_STOPPER_MASK   = 0x3u << SYNC_STOPPER_SHIFT;

// ============================================================================
// ATINCGET Helpers
// ============================================================================

// Use architecture-specific ATINCGET macro shape
#if defined(ARCH_QUASAR)
#define PERF_COUNTERS_TTI_ATINCGET(WrapVal, Sel32b, DataRegIndex, AddrRegIndex) TTI_ATINCGET(WrapVal, Sel32b, DataRegIndex, AddrRegIndex)
#else
#define PERF_COUNTERS_TTI_ATINCGET(WrapVal, Sel32b, DataRegIndex, AddrRegIndex) TTI_ATINCGET(0, WrapVal, Sel32b, DataRegIndex, AddrRegIndex)
#endif

#ifndef PERF_COUNTERS_USE_ATINCGET
#define PERF_COUNTERS_USE_ATINCGET 0
#endif

constexpr std::uint32_t ATINCGET_WIDTH_32    = 31u; // IntWidth for 32-bit
constexpr std::uint32_t PERF_COUNTER_THREADS = PERF_COUNTERS_THREAD_COUNT;

// ============================================================================
// Counter Bank Enumeration
// ============================================================================

enum class counter_bank : std::uint8_t
{
    instrn_thread = 0,
    fpu           = 1,
    tdma_unpack   = 2,
    l1            = 3,
    tdma_pack     = 4,
};

constexpr std::uint32_t COUNTER_BANK_COUNT = 5;
constexpr std::uint32_t COUNTER_SLOT_COUNT = PERF_COUNTERS_CONFIG_WORDS;

// ============================================================================
// Helper Functions
// ============================================================================

// Hardware register access functions for performance counter control
namespace hw_access
{
// Write a 32-bit value to a hardware register address
inline void write_reg(std::uint32_t addr, std::uint32_t value)
{
    *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(addr) = value;
}

inline std::uint32_t read_reg(std::uint32_t addr)
{
    return *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(addr);
}

// Get the base configuration register address for a counter bank
// Used to configure and control counter operation (period, mode, start/stop)
// Use arithmetic to compute addresses instead of arrays or switch tables.
// Both arrays and switch statements generate .ldm_data lookup tables that
// shift global variable addresses, causing GP-offset differences between
// counter and non-counter builds → different run_kernel codegen.
//
// Counter bank register layout (base = 0xFFB12000):
//   INSTRN_THREAD: offset 0x000, FPU: 0x018, TDMA_UNPACK: 0x00C,
//   L1: 0x030, TDMA_PACK: 0x0F0
// Output registers: low at base+0x100+bank*8, high at base+0x104+bank*8
// (approximate — actual offsets are irregular, so we use if-else)
// Prevent GCC from generating .data lookup tables (CSWTCH) by using
// __attribute__((optimize("Os"))) which disables switch-to-table optimization.
// This keeps .ldm_data identical between counter and non-counter builds.
// Use volatile cast to prevent GCC from building CSWTCH lookup tables.
// GCC can't prove the volatile read returns a constant, so it emits branches.
inline std::uint32_t get_counter_base_addr(counter_bank bank)
{
    volatile auto b = static_cast<std::uint32_t>(bank);
    if (b == 0)
    {
        return RISCV_DEBUG_REG_PERF_CNT_INSTRN_THREAD0;
    }
    if (b == 1)
    {
        return RISCV_DEBUG_REG_PERF_CNT_FPU0;
    }
    if (b == 2)
    {
        return RISCV_DEBUG_REG_PERF_CNT_TDMA_UNPACK0;
    }
    if (b == 3)
    {
        return RISCV_DEBUG_REG_PERF_CNT_L1_0;
    }
    if (b == 4)
    {
        return RISCV_DEBUG_REG_PERF_CNT_TDMA_PACK0;
    }
    return 0u;
}

inline std::uint32_t get_counter_output_low_addr(counter_bank bank)
{
    volatile auto b = static_cast<std::uint32_t>(bank);
    if (b == 0)
    {
        return RISCV_DEBUG_REG_PERF_CNT_OUT_L_INSTRN_THREAD;
    }
    if (b == 1)
    {
        return RISCV_DEBUG_REG_PERF_CNT_OUT_L_FPU;
    }
    if (b == 2)
    {
        return RISCV_DEBUG_REG_PERF_CNT_OUT_L_TDMA_UNPACK;
    }
    if (b == 3)
    {
        return RISCV_DEBUG_REG_PERF_CNT_OUT_L_DBG_L1;
    }
    if (b == 4)
    {
        return RISCV_DEBUG_REG_PERF_CNT_OUT_L_TDMA_PACK;
    }
    return 0u;
}

inline std::uint32_t get_counter_output_high_addr(counter_bank bank)
{
    volatile auto b = static_cast<std::uint32_t>(bank);
    if (b == 0)
    {
        return RISCV_DEBUG_REG_PERF_CNT_OUT_H_INSTRN_THREAD;
    }
    if (b == 1)
    {
        return RISCV_DEBUG_REG_PERF_CNT_OUT_H_FPU;
    }
    if (b == 2)
    {
        return RISCV_DEBUG_REG_PERF_CNT_OUT_H_TDMA_UNPACK;
    }
    if (b == 3)
    {
        return RISCV_DEBUG_REG_PERF_CNT_OUT_H_DBG_L1;
    }
    if (b == 4)
    {
        return RISCV_DEBUG_REG_PERF_CNT_OUT_H_TDMA_PACK;
    }
    return 0u;
}
} // namespace hw_access

// Thread identification and sync bit helpers
// Each TRISC thread (Unpack/Math/Pack) needs to identify itself for synchronization
namespace thread_info
{
// Get the current thread ID based on compile-time defines
// Returns: 0 (UNPACK), 1 (MATH), 2 (PACK), 3 (SFPU on Quasar only)
constexpr std::uint32_t get_thread_id()
{
#if defined(LLK_TRISC_UNPACK)
    return 0u;
#elif defined(LLK_TRISC_MATH)
    return 1u;
#elif defined(LLK_TRISC_PACK)
    return 2u;
#elif defined(LLK_TRISC_ISOLATE_SFPU)
    return 3u;
#else
    return 0u; // BRISC or unknown — default to thread 0
#endif
}

// Get the bit mask for this thread's start flag in sync control word
// Returns: bit 0 (UNPACK), bit 1 (MATH), bit 2 (PACK), or bit 3 (SFPU)
constexpr std::uint32_t get_thread_start_bit()
{
    return 1u << get_thread_id();
}

// Get the bit mask for this thread's stop flag in sync control word
constexpr std::uint32_t get_thread_stop_bit()
{
    return get_thread_start_bit() << SYNC_STOP_BIT_SHIFT;
}
} // namespace thread_info

// ============================================================================
// Performance Counter Manager (Singleton)
// ============================================================================

// Stateless counter manager — all state lives in L1 at fixed addresses.
// No C++ member variables → no .ldm_data footprint → no GP offset shift.
class PerfCounterManager
{
private:
    PerfCounterManager() = default;

    // Read metadata directly from L1 (written by BRISC in configure_all_zones).
    static std::uint32_t get_active_bank_mask()
    {
        return *reinterpret_cast<volatile std::uint32_t*>(PERF_COUNTERS_BANK_MASK_ADDR);
    }

    static std::uint32_t get_valid_count(std::uint32_t zone)
    {
        return reinterpret_cast<volatile std::uint32_t*>(PERF_COUNTERS_VALID_COUNT_ADDR)[zone];
    }

    // Get pointer to L1 config buffer (86 words of counter metadata)
    const volatile std::uint32_t* get_config_mem(std::uint32_t zone)
    {
        return reinterpret_cast<volatile std::uint32_t*>(perf_counters_config_addr(zone));
    }

    // Get pointer to L1 data buffer (172 words: cycles + count per counter)
    volatile std::uint32_t* get_data_mem(std::uint32_t zone)
    {
        return reinterpret_cast<volatile std::uint32_t*>(perf_counters_data_addr(zone));
    }

    static bool is_enabled()
    {
        return *reinterpret_cast<volatile std::uint32_t*>(PERF_COUNTERS_ENABLED_FLAG_ADDR) != 0u;
    }

    void compute_metadata()
    {
        ckernel::invalidate_data_cache();
    }

    // Get pointer to sync control word (thread coordination flags)
    volatile std::uint32_t* get_sync_ctrl_mem(std::uint32_t zone)
    {
        return reinterpret_cast<volatile std::uint32_t*>(perf_counters_sync_ctrl_addr(zone));
    }

    // Read an L1 word with a cache invalidation to improve visibility across threads.
    inline std::uint32_t read_l1_word(volatile std::uint32_t* addr)
    {
        ckernel::invalidate_data_cache();
        return *addr;
    }

    // Issue ATINCGET in L1 and return the original value.
    // Uses regfile indices reserved for perf counters, and restores them afterward.
    inline std::uint32_t atincget_l1(std::uint32_t addr, std::uint32_t increment)
    {
        constexpr std::uint32_t kDataReg = ckernel::p_gpr::DBG_RESERVED;
        constexpr std::uint32_t kAddrReg = ckernel::p_gpr::DBG_MSG;

        const std::uint32_t base16 = addr & ~0xFu;
        const std::uint32_t sel32b = (addr >> 2) & 0x3u;

        // No save/restore: DBG_RESERVED and DBG_MSG are reserved for perf counters.
        // Skipping save/restore saves ~8 cycles per atincget call.

        // Store to GPRs with explicit ordering (sw -> lw -> addi)
        volatile std::uint32_t* data_ptr = &ckernel::regfile[kDataReg];
        volatile std::uint32_t* addr_ptr = &ckernel::regfile[kAddrReg];
        std::uint32_t tmp;
        const std::uint32_t inc_val  = increment;
        const std::uint32_t addr_val = base16 >> 4;
        asm volatile(
            "sw %2, 0(%1)\n"
            "lw %0, 0(%1)\n"
            "addi x0, %0, 0\n"
            : "=&r"(tmp)
            : "r"(data_ptr), "r"(inc_val)
            : "memory");
        asm volatile(
            "sw %2, 0(%1)\n"
            "lw %0, 0(%1)\n"
            "addi x0, %0, 0\n"
            : "=&r"(tmp)
            : "r"(addr_ptr), "r"(addr_val)
            : "memory");

        PERF_COUNTERS_TTI_ATINCGET(ATINCGET_WIDTH_32, sel32b, kDataReg, kAddrReg);

        // Wait for ATINCGET result by spinning on the GPR (RISC-V loads only, no coprocessor FIFO pollution).
        // The ATINCGET overwrites kDataReg with the old L1 value. We detect completion by
        // reading until the value differs from the increment we stored, or a timeout.
        std::uint32_t old_value;
        for (std::uint32_t i = 0; i < 1000; ++i)
        {
            old_value = ckernel::regfile[kDataReg];
            if (old_value != inc_val)
            {
                break;
            }
            // Pure RISC-V delay — no coprocessor instructions
            asm volatile("nop");
        }

        return old_value;
    }

    // Configure banks for a zone: L1 MUX, reference period, mode register.
    void configure_hardware(std::uint32_t zone)
    {
        const volatile std::uint32_t* config_mem = get_config_mem(zone);
        std::uint32_t configured_mask            = 0;

        for (std::uint32_t i = 0; i < COUNTER_SLOT_COUNT; i++)
        {
            const std::uint32_t metadata = config_mem[i];
            if ((metadata & 0x80000000u) == 0)
            {
                continue;
            }

            const std::uint8_t bank_id   = static_cast<std::uint8_t>(metadata);
            const std::uint32_t bank_bit = 1u << bank_id;

            if (configured_mask & bank_bit)
            {
                continue;
            }

            const counter_bank bank = static_cast<counter_bank>(bank_id);

            if (bank == counter_bank::l1)
            {
                // L1 mux: WH = 1-bit (bit 4), BH = 3-bit (bits 6:4). Config-word
                // bits 19:17 encode mux (3 bits) — upper 2 unused on WH.
                const std::uint8_t l1_mux = (metadata >> 17) & 0x7;
                std::uint32_t cur         = hw_access::read_reg(RISCV_DEBUG_REG_PERF_CNT_MUX_CTRL);
                hw_access::write_reg(RISCV_DEBUG_REG_PERF_CNT_MUX_CTRL, (cur & ~(0x7u << 4)) | ((l1_mux & 0x7u) << 4));
            }

            std::uint32_t counter_base = hw_access::get_counter_base_addr(bank);
            hw_access::write_reg(counter_base, 0xFFFFFFFF); // Reference period (used by mode 1 only)
            hw_access::write_reg(counter_base + 4, 0);      // mode=0 continuous

            configured_mask |= bank_bit;
        }
    }

    // Arm only banks that have configured counters: clear + start.
    void arm_hardware()
    {
        // Working tt-metal sequence: write 1 then write 0.
        // Rising edge 0→1 on start bit clears counters and starts counting.
        // Writing 0 after resets control to idle (so subsequent 0→2 stop is clean edge).
        for (std::uint32_t b = 0; b < COUNTER_BANK_COUNT; ++b)
        {
            if (!(get_active_bank_mask() & (1u << b)))
            {
                continue;
            }
            std::uint32_t counter_base = hw_access::get_counter_base_addr(static_cast<counter_bank>(b));
            hw_access::write_reg(counter_base + 8, 1); // Start (rising edge clears + starts)
            hw_access::write_reg(counter_base + 8, 0); // Reset to idle
        }
        // PERF_CNT_ALL: broadcast start to INSTRN_THREAD and FPU banks.
        // These banks may require global arm separate from per-bank control.
        hw_access::write_reg(RISCV_DEBUG_REG_PERF_CNT_ALL, 1);
        hw_access::write_reg(RISCV_DEBUG_REG_PERF_CNT_ALL, 0);
    }

    void freeze_hardware()
    {
        // Working tt-metal sequence: write 2 then write 0.
        // Rising edge 0→2 on stop bit freezes counters.
        // PERF_CNT_ALL: broadcast stop to INSTRN_THREAD and FPU banks.
        hw_access::write_reg(RISCV_DEBUG_REG_PERF_CNT_ALL, 2);
        hw_access::write_reg(RISCV_DEBUG_REG_PERF_CNT_ALL, 0);
        for (std::uint32_t b = 0; b < COUNTER_BANK_COUNT; ++b)
        {
            if (!(get_active_bank_mask() & (1u << b)))
            {
                continue;
            }
            std::uint32_t counter_base = hw_access::get_counter_base_addr(static_cast<counter_bank>(b));
            hw_access::write_reg(counter_base + 8, 2); // Stop (rising edge freezes)
            hw_access::write_reg(counter_base + 8, 0); // Reset to idle
        }
    }

    // Initialize and start hardware counters (called by first thread only)
    void start_hardware(std::uint32_t zone)
    {
        configure_hardware(zone);
        arm_hardware();
    }

    // Reset counter banks to unconfigured state so hardware has zero monitoring
    // overhead on subsequent execution (especially FPU pipeline → MATH_ISOLATE).
    void deconfigure_hardware()
    {
        for (std::uint32_t b = 0; b < COUNTER_BANK_COUNT; ++b)
        {
            if (!(get_active_bank_mask() & (1u << b)))
            {
                continue;
            }
            std::uint32_t counter_base = hw_access::get_counter_base_addr(static_cast<counter_bank>(b));
            hw_access::write_reg(counter_base, 0);     // Clear reference period
            hw_access::write_reg(counter_base + 4, 0); // Clear mode register
            hw_access::write_reg(counter_base + 8, 0); // Clear control (stop + disarm)
        }
    }

    // Freeze only banks that have configured counters.
    // After this, all counter values are frozen and can be read at leisure.
    // Read frozen counter values to L1 for a zone.
    // Skips zones with no configured counters (e.g. zone 1/2 when only zone 0 has config).
    // Stops scanning config slots as soon as all valid counters are processed.
    // Must be called after freeze_hardware().
    void read_hardware(std::uint32_t zone)
    {
        const std::uint32_t count = get_valid_count(zone);
        if (count == 0)
        {
            return;
        }

        const volatile std::uint32_t* config_mem = get_config_mem(zone);
        volatile std::uint32_t* data_mem         = get_data_mem(zone);

        std::uint32_t result_idx = 0;

        for (std::uint32_t i = 0; i < COUNTER_SLOT_COUNT && result_idx < count; i++)
        {
            const std::uint32_t metadata = config_mem[i];
            if ((metadata & 0x80000000u) == 0)
            {
                continue;
            }

            const std::uint8_t bank_id     = static_cast<std::uint8_t>(metadata);
            const std::uint16_t counter_id = (metadata >> 8) & 0x1FF;
            // L1 mux: 3 bits (bits 19:17). WH uses 1 bit, BH uses 3 (banks 0..4).
            const std::uint8_t l1_mux = (metadata >> 17) & 0x7;

            const counter_bank bank = static_cast<counter_bank>(bank_id);

            // Configure L1 MUX before reading. Use 3-bit mask so BH mux 2..4
            // bits get properly cleared/set (WH ignores bits 5..6).
            if (bank == counter_bank::l1)
            {
                std::uint32_t cur = hw_access::read_reg(RISCV_DEBUG_REG_PERF_CNT_MUX_CTRL);
                hw_access::write_reg(RISCV_DEBUG_REG_PERF_CNT_MUX_CTRL, (cur & ~(0x7u << 4)) | ((l1_mux & 0x7u) << 4));
            }

            std::uint32_t counter_base = hw_access::get_counter_base_addr(bank);
            hw_access::write_reg(counter_base + 4, static_cast<std::uint32_t>(counter_id) << 8);

            // Single dummy read for mux settling
            std::uint32_t output_low_addr  = hw_access::get_counter_output_low_addr(bank);
            std::uint32_t output_high_addr = hw_access::get_counter_output_high_addr(bank);
            (void)hw_access::read_reg(output_low_addr);

            // Write to L1
            data_mem[result_idx * 2]     = hw_access::read_reg(output_low_addr);
            data_mem[result_idx * 2 + 1] = hw_access::read_reg(output_high_addr);

            result_idx++;
        }
    }

public:
    // Public wrappers for BRISC access
    void arm()
    {
        arm_hardware();
    }

    void read_counters(std::uint32_t zone)
    {
        read_hardware(zone);
    }

    bool is_metadata_ready() const
    {
        return get_active_bank_mask() != 0;
    }

    void init_metadata()
    {
        compute_metadata();
    }

    // Write only L1 metadata (bank mask, valid counts, enabled flag) without
    // touching hardware registers. Called by BRISC — hardware configure+arm
    // is deferred to TRISC start_perf_counters() to avoid counter hw overhead.
    void configure_all_zones_metadata_only()
    {
        bool found_valid                                    = false;
        std::uint32_t bank_mask                             = 0;
        std::uint32_t valid_counts[PERF_COUNTERS_MAX_ZONES] = {};

        for (std::uint32_t zone = 0; zone < PERF_COUNTERS_MAX_ZONES; ++zone)
        {
            const volatile std::uint32_t* config_mem = get_config_mem(zone);
            std::uint32_t count                      = 0;
            for (std::uint32_t i = 0; i < COUNTER_SLOT_COUNT; i++)
            {
                const std::uint32_t metadata = config_mem[i];
                if (metadata & 0x80000000u)
                {
                    found_valid = true;
                    count++;
                    bank_mask |= (1u << (metadata & 0xFFu));
                }
            }
            valid_counts[zone] = count;
        }

        // Write pre-computed metadata to shared L1 for TRISCs to read.
        volatile std::uint32_t* enabled_flag = reinterpret_cast<volatile std::uint32_t*>(PERF_COUNTERS_ENABLED_FLAG_ADDR);
        *enabled_flag                        = found_valid ? 1u : 0u;

        volatile std::uint32_t* bank_mask_ptr = reinterpret_cast<volatile std::uint32_t*>(PERF_COUNTERS_BANK_MASK_ADDR);
        *bank_mask_ptr                        = bank_mask;

        volatile std::uint32_t* valid_count_ptr = reinterpret_cast<volatile std::uint32_t*>(PERF_COUNTERS_VALID_COUNT_ADDR);
        for (std::uint32_t zone = 0; zone < PERF_COUNTERS_MAX_ZONES; ++zone)
        {
            valid_count_ptr[zone] = valid_counts[zone];
        }
    }

    // Pre-configure all zones. Called by BRISC before releasing TRISCs.
    // Also pre-computes bank mask and valid counts so TRISCs skip compute_metadata().
    void configure_all_zones()
    {
        bool found_valid                                    = false;
        std::uint32_t bank_mask                             = 0;
        std::uint32_t valid_counts[PERF_COUNTERS_MAX_ZONES] = {};

        for (std::uint32_t zone = 0; zone < PERF_COUNTERS_MAX_ZONES; ++zone)
        {
            const volatile std::uint32_t* config_mem = get_config_mem(zone);
            std::uint32_t count                      = 0;
            for (std::uint32_t i = 0; i < COUNTER_SLOT_COUNT; i++)
            {
                const std::uint32_t metadata = config_mem[i];
                if (metadata & 0x80000000u)
                {
                    found_valid = true;
                    count++;
                    bank_mask |= (1u << (metadata & 0xFFu));
                }
            }
            valid_counts[zone] = count;
        }

        // Write metadata to L1 FIRST — configure_hardware + arm_hardware read
        // bank_mask from L1 via get_active_bank_mask(). Must be set before they run.
        volatile std::uint32_t* enabled_flag = reinterpret_cast<volatile std::uint32_t*>(PERF_COUNTERS_ENABLED_FLAG_ADDR);
        *enabled_flag                        = found_valid ? 1u : 0u;

        volatile std::uint32_t* bank_mask_ptr = reinterpret_cast<volatile std::uint32_t*>(PERF_COUNTERS_BANK_MASK_ADDR);
        *bank_mask_ptr                        = bank_mask;

        volatile std::uint32_t* valid_count_ptr = reinterpret_cast<volatile std::uint32_t*>(PERF_COUNTERS_VALID_COUNT_ADDR);
        for (std::uint32_t zone = 0; zone < PERF_COUNTERS_MAX_ZONES; ++zone)
        {
            valid_count_ptr[zone] = valid_counts[zone];
        }

        if (found_valid)
        {
            hw_access::write_reg(RISCV_DEBUG_REG_DBG_FEATURE_DISABLE, 0);
            configure_hardware(0);
            arm_hardware();
        }
    }

    // Get singleton instance (Meyer's singleton pattern)
    static PerfCounterManager& instance()
    {
        static PerfCounterManager instance;
        return instance;
    }

    // Delete copy/move constructors and assignment operators
    PerfCounterManager(const PerfCounterManager&)            = delete;
    PerfCounterManager& operator=(const PerfCounterManager&) = delete;
    PerfCounterManager(PerfCounterManager&&)                 = delete;
    PerfCounterManager& operator=(PerfCounterManager&&)      = delete;

    // Per-zone start: BRISC handles all CSR operations.
    // TRISCs only signal zone boundaries via L1 flags.
    // For zone > 0, all TRISCs wait for BRISC to finish counter ops + re-arm.
    void start(std::uint32_t zone)
    {
        // Delay + ready check moved to _profiler_counter_start hook
        // for icache parity with NC (both delays in same section).
        asm volatile("" : : "r"(zone) : "memory");
    }

    void stop(std::uint32_t zone)
    {
        // Each thread signals its arrival via per-thread flag.
        // BRISC waits until ALL threads have signaled before freezing counters.
        // All threads write (no branch divergence) — different addresses per thread.
        volatile std::uint32_t* stop_flags       = reinterpret_cast<volatile std::uint32_t*>(perf_counters_stop_flags_addr(zone));
        stop_flags[thread_info::get_thread_id()] = 1;
    }

    // Freeze + read + deconfigure for BRISC use (last zone, after TRISCs complete).
    void freeze_and_read(std::uint32_t zone)
    {
        if (get_active_bank_mask() == 0 || get_valid_count(zone) == 0)
        {
            return;
        }
        freeze_hardware();
        read_hardware(zone);
        deconfigure_hardware();

        volatile std::uint32_t* sync_ctrl = get_sync_ctrl_mem(zone);
        *sync_ctrl                        = SYNC_ZONE_COMPLETE;
    }

    // Freeze + read + re-arm for BRISC use (between zones, counters restart for next zone).
    void freeze_read_and_rearm(std::uint32_t zone)
    {
        if (get_active_bank_mask() == 0 || get_valid_count(zone) == 0)
        {
            return;
        }
        // Between zones: PERF_CNT_ALL broadcast freeze+arm (3 CSR writes).
        // This re-arms FPU + INSTRN_THREAD for the next zone (per-zone separation).
        // TDMA/L1 banks are not re-armed — their zone 1 counts are cumulative
        // (INIT + TILE_LOOP) which is acceptable given zero-overhead goal.
        hw_access::write_reg(RISCV_DEBUG_REG_PERF_CNT_ALL, 2); // stop
        hw_access::write_reg(RISCV_DEBUG_REG_PERF_CNT_ALL, 1); // start (rising edge clears + starts)
        hw_access::write_reg(RISCV_DEBUG_REG_PERF_CNT_ALL, 0); // reset to idle

        // Mark zone complete so Python can find the data
        volatile std::uint32_t* sync_ctrl = get_sync_ctrl_mem(zone);
        *sync_ctrl                        = SYNC_ZONE_COMPLETE;
    }
};

// ============================================================================
// Public API
// ============================================================================

// start_perf_counters and stop_perf_counters are defined above (always compiled).
// They check PERF_COUNTERS_ENABLED_FLAG_ADDR at runtime.

inline void init_perf_counter_metadata()
{
    auto& mgr = PerfCounterManager::instance();
    if (!mgr.is_metadata_ready())
    {
        mgr.init_metadata();
    }
}

// Pre-configure all counter banks (called by BRISC before TRISCs start)
inline void configure_perf_counters_from_brisc()
{
    PerfCounterManager::instance().configure_all_zones();
}

// Configure + arm from BRISC (before releasing TRISCs).
// TRISCs have zero counter code in run_kernel.
// Built-in counter config: arch-specific list of HW counters that BRISC writes
// to L1 (local write, no NOC) instead of Python — avoids ~7 cyc Float16 unpack
// overhead from L1 controller state changes.
//
// Source of truth: tt_metal/hw/inc/internal/tt-1xx/{wormhole,blackhole}/hw_counters.h
// Same per-arch dispatch pattern as tt_metal/tools/profiler/perf_counters.hpp.
//
// Config word format: valid(31) | l1_mux<<17 (3 bits) | counter_id<<8 (9 bits) | bank_id (8 bits)
//   bank_id: 0=INSTRN_THREAD, 1=FPU, 2=TDMA_UNPACK, 3=L1, 4=TDMA_PACK
//   l1_mux: 0..1 on WH, 0..4 on BH (L1 only; ignored for other banks)
constexpr std::uint32_t _perf_cfg(std::uint8_t bank, std::uint16_t cid, std::uint8_t mux = 0)
{
    return 0x80000000u | (static_cast<std::uint32_t>(mux & 0x7u) << 17) | (static_cast<std::uint32_t>(cid & 0x1FFu) << 8) | static_cast<std::uint32_t>(bank);
}

// clang-format off
#if defined(ARCH_BLACKHOLE)
// BH = 169 counters: 59 INSTRN + 3 FPU + 22 TDMA_UNPACK + 5 TDMA_PACK + 80 L1
//                    (16 each × 5 L1 mux banks 0..4 — BH-specific)
constexpr std::uint32_t BUILTIN_COUNTER_CONFIG[] = {
    // INSTRN_THREAD (bank 0) — 59 entries, contiguous IDs except gaps 9..11.
    _perf_cfg(0,   0), _perf_cfg(0,   1), _perf_cfg(0,   2), _perf_cfg(0,   3), _perf_cfg(0,   4), _perf_cfg(0,   5),
    _perf_cfg(0,   6), _perf_cfg(0,   7), _perf_cfg(0,   8), _perf_cfg(0,  12), _perf_cfg(0,  13), _perf_cfg(0,  14),
    _perf_cfg(0,  15), _perf_cfg(0,  16), _perf_cfg(0,  17), _perf_cfg(0,  18), _perf_cfg(0,  19), _perf_cfg(0,  20),
    _perf_cfg(0,  21), _perf_cfg(0,  22), _perf_cfg(0,  23), _perf_cfg(0,  24), _perf_cfg(0,  25), _perf_cfg(0,  26),
    _perf_cfg(0,  27), _perf_cfg(0,  28), _perf_cfg(0,  29), _perf_cfg(0,  30), _perf_cfg(0,  31), _perf_cfg(0,  32),
    _perf_cfg(0,  33), _perf_cfg(0,  34), _perf_cfg(0,  35), _perf_cfg(0,  36), _perf_cfg(0,  37), _perf_cfg(0,  38),
    _perf_cfg(0,  39), _perf_cfg(0,  40), _perf_cfg(0,  41), _perf_cfg(0,  42), _perf_cfg(0,  43), _perf_cfg(0,  44),
    _perf_cfg(0,  45), _perf_cfg(0,  46), _perf_cfg(0,  47), _perf_cfg(0,  48), _perf_cfg(0,  49), _perf_cfg(0,  50),
    _perf_cfg(0,  51), _perf_cfg(0,  52), _perf_cfg(0,  53), _perf_cfg(0,  54), _perf_cfg(0,  55), _perf_cfg(0,  56),
    _perf_cfg(0,  57), _perf_cfg(0, 256), _perf_cfg(0, 264), _perf_cfg(0, 272), _perf_cfg(0, 283),
    // FPU (bank 1) — 3 entries
    _perf_cfg(1, 0), _perf_cfg(1, 1), _perf_cfg(1, 257),
    // TDMA_UNPACK (bank 2) — 22 entries
    _perf_cfg(2,   0), _perf_cfg(2,   1), _perf_cfg(2,   2), _perf_cfg(2,   3), _perf_cfg(2,   4), _perf_cfg(2,   5),
    _perf_cfg(2,   6), _perf_cfg(2,   7), _perf_cfg(2,   8), _perf_cfg(2,   9), _perf_cfg(2,  10),
    _perf_cfg(2, 256), _perf_cfg(2, 257), _perf_cfg(2, 258), _perf_cfg(2, 259), _perf_cfg(2, 260), _perf_cfg(2, 261),
    _perf_cfg(2, 262), _perf_cfg(2, 263), _perf_cfg(2, 264), _perf_cfg(2, 265), _perf_cfg(2, 266),
    // TDMA_PACK (bank 4) — 5 entries (BH has only 1 packer engine)
    _perf_cfg(4, 11), _perf_cfg(4, 18), _perf_cfg(4, 267), _perf_cfg(4, 271), _perf_cfg(4, 272),
    // L1 (bank 3) mux 0 — 16 entries: unpacker, TDMA bundles, NOC Ring 0
    _perf_cfg(3, 0, 0), _perf_cfg(3, 1, 0), _perf_cfg(3, 2, 0), _perf_cfg(3, 3, 0),
    _perf_cfg(3, 4, 0), _perf_cfg(3, 5, 0), _perf_cfg(3, 6, 0), _perf_cfg(3, 7, 0),
    _perf_cfg(3, 256, 0), _perf_cfg(3, 257, 0), _perf_cfg(3, 258, 0), _perf_cfg(3, 259, 0),
    _perf_cfg(3, 260, 0), _perf_cfg(3, 261, 0), _perf_cfg(3, 262, 0), _perf_cfg(3, 263, 0),
    // L1 mux 1 — 16: RISC core, ext unpacker, NOC Ring 1
    _perf_cfg(3, 0, 1), _perf_cfg(3, 1, 1), _perf_cfg(3, 2, 1), _perf_cfg(3, 3, 1),
    _perf_cfg(3, 4, 1), _perf_cfg(3, 5, 1), _perf_cfg(3, 6, 1), _perf_cfg(3, 7, 1),
    _perf_cfg(3, 256, 1), _perf_cfg(3, 257, 1), _perf_cfg(3, 258, 1), _perf_cfg(3, 259, 1),
    _perf_cfg(3, 260, 1), _perf_cfg(3, 261, 1), _perf_cfg(3, 262, 1), _perf_cfg(3, 263, 1),
    // L1 mux 2 — 16: NOC Ring 2 ports (BH-only)
    _perf_cfg(3, 0, 2), _perf_cfg(3, 1, 2), _perf_cfg(3, 2, 2), _perf_cfg(3, 3, 2),
    _perf_cfg(3, 4, 2), _perf_cfg(3, 5, 2), _perf_cfg(3, 6, 2), _perf_cfg(3, 7, 2),
    _perf_cfg(3, 256, 2), _perf_cfg(3, 257, 2), _perf_cfg(3, 258, 2), _perf_cfg(3, 259, 2),
    _perf_cfg(3, 260, 2), _perf_cfg(3, 261, 2), _perf_cfg(3, 262, 2), _perf_cfg(3, 263, 2),
    // L1 mux 3 — 16: NOC Ring 3 ports (BH-only)
    _perf_cfg(3, 0, 3), _perf_cfg(3, 1, 3), _perf_cfg(3, 2, 3), _perf_cfg(3, 3, 3),
    _perf_cfg(3, 4, 3), _perf_cfg(3, 5, 3), _perf_cfg(3, 6, 3), _perf_cfg(3, 7, 3),
    _perf_cfg(3, 256, 3), _perf_cfg(3, 257, 3), _perf_cfg(3, 258, 3), _perf_cfg(3, 259, 3),
    _perf_cfg(3, 260, 3), _perf_cfg(3, 261, 3), _perf_cfg(3, 262, 3), _perf_cfg(3, 263, 3),
    // L1 mux 4 — 16: misc ports (BH-only)
    _perf_cfg(3, 0, 4), _perf_cfg(3, 1, 4), _perf_cfg(3, 2, 4), _perf_cfg(3, 3, 4),
    _perf_cfg(3, 4, 4), _perf_cfg(3, 5, 4), _perf_cfg(3, 6, 4), _perf_cfg(3, 7, 4),
    _perf_cfg(3, 256, 4), _perf_cfg(3, 257, 4), _perf_cfg(3, 258, 4), _perf_cfg(3, 259, 4),
    _perf_cfg(3, 260, 4), _perf_cfg(3, 261, 4), _perf_cfg(3, 262, 4), _perf_cfg(3, 263, 4),
};
#else
// WH = 130 counters: 59 INSTRN + 3 FPU + 22 TDMA_UNPACK + 14 TDMA_PACK + 32 L1
//                    (2 L1 mux banks). INSTRN uses gap IDs 27/30/33/36 and 39..65.
constexpr std::uint32_t BUILTIN_COUNTER_CONFIG[] = {
    // INSTRN_THREAD (bank 0) — 59 entries with gap pattern.
    _perf_cfg(0,   0), _perf_cfg(0,   1), _perf_cfg(0,   2), _perf_cfg(0,   3), _perf_cfg(0,   4), _perf_cfg(0,   5),
    _perf_cfg(0,   6), _perf_cfg(0,   7), _perf_cfg(0,   8), _perf_cfg(0,  12), _perf_cfg(0,  13), _perf_cfg(0,  14),
    _perf_cfg(0,  15), _perf_cfg(0,  16), _perf_cfg(0,  17), _perf_cfg(0,  18), _perf_cfg(0,  19), _perf_cfg(0,  20),
    _perf_cfg(0,  21), _perf_cfg(0,  22), _perf_cfg(0,  23), _perf_cfg(0,  24), _perf_cfg(0,  25), _perf_cfg(0,  26),
    _perf_cfg(0,  27), _perf_cfg(0,  30), _perf_cfg(0,  33), _perf_cfg(0,  36), _perf_cfg(0,  39), _perf_cfg(0,  40),
    _perf_cfg(0,  41), _perf_cfg(0,  42), _perf_cfg(0,  43), _perf_cfg(0,  44), _perf_cfg(0,  45), _perf_cfg(0,  46),
    _perf_cfg(0,  47), _perf_cfg(0,  48), _perf_cfg(0,  49), _perf_cfg(0,  50), _perf_cfg(0,  51), _perf_cfg(0,  52),
    _perf_cfg(0,  53), _perf_cfg(0,  54), _perf_cfg(0,  55), _perf_cfg(0,  56), _perf_cfg(0,  57), _perf_cfg(0,  58),
    _perf_cfg(0,  59), _perf_cfg(0,  60), _perf_cfg(0,  61), _perf_cfg(0,  62), _perf_cfg(0,  63), _perf_cfg(0,  64),
    _perf_cfg(0,  65), _perf_cfg(0, 256), _perf_cfg(0, 264), _perf_cfg(0, 272), _perf_cfg(0, 283),
    // FPU (bank 1) — 3 entries
    _perf_cfg(1, 0), _perf_cfg(1, 1), _perf_cfg(1, 257),
    // TDMA_UNPACK (bank 2) — 22 entries
    _perf_cfg(2,   0), _perf_cfg(2,   1), _perf_cfg(2,   2), _perf_cfg(2,   3), _perf_cfg(2,   4), _perf_cfg(2,   5),
    _perf_cfg(2,   6), _perf_cfg(2,   7), _perf_cfg(2,   8), _perf_cfg(2,   9), _perf_cfg(2,  10),
    _perf_cfg(2, 256), _perf_cfg(2, 257), _perf_cfg(2, 258), _perf_cfg(2, 259), _perf_cfg(2, 260), _perf_cfg(2, 261),
    _perf_cfg(2, 262), _perf_cfg(2, 263), _perf_cfg(2, 264), _perf_cfg(2, 265), _perf_cfg(2, 266),
    // TDMA_PACK (bank 4) — 14 entries (WH has 4 packer engines)
    _perf_cfg(4, 11), _perf_cfg(4, 12), _perf_cfg(4, 13), _perf_cfg(4, 14), _perf_cfg(4, 15), _perf_cfg(4, 16),
    _perf_cfg(4, 17), _perf_cfg(4, 18),
    _perf_cfg(4, 267), _perf_cfg(4, 268), _perf_cfg(4, 269), _perf_cfg(4, 270), _perf_cfg(4, 271), _perf_cfg(4, 272),
    // L1 (bank 3) mux 0 — 16
    _perf_cfg(3, 0, 0), _perf_cfg(3, 1, 0), _perf_cfg(3, 2, 0), _perf_cfg(3, 3, 0),
    _perf_cfg(3, 4, 0), _perf_cfg(3, 5, 0), _perf_cfg(3, 6, 0), _perf_cfg(3, 7, 0),
    _perf_cfg(3, 256, 0), _perf_cfg(3, 257, 0), _perf_cfg(3, 258, 0), _perf_cfg(3, 259, 0),
    _perf_cfg(3, 260, 0), _perf_cfg(3, 261, 0), _perf_cfg(3, 262, 0), _perf_cfg(3, 263, 0),
    // L1 mux 1 — 16
    _perf_cfg(3, 0, 1), _perf_cfg(3, 1, 1), _perf_cfg(3, 2, 1), _perf_cfg(3, 3, 1),
    _perf_cfg(3, 4, 1), _perf_cfg(3, 5, 1), _perf_cfg(3, 6, 1), _perf_cfg(3, 7, 1),
    _perf_cfg(3, 256, 1), _perf_cfg(3, 257, 1), _perf_cfg(3, 258, 1), _perf_cfg(3, 259, 1),
    _perf_cfg(3, 260, 1), _perf_cfg(3, 261, 1), _perf_cfg(3, 262, 1), _perf_cfg(3, 263, 1),
};
#endif
constexpr std::uint32_t BUILTIN_COUNTER_COUNT = sizeof(BUILTIN_COUNTER_CONFIG) / sizeof(BUILTIN_COUNTER_CONFIG[0]);
// clang-format on

inline void configure_and_arm_from_brisc()
{
    // Shared config (single buffer for all zones): write BUILTIN_COUNTER_CONFIG
    // once, pad remaining slots with 0. Per-zone L1 holds only result buffers.
    volatile std::uint32_t* shared_config = reinterpret_cast<volatile std::uint32_t*>(PERF_COUNTERS_SHARED_CONFIG_ADDR);
    for (std::uint32_t i = 0; i < BUILTIN_COUNTER_COUNT; i++)
    {
        shared_config[i] = BUILTIN_COUNTER_CONFIG[i];
    }
    for (std::uint32_t i = BUILTIN_COUNTER_COUNT; i < COUNTER_SLOT_COUNT; i++)
    {
        shared_config[i] = 0;
    }

    // Clear per-zone data and sync regions so leftover state from previous run
    // can't be mistaken for a fresh measurement.
    for (std::uint32_t zone = 0; zone < PERF_COUNTERS_MAX_ZONES; ++zone)
    {
        volatile std::uint32_t* data_mem = reinterpret_cast<volatile std::uint32_t*>(perf_counters_zone_data_addr(zone));
        for (std::uint32_t i = 0; i < PERF_COUNTERS_BANK_CYCLES_WORDS + PERF_COUNTERS_DATA_WORDS; i++)
        {
            data_mem[i] = 0;
        }
        volatile std::uint32_t* sync_mem = reinterpret_cast<volatile std::uint32_t*>(perf_counters_sync_ctrl_addr(zone));
        for (std::uint32_t i = 0; i < 10; i++)
        {
            sync_mem[i] = 0;
        }
    }

    // Configure + arm hardware from BRISC — TRISCs do no hw work.
    auto& mgr = PerfCounterManager::instance();
    mgr.configure_all_zones(); // L1 metadata + hw configure + arm
}

// ============================================================================
// Profiler-buffer based zone monitoring (BRISC-only).
//
// Replaces the old stop_flags signaling: TRISCs no longer write per-thread L1
// flags at zone end (zero TRISC overhead — WC TRISC ELF is bit-identical to NC).
// Instead, BRISC scans each TRISC's profiler ring buffer (BUFFERS_START..)
// looking for top-level ZONE_END entries that the existing zone_scoped class
// already writes regardless of build mode.
// ============================================================================

// Mirrors of llk_profiler buffer layout (profiler.h is not included from BRISC).
constexpr std::uint32_t PROFILER_BUFFER_LENGTH = 0x400;
#if defined(ARCH_QUASAR)
constexpr std::uint32_t PROFILER_NUM_CORES   = 4;
constexpr std::uint32_t PROFILER_BUFFERS_END = 0x16F000;
#else
constexpr std::uint32_t PROFILER_NUM_CORES   = 3;
constexpr std::uint32_t PROFILER_BUFFERS_END = 0x16E000;
#endif
constexpr std::uint32_t PROFILER_BUFFERS_START = PROFILER_BUFFERS_END - (PROFILER_NUM_CORES * PROFILER_BUFFER_LENGTH * 4u);

// Initial wait so BRISC doesn't start polling the profiler buffer before
// TRISC's own llk_profiler::reset() has zeroed it. We deliberately do NOT
// pre-clear from BRISC: that 12KB L1 write burst leaves the L1 fabric busy
// at the exact moment TRISCs are released, slowing INIT-zone L1 access by
// ~13 cycles. Instead, BRISC just waits long enough for TRISC reset() to
// memset the 4KB-per-core buffer. ~8000 cycles is plenty.
constexpr std::uint32_t PERF_COUNTERS_INITIAL_WAIT_CYCLES = 8000;

inline void perf_counters_initial_wait()
{
    for (std::uint32_t i = 0; i < PERF_COUNTERS_INITIAL_WAIT_CYCLES; ++i)
    {
        asm volatile("nop");
    }
}

constexpr std::uint32_t PROFILER_ENTRY_TYPE_SHAMT    = 28u;
constexpr std::uint32_t PROFILER_ENTRY_EXISTS_BIT    = 0x80000000u;
constexpr std::uint32_t PROFILER_TYPE_TIMESTAMP      = 0b1000;
constexpr std::uint32_t PROFILER_TYPE_TIMESTAMP_DATA = 0b1001;
constexpr std::uint32_t PROFILER_TYPE_ZONE_START     = 0b1010;
constexpr std::uint32_t PROFILER_TYPE_ZONE_END       = 0b1011;

// Per-TRISC scan state. Tracks position in the ring buffer and current nesting
// depth so we only treat depth==0 ZONE_END as a top-level zone boundary.
struct TriscScanState
{
    std::uint32_t write_idx;
    std::uint32_t zone_ends_seen;
};

// Advance the scan for one TRISC by one entry, updating zone_ends_seen.
// Stops without consuming an entry when the next slot is unwritten — the
// caller polls again later.
inline void advance_trisc_scan(TriscScanState& state, std::uint32_t trisc_id)
{
    volatile std::uint32_t* buf = reinterpret_cast<volatile std::uint32_t*>(PROFILER_BUFFERS_START + trisc_id * PROFILER_BUFFER_LENGTH * 4u);

    ckernel::invalidate_data_cache();
    std::uint32_t meta = buf[state.write_idx];
    if ((meta & PROFILER_ENTRY_EXISTS_BIT) == 0u)
    {
        return; // not yet written
    }
    std::uint32_t type = (meta >> PROFILER_ENTRY_TYPE_SHAMT) & 0xFu;
    if (type == PROFILER_TYPE_TIMESTAMP_DATA)
    {
        state.write_idx += 4u; // meta + ts_low + data_high + data_low
    }
    else
    {
        state.write_idx += 2u;
    }
    if (type == PROFILER_TYPE_ZONE_END)
    {
        ++state.zone_ends_seen;
    }
}

// BRISC poll backoff between L1 reads. A tight `while (!done) { read L1 }` loop
// hammers the L1 fabric and creates contention with TRISC unpack/pack traffic
// (~0.5 cyc/iter slow-down on UNPACK/PACK/L1_TO_L1). 256 BRISC cycles (~190ns
// on BH) is short enough to catch zone boundaries with negligible drift, but
// long enough that BRISC reads only ~once per a TRISC tile iteration —
// removing the L1 contention.
constexpr std::uint32_t PERF_COUNTERS_POLL_BACKOFF_CYCLES = 64;

inline void perf_counters_poll_backoff()
{
    for (std::uint32_t i = 0; i < PERF_COUNTERS_POLL_BACKOFF_CYCLES; ++i)
    {
        asm volatile("nop");
    }
}

// Spin until every TRISC has emitted at least `target` ZONE_END entries.
inline void wait_for_zone_end_count(TriscScanState (&states)[PERF_COUNTERS_THREAD_COUNT], std::uint32_t target)
{
    while (true)
    {
        bool all_done = true;
        for (std::uint32_t t = 0; t < PERF_COUNTERS_THREAD_COUNT; ++t)
        {
            if (states[t].zone_ends_seen < target)
            {
                advance_trisc_scan(states[t], t);
                if (states[t].zone_ends_seen < target)
                {
                    all_done = false;
                }
            }
        }
        if (all_done)
        {
            return;
        }
        perf_counters_poll_backoff();
    }
}

// Passive waiter — TRISCs now snapshot their own counter values per zone via
// perf_counter_scoped::~dtor (LIFO-ordered with ZONE_END). BRISC just stays
// out of the way so it doesn't trample the new shared-config / per-zone-data
// L1 layout. The old freeze_read_and_rearm + freeze_and_read paths wrote
// (cycles, count) pairs in the legacy 2-word layout and would overflow into
// adjacent zones with the new MAX_ZONES=8 / shared-config design.
inline void monitor_zones_from_brisc()
{
    // Wait for TRISC reset() to memset the profiler buffer before returning,
    // so subsequent BRISC reads see fresh state from this run.
    perf_counters_initial_wait();
    // No-op past this point — TRISC's perf_counter_scoped writes per-zone
    // bank cycles + counter counts + sync_word directly to L1 from each
    // TRISC. Host reads via Python's _read_zone_counters() after BRISC's
    // kernel-complete signal.
}

// ============================================================================
// Counter ID Constants (for reference/documentation)
// ============================================================================

namespace counter_id
{
namespace instrn_thread
{
// Instruction availability counters (per-thread: add thread offset 0, 1, or 2)
constexpr std::uint32_t CFG_INSTRN_AVAILABLE_0     = 0;
constexpr std::uint32_t CFG_INSTRN_AVAILABLE_1     = 1;
constexpr std::uint32_t CFG_INSTRN_AVAILABLE_2     = 2;
constexpr std::uint32_t SYNC_INSTRN_AVAILABLE_0    = 3;
constexpr std::uint32_t SYNC_INSTRN_AVAILABLE_1    = 4;
constexpr std::uint32_t SYNC_INSTRN_AVAILABLE_2    = 5;
constexpr std::uint32_t THCON_INSTRN_AVAILABLE_0   = 6;
constexpr std::uint32_t THCON_INSTRN_AVAILABLE_1   = 7;
constexpr std::uint32_t THCON_INSTRN_AVAILABLE_2   = 8;
constexpr std::uint32_t XSEARCH_INSTRN_AVAILABLE_0 = 9;
constexpr std::uint32_t XSEARCH_INSTRN_AVAILABLE_1 = 10;
constexpr std::uint32_t XSEARCH_INSTRN_AVAILABLE_2 = 11;
constexpr std::uint32_t MOVE_INSTRN_AVAILABLE_0    = 12;
constexpr std::uint32_t MOVE_INSTRN_AVAILABLE_1    = 13;
constexpr std::uint32_t MOVE_INSTRN_AVAILABLE_2    = 14;
constexpr std::uint32_t FPU_INSTRN_AVAILABLE_0     = 15;
constexpr std::uint32_t FPU_INSTRN_AVAILABLE_1     = 16;
constexpr std::uint32_t FPU_INSTRN_AVAILABLE_2     = 17;
constexpr std::uint32_t UNPACK_INSTRN_AVAILABLE_0  = 18;
constexpr std::uint32_t UNPACK_INSTRN_AVAILABLE_1  = 19;
constexpr std::uint32_t UNPACK_INSTRN_AVAILABLE_2  = 20;
constexpr std::uint32_t PACK_INSTRN_AVAILABLE_0    = 21;
constexpr std::uint32_t PACK_INSTRN_AVAILABLE_1    = 22;
constexpr std::uint32_t PACK_INSTRN_AVAILABLE_2    = 23;
// Thread stalls
constexpr std::uint32_t THREAD_STALLS_0 = 24;
constexpr std::uint32_t THREAD_STALLS_1 = 25;
constexpr std::uint32_t THREAD_STALLS_2 = 26;
// Wait counters (shared across threads)
constexpr std::uint32_t WAITING_FOR_SRCA_CLEAR = 27;
constexpr std::uint32_t WAITING_FOR_SRCB_CLEAR = 28;
constexpr std::uint32_t WAITING_FOR_SRCA_VALID = 29;
constexpr std::uint32_t WAITING_FOR_SRCB_VALID = 30;
// Per-thread wait counters
constexpr std::uint32_t WAITING_FOR_THCON_IDLE_0  = 31;
constexpr std::uint32_t WAITING_FOR_THCON_IDLE_1  = 32;
constexpr std::uint32_t WAITING_FOR_THCON_IDLE_2  = 33;
constexpr std::uint32_t WAITING_FOR_UNPACK_IDLE_0 = 34;
constexpr std::uint32_t WAITING_FOR_UNPACK_IDLE_1 = 35;
constexpr std::uint32_t WAITING_FOR_UNPACK_IDLE_2 = 36;
constexpr std::uint32_t WAITING_FOR_PACK_IDLE_0   = 37;
constexpr std::uint32_t WAITING_FOR_PACK_IDLE_1   = 38;
constexpr std::uint32_t WAITING_FOR_PACK_IDLE_2   = 39;
constexpr std::uint32_t WAITING_FOR_MATH_IDLE_0   = 40;
constexpr std::uint32_t WAITING_FOR_MATH_IDLE_1   = 41;
constexpr std::uint32_t WAITING_FOR_MATH_IDLE_2   = 42;
constexpr std::uint32_t WAITING_FOR_NONZERO_SEM_0 = 43;
constexpr std::uint32_t WAITING_FOR_NONZERO_SEM_1 = 44;
constexpr std::uint32_t WAITING_FOR_NONZERO_SEM_2 = 45;
constexpr std::uint32_t WAITING_FOR_NONFULL_SEM_0 = 46;
constexpr std::uint32_t WAITING_FOR_NONFULL_SEM_1 = 47;
constexpr std::uint32_t WAITING_FOR_NONFULL_SEM_2 = 48;
constexpr std::uint32_t WAITING_FOR_MOVE_IDLE_0   = 49;
constexpr std::uint32_t WAITING_FOR_MOVE_IDLE_1   = 50;
constexpr std::uint32_t WAITING_FOR_MOVE_IDLE_2   = 51;
constexpr std::uint32_t WAITING_FOR_MMIO_IDLE_0   = 52;
constexpr std::uint32_t WAITING_FOR_MMIO_IDLE_1   = 53;
constexpr std::uint32_t WAITING_FOR_MMIO_IDLE_2   = 54;
constexpr std::uint32_t WAITING_FOR_SFPU_IDLE_0   = 55;
constexpr std::uint32_t WAITING_FOR_SFPU_IDLE_1   = 56;
constexpr std::uint32_t WAITING_FOR_SFPU_IDLE_2   = 57;
// Per-type instruction issue counts (grant counters, bit 8 set = ID 256+n)
constexpr std::uint32_t CFG_INSTRUCTIONS_0     = 256;
constexpr std::uint32_t CFG_INSTRUCTIONS_1     = 257;
constexpr std::uint32_t CFG_INSTRUCTIONS_2     = 258;
constexpr std::uint32_t SYNC_INSTRUCTIONS_0    = 259;
constexpr std::uint32_t SYNC_INSTRUCTIONS_1    = 260;
constexpr std::uint32_t SYNC_INSTRUCTIONS_2    = 261;
constexpr std::uint32_t THCON_INSTRUCTIONS_0   = 262;
constexpr std::uint32_t THCON_INSTRUCTIONS_1   = 263;
constexpr std::uint32_t THCON_INSTRUCTIONS_2   = 264;
constexpr std::uint32_t XSEARCH_INSTRUCTIONS_0 = 265;
constexpr std::uint32_t XSEARCH_INSTRUCTIONS_1 = 266;
constexpr std::uint32_t XSEARCH_INSTRUCTIONS_2 = 267;
constexpr std::uint32_t MOVE_INSTRUCTIONS_0    = 268;
constexpr std::uint32_t MOVE_INSTRUCTIONS_1    = 269;
constexpr std::uint32_t MOVE_INSTRUCTIONS_2    = 270;
constexpr std::uint32_t MATH_INSTRUCTIONS_0    = 271;
constexpr std::uint32_t MATH_INSTRUCTIONS_1    = 272;
constexpr std::uint32_t MATH_INSTRUCTIONS_2    = 273;
constexpr std::uint32_t UNPACK_INSTRUCTIONS_0  = 274;
constexpr std::uint32_t UNPACK_INSTRUCTIONS_1  = 275;
constexpr std::uint32_t UNPACK_INSTRUCTIONS_2  = 276;
constexpr std::uint32_t PACK_INSTRUCTIONS_0    = 277;
constexpr std::uint32_t PACK_INSTRUCTIONS_1    = 278;
constexpr std::uint32_t PACK_INSTRUCTIONS_2    = 279;
} // namespace instrn_thread

namespace fpu
{
constexpr std::uint32_t FPU_INSTRUCTION    = 0;
constexpr std::uint32_t SFPU_INSTRUCTION   = 1;
constexpr std::uint32_t FPU_OR_SFPU_INSTRN = 257;
} // namespace fpu

namespace tdma_unpack
{
constexpr std::uint32_t MATH_NOT_BLOCKED_BY_SRC          = 0;
constexpr std::uint32_t DATA_HAZARD_STALLS_MOVD2A        = 1;
constexpr std::uint32_t FIDELITY_PHASE_STALLS            = 2;
constexpr std::uint32_t MATH_INSTRN_STARTED              = 3;
constexpr std::uint32_t MATH_INSTRN_AVAILABLE            = 4;
constexpr std::uint32_t SRCB_WRITE_AVAILABLE             = 5;
constexpr std::uint32_t SRCA_WRITE_AVAILABLE             = 6;
constexpr std::uint32_t UNPACK0_BUSY_THREAD0             = 7;
constexpr std::uint32_t UNPACK1_BUSY_THREAD0             = 8;
constexpr std::uint32_t UNPACK0_BUSY_THREAD1             = 9;
constexpr std::uint32_t UNPACK1_BUSY_THREAD1             = 10;
constexpr std::uint32_t MATH_NOT_BLOCKED_BY_SRC_GRANT    = 256;
constexpr std::uint32_t INSTRN_2HF_CYCLES                = 257;
constexpr std::uint32_t INSTRN_1HF_CYCLE                 = 258;
constexpr std::uint32_t SRCB_WRITE                       = 259;
constexpr std::uint32_t SRCA_WRITE_NOT_BLOCKED_OVERWRITE = 260;
constexpr std::uint32_t SRCA_WRITE                       = 261;
constexpr std::uint32_t SRCB_WRITE_NOT_BLOCKED_PORT      = 262;
constexpr std::uint32_t SRCA_WRITE_THREAD0               = 263;
constexpr std::uint32_t SRCB_WRITE_THREAD0               = 264;
constexpr std::uint32_t SRCA_WRITE_THREAD1               = 265;
constexpr std::uint32_t SRCB_WRITE_THREAD1               = 266;
} // namespace tdma_unpack

namespace l1
{
// l1_mux = 0
constexpr std::uint32_t NOC_RING0_INCOMING_1 = 0;
constexpr std::uint32_t NOC_RING0_INCOMING_0 = 1;
constexpr std::uint32_t NOC_RING0_OUTGOING_1 = 2;
constexpr std::uint32_t NOC_RING0_OUTGOING_0 = 3;
constexpr std::uint32_t L1_ARB_TDMA_BUNDLE_1 = 4;
constexpr std::uint32_t L1_ARB_TDMA_BUNDLE_0 = 5;
constexpr std::uint32_t L1_ARB_UNPACKER      = 6;
constexpr std::uint32_t L1_NO_ARB_UNPACKER   = 7;

// l1_mux = 1 (same IDs, different mux setting)
constexpr std::uint32_t NOC_RING1_INCOMING_1 = 0;
constexpr std::uint32_t NOC_RING1_INCOMING_0 = 1;
constexpr std::uint32_t NOC_RING1_OUTGOING_1 = 2;
constexpr std::uint32_t NOC_RING1_OUTGOING_0 = 3;
constexpr std::uint32_t TDMA_BUNDLE_1_ARB    = 4;
constexpr std::uint32_t TDMA_BUNDLE_0_ARB    = 5;
constexpr std::uint32_t TDMA_EXT_UNPACK_9_10 = 6;
constexpr std::uint32_t TDMA_PACKER_2_WR     = 7;
} // namespace l1

namespace tdma_pack
{
constexpr std::uint32_t PACKER_DEST_READ_AVAILABLE_0  = 11;
constexpr std::uint32_t PACKER_DEST_READ_AVAILABLE_1  = 12;
constexpr std::uint32_t PACKER_DEST_READ_AVAILABLE_2  = 13;
constexpr std::uint32_t PACKER_DEST_READ_AVAILABLE_3  = 14;
constexpr std::uint32_t PACKER_BUSY_0                 = 15;
constexpr std::uint32_t PACKER_BUSY_1                 = 16;
constexpr std::uint32_t PACKER_BUSY_2                 = 17;
constexpr std::uint32_t PACKER_BUSY                   = 18;
constexpr std::uint32_t DEST_READ_GRANTED_0           = 267;
constexpr std::uint32_t DEST_READ_GRANTED_1           = 268;
constexpr std::uint32_t DEST_READ_GRANTED_2           = 269;
constexpr std::uint32_t DEST_READ_GRANTED_3           = 270;
constexpr std::uint32_t MATH_NOT_STALLED_BY_DEST_PORT = 271;
constexpr std::uint32_t AVAILABLE_MATH                = 272;
} // namespace tdma_pack
} // namespace counter_id

} // namespace llk_perf

namespace llk_perf
{
// ── Runtime zone allocator ──────────────────────────────────────────
// Maps zone-name 32-bit DJB2 hashes → sequential zone IDs (0..MAX_ZONES-1).
// Linear table (≤ 8 entries) with full-hash compare: collision-safe vs the old
// modulo-indexed scheme. Each TRISC's BSS gets its own copy; all TRISCs see
// MEASURE_PERF_COUNTERS calls in source order, so IDs assigned consistently.
namespace detail
{
__attribute__((section(".bss.perf_counters"))) static std::uint32_t zone_hashes[PERF_COUNTERS_MAX_ZONES]; // 0 = empty
__attribute__((section(".bss.perf_counters"))) static std::uint32_t next_zone_id;

#ifndef _LLK_PERF_ZONE_ALLOCATOR_DEFINED_
#define _LLK_PERF_ZONE_ALLOCATOR_DEFINED_

// Full 32-bit DJB2; nudge 0 → 1 so 0 stays unique as "empty slot" sentinel.
constexpr std::uint32_t zone_name_hash(const char* s)
{
    std::uint32_t h = 5381u;
    while (*s)
    {
        h = h * 33u + static_cast<std::uint32_t>(*s++);
    }
    return h ? h : 1u;
}
#endif // _LLK_PERF_ZONE_ALLOCATOR_DEFINED_
} // namespace detail

// Always-inline so the call site doesn't pay JAL + register save/restore costs.
// Linear search over an 8-slot table is ~8 loads + compares = trivial; first
// call additionally writes the new hash + bumps next_zone_id (cold path).
__attribute__((always_inline)) inline std::uint32_t get_zone_id(std::uint32_t hash_val)
{
    std::uint32_t n = detail::next_zone_id;
    for (std::uint32_t i = 0; i < n; ++i)
    {
        if (detail::zone_hashes[i] == hash_val)
        {
            return i;
        }
    }
    if (n < PERF_COUNTERS_MAX_ZONES)
    {
        detail::zone_hashes[n] = hash_val;
        detail::next_zone_id   = n + 1;
        return n;
    }
    return 0; // overflow: more than 8 distinct zone names — silent fallback
}

// Hooks (start_perf_counters / stop_perf_counters) are defined above as no-ops.

} // namespace llk_perf

namespace llk_perf // reopen
{

// RAII wrapper for automatic per-zone counter start/stop.
// May already be defined by perf.h (forward-declaration version).
#ifndef _LLK_PERF_COUNTER_SCOPED_DEFINED_
#define _LLK_PERF_COUNTER_SCOPED_DEFINED_

// Runtime zone_id (not template) — enables string-only MEASURE_PERF_COUNTERS
// macro via get_zone_id(zone_name_hash(name)). Runtime arithmetic for L1
// offsets adds ~5 cyc to the dtor, all OUTSIDE the wall_clock window (zone_dtor
// captures t_end before this runs via LIFO destruction).
//
// Data layout per zone (matches counters.py parser):
//   word [0..4]:   per-bank cycles (OUT_L of INSTRN, FPU, TDMA_UNPACK, L1, TDMA_PACK)
//   word [5..5+N): per-counter event counts (OUT_H), one slot per valid config entry
//   word [last]:   sync_word = 0xFF (set after writes complete)
struct perf_counter_scoped
{
    std::uint32_t zone_id;

    perf_counter_scoped(const perf_counter_scoped&)            = delete;
    perf_counter_scoped(perf_counter_scoped&&)                 = delete;
    perf_counter_scoped& operator=(const perf_counter_scoped&) = delete;
    perf_counter_scoped& operator=(perf_counter_scoped&&)      = delete;

    inline __attribute__((always_inline)) explicit perf_counter_scoped(std::uint32_t zid) : zone_id(zid)
    {
        // ARM all 5 banks: clear + start (rising edge 0→1) on each bank's own
        // base+8 control register. Mirrors tt-metal's start_single_group. The
        // PERF_CNT_ALL helper only re-arms INSTRN+FPU as a pair, but per-bank
        // explicit arm is required so FPU/L1 re-arm reliably after a freeze.
        // Runs BEFORE zone_scoped::ctor — OUTSIDE wall_clock window.
        asm volatile("" ::: "memory");
        // PERF_CNT_ALL arms INSTRN_THREAD + FPU as a pair. Don't touch their
        // per-bank base+8 — doing so seems to leave FPU OUT_L unable to
        // re-latch on subsequent zones. TDMA_UNPACK, L1, TDMA_PACK each need
        // their own base+8 arm because they aren't gated by PERF_CNT_ALL.
        *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(0xFFB1203Cu) = 1u; // PERF_CNT_ALL (INSTRN+FPU)
        *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(0xFFB1203Cu) = 0u;
        *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(0xFFB12014u) = 1u; // TDMA_UNPACK
        *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(0xFFB12014u) = 0u;
        *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(0xFFB12038u) = 1u; // L1
        *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(0xFFB12038u) = 0u;
        *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(0xFFB120F8u) = 1u; // TDMA_PACK
        *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(0xFFB120F8u) = 0u;
        asm volatile("" ::: "memory");
    }

    inline __attribute__((always_inline)) ~perf_counter_scoped()
    {
        // Runs AFTER zone_scoped::dtor (LIFO destruction) — OUTSIDE wall_clock.
        // Sequence: freeze all banks → read 5 OUT_L (per-bank cycles) → iterate
        // shared config and read OUT_H per counter → write sync word.
        asm volatile("" ::: "memory");
        // FREEZE mirror of ctor: global stop for INSTRN+FPU, per-bank stop for
        // TDMA_UNPACK / L1 / TDMA_PACK.
        *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(0xFFB1203Cu) = 2u; // PERF_CNT_ALL
        *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(0xFFB1203Cu) = 0u;
        *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(0xFFB12014u) = 2u; // TDMA_UNPACK
        *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(0xFFB12014u) = 0u;
        *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(0xFFB12038u) = 2u; // L1
        *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(0xFFB12038u) = 0u;
        *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(0xFFB120F8u) = 2u; // TDMA_PACK
        *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(0xFFB120F8u) = 0u;
        asm volatile("" ::: "memory");

        // Per-bank register table (OUT_L address + counter_sel reg).
        struct bank_regs
        {
            std::uint32_t mode_reg;
            std::uint32_t out_l;
        };

        static constexpr bank_regs banks[5] = {
            {0xFFB12004u, 0xFFB12100u}, // 0 INSTRN_THREAD
            {0xFFB1201Cu, 0xFFB12120u}, // 1 FPU
            {0xFFB12010u, 0xFFB12108u}, // 2 TDMA_UNPACK
            {0xFFB12034u, 0xFFB12118u}, // 3 L1
            {0xFFB120F4u, 0xFFB12110u}, // 4 TDMA_PACK
        };

        // Compute zone's L1 addresses (runtime since zone_id is runtime).
        std::uint32_t cycles_base              = PERF_COUNTERS_ZONES_BASE + zone_id * PERF_COUNTERS_ZONE_SIZE;
        volatile std::uint32_t* bank_cycles    = reinterpret_cast<volatile std::uint32_t*>(cycles_base);
        volatile std::uint32_t* counter_counts = bank_cycles + PERF_COUNTERS_BANK_CYCLES_WORDS;

        // Sample bank cycles up front by reading INSTRN_THREAD's OUT_L. All 5
        // banks are armed and frozen together so their OUT_L values agree
        // within ±30 cyc; the FPU/L1 OUT_L registers also exhibit a HW quirk
        // where reads after a high counter_sel (e.g. FPU MATH_COUNTER=257)
        // return 0 on the second+ zone. Using INSTRN cycles for every bank is
        // a stable, simple workaround that keeps metric math non-degenerate.
        std::uint32_t shared_cycles = *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(banks[0].out_l);
        bank_cycles[0]              = shared_cycles;
        bank_cycles[1]              = shared_cycles;
        bank_cycles[2]              = shared_cycles;
        bank_cycles[3]              = shared_cycles;
        bank_cycles[4]              = shared_cycles;

        // Iterate shared config — set counter_sel per slot, read OUT_H (count).
        const volatile std::uint32_t* cfg = reinterpret_cast<volatile std::uint32_t*>(PERF_COUNTERS_SHARED_CONFIG_ADDR);
        std::uint32_t out_idx             = 0;
#pragma GCC unroll 0
        for (std::uint32_t i = 0; i < PERF_COUNTERS_CONFIG_WORDS; ++i)
        {
            std::uint32_t cw = cfg[i];
            if (!(cw & 0x80000000u))
            {
                continue;
            }
            std::uint32_t bank_id    = cw & 0xFFu;
            std::uint32_t counter_id = (cw >> 8) & 0x1FFu;
            std::uint32_t l1_mux     = (cw >> 17) & 0x7u;
            const bank_regs& br      = banks[bank_id];
            if (bank_id == 3u)
            {
                volatile std::uint32_t tt_reg_ptr* mux = reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(0xFFB12218u);
                *mux                                   = (*mux & ~(0x7u << 4)) | (l1_mux << 4);
            }
            *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(br.mode_reg) = counter_id << 8;
            counter_counts[out_idx]                                            = *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(br.out_l + 4u);
            ++out_idx;
        }

        // 3) Sync word marks zone complete.
        std::uint32_t sync_addr                               = PERF_COUNTERS_ZONES_BASE + zone_id * PERF_COUNTERS_ZONE_SIZE + PERF_COUNTERS_ZONE_DATA_BYTES;
        *reinterpret_cast<volatile std::uint32_t*>(sync_addr) = SYNC_ZONE_COMPLETE;
        asm volatile("" ::: "memory");
        // 32 nops padding AFTER read+sync.
        asm volatile(
            "nop\nnop\nnop\nnop\nnop\nnop\nnop\nnop\n"
            "nop\nnop\nnop\nnop\nnop\nnop\nnop\nnop\n"
            "nop\nnop\nnop\nnop\nnop\nnop\nnop\nnop\n"
            "nop\nnop\nnop\nnop\nnop\nnop\nnop\nnop\n" ::
                : "memory");
        asm volatile("" ::: "memory");
    }
};
#endif // _LLK_PERF_COUNTER_SCOPED_DEFINED_

} // namespace llk_perf

// String-only API: zone_id allocated at runtime from zone_name's DJB2 hash.
// Single shared L1 layout supports up to PERF_COUNTERS_MAX_ZONES distinct names.
// Override-safe: only define if perf.h hasn't already.
#ifndef PERF_COUNTER_VAR_CONCAT_
#undef MEASURE_PERF_COUNTERS
#define PERF_COUNTER_VAR_CONCAT_(a, b)   a##b
#define PERF_COUNTER_VAR_(line)          PERF_COUNTER_VAR_CONCAT_(_perf_ctr_, line)
#define MEASURE_PERF_COUNTERS(zone_name) \
    const llk_perf::perf_counter_scoped PERF_COUNTER_VAR_(__LINE__)(llk_perf::get_zone_id(llk_perf::detail::zone_name_hash(zone_name)));
#endif

#endif // PERF_COUNTERS_COMPILED
