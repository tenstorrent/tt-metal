// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Default: MEASURE_PERF_COUNTERS expands to nothing.
// perf.h may provide its own definition (with forward declarations), so guard this.
#ifndef MEASURE_PERF_COUNTERS
#define MEASURE_PERF_COUNTERS(zone_name)
#endif

#ifndef PERF_COUNTERS_COMPILED

// Stub functions for non-counter build — called unconditionally from
// trisc.cpp and brisc.cpp so both builds produce identical .text layout.
// noinline + memory clobber so the compiler treats them identically to
// the real counter functions (same call-site code in main()).
namespace llk_perf
{
__attribute__((noinline)) inline void start_perf_counters(unsigned int)
{
    asm volatile("" ::: "memory");
}

__attribute__((noinline)) inline void stop_perf_counters(unsigned int)
{
    asm volatile("" ::: "memory");
}

__attribute__((noinline)) inline void configure_and_arm_from_brisc()
{
    asm volatile("" ::: "memory");
}

__attribute__((noinline)) inline void write_counter_config_from_brisc()
{
    asm volatile("" ::: "memory");
}

inline void init_perf_counter_metadata()
{
}
} // namespace llk_perf

#else

#include <cstdint>

#include "ckernel.h"

namespace llk_perf
{

// ============================================================================
// L1 Memory Layout (Single Shared Buffer)
// ============================================================================

// SOURCE OF TRUTH: tests/python_tests/helpers/test_config.py (TestConfig class)
// Memory budget: 0x169000 to 0x16AFF3
#define PERF_COUNTERS_BASE_ADDR    0x169000
#define PERF_COUNTERS_CONFIG_WORDS 137 // Counter configuration slots (all Wormhole hw counters)
#define PERF_COUNTERS_DATA_WORDS   274 // Counter data (cycles + count per slot)
#define PERF_COUNTERS_BUFFER_SIZE  ((PERF_COUNTERS_CONFIG_WORDS + PERF_COUNTERS_DATA_WORDS) * 4)

constexpr std::uint32_t PERF_COUNTERS_ZONE_SIZE = PERF_COUNTERS_BUFFER_SIZE + 40; // +40 for sync words
constexpr std::uint32_t PERF_COUNTERS_MAX_ZONES = 3;

constexpr std::uint32_t perf_counters_config_addr(std::uint32_t zone)
{
    return PERF_COUNTERS_BASE_ADDR + zone * PERF_COUNTERS_ZONE_SIZE;
}

constexpr std::uint32_t perf_counters_data_addr(std::uint32_t zone)
{
    return perf_counters_config_addr(zone) + PERF_COUNTERS_CONFIG_WORDS * 4;
}

constexpr std::uint32_t perf_counters_sync_ctrl_addr(std::uint32_t zone)
{
    return perf_counters_config_addr(zone) + PERF_COUNTERS_BUFFER_SIZE;
}

// Thread count for perf counter synchronization
// Quasar: 4 TRISCs (UNPACK, MATH, PACK, SFPU); Wormhole/Blackhole: 3 TRISCs
#if defined(ARCH_QUASAR)
#define PERF_COUNTERS_THREAD_COUNT 4
#else
#define PERF_COUNTERS_THREAD_COUNT 3
#endif

// Atomic counters for ATINCGET-based synchronization
// Per-thread stop arrival flags (3 words at sync_ctrl + 4) — used by lightweight stop
constexpr std::uint32_t perf_counters_stop_flags_addr(std::uint32_t zone)
{
    return perf_counters_sync_ctrl_addr(zone) + 4;
}

// Atomic counters for ATINCGET-based synchronization (overlaps stop_flags when using lightweight mode)
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

// Barriers to synchronize all threads after start_hardware / stop_hardware.
// Ensures all threads enter/exit profiler zones at the same time,
// preventing inter-thread timing skew from leaking into profiler measurements.
constexpr std::uint32_t perf_counters_start_barrier_addr(std::uint32_t zone)
{
    return perf_counters_stop_elect_addr(zone) + 4;
}

constexpr std::uint32_t perf_counters_stop_barrier_addr(std::uint32_t zone)
{
    return perf_counters_start_barrier_addr(zone) + 4;
}

// Global enabled flag — set by BRISC, read by TRISCs. Located after all zone data.
constexpr std::uint32_t PERF_COUNTERS_ENABLED_FLAG_ADDR = PERF_COUNTERS_BASE_ADDR + PERF_COUNTERS_MAX_ZONES * PERF_COUNTERS_ZONE_SIZE;

// Pre-computed metadata written by BRISC, read by TRISCs.
// This moves ~400 cycles of compute_metadata() from TRISC init to BRISC setup.
constexpr std::uint32_t PERF_COUNTERS_BANK_MASK_ADDR = PERF_COUNTERS_ENABLED_FLAG_ADDR + 4;
// valid_count per zone: PERF_COUNTERS_MAX_ZONES words starting after bank_mask
constexpr std::uint32_t PERF_COUNTERS_VALID_COUNT_ADDR = PERF_COUNTERS_BANK_MASK_ADDR + 4;

// ============================================================================
// Sync Control Word Bit Layout
// ============================================================================

// Sync control word bit layout (layout differs for 3 vs 4 TRISCs):
// 3 TRISCs: Bits 0-2 start, 3-5 stop, 6 started, 7 stopped, 8-9 starter, 10-11 stopper
// 4 TRISCs: Bits 0-3 start, 4-7 stop, 8 started, 9 stopped, 10-11 starter, 12-13 stopper

// Lightweight sync: zone complete marker
constexpr std::uint32_t SYNC_ZONE_COMPLETE  = 0xFFu;
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
constexpr std::uint32_t COUNTER_SLOT_COUNT = 137;

// ============================================================================
// Helper Functions
// ============================================================================

// Hardware register access functions for performance counter control
namespace hw_access
{
// Write a 32-bit value to a hardware register address
inline void write_reg(std::uint32_t addr, std::uint32_t value)
{
    *reinterpret_cast<volatile std::uint32_t*>(addr) = value;
}

// Read a 32-bit value from a hardware register address
inline std::uint32_t read_reg(std::uint32_t addr)
{
    return *reinterpret_cast<volatile std::uint32_t*>(addr);
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
                const std::uint8_t l1_mux = (metadata >> 17) & 0x1;
                std::uint32_t cur         = hw_access::read_reg(RISCV_DEBUG_REG_PERF_CNT_MUX_CTRL);
                hw_access::write_reg(RISCV_DEBUG_REG_PERF_CNT_MUX_CTRL, (cur & ~(1u << 4)) | ((l1_mux & 0x1u) << 4));
            }

            std::uint32_t counter_base = hw_access::get_counter_base_addr(bank);
            hw_access::write_reg(counter_base, 0xFFFFFFFF); // Reference period
            hw_access::write_reg(counter_base + 4, 0);      // Mode register

            configured_mask |= bank_bit;
        }
    }

    // Arm only banks that have configured counters: clear + start.
    void arm_hardware()
    {
        for (std::uint32_t b = 0; b < COUNTER_BANK_COUNT; ++b)
        {
            if (!(get_active_bank_mask() & (1u << b)))
            {
                continue;
            }
            std::uint32_t counter_base = hw_access::get_counter_base_addr(static_cast<counter_bank>(b));
            hw_access::write_reg(counter_base + 8, 0); // Clear
            hw_access::write_reg(counter_base + 8, 1); // Start
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
    void freeze_hardware()
    {
        for (std::uint32_t b = 0; b < COUNTER_BANK_COUNT; ++b)
        {
            if (!(get_active_bank_mask() & (1u << b)))
            {
                continue;
            }
            std::uint32_t counter_base = hw_access::get_counter_base_addr(static_cast<counter_bank>(b));
            hw_access::write_reg(counter_base + 8, 0); // Clear start/stop bits
            hw_access::write_reg(counter_base + 8, 2); // Stop (freeze)
        }
    }

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
            const std::uint8_t l1_mux      = (metadata >> 17) & 0x1;

            const counter_bank bank = static_cast<counter_bank>(bank_id);

            // Configure L1 MUX before reading
            if (bank == counter_bank::l1)
            {
                std::uint32_t cur = hw_access::read_reg(RISCV_DEBUG_REG_PERF_CNT_MUX_CTRL);
                hw_access::write_reg(RISCV_DEBUG_REG_PERF_CNT_MUX_CTRL, (cur & ~(1u << 4)) | ((l1_mux & 0x1u) << 4));
            }

            std::uint32_t counter_base = hw_access::get_counter_base_addr(bank);
            hw_access::write_reg(counter_base + 4, static_cast<std::uint32_t>(counter_id) << 8);

            // Single dummy read for settling after mode register write.
            // The actual reads follow 2+ instruction cycles later — sufficient
            // for the hardware mux to propagate the selected counter value.
            std::uint32_t output_low_addr  = hw_access::get_counter_output_low_addr(bank);
            std::uint32_t output_high_addr = hw_access::get_counter_output_high_addr(bank);
            (void)hw_access::read_reg(output_low_addr);

            // Actual read and write directly to L1 buffer
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

        if (found_valid)
        {
            for (std::uint32_t zone = 0; zone < PERF_COUNTERS_MAX_ZONES; ++zone)
            {
                configure_hardware(zone);
            }
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

    // Per-zone start: first thread to arrive configures + arms hardware.
    // On first call, reads pre-computed metadata from L1 (~5 cycles, written by BRISC).
    void start(std::uint32_t zone)
    {
        compute_metadata();

        // Skip if no valid counters configured for this zone.
        if (get_active_bank_mask() == 0 || get_valid_count(zone) == 0)
        {
            return;
        }

        // Thread 0 configures (first zone only) and arms counters.
        // configure_hardware writes ref_period/mode registers — only needed once
        // since all zones use the same counter configuration. Subsequent zones
        // just re-arm (clear+start) which resets counter values to 0 per RTL.
        if (thread_info::get_thread_id() == 0)
        {
            if (zone == 0)
            {
                configure_hardware(zone);
            }
            arm_hardware();
        }
        else
        {
            // Match arm_hardware's bus latency with dummy reads.
            // Zone 0: configure+arm = more latency (4 reads/bank).
            // Zone > 0: arm only = less latency (1 read/bank).
            const std::uint32_t reads_per_bank = (zone == 0) ? 4 : 1;
            for (std::uint32_t b = 0; b < COUNTER_BANK_COUNT; ++b)
            {
                if (get_active_bank_mask() & (1u << b))
                {
                    std::uint32_t base = hw_access::get_counter_base_addr(static_cast<counter_bank>(b));
                    for (std::uint32_t r = 0; r < reads_per_bank; ++r)
                    {
                        (void)hw_access::read_reg(base + 8);
                    }
                }
            }
        }

        // L1-flag barrier: all threads must arrive before entering profiler zone.
        // Required for ALL zones — without it, threads enter zones at different
        // times, causing inconsistent ZONE_START timestamps and biased measurements
        // (L1_TO_L1 TILE_LOOP shows +17% without barrier vs +0.3% with it).
        volatile std::uint32_t* start_flags       = reinterpret_cast<volatile std::uint32_t*>(perf_counters_stop_counter_addr(zone));
        start_flags[thread_info::get_thread_id()] = 1;
        asm volatile("fence" ::: "memory");
        for (std::uint32_t t = 0; t < PERF_COUNTER_THREADS; ++t)
        {
            while (!start_flags[t])
            {
                asm volatile("fence" ::: "memory");
            }
        }
    }

    // Per-zone stop: last thread to leave freezes + reads + deconfigures.
    // Early threads wait for the last thread to finish all hardware ops.
    void stop(std::uint32_t zone)
    {
        // Skip if no valid counters configured for this zone.
        if (get_active_bank_mask() == 0 || get_valid_count(zone) == 0)
        {
            return;
        }

        volatile std::uint32_t* stop_flags = reinterpret_cast<volatile std::uint32_t*>(perf_counters_stop_flags_addr(zone));
        const std::uint32_t thread_id      = thread_info::get_thread_id();

        // Signal arrival.
        stop_flags[thread_id] = 1;
        asm volatile("fence" ::: "memory");

        // Check if we're the last thread (other flags already set).
        bool is_last = true;
        for (std::uint32_t t = 0; t < PERF_COUNTER_THREADS; ++t)
        {
            if (t != thread_id && !stop_flags[t])
            {
                is_last = false;
                break;
            }
        }

        if (!is_last)
        {
            // Early threads must wait until the last thread finishes
            // freeze + read + deconfigure. Otherwise they enter the next zone
            // while counter hardware is still active, causing bus contention.
            volatile std::uint32_t* sync_ctrl = get_sync_ctrl_mem(zone);
            while (read_l1_word(sync_ctrl) == 0)
            {
                asm volatile("fence" ::: "memory");
            }
            return;
        }

        // Last thread out: freeze counters and read values.
        // Skip deconfigure for zone 0 — zone 1 reuses the same hw config.
        // Only deconfigure after the last zone to avoid register writes
        // that cause bus contention with the next zone's FPU operations.
        freeze_hardware();
        read_hardware(zone);
        if (zone > 0)
        {
            deconfigure_hardware();
        }

        // Clear stop flags AND start flags for next zone.
        stop_flags[0] = 0;
        stop_flags[1] = 0;
        stop_flags[2] = 0;

        volatile std::uint32_t* start_flags = reinterpret_cast<volatile std::uint32_t*>(perf_counters_stop_counter_addr(zone));
        start_flags[0]                      = 0;
        start_flags[1]                      = 0;
        start_flags[2]                      = 0;

        // Signal early threads that ALL hardware ops are done.
        volatile std::uint32_t* sync_ctrl = get_sync_ctrl_mem(zone);
        *sync_ctrl                        = SYNC_ZONE_COMPLETE | (thread_id << SYNC_STOPPER_SHIFT);
        ckernel::invalidate_data_cache();
    }
};

// ============================================================================
// Public API
// ============================================================================

// Start performance counters (call from all threads)
// noinline + cold + aligned(64): keep counter code out of the hot instruction cache
// and ensure it occupies whole cache lines so it doesn't shift TILE_LOOP code alignment.
__attribute__((noinline, section(".text.zzz_perf_counters"))) inline void start_perf_counters(std::uint32_t zone)
{
    PerfCounterManager::instance().start(zone);
}

// Initialize counter metadata without arming. Called from trisc.cpp
// before stop_perf_counters so stop knows which banks to read.
inline void init_perf_counter_metadata()
{
    auto& mgr = PerfCounterManager::instance();
    if (!mgr.is_metadata_ready())
    {
        mgr.init_metadata();
    }
}

// Stop performance counters (call from all threads)
__attribute__((noinline, section(".text.zzz_perf_counters"))) inline void stop_perf_counters(std::uint32_t zone)
{
    PerfCounterManager::instance().stop(zone);
}

// Pre-configure all counter banks (called by BRISC before TRISCs start)
inline void configure_perf_counters_from_brisc()
{
    PerfCounterManager::instance().configure_all_zones();
}

// Configure + arm from BRISC (before releasing TRISCs).
// TRISCs have zero counter code in run_kernel.
// Built-in counter config: all 137 Wormhole hardware counters.
// Written to L1 by BRISC (local write, no NOC) instead of by Python host.
// This eliminates the Python NOC write which changes L1 controller state
// and causes ~7 cycle overhead on Float16 unpack operations.
// clang-format off
constexpr std::uint32_t BUILTIN_COUNTER_CONFIG[] = {
    0x80000000, 0x80000100, 0x80000200, 0x80000300, 0x80000400, 0x80000500, 0x80000600, 0x80000700,
    0x80000800, 0x80000900, 0x80000A00, 0x80000B00, 0x80000C00, 0x80000D00, 0x80000E00, 0x80000F00,
    0x80001000, 0x80001100, 0x80001200, 0x80001300, 0x80001400, 0x80001500, 0x80001600, 0x80001700,
    0x80001800, 0x80001900, 0x80001A00, 0x80001B00, 0x80001C00, 0x80001D00, 0x80001E00, 0x80001F00,
    0x80002000, 0x80002100, 0x80002200, 0x80002300, 0x80002400, 0x80002500, 0x80002600, 0x80002700,
    0x80002800, 0x80002900, 0x80002A00, 0x80002B00, 0x80002C00, 0x80002D00, 0x80002E00, 0x80002F00,
    0x80003000, 0x80003100, 0x80003200, 0x80003300, 0x80003400, 0x80003500, 0x80003600, 0x80003700,
    0x80003800, 0x80003900, 0x80003A00, 0x80003B00, 0x80003C00, 0x80003D00, 0x80003E00, 0x80003F00,
    0x80004000, 0x80004100, 0x80004200, 0x80004300, 0x80004400, 0x80004500, 0x80004600, 0x80004700,
    0x80004800, 0x80004900, 0x80004A00, 0x80004B00, 0x80004C00, 0x80004D00, 0x80004E00, 0x80004F00,
    0x80005000, 0x80005100, 0x80000001, 0x80000101, 0x80010101, 0x80000002, 0x80000102, 0x80000202,
    0x80000302, 0x80000402, 0x80000502, 0x80000602, 0x80000702, 0x80000802, 0x80000902, 0x80000A02,
    0x80010002, 0x80010102, 0x80010202, 0x80010302, 0x80010402, 0x80010502, 0x80010602, 0x80010702,
    0x80010802, 0x80010902, 0x80010A02, 0x80000003, 0x80000103, 0x80000203, 0x80000303, 0x80000403,
    0x80000503, 0x80000603, 0x80000703, 0x80020003, 0x80020103, 0x80020203, 0x80020303, 0x80020403,
    0x80020503, 0x80020603, 0x80020703, 0x80000004, 0x80000104, 0x80000B04, 0x80000C04, 0x80000D04,
    0x80000E04, 0x80000F04, 0x80001004, 0x80010004, 0x80010104, 0x80010204, 0x80010304, 0x80010404,
    0x80010504,
};
constexpr std::uint32_t BUILTIN_COUNTER_COUNT = sizeof(BUILTIN_COUNTER_CONFIG) / sizeof(BUILTIN_COUNTER_CONFIG[0]);
// clang-format on

inline void configure_and_arm_from_brisc()
{
    // Write built-in config to ALL zone L1 buffers (local write, no NOC).
    // Each zone needs its own copy so configure_hardware(zone) can read it.
    for (std::uint32_t zone = 0; zone < 2; ++zone)
    {
        volatile std::uint32_t* config_mem = reinterpret_cast<volatile std::uint32_t*>(perf_counters_config_addr(zone));
        for (std::uint32_t i = 0; i < BUILTIN_COUNTER_COUNT; i++)
        {
            config_mem[i] = BUILTIN_COUNTER_CONFIG[i];
        }
        for (std::uint32_t i = BUILTIN_COUNTER_COUNT; i < COUNTER_SLOT_COUNT; i++)
        {
            config_mem[i] = 0;
        }

        // Clear data buffer for this zone.
        volatile std::uint32_t* data_mem = reinterpret_cast<volatile std::uint32_t*>(perf_counters_data_addr(zone));
        for (std::uint32_t i = 0; i < PERF_COUNTERS_DATA_WORDS; i++)
        {
            data_mem[i] = 0;
        }

        // Clear sync control + stop flags + barriers for this zone.
        // 10 words covers: sync_ctrl(1) + stop_flags(3) + start_counter(3) +
        // stop_counter(3) + stop_elect(1) + start_barrier(1) + stop_barrier(1)
        volatile std::uint32_t* sync_mem = reinterpret_cast<volatile std::uint32_t*>(perf_counters_sync_ctrl_addr(zone));
        for (std::uint32_t i = 0; i < 10; i++)
        {
            sync_mem[i] = 0;
        }
    }

    // Compute and write L1 metadata for all zones — no hardware register writes.
    // Hardware configure+arm is done by TRISC start_perf_counters() inside
    // the counter zone. Writing hw registers from BRISC causes ~5 cycle
    // overhead on pack operations.
    auto& mgr = PerfCounterManager::instance();
    mgr.configure_all_zones_metadata_only();
}

// Read frozen counter values from hardware into L1 data buffer.
// Called by BRISC after TRISCs finish (counters were frozen by TRISC asm).
inline void read_counters_from_brisc()
{
    PerfCounterManager::instance().read_counters(0);
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
// Maps compile-time zone name hashes to sequential zone IDs (0, 1, 2).
// zone_name_hash may already be defined by perf.h (constexpr, can't duplicate).
// Static storage and get_zone_id MUST come from counters.h (single definition).
namespace detail
{
constexpr std::uint32_t ZONE_UNALLOCATED = 0xFF;
constexpr std::uint32_t ZONE_LOOKUP_SIZE = 32;
__attribute__((section(".bss.perf_counters"))) static std::uint32_t zone_lookup[ZONE_LOOKUP_SIZE];
__attribute__((section(".bss.perf_counters"))) static std::uint32_t next_zone_id;
__attribute__((section(".bss.perf_counters"))) static bool zone_lookup_ready;

#ifndef _LLK_PERF_ZONE_ALLOCATOR_DEFINED_
#define _LLK_PERF_ZONE_ALLOCATOR_DEFINED_

constexpr std::uint32_t zone_name_hash(const char* s)
{
    std::uint32_t h = 5381;
    while (*s)
    {
        h = h * 33 + static_cast<std::uint32_t>(*s++);
    }
    return h % ZONE_LOOKUP_SIZE;
}
#endif // _LLK_PERF_ZONE_ALLOCATOR_DEFINED_
} // namespace detail

__attribute__((noinline, cold)) inline std::uint32_t get_zone_id(std::uint32_t hash_val)
{
    if (!detail::zone_lookup_ready)
    {
        for (std::uint32_t i = 0; i < detail::ZONE_LOOKUP_SIZE; ++i)
        {
            detail::zone_lookup[i] = detail::ZONE_UNALLOCATED;
        }
        detail::zone_lookup_ready = true;
    }
    if (hash_val < detail::ZONE_LOOKUP_SIZE && detail::zone_lookup[hash_val] == detail::ZONE_UNALLOCATED)
    {
        detail::zone_lookup[hash_val] = detail::next_zone_id++;
    }
    return (hash_val < detail::ZONE_LOOKUP_SIZE) ? detail::zone_lookup[hash_val] : 0;
}

// Profiler-integrated counter hooks (called from zone_scoped ctor/dtor).
// Auto-incrementing zone_id: first call = zone 0 (INIT), second = zone 1 (TILE_LOOP).
// The static counter resets naturally because BRISC clears .bss between runs.
namespace detail
{
__attribute__((section(".bss.perf_counters"))) static std::uint32_t profiler_zone_counter;
} // namespace detail

} // namespace llk_perf

namespace llk_profiler
{
__attribute__((noinline)) void _profiler_counter_start(std::uint32_t& zone_out, bool& active_out)
{
    zone_out   = llk_perf::detail::profiler_zone_counter++;
    active_out = true;
    llk_perf::start_perf_counters(zone_out);
}

__attribute__((noinline)) void _profiler_counter_stop(std::uint32_t zone, bool active)
{
    if (active)
    {
        llk_perf::stop_perf_counters(zone);
    }
}
} // namespace llk_profiler

namespace llk_perf // reopen
{

// RAII wrapper for automatic per-zone counter start/stop.
// May already be defined by perf.h (forward-declaration version).
#ifndef _LLK_PERF_COUNTER_SCOPED_DEFINED_
#define _LLK_PERF_COUNTER_SCOPED_DEFINED_

class perf_counter_scoped
{
    std::uint32_t m_zone;

public:
    perf_counter_scoped(const perf_counter_scoped&)            = delete;
    perf_counter_scoped(perf_counter_scoped&&)                 = delete;
    perf_counter_scoped& operator=(const perf_counter_scoped&) = delete;
    perf_counter_scoped& operator=(perf_counter_scoped&&)      = delete;

    __attribute__((noinline, cold)) explicit perf_counter_scoped(std::uint32_t hash) : m_zone(get_zone_id(hash))
    {
        start_perf_counters(m_zone);
    }

    __attribute__((noinline, cold)) ~perf_counter_scoped()
    {
        stop_perf_counters(m_zone);
    }
};
#endif // _LLK_PERF_COUNTER_SCOPED_DEFINED_

} // namespace llk_perf

// MEASURE_PERF_COUNTERS may already be defined by perf.h.
// Only redefine here if perf.h was not included first.
#ifndef PERF_COUNTER_VAR_CONCAT_
#undef MEASURE_PERF_COUNTERS
#define PERF_COUNTER_VAR_CONCAT_(a, b)   a##b
#define PERF_COUNTER_VAR_(line)          PERF_COUNTER_VAR_CONCAT_(_perf_ctr_, line)
#define MEASURE_PERF_COUNTERS(zone_name) const llk_perf::perf_counter_scoped PERF_COUNTER_VAR_(__LINE__)(llk_perf::detail::zone_name_hash(zone_name));
#endif

#endif // PERF_COUNTERS_COMPILED
