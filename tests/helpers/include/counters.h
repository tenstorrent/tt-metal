// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>

#include "ckernel.h"

namespace llk_perf
{

// ============================================================================
// Hardware Register Addresses
// ============================================================================

// TDMA_UNPACK registers (missing from hw headers)
#ifndef RISCV_DEBUG_REG_PERF_CNT_TDMA_UNPACK0
#define RISCV_DEBUG_REG_PERF_CNT_TDMA_UNPACK0 (RISCV_DEBUG_REGS_START_ADDR | 0x00C)
#define RISCV_DEBUG_REG_PERF_CNT_TDMA_UNPACK1 (RISCV_DEBUG_REGS_START_ADDR | 0x010)
#define RISCV_DEBUG_REG_PERF_CNT_TDMA_UNPACK2 (RISCV_DEBUG_REGS_START_ADDR | 0x014)
#endif

// L1 counter MUX control register
#ifndef RISCV_DEBUG_REG_PERF_CNT_MUX_CTRL
#define RISCV_DEBUG_REG_PERF_CNT_MUX_CTRL (RISCV_DEBUG_REGS_START_ADDR | 0x218)
#endif

// Performance counter output registers
// OUT_L: Reference cycle count (independent of event)
// OUT_H: Event-specific count (depends on selected counter)
#define RISCV_DEBUG_REG_PERF_CNT_OUT_L_INSTRN_THREAD (RISCV_DEBUG_REGS_START_ADDR | 0x100)
#define RISCV_DEBUG_REG_PERF_CNT_OUT_H_INSTRN_THREAD (RISCV_DEBUG_REGS_START_ADDR | 0x104)
#define RISCV_DEBUG_REG_PERF_CNT_OUT_L_TDMA_UNPACK   (RISCV_DEBUG_REGS_START_ADDR | 0x108)
#define RISCV_DEBUG_REG_PERF_CNT_OUT_H_TDMA_UNPACK   (RISCV_DEBUG_REGS_START_ADDR | 0x10C)
#define RISCV_DEBUG_REG_PERF_CNT_OUT_L_TDMA_PACK     (RISCV_DEBUG_REGS_START_ADDR | 0x110)
#define RISCV_DEBUG_REG_PERF_CNT_OUT_H_TDMA_PACK     (RISCV_DEBUG_REGS_START_ADDR | 0x114)
#define RISCV_DEBUG_REG_PERF_CNT_OUT_L_DBG_L1        (RISCV_DEBUG_REGS_START_ADDR | 0x118)
#define RISCV_DEBUG_REG_PERF_CNT_OUT_H_DBG_L1        (RISCV_DEBUG_REGS_START_ADDR | 0x11C)
#define RISCV_DEBUG_REG_PERF_CNT_OUT_L_FPU           (RISCV_DEBUG_REGS_START_ADDR | 0x120)
#define RISCV_DEBUG_REG_PERF_CNT_OUT_H_FPU           (RISCV_DEBUG_REGS_START_ADDR | 0x124)

// ============================================================================
// L1 Memory Layout (Single Shared Buffer)
// ============================================================================

// SOURCE OF TRUTH: tests/python_tests/helpers/test_config.py (TestConfig class)
// Must be below profiler buffers which start at 0x16B000
#define PERF_COUNTERS_BASE_ADDR    0x16A000
#define PERF_COUNTERS_CONFIG_WORDS 86  // Counter configuration slots
#define PERF_COUNTERS_DATA_WORDS   172 // Counter data (cycles + count per slot)
#define PERF_COUNTERS_BUFFER_SIZE  ((PERF_COUNTERS_CONFIG_WORDS + PERF_COUNTERS_DATA_WORDS) * 4)

#define PERF_COUNTERS_CONFIG_ADDR    (PERF_COUNTERS_BASE_ADDR)
#define PERF_COUNTERS_DATA_ADDR      (PERF_COUNTERS_BASE_ADDR + PERF_COUNTERS_CONFIG_WORDS * 4)
#define PERF_COUNTERS_SYNC_CTRL_ADDR (PERF_COUNTERS_BASE_ADDR + PERF_COUNTERS_BUFFER_SIZE)

// Thread count for perf counter synchronization
#define PERF_COUNTERS_THREAD_COUNT 3

// Atomic counters for ATINCGET-based synchronization
#define PERF_COUNTERS_START_COUNTER_ADDR (PERF_COUNTERS_SYNC_CTRL_ADDR + 4)
#define PERF_COUNTERS_STOP_COUNTER_ADDR  (PERF_COUNTERS_START_COUNTER_ADDR + (PERF_COUNTERS_THREAD_COUNT * 4))
#define PERF_COUNTERS_STOP_ELECT_ADDR    (PERF_COUNTERS_STOP_COUNTER_ADDR + (PERF_COUNTERS_THREAD_COUNT * 4))

// ============================================================================
// Sync Control Word Bit Layout
// ============================================================================

// Bit 0: UNPACK thread started flag
// Bit 1: MATH thread started flag
// Bit 2: PACK thread started flag
// Bit 3: UNPACK thread stopped flag
// Bit 4: MATH thread stopped flag
// Bit 5: PACK thread stopped flag
// Bit 6: Global started flag (at least one thread started)
// Bit 7: Global stopped flag (all threads stopped)
// Bits 8-9: Starter thread ID (0=UNPACK, 1=MATH, 2=PACK) - thread that initialized hardware
// Bits 10-11: Stopper thread ID (0=UNPACK, 1=MATH, 2=PACK) - thread that read hardware
// Bits 12-31: Reserved

constexpr std::uint32_t SYNC_START_MASK    = (1u << 0) | (1u << 1) | (1u << 2);
constexpr std::uint32_t SYNC_STOP_MASK     = SYNC_START_MASK << 3;
constexpr std::uint32_t SYNC_STARTED_FLAG  = 1u << 6;
constexpr std::uint32_t SYNC_STOPPED_FLAG  = 1u << 7;
constexpr std::uint32_t SYNC_STARTER_SHIFT = 8u;
constexpr std::uint32_t SYNC_STARTER_MASK  = 0x3u << SYNC_STARTER_SHIFT;
constexpr std::uint32_t SYNC_STOPPER_SHIFT = 10u;
constexpr std::uint32_t SYNC_STOPPER_MASK  = 0x3u << SYNC_STOPPER_SHIFT;

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
#define PERF_COUNTERS_USE_ATINCGET 1
#endif

constexpr std::uint32_t ATINCGET_WIDTH_32    = 31u; // IntWidth for 32-bit
constexpr std::uint32_t ATINCGET_DMANOP_WAIT = 96u;
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
constexpr std::uint32_t COUNTER_SLOT_COUNT = 86;

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
inline std::uint32_t get_counter_base_addr(counter_bank bank)
{
    constexpr std::uint32_t base_addrs[COUNTER_BANK_COUNT] = {
        RISCV_DEBUG_REG_PERF_CNT_INSTRN_THREAD0,
        RISCV_DEBUG_REG_PERF_CNT_FPU0,
        RISCV_DEBUG_REG_PERF_CNT_TDMA_UNPACK0,
        RISCV_DEBUG_REG_PERF_CNT_L1_0,
        RISCV_DEBUG_REG_PERF_CNT_TDMA_PACK0,
    };
    const std::uint32_t idx = static_cast<std::uint32_t>(bank);
    return idx < COUNTER_BANK_COUNT ? base_addrs[idx] : 0u;
}

// Get the output register address for cycle counts (reference counter)
// This counter runs continuously regardless of events
inline std::uint32_t get_counter_output_low_addr(counter_bank bank)
{
    constexpr std::uint32_t low_addrs[COUNTER_BANK_COUNT] = {
        RISCV_DEBUG_REG_PERF_CNT_OUT_L_INSTRN_THREAD,
        RISCV_DEBUG_REG_PERF_CNT_OUT_L_FPU,
        RISCV_DEBUG_REG_PERF_CNT_OUT_L_TDMA_UNPACK,
        RISCV_DEBUG_REG_PERF_CNT_OUT_L_DBG_L1,
        RISCV_DEBUG_REG_PERF_CNT_OUT_L_TDMA_PACK,
    };
    const std::uint32_t idx = static_cast<std::uint32_t>(bank);
    return idx < COUNTER_BANK_COUNT ? low_addrs[idx] : 0u;
}

// Get the output register address for event counts
// This counter increments based on the selected counter/event type
inline std::uint32_t get_counter_output_high_addr(counter_bank bank)
{
    constexpr std::uint32_t high_addrs[COUNTER_BANK_COUNT] = {
        RISCV_DEBUG_REG_PERF_CNT_OUT_H_INSTRN_THREAD,
        RISCV_DEBUG_REG_PERF_CNT_OUT_H_FPU,
        RISCV_DEBUG_REG_PERF_CNT_OUT_H_TDMA_UNPACK,
        RISCV_DEBUG_REG_PERF_CNT_OUT_H_DBG_L1,
        RISCV_DEBUG_REG_PERF_CNT_OUT_H_TDMA_PACK,
    };
    const std::uint32_t idx = static_cast<std::uint32_t>(bank);
    return idx < COUNTER_BANK_COUNT ? high_addrs[idx] : 0u;
}
} // namespace hw_access

// Thread identification and sync bit helpers
// Each TRISC thread (Unpack/Math/Pack) needs to identify itself for synchronization
namespace thread_info
{
// Get the current thread ID based on compile-time defines
// Returns: 0 (UNPACK), 1 (MATH), 2 (PACK)
constexpr std::uint32_t get_thread_id()
{
#if defined(LLK_TRISC_UNPACK)
    return 0u;
#elif defined(LLK_TRISC_MATH)
    return 1u;
#elif defined(LLK_TRISC_PACK)
    return 2u;
#else
    return 1u;
#endif
}

// Get the bit mask for this thread's start flag in sync control word
// Returns: bit 0 (UNPACK), bit 1 (MATH), or bit 2 (PACK)
constexpr std::uint32_t get_thread_start_bit()
{
    return 1u << get_thread_id();
}

// Get the bit mask for this thread's stop flag in sync control word
// Returns: bit 3 (UNPACK), bit 4 (MATH), or bit 5 (PACK)
constexpr std::uint32_t get_thread_stop_bit()
{
    return get_thread_start_bit() << 3;
}
} // namespace thread_info

// ============================================================================
// Performance Counter Manager (Singleton)
// ============================================================================

class PerfCounterManager
{
private:
    PerfCounterManager() = default;

    // Get pointer to L1 config buffer (86 words of counter metadata)
    const volatile std::uint32_t* get_config_mem()
    {
        return reinterpret_cast<volatile std::uint32_t*>(PERF_COUNTERS_CONFIG_ADDR);
    }

    // Get pointer to L1 data buffer (172 words: cycles + count per counter)
    volatile std::uint32_t* get_data_mem()
    {
        return reinterpret_cast<volatile std::uint32_t*>(PERF_COUNTERS_DATA_ADDR);
    }

    // Get pointer to sync control word (thread coordination flags)
    volatile std::uint32_t* get_sync_ctrl_mem()
    {
        return reinterpret_cast<volatile std::uint32_t*>(PERF_COUNTERS_SYNC_CTRL_ADDR);
    }

    // Force L1 cache flush by doing uncached write followed by uncached read
    // This ensures our write is visible to other cores in cache-incoherent system
    inline void flush_l1_cache(volatile std::uint32_t* addr)
    {
        // Read-modify-write to force cache flush
        // The asm volatile prevents compiler reordering
        std::uint32_t tmp;
        asm volatile(
            "lw %0, 0(%1)\n" // Load from address
            "sw %0, 0(%1)\n" // Store back same value (forces writeback)
            "lw %0, 0(%1)\n" // Load again (forces cache line fetch)
            : "=&r"(tmp)
            : "r"(addr)
            : "memory");
    }

    // Read an L1 word with a flush to improve visibility across threads.
    inline std::uint32_t read_l1_word(volatile std::uint32_t* addr)
    {
        flush_l1_cache(addr);
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

        const std::uint32_t saved_data = ckernel::regfile[kDataReg];
        const std::uint32_t saved_addr = ckernel::regfile[kAddrReg];

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

        // Wait until ATINCGET result is written back to the GPR
        for (std::uint32_t i = 0; i < ATINCGET_DMANOP_WAIT; ++i)
        {
            TTI_DMANOP;
        }

        const std::uint32_t old_value = ckernel::regfile[kDataReg];

        // Restore GPRs using the same ordered store sequence
        asm volatile(
            "sw %2, 0(%1)\n"
            "lw %0, 0(%1)\n"
            "addi x0, %0, 0\n"
            : "=&r"(tmp)
            : "r"(data_ptr), "r"(saved_data)
            : "memory");
        asm volatile(
            "sw %2, 0(%1)\n"
            "lw %0, 0(%1)\n"
            "addi x0, %0, 0\n"
            : "=&r"(tmp)
            : "r"(addr_ptr), "r"(saved_addr)
            : "memory");

        return old_value;
    }

    // Initialize and start hardware counters (called by first thread only)
    // Reads config from L1, configures L1 MUX if needed, and starts each bank
    void start_hardware()
    {
        const volatile std::uint32_t* config_mem = get_config_mem();
        std::uint32_t started_mask               = 0;

        for (std::uint32_t i = 0; i < COUNTER_SLOT_COUNT; i++)
        {
            const std::uint32_t metadata = config_mem[i];
            if ((metadata & 0x80000000u) == 0)
            {
                continue;
            }

            const std::uint8_t bank_id   = static_cast<std::uint8_t>(metadata);
            const std::uint32_t bank_bit = 1u << bank_id;

            if (started_mask & bank_bit)
            {
                continue;
            }

            const counter_bank bank = static_cast<counter_bank>(bank_id);

            // Configure L1 MUX if needed
            if (bank == counter_bank::l1)
            {
                const std::uint8_t l1_mux = (metadata >> 17) & 0x1;
                std::uint32_t cur         = hw_access::read_reg(RISCV_DEBUG_REG_PERF_CNT_MUX_CTRL);
                hw_access::write_reg(RISCV_DEBUG_REG_PERF_CNT_MUX_CTRL, (cur & ~(1u << 4)) | ((l1_mux & 0x1u) << 4));
            }

            // Start the bank
            std::uint32_t counter_base = hw_access::get_counter_base_addr(bank);
            hw_access::write_reg(counter_base, 0xFFFFFFFF); // Reference period
            hw_access::write_reg(counter_base + 4, 0);      // Mode register
            hw_access::write_reg(counter_base + 8, 0);      // Clear
            hw_access::write_reg(counter_base + 8, 1);      // Start

            started_mask |= bank_bit;
        }
    }

    // Stop hardware counters and read all results (called by last thread only)
    // Stops each bank, configures counter selectors, reads cycle/count pairs, writes to L1
    void stop_hardware()
    {
        const volatile std::uint32_t* config_mem = get_config_mem();
        volatile std::uint32_t* data_mem         = get_data_mem();

        std::uint32_t stopped_mask = 0;
        std::uint32_t result_idx   = 0;

        for (std::uint32_t i = 0; i < COUNTER_SLOT_COUNT; i++)
        {
            const std::uint32_t metadata = config_mem[i];
            if ((metadata & 0x80000000u) == 0)
            {
                continue;
            }

            const std::uint8_t bank_id     = static_cast<std::uint8_t>(metadata);
            const std::uint16_t counter_id = (metadata >> 8) & 0x1FF;
            const std::uint8_t l1_mux      = (metadata >> 17) & 0x1;
            const std::uint32_t bank_bit   = 1u << bank_id;

            const counter_bank bank = static_cast<counter_bank>(bank_id);

            // Stop bank on first encounter
            if (!(stopped_mask & bank_bit))
            {
                std::uint32_t counter_base = hw_access::get_counter_base_addr(bank);
                hw_access::write_reg(counter_base + 8, 0); // Clear
                hw_access::write_reg(counter_base + 8, 2); // Stop
                stopped_mask |= bank_bit;
            }

            // Configure L1 MUX before reading
            if (bank == counter_bank::l1)
            {
                std::uint32_t cur = hw_access::read_reg(RISCV_DEBUG_REG_PERF_CNT_MUX_CTRL);
                hw_access::write_reg(RISCV_DEBUG_REG_PERF_CNT_MUX_CTRL, (cur & ~(1u << 4)) | ((l1_mux & 0x1u) << 4));
            }

            std::uint32_t counter_base = hw_access::get_counter_base_addr(bank);
            hw_access::write_reg(counter_base + 4, static_cast<std::uint32_t>(counter_id) << 8);

            // Dummy read for settling
            std::uint32_t output_low_addr  = hw_access::get_counter_output_low_addr(bank);
            std::uint32_t output_high_addr = hw_access::get_counter_output_high_addr(bank);
            (void)hw_access::read_reg(output_low_addr);
            (void)hw_access::read_reg(output_high_addr);

            // Actual read and write directly to L1 buffer
            data_mem[result_idx * 2]     = hw_access::read_reg(output_low_addr);
            data_mem[result_idx * 2 + 1] = hw_access::read_reg(output_high_addr);

            result_idx++;
        }
    }

public:
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

    // Thread-safe start: CAS for flags + ATINCGET counter
    void start()
    {
        volatile std::uint32_t* sync_ctrl     = get_sync_ctrl_mem();
        volatile std::uint32_t* start_counter = reinterpret_cast<volatile std::uint32_t*>(PERF_COUNTERS_START_COUNTER_ADDR);
        const std::uint32_t thread_bit        = thread_info::get_thread_start_bit();
        const std::uint32_t thread_id         = thread_info::get_thread_id();

        __sync_synchronize();

#if PERF_COUNTERS_USE_ATINCGET
        // ATINCGET-based arrival counter (optional, per-thread address to avoid contention)
        (void)atincget_l1(reinterpret_cast<std::uint32_t>(start_counter) + (thread_id * sizeof(std::uint32_t)), 1u);
        flush_l1_cache(start_counter + thread_id);
#else
        (void)start_counter;
#endif

        // Simple CAS to determine first thread (for hardware init)
        bool is_first         = false;
        const int MAX_RETRIES = 1000;
        int retry_count       = 0;

        while (retry_count < MAX_RETRIES)
        {
            volatile std::uint32_t old_state = *sync_ctrl;

            // Check if we're first (no started flag yet)
            is_first = !(old_state & SYNC_STARTED_FLAG);

            // Set our start bit
            volatile std::uint32_t new_state = old_state | thread_bit;

            // If first, also set started flag
            if (is_first)
            {
                new_state |= SYNC_STARTED_FLAG;
                new_state = (new_state & ~SYNC_STARTER_MASK) | (thread_id << SYNC_STARTER_SHIFT);
            }

            flush_l1_cache(sync_ctrl);

            volatile std::uint32_t current_state = *sync_ctrl;
            if (current_state == old_state)
            {
                *sync_ctrl = new_state;
                flush_l1_cache(sync_ctrl);

                volatile std::uint32_t verify = *sync_ctrl;
                if ((verify & thread_bit) == thread_bit)
                {
                    break;
                }
            }
            retry_count++;
        }

        if (is_first)
        {
            start_hardware();
        }

        __sync_synchronize();
    }

    // Thread-safe stop: CAS for flags + ATINCGET counter
    void stop()
    {
        volatile std::uint32_t* sync_ctrl     = get_sync_ctrl_mem();
        volatile std::uint32_t* stop_counter  = reinterpret_cast<volatile std::uint32_t*>(PERF_COUNTERS_STOP_COUNTER_ADDR);
        volatile std::uint32_t* start_counter = reinterpret_cast<volatile std::uint32_t*>(PERF_COUNTERS_START_COUNTER_ADDR);
        volatile std::uint32_t* stop_elect    = reinterpret_cast<volatile std::uint32_t*>(PERF_COUNTERS_STOP_ELECT_ADDR);
        const std::uint32_t thread_bit        = thread_info::get_thread_stop_bit();
        const std::uint32_t thread_id         = thread_info::get_thread_id();

        __sync_synchronize();

        // Phase 1: Set our stop bit
        const int MAX_RETRIES = 1000;
        int retry_count       = 0;
        while (retry_count < MAX_RETRIES)
        {
            volatile std::uint32_t old_state = *sync_ctrl;
            volatile std::uint32_t new_state = old_state | thread_bit;

            flush_l1_cache(sync_ctrl);

            volatile std::uint32_t current_state = *sync_ctrl;
            if (current_state == old_state)
            {
                *sync_ctrl = new_state;
                flush_l1_cache(sync_ctrl);

                volatile std::uint32_t verify = *sync_ctrl;
                if ((verify & thread_bit) == thread_bit)
                {
                    break;
                }
            }
            retry_count++;
        }

#if PERF_COUNTERS_USE_ATINCGET
        // ATINCGET-based arrival counter (optional, per-thread address to avoid contention)
        (void)atincget_l1(reinterpret_cast<std::uint32_t>(stop_counter) + (thread_id * sizeof(std::uint32_t)), 1u);
        flush_l1_cache(stop_counter + thread_id);
#else
        (void)stop_counter;
#endif

        // Delay for write propagation
        for (volatile int i = 0; i < 100; i++)
            ;
        __sync_synchronize();

        // Phase 2: Use stop_elect as the last-arrival barrier.
        bool is_last               = false;
        const std::uint32_t ticket = atincget_l1(reinterpret_cast<std::uint32_t>(stop_elect), 1u);
        if (ticket + 1u == PERF_COUNTER_THREADS)
        {
            is_last = true;
        }

        if (is_last)
        {
            // Stop hardware immediately after the last thread arrives (minimize measured overhead).
            stop_hardware();

            // Consolidate sync_ctrl after counters are frozen.
            std::uint32_t start_bits = 0;
            std::uint32_t stop_bits  = 0;
            for (std::uint32_t tid = 0; tid < PERF_COUNTER_THREADS; ++tid)
            {
                if (read_l1_word(start_counter + tid) != 0u)
                {
                    start_bits |= (1u << tid);
                }
                if (read_l1_word(stop_counter + tid) != 0u)
                {
                    stop_bits |= (1u << tid);
                }
            }

            // If visibility is still incomplete, force bits to avoid false warnings.
            if (start_bits != SYNC_START_MASK)
            {
                start_bits = SYNC_START_MASK;
            }
            if (stop_bits != SYNC_START_MASK)
            {
                stop_bits = SYNC_START_MASK;
            }

            std::uint32_t final_state = *sync_ctrl;
            final_state &= ~(SYNC_START_MASK | SYNC_STOP_MASK | SYNC_STARTED_FLAG | SYNC_STOPPED_FLAG);
            final_state |= start_bits;
            final_state |= (stop_bits << 3);
            if (start_bits != 0u)
            {
                final_state |= SYNC_STARTED_FLAG;
            }
            if (stop_bits == SYNC_START_MASK)
            {
                final_state |= SYNC_STOPPED_FLAG;
            }

            // Preserve starter id, update stopper id to this thread.
            final_state = (final_state & ~SYNC_STARTER_MASK) | (*sync_ctrl & SYNC_STARTER_MASK);
            final_state = (final_state & ~SYNC_STOPPER_MASK) | (thread_id << SYNC_STOPPER_SHIFT);

            *sync_ctrl = final_state;
            flush_l1_cache(sync_ctrl);
        }

        __sync_synchronize();
    }
};

// ============================================================================
// Public API
// ============================================================================

// Start performance counters (call from all threads)
inline void start_perf_counters()
{
    PerfCounterManager::instance().start();
}

// Stop performance counters (call from all threads)
inline void stop_perf_counters()
{
    PerfCounterManager::instance().stop();
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
// Thread instruction counts (bit 8 set = ID 256+n)
constexpr std::uint32_t THREAD_INSTRUCTIONS_0 = 256;
constexpr std::uint32_t THREAD_INSTRUCTIONS_1 = 257;
constexpr std::uint32_t THREAD_INSTRUCTIONS_2 = 258;
} // namespace instrn_thread

namespace fpu
{
constexpr std::uint32_t FPU_INSTRUCTION    = 0;
constexpr std::uint32_t SFPU_INSTRUCTION   = 1;
constexpr std::uint32_t FPU_OR_SFPU_INSTRN = 257;
} // namespace fpu

namespace tdma_unpack
{
constexpr std::uint32_t DATA_HAZARD_STALLS_MOVD2A = 1;
constexpr std::uint32_t MATH_INSTRN_STARTED       = 3;
constexpr std::uint32_t MATH_INSTRN_AVAILABLE     = 4;
constexpr std::uint32_t SRCB_WRITE_AVAILABLE      = 5;
constexpr std::uint32_t SRCA_WRITE_AVAILABLE      = 6;
constexpr std::uint32_t UNPACK0_BUSY_THREAD0      = 7;
constexpr std::uint32_t UNPACK1_BUSY_THREAD0      = 8;
constexpr std::uint32_t UNPACK0_BUSY_THREAD1      = 9;
constexpr std::uint32_t UNPACK1_BUSY_THREAD1      = 10;
constexpr std::uint32_t SRCB_WRITE                = 259;
constexpr std::uint32_t SRCA_WRITE                = 261;
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
constexpr std::uint32_t PACKER_DEST_READ_AVAILABLE = 11;
constexpr std::uint32_t PACKER_BUSY                = 18;
constexpr std::uint32_t AVAILABLE_MATH             = 272;
} // namespace tdma_pack
} // namespace counter_id

} // namespace llk_perf
