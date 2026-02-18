// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>

#include "ckernel.h"

namespace llk_perf
{

// Note: Most counter base register addresses are defined in hw_specific/<arch>/inc/tensix.h
// TDMA_UNPACK registers are missing from hardware headers, so we define them here
#ifndef RISCV_DEBUG_REG_PERF_CNT_TDMA_UNPACK0
#define RISCV_DEBUG_REG_PERF_CNT_TDMA_UNPACK0 (RISCV_DEBUG_REGS_START_ADDR | 0x00C)
#define RISCV_DEBUG_REG_PERF_CNT_TDMA_UNPACK1 (RISCV_DEBUG_REGS_START_ADDR | 0x010)
#define RISCV_DEBUG_REG_PERF_CNT_TDMA_UNPACK2 (RISCV_DEBUG_REGS_START_ADDR | 0x014)
#endif

// L1 counter MUX control register - bit 4 selects which L1 counter set is active
#ifndef RISCV_DEBUG_REG_PERF_CNT_MUX_CTRL
#define RISCV_DEBUG_REG_PERF_CNT_MUX_CTRL (RISCV_DEBUG_REGS_START_ADDR | 0x218)
#endif

// Performance counter output registers (for reading cycle/count results)
// NOTE: Table shows TDMA_UNPACK outputs at 0x018/0x01C but empirically 0x108/0x10C works
// IMPORTANT: The low output (`OUT_L`) returns the reference cycle count for the bank's
// measurement window. This value is independent of the selected event and will be
// identical across selections when scanning multiple counters in the same bank.
// The high output (`OUT_H`) returns the event-specific count for the currently selected
// counter (set via bits [15:8] of the mode register).
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

// L1 Memory addresses - separate per TRISC thread
// SOURCE OF TRUTH: tests/python_tests/helpers/test_config.py (TestConfig class)
// These values MUST match TestConfig.PERF_COUNTERS_* addresses.
// Must be below profiler buffers which start at 0x16B000
// Layout: 86 config words (344 bytes) + 172 data words (688 bytes) = 1032 bytes per thread
#define PERF_COUNTERS_BASE_ADDR    0x16A000
#define PERF_COUNTERS_SIZE         0xC18 // 3096 bytes for all 3 threads
#define PERF_COUNTERS_CONFIG_WORDS 86
#define PERF_COUNTERS_DATA_WORDS   172
#define PERF_COUNTERS_THREAD_SIZE  ((PERF_COUNTERS_CONFIG_WORDS + PERF_COUNTERS_DATA_WORDS) * 4) // 1032 bytes
// Computed addresses (UNPACK=thread 0, MATH=thread 1, PACK=thread 2)
#define PERF_COUNTER_UNPACK_CONFIG_ADDR (PERF_COUNTERS_BASE_ADDR)
#define PERF_COUNTER_UNPACK_DATA_ADDR   (PERF_COUNTERS_BASE_ADDR + PERF_COUNTERS_CONFIG_WORDS * 4)
#define PERF_COUNTER_MATH_CONFIG_ADDR   (PERF_COUNTERS_BASE_ADDR + PERF_COUNTERS_THREAD_SIZE)
#define PERF_COUNTER_MATH_DATA_ADDR     (PERF_COUNTERS_BASE_ADDR + PERF_COUNTERS_THREAD_SIZE + PERF_COUNTERS_CONFIG_WORDS * 4)
#define PERF_COUNTER_PACK_CONFIG_ADDR   (PERF_COUNTERS_BASE_ADDR + 2 * PERF_COUNTERS_THREAD_SIZE)
#define PERF_COUNTER_PACK_DATA_ADDR     (PERF_COUNTERS_BASE_ADDR + 2 * PERF_COUNTERS_THREAD_SIZE + PERF_COUNTERS_CONFIG_WORDS * 4)

// Configuration word format: [valid(31), l1_mux(17), counter_sel(8-16), bank_id(0-7)]
// Note: counter_sel uses 9 bits (bits 8-16) to support counter IDs up to 511
// Use compact underlying type to reduce memory footprint
enum class CounterBank : std::uint8_t
{
    INSTRN_THREAD = 0,
    FPU           = 1,
    TDMA_UNPACK   = 2,
    L1            = 3,
    TDMA_PACK     = 4,
};

// Number of counter banks represented by CounterBank enum
inline constexpr std::uint32_t COUNTER_BANK_COUNT = 5;
// Number of counter slots supported per thread (config words and data pairs)
inline constexpr std::uint32_t COUNTER_SLOT_COUNT = 86;

inline constexpr std::uint32_t get_counter_base_addr(CounterBank bank)
{
    constexpr std::uint32_t base_addrs[COUNTER_BANK_COUNT] = {
        RISCV_DEBUG_REG_PERF_CNT_INSTRN_THREAD0, // INSTRN_THREAD
        RISCV_DEBUG_REG_PERF_CNT_FPU0,           // FPU
        RISCV_DEBUG_REG_PERF_CNT_TDMA_UNPACK0,   // TDMA_UNPACK
        RISCV_DEBUG_REG_PERF_CNT_L1_0,           // L1
        RISCV_DEBUG_REG_PERF_CNT_TDMA_PACK0,     // TDMA_PACK
    };

    const std::uint32_t idx = static_cast<std::uint32_t>(bank);
    return idx < COUNTER_BANK_COUNT ? base_addrs[idx] : 0u;
}

inline constexpr std::uint32_t get_counter_output_low_addr(CounterBank bank)
{
    constexpr std::uint32_t low_addrs[COUNTER_BANK_COUNT] = {
        RISCV_DEBUG_REG_PERF_CNT_OUT_L_INSTRN_THREAD, // INSTRN_THREAD
        RISCV_DEBUG_REG_PERF_CNT_OUT_L_FPU,           // FPU
        RISCV_DEBUG_REG_PERF_CNT_OUT_L_TDMA_UNPACK,   // TDMA_UNPACK
        RISCV_DEBUG_REG_PERF_CNT_OUT_L_DBG_L1,        // L1
        RISCV_DEBUG_REG_PERF_CNT_OUT_L_TDMA_PACK,     // TDMA_PACK
    };

    const std::uint32_t idx = static_cast<std::uint32_t>(bank);
    return idx < COUNTER_BANK_COUNT ? low_addrs[idx] : 0u;
}

inline constexpr std::uint32_t get_counter_output_high_addr(CounterBank bank)
{
    constexpr std::uint32_t high_addrs[COUNTER_BANK_COUNT] = {
        RISCV_DEBUG_REG_PERF_CNT_OUT_H_INSTRN_THREAD, // INSTRN_THREAD
        RISCV_DEBUG_REG_PERF_CNT_OUT_H_FPU,           // FPU
        RISCV_DEBUG_REG_PERF_CNT_OUT_H_TDMA_UNPACK,   // TDMA_UNPACK
        RISCV_DEBUG_REG_PERF_CNT_OUT_H_DBG_L1,        // L1
        RISCV_DEBUG_REG_PERF_CNT_OUT_H_TDMA_PACK,     // TDMA_PACK
    };

    const std::uint32_t idx = static_cast<std::uint32_t>(bank);
    return idx < COUNTER_BANK_COUNT ? high_addrs[idx] : 0u;
}

// Note: Request/grant mode bit was removed - counter_sel now uses all 9 bits (8-16)

// Helper functions for direct register access
namespace detail
{
inline void write_reg(std::uint32_t addr, std::uint32_t value)
{
    *reinterpret_cast<volatile std::uint32_t*>(addr) = value;
}

inline std::uint32_t read_reg(std::uint32_t addr)
{
    return *reinterpret_cast<volatile std::uint32_t*>(addr);
}
} // namespace detail

struct CounterResult
{
    std::uint32_t cycles;
    std::uint32_t count;
    CounterBank bank;
    std::uint32_t counter_id;
};

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
constexpr std::uint32_t FPU_OR_SFPU_INSTRN = 257; // Combined FPU/SFPU
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
constexpr std::uint32_t SRCB_WRITE                = 259; // Bit 8 set
constexpr std::uint32_t SRCA_WRITE                = 261; // Bit 8 set
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
constexpr std::uint32_t AVAILABLE_MATH             = 272; // Bit 8 set
} // namespace tdma_pack
} // namespace counter_id

class PerfCounters
{
private:
    std::uint32_t num_counters {0};

    static inline volatile std::uint32_t* get_config_mem()
    {
#if defined(LLK_TRISC_UNPACK)
        constexpr std::uint32_t addr = PERF_COUNTER_UNPACK_CONFIG_ADDR;
#elif defined(LLK_TRISC_MATH)
        constexpr std::uint32_t addr = PERF_COUNTER_MATH_CONFIG_ADDR;
#elif defined(LLK_TRISC_PACK)
        constexpr std::uint32_t addr = PERF_COUNTER_PACK_CONFIG_ADDR;
#else
        constexpr std::uint32_t addr = PERF_COUNTER_MATH_CONFIG_ADDR;
#endif
        return reinterpret_cast<volatile std::uint32_t*>(addr);
    }

    static inline volatile std::uint32_t* get_data_mem()
    {
#if defined(LLK_TRISC_UNPACK)
        constexpr std::uint32_t addr = PERF_COUNTER_UNPACK_DATA_ADDR;
#elif defined(LLK_TRISC_MATH)
        constexpr std::uint32_t addr = PERF_COUNTER_MATH_DATA_ADDR;
#elif defined(LLK_TRISC_PACK)
        constexpr std::uint32_t addr = PERF_COUNTER_PACK_DATA_ADDR;
#else
        constexpr std::uint32_t addr = PERF_COUNTER_MATH_DATA_ADDR;
#endif
        return reinterpret_cast<volatile std::uint32_t*>(addr);
    }

public:
    PerfCounters() = default;

    void start()
    {
        volatile std::uint32_t* config_mem = get_config_mem();
        std::uint32_t started_mask         = 0;
        num_counters                       = 0;

        // Single pass: count counters and start each bank once
        for (std::uint32_t i = 0; i < COUNTER_SLOT_COUNT; i++)
        {
            const std::uint32_t metadata = config_mem[i];
            if ((metadata & 0x80000000u) == 0)
            {
                continue;
            }

            const std::uint8_t bank_id   = metadata & 0xFF;
            const std::uint32_t bank_bit = 1u << bank_id;
            num_counters++;

            // Skip if bank already started
            if (started_mask & bank_bit)
            {
                continue;
            }

            CounterBank bank = static_cast<CounterBank>(bank_id);

            // Configure L1 MUX if needed
            if (bank == CounterBank::L1)
            {
                const std::uint8_t l1_mux = (metadata >> 17) & 0x1;
                std::uint32_t cur         = detail::read_reg(RISCV_DEBUG_REG_PERF_CNT_MUX_CTRL);
                detail::write_reg(RISCV_DEBUG_REG_PERF_CNT_MUX_CTRL, (cur & ~(1u << 4)) | ((l1_mux & 0x1u) << 4));
            }

            // Start the bank
            std::uint32_t counter_base = get_counter_base_addr(bank);
            detail::write_reg(counter_base, 0xFFFFFFFF); // Reference period
            detail::write_reg(counter_base + 4, 0);      // Mode register
            detail::write_reg(counter_base + 8, 0);      // Clear start/stop
            detail::write_reg(counter_base + 8, 1);      // Start (rising edge)

            started_mask |= bank_bit;
        }
    }

    std::array<CounterResult, COUNTER_SLOT_COUNT> stop()
    {
        std::array<CounterResult, COUNTER_SLOT_COUNT> results;
        volatile std::uint32_t* config_mem = get_config_mem();
        volatile std::uint32_t* data_mem   = get_data_mem();

        std::uint32_t stopped_mask = 0;
        std::uint32_t result_idx   = 0;

        // Single pass: stop each bank once and read all counters
        for (std::uint32_t i = 0; i < COUNTER_SLOT_COUNT; i++)
        {
            const std::uint32_t metadata = config_mem[i];
            if ((metadata & 0x80000000u) == 0)
            {
                continue;
            }

            const std::uint8_t bank_id = metadata & 0xFF;
            // Counter ID uses 9 bits: bits 8-16 (values 0-511)
            const std::uint16_t counter_id = (metadata >> 8) & 0x1FF;
            const std::uint8_t l1_mux      = (metadata >> 17) & 0x1;
            const std::uint32_t bank_bit   = 1u << bank_id;

            CounterBank bank = static_cast<CounterBank>(bank_id);

            // Stop bank on first encounter
            if (!(stopped_mask & bank_bit))
            {
                std::uint32_t counter_base = get_counter_base_addr(bank);
                detail::write_reg(counter_base + 8, 0); // Clear
                detail::write_reg(counter_base + 8, 2); // Stop (bit1 0->1)
                stopped_mask |= bank_bit;
            }

            // Configure L1 MUX if needed before reading
            if (bank == CounterBank::L1)
            {
                std::uint32_t cur = detail::read_reg(RISCV_DEBUG_REG_PERF_CNT_MUX_CTRL);
                detail::write_reg(RISCV_DEBUG_REG_PERF_CNT_MUX_CTRL, (cur & ~(1u << 4)) | ((l1_mux & 0x1u) << 4));
            }

            std::uint32_t counter_base = get_counter_base_addr(bank);

            // Select the desired counter in mode register (counter_id in bits 8-16)
            detail::write_reg(counter_base + 4, static_cast<std::uint32_t>(counter_id) << 8);

            // Allow selection/mux to settle: perform a dummy read sequence
            std::uint32_t output_low_addr  = get_counter_output_low_addr(bank);
            std::uint32_t output_high_addr = get_counter_output_high_addr(bank);
            (void)detail::read_reg(output_low_addr);
            (void)detail::read_reg(output_high_addr);

            // Read outputs again for the actual value
            results[result_idx].cycles     = detail::read_reg(output_low_addr);
            results[result_idx].count      = detail::read_reg(output_high_addr);
            results[result_idx].bank       = bank;
            results[result_idx].counter_id = counter_id;

            // Write to L1 memory for Python to read (2 words per counter: cycles, count)
            data_mem[result_idx * 2]     = results[result_idx].cycles;
            data_mem[result_idx * 2 + 1] = results[result_idx].count;

            result_idx++;
        }

        return results;
    }
};

// RAII helper: reads config from L1, starts counters on construction, stops on destruction.
// Configuration is done externally via Python before running the kernel.
class ScopedPerfCounters
{
private:
    PerfCounters counters_;
    bool stopped_ = false;

public:
    ScopedPerfCounters()
    {
        counters_.start();
    }

    ~ScopedPerfCounters()
    {
        if (!stopped_)
        {
            counters_.stop();
        }
    }

    // Manually stop and return results; destructor will not stop again
    std::array<CounterResult, COUNTER_SLOT_COUNT> stop()
    {
        stopped_ = true;
        return counters_.stop();
    }

    // Non-copyable, non-movable
    ScopedPerfCounters(const ScopedPerfCounters&)            = delete;
    ScopedPerfCounters& operator=(const ScopedPerfCounters&) = delete;
    ScopedPerfCounters(ScopedPerfCounters&&)                 = delete;
    ScopedPerfCounters& operator=(ScopedPerfCounters&&)      = delete;
};

} // namespace llk_perf
