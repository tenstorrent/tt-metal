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
// L1 layout — always compiled. Shared config + per-zone data.
// ============================================================================

#define PERF_COUNTERS_BASE_ADDR         0x169000
#define PERF_COUNTERS_CONFIG_WORDS      200
#define PERF_COUNTERS_DATA_WORDS        200
#define PERF_COUNTERS_BANK_CYCLES_WORDS 5

namespace llk_perf
{

constexpr std::uint32_t PERF_COUNTERS_MAX_ZONES = 8;
constexpr std::uint32_t SYNC_ZONE_COMPLETE      = 0xFFu;

constexpr std::uint32_t PERF_COUNTERS_ZONE_DATA_BYTES = (PERF_COUNTERS_BANK_CYCLES_WORDS + PERF_COUNTERS_DATA_WORDS) * 4;
constexpr std::uint32_t PERF_COUNTERS_ZONE_SIZE       = PERF_COUNTERS_ZONE_DATA_BYTES + 40;

constexpr std::uint32_t PERF_COUNTERS_SHARED_CONFIG_ADDR = PERF_COUNTERS_BASE_ADDR;
constexpr std::uint32_t PERF_COUNTERS_ZONES_BASE         = PERF_COUNTERS_BASE_ADDR + PERF_COUNTERS_CONFIG_WORDS * 4;

constexpr std::uint32_t perf_counters_zone_data_addr(std::uint32_t zone)
{
    return PERF_COUNTERS_ZONES_BASE + zone * PERF_COUNTERS_ZONE_SIZE;
}

constexpr std::uint32_t perf_counters_sync_ctrl_addr(std::uint32_t zone)
{
    return perf_counters_zone_data_addr(zone) + PERF_COUNTERS_ZONE_DATA_BYTES;
}

static_assert(PERF_COUNTERS_ZONES_BASE + PERF_COUNTERS_MAX_ZONES * PERF_COUNTERS_ZONE_SIZE <= 0x16AFF4u, "Perf counter L1 layout overflows profiler region");

constexpr std::uint32_t PERF_COUNTERS_ENABLED_FLAG_ADDR = PERF_COUNTERS_ZONES_BASE + PERF_COUNTERS_MAX_ZONES * PERF_COUNTERS_ZONE_SIZE;

} // namespace llk_perf

// ============================================================================
// BRISC entry points.
// ============================================================================

namespace llk_perf
{

constexpr std::uint32_t PERF_COUNTERS_BANK_MASK_ADDR   = PERF_COUNTERS_ENABLED_FLAG_ADDR + 4;
constexpr std::uint32_t PERF_COUNTERS_VALID_COUNT_ADDR = PERF_COUNTERS_BANK_MASK_ADDR + 4;

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

namespace hw_access
{
inline void write_reg(std::uint32_t addr, std::uint32_t value)
{
    *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(addr) = value;
}

inline std::uint32_t read_reg(std::uint32_t addr)
{
    return *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(addr);
}

// Counter bank base register address. Volatile cast prevents GCC from
// building a CSWTCH lookup table that would shift GP-offsets.
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
} // namespace hw_access

// Stateless — all state lives in L1 at fixed addresses.
class PerfCounterManager
{
private:
    PerfCounterManager() = default;

    static std::uint32_t get_active_bank_mask()
    {
        return *reinterpret_cast<volatile std::uint32_t*>(PERF_COUNTERS_BANK_MASK_ADDR);
    }

    const volatile std::uint32_t* get_config_mem(std::uint32_t /*zone*/)
    {
        return reinterpret_cast<volatile std::uint32_t*>(PERF_COUNTERS_SHARED_CONFIG_ADDR);
    }

    // Configure bank reference period + mode register from shared config.
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
                const std::uint8_t l1_mux = (metadata >> 17) & 0x7;
                std::uint32_t cur         = hw_access::read_reg(RISCV_DEBUG_REG_PERF_CNT_MUX_CTRL);
                hw_access::write_reg(RISCV_DEBUG_REG_PERF_CNT_MUX_CTRL, (cur & ~(0x7u << 4)) | ((l1_mux & 0x7u) << 4));
            }
            std::uint32_t counter_base = hw_access::get_counter_base_addr(bank);
            hw_access::write_reg(counter_base, 0xFFFFFFFF);
            hw_access::write_reg(counter_base + 4, 0);
            configured_mask |= bank_bit;
        }
    }

    // Per-bank arm + global PERF_CNT_ALL broadcast (rising edge 0→1 clears + starts).
    void arm_hardware()
    {
        for (std::uint32_t b = 0; b < COUNTER_BANK_COUNT; ++b)
        {
            if (!(get_active_bank_mask() & (1u << b)))
            {
                continue;
            }
            std::uint32_t counter_base = hw_access::get_counter_base_addr(static_cast<counter_bank>(b));
            hw_access::write_reg(counter_base + 8, 1);
            hw_access::write_reg(counter_base + 8, 0);
        }
        hw_access::write_reg(RISCV_DEBUG_REG_PERF_CNT_ALL, 1);
        hw_access::write_reg(RISCV_DEBUG_REG_PERF_CNT_ALL, 0);
    }

public:
    static PerfCounterManager& instance()
    {
        static PerfCounterManager instance;
        return instance;
    }

    PerfCounterManager(const PerfCounterManager&)            = delete;
    PerfCounterManager& operator=(const PerfCounterManager&) = delete;
    PerfCounterManager(PerfCounterManager&&)                 = delete;
    PerfCounterManager& operator=(PerfCounterManager&&)      = delete;

    // Scan shared config, pre-compute bank_mask + valid_counts to L1, configure
    // + arm hw. Called once from BRISC before releasing TRISCs.
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

        *reinterpret_cast<volatile std::uint32_t*>(PERF_COUNTERS_ENABLED_FLAG_ADDR) = found_valid ? 1u : 0u;
        *reinterpret_cast<volatile std::uint32_t*>(PERF_COUNTERS_BANK_MASK_ADDR)    = bank_mask;
        volatile std::uint32_t* valid_count_ptr                                     = reinterpret_cast<volatile std::uint32_t*>(PERF_COUNTERS_VALID_COUNT_ADDR);
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
};

// ============================================================================
// Per-arch built-in counter inventory.
// Source: tt_metal/hw/inc/internal/tt-1xx/{wormhole,blackhole}/hw_counters.h
// Config word: valid(31) | l1_mux<<17 (3b) | counter_id<<8 (9b) | bank_id (8b)
// ============================================================================

constexpr std::uint32_t _perf_cfg(std::uint8_t bank, std::uint16_t cid, std::uint8_t mux = 0)
{
    return 0x80000000u | (static_cast<std::uint32_t>(mux & 0x7u) << 17) | (static_cast<std::uint32_t>(cid & 0x1FFu) << 8) | static_cast<std::uint32_t>(bank);
}

// clang-format off
#if defined(ARCH_BLACKHOLE)
// BH = 169 counters: 59 INSTRN + 3 FPU + 22 TDMA_UNPACK + 5 TDMA_PACK + 80 L1 (16 × 5 mux banks).
constexpr std::uint32_t BUILTIN_COUNTER_CONFIG[] = {
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
    _perf_cfg(1, 0), _perf_cfg(1, 1), _perf_cfg(1, 257),
    _perf_cfg(2,   0), _perf_cfg(2,   1), _perf_cfg(2,   2), _perf_cfg(2,   3), _perf_cfg(2,   4), _perf_cfg(2,   5),
    _perf_cfg(2,   6), _perf_cfg(2,   7), _perf_cfg(2,   8), _perf_cfg(2,   9), _perf_cfg(2,  10),
    _perf_cfg(2, 256), _perf_cfg(2, 257), _perf_cfg(2, 258), _perf_cfg(2, 259), _perf_cfg(2, 260), _perf_cfg(2, 261),
    _perf_cfg(2, 262), _perf_cfg(2, 263), _perf_cfg(2, 264), _perf_cfg(2, 265), _perf_cfg(2, 266),
    _perf_cfg(4, 11), _perf_cfg(4, 18), _perf_cfg(4, 267), _perf_cfg(4, 271), _perf_cfg(4, 272),
    _perf_cfg(3, 0, 0), _perf_cfg(3, 1, 0), _perf_cfg(3, 2, 0), _perf_cfg(3, 3, 0),
    _perf_cfg(3, 4, 0), _perf_cfg(3, 5, 0), _perf_cfg(3, 6, 0), _perf_cfg(3, 7, 0),
    _perf_cfg(3, 256, 0), _perf_cfg(3, 257, 0), _perf_cfg(3, 258, 0), _perf_cfg(3, 259, 0),
    _perf_cfg(3, 260, 0), _perf_cfg(3, 261, 0), _perf_cfg(3, 262, 0), _perf_cfg(3, 263, 0),
    _perf_cfg(3, 0, 1), _perf_cfg(3, 1, 1), _perf_cfg(3, 2, 1), _perf_cfg(3, 3, 1),
    _perf_cfg(3, 4, 1), _perf_cfg(3, 5, 1), _perf_cfg(3, 6, 1), _perf_cfg(3, 7, 1),
    _perf_cfg(3, 256, 1), _perf_cfg(3, 257, 1), _perf_cfg(3, 258, 1), _perf_cfg(3, 259, 1),
    _perf_cfg(3, 260, 1), _perf_cfg(3, 261, 1), _perf_cfg(3, 262, 1), _perf_cfg(3, 263, 1),
    _perf_cfg(3, 0, 2), _perf_cfg(3, 1, 2), _perf_cfg(3, 2, 2), _perf_cfg(3, 3, 2),
    _perf_cfg(3, 4, 2), _perf_cfg(3, 5, 2), _perf_cfg(3, 6, 2), _perf_cfg(3, 7, 2),
    _perf_cfg(3, 256, 2), _perf_cfg(3, 257, 2), _perf_cfg(3, 258, 2), _perf_cfg(3, 259, 2),
    _perf_cfg(3, 260, 2), _perf_cfg(3, 261, 2), _perf_cfg(3, 262, 2), _perf_cfg(3, 263, 2),
    _perf_cfg(3, 0, 3), _perf_cfg(3, 1, 3), _perf_cfg(3, 2, 3), _perf_cfg(3, 3, 3),
    _perf_cfg(3, 4, 3), _perf_cfg(3, 5, 3), _perf_cfg(3, 6, 3), _perf_cfg(3, 7, 3),
    _perf_cfg(3, 256, 3), _perf_cfg(3, 257, 3), _perf_cfg(3, 258, 3), _perf_cfg(3, 259, 3),
    _perf_cfg(3, 260, 3), _perf_cfg(3, 261, 3), _perf_cfg(3, 262, 3), _perf_cfg(3, 263, 3),
    _perf_cfg(3, 0, 4), _perf_cfg(3, 1, 4), _perf_cfg(3, 2, 4), _perf_cfg(3, 3, 4),
    _perf_cfg(3, 4, 4), _perf_cfg(3, 5, 4), _perf_cfg(3, 6, 4), _perf_cfg(3, 7, 4),
    _perf_cfg(3, 256, 4), _perf_cfg(3, 257, 4), _perf_cfg(3, 258, 4), _perf_cfg(3, 259, 4),
    _perf_cfg(3, 260, 4), _perf_cfg(3, 261, 4), _perf_cfg(3, 262, 4), _perf_cfg(3, 263, 4),
};
#else
// WH = 130 counters: 59 INSTRN + 3 FPU + 22 TDMA_UNPACK + 14 TDMA_PACK + 32 L1 (2 mux banks).
constexpr std::uint32_t BUILTIN_COUNTER_CONFIG[] = {
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
    _perf_cfg(1, 0), _perf_cfg(1, 1), _perf_cfg(1, 257),
    _perf_cfg(2,   0), _perf_cfg(2,   1), _perf_cfg(2,   2), _perf_cfg(2,   3), _perf_cfg(2,   4), _perf_cfg(2,   5),
    _perf_cfg(2,   6), _perf_cfg(2,   7), _perf_cfg(2,   8), _perf_cfg(2,   9), _perf_cfg(2,  10),
    _perf_cfg(2, 256), _perf_cfg(2, 257), _perf_cfg(2, 258), _perf_cfg(2, 259), _perf_cfg(2, 260), _perf_cfg(2, 261),
    _perf_cfg(2, 262), _perf_cfg(2, 263), _perf_cfg(2, 264), _perf_cfg(2, 265), _perf_cfg(2, 266),
    _perf_cfg(4, 11), _perf_cfg(4, 12), _perf_cfg(4, 13), _perf_cfg(4, 14), _perf_cfg(4, 15), _perf_cfg(4, 16),
    _perf_cfg(4, 17), _perf_cfg(4, 18),
    _perf_cfg(4, 267), _perf_cfg(4, 268), _perf_cfg(4, 269), _perf_cfg(4, 270), _perf_cfg(4, 271), _perf_cfg(4, 272),
    _perf_cfg(3, 0, 0), _perf_cfg(3, 1, 0), _perf_cfg(3, 2, 0), _perf_cfg(3, 3, 0),
    _perf_cfg(3, 4, 0), _perf_cfg(3, 5, 0), _perf_cfg(3, 6, 0), _perf_cfg(3, 7, 0),
    _perf_cfg(3, 256, 0), _perf_cfg(3, 257, 0), _perf_cfg(3, 258, 0), _perf_cfg(3, 259, 0),
    _perf_cfg(3, 260, 0), _perf_cfg(3, 261, 0), _perf_cfg(3, 262, 0), _perf_cfg(3, 263, 0),
    _perf_cfg(3, 0, 1), _perf_cfg(3, 1, 1), _perf_cfg(3, 2, 1), _perf_cfg(3, 3, 1),
    _perf_cfg(3, 4, 1), _perf_cfg(3, 5, 1), _perf_cfg(3, 6, 1), _perf_cfg(3, 7, 1),
    _perf_cfg(3, 256, 1), _perf_cfg(3, 257, 1), _perf_cfg(3, 258, 1), _perf_cfg(3, 259, 1),
    _perf_cfg(3, 260, 1), _perf_cfg(3, 261, 1), _perf_cfg(3, 262, 1), _perf_cfg(3, 263, 1),
};
#endif
constexpr std::uint32_t BUILTIN_COUNTER_COUNT = sizeof(BUILTIN_COUNTER_CONFIG) / sizeof(BUILTIN_COUNTER_CONFIG[0]);
// clang-format on

// Write shared config to L1, clear per-zone data, then configure + arm hw.
inline void configure_and_arm_from_brisc()
{
    volatile std::uint32_t* shared_config = reinterpret_cast<volatile std::uint32_t*>(PERF_COUNTERS_SHARED_CONFIG_ADDR);
    for (std::uint32_t i = 0; i < BUILTIN_COUNTER_COUNT; i++)
    {
        shared_config[i] = BUILTIN_COUNTER_CONFIG[i];
    }
    for (std::uint32_t i = BUILTIN_COUNTER_COUNT; i < COUNTER_SLOT_COUNT; i++)
    {
        shared_config[i] = 0;
    }

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

    PerfCounterManager::instance().configure_all_zones();
}

} // namespace llk_perf

// ============================================================================
// String-only API: zone name → 32-bit DJB2 → sequential zone id (0..MAX_ZONES-1).
// ============================================================================

namespace llk_perf
{
namespace detail
{
__attribute__((section(".bss.perf_counters"))) static std::uint32_t zone_hashes[PERF_COUNTERS_MAX_ZONES];
__attribute__((section(".bss.perf_counters"))) static std::uint32_t next_zone_id;

#ifndef _LLK_PERF_ZONE_ALLOCATOR_DEFINED_
#define _LLK_PERF_ZONE_ALLOCATOR_DEFINED_

constexpr std::uint32_t zone_name_hash(const char* s)
{
    std::uint32_t h = 5381u;
    while (*s)
    {
        h = h * 33u + static_cast<std::uint32_t>(*s++);
    }
    return h ? h : 1u;
}
#endif
} // namespace detail

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
    return 0;
}

// ============================================================================
// perf_counter_scoped: RAII arm-at-ctor / freeze+read-at-dtor.
// LIFO-ordered with zone_scoped: perf_ctor → zone_ctor → body → zone_dtor → perf_dtor.
// ============================================================================

#ifndef _LLK_PERF_COUNTER_SCOPED_DEFINED_
#define _LLK_PERF_COUNTER_SCOPED_DEFINED_

struct perf_counter_scoped
{
    std::uint32_t zone_id;

    perf_counter_scoped(const perf_counter_scoped&)            = delete;
    perf_counter_scoped(perf_counter_scoped&&)                 = delete;
    perf_counter_scoped& operator=(const perf_counter_scoped&) = delete;
    perf_counter_scoped& operator=(perf_counter_scoped&&)      = delete;

    inline __attribute__((always_inline)) explicit perf_counter_scoped(std::uint32_t zid) : zone_id(zid)
    {
        asm volatile("" ::: "memory");
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
        asm volatile("" ::: "memory");
        *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(0xFFB1203Cu) = 2u; // PERF_CNT_ALL
        *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(0xFFB1203Cu) = 0u;
        *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(0xFFB12014u) = 2u; // TDMA_UNPACK
        *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(0xFFB12014u) = 0u;
        *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(0xFFB12038u) = 2u; // L1
        *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(0xFFB12038u) = 0u;
        *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(0xFFB120F8u) = 2u; // TDMA_PACK
        *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(0xFFB120F8u) = 0u;

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

        std::uint32_t cycles_base              = PERF_COUNTERS_ZONES_BASE + zone_id * PERF_COUNTERS_ZONE_SIZE;
        volatile std::uint32_t* bank_cycles    = reinterpret_cast<volatile std::uint32_t*>(cycles_base);
        volatile std::uint32_t* counter_counts = bank_cycles + PERF_COUNTERS_BANK_CYCLES_WORDS;

        // INSTRN OUT_L replicated to all banks: FPU/L1 OUT_L return 0 on 2nd+ zone
        // when counter_sel is high (HW quirk); INSTRN cycles agree within ±30 cyc.
        std::uint32_t shared_cycles = *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(banks[0].out_l);
        bank_cycles[0]              = shared_cycles;
        bank_cycles[1]              = shared_cycles;
        bank_cycles[2]              = shared_cycles;
        bank_cycles[3]              = shared_cycles;
        bank_cycles[4]              = shared_cycles;

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

        std::uint32_t sync_addr                               = PERF_COUNTERS_ZONES_BASE + zone_id * PERF_COUNTERS_ZONE_SIZE + PERF_COUNTERS_ZONE_DATA_BYTES;
        *reinterpret_cast<volatile std::uint32_t*>(sync_addr) = SYNC_ZONE_COMPLETE;
        asm volatile("" ::: "memory");
    }
};
#endif // _LLK_PERF_COUNTER_SCOPED_DEFINED_

} // namespace llk_perf

#ifndef PERF_COUNTER_VAR_CONCAT_
#undef MEASURE_PERF_COUNTERS
#define PERF_COUNTER_VAR_CONCAT_(a, b)   a##b
#define PERF_COUNTER_VAR_(line)          PERF_COUNTER_VAR_CONCAT_(_perf_ctr_, line)
#define MEASURE_PERF_COUNTERS(zone_name) \
    const llk_perf::perf_counter_scoped PERF_COUNTER_VAR_(__LINE__)(llk_perf::get_zone_id(llk_perf::detail::zone_name_hash(zone_name)));
#endif
