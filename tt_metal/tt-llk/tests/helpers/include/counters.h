// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "perf.h" // PerfRunType lives here

#ifdef PERF_COUNTERS_COMPILED

#include "ckernel.h"

// Quasar gets a 4th compute thread (SFPU) on top of unpack/math/pack, and the
// arm/freeze barrier below assumes exactly 3 threads (1 arm + 2 followers). The
// build system is supposed to keep `-DPERF_COUNTERS_COMPILED` off for Quasar
// (see `test_config.py::build_kernel_part`); this guard catches a silent
// regression of that gate.
#ifdef ARCH_QUASAR
#error \
    "Perf counters do not support Quasar yet: the entry/exit barrier hardcodes a 3-thread post count and `is_arm_thread`/`is_freeze_thread` have no LLK_TRISC_ISOLATE_SFPU path. Re-enabling for Quasar requires parameterizing PERF_NUM_SPINWAITERS by the active thread count."
#endif

// L1 region constants (PERF_COUNTERS_BASE_ADDR and friends) live in perf.h next to the
// stimuli buffer addresses so the disjoint L1 ranges are visible side-by-side.

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

// sync_ctrl block: +0 SYNC_ZONE_COMPLETE flag, +16 entry atomic, +32 exit atomic (ATINCGET-aligned).
constexpr std::uint32_t perf_counters_entry_atomic_addr(std::uint32_t zone)
{
    return perf_counters_sync_ctrl_addr(zone) + 16;
}

constexpr std::uint32_t perf_counters_exit_atomic_addr(std::uint32_t zone)
{
    return perf_counters_sync_ctrl_addr(zone) + 32;
}

constexpr std::uint32_t PERF_COUNTERS_ENABLED_FLAG_ADDR = PERF_COUNTERS_ZONES_BASE + PERF_COUNTERS_MAX_ZONES * PERF_COUNTERS_ZONE_SIZE;
constexpr std::uint32_t PERF_COUNTERS_BANK_MASK_ADDR    = PERF_COUNTERS_ENABLED_FLAG_ADDR + 4;
constexpr std::uint32_t PERF_COUNTERS_VALID_COUNT_ADDR  = PERF_COUNTERS_BANK_MASK_ADDR + 4;

// Total L1 footprint covers shared config + per-zone blocks + the trailing
// metadata (enabled flag, bank mask, per-zone valid_count[]) written by
// PerfCounterManager::validate_and_set_enabled().
constexpr std::uint32_t PERF_COUNTERS_LAYOUT_END = PERF_COUNTERS_VALID_COUNT_ADDR + PERF_COUNTERS_MAX_ZONES * 4;

static_assert(PERF_COUNTERS_LAYOUT_END <= 0x16AFF4u, "Perf counter L1 layout overflows profiler region");

} // namespace llk_perf

// BRISC entry points.

namespace llk_perf
{

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

// Config word layout (32-bit). Built by _perf_cfg(); decoded by configure_hardware
// and freeze_and_read_all_counters; mirrored on the host side in counters.py.
constexpr std::uint32_t PERF_CFG_VALID_BIT     = 1u << 31; // bit 31: slot active
constexpr std::uint32_t PERF_CFG_L1_MUX_SHIFT  = 17;       // bits 19:17
constexpr std::uint32_t PERF_CFG_L1_MUX_MASK   = 0x7u;
constexpr std::uint32_t PERF_CFG_COUNTER_SHIFT = 8; // bits 16:8 (9-bit counter_sel)
constexpr std::uint32_t PERF_CFG_COUNTER_MASK  = 0x1FFu;
constexpr std::uint32_t PERF_CFG_BANK_MASK     = 0xFFu; // bits 7:0

// L1 mux position inside PERF_CNT_MUX_CTRL (bits 6:4).
constexpr std::uint32_t PERF_CNT_MUX_CTRL_SHIFT = 4;

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

// Volatile index cast prevents GCC CSWTCH (would shift GP-offsets and break NC/WC bit-identity).
inline std::uint32_t get_counter_base_addr(counter_bank bank)
{
    static constexpr std::uint32_t base_addrs[COUNTER_BANK_COUNT] = {
        RISCV_DEBUG_REG_PERF_CNT_INSTRN_THREAD0,
        RISCV_DEBUG_REG_PERF_CNT_FPU0,
        RISCV_DEBUG_REG_PERF_CNT_TDMA_UNPACK0,
        RISCV_DEBUG_REG_PERF_CNT_L1_0,
        RISCV_DEBUG_REG_PERF_CNT_TDMA_PACK0,
    };
    // Guarantees every valid `counter_bank` enumerator is a safe index into base_addrs[],
    // so the runtime check below is purely defensive against future enum growth.
    static_assert(
        static_cast<std::uint32_t>(counter_bank::tdma_pack) == COUNTER_BANK_COUNT - 1, "counter_bank enumerators must be contiguous 0..COUNTER_BANK_COUNT-1");
    volatile auto b = static_cast<std::uint32_t>(bank);
    return b < COUNTER_BANK_COUNT ? base_addrs[b] : 0u;
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
            if ((metadata & PERF_CFG_VALID_BIT) == 0)
            {
                continue;
            }
            const std::uint8_t bank_id   = static_cast<std::uint8_t>(metadata & PERF_CFG_BANK_MASK);
            const std::uint32_t bank_bit = 1u << bank_id;
            if (configured_mask & bank_bit)
            {
                continue;
            }
            const counter_bank bank = static_cast<counter_bank>(bank_id);
            if (bank == counter_bank::l1)
            {
                const std::uint8_t l1_mux = (metadata >> PERF_CFG_L1_MUX_SHIFT) & PERF_CFG_L1_MUX_MASK;
                std::uint32_t cur         = hw_access::read_reg(RISCV_DEBUG_REG_PERF_CNT_MUX_CTRL);
                hw_access::write_reg(
                    RISCV_DEBUG_REG_PERF_CNT_MUX_CTRL, (cur & ~(PERF_CFG_L1_MUX_MASK << PERF_CNT_MUX_CTRL_SHIFT)) | (l1_mux << PERF_CNT_MUX_CTRL_SHIFT));
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
                if (metadata & PERF_CFG_VALID_BIT)
                {
                    found_valid = true;
                    count++;
                    bank_mask |= (1u << (metadata & PERF_CFG_BANK_MASK));
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

// Per-arch built-in counter inventory. Config word layout matches the PERF_CFG_* constants above.

constexpr std::uint32_t _perf_cfg(std::uint8_t bank, std::uint16_t cid, std::uint8_t mux = 0)
{
    return PERF_CFG_VALID_BIT | (static_cast<std::uint32_t>(mux & PERF_CFG_L1_MUX_MASK) << PERF_CFG_L1_MUX_SHIFT) |
           (static_cast<std::uint32_t>(cid & PERF_CFG_COUNTER_MASK) << PERF_CFG_COUNTER_SHIFT) | static_cast<std::uint32_t>(bank);
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

// String-only API: zone name → 32-bit DJB2 → sequential zone id (0..MAX_ZONES-1).

namespace llk_perf
{
namespace detail
{
// `.bss.perf_counters` is NOBITS (the `.bss.*` prefix is honored by GCC) and is
// absorbed by the catch-all `*(.bss .bss.* ...)` line in sections.ld, so these
// symbols live inside [__ldm_bss_start, __ldm_bss_end) and are zero-initialized
// by do_crt0() on every cold start. The named section also lets the linker
// script document the WC-only zone allocator and keeps these symbols grouped
// for objdump inspection.
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

// perf_counter_scoped: RAII arm/freeze. LIFO with zone_scoped: perf_ctor → zone_ctor → body → zone_dtor → perf_dtor.

#ifndef _LLK_PERF_COUNTER_SCOPED_DEFINED_
#define _LLK_PERF_COUNTER_SCOPED_DEFINED_

// PERF_CNT_ALL is a broadcast command register shared by INSTRN+FPU; TDMA_UNPACK,
// L1 and TDMA_PACK each have their own command register at the *2 alias. Writing
// 1 arms (rising-edge start), 2 freezes (rising-edge stop).
inline __attribute__((always_inline)) void arm_all_counters()
{
    ckernel::fence_compiler();
    hw_access::write_reg(RISCV_DEBUG_REG_PERF_CNT_ALL, 1u);
    hw_access::write_reg(RISCV_DEBUG_REG_PERF_CNT_TDMA_UNPACK2, 1u);
    hw_access::write_reg(RISCV_DEBUG_REG_PERF_CNT_L1_2, 1u);
    hw_access::write_reg(RISCV_DEBUG_REG_PERF_CNT_TDMA_PACK2, 1u);
    ckernel::fence_compiler();
}

inline __attribute__((always_inline)) void freeze_and_read_all_counters(std::uint32_t zone_id)
{
    ckernel::fence_compiler();
    hw_access::write_reg(RISCV_DEBUG_REG_PERF_CNT_ALL, 2u);
    hw_access::write_reg(RISCV_DEBUG_REG_PERF_CNT_TDMA_UNPACK2, 2u);
    hw_access::write_reg(RISCV_DEBUG_REG_PERF_CNT_L1_2, 2u);
    hw_access::write_reg(RISCV_DEBUG_REG_PERF_CNT_TDMA_PACK2, 2u);

    struct bank_regs
    {
        std::uint32_t mode_reg;
        std::uint32_t out_l;
    };

    // Per-bank readout pair: mode_reg drives counter_sel; out_l is the bank's
    // OUT_L (shared cycles); OUT_H sits at out_l + 4 and is sampled per slot.
    static constexpr bank_regs banks[5] = {
        {RISCV_DEBUG_REG_PERF_CNT_INSTRN_THREAD1, RISCV_DEBUG_REG_PERF_CNT_OUT_L_INSTRN_THREAD},
        {RISCV_DEBUG_REG_PERF_CNT_FPU1, RISCV_DEBUG_REG_PERF_CNT_OUT_L_FPU},
        {RISCV_DEBUG_REG_PERF_CNT_TDMA_UNPACK1, RISCV_DEBUG_REG_PERF_CNT_OUT_L_TDMA_UNPACK},
        {RISCV_DEBUG_REG_PERF_CNT_L1_1, RISCV_DEBUG_REG_PERF_CNT_OUT_L_DBG_L1},
        {RISCV_DEBUG_REG_PERF_CNT_TDMA_PACK1, RISCV_DEBUG_REG_PERF_CNT_OUT_L_TDMA_PACK},
    };

    std::uint32_t cycles_base              = PERF_COUNTERS_ZONES_BASE + zone_id * PERF_COUNTERS_ZONE_SIZE;
    volatile std::uint32_t* bank_cycles    = reinterpret_cast<volatile std::uint32_t*>(cycles_base);
    volatile std::uint32_t* counter_counts = bank_cycles + PERF_COUNTERS_BANK_CYCLES_WORDS;
    std::uint32_t shared_cycles            = *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(banks[0].out_l);
    bank_cycles[0]                         = shared_cycles;
    bank_cycles[1]                         = shared_cycles;
    bank_cycles[2]                         = shared_cycles;
    bank_cycles[3]                         = shared_cycles;
    bank_cycles[4]                         = shared_cycles;

    const volatile std::uint32_t* cfg = reinterpret_cast<volatile std::uint32_t*>(PERF_COUNTERS_SHARED_CONFIG_ADDR);
    std::uint32_t out_idx             = 0;
#pragma GCC unroll 0
    for (std::uint32_t i = 0; i < PERF_COUNTERS_CONFIG_WORDS; ++i)
    {
        std::uint32_t cw = cfg[i];
        if (!(cw & PERF_CFG_VALID_BIT))
        {
            continue;
        }
        std::uint32_t bank_id    = cw & PERF_CFG_BANK_MASK;
        std::uint32_t counter_id = (cw >> PERF_CFG_COUNTER_SHIFT) & PERF_CFG_COUNTER_MASK;
        std::uint32_t l1_mux     = (cw >> PERF_CFG_L1_MUX_SHIFT) & PERF_CFG_L1_MUX_MASK;
        const bank_regs& br      = banks[bank_id];
        if (bank_id == static_cast<std::uint32_t>(counter_bank::l1))
        {
            volatile std::uint32_t tt_reg_ptr* mux = reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(RISCV_DEBUG_REG_PERF_CNT_MUX_CTRL);
            *mux                                   = (*mux & ~(PERF_CFG_L1_MUX_MASK << PERF_CNT_MUX_CTRL_SHIFT)) | (l1_mux << PERF_CNT_MUX_CTRL_SHIFT);
        }
        *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(br.mode_reg) = counter_id << PERF_CFG_COUNTER_SHIFT;
        counter_counts[out_idx]                                            = *reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(br.out_l + 4u);
        ++out_idx;
    }

    std::uint32_t sync_addr                               = perf_counters_sync_ctrl_addr(zone_id);
    *reinterpret_cast<volatile std::uint32_t*>(sync_addr) = SYNC_ZONE_COMPLETE;
}

// L1_TO_L1/L1_CONGESTION: unpack arms (pipeline source), pack freezes (sink). ISOLATE: same thread arms and freezes.
template <PerfRunType run_type>
constexpr bool is_arm_thread()
{
#if defined(LLK_TRISC_UNPACK)
    return run_type == PerfRunType::L1_TO_L1 || run_type == PerfRunType::L1_CONGESTION || run_type == PerfRunType::UNPACK_ISOLATE;
#elif defined(LLK_TRISC_MATH)
    return run_type == PerfRunType::MATH_ISOLATE;
#elif defined(LLK_TRISC_PACK)
    return run_type == PerfRunType::PACK_ISOLATE;
#else
    return false;
#endif
}

template <PerfRunType run_type>
constexpr bool is_freeze_thread()
{
#if defined(LLK_TRISC_UNPACK)
    return run_type == PerfRunType::UNPACK_ISOLATE;
#elif defined(LLK_TRISC_MATH)
    return run_type == PerfRunType::MATH_ISOLATE;
#elif defined(LLK_TRISC_PACK)
    return run_type == PerfRunType::L1_TO_L1 || run_type == PerfRunType::L1_CONGESTION || run_type == PerfRunType::PACK_ISOLATE;
#else
    return false;
#endif
}

// pc_buf-semaphore barriers: arm/freeze thread sempost ×N, non-active threads spinwait+semget.
constexpr std::uint8_t PERF_ENTRY_SEM        = ckernel::semaphore::FPU_SFPU;
constexpr std::uint8_t PERF_EXIT_SEM         = ckernel::semaphore::UNPACK_TO_DEST;
constexpr std::uint32_t PERF_NUM_SPINWAITERS = 2;

template <PerfRunType run_type>
struct perf_counter_scoped
{
    std::uint32_t zone_id;

    perf_counter_scoped(const perf_counter_scoped&)            = delete;
    perf_counter_scoped(perf_counter_scoped&&)                 = delete;
    perf_counter_scoped& operator=(const perf_counter_scoped&) = delete;
    perf_counter_scoped& operator=(perf_counter_scoped&&)      = delete;

    inline __attribute__((always_inline)) explicit perf_counter_scoped(std::uint32_t zid) : zone_id(zid)
    {
        ckernel::fence_compiler();
        if constexpr (is_arm_thread<run_type>())
        {
            arm_all_counters();
            for (std::uint32_t i = 0; i < PERF_NUM_SPINWAITERS; ++i)
            {
                ckernel::semaphore_post(PERF_ENTRY_SEM);
            }
        }
        else
        {
            while (ckernel::semaphore_read(PERF_ENTRY_SEM) == 0)
            {
                asm volatile("nop");
            }
            ckernel::semaphore_get(PERF_ENTRY_SEM);
        }
        ckernel::fence_compiler();
    }

    inline __attribute__((always_inline)) ~perf_counter_scoped()
    {
        ckernel::fence_compiler();
        if constexpr (is_freeze_thread<run_type>())
        {
            freeze_and_read_all_counters(zone_id);
            for (std::uint32_t i = 0; i < PERF_NUM_SPINWAITERS; ++i)
            {
                ckernel::semaphore_post(PERF_EXIT_SEM);
            }
        }
        else
        {
            while (ckernel::semaphore_read(PERF_EXIT_SEM) == 0)
            {
                asm volatile("nop");
            }
            ckernel::semaphore_get(PERF_EXIT_SEM);
        }
        ckernel::fence_compiler();
    }
};
#endif // _LLK_PERF_COUNTER_SCOPED_DEFINED_

} // namespace llk_perf

#define PERF_COUNTER_VAR_CONCAT_(a, b)   a##b
#define PERF_COUNTER_VAR_(line)          PERF_COUNTER_VAR_CONCAT_(_perf_ctr_, line)
#define MEASURE_PERF_COUNTERS(zone_name) \
    const llk_perf::perf_counter_scoped<PERF_RUN_TYPE> PERF_COUNTER_VAR_(__LINE__)(llk_perf::get_zone_id(llk_perf::detail::zone_name_hash(zone_name)));

#else // !PERF_COUNTERS_COMPILED

#define MEASURE_PERF_COUNTERS(zone_name)

// NC build stubs — let callers invoke unconditionally; compiler folds these to nothing.
namespace llk_perf
{
inline void configure_and_arm_from_brisc()
{
}
} // namespace llk_perf

#endif // PERF_COUNTERS_COMPILED

// Convenience macro for the conventional pairing at the top of a measured scope.
// Expands to MEASURE_PERF_COUNTERS + ZONE_SCOPED with the same `name`, so the
// counter zone and profiler marker stay in sync. One of the two halves is empty
// in any given build, so this never emits both timing paths at once.
#define START_PERF_MEASURE(zone_name) \
    MEASURE_PERF_COUNTERS(zone_name)  \
    ZONE_SCOPED(zone_name)
