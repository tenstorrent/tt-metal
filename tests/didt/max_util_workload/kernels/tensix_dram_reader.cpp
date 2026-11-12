// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Tensix DRAM streaming kernel (NCRISC). Every worker reads from all DRAM banks
// in rotation into a private L1 staging slot, for a fixed wall-clock duration,
// so DRAM is loaded even on boards without active Ethernet links.
//
// Throttling mirrors the syseng ttx NOC power virus knobs:
//   burst      – transfers per burst; 0 = continuous (no idle period)
//   idle_units – idle time after each burst, in units of 512 wall-clock ticks
//
// Only the low half of the wall clock is read: reading WALL_CLOCK_L latches
// WALL_CLOCK_H, which TRISC's 64-bit FPU timing reads also rely on.
//
// Compile-time args:
//   0: num_banks        – number of DRAM banks
//   1: pages_per_bank   – distinct pages per bank to cycle through
//   2: xfer_bytes       – bytes per NOC read (<= NOC max burst)
//   3: burst            – transfers per burst (0 = continuous)
//   4: idle_units       – idle after burst, units of 512 wall-clock ticks
//   5: duration_ticks_lo – run duration in wall-clock ticks (low 32 bits)
//   6: duration_ticks_hi – run duration in wall-clock ticks (high 32 bits)
//   7: period_units     – phase-locked mode period (units of 512 ticks, power
//                         of two); 0 = off. When on, burst/idle are ignored and
//                         every core issues reads only while
//                         (wall_clock / 512) % period < active_units, so all
//                         workers toggle DRAM traffic in lock-step.
//   8: active_units     – active part of each period (units of 512 ticks)
//
// Runtime args:
//   0: dram_src_addr  – interleaved DRAM buffer base address
//   1: l1_staging_addr – private L1 destination (xfer_bytes)
//   2: l1_timing_addr – 16 B result slot: [elapsed_lo, elapsed_hi, xfers_lo, xfers_hi]
//   3: start_bank     – first bank for this core (spreads load across banks)

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

namespace {

constexpr uint32_t kIdleUnitTicks = 512;
constexpr uint32_t kMaxBanks = 16;
constexpr uint32_t kBarrierInterval = 64;

FORCE_INLINE uint32_t wall_clock_l() {
    return *reinterpret_cast<volatile uint32_t tt_reg_ptr*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
}

class Stopwatch {
public:
    Stopwatch() : prev_(wall_clock_l()) {}
    uint64_t elapsed() {
        const uint32_t t = wall_clock_l();
        elapsed_ += t - prev_;
        prev_ = t;
        return elapsed_;
    }

private:
    uint32_t prev_;
    uint64_t elapsed_ = 0;
};

}  // namespace

void kernel_main() {
    constexpr uint32_t num_banks = get_compile_time_arg_val(0);
    constexpr uint32_t pages_per_bank = get_compile_time_arg_val(1);
    constexpr uint32_t xfer_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t burst = get_compile_time_arg_val(3);
    constexpr uint32_t idle_units = get_compile_time_arg_val(4);
    constexpr uint64_t duration_ticks =
        (static_cast<uint64_t>(get_compile_time_arg_val(6)) << 32) | get_compile_time_arg_val(5);
    constexpr uint32_t period_units = get_compile_time_arg_val(7);
    constexpr uint32_t active_units = get_compile_time_arg_val(8);
    constexpr bool phased = period_units != 0;
    static_assert(!phased || (period_units & (period_units - 1)) == 0, "period_units must be a power of two");
    // Phase-locked mode keeps few reads outstanding so traffic stops promptly
    // when the active window closes.
    constexpr uint32_t batch = phased ? 1 : (burst == 0 ? kBarrierInterval : burst);
    constexpr uint32_t barrier_interval = phased ? 4 : kBarrierInterval;
    static_assert(num_banks > 0 && num_banks <= kMaxBanks, "unsupported DRAM bank count");
    static_assert(pages_per_bank > 0, "pages_per_bank must be non-zero");

    const uint32_t dram_src_addr = get_arg_val<uint32_t>(0);
    const uint32_t l1_staging_addr = get_arg_val<uint32_t>(1);
    const uint32_t l1_timing_addr = get_arg_val<uint32_t>(2);
    const uint32_t start_bank = get_arg_val<uint32_t>(3);

    uint64_t bank_base[num_banks];
    for (uint32_t b = 0; b < num_banks; b++) {
        bank_base[b] = get_noc_addr_from_bank_id<true>(b, dram_src_addr);
    }

    uint32_t bank = start_bank % num_banks;
    uint32_t page = 0;
    uint64_t xfers = 0;
    uint32_t since_barrier = 0;

    Stopwatch stopwatch;
    uint64_t elapsed = 0;
    while (elapsed < duration_ticks) {
        if constexpr (phased) {
            // 2^32 / 512 is a multiple of period_units, so the low clock word
            // gives a consistent phase across wrap-around.
            if (((wall_clock_l() / kIdleUnitTicks) & (period_units - 1)) >= active_units) {
                noc_async_read_barrier();
                since_barrier = 0;
                elapsed = stopwatch.elapsed();
                continue;
            }
        }
        constexpr uint32_t n = batch;
        for (uint32_t i = 0; i < n; i++) {
            noc_async_read_one_packet(bank_base[bank] + page * xfer_bytes, l1_staging_addr, xfer_bytes);
            if (++bank == num_banks) {
                bank = 0;
                if (++page == pages_per_bank) {
                    page = 0;
                }
            }
            if (++since_barrier == barrier_interval) {
                noc_async_read_barrier();
                since_barrier = 0;
            }
        }
        xfers += n;

        if constexpr (!phased && burst != 0 && idle_units != 0) {
            noc_async_read_barrier();
            since_barrier = 0;
            const uint64_t idle_end = stopwatch.elapsed() + static_cast<uint64_t>(idle_units) * kIdleUnitTicks;
            while (stopwatch.elapsed() < idle_end) {
            }
        }
        elapsed = stopwatch.elapsed();
    }
    noc_async_read_barrier();
    elapsed = stopwatch.elapsed();

    volatile tt_l1_ptr uint32_t* timing = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_timing_addr);
    timing[0] = static_cast<uint32_t>(elapsed);
    timing[1] = static_cast<uint32_t>(elapsed >> 32);
    timing[2] = static_cast<uint32_t>(xfers);
    timing[3] = static_cast<uint32_t>(xfers >> 32);
}
