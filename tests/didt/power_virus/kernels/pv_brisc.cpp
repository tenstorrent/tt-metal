// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Power-virus BRISC kernel: optional super-sync barrier, then optional DRAM
// reads until the host sets STOP.
//
// Super-sync (from tests/didt/max_util_workload/kernels/super_sync.cpp): every
// core reports ready to the sender, the sender multicasts the release, and each
// receiver waits 220 - 9 × (hops from the sender) cycles so they all start
// together. The barrier then sets GO, which the TRISC, NCRISC and DRAM loops
// wait for, so every generator on the chip starts in the same window.
//
// DRAM: reads round-robin across all DRAM banks into a private L1 staging slot.
// It runs on this core's BRISC NOC, which the host sets opposite to NCRISC's.
//
// Throttle: see pv_common.hpp. Smooth mode times each batch of reads (to
// completion) and then idles in proportion; burst counts reads.
//
// Compile-time args:
//   0: ctrl_base
//   1: super_sync
//   2: wait_for_trisc   – barrier waits for TRISC READY (FPU on)
//   3: sender_sem_id
//   4: receiver_sem_id
//   5: dram_on
//   6: num_banks
//   7: pages_per_bank
//   8: xfer_bytes
//   9: mode             – pv::kMode*
//  10: a
//  11: b
//  12: duty_period      – 0 = no duty cycle
//  13: duty_active
//
// Runtime args:
//   0: is_sender
//   1: sender x          (virtual)
//   2: sender y
//   3: mcast start x     (ordered for this kernel's NOC)
//   4: mcast start y
//   5: mcast end x
//   6: mcast end y
//   7: num_dests
//   8: own x             (virtual)
//   9: own y
//  10: dram_src_addr
//  11: l1_staging_addr
//  12: start_bank

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "tt_metal/tt-llk/tt_llk_blackhole/common/inc/ckernel.h"
#include "pv_common.hpp"

namespace {

constexpr uint32_t kMaxBanks = 16;
constexpr uint32_t kBarrierInterval = 64;

// Legacy Blackhole launch-alignment values from the MaxUtil super-sync.
constexpr uint32_t kReceiverDelayCyclesPerHop = 9;
constexpr uint32_t kReceiverBaseDelayCycles = 220;
constexpr uint32_t kSenderPostMulticastDelayCycles = 600;

// Only the low half of the wall clock is read: reading WALL_CLOCK_L latches
// WALL_CLOCK_H, which other RISCs' 64-bit reads also rely on.
FORCE_INLINE uint32_t wall_clock_l() {
    return *reinterpret_cast<volatile uint32_t tt_reg_ptr*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
}

FORCE_INLINE volatile tt_l1_ptr uint32_t* ctrl_word(uint32_t base, uint32_t offset) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + offset);
}

FORCE_INLINE uint32_t abs_diff(uint32_t a, uint32_t b) { return a > b ? a - b : b - a; }

void super_sync_barrier(uint32_t ctrl_base, bool wait_for_trisc, uint32_t sender_sem_id, uint32_t receiver_sem_id) {
    const uint32_t is_sender = get_arg_val<uint32_t>(0);
    const uint32_t sender_x = get_arg_val<uint32_t>(1);
    const uint32_t sender_y = get_arg_val<uint32_t>(2);
    const uint32_t mcast_start_x = get_arg_val<uint32_t>(3);
    const uint32_t mcast_start_y = get_arg_val<uint32_t>(4);
    const uint32_t mcast_end_x = get_arg_val<uint32_t>(5);
    const uint32_t mcast_end_y = get_arg_val<uint32_t>(6);
    const uint32_t num_dests = get_arg_val<uint32_t>(7);
    const uint32_t own_x = get_arg_val<uint32_t>(8);
    const uint32_t own_y = get_arg_val<uint32_t>(9);

    // Farther cores wait less to compensate for multicast propagation delay.
    const uint32_t hops = abs_diff(own_x, sender_x) + abs_diff(own_y, sender_y);
    const uint32_t compensation = hops * kReceiverDelayCyclesPerHop;
    const uint32_t cycles_to_wait =
        compensation < kReceiverBaseDelayCycles ? kReceiverBaseDelayCycles - compensation : 0;

    const uint32_t sender_sem_addr = get_semaphore(sender_sem_id);
    const uint32_t receiver_sem_addr = get_semaphore(receiver_sem_id);
    volatile tt_l1_ptr uint32_t* sender_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sender_sem_addr);
    volatile tt_l1_ptr uint32_t* receiver_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receiver_sem_addr);
    volatile tt_l1_ptr uint32_t* ready = ctrl_word(ctrl_base, pv::kReady);

    if (wait_for_trisc) {
        do {
            invalidate_l1_cache();
        } while (*ready != 1);
    }

    if (is_sender) {
        *receiver_sem = VALID;
        noc_semaphore_wait(sender_sem, num_dests);
        noc_semaphore_set(sender_sem, 0);
        const uint64_t mcast_addr =
            get_noc_multicast_addr(mcast_start_x, mcast_start_y, mcast_end_x, mcast_end_y, receiver_sem_addr);
        noc_semaphore_set_multicast(receiver_sem_addr, mcast_addr, num_dests);
        ckernel::wait(kSenderPostMulticastDelayCycles);
    } else {
        noc_semaphore_set(receiver_sem, INVALID);
        noc_semaphore_inc(get_noc_addr(sender_x, sender_y, sender_sem_addr), 1);
        noc_semaphore_wait(receiver_sem, VALID);
        ckernel::wait(cycles_to_wait);
    }
    *ctrl_word(ctrl_base, pv::kGo) = 1;
}

template <
    uint32_t num_banks,
    uint32_t pages_per_bank,
    uint32_t xfer_bytes,
    uint32_t mode,
    uint32_t a,
    uint32_t b,
    uint32_t duty_period,
    uint32_t duty_active>
void dram_reads(uint32_t ctrl_base) {
    using T = pv::Throttle<mode, a, b, duty_period, duty_active>;
    static_assert(num_banks > 0 && num_banks <= kMaxBanks, "unsupported DRAM bank count");
    // Phase-locked and rate modes issue one read at a time, so traffic follows
    // the gate closely; smooth mode uses short batches so the load stays even.
    constexpr uint32_t batch = mode == pv::kModeFull     ? kBarrierInterval
                               : mode == pv::kModeSmooth ? 8
                               : mode == pv::kModeBurst  ? a
                                                         : 1;
    constexpr uint32_t barrier_interval = mode == pv::kModePhase ? 4 : kBarrierInterval;
    constexpr bool pace = mode == pv::kModeSmooth || mode == pv::kModeBurst;
    constexpr bool rate = mode == pv::kModeRate;
    uint64_t next_issue = 0;  // rate mode: elapsed tick of the next read

    const uint32_t dram_src_addr = get_arg_val<uint32_t>(10);
    const uint32_t l1_staging_addr = get_arg_val<uint32_t>(11);
    const uint32_t start_bank = get_arg_val<uint32_t>(12);

    uint64_t bank_base[num_banks];
    for (uint32_t bank_id = 0; bank_id < num_banks; bank_id++) {
        bank_base[bank_id] = get_noc_addr_from_bank_id<true>(bank_id, dram_src_addr);
    }

    volatile tt_l1_ptr uint32_t* stop = ctrl_word(ctrl_base, pv::kStop);
    uint32_t bank = start_bank % num_banks;
    uint32_t page = 0;
    uint64_t xfers = 0;
    uint32_t since_barrier = 0;
    uint32_t prev_clk = wall_clock_l();
    uint64_t elapsed = 0;

    while (true) {
        invalidate_l1_cache();
        if (*stop != 0) {
            break;
        }
        const uint32_t now = wall_clock_l();
        elapsed += now - prev_clk;
        prev_clk = now;
        if constexpr (T::kGated) {
            if (!T::open(now)) {
                noc_async_read_barrier();
                since_barrier = 0;
                continue;
            }
        }
        if constexpr (rate) {
            if (elapsed < next_issue) {
                continue;
            }
            // Catch up after a stall, but by at most kRateMaxCatchUp reads.
            const uint64_t floor = elapsed > pv::kRateMaxCatchUp * a ? elapsed - pv::kRateMaxCatchUp * a : 0;
            next_issue = (next_issue > floor ? next_issue : floor) + a;
        }
        for (uint32_t i = 0; i < batch; i++) {
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
        xfers += batch;

        if constexpr (pace) {
            noc_async_read_barrier();
            since_barrier = 0;
            const uint32_t idle_start = wall_clock_l();
            const uint32_t pause = T::pause_after(idle_start - now);
            while (wall_clock_l() - idle_start < pause) {
            }
        }
    }
    noc_async_read_barrier();
    elapsed += wall_clock_l() - prev_clk;

    volatile tt_l1_ptr uint32_t* record = ctrl_word(ctrl_base, pv::kDramRecord);
    record[0] = static_cast<uint32_t>(elapsed);
    record[1] = static_cast<uint32_t>(elapsed >> 32);
    record[2] = static_cast<uint32_t>(xfers);
    record[3] = static_cast<uint32_t>(xfers >> 32);
}

}  // namespace

void kernel_main() {
    constexpr uint32_t ctrl_base = get_compile_time_arg_val(0);
    constexpr bool super_sync = get_compile_time_arg_val(1) != 0;
    constexpr bool wait_for_trisc = get_compile_time_arg_val(2) != 0;
    constexpr uint32_t sender_sem_id = get_compile_time_arg_val(3);
    constexpr uint32_t receiver_sem_id = get_compile_time_arg_val(4);
    constexpr bool dram_on = get_compile_time_arg_val(5) != 0;

    if constexpr (super_sync) {
        super_sync_barrier(ctrl_base, wait_for_trisc, sender_sem_id, receiver_sem_id);
    }
    if constexpr (dram_on) {
        dram_reads<
            get_compile_time_arg_val(6),
            get_compile_time_arg_val(7),
            get_compile_time_arg_val(8),
            get_compile_time_arg_val(9),
            get_compile_time_arg_val(10),
            get_compile_time_arg_val(11),
            get_compile_time_arg_val(12),
            get_compile_time_arg_val(13)>(ctrl_base);
    }
}
