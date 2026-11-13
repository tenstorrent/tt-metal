// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Power-virus NOC kernel (NCRISC). Cores that have a route keep issuing posted
// writes of their L1 buffer to the same address on the destination core. The
// host computes the routes (noc_virus_routes.hpp): one transfer per Tensix row
// and per column on each NOC, each going to the adjacent live column/row
// upstream so it travels almost the whole ring.
//
// Throttle: see pv_common.hpp. Smooth mode times each transfer until it has
// left the NIU and then idles in proportion; burst counts transfers.
//
// Compile-time args:
//   0: ctrl_base
//   1: super_sync     – wait for GO from the BRISC barrier
//   2: xfer_bytes
//   3: mode           – pv::kMode*
//   4: a
//   5: b
//   6: duty_period    – 0 = no duty cycle
//   7: duty_active
//
// Runtime args:
//   0: buffer_addr    – L1 source; also the destination address on the peer
//   1: dst_x          (virtual)
//   2: dst_y

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "pv_common.hpp"

namespace {

constexpr uint32_t kFlushInterval = 16;

FORCE_INLINE uint32_t wall_clock_l() {
    return *reinterpret_cast<volatile uint32_t tt_reg_ptr*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
}

FORCE_INLINE volatile tt_l1_ptr uint32_t* ctrl_word(uint32_t base, uint32_t offset) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + offset);
}

}  // namespace

void kernel_main() {
    constexpr uint32_t ctrl_base = get_compile_time_arg_val(0);
    constexpr bool super_sync = get_compile_time_arg_val(1) != 0;
    constexpr uint32_t xfer_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t mode = get_compile_time_arg_val(3);
    constexpr uint32_t a = get_compile_time_arg_val(4);
    constexpr uint32_t b = get_compile_time_arg_val(5);
    constexpr uint32_t duty_period = get_compile_time_arg_val(6);
    constexpr uint32_t duty_active = get_compile_time_arg_val(7);
    using T = pv::Throttle<mode, a, b, duty_period, duty_active>;
    constexpr uint32_t batch = mode == pv::kModeFull ? kFlushInterval : mode == pv::kModeBurst ? a : 1;
    constexpr bool pace = mode == pv::kModeSmooth || mode == pv::kModeBurst;

    const uint32_t buffer_addr = get_arg_val<uint32_t>(0);
    const uint32_t dst_x = get_arg_val<uint32_t>(1);
    const uint32_t dst_y = get_arg_val<uint32_t>(2);
    const uint64_t dst = get_noc_addr(dst_x, dst_y, buffer_addr);

    volatile tt_l1_ptr uint32_t* stop = ctrl_word(ctrl_base, pv::kStop);
    if constexpr (super_sync) {
        volatile tt_l1_ptr uint32_t* go = ctrl_word(ctrl_base, pv::kGo);
        do {
            invalidate_l1_cache();
        } while (*go != 1 && *stop == 0);
    }

    uint64_t xfers = 0;
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
                continue;
            }
        }
        for (uint32_t i = 0; i < batch; i++) {
            // Posted writes: no acks, so the NOC carries only payload.
            noc_async_write<NOC_MAX_BURST_SIZE + 1, false, true>(buffer_addr, dst, xfer_bytes);
        }
        noc_async_posted_writes_flushed();
        xfers += batch;

        if constexpr (pace) {
            const uint32_t idle_start = wall_clock_l();
            const uint32_t pause = T::pause_after(idle_start - now);
            while (wall_clock_l() - idle_start < pause) {
            }
        }
    }
    noc_async_posted_writes_flushed();
    elapsed += wall_clock_l() - prev_clk;

    volatile tt_l1_ptr uint32_t* record = ctrl_word(ctrl_base, pv::kNocRecord);
    record[0] = static_cast<uint32_t>(elapsed);
    record[1] = static_cast<uint32_t>(elapsed >> 32);
    record[2] = static_cast<uint32_t>(xfers);
    record[3] = static_cast<uint32_t>(xfers >> 32);
}
