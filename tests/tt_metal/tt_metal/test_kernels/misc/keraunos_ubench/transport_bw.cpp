// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Transport bandwidth from one Keraunos CCE hart (IDLE_ETH). H harts of one core run this kernel together, each on its
// own source and destination slice, released at once by the start gate.
//
// Per hart: W warm-up transfers (then drain), the gate, then the timed phase:
//   t0; N transfers of SIZE bytes; t_inj; wait until every transfer is acked (CPU: fence); t_drain.
//   INJECTION = t_inj - t0, DRAINED = t_drain - t0. Bandwidth = N * SIZE / interval, per hart; the host reports the run
//   rate over the slowest hart's interval, never by subtracting timestamps of different harts.
//
// Mechanisms (TransportMech):
//   IdmaMetal   cce_noc::copy() per transfer: reset and program the whole command buffer, issue, wait for space.
//   IdmaLean    the command buffer programmed once (as cce_noc::copy() does); per transfer: source, destination,
//               length, issue, wait for space.
//   CpuStore64  SIZE/8 64-bit stores of this hart's pattern through the destination's CPU address.
// Every transfer of a run goes to the same destination slice; the host checks the slice afterwards.
//
// Runtime args:
//   0 base, 1 hart, 2 harts in the run, 3 mechanism, 4 size, 5 transfers N, 6 warm-up W, 7 polls per wait, 8 epoch,
//   9-10 source iDMA address, 11-12 destination iDMA address, 13-14 destination CPU address.
//
// Result: cycles[kInjection], cycles[kDrained], cycles[kGateWait].

#include <cstdint>

#include "internal/tt-2xx/quasar/cce_noc.h"
#include "tests/tt_metal/tt_metal/test_kernels/misc/keraunos_ubench/keraunos_ubench.h"

using namespace keraunos_ubench;
namespace tp = keraunos_ubench::transport;
namespace cce_noc = internal_::cce_noc;

namespace {

constexpr overlay::CmdBuf kCmdBuf = cce_noc::kCmdBuf;

// What cce_noc::copy() programs, once.
void lean_setup(uint32_t len) {
    overlay::reset_cmdbuf<kCmdBuf>();
    overlay::idma_setup_as_copy_cmdbuf<kCmdBuf>();
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        kCmdBuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_REQ_VC_REG_OFFSET / 8, cce_noc::kReqVc);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        kCmdBuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_RESP_VC_REG_OFFSET / 8, cce_noc::kRespVc);
    overlay::setup_trids_cmdbuf<kCmdBuf>(cce_noc::kTrid);
    overlay::set_len_cmdbuf<kCmdBuf>(len);
}

inline __attribute__((always_inline)) void lean_issue(uint64_t src, uint64_t dst, uint32_t len) {
    overlay::set_src_cmdbuf<kCmdBuf>(src, 0, 0, 0);
    overlay::set_dest_cmdbuf<kCmdBuf>(dst, 0, 0, 0);
    overlay::set_len_cmdbuf<kCmdBuf>(len);
    overlay::issue_cmdbuf<kCmdBuf>();
    while (overlay::idma_free_space_cmdbuf<kCmdBuf>() == 0) {
    }
}

inline __attribute__((always_inline)) void cpu_store(volatile uint64_t* dst, uint32_t words64, uint32_t hart) {
    for (uint32_t j = 0; j < words64; j++) {
        const uint64_t lo = pattern_word(hart, 2 * j);
        const uint64_t hi = pattern_word(hart, 2 * j + 1);
        dst[j] = lo | (hi << 32);
    }
}

bool drain(uint32_t polls) {
    for (uint32_t p = 0; p < polls; p++) {
        if (cce_noc::copies_done()) {
            return true;
        }
    }
    return cce_noc::copies_done();
}

}  // namespace

void kernel_main() {
    const uint32_t base = get_arg_val<uint32_t>(0);
    const uint32_t hart = get_arg_val<uint32_t>(1);
    const uint32_t num_harts = get_arg_val<uint32_t>(2);
    const auto mech = static_cast<TransportMech>(get_arg_val<uint32_t>(3));
    const uint32_t size = get_arg_val<uint32_t>(4);
    const uint32_t n = get_arg_val<uint32_t>(5);
    const uint32_t warmup = get_arg_val<uint32_t>(6);
    const uint32_t polls = get_arg_val<uint32_t>(7);
    const uint32_t epoch = get_arg_val<uint32_t>(8);
    const uint64_t src_idma = arg_u64(9);
    const uint64_t dst_idma = arg_u64(11);
    const uint64_t dst_cpu = arg_u64(13);

    volatile Result* r = result_block(base, hart);
    r->kernel = static_cast<uint32_t>(KernelId::TransportBw);
    r->hart = hart;
    r->status = kStatusRunning;

    uint32_t bad = 0;
    if (hart >= num_harts || num_harts > kMaxHarts) {
        bad = 1;
    } else if (size == 0 || size > kMaxTransferBytes || size % 64 != 0) {
        bad = 2;
    } else if (n == 0) {
        bad = 3;
    } else if (static_cast<uint32_t>(mech) > static_cast<uint32_t>(TransportMech::CpuStore64)) {
        bad = 4;
    }
    if (bad != 0) {
        r->detail = bad;
        finish(r, kStatusBadArgs);
        return;
    }

    volatile uint64_t* cpu_dst = reinterpret_cast<volatile uint64_t*>(static_cast<uintptr_t>(dst_cpu));
    const uint32_t words64 = size / sizeof(uint64_t);

    auto one = [&]() {
        switch (mech) {
            case TransportMech::IdmaMetal: cce_noc::copy(src_idma, dst_idma, size); break;
            case TransportMech::IdmaLean: lean_issue(src_idma, dst_idma, size); break;
            case TransportMech::CpuStore64: cpu_store(cpu_dst, words64, hart); break;
        }
    };

    if (mech == TransportMech::IdmaLean) {
        lean_setup(size);
    }

    // Warm-up, drained before the gate so no warm-up transfer overlaps the timed phase.
    for (uint32_t i = 0; i < warmup; i++) {
        one();
    }
    asm volatile("fence iorw, iorw" ::: "memory");
    if (mech != TransportMech::CpuStore64 && !drain(polls)) {
        r->detail = 1;  // warm-up drain
        finish(r, kStatusTimeout);
        return;
    }

    const uint64_t g0 = now();
    if (!gate(base, hart, num_harts, epoch, polls)) {
        r->detail = 2;  // gate
        finish(r, kStatusTimeout);
        return;
    }
    const uint64_t t0 = now();
    r->cycles[tp::kGateWait] = t0 - g0;

    for (uint32_t i = 0; i < n; i++) {
        one();
    }
    // CPU stores: the timestamp's own leading fence waits for the stores, so t_inj is their retirement.
    const uint64_t t_inj = now();
    if (mech != TransportMech::CpuStore64 && !drain(polls)) {
        r->detail = 3;  // timed drain
        finish(r, kStatusTimeout);
        return;
    }
    const uint64_t t_drain = now();

    r->cycles[tp::kInjection] = t_inj - t0;
    r->cycles[tp::kDrained] = t_drain - t0;
    r->iterations_done = n;
    finish(r, kStatusDone);
}
