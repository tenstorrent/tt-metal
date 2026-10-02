// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Measures what programming and issuing a NoC read through the RoCC command buffer costs a Quasar DM core.
//
// Each phase records six 64-bit counter deltas: mcycle, minstret, and four Rocket HPM counters programmed to
// D$ blocked cycles, long-latency interlock cycles, D$ misses and load-use interlock cycles.
//
// Phases, each repeated num_iters times:
//   0 empty region (measurement overhead)
//   1 one cmdbuf register write
//   2 two cmdbuf register writes
//   3 the six cmdbuf register writes of a read, without the issue
//   4 one complete 16B read issue, with the previous read already finished
//   5 four read issues back to back, so the later ones go out while earlier reads are in flight
//   6 the prefetcher's sequence: set trid, issue a read, reset trid to 0
// Phases 1 to 3 time the whole loop. Phases 4 to 6 time only the issue, and wait for the read outside it.
//
// Result layout at result_addr, as 32-bit words:
//   [0] mhpmevent3 read back after programming it
//   [1..3] unused
//   then kNumPhases * kNumCounters 64-bit values, low word first
//   then kNumCounters 64-bit values for a counter read pair with nothing between, the baseline for phases 4 to 6

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "dev_mem_map.h"
#include "experimental/kernel_args.h"

namespace {

constexpr std::uint32_t kNumPhases = 7;
constexpr std::uint32_t kNumCounters = 6;
constexpr std::uint32_t kReadBytes = 16;
constexpr std::uint32_t kTrid = 5;

constexpr std::uint64_t kEventDcacheBlocked = 1 | (1ull << (8 + 4));
constexpr std::uint64_t kEventLongLatency = 1 | (1ull << (8 + 1));
constexpr std::uint64_t kEventDcacheMiss = 2 | (1ull << (8 + 1));
constexpr std::uint64_t kEventLoadUse = 1 | (1ull << (8 + 0));

#define READ_CSR(csr)                                         \
    ({                                                        \
        std::uint64_t value;                                  \
        __asm__ __volatile__("csrr %0, " #csr : "=r"(value)); \
        value;                                                \
    })
#define WRITE_CSR(csr, value) __asm__ __volatile__("csrw " #csr ", %0" ::"r"(value))

struct Counters {
    std::uint64_t value[kNumCounters];
};

inline __attribute__((always_inline)) Counters read_counters() {
    Counters c;
    c.value[0] = READ_CSR(0xB00);
    c.value[1] = READ_CSR(0xB02);
    c.value[2] = READ_CSR(0xB03);
    c.value[3] = READ_CSR(0xB04);
    c.value[4] = READ_CSR(0xB05);
    c.value[5] = READ_CSR(0xB06);
    return c;
}

inline __attribute__((always_inline)) void accumulate(Counters& sum, const Counters& start, const Counters& end) {
    for (std::uint32_t i = 0; i < kNumCounters; i++) {
        sum.value[i] += end.value[i] - start.value[i];
    }
}

inline __attribute__((always_inline)) void write_len(std::uint32_t len) {
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        read_cmd_buf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_LEN_BYTES_REG_OFFSET / 8, len);
}
inline __attribute__((always_inline)) void write_dest(std::uint32_t addr) {
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        read_cmd_buf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_ADDR_REG_OFFSET / 8, addr);
}

// The six register writes ncrisc_noc_fast_read makes in the V2 API, without the issue.
inline __attribute__((always_inline)) void write_read_regs(std::uint64_t src_noc_addr, std::uint32_t dst_addr) {
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        read_cmd_buf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_REQ_VC_REG_OFFSET / 8, NOC_UNICAST_WRITE_VC);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        read_cmd_buf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_RESP_VC_REG_OFFSET / 8, NOC_OVERLAY_RD_RESP_VC);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        read_cmd_buf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SRC_ADDR_REG_OFFSET / 8, (std::uint32_t)src_noc_addr);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        read_cmd_buf,
        TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SRC_COORD_REG_OFFSET / 8,
        (std::uint32_t)(src_noc_addr >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK);
    write_dest(dst_addr);
    write_len(kReadBytes);
}

}  // namespace

void kernel_main() {
    std::uint32_t src_addr = get_arg(args::src_addr);
    std::uint32_t dst_addr = get_arg(args::dst_addr);
    std::uint32_t num_iters = get_arg(args::num_iters);
    std::uint32_t result_addr = get_arg(args::result_addr);

    WRITE_CSR(0x323, kEventDcacheBlocked);
    WRITE_CSR(0x324, kEventLongLatency);
    WRITE_CSR(0x325, kEventDcacheMiss);
    WRITE_CSR(0x326, kEventLoadUse);
    const std::uint64_t event3_readback = READ_CSR(0x323);

    const std::uint64_t src_noc_addr = get_noc_addr(src_addr);
    Counters phases[kNumPhases] = {};

    // Phases 0 to 3 time the whole loop between fences.
    auto loop = [&](std::uint32_t phase, auto body) {
        noc_async_read_barrier();
        __asm__ __volatile__("fence" ::: "memory");
        const Counters start = read_counters();
        for (std::uint32_t i = 0; i < num_iters; i++) {
            body();
        }
        __asm__ __volatile__("fence" ::: "memory");
        accumulate(phases[phase], start, read_counters());
    };
    loop(0, []() {});
    loop(1, []() { write_len(kReadBytes); });
    loop(2, [&]() {
        write_len(kReadBytes);
        write_dest(dst_addr);
    });
    loop(3, [&]() { write_read_regs(src_noc_addr, dst_addr); });

    // Phases 4 to 6 time only the issue, reading the counters without fences the way the prefetcher's own probes
    // do. The fenced phase 0 is no baseline for them, so a bare counter read pair is recorded after them.
    for (std::uint32_t i = 0; i < num_iters; i++) {
        noc_async_read_barrier();
        const Counters start = read_counters();
        noc_async_read(src_noc_addr, dst_addr, kReadBytes);
        accumulate(phases[4], start, read_counters());
    }
    for (std::uint32_t i = 0; i < num_iters; i++) {
        noc_async_read_barrier();
        const Counters start = read_counters();
        noc_async_read(src_noc_addr, dst_addr, kReadBytes);
        noc_async_read(src_noc_addr, dst_addr + kReadBytes, kReadBytes);
        noc_async_read(src_noc_addr, dst_addr + 2 * kReadBytes, kReadBytes);
        noc_async_read(src_noc_addr, dst_addr + 3 * kReadBytes, kReadBytes);
        accumulate(phases[5], start, read_counters());
    }
    for (std::uint32_t i = 0; i < num_iters; i++) {
        noc_async_read_barrier();
        const Counters start = read_counters();
        noc_async_read_set_trid(kTrid);
        noc_async_read(src_noc_addr, dst_addr, kReadBytes);
        noc_async_read_set_trid(0);
        accumulate(phases[6], start, read_counters());
    }
    noc_async_read_barrier();

    Counters probe = {};
    for (std::uint32_t i = 0; i < num_iters; i++) {
        const Counters start = read_counters();
        accumulate(probe, start, read_counters());
    }

    volatile std::uint32_t* out = reinterpret_cast<volatile std::uint32_t*>(result_addr + MEM_L1_UNCACHED_BASE);
    out[0] = static_cast<std::uint32_t>(event3_readback);
    std::uint32_t w = 4;
    for (std::uint32_t p = 0; p < kNumPhases; p++) {
        for (std::uint32_t c = 0; c < kNumCounters; c++) {
            out[w++] = static_cast<std::uint32_t>(phases[p].value[c]);
            out[w++] = static_cast<std::uint32_t>(phases[p].value[c] >> 32);
        }
    }
    for (std::uint32_t c = 0; c < kNumCounters; c++) {
        out[w++] = static_cast<std::uint32_t>(probe.value[c]);
        out[w++] = static_cast<std::uint32_t>(probe.value[c] >> 32);
    }
}
