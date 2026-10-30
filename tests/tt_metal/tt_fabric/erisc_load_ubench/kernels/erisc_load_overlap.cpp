// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// How much read latency an ERISC can hide (see test_erisc_load_overlap.cpp). Each timed loop, counter and branch
// included, is one inline-asm block aligned to 32 bytes, so the instruction order and code layout are the same for
// every pattern: loads, K independent ALU instructions ("filler"), then the instructions that use the loaded values.
// The host subtracts a same-shape baseline without loads, leaving what the loads cost. Besides stream registers,
// L1 and local memory, the register patterns read a TX queue's command and status registers and NoC status and
// command-buffer registers; none is written, so nothing is sent. The L1 data cache is off
// during the measurement, as in the router (ENABLE_RISC_CPU_DATA_CACHE = 0); define ERISC_LOAD_DATA_CACHE=1 to
// measure with it on.

#include <cstdint>
#include <utility>

#include "risc_common.h"
#include "noc_nonblocking_api.h"
#include "internal/ethernet/tt_eth_ss_regs.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_stream_regs.hpp"
#include "tests/tt_metal/tt_fabric/erisc_load_ubench/kernels/erisc_load_overlap_common.hpp"

using namespace erisc_load_overlap;

// A word in the core's local data memory, where the stack and spills live.
volatile uint32_t ldm_word = 7;
// Two more local-memory words, and one holding an L1 address (a parked send cursor), for the cursor patterns.
volatile uint32_t ldm_word2 = 9;
volatile uint32_t ldm_word3 = 11;
volatile uint32_t ldm_slot = 0;

namespace {

#define FILL_K ".rept %[k]\n addi t1,t1,1\n .endr\n"
#define FILL_H ".rept %[h]\n addi t1,t1,1\n .endr\n"
#define FILL_R ".rept %[r]\n addi t1,t1,1\n .endr\n"
#define USE(reg) "add %[a],%[a]," reg "\n"
#define TIMED(BODY)                                                                \
    asm volatile("csrr %[c0], mcycle\n li %[n], %[iters]\n .p2align 5\n 1:\n" BODY \
                 "addi %[n], %[n], -1\n bnez %[n], 1b\n csrr %[c1], mcycle\n"      \
                 : [c0] "=&r"(c0), [c1] "=&r"(c1), [n] "=&r"(n), [a] "+r"(acc)     \
                 : [iters] "n"(kIterations),                                       \
                   [k] "n"(K),                                                     \
                   [h] "n"(K / 2),                                                 \
                   [r] "n"(K - K / 2),                                             \
                   [s] "r"(s),                                                     \
                   [s2] "r"(s2),                                                   \
                   [s3] "r"(s3),                                                   \
                   [s4] "r"(s4),                                                   \
                   [l] "r"(l),                                                     \
                   [l2] "r"(l2),                                                   \
                   [d] "r"(d),                                                     \
                   [tc] "r"(regs.txq_cmd),                                         \
                   [tq] "r"(regs.txq_status),                                      \
                   [nw] "r"(regs.niu),                                             \
                   [cb] "r"(regs.cmdbuf)                                           \
                 : "t1", "t2", "t3", "t4", "t5", "t6", "memory")

// The cursor patterns' timed block: stream registers, the L1 word, three local-memory words and the parked slot.
#define TIMED_CURSOR(BODY)                                                         \
    asm volatile("csrr %[c0], mcycle\n li %[n], %[iters]\n .p2align 5\n 1:\n" BODY \
                 "addi %[n], %[n], -1\n bnez %[n], 1b\n csrr %[c1], mcycle\n"      \
                 : [c0] "=&r"(c0), [c1] "=&r"(c1), [n] "=&r"(n), [a] "+r"(acc)     \
                 : [iters] "n"(kIterations),                                       \
                   [k] "n"(K),                                                     \
                   [s] "r"(s),                                                     \
                   [s2] "r"(s2),                                                   \
                   [s3] "r"(s3),                                                   \
                   [s4] "r"(s4),                                                   \
                   [l] "r"(l),                                                     \
                   [d] "r"(d),                                                     \
                   [d2] "r"(reinterpret_cast<uint32_t>(&ldm_word2)),               \
                   [d3] "r"(reinterpret_cast<uint32_t>(&ldm_word3)),               \
                   [ds] "r"(reinterpret_cast<uint32_t>(&ldm_slot))                 \
                 : "t1", "t2", "t3", "t4", "t5", "t6", "a6", "a7", "memory")

// Addresses of the non-stream MMIO registers read by the register patterns.
struct Regs {
    uint32_t txq_cmd;
    uint32_t txq_status;
    uint32_t niu;
    uint32_t cmdbuf;
};

template <uint32_t P, uint32_t K>
uint32_t time_block(
    uint32_t& acc,
    uint32_t s,
    uint32_t s2,
    uint32_t s3,
    uint32_t s4,
    uint32_t l,
    uint32_t l2,
    uint32_t d,
    const Regs& regs) {
    uint32_t best = UINT32_MAX;
#pragma GCC unroll 1
    for (uint32_t rep = 0; rep < kRepeats; rep++) {
        uint32_t c0, c1, n;
        if constexpr (P == BASE2) {
            TIMED(FILL_K USE("t1") USE("t1"));
        } else if constexpr (P == DEP_STREAM) {
            TIMED("lw t2,0(%[s])\n" USE("t2") FILL_K USE("t1"));
        } else if constexpr (P == DEP_L1) {
            TIMED("lw t3,0(%[l])\n" USE("t3") FILL_K USE("t1"));
        } else if constexpr (P == DEP_BOTH) {
            TIMED("lw t2,0(%[s])\n" USE("t2") "lw t3,0(%[l])\n" USE("t3") FILL_K);
        } else if constexpr (P == PRE_STREAM) {
            TIMED("lw t2,0(%[s])\n" FILL_K USE("t2") USE("t1"));
        } else if constexpr (P == PRE_L1) {
            TIMED("lw t3,0(%[l])\n" FILL_K USE("t3") USE("t1"));
        } else if constexpr (P == PRE_BOTH) {
            TIMED("lw t2,0(%[s])\n lw t3,0(%[l])\n" FILL_K USE("t2") USE("t3"));
        } else if constexpr (P == PRE_BOTH_REV) {
            TIMED("lw t3,0(%[l])\n lw t2,0(%[s])\n" FILL_K USE("t2") USE("t3"));
        } else if constexpr (P == PRE_STREAM_LDM) {
            TIMED("lw t2,0(%[s])\n lw t3,0(%[d])\n" FILL_K USE("t2") USE("t3"));
        } else if constexpr (P == PRE_STREAM2) {
            TIMED("lw t2,0(%[s])\n lw t3,0(%[s2])\n" FILL_K USE("t2") USE("t3"));
        } else if constexpr (P == PRE_L1_LDM) {
            TIMED("lw t3,0(%[l])\n lw t2,0(%[d])\n" FILL_K USE("t2") USE("t3"));
        } else if constexpr (P == PRE_L1_2) {
            TIMED("lw t3,0(%[l])\n lw t2,0(%[l2])\n" FILL_K USE("t2") USE("t3"));
        } else if constexpr (P == PRE_STREAM_X0) {
            TIMED("lw t2,0(%[s])\n lw zero,0(%[d])\n" FILL_K USE("t2") USE("t1"));
        } else if constexpr (P == BASE3) {
            TIMED(FILL_K USE("t1") USE("t1") USE("t1"));
        } else if constexpr (P == DEP_LDM) {
            TIMED("lw t4,0(%[d])\n" USE("t4") FILL_K USE("t1") USE("t1"));
        } else if constexpr (P == DEP_STREAM2) {
            TIMED("lw t4,0(%[s2])\n" USE("t4") FILL_K USE("t1") USE("t1"));
        } else if constexpr (P == PRE_BOTH_LDM) {
            TIMED("lw t2,0(%[s])\n lw t3,0(%[l])\n" FILL_H "lw t4,0(%[d])\n" USE("t4") FILL_R USE("t2") USE("t3"));
        } else if constexpr (P == PRE_BOTH_MMIO) {
            TIMED("lw t2,0(%[s])\n lw t3,0(%[l])\n" FILL_H "lw t4,0(%[s2])\n" USE("t4") FILL_R USE("t2") USE("t3"));
        } else if constexpr (P == PRE_BOTH_TAIL) {
            TIMED("lw t2,0(%[s])\n lw t3,0(%[l])\n lw t4,0(%[d])\n" FILL_K USE("t2") USE("t3") USE("t4"));
        } else if constexpr (P == PRE_S2L) {
            TIMED("lw t2,0(%[s])\n lw t4,0(%[s2])\n lw t3,0(%[l])\n" FILL_K USE("t2") USE("t3") USE("t4"));
        } else if constexpr (P == BASE4) {
            TIMED(FILL_K USE("t1") USE("t1") USE("t1") USE("t1"));
        } else if constexpr (P == PRE_S4) {
            TIMED("lw t2,0(%[s])\n lw t3,0(%[s2])\n lw t4,0(%[s3])\n lw t5,0(%[s4])\n" FILL_K USE("t2") USE("t3")
                      USE("t4") USE("t5"));
        } else if constexpr (P == BASE5) {
            TIMED(FILL_K USE("t1") USE("t1") USE("t1") USE("t1") USE("t1"));
        } else if constexpr (P == PRE_S4_L) {
            TIMED("lw t2,0(%[s])\n lw t3,0(%[s2])\n lw t4,0(%[s3])\n lw t5,0(%[s4])\n lw t6,0(%[l])\n" FILL_K USE("t2")
                      USE("t3") USE("t4") USE("t5") USE("t6"));
        } else if constexpr (P == DEP_TXQ) {
            TIMED("lw t2,0(%[tc])\n" USE("t2") FILL_K USE("t1"));
        } else if constexpr (P == DEP_TXQ_BUSY) {
            TIMED("lw t3,0(%[tc])\n lw t2,0(%[tq])\n" USE("t2") FILL_K USE("t1"));
        } else if constexpr (P == DEP_NIU) {
            TIMED("lw t2,0(%[nw])\n" USE("t2") FILL_K USE("t1"));
        } else if constexpr (P == DEP_CMDBUF) {
            TIMED("lw t2,0(%[cb])\n" USE("t2") FILL_K USE("t1"));
        } else if constexpr (P == PRE_TXQ_BUSY) {
            TIMED("lw t3,0(%[tc])\n lw t2,0(%[tq])\n" FILL_K USE("t2") USE("t1"));
        } else if constexpr (P == PRE_S_TXQ) {
            TIMED("lw t2,0(%[s])\n lw t3,0(%[tc])\n lw t4,0(%[tq])\n" FILL_K USE("t2") USE("t4"));
        } else if constexpr (P == PRE_S_NIU) {
            TIMED("lw t2,0(%[s])\n lw t4,0(%[nw])\n" FILL_K USE("t2") USE("t4"));
        } else if constexpr (P == PRE_E1_GROUP) {
            TIMED("lw t2,0(%[s])\n lw t3,0(%[tc])\n lw t4,0(%[tq])\n lw t5,0(%[nw])\n" FILL_K USE("t2") USE("t4")
                      USE("t5"));
        } else if constexpr (P == DEP_TXQ_BUSY_WAW) {
            TIMED("lw t2,0(%[tc])\n lw t2,0(%[tq])\n" USE("t2") FILL_K USE("t1"));
        } else if constexpr (P == PRE_E1_GROUP_WAW) {
            TIMED("lw t2,0(%[s])\n lw t4,0(%[tc])\n lw t4,0(%[tq])\n lw t5,0(%[nw])\n" FILL_K USE("t2") USE("t4")
                      USE("t5"));
        } else if constexpr (P == PRE_S4_TXQ) {
            TIMED(
                "lw t2,0(%[s])\n lw t3,0(%[s2])\n lw t4,0(%[s3])\n lw t5,0(%[s4])\n lw t6,0(%[tc])\n"
                " lw t6,0(%[tq])\n" FILL_K USE("t2") USE("t3") USE("t4") USE("t5") USE("t6"));
        } else if constexpr (P == BASE7) {
            TIMED_CURSOR(FILL_K USE("t1") USE("t1") USE("t1") USE("t1") USE("t1") USE("t1") USE("t1"));
        } else if constexpr (P == PRE_S4_D) {
            TIMED_CURSOR("lw t2,0(%[s])\n lw t3,0(%[s2])\n lw t4,0(%[s3])\n lw t5,0(%[s4])\n lw t6,0(%[d])\n" FILL_K
                             USE("t2") USE("t3") USE("t4") USE("t5") USE("t6"));
        } else if constexpr (P == PRE_D_S4) {
            TIMED_CURSOR("lw t6,0(%[d])\n lw t2,0(%[s])\n lw t3,0(%[s2])\n lw t4,0(%[s3])\n lw t5,0(%[s4])\n" FILL_K
                             USE("t2") USE("t3") USE("t4") USE("t5") USE("t6"));
        } else if constexpr (P == PRE_S4_D3) {
            TIMED_CURSOR(
                "lw t2,0(%[s])\n lw t3,0(%[s2])\n lw t4,0(%[s3])\n lw t5,0(%[s4])\n"
                " lw t6,0(%[d])\n lw a6,0(%[d2])\n lw a7,0(%[d3])\n" FILL_K USE("t2") USE("t3") USE("t4") USE("t5")
                    USE("t6") USE("a6") USE("a7"));
        } else if constexpr (P == PRE_D3_S4) {
            TIMED_CURSOR(
                "lw t6,0(%[d])\n lw a6,0(%[d2])\n lw a7,0(%[d3])\n"
                " lw t2,0(%[s])\n lw t3,0(%[s2])\n lw t4,0(%[s3])\n lw t5,0(%[s4])\n" FILL_K USE("t2") USE("t3")
                    USE("t4") USE("t5") USE("t6") USE("a6") USE("a7"));
        } else if constexpr (P == PRE_S4_HDR) {
            TIMED_CURSOR("lw t2,0(%[s])\n lw t3,0(%[s2])\n lw t4,0(%[s3])\n lw t5,0(%[s4])\n" FILL_K USE("t2")
                             USE("t3") USE("t4") USE("t5") "lw t6,0(%[l])\n" USE("t6"));
        } else if constexpr (P == PRE_S4_DS_HDR) {
            TIMED_CURSOR("lw t2,0(%[s])\n lw t3,0(%[s2])\n lw t4,0(%[s3])\n lw t5,0(%[s4])\n lw a6,0(%[ds])\n" FILL_K
                             USE("t2") USE("t3") USE("t4") USE("t5") "lw t6,0(a6)\n" USE("t6"));
        } else if constexpr (P == DEP_DS_HDR) {
            TIMED_CURSOR("lw a6,0(%[ds])\n lw t6,0(a6)\n" USE("t6") FILL_K USE("t1"));
        }
        const uint32_t t = c1 - c0;
        best = t < best ? t : best;
    }
    return best;
}

#ifndef ERISC_LOAD_GROUP
#define ERISC_LOAD_GROUP 0
#endif
constexpr uint32_t kFirst = kGroups[ERISC_LOAD_GROUP][0];
constexpr uint32_t kLast = kGroups[ERISC_LOAD_GROUP][1];

template <uint32_t P, size_t... KI>
void time_pattern(
    volatile uint32_t* results,
    uint32_t& acc,
    uint32_t s,
    uint32_t s2,
    uint32_t s3,
    uint32_t s4,
    uint32_t l,
    uint32_t l2,
    uint32_t d,
    const Regs& regs,
    std::index_sequence<KI...>) {
    if constexpr (P >= kFirst && P < kLast) {
        ((results[P * kNumFill + KI] = time_block<P, kFill[KI]>(acc, s, s2, s3, s4, l, l2, d, regs)), ...);
    }
}

template <size_t... PI>
void time_all(
    volatile uint32_t* results,
    uint32_t& acc,
    uint32_t s,
    uint32_t s2,
    uint32_t s3,
    uint32_t s4,
    uint32_t l,
    uint32_t l2,
    uint32_t d,
    const Regs& regs,
    std::index_sequence<PI...>) {
    (time_pattern<PI>(results, acc, s, s2, s3, s4, l, l2, d, regs, std::make_index_sequence<kNumFill>{}), ...);
}

}  // namespace

void kernel_main() {
    const uint32_t base = get_arg_val<uint32_t>(0);
    volatile uint32_t* results = reinterpret_cast<volatile uint32_t*>(base + kResultsOffset);
    *reinterpret_cast<volatile uint32_t*>(base + kL1WordOffset) = 3;
    *reinterpret_cast<volatile uint32_t*>(base + kL1Word2Offset) = 5;
    ldm_slot = base + kL1WordOffset;

    uint32_t cache_csr;
    asm volatile("csrr %0, 0x7c0" : "=r"(cache_csr));
    results[kNumPatterns * kNumFill + 2] = cache_csr;
#if defined(ERISC_LOAD_DATA_CACHE) && ERISC_LOAD_DATA_CACHE
    set_l1_data_cache<true>();
#else
    set_l1_data_cache<false>();
#endif

    const Regs regs = {
        .txq_cmd = ETH_TXQ0_REGS_START + kTxq * ETH_TXQ_REGS_SIZE + ETH_TXQ_CMD,
        .txq_status = ETH_TXQ0_REGS_START + kTxq * ETH_TXQ_REGS_SIZE + ETH_TXQ_STATUS,
        .niu = (kNiuNoc << NOC_INSTANCE_OFFSET_BIT) + NOC_STATUS(NIU_MST_WRITE_REQS_OUTGOING_ID(0)),
        .cmdbuf = NOC_CMD_BUF_INSTANCE_OFFSET(kNiuNoc, 0) + NOC_CMD_CTRL,
    };
    uint32_t acc = 0;
    time_all(
        results,
        acc,
        get_stream_reg_read_addr<kStream>(),
        get_stream_reg_read_addr<kStream2>(),
        get_stream_reg_read_addr<kStream3>(),
        get_stream_reg_read_addr<kStream4>(),
        base + kL1WordOffset,
        base + kL1Word2Offset,
        reinterpret_cast<uint32_t>(&ldm_word),
        regs,
        std::make_index_sequence<kNumPatterns>{});

    asm volatile("fence\n csrw 0x7c0, %0" ::"r"(cache_csr));  // back to the state firmware left
    results[kNumPatterns * kNumFill] = acc;                   // keep the work observable
    results[kNumPatterns * kNumFill + 1] = kDone;
}
