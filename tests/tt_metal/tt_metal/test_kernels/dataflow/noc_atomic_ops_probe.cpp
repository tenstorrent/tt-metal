// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Probes the NoC atomic operations beyond increment. The host picks one mode
// per -D define: decrement via INCR_GET, decrement via a raw RISCV_AMO,
// a raw 4-bit CAS, or the CAS return-value path. The raw encodings come from the
// RTL (tt_t6_l1_sub_bank_atomic.sv/tt_t6_l1_pkg.sv).

#include "api/dataflow/dataflow_api.h"
#include "experimental/kernel_args.h"

#if (defined(PROBE_DECR_AMO) || defined(PROBE_CAS)) && defined(ARCH_QUASAR)
// Sends one raw NoC atomic, built the same way as noc_fast_atomic_increment but with a
// caller-supplied at_len. It never asks for a return value, so the tests check results
// by reading the target word back from L1.
inline __attribute__((always_inline)) void noc_raw_atomic(uint64_t noc_addr, uint64_t at_len, uint32_t operand) {
    const uint32_t noc = noc_index;
    const uint64_t misc = CMD_BUF_MISC_ATOMIC_TRANS | CMD_BUF_MISC_SRC_INCLUDE;
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_MISC_REG_OFFSET / 8, misc);
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(
        TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_REQ_VC_REG_OFFSET / 8, NOC_UNICAST_WRITE_VC);
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(
        TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_RESP_VC_REG_OFFSET / 8, NOC_V2_WR_RESP_VC);
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(
        TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_ADDR_REG_OFFSET / 8, (uint32_t)(noc_addr & 0xFFFFFFFF));
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(
        TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_COORD_REG_OFFSET / 8,
        (uint32_t)(noc_addr >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK);
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_LEN_BYTES_REG_OFFSET / 8, at_len);
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(
        TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_INLINE_DATA_REG_OFFSET / 8, (uint64_t)operand);
    __builtin_riscv_ttrocc_scmdbuf_issue_trans();
    // The Quasar atomic barrier waits on the HW ack counter; bump the SW count too so
    // context save/restore sees a consistent value.
    noc_nonposted_atomics_acked[noc] += 1;
}
#endif

#if defined(PROBE_CAS_RET) && defined(ARCH_QUASAR)
// Uncached (TL1) view of an L1 address. NoC atomics and their responses land at TL1, so
// these words must be read and written around the DM write-back cache.
inline volatile tt_l1_ptr uint32_t* uncached(uint32_t addr) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(static_cast<uintptr_t>(addr) + MEM_L1_UNCACHED_BASE);
}
#endif

void kernel_main() {
    const uint32_t sem_addr = get_arg(args::sem_addr);
    const uint32_t increment_times = get_arg(args::increment_times);
    const uint64_t self_noc_addr = get_noc_addr(sem_addr);  // loopback to this node

#if defined(PROBE_DECR_INCRGET)
    // Decrement through the INCR_GET path. noc_semaphore_inc uses wrap=31, so
    // adding -1 is a full 32-bit modular subtract.
    for (uint32_t i = 0; i < increment_times; i++) {
        noc_semaphore_inc(self_noc_addr, (uint32_t)(-1));
        noc_async_atomic_barrier();
    }

#elif defined(PROBE_DECR_AMO)
#if defined(ARCH_QUASAR)
    // Decrement with a raw RISCV_AMO.
    const uint64_t at_len_amoadd = NOC_AT_INS(NOC_AT_INS_RISCV_AMO) | ((uint64_t)0x8 << 8) | NOC_AT_IND_32(0);
    for (uint32_t i = 0; i < increment_times; i++) {
        noc_raw_atomic(self_noc_addr, at_len_amoadd, (uint32_t)(-1));
        noc_async_atomic_barrier();
    }
#endif

#elif defined(PROBE_CAS)
#if defined(ARCH_QUASAR)
    // Raw 4-bit CAS, single writer: one CAS that matches (5 -> 9), then one that
    // mismatches and must leave the word alone. The host preloads 5 and expects the
    // word to end at 9.
    uint64_t hart;
    asm volatile("csrr %0, mhartid" : "=r"(hart));
    if (hart == 2) {  // Metal 2.0 reserves DM0/DM1; lowest user DM is 2
        const uint32_t lane = (uint32_t)((sem_addr >> 2) & 0x3);
        const uint64_t at_len_cas_ok =
            NOC_AT_INS(NOC_AT_INS_CAS) | ((uint64_t)(9 & 0xF) << 6) | ((uint64_t)(5 & 0xF) << 2) | NOC_AT_IND_32(lane);
        noc_raw_atomic(self_noc_addr, at_len_cas_ok, 0);
        noc_async_atomic_barrier();
        const uint64_t at_len_cas_fail =
            NOC_AT_INS(NOC_AT_INS_CAS) | ((uint64_t)(2 & 0xF) << 6) | ((uint64_t)(5 & 0xF) << 2) | NOC_AT_IND_32(lane);
        noc_raw_atomic(self_noc_addr, at_len_cas_fail, 0);
        noc_async_atomic_barrier();
    }
#endif

#elif defined(PROBE_CAS_RET)
#if defined(ARCH_QUASAR)
    // CAS return-value check. The CAS response writes the word's PRE-OP value into this
    // hart's return slot, whether the CAS succeeded or failed. Each CAS here gets its
    // own slot, pre-set to a sentinel and polled until the response overwrites it.
    // report[0] and report[2] are instead sampled right after the atomic barrier with
    // no poll, to show whether the barrier also orders the return write.
    uint64_t hart;
    asm volatile("csrr %0, mhartid" : "=r"(hart));
    if (hart == 2) {
        constexpr uint32_t SENTINEL = 0xFFFFFFFFu;
        const uint32_t word2_addr = sem_addr + 16;
        const uint32_t slot_a = sem_addr + 32;
        const uint32_t slot_b = sem_addr + 48;
        const uint32_t slot_c = sem_addr + 64;
        // The report goes through the uncached alias so the host can read it without a flush.
        volatile tt_l1_ptr uint32_t* report = uncached(sem_addr + 128);

        // 1. A successful CAS (5 -> 9) must return the pre-op 5.
        *uncached(slot_a) = SENTINEL;
        noc_fast_atomic_cas4<DM_DEDICATED_NOC>(
            noc_index, self_noc_addr, NOC_UNICAST_WRITE_VC, 5 /*cmp*/, 9 /*swap*/, slot_a);
        noc_async_atomic_barrier();
        report[0] = *uncached(slot_a);  // immediate, no poll
        while (*uncached(slot_a) == SENTINEL) {
        }
        report[1] = *uncached(slot_a);  // polled: expect 5

        // 2. A failed CAS (cmp=5, but the word is 9) must still return the pre-op word.
        *uncached(slot_b) = SENTINEL;
        noc_fast_atomic_cas4<DM_DEDICATED_NOC>(noc_index, self_noc_addr, NOC_UNICAST_WRITE_VC, 5, 2, slot_b);
        noc_async_atomic_barrier();
        report[2] = *uncached(slot_b);  // immediate, no poll
        while (*uncached(slot_b) == SENTINEL) {
        }
        report[3] = *uncached(slot_b);  // polled: expect 9

        // 3. The failed CAS left the word unchanged.
        report[4] = *uncached(sem_addr);  // expect 9

        // 4. Success also requires word[31:4]==0, so the CAS on word2=0x15 must fail
        //    even though the low nibble matches cmp, and must still return the
        //    pre-op word.
        *uncached(slot_c) = SENTINEL;
        noc_fast_atomic_cas4<DM_DEDICATED_NOC>(noc_index, get_noc_addr(word2_addr), NOC_UNICAST_WRITE_VC, 5, 9, slot_c);
        noc_async_atomic_barrier();
        while (*uncached(slot_c) == SENTINEL) {
        }
        report[5] = *uncached(slot_c);      // expect 0x15 (pre-op returned on failure)
        report[6] = *uncached(word2_addr);  // expect 0x15 (unchanged)
    }
#endif
#endif
}
