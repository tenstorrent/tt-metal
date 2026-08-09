// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// PROBE kernel: NoC atomic ops beyond plain increment on Quasar. The NoC atomic HW defines
// SWAP/CAS/ACC/RISCV_AMO opcodes (noc_parameters.h) but tt-metal only ever emits INCR_GET;
// this probe confirms cross-domain atomic DECREMENT and CAS work.
//
// Encodings below are RTL-confirmed (aether tt_t6_l1_sub_bank_atomic.sv /
// tt_t6_l1_pkg.sv). Modes selected by a -D from the host:
//   PROBE_DECR_INCRGET : atomic decrement via the EXISTING INCR_GET path
//                        (noc_semaphore_inc with incr = -1, wrap=31 => modular sub).
//   PROBE_DECR_AMO     : atomic decrement via a raw NOC_AT_INS_RISCV_AMO (AMOADD, -1). Quasar-only.
//   PROBE_CAS          : 4-bit compare-and-swap via a raw NOC_AT_INS_CAS. Quasar-only.
//   PROBE_CAS_RET      : noc_fast_atomic_cas4 with program_ret_addr=true -- the response
//                        must return the PRE-OP word on success AND failure (Quasar-only).

#include "api/dataflow/dataflow_api.h"
#include "experimental/kernel_args.h"

#if (defined(PROBE_DECR_AMO) || defined(PROBE_CAS)) && defined(ARCH_QUASAR)
// Raw NoC-atomic emit, mirroring noc_fast_atomic_increment (noc_nonblocking_api_v2.h)
// but with a caller-supplied at_len (opcode+fields) and inline operand. Does NOT program
// the return-value addr (like noc_semaphore_inc); result is verified by reading L1 back.
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
    // Mirror noc_fast_atomic_increment's software bookkeeping (the Quasar atomic barrier waits
    // on the HW ack counter, but keeping this in sync avoids surprising context save/restore).
    noc_nonposted_atomics_acked[noc] += 1;
}
#endif

#if defined(PROBE_CAS_RET) && defined(ARCH_QUASAR)
// TL1 view of an L1 address: the CAS response (and NoC atomics generally) land at TL1,
// so reads/writes here must bypass the DM write-back cache.
inline volatile tt_l1_ptr uint32_t* uncached(uint32_t addr) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(static_cast<uintptr_t>(addr) + MEM_L1_UNCACHED_BASE);
}
#endif

void kernel_main() {
    const uint32_t sem_addr = get_arg(args::sem_addr);
    const uint32_t increment_times = get_arg(args::increment_times);
    const uint64_t self_noc_addr = get_noc_addr(sem_addr);  // loopback to this node

#if defined(PROBE_DECR_INCRGET)
    // Atomic decrement via the EXISTING path: incr = -1 with wrap=31 (set by
    // noc_semaphore_inc) is a full 32-bit modular add, i.e. subtract 1.
    for (uint32_t i = 0; i < increment_times; i++) {
        noc_semaphore_inc(self_noc_addr, (uint32_t)(-1));
        noc_async_atomic_barrier();
    }

#elif defined(PROBE_DECR_AMO)
#if defined(ARCH_QUASAR)
    // Atomic decrement via raw NOC_AT_INS_RISCV_AMO / AMOADD (briscv_req_type 0x8 at
    // at_len[11:8]); lane 0 requires a 16B-aligned target (l1_unreserved_base is).
    const uint64_t at_len_amoadd = NOC_AT_INS(NOC_AT_INS_RISCV_AMO) | ((uint64_t)0x8 << 8) | NOC_AT_IND_32(0);
    for (uint32_t i = 0; i < increment_times; i++) {
        noc_raw_atomic(self_noc_addr, at_len_amoadd, (uint32_t)(-1));
        noc_async_atomic_barrier();
    }
#endif

#elif defined(PROBE_CAS)
#if defined(ARCH_QUASAR)
    // 4-bit compare-and-swap (single writer). Host inits the word to 5.
    //   CAS(cmp=5, swap=9) -> succeeds (word 5 -> 9)
    //   CAS(cmp=5, swap=2) -> fails (word is 9) -> unchanged
    // Host verifies the final word == 9. Only the lowest user DM acts.
    uint64_t hart;
    asm volatile("csrr %0, mhartid" : "=r"(hart));
    if (hart == 2) {  // Metal 2.0 reserves DM0/DM1; lowest user DM is 2
        // NOC_AT_IND_32 selects the DESTINATION 32-bit lane within the target's 16B row (at_len[1:0]).
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
    // CAS return value (program_ret_addr=true): the response writes the PRE-OP word to the slot
    // programmed into this hart's sticky R_SRC_ADDR -- on success AND on failure. Single writer;
    // each CAS gets a private slot pre-set to a sentinel via the uncached alias, then polled
    // until the response overwrites it. report[0]/report[2] are read IMMEDIATELY after the
    // atomic barrier (no poll) to probe whether the barrier also orders the return write.
    // Scratch layout (offsets from sem_addr; must match TestAtomicCasReturnsPreOpValue):
    //   +0 word (host preloads 5)   +16 word2 (host preloads 0x15)
    //   +32/+48/+64 slotA/B/C       +128 report[7]
    uint64_t hart;
    asm volatile("csrr %0, mhartid" : "=r"(hart));
    if (hart == 2) {                                // Metal 2.0 reserves DM0/DM1; lowest user DM is 2
        constexpr uint32_t SENTINEL = 0xFFFFFFFFu;  // never a legal pre-op word here (5/9/0x15)
        const uint32_t word2_addr = sem_addr + 16;
        const uint32_t slot_a = sem_addr + 32;
        const uint32_t slot_b = sem_addr + 48;
        const uint32_t slot_c = sem_addr + 64;
        // Report is written via the uncached alias: lands at TL1, no flush needed for readback.
        volatile tt_l1_ptr uint32_t* report = uncached(sem_addr + 128);

        // 1. Successful CAS (5 -> 9) returns the pre-op 5.
        *uncached(slot_a) = SENTINEL;
        noc_fast_atomic_cas4<DM_DEDICATED_NOC, true /*program_ret_addr*/>(
            noc_index,
            0 /*cmd_buf unused*/,
            self_noc_addr,
            NOC_UNICAST_WRITE_VC,
            5 /*cmp*/,
            9 /*swap*/,
            false /*linked*/,
            false /*posted*/,
            slot_a);
        noc_async_atomic_barrier();
        report[0] = *uncached(slot_a);  // immediate, no poll
        while (*uncached(slot_a) == SENTINEL) {
        }
        report[1] = *uncached(slot_a);  // polled: expect 5

        // 2. FAILED CAS (cmp=5 but word is 9) also returns the pre-op word.
        *uncached(slot_b) = SENTINEL;
        noc_fast_atomic_cas4<DM_DEDICATED_NOC, true>(
            noc_index, 0, self_noc_addr, NOC_UNICAST_WRITE_VC, 5, 2, false, false, slot_b);
        noc_async_atomic_barrier();
        report[2] = *uncached(slot_b);  // immediate, no poll
        while (*uncached(slot_b) == SENTINEL) {
        }
        report[3] = *uncached(slot_b);  // polled: expect 9

        // 3. The failed CAS left the word unchanged.
        report[4] = *uncached(sem_addr);  // expect 9

        // 4. Upper-28 rule: success requires word[31:4]==0, so word2=0x15 must FAIL the
        //    CAS even though word2[3:0] == cmp4 == 5 -- and still return the pre-op word.
        *uncached(slot_c) = SENTINEL;
        noc_fast_atomic_cas4<DM_DEDICATED_NOC, true>(
            noc_index, 0, get_noc_addr(word2_addr), NOC_UNICAST_WRITE_VC, 5, 9, false, false, slot_c);
        noc_async_atomic_barrier();
        while (*uncached(slot_c) == SENTINEL) {
        }
        report[5] = *uncached(slot_c);      // expect 0x15 (pre-op returned on failure)
        report[6] = *uncached(word2_addr);  // expect 0x15 (unchanged)
    }
#endif
#endif
}
