// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// PROBE kernel: verify NoC atomic operations BEYOND plain increment on Quasar.
//
// The auto-path semaphore plan's "cross-domain atomic is increment-only" limit was
// actually a SOFTWARE gap: the NoC atomic HW defines SWAP/CAS/ACC/RISCV_AMO opcodes
// (noc_parameters.h:351-361) but tt-metal SW only ever emits INCR_GET. This probe
// exercises the missing ops to confirm cross-domain atomic DECREMENT and CAS work,
// which would make "all cross-domain ops atomic" a software deliverable on EXTERNAL.
//
// Encodings below are RTL-confirmed (aether tt_t6_l1_sub_bank_atomic.sv /
// tt_t6_l1_pkg.sv). Modes selected by a -D from the host:
//   PROBE_DECR_INCRGET : atomic decrement via the EXISTING INCR_GET path
//                        (noc_semaphore_inc with incr = -1, wrap=31 => modular sub).
//   PROBE_DECR_AMO     : atomic decrement via a raw NOC_AT_INS_RISCV_AMO (AMOADD, -1).
//   PROBE_CAS          : 4-bit compare-and-swap via a raw NOC_AT_INS_CAS.

#include "api/dataflow/dataflow_api.h"
#include "experimental/kernel_args.h"

#if defined(PROBE_DECR_AMO) || defined(PROBE_CAS)
// Raw NoC-atomic emit, mirroring noc_fast_atomic_increment (noc_nonblocking_api_v2.h:768)
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
    // Match noc_fast_atomic_increment's bookkeeping so noc_async_atomic_barrier() waits for it.
    noc_nonposted_atomics_acked[noc] += 1;
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
    // Atomic decrement via raw NOC_AT_INS_RISCV_AMO / AMOADD (briscv_req_type 0x8 at
    // at_len[11:8]); src_index 0 requires a 16B-aligned target (l1_unreserved_base is).
    const uint64_t at_len_amoadd = NOC_AT_INS(NOC_AT_INS_RISCV_AMO) | ((uint64_t)0x8 << 8) | NOC_AT_IND_32(0);
    for (uint32_t i = 0; i < increment_times; i++) {
        noc_raw_atomic(self_noc_addr, at_len_amoadd, (uint32_t)(-1));
        noc_async_atomic_barrier();
    }

#elif defined(PROBE_CAS)
    // 4-bit compare-and-swap (single writer). Host inits the word to 5.
    //   CAS(cmp=5, swap=9) -> succeeds (word 5 -> 9)
    //   CAS(cmp=5, swap=2) -> fails (word is 9) -> unchanged
    // Host verifies the final word == 9. Only the lowest user DM acts.
    bool is_writer = true;
#if defined(ARCH_QUASAR)
    uint64_t hart = 0;
    asm volatile("csrr %0, mhartid" : "=r"(hart));
    is_writer = (hart == 2);  // Metal 2.0 reserves DM0/DM1; lowest user DM is 2
#elif !defined(COMPILE_FOR_BRISC)
    is_writer = false;  // Gen1: only BRISC acts
#endif
    if (is_writer) {
        const uint32_t src_index = (uint32_t)((sem_addr >> 2) & 0x3);
        const uint64_t at_len_cas_ok =
            NOC_AT_INS(NOC_AT_INS_CAS) | ((uint64_t)(9 & 0xF) << 6) | ((uint64_t)(5 & 0xF) << 2) | NOC_AT_IND_32(src_index);
        noc_raw_atomic(self_noc_addr, at_len_cas_ok, 0);
        noc_async_atomic_barrier();
        const uint64_t at_len_cas_fail =
            NOC_AT_INS(NOC_AT_INS_CAS) | ((uint64_t)(2 & 0xF) << 6) | ((uint64_t)(5 & 0xF) << 2) | NOC_AT_IND_32(src_index);
        noc_raw_atomic(self_noc_addr, at_len_cas_fail, 0);
        noc_async_atomic_barrier();
    }
#endif
}
