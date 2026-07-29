/*
 * entry.S - Minimal bare-metal boot stub for the SiFive X280 (L2CPU) on
 *           Blackhole, for the LIM-resident single-hart heartbeat FW.
 *
 * This is a deliberately small Level-2 boot stub (gp + mtvec + feature
 * enable + per-hart sp + bss zero + call main). It drops the IPI and
 * register-probe trap modes of the full tt-llm-engine entry.S: step 1
 * runs C on hart 0 only and never sends an IPI, so the trap handler just
 * records mcause/mepc/mtval to fixed LIM diag addresses and CEASEs.
 *
 * Boot sequence (SiFive coreip_21G3.04.00 manual S5.17):
 *   1. gp  (global pointer; relaxation must be disabled around the la)
 *   2. mtvec -> _trap_handler
 *   3. clear Feature Disable CSR 0x7c1 (enable full uarch features)
 *   4. mstatus.VS = Initial (bit 9) so any future vector insn won't trap
 *   5. per-hart sp = _stack_top - hartid * STACK_SIZE
 *   6. zero .bss (hart 0 only)
 *   7. call main(hartid)  -- a0 = mhartid
 *   8. park if main returns
 *
 * Linked with ld/x280-lim.ld (link/load/entry base = LIM_BASE = 0x08000000).
 */

.section .text.start, "ax"
.global _start
_start:
    /* gp (disable relaxation around the la) */
    .option push
    .option norelax
    la    gp, __global_pointer$
    .option pop

    /* trap vector */
    la    t0, _trap_handler
    csrw  mtvec, t0

    /* clear Feature Disable CSR 0x7c1 */
    csrw  0x7c1, zero

    /* mstatus.VS = Initial (bit 9) */
    li    t0, (1 << 9)
    csrs  mstatus, t0

    /* per-hart stack: sp = _stack_top - hartid * STACK_SIZE (4 KiB).
     * 4 KiB stride keeps all 4 harts' stacks inside the LIM region (4 x 4 KiB
     * = 16 KiB below _stack_top); the FWs use little stack (no recursion). */
    csrr  a0, mhartid
    la    sp, _stack_top
    li    t0, 0x1000
    mul   t1, a0, t0
    sub   sp, sp, t1

    /* zero .bss on hart 0 only (a0 = mhartid preserved) */
    bnez  a0, _skip_bss
    la    t0, _bss_start
    la    t1, _bss_end
1:  bgeu  t0, t1, 2f
    sd    zero, 0(t0)
    addi  t0, t0, 8
    j     1b
2:
_skip_bss:
    /* main(hartid) -- a0 already = mhartid */
    call  main

3:  j     3b

/*
 * Trap handler: record mcause/mepc/mtval to fixed LIM diag addresses
 * (host-readable) then CEASE. Uses only t0/t1.
 */
.global _trap_handler
_trap_handler:
    csrr  t0, mcause
    la    t1, _trap_mcause
    sd    t0, 0(t1)
    csrr  t0, mepc
    la    t1, _trap_mepc
    sd    t0, 0(t1)
    csrr  t0, mtval
    la    t1, _trap_mtval
    sd    t0, 0(t1)
    /* CEASE (SiFive custom, 0x30500073): halt this hart */
    .word 0x30500073
    j     .
