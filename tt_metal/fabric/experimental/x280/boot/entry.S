/*
 * entry.S - bare-metal boot stub for SiFive X280 on Blackhole
 *
 * Boot sequence (SiFive coreip_21G3.04.00 manual §5.17):
 *   1. Initialize gp (global pointer for relaxation)
 *   2. Set mtvec to diagnostic trap handler
 *   3. Clear Feature Disable CSR 0x7c1
 *   3b. Enable vector unit (mstatus.VS = Initial, bit 9)
 *   4. Per-hart stack pointer: sp = _stack_top - hartid * _stack_size
 *   5. Zero BSS section (hart 0 only; absent under X280_PER_HART_PROTOCOL,
 *      whose images carry no .bss)
 *   6. Call main(hartid) -- a0 = mhartid for all harts
 *   7. Infinite loop if main returns
 *
 * Trap handler supports three modes:
 *   IPI MODE (mcause=0x8000000000000003): machine software interrupt.
 *     Clears msip[hartid], writes DDR ack, mret.
 *   PROBE MODE (this hart's probe state != 0): skip access faults (5, 7),
 *     set it to 2, mret.
 *   NORMAL MODE: write trap diagnostics for host readback, CEASE. Under
 *     X280_PER_HART_PROTOCOL that is this hart's trap record; otherwise this
 *     hart's linker-placed trap block.
 */

#include "x280_hart_protocol.h"

.section .text.start, "ax"
.global _start

_start:
    /* Global-pointer initialization must disable relaxation. */
    .option push
    .option norelax
    la    gp, __global_pointer$
    .option pop

    /* The handler's own alignment selects mtvec.MODE -- see the .balign 4 on
     * _trap_handler below. */
    la    t0, _trap_handler
    csrw  mtvec, t0

    /* Enable full microarchitecture features (SiFive §5.17 step 4). */
    csrw  0x7c1, zero

    /* mstatus.VS = bits[10:9]. At reset VS=0b00 (Off) -- any vector instruction traps. */
    /* Write bit[9] = 1 to set VS = Initial. Must precede any vector instruction. */
    /* Source: SiFive coreip_21G3.04.00 §5.8 */
    li    t0, (1 << 9)
    csrs  mstatus, t0

    /* sp = _stack_top - hartid * _stack_size. The stride is the linking
     * script's own STACK_SIZE, because the regions differ by two orders of
     * magnitude: 32 KiB per hart in the active-FW images, 256 B in
     * the 4 KiB resident idle. A fixed stride puts three of the idle image's
     * four stacks outside its region. %hi/%lo rather than `la`, since the
     * stride is an absolute value, not an address. */
    /* a0 = mhartid (preserved through BSS zero; passed to main as first argument) */
    csrr  a0, mhartid
    la    sp, _stack_top
    lui   t0, %hi(_stack_size)
    addi  t0, t0, %lo(_stack_size)
    mul   t1, a0, t0
    sub   sp, sp, t1              /* sp = _stack_top - hartid * _stack_size */

#ifndef X280_PER_HART_PROTOCOL
    /* Non-hart-0 harts skip BSS zero to prevent data corruption. Nothing
     * holds them behind hart 0 finishing, so an image whose harts read .bss
     * races the zeroing. Per-hart images carry no .bss (asserted at link) and
     * skip this step entirely. */
    /* BSS zero uses only t0/t1, so a0 (mhartid) is preserved. */
    bnez  a0, _hart_skip_bss

    la    t0, _bss_start
    la    t1, _bss_end
1:  bgeu  t0, t1, 2f
    sd    zero, 0(t0)
    addi  t0, t0, 8
    j     1b
2:

_hart_skip_bss:
#endif
#ifdef X280_ACTIVE_HARNESS
    /* The harness is noreturn, so the spin below is dead for these images. */
    call  x280_active_start
#else
    call  main
#endif

3:  j     3b


/*
 * Trap handler: three modes.
 *
 * IPI MODE (mcause = 0x8000000000000003, machine software interrupt):
 *   Clear msip[hartid] to acknowledge, write DDR ack flag, mret.
 *   Must come BEFORE probe check to allow harts 1-3 to receive IPIs.
 *
 * PROBE MODE (this hart's probe state != 0):
 *   If mcause is illegal instruction (2), load access fault (5), or
 *   store/AMO access fault (7), skip the faulting instruction (mepc += 4),
 *   set this hart's probe state to 2 ("fault occurred"), and mret.
 *   mcause=2 covers SiFive custom / CBO instructions not present in ISA.
 *   This enables safe probing of unknown hardware registers and instructions.
 *
 * NORMAL MODE (this hart's probe state == 0):
 *   Write mcause/mepc/mtval to this hart's known addresses for host
 *   readback, then halt via CEASE.
 *
 * Uses only caller-saved temporaries to avoid corrupting state.
 * IPI handler does NOT modify mstatus -- vector state is preserved.
 */
.global _trap_handler
/* mtvec's low two bits are its MODE field, so this address doubles as the mode
 * selector: 4-byte aligned is MODE=Direct, and 2 mod 4 is a reserved mode whose
 * masked base lands in the spin loop above. Compressed instructions in _start
 * make the alignment incidental, so it is forced here and asserted at link. */
.balign 4
_trap_handler:
    /* Check for machine software interrupt (IPI) first -- mcause has bit 63 set for interrupts */
    /* Machine software interrupt = interrupt bit (bit 63) set + cause 3 */
    csrr  t0, mcause
    li    t1, 3
    bltu  t0, t1, _trap_probe_check  /* if mcause < 3, not MSI (also handles synchronous) */
    beq   t0, t1, _trap_probe_check  /* if mcause == 3 exactly (synchronous cause 3), check probe */
    /* mcause > 3: could be interrupt. Check if it equals 0x8000000000000003 */
    li    t1, 0x8000000000000003
    beq   t0, t1, _trap_ipi

_trap_probe_check:
    /* This hart's probe-state word into t2, kept live for _trap_skip. One
     * shared word would let a hart probing an unknown register suppress the
     * fault report of another hart trapping for real. */
    csrr  t2, mhartid
#ifdef X280_PER_HART_PROTOCOL
    slli  t2, t2, X280_HART_RECORD_SHIFT
    li    t0, X280_HART_TRAP_ADDR(0) + X280_HART_TRAP_PROBE_STATE_OFFSET
    add   t2, t2, t0
#else
    lui   t0, %hi(_trap_block_size)
    addi  t0, t0, %lo(_trap_block_size)
    mul   t2, t2, t0
    la    t0, _trap_block
    add   t2, t2, t0
    addi  t2, t2, 0x18
#endif
    ld    t0, 0(t2)
    beqz  t0, _trap_normal

    /* Probe mode: check if this is a fault we can skip */
    csrr  t0, mcause
    li    t1, 2             /* illegal instruction (e.g. unsupported cbo.* / SiFive custom insn) */
    beq   t0, t1, _trap_skip
    li    t1, 5             /* load access fault */
    beq   t0, t1, _trap_skip
    li    t1, 7             /* store/AMO access fault */
    beq   t0, t1, _trap_skip
    /* Not a skippable fault -- fall through to normal handler */

_trap_normal:
#ifdef X280_PER_HART_PROTOCOL
    /* x280_hart_protocol_t groups one control and one trap record per hart. */
    csrr  t0, mhartid
    slli  t0, t0, X280_HART_RECORD_SHIFT
    li    t1, X280_HART_TRAP_ADDR(0)
    add   t1, t1, t0

    li    t2, X280_HART_CONTROL_ADDR(0)
    add   t2, t2, t0
    ld    t2, X280_HART_CONTROL_RUN_VALUE_OFFSET(t2)
    sd    t2, X280_HART_TRAP_RUN_VALUE_OFFSET(t1)

    csrr  t2, mcause
    sd    t2, X280_HART_TRAP_MCAUSE_OFFSET(t1)
    csrr  t2, mepc
    sd    t2, X280_HART_TRAP_MEPC_OFFSET(t1)
    csrr  t2, mtval
    sd    t2, X280_HART_TRAP_MTVAL_OFFSET(t1)
    /* probe_state needs no write: it is where this hart's probe state already
     * lives, and the host cleared it with the rest of the record. */
    fence ow, ow
    li    t2, 1
    sd    t2, X280_HART_TRAP_VALID_OFFSET(t1)
#else
    /* One linker-placed block per hart, at _trap_block + mhartid *
     * _trap_block_size. A single shared block would have the four harts these
     * images run overwrite each other's diagnostics. */
    csrr  t0, mhartid
    lui   t1, %hi(_trap_block_size)
    addi  t1, t1, %lo(_trap_block_size)
    mul   t0, t0, t1
    la    t1, _trap_block
    add   t1, t1, t0

    csrr  t0, mcause
    sd    t0, 0x00(t1)
    csrr  t0, mepc
    sd    t0, 0x08(t1)
    csrr  t0, mtval
    sd    t0, 0x10(t1)
#endif

    /* CEASE: halt the hart (SiFive custom, encoding 0x30500073) */
    .word 0x30500073

    /* Fallback: infinite loop if CEASE is not supported */
    j     .

_trap_skip:
    /* Mark that a fault occurred: this hart's probe state = 2. t2 still holds
     * its address from _trap_probe_check, the only path that reaches here. */
    li    t0, 2
    sd    t0, 0(t2)

    /* Skip faulting instruction: mepc += 4 */
    csrr  t0, mepc
    addi  t0, t0, 4
    csrw  mepc, t0

    mret

/*
 * _trap_ipi: Machine software interrupt handler (IPI from hart 0 via CLINT msip).
 *
 * Protocol:
 *   1. Disable MSIE (mie bit 3) via CSR clear -- prevents re-fire without touching CLINT MMIO.
 *      The CLINT msip register at 0x02000000 may not be mapped;
 *      clearing via MMIO store risks a store access fault (double-trap -> CEASE).
 *      Clearing mie.MSIE (bit 3) is a CSR write -- always accessible, same effect: suppresses MSI.
 *   2. Write X280_IPI_ACK_FLAG to X280_IPI_ACK_ADDR[hartid].
 *   3. mret to resume execution (WFI returns, or next instruction after WFI).
 *
 * Register usage: t0, t1 only (caller-saved). Does NOT modify mstatus.VS (vector state preserved).
 * Source: SiFive coreip_21G3.04.00 §8.5-8.6, Table 150.
 */
_trap_ipi:
    /* Step 1: Clear MSIE (mie bit 3) to suppress further machine software interrupts.
     * csrci mie, 8 clears bit 3 (MSIE) without touching mstatus or other mie bits.
     * This is equivalent to clearing msip at the CLINT but uses only CSR (no MMIO). */
    csrci mie, 8

    /* Step 2: Write this hart's IPI acknowledgement to DDR. */
    csrr  t0, mhartid
    li    t1, X280_IPI_ACK_ADDR
    slli  t0, t0, X280_IPI_ACK_RECORD_SHIFT
    add   t1, t1, t0           /* t1 = &X280_IPI_ACK_ADDR[hartid] */
    li    t0, X280_IPI_ACK_FLAG
    sd    t0, 0(t1)

    mret
