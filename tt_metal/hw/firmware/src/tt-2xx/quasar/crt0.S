#include "asm_tb.h"
#include "riscv-pk/encoding.h"

    .section .text.init
    .global _start
    .type _start, @function
_start:
    .cfi_startproc

    li x1, 0
    li x2, 0
    li x3, 0
    li x4, 0
    li x5, 0
    li x6, 0
    li x7, 0
    li x8, 0
    li x9, 0
    li x10, 0
    li x11, 0
    li x12, 0
    li x13, 0
    li x14, 0
    li x15, 0
    li x16, 0
    li x17, 0
    li x18, 0
    li x19, 0
    li x20, 0
    li x21, 0
    li x22, 0
    li x23, 0
    li x24, 0
    li x25, 0
    li x26, 0
    li x27, 0
    li x28, 0
    li x29, 0
    li x30, 0
    li x31, 0

    li t0, (MSTATUS_FS | MSTATUS_XS)
    csrs mstatus, t0

    /* Initialize global pointer; disable relaxation to avoid relaxing
       the address calculation to "addi gp, gp, 0" */
    .option push
    .option norelax
    la gp, __global_pointer$
    .option pop

    /* Initialize thread pointer */
    csrr s0, mhartid
    lui t0, %hi(__stack_shift)
    addi t0, t0, %lo(__stack_shift)
    la tp, __stack_start
    sll t0, s0, t0
    add tp, tp, t0

    /* Initialize stack pointer */
    lui t0, %hi(__stack_size)
    addi t0, t0, %lo(__stack_size)
    add sp, tp, t0

    /* Initialize trap vector */
#ifdef METAL_LIB_INC
    la t0, early_trap_vector
#else
    la t0, early_trap_vector_overlay_lib
#endif
    csrw mtvec, t0

    /* Skip global initialization if not the designated boot hart */
    la t0, __boot_hart
    bne s0, t0, MAIN_CALL

    /* Zero BSS segment */
    la t0, __bss_start
    la t1, __bss_end
    bgeu t0, t1, MAIN_CALL
BSS_BRANCH:
    SREG zero, (t0)
    addi t0, t0, REGBYTES
    bltu t0, t1, BSS_BRANCH
MAIN_CALL:
    /* Call main function */
    call __main

    /* Post-main */
    wfi

    .cfi_endproc

    /*
    * For sanity's sake we set up an early trap vector that just does nothing.
    * If you end up here then there's a bug in the early boot code somewhere.
    */
    .section .text.init.trapvec
    .global early_trap_vector_overlay_lib
    .align 2
    early_trap_vector_overlay_lib:
        .cfi_startproc
        csrr t0, mcause
        csrr t1, mepc
        csrr t2, mtval
        j early_trap_vector_overlay_lib
        .cfi_endproc
