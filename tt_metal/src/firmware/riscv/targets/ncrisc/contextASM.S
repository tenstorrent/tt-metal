
/* context is 28 registers */
#define CONTEXT_SIZE ( 28 * 4)
#define BRISC_MAILBOX 0x5020

.global context_save
.global context_restore
.global call_with_cpu_flush_args
.global call_with_cpu_flush_args2
.global call_with_cpu_flush
.extern pContext
.extern RtosScheduler

.align 4
.func
call_with_cpu_flush_args:
	j call_with_cpu_flush

.endfunc

.align 4
.func
call_with_cpu_flush_args2:
	j call_with_cpu_flush

.endfunc

.align 4
.func
call_with_cpu_flush:
	/* Save Context of Running Task */
	addi sp, sp, -CONTEXT_SIZE
	sw x1, 0 * 4( sp )
	sw x5, 1 * 4( sp )
	sw x6, 2 * 4( sp )
	sw x7, 3 * 4( sp )
	sw x8, 4 * 4( sp )
	sw x9, 5 * 4( sp )
	sw x10, 6 * 4( sp )
	sw x11, 7 * 4( sp )
	sw x12, 8 * 4( sp )
	sw x13, 9 * 4( sp )
	sw x14, 10 * 4( sp )
	sw x15, 11 * 4( sp )
	sw x16, 12 * 4( sp )
	sw x17, 13 * 4( sp )
	sw x18, 14 * 4( sp )
	sw x19, 15 * 4( sp )
	sw x20, 16 * 4( sp )
	sw x21, 17 * 4( sp )
	sw x22, 18 * 4( sp )
	sw x23, 19 * 4( sp )
	sw x24, 20 * 4( sp )
	sw x25, 21 * 4( sp )
	sw x26, 22 * 4( sp )
	sw x27, 23 * 4( sp )
	sw x28, 24 * 4( sp )
	sw x29, 25 * 4( sp )
	sw x30, 26 * 4( sp )
	sw x31, 27 * 4( sp )

	lw  t0, pContext			/* Load pointer to context. */
	sw  sp, 0( t0 )				/* save context stack pointer */
	li  t0, BRISC_MAILBOX
	la  t1, context_restore
	sw  t1, 4(t0)
	li   t1, 0x1
	sw  t1, 0(t0)

	/*Signal to BRISC to reset NCRISC*/
	/*Set the reset vector for ncrisc to restore_context*/
	/*Save stack pointer to pCpuSP*/
	/*Program flow will end here as ncrisc will go into reset*/
halt:
	j halt

.endfunc

.section code_l1
.align 4
.func
context_restore:
  # Initialize global pointer
    .option push
    .option norelax
1:  auipc gp, %pcrel_hi(__global_pointer$)
    addi  gp, gp, %pcrel_lo(1b)
    .option pop
	lw  t1, pContext		/* Load address where CpuSP is saved */
	lw  sp, 0( t1 )			/* Initialize riscv SP */
	/* Restore Context */
	lw  x1, 0 * 4( sp )		/* ra */
	lw  x5, 1 * 4( sp )		/* t0 */
	lw  x6, 2 * 4( sp )		/* t1 */
	lw  x7, 3 * 4( sp )		/* t2 */
	lw  x8, 4 * 4( sp )		/* s0/fp */
	lw  x9, 5 * 4( sp )		/* s1 */
	lw  x10, 6 * 4( sp )	/* a0 */
	lw  x11, 7 * 4( sp )	/* a1 */
	lw  x12, 8 * 4( sp )	/* a2 */
	lw  x13, 9 * 4( sp )	/* a3 */
	lw  x14, 10 * 4( sp )	/* a4 */
	lw  x15, 11 * 4( sp )	/* a5 */
	lw  x16, 12 * 4( sp )	/* a6 */
	lw  x17, 13 * 4( sp )	/* a7 */
	lw  x18, 14 * 4( sp )	/* s2 */
	lw  x19, 15 * 4( sp )	/* s3 */
	lw  x20, 16 * 4( sp )	/* s4 */
	lw  x21, 17 * 4( sp )	/* s5 */
	lw  x22, 18 * 4( sp )	/* s6 */
	lw  x23, 19 * 4( sp )	/* s7 */
	lw  x24, 20 * 4( sp )	/* s8 */
	lw  x25, 21 * 4( sp )	/* s9 */
	lw  x26, 22 * 4( sp )	/* s10 */
	lw  x27, 23 * 4( sp )	/* s11 */
	lw  x28, 24 * 4( sp )	/* t3 */
	lw  x29, 25 * 4( sp )	/* t4 */
	lw  x30, 26 * 4( sp )	/* t5 */
	lw  x31, 27 * 4( sp )	/* t6 */
	addi sp, sp, CONTEXT_SIZE

	/* x10 is a0 which is the argument for function call*/
	/* argument contains the pointer to the funciton to be called*/
	jr x10
	.endfunc
