// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include <dev_mem_map.h>

.section .text

.extern halt_stack_ptr_save
.global notify_brisc_and_halt
.global ncrisc_resume

/*
  brisc copies ncrisc's kernel to iram
  copying to iram requires that ncrisc goes into reset (asserts reset)
  this code saves the register state on the stack, and the stack
  ptr to a well-known-address and notifies brisc it is done

  no need to save caller saves regs or the gp or tp
*/

/* save 13 registers */
#define CONTEXT_SIZE (13 * 4)

.align 4
.func
notify_brisc_and_halt:
	/* Save context */
	addi sp, sp, -CONTEXT_SIZE
	sw x1, 0 * 4( sp )
	sw x8, 1 * 4( sp )
	sw x9, 2 * 4( sp )
	sw x18, 3 * 4( sp )
	sw x19, 4 * 4( sp )
	sw x20, 5 * 4( sp )
	sw x21, 6 * 4( sp )
	sw x22, 7 * 4( sp )
	sw x23, 8 * 4( sp )
	sw x24, 9 * 4( sp )
	sw x25, 10 * 4( sp )
	sw x26, 11 * 4( sp )
	sw x27, 12 * 4( sp )
	sw  sp, MEM_NCRISC_HALT_STACK_MAILBOX_ADDRESS( zero )
	sb x10, MEM_SLAVE_RUN_MAILBOX_ADDRESS( zero )   /* Tell brisc we're done */

	/*Program flow will end here as ncrisc will go into reset*/
halt:
	j halt

.endfunc

/*
  ncrisc fw saves the address of this routine so brisc can start ncrisc here
  when a new kernel is ready to run
*/
.align 4
.func
ncrisc_resume:
    .option push
    .option norelax
	// Initialize global pointer,
	// Use an absolute reloc, so text placement is irrelevant.
	lui gp,%hi(__global_pointer$)
	addi gp,gp,%lo(__global_pointer$)
    .option pop
	lw  sp, MEM_NCRISC_HALT_STACK_MAILBOX_ADDRESS( zero )
	/* Restore context */
	lw  x1, 0 * 4( sp )
	lw  x8, 1 * 4( sp )
	lw  x9, 2 * 4( sp )
	lw  x18, 3 * 4( sp )
	lw  x19, 4 * 4( sp )
	lw  x20, 5 * 4( sp )
	lw  x21, 6 * 4( sp )
	lw  x22, 7 * 4( sp )
	lw  x23, 8 * 4( sp )
	lw  x24, 9 * 4( sp )
	lw  x25, 10 * 4( sp )
	lw  x26, 11 * 4( sp )
	lw  x27, 12 * 4( sp )

	addi sp, sp, CONTEXT_SIZE

	ret
.endfunc
