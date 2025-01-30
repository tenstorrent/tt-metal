// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include <dev_mem_map.h>

.section .text

.global notify_brisc_and_halt_to_iram

/*
  switching execution from L1 to IRAM requires that ncrisc goes into reset, or else strange behavior results.
  this code saves the register state on the stack, and the stack
  ptr to a well-known-address and notifies brisc it is done. The resume happens in tmu-crt0k-ncrisc.S.

  no need to save caller saves regs or the gp or tp.
*/

/* save 14 registers, but round up to keep 16 byte aligned */
#define CONTEXT_SIZE (16 * 4)

.align 4
.func
notify_brisc_and_halt_to_iram:
	/* Save context */
	addi sp, sp, -CONTEXT_SIZE
	sw x1, 0 * 4( sp ) // ra
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
	sw a1,  13 * 4( sp )
	sw  sp, MEM_NCRISC_HALT_STACK_MAILBOX_ADDRESS( zero )
	sb a0, MEM_SLAVE_RUN_MAILBOX_ADDRESS( zero )   /* Tell brisc we're done */

halt:
	/*Program flow will end here as ncrisc will go into reset*/
	j halt

.endfunc
