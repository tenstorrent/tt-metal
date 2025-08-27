// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <dev_mem_map.h>

.section .start,"ax",@progbits

// This file is used on NCRISC on wormhole, and reloads registers stored in ncrisc-halt-wormhole.S.

.global _start
.type   _start, @function
.func	_start
_start:
	// kernel_launch is responsible for the rest of crt -- clear bss, copy data image, run global constructors
	.option push
	.option norelax
	lui gp,%hi(__global_pointer$)
	addi gp,gp,%lo(__global_pointer$)
	.option pop
	lw  sp, MEM_NCRISC_HALT_STACK_MAILBOX_ADDRESS( zero )
	lw  a0, 13 * 4( sp )
	call  kernel_launch
	// Store function return (kernel stack usage)
	sw a0, 13 * 4(sp)
	
	lw ra, 14 * 4(sp)
	sw ra,  MEM_NCRISC_RESUME_ADDR_MAILBOX_ADDRESS( zero )
	sb zero, MEM_SUBORDINATE_RUN_MAILBOX_ADDRESS( zero )   /* Tell brisc we're done */

halt:
	/*Program flow will end here as ncrisc will go into reset*/
	j halt
	.size  _start, .-_start
.size _start, .-_start
.endfunc
