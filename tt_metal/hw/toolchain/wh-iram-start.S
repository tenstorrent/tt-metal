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
	lw  ra, 0 * 4( sp )
	lw  s0, 1 * 4( sp )
	lw  s1, 2 * 4( sp )
	lw  s2, 3 * 4( sp )
	lw  s3, 4 * 4( sp )
	lw  s4, 5 * 4( sp )
	lw  s5, 6 * 4( sp )
	lw  s6, 7 * 4( sp )
	lw  s7, 8 * 4( sp )
	lw  s8, 9 * 4( sp )
	lw  s9, 10 * 4( sp )
	lw  s10, 11 * 4( sp )
	lw  s11, 12 * 4( sp )
	lw  a0, 13 * 4( sp )
	addi sp, sp, (16 * 4)
	tail    kernel_launch
.size _start, .-_start
.endfunc
