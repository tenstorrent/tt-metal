// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <dev_mem_map.h>

.section .start,"ax",@progbits
.global _start
.type   _start, @function

// This file is used on NCRISC on wormhole, and reloads registers stored in ncrisc-halt-wormhole.S.

_start:
	// kernel_launch is responsible for the rest of crt -- clear bss, copy data image, run global constructors
	// Enable GPREL optimizations
	.reloc _start, R_RISCV_NONE, __global_pointer$
    .option push
    .option norelax
	lui gp,%hi(__global_pointer$)
	addi gp,gp,%lo(__global_pointer$)
	.option pop
	lw  sp, MEM_NCRISC_HALT_STACK_MAILBOX_ADDRESS( zero )
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
	lw  a0, 13 * 4( sp )
	addi sp, sp, (16 * 4)
	tail    _Z13kernel_launchm
	.size  _start, .-_start
