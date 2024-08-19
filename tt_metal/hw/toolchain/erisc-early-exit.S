// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

.section .text

.global erisc_save_sp_and_ra
.global erisc_early_exit

/*
    This file contains an early exit function for use in erisc kernels + erisc metal fw (not base
    layer fw). This needs to exist so that when watcher detects an illegal noc transaction it can
    exit from the kernel or metal fw. On brisc/ncrisc, watcher will hang the core, whichwill then
    be recovered when a new program is written. Because erisc doesn't get deasserted when a new
    program is written, we instead need to early exit to ensure no hang. We accomplish this by
    saving registers to the stack when metal fw starts and saving the stack pointer to a known
    address. Then in this early exit function we restore all registers and return to the base fw
    layer.
*/

/* save 13 registers */
#define CONTEXT_SIZE (13 * 4)

.align 4
.func
erisc_early_exit:
	lw  sp, 0( x10 )
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

    /* Directly return to the new return address. */
    ret
.endfunc
