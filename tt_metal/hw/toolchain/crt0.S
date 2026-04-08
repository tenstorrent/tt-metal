// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

	.section .start,"ax",@progbits
	.global _start
	.type _start,@function
	/* s1 is coordinates.  */
_start:
#if 0 /* TODO: Enable gprel in quasar runtime #32892  */
	.option push
	.option norelax
	// Initialize global pointer,
	lui gp,%hi(__global_pointer$)
	addi gp,gp,%lo(__global_pointer$)
	.option pop
#endif

	/* Compute TP and SP at either end of the locals */
	lui	tp,%hi(__local_base)
	addi	tp,tp,%lo(__local_base)
	lui	sp,%hi(__local_stride)
	addi	sp,sp,%lo(__local_stride)
// for TRISCs __local_base and __local_stride are already set by the linker to the correct internal memory addresses
#ifndef COMPILE_FOR_TRISC
	csrr	s2,mhartid
	mul	s2,sp,s2
	add	tp,tp,s2
#endif
	add	sp,tp,sp

	/* Enter high-level start.  */
	tail _start1

	.size _start, .-_start
