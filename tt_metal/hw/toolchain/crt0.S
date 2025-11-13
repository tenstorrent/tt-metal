
	.section .start,"ax",@progbits
	.global _start
	.type _start,@function
	/* s1 is coordinates.  */
_start:
#if 0 /* FIXME: Enable global pointer at some point.  */
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
	csrr	s2,mhartid
	mul	s2,sp,s2
	add	tp,tp,s2
	add	sp,tp,sp

	/* Just in case someone thinks main has useful argument values.  */
	mv a0,zero
	mv a1,zero
	mv a2,zero
	call main
	tail exit

	.size _start, .-_start
