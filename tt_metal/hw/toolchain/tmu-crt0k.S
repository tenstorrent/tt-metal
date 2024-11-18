.section .start,"ax",@progbits
.global _start
.type   _start, @function

_start:
	// kernel_launch is responsible for the rest of crt -- clear bss, copy data image, run global constructors
	// Enable GPREL optimizations
	.reloc _start, R_RISCV_NONE, __global_pointer$
	tail    _Z13kernel_launchm
	.size  _start, .-_start
