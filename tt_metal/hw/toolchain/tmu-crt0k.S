.section .start,"ax",@progbits
.global _start
.type   _start, @function

_start:
	tail    _Z13kernel_launchm
	.size  _start, .-_start
