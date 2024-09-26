.section .start,"ax",@progbits
.global _start
.type   _start, @function

_start:
	tail    _Z13kernel_launchv
	.size  _start, .-_start
