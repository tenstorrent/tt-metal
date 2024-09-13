.section .init
.global _start
.type   _start, @function

_start:

CRT_START:
  # Initialize global pointer
  .option push
  .option norelax

  tail    _Z13kernel_launchv
  .size  _start, .-_start
