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

  .global _init
  .type   _init, @function
  .global _fini
  .type   _fini, @function
_init:
_fini:
  # These don't have to do anything since we use init_array/fini_array.
  ret
  .size  _init, .-_init
  .size  _fini, .-_fini
