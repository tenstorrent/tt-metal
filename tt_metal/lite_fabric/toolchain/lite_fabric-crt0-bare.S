.section .start,"ax",@progbits
.global _start
.type   _start, @function

_start:
  tail    main
  .size  _start, .-_start
