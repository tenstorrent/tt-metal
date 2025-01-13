.section .start,"ax",@progbits
.global _start
.type   _start, @function

_start:
	// Use explicit lui/addi pairs here to generate absolute
	// addresses, these will get relaxed to gp-relative
	// computations where possible.

	.option push
	.option norelax
	// Initialize global pointer,
	lui gp,%hi(__global_pointer$)
	addi gp,gp,%lo(__global_pointer$)
	.option pop

	// set stack pointer, reserve 16 bytes for main's arguments
	lui	sp, %hi(__stack_top - 16)
	addi	sp, sp, %lo(__stack_top - 16)

	// main is responsible for the rest of crt -- clear bss, copy data image, run global constructors

  /* Pass in the tensix coordinates as argv[0][0] through argv[0][3].
     argc = 1, envp = NULL. In memory, we'll have
   * sp+0: argv[0] -> sp+8
   * sp+4: argv[1] = NULL
   * sp+8: s1
   * sp+c: 0
   */
  addi    a0, sp, 8
  sw      a0, 0(sp) // argv[0]
  sw      zero, 4(sp) // argv[1]
  sw      s1, 8(sp) // argv[0][0..3]
  sw      zero, 12(sp) // argv[0][4..7]

  li      a0, 1 // argc = 1
  mv      a1, sp // argv
  mv      a2, zero // env

  call    main
  tail    exit
  .size  _start, .-_start
