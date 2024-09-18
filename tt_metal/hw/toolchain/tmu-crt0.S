.section .init
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

	// set stack pointer
	lui	sp, %hi(__stack_top)
	addi	sp, sp, %lo(__stack_top)

	// Clear bss
	lui	a0, %hi(__ldm_bss_start)
	addi	a0, a0, %lo(__ldm_bss_start)
	lui	a1, %hi(__ldm_bss_end)
	addi	a1, a1, %lo(__ldm_bss_end)
	call    wzerorange

	// Run global initializers
	lui	s2, %hi(__init_array_start)
	addi	s2, s2, %lo(__init_array_start)
	lui	s3, %hi(__init_array_end)
	addi	s3, s3, %lo(__init_array_end)
	beq	s2, s3, 2f
1:	lw      a0, 0(s2)
	jalr    a0
	addi  	s2, s2, 4
	bne	s2, s3, 1b
2:

  /* Pass in the tensix coordinates as argv[0][0] through argv[0][3].
     argc = 1, envp = NULL. In memory, we'll have
   * sp+0: argv[0] -> sp+8
   * sp+4: argv[1] = NULL
   * sp+8: s1
   * sp+c: 0
   */
  addi    sp, sp, -16 /* (stack is aligned to 16 bytes in riscv calling convention) */
  addi    a0, sp, 8
  sw      a0, 0(sp)
  sw      zero, 4(sp)
  sw      s1, 8(sp)
  sw      zero, 12(sp)

  li      a0, 1 # argc = 1
  mv      a1, sp
  mv      a2, zero

  call    main
  tail    exit
  .size  _start, .-_start
