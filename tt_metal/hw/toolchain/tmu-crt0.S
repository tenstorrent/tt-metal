.section .init
.global _start
.type   _start, @function

_start:
.option push
.option norelax
	// Initialize global pointer,
	// Use an absolute reloc, so text placement is irrelevant.
	lui gp,%hi(__global_pointer$)
	addi gp,gp,%lo(__global_pointer$)
.option pop

  /* set stack pointer */
  la sp, __stack_top

  # Clear the bss segment
  la      a0, __ldm_bss_start
  la      a1, __ldm_bss_end
  call    wzerorange

  la      s2, __init_array_start
  la      s3, __init_array_end
  j       2f
1:lw      a0, 0(s2)
  jalr    a0
  addi    s2, s2, 4
2:bne     s2, s3, 1b

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
