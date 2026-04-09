// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

.section .init
.global _start
.type   _start, @function

_start:

CRT_START:
  # Initialize global pointer
  .option push
  .option norelax
  la gp, __global_pointer$
  .option pop                          # Restore the previous assembler options

  la sp, __stack_top # Set stack pointer to the end of L0

  # Clear the bss segment
  la      t0, __ldm_bss_start          # Load the address of __ldm_bss_start into t0
  la      t1, __ldm_bss_end            # Load the address of __ldm_bss_end into t1
1:beq     t0, t1, 2f                   # If t0 == t1, break out of loop
  sw      zero, 0(t0)                  # Store 0 at address pointed by t0
  addi    t0, t0, 1                    # Move to the next word (32-bit) address
  j       1b                           # Jump back to label 1

  # Copy memory from .loader_init to .ldm_data
2:la      t0, __loader_init_start      # Load the address of __loader_init_start into t0
  la      t1, __loader_init_end        # Load the address of __loader_init_end into t1
  beq     t0, t1, 4f                   # If t0 == t1, skip; nothing should be loaded into local memory
  la      t1, __ldm_data_end           # Load the address of __ldm_data_end into t1
  la      t2, __ldm_data_start         # Load the address of __ldm_data_start into t2
3:beq     t2, t1, 4f                   # If t2 == t1, break out of loop
  lw      t3, 0(t0)                    # Load word from .loader_init
  sw      t3, 0(t2)                    # Store word to .ldm_data
  addi    t0, t0, 4                    # Move to the next word in .loader_init
  addi    t2, t2, 4                    # Move to the next word in .ldm_data
  j       3b                           # Jump back to label 3

  # Execute global initialization
4:la      s2, __init_array_start       # Load the address of __init_array_start into s2
  la      s3, __init_array_end         # Load the address of __init_array_end into s3
  j       6f                           # Jump to label 6
5:lw      a0, 0(s2)                    # Load word from address pointed by s2 into a0
  jalr    a0                           # Jump and link register to the address in a0
  addi    s2, s2, 4                    # Move to the next word in the init array
6:bne     s2, s3, 5b                   # If s2 != s3, jump back to label 5

  # We don't have any arguments that we pass to main, so no a0 for argc and a1 for argv need be initialized
  call    main

  # Automatically dump gcda into L1 when main returns if built with coverage.
  #ifdef COVERAGE
  call gcov_dump
  #endif

  # Loop infinitely.
  10: j 10b

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
