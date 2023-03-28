.section .init
.global _start
.type   _start, @function

_start:

# produce an empty word at a known location
  j 1f
.balign 4, 0 # in case j is 2-byte compressed insn
TEST_MAILBOX:
.global TEST_MAILBOX
.word 0

# Small mailbox to allow for logging from firmware
FWLOG_MAILBOX:
.global FWLOG_MAILBOX
.word 0
.word 0
.word 0
.word 0
.word 0
FWEVENT_MAILBOX:
.global FWEVENT_MAILBOX
.word 0

FW_TEST_RESULT:
.global FW_TEST_RESULT
.space 64

WALL_CLOCK_MAILBOX:
.global WALL_CLOCK_MAILBOX
.space 16 # 64-bit start, 64-bit end. AICLK cycles

DEBUG_MAILBOX:
.global DEBUG_MAILBOX
.space 256

CQ_MAILBOX:
.global CQ_MAILBOX
.space 4

# STOP. Do not add new fields here. Put them immediately before CRT_START.

FIRMWARE_VERSION:
.global FIRMWARE_VERSION
# Prior to the introduction of FIRMWARE_VERSION, this location was occupied by the auipc gp instruction that follows.
# Uncompressed instructions always have 11 as the low bits so pad the version with zeroes so that we can distinguish it
# from an instruction.
.word 1 << 2
.word 0x5da30992 # special magic for FIRMWARE_VERSION, this must immediately follow the version word.

CRT_START:
  # Initialize global pointer
  .option push
  .option norelax
1:auipc gp, %pcrel_hi(__global_pointer$)
  addi  gp, gp, %pcrel_lo(1b)

/* set stack pointer */
la sp, __stack_top

  # Clear the bss segment
  la      a0, __l1_bss_start
  la      a1, __l1_bss_end
  call    wzerorange

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
  .option pop

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
