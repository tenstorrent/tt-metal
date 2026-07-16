/*
 * x280-lim-idle.ld - Linker script for the resident X280 idle firmware.
 *
 * The idle FW owns the lowest 4 KiB of LIM (0x08000000..0x08000FFF) and is
 * loaded exactly once per chip-reset. The reset vector points here; after the
 * one-time release the L2CPU never re-enters reset -- the active FW (the
 * drainer, profzone) is loaded at 0x08001000 (ld/x280-lim.ld) and reached by
 * an indirect JUMP from this idle FW. See tools/x280_bm/include/x280_boot.h.
 *
 * Memory map:
 *   [0x08000000, 0x08000FE0)  .text/.rodata/.data/.bss + small scratch stack
 *   [0x08000FE0, 0x08001000)  trap-diag block (mcause/mepc/mtval, 32 B)
 *
 * The idle FW runs C on all 4 harts (hart 0 arms the mailboxes; harts 1..3
 * gate on the heartbeat then poll their per-hart wake), but uses only a tiny
 * stack (no recursion, a few locals) so 4 KiB is ample. entry.S uses a 4 KiB
 * per-hart stride from _stack_top; here the harts touch almost no stack.
 */
OUTPUT_ARCH(riscv)
ENTRY(_start)

LIM_IDLE_BASE = 0x08000000;
LIM_IDLE_LEN  = 0x00001000;   /* 4 KiB */
STACK_SIZE    = 0x00000100;   /* 256 B (guard reservation) */

TRAP_DIAG_ADDR = LIM_IDLE_BASE + LIM_IDLE_LEN - 0x20;   /* 0x08000FE0 */

MEMORY {
  lim_idle (rwx) : ORIGIN = LIM_IDLE_BASE, LENGTH = LIM_IDLE_LEN
}

SECTIONS {
  .text LIM_IDLE_BASE : {
    KEEP(*(.text.start))      /* boot stub entry must be first */
    *(.text .text.*)
    *(.rodata .rodata.*)
    . = ALIGN(8);
  } > lim_idle

  /* GP convention: DATA_START + 0x7f0 (SiFive S5.18) */
  PROVIDE(__global_pointer$ = . + 0x7f0);

  .data : {
    _data_start = .;
    *(.data .data.*)
    *(.sdata .sdata.*)
    . = ALIGN(8);
    _data_end = .;
  } > lim_idle

  .bss (NOLOAD) : {
    _bss_start = .;
    *(.bss .bss.*)
    *(.sbss .sbss.*)
    *(COMMON)
    . = ALIGN(8);
    _bss_end = .;
  } > lim_idle

  /* Trap-diag at fixed absolute addresses (matches entry.S references). */
  _trap_mcause = TRAP_DIAG_ADDR + 0x00;
  _trap_mepc   = TRAP_DIAG_ADDR + 0x08;
  _trap_mtval  = TRAP_DIAG_ADDR + 0x10;

  . = ALIGN(16);
  _stack_top = TRAP_DIAG_ADDR;

  /DISCARD/ : {
    *(.eh_frame .eh_frame_hdr)
    *(.note.GNU-stack)
  }
}

ASSERT(_bss_end <= (TRAP_DIAG_ADDR - STACK_SIZE),
       "x280-lim-idle.ld: idle FW code/data/bss overflow the 4 KiB region")
