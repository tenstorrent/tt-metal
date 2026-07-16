/*
 * x280-lim.ld - Linker script for the LIM-resident drainer FW (now the
 *               "active FW" in the idle-FW + JUMP-handoff scheme).
 *
 * Link / load base = LIM_BASE = 0x08001000. The lowest 4 KiB of LIM
 * (0x08000000..0x08000FFF) is owned by the resident idle FW (ld/x280-lim-idle.ld);
 * the reset vector points there, and this drainer is loaded at 0x08001000 and
 * reached by an indirect JUMP from the idle FW -- so its .text.start (_start)
 * is at 0x08001000, NOT 0x08000000. See tools/x280_bm/include/x280_boot.h.
 *
 * The managed region still ends at 0x08010000 (top unchanged), so the HB
 * counter (0x08010000) and the mailboxes/mirrors above are untouched -- only
 * the base moved up one 4 KiB page. Actual code+data+bss is ~5.5 KiB, so the
 * ~60 KiB region (minus the 32 KiB stack reservation) is far more than enough.
 *
 * Memory map:
 *   [0x08000000, 0x08001000)  resident idle FW (separate linker)
 *   [0x08001000, 0x0800FFE0)  .text/.rodata/.data/.bss + stack (grows down)
 *   [0x0800FFE0, 0x08010000)  trap-diag block (mcause/mepc/mtval, 32 B)
 *    0x08010000               HB_COUNTER_ADDR (FW heartbeat counter, free SRAM)
 */
OUTPUT_ARCH(riscv)
ENTRY(_start)

LIM_BASE   = 0x08001000;
LIM_LEN    = 0x0000F000;   /* 60 KiB managed region (idle FW owns the low 4 KiB) */
STACK_SIZE = 0x00008000;   /* 32 KiB per-hart reservation (real stacks use 4x4 KiB) */

/* Trap-diag block: last 32 B of the managed region, fixed absolute address
 * so the host can read mcause/mepc/mtval at a known offset. */
TRAP_DIAG_ADDR = LIM_BASE + LIM_LEN - 0x20;   /* 0x0800FFE0 */

MEMORY {
  lim (rwx) : ORIGIN = LIM_BASE, LENGTH = LIM_LEN
}

SECTIONS {
  .text LIM_BASE : {
    KEEP(*(.text.start))      /* boot stub entry must be first */
    *(.text .text.*)
    *(.rodata .rodata.*)
    . = ALIGN(8);
  } > lim

  /* GP convention: DATA_START + 0x7f0 (SiFive S5.18) */
  PROVIDE(__global_pointer$ = . + 0x7f0);

  .data : {
    _data_start = .;
    *(.data .data.*)
    *(.sdata .sdata.*)
    . = ALIGN(8);
    _data_end = .;
  } > lim

  .bss (NOLOAD) : {
    _bss_start = .;
    *(.bss .bss.*)
    *(.sbss .sbss.*)
    *(COMMON)
    . = ALIGN(8);
    _bss_end = .;
  } > lim

  /* Trap-diag at fixed absolute addresses (matches entry.S references). */
  _trap_mcause = TRAP_DIAG_ADDR + 0x00;
  _trap_mepc   = TRAP_DIAG_ADDR + 0x08;
  _trap_mtval  = TRAP_DIAG_ADDR + 0x10;

  /* Stack at the top of the managed region, just below the trap-diag block. */
  . = ALIGN(16);
  _stack_top = TRAP_DIAG_ADDR;

  /DISCARD/ : {
    *(.eh_frame .eh_frame_hdr)
    *(.note.GNU-stack)
  }
}

ASSERT(_bss_end <= (TRAP_DIAG_ADDR - STACK_SIZE),
       "x280-lim.ld: code/data/bss overflow into stack/trap-diag")
