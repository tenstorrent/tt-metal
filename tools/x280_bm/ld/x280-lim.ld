/*
 * x280-lim.ld - Linker script for the LIM-resident single-hart heartbeat FW.
 *
 * Link / load / entry base = LIM_BASE = 0x08000000 (L3 LIM SRAM, available
 * at reset with WayEnable=0). This matches the host loader, which NOC-writes
 * the .bin to 0x08000000 and sets the reset vector to 0x08000000.
 *
 * We bound the linker-managed region to the low 64 KiB of LIM so the
 * heartbeat counter the FW publishes (HB_COUNTER_ADDR = 0x08010000, just
 * above this region) is guaranteed clear of .text/.data/.bss/stack. The
 * X280's full 1.875 MiB LIM is larger; we only manage what the FW needs.
 *
 * Memory map:
 *   [0x08000000, 0x0800FFE0)  .text/.rodata/.data/.bss + stack (grows down)
 *   [0x0800FFE0, 0x08010000)  trap-diag block (mcause/mepc/mtval, 32 B)
 *    0x08010000               HB_COUNTER_ADDR (FW heartbeat counter, free SRAM)
 */
OUTPUT_ARCH(riscv)
ENTRY(_start)

LIM_BASE   = 0x08000000;
LIM_LEN    = 0x00010000;   /* 64 KiB managed region */
STACK_SIZE = 0x00008000;   /* 32 KiB per-hart */

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
