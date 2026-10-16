/*
 * x280-gddr.ld - Bare-metal linker script for SiFive X280 on Blackhole
 *                (active-FW region, PLACEMENT gddr).
 *
 * Places code, read-only data, data and every hart's stack in the local
 * GDDR tile through its cached alias, in the region x280_address_map.h
 * describes as X280_ACTIVE_GDDR_*. x280_image.cmake runs this script through
 * the C preprocessor, so the constants come from that header.
 * The trap-diag block stays at its fixed LIM address, as in x280.ld.
 *
 * Reference: SiFive coreip_21G3.04.00 manual §5.18
 */
#include "x280_address_map.h"

OUTPUT_ARCH(riscv)
ENTRY(_start)

REGION_BASE  = X280_ACTIVE_GDDR_REGION_BASE;
REGION_SIZE  = X280_ACTIVE_GDDR_REGION_SIZE;
LOAD_ADDR    = X280_ACTIVE_GDDR_LOAD_ADDR;
STACK_STRIDE = X280_ACTIVE_GDDR_STACK_STRIDE;
NUM_HARTS    = 4;

TRAP_DIAG_SIZE = X280_TRAP_DIAG_STRIDE;
TRAP_DIAG_ADDR = X280_ACTIVE_TRAP_DIAG_ADDR;

MEMORY {
  gddr (rwx) : ORIGIN = LOAD_ADDR, LENGTH = REGION_BASE + REGION_SIZE - LOAD_ADDR
}

SECTIONS {
  .text LOAD_ADDR : {
    KEEP(*(.text.start))   /* boot stub entry point must be first */
    *(.text .text.*)
    *(.rodata .rodata.*)
    . = ALIGN(8);
  } > gddr

  /* GP convention: DATA_START + 0x7f0 (SiFive §5.18) */
  PROVIDE(__global_pointer$ = . + 0x7f0);

  _data_lma = LOADADDR(.data);
  .data : {
    _data_start = .;
    *(.data .data.*)
    *(.sdata .sdata.*)
    . = ALIGN(8);
    KEEP(*(.x280_image_tag))   /* entry.S's image tag: the last bytes of the file */
    _data_end = .;
  } > gddr

  .bss (NOLOAD) : {
    _bss_start = .;
    *(.bss .bss.*)
    *(.sbss .sbss.*)
    *(COMMON)
    . = ALIGN(8);
    _bss_end = .;
  } > gddr

  _trap_block      = TRAP_DIAG_ADDR;
  _trap_block_size = TRAP_DIAG_SIZE;
  _probe_active    = TRAP_DIAG_ADDR + 0x18;

  /* Hart N's stack tops at _stack_top - N * _stack_size; entry.S reads
   * _stack_size as the stride. */
  _stack_top = REGION_BASE + REGION_SIZE;
  _stack_size = STACK_STRIDE;

  /DISCARD/ : {
    *(.eh_frame .eh_frame_hdr)
    *(.note.GNU-stack)
  }
}

/* Code+data+bss must leave room for every hart's stack. This is the bound
 * x280_state.cpp enforces on a loaded image. */
ASSERT(_bss_end <= X280_ACTIVE_GDDR_IMAGE_END,
       "x280-gddr.ld: active-FW code/data/bss overflow into the per-hart stacks")

/* Images built with X280_PER_HART_PROTOCOL zero no .bss -- entry.S has no
 * hart-0 zeroing loop for them. */
ASSERT(DEFINED(X280_REQUIRE_NO_BSS) ? (_bss_end == _bss_start) : 1,
       "x280-gddr.ld: per-hart images must carry no .bss -- entry.S zeroes none")

/* entry.S writes this into mtvec, whose low two bits are the MODE field, so a
 * misaligned handler selects a reserved mode and traps never arrive. */
ASSERT((_trap_handler & 3) == 0,
       "x280-gddr.ld: _trap_handler must be 4-byte aligned for mtvec MODE=Direct")
