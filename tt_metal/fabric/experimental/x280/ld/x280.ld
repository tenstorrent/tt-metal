/*
 * x280.ld - Bare-metal linker script for SiFive X280 on Blackhole
 *           (active-FW region).
 *
 * Places all code and data starting just above the resident idle FW,
 * which owns the lowest 4 KiB of LIM (0x08000000..0x08000FFF). See
 * x280/ld/x280-lim-idle.ld, x280/include/x280_address_map.h, and
 * x280/include/x280_hart_protocol.h for the split and handoff protocol.
 *
 * Stack grows downward from top of allocation. Trap-diag block is
 * pinned to a fixed absolute LIM address (0x0811FF80) so the host
 * can read mcause / mepc / mtval at a known offset regardless of
 * build size.
 *
 * Reference: SiFive coreip_21G3.04.00 manual §5.18
 */
OUTPUT_ARCH(riscv)
ENTRY(_start)

/* X280 L3 LIM: total region [0x08000000, 0x081E0000), 1.875 MiB.
 *   [0x08000000, 0x08001000)   resident idle FW (separate linker)
 *   [0x08001000, 0x08120000)   active FW          (this script)
 *   [0x08130000, 0x08131B00)   per-hart protocol at 0x08131800
 *                              (x280_hart_protocol.h)
 *   [0x08131B00, 0x081E0000)   host runtime available_lim() (Legacy layout)
 * Active-FW capacity ~1.121 MiB. */
LIM_BASE = 0x08001000;
LIM_SIZE = 0x0011F000;   /* ~1.121 MiB */
STACK_SIZE = 0x8000;     /* 32 KiB per-hart */
NUM_HARTS  = 4;

/* Trap-diag: one 32 B block per hart, occupying the last 128 B of the region
 * at a fixed absolute address. Hart N is at TRAP_DIAG_ADDR + N * 0x20; the
 * host reads hart 0's block via X280_ACTIVE_TRAP_DIAG_ADDR (= 0x0811FF80). */
TRAP_DIAG_SIZE = 0x20;
TRAP_DIAG_ADDR = LIM_BASE + LIM_SIZE - NUM_HARTS * TRAP_DIAG_SIZE;  /* 0x0811FF80 */

MEMORY {
  lim (rwx) : ORIGIN = LIM_BASE, LENGTH = LIM_SIZE
}

SECTIONS {
  .text LIM_BASE : {
    KEEP(*(.text.start))   /* boot stub entry point must be first */
    *(.text .text.*)
    *(.rodata .rodata.*)
    . = ALIGN(8);
  } > lim

  /* GP convention: DATA_START + 0x7f0 (SiFive §5.18) */
  PROVIDE(__global_pointer$ = . + 0x7f0);

  _data_lma = LOADADDR(.data);
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

  /* Trap-diag at FIXED absolute address (matches
   * X280_ACTIVE_TRAP_DIAG_ADDR in x280_address_map.h). entry.S indexes it by
   * mhartid; _probe_active names hart 0's probe word for C code that arms it. */
  _trap_block      = TRAP_DIAG_ADDR;
  _trap_block_size = TRAP_DIAG_SIZE;
  _probe_active    = TRAP_DIAG_ADDR + 0x18;

  /* Stacks below the trap-diag blocks, at the top of the LIM allocation.
   * entry.S reads _stack_size and subtracts hartid * _stack_size. */
  . = ALIGN(16);
  _stack_top = TRAP_DIAG_ADDR;
  _stack_size = STACK_SIZE;

  /DISCARD/ : {
    *(.eh_frame .eh_frame_hdr)
    *(.note.GNU-stack)
  }
}

/* Sanity: code+data+bss must leave room for every hart's stack, not just
 * hart 0's. This is the bound x280_state.cpp enforces on a loaded image. */
ASSERT(_bss_end <= (_stack_top - NUM_HARTS * STACK_SIZE),
       "x280.ld: active-FW code/data/bss overflow into stack/trap-diag")

/* Images built with X280_PER_HART_PROTOCOL zero no .bss -- entry.S has no
 * hart-0 zeroing loop for them, because nothing holds the other harts behind it. Their link
 * rules pass --defsym=X280_REQUIRE_NO_BSS=1 to hold the precondition here. */
ASSERT(DEFINED(X280_REQUIRE_NO_BSS) ? (_bss_end == _bss_start) : 1,
       "x280.ld: per-hart images must carry no .bss -- entry.S zeroes none")

/* Packet injection profile requires image to end at or below the lower arena boundary */
ASSERT(DEFINED(X280_PACKET_INJECTION_LAYOUT) ? (_bss_end <= 0x08020000) : 1,
       "x280.ld: packet injection active-FW image overflows 0x08020000 lower arena bound")

/* entry.S writes this into mtvec, whose low two bits are the MODE field, so a
 * misaligned handler selects a reserved mode and traps never arrive. */
ASSERT((_trap_handler & 3) == 0,
       "x280.ld: _trap_handler must be 4-byte aligned for mtvec MODE=Direct")
