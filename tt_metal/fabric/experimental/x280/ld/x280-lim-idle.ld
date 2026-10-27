/*
 * x280-lim-idle.ld - Linker script for the resident X280 idle firmware.
 *
 * The idle firmware lives in the lowest 4 KiB of LIM and is loaded
 * exactly once per chip-reset by the host. After that initial load
 * the L2CPU never re-enters reset; subsequent firmwares (the "active
 * FW") are loaded into the upper
 * portion of LIM and are reached by an indirect jump from this idle
 * firmware. See:
 *   - x280/include/x280_address_map.h   (resident/active regions)
 *   - x280/include/x280_hart_protocol.h (per-hart lifecycle records)
 *   - x280/ld/x280.ld              (active-FW linker, LIM_BASE = 4 KiB)
 *
 * Memory budget:
 *
 *   [0x08000000, 0x08000F80)   text + rodata + data + bss + stack
 *                              (= 3968 B usable)
 *   [0x08000F80, 0x08001000)   trap-diag blocks (mcause / mepc /
 *                              mtval / probe state, 32 B per hart),
 *                              addresses pinned for host readback
 *
 * All four harts run the idle FW's C: entry.S calls main(mhartid) on
 * each, and each polls its own lifecycle record. It has no recursion
 * and no FP, and its four stacks are 256 B apiece inside the region.
 * The ASSERTs at the bottom refuse a build that overflows the budget or
 * carries .bss.
 */
OUTPUT_ARCH(riscv)
ENTRY(_start)

LIM_IDLE_BASE      = 0x08000000;
LIM_IDLE_SIZE      = 0x00001000;   /* 4 KiB */

/* Trap-diag is the last 128 B: one 32 B block (4 x u64) per hart, so four
 * harts do not race one set of words. The stacks start immediately below,
 * growing down, and are intentionally small (256 B each): the idle FW has no
 * recursion and a few local u64s only. */
NUM_HARTS          = 4;
TRAP_DIAG_SIZE     = 0x20;                                    /* per hart */
TRAP_DIAG_ADDR     = LIM_IDLE_BASE + LIM_IDLE_SIZE
                     - NUM_HARTS * TRAP_DIAG_SIZE;            /* 0x08000F80 */
STACK_TOP          = TRAP_DIAG_ADDR;                          /* growing down */
STACK_SIZE         = 0x100;                                   /* 256 B per hart */

/* Host-only ECC-prime marker slot (see x280_address_map.h).
 * The host prime seeds a magic u64 here and reads it back after the ASIC reset
 * to detect an already-primed machine. It must stay clear of the idle FW's
 * code/data/bss; the ASSERT below guards that. The four stacks at
 * 0x0B80..0x0F80 are above this slot, so no stack conflict. */
ECC_MARKER_ADDR    = LIM_IDLE_BASE + 0x800;                  /* 0x08000800 */

MEMORY {
  lim_idle (rwx) : ORIGIN = LIM_IDLE_BASE, LENGTH = LIM_IDLE_SIZE
}

SECTIONS {
  .text LIM_IDLE_BASE : {
    KEEP(*(.text.start))
    *(.text .text.*)
    *(.rodata .rodata.*)
    . = ALIGN(8);
  } > lim_idle

  /* GP convention: DATA_START + 0x7f0 (SiFive §5.18) */
  PROVIDE(__global_pointer$ = . + 0x7f0);

  _data_lma = LOADADDR(.data);
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

  /* Trap-diag blocks at the FIXED top of the region. The host reads hart 0's
   * absolute address (X280_IDLE_TRAP_DIAG_ADDR) for crash diagnostics
   * regardless of how big .text/.bss grow; entry.S indexes by mhartid, and
   * _probe_active names hart 0's probe word for the C code that arms it. */
  _trap_block      = TRAP_DIAG_ADDR;
  _trap_block_size = TRAP_DIAG_SIZE;
  _probe_active    = TRAP_DIAG_ADDR + 0x18;

  /* Stack lives just below the trap-diag block. All four harts run C here,
   * so all four stacks must fit inside this 4 KiB region: entry.S strides by
   * _stack_size, putting them in [0x08000B80, 0x08000F80) -- above the
   * ECC-prime marker at LIM+0x800. A 32 KiB stride would place three of them
   * outside the region entirely. */
  . = ALIGN(16);
  _stack_top = STACK_TOP;
  _stack_size = STACK_SIZE;

  /DISCARD/ : {
    *(.eh_frame .eh_frame_hdr)
    *(.note.GNU-stack)
  }
}

/* Sanity: code+data+bss must fit before every hart's stack scratch. */
ASSERT(_bss_end <= (_stack_top - NUM_HARTS * STACK_SIZE),
       "x280-lim-idle: code/data/bss overflow the 4 KiB idle region")

/* Images built with X280_PER_HART_PROTOCOL zero no .bss -- entry.S has no
 * hart-0 zeroing loop for them, because nothing holds the other harts behind it. Their link
 * rules pass --defsym=X280_REQUIRE_NO_BSS=1 to hold the precondition here. */
ASSERT(DEFINED(X280_REQUIRE_NO_BSS) ? (_bss_end == _bss_start) : 1,
       "x280-lim-idle: per-hart images must carry no .bss -- entry.S zeroes none")

/* Sanity: code+data+bss must not grow into the host-only ECC-prime marker slot
 * at LIM+0x800 (see x280_address_map.h X280_ECC_PRIME_MARKER_*). */
ASSERT(_bss_end <= ECC_MARKER_ADDR,
       "x280-lim-idle: .bss overlaps the ECC-prime marker at LIM+0x800")

/* entry.S writes this into mtvec, whose low two bits are the MODE field, so a
 * misaligned handler selects a reserved mode and traps never arrive. */
ASSERT((_trap_handler & 3) == 0,
       "x280-lim-idle: _trap_handler must be 4-byte aligned for mtvec MODE=Direct")
