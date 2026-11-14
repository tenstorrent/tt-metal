/* SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
/* SPDX-FileCopyrightText: © 2026 Kansei Motoe (kinginu)
 *
 * SPDX-License-Identifier: Apache-2.0
 */
OUTPUT_ARCH(riscv)
ENTRY(_start)
MEMORY { RAM (rwx) : ORIGIN = 0x80000000, LENGTH = 240M }
SECTIONS {
  .text : { *(.text.init) *(.text*) } > RAM
  .rodata : { *(.rodata*) *(.srodata*) } > RAM
  .data : { *(.data*) *(.sdata*) } > RAM
  . = ALIGN(8);
  _bss_start = .;
  .bss : { *(.bss*) *(.sbss*) *(COMMON) } > RAM
  . = ALIGN(8);
  _bss_end = .;
  . = ALIGN(16);
  . = . + 0x40000;
  _stack_top = .;
}
