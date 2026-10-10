/* SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
/* SPDX-FileCopyrightText: © 2026 Kansei Motoe (kinginu)
 *
 * SPDX-License-Identifier: Apache-2.0
 */
OUTPUT_ARCH(riscv)
ENTRY(_start)
SECTIONS
{
  . = 0;
  .text : { KEEP(*(.text.entry)) *(.text .text.*) }
  .rodata : { *(.rodata .rodata.* .srodata .srodata.*) }
  .data : { *(.data .data.* .sdata .sdata.*) }
  .bss : { *(.bss .bss.* .sbss .sbss.* COMMON) }
  . = ALIGN(16);
  . = . + 0x2000;
  __stack_top = .;
  ASSERT(. <= 0x10000, "responder image too large")
  /DISCARD/ : { *(.comment) *(.riscv.attributes) *(.note*) *(.eh_frame*) }
}
