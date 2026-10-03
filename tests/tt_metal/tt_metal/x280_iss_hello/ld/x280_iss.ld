/*
 * SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Active-FW LIM window matching tt-llm-engine x280/ld/x280.ld.
 * Image is linked at 0x08001000 so the ISS can execute at the real X280
 * load address (idle FW keeps 0x08000000..0x08000FFF).
 */
OUTPUT_ARCH(riscv)
ENTRY(_start)

LIM_BASE  = 0x08001000;
LIM_SIZE  = 0x0011F000;
STACK_SIZE = 0x4000;
NUM_HARTS = 4;

MEMORY {
    lim (rwx) : ORIGIN = LIM_BASE, LENGTH = LIM_SIZE
}

SECTIONS {
    .text LIM_BASE : {
        KEEP(*(.text.start))
        *(.text .text.*)
        *(.rodata .rodata.*)
        . = ALIGN(8);
    } > lim

    PROVIDE(__global_pointer$ = . + 0x7f0);

    .data : {
        *(.data .data.*)
        *(.sdata .sdata.*)
        . = ALIGN(8);
    } > lim

    .tohost : {
        . = ALIGN(64);
        KEEP(*(.tohost))
        . = ALIGN(64);
    } > lim

    .bss (NOLOAD) : {
        *(.bss .bss.*)
        *(.sbss .sbss.*)
        *(COMMON)
        . = ALIGN(16);
        _stack_bottom = .;
        . += STACK_SIZE * NUM_HARTS;
        _stack_top = .;
    } > lim

    /DISCARD/ : { *(.comment) *(.note*) *(.eh_frame*) }
}
