/* SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
/* SPDX-FileCopyrightText: © 2026 Kansei Motoe (kinginu)
 *
 * SPDX-License-Identifier: Apache-2.0
 */
/* link.ld: flat image linked at BASE (make BASE=...). The code is position independent (medany,
 * pc-relative only; `make pic-check` proves it), so the binary also runs at any other load address.
 * First byte of the image = _start = the reset vector. */
OUTPUT_ARCH(riscv)
ENTRY(_start)

SECTIONS
{
    . = BASE;
    _image_start = .;
    .text : {
        KEEP(*(.text.entry))
        *(.text .text.*)
    }
    .rodata : ALIGN(16) {
        *(.rodata .rodata.*)
        *(.srodata .srodata.*)
    }
    .data : ALIGN(64) {
        *(.data .data.*)
    }
    .sdata : ALIGN(16) {
        __global_pointer$ = . + 0x800;
        *(.sdata .sdata.*)
    }
    _load_end = .;
    .bss (NOLOAD) : ALIGN(64) {
        _bss_start = .;
        *(.sbss .sbss.*)
        *(.bss .bss.*)
        *(COMMON)
        . = ALIGN(64);
        _bss_end = .;
    }
    _image_end = .;
    ASSERT(_image_end - _image_start <= 0x100000, "image (incl. bss) exceeds L2CPU_FW_MAX_SIZE")
    /DISCARD/ : { *(.eh_frame*) *(.comment) *(.note*) }
}
