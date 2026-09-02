/* Bare-metal layout for x280 echo firmware: everything in the first 512 KiB of
 * the L2CPU tile's LIM. The mailbox lives at LIM + 1 MiB, well clear of this. */
OUTPUT_ARCH(riscv)
ENTRY(_start)

MEMORY {
    LIM : ORIGIN = 0x08000000, LENGTH = 0x80000
}

SECTIONS {
    .text : { *(.text.start) *(.text*) } > LIM
    .rodata : { *(.rodata*) *(.srodata*) } > LIM
    .data : { *(.data*) *(.sdata*) } > LIM
    .bss : { *(.bss*) *(.sbss*) *(COMMON) } > LIM
    . = ALIGN(16);
    . = . + 0x4000;
    _stack_top = .;
}
