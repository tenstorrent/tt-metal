/* Reference-faithful layout: x280 firmware runs from CACHED GDDR, exactly like
 * tt-bh-linux (opensbi at 0x4000_3000_0000). The mailbox lives separately in the
 * UNCACHED GDDR alias (0x3010_0000) so the firmware's stores are instantly
 * NOC-visible without cache flushes. Code at GDDR offset 0; mailbox at +1 MiB. */
OUTPUT_ARCH(riscv)
ENTRY(_start)

MEMORY {
    DRAM : ORIGIN = 0x0000400030000000, LENGTH = 0x80000
}

SECTIONS {
    .text : { *(.text.start) *(.text*) } > DRAM
    .rodata : { *(.rodata*) *(.srodata*) } > DRAM
    .data : { *(.data*) *(.sdata*) } > DRAM
    .bss : { *(.bss*) *(.sbss*) *(COMMON) } > DRAM
    . = ALIGN(16);
    . = . + 0x4000;
    _stack_top = .;
}
