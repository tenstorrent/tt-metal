/*
 * SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
 * SPDX-License-Identifier: Apache-2.0
 *
 * Lite Fabric Linker Script for Blackhole RISCV Cores
 *
 * This linker script defines the memory layout for the lite fabric firmware
 * that runs on ERISC1.
 */

OUTPUT_FORMAT("elf32-littleriscv", "elf32-littleriscv", "elf32-littleriscv")
OUTPUT_ARCH(riscv)
ENTRY(_start)

/* Not to be confused with dev_mem_map.h which is for tt-metal kernels and firmware */
#include "lf_dev_mem_map.hpp"

PHDRS {
    attributes 0x70000003;
    text PT_LOAD;
    data PT_LOAD;
}

SECTIONS
{
    /* Text section starts at firmware base address */
    .text LITE_FABRIC_TEXT_START :
    {
        *(.start)
        *(.text.unlikely .text.*_unlikely .text.unlikely.*)
        *(.text.exit .text.exit.*)
        *(.text.startup .text.startup.*)
        *(.text.hot .text.hot.*)
        *(.text .stub .text.* .gnu.linkonce.t.*)
        *(.gnu.warning)
        . = ALIGN(4);
    } :text

    .empty.init.fini :
    {
        /* Check that these sections are empty */
        KEEP (*(.init .fini))
    } :text

    /* Align text end to 16-byte boundary */
    . = ALIGN(ABSOLUTE(.) + 0, 16);
    __fw_export_text_end = ABSOLUTE(.);

    /* Set up global pointer for efficient access to data */
    PROVIDE(__global_pointer$ = LITE_FABRIC_DATA_START + 0x7f0);

    /* Data section starts at local memory base */
    .data LITE_FABRIC_DATA_START :
    {
        __ldm_data_start = .;

        /* Read-only data */
        *(.rodata .rodata.* .gnu.linkonce.r.*)
        *(.rodata1)

        /* Dynamic linking data */
        *(.dynamic)
        *(.data.rel.ro.local* .gnu.linkonce.d.rel.ro.local.*)
        *(.data.rel.ro .data.rel.ro.* .gnu.linkonce.d.rel.ro.*)

        /* Small read-only data */
        *(.srodata.cst16) *(.srodata.cst8) *(.srodata.cst4) *(.srodata.cst2) *(.srodata .srodata.*)

        /* Small data */
        *(.sdata .sdata.* .gnu.linkonce.s.*)
        *(.sdata2 .sdata2.* .gnu.linkonce.s2.*)

        /* Initialized data */
        *(.data .data.* .gnu.linkonce.d.*)
        *(.data1)

        /* Global offset table */
        *(.got.plt) *(.igot.plt) *(.got) *(.igot)

        . = ALIGN(4);
        __ldm_data_end = .;
    } :data

    .empty.ctors.dtors :
    {
        /* We don't use constructors/destructors - ensure they're empty */
        KEEP (*(.ctors .ctors.* .dtors .dtors.*))
    } :data

    .empty.init_array.fini_array :
    {
        /* No global static constructors or destructors */
        KEEP (*(.preinit_array))
        KEEP (*(SORT_BY_INIT_PRIORITY(.init_array.*)))
        KEEP (*(.init_array))
        KEEP (*(SORT_BY_INIT_PRIORITY(.fini_array.*)))
        KEEP (*(.fini_array))
    } :data

    /* Uninitialized data (BSS) */
    .bss : ALIGN(4)
    {
        __ldm_bss_start = .;
        *(.sbss2 .sbss2.* .gnu.linkonce.sb2.*)
        *(.dynsbss)
        *(.sbss .sbss.* .gnu.linkonce.sb.*)
        *(.scommon)
        *(.dynbss)
        *(.bss .bss.* .gnu.linkonce.b.*)
        *(COMMON)
        . = ALIGN(4);
        __ldm_bss_end = .;
    } :data

    /* Align data end to 16-byte boundary */
    . = ALIGN(ABSOLUTE(.), 16);
    __fw_export_ldm_end = ABSOLUTE(.);

    /* Stack grows downward */
    __stack_top = LITE_FABRIC_STACK_START + LITE_FABRIC_STACK_SIZE;

    /* Loader information for size checks */
    .phdrs 0 (INFO) :
    {
        LONG(LITE_FABRIC_TEXT_SIZE)
        LONG(LITE_FABRIC_DATA_SIZE)
    }

    /* RISC-V attributes */
    .riscv.attributes 0 : { *(.riscv.attributes) } :attributes

    /* Debug sections */
    .stab 0 : { *(.stab) }
    .stabstr 0 : { *(.stabstr) }
    .stab.excl 0 : { *(.stab.excl) }
    .stab.exclstr 0 : { *(.stab.exclstr) }
    .stab.index 0 : { *(.stab.index) }
    .stab.indexstr 0 : { *(.stab.indexstr) }
    .comment 0 : { *(.comment) }

    /* DWARF debug sections */
    .debug 0 : { *(.debug) }
    .line 0 : { *(.line) }
    .debug_srcinfo 0 : { *(.debug_srcinfo) }
    .debug_sfnames 0 : { *(.debug_sfnames) }
    .debug_aranges 0 : { *(.debug_aranges) }
    .debug_pubnames 0 : { *(.debug_pubnames) }
    .debug_info 0 : { *(.debug_info .gnu.linkonce.wi.*) }
    .debug_abbrev 0 : { *(.debug_abbrev) }
    .debug_line 0 : { *(.debug_line .debug_line.* .debug_line_end ) }
    .debug_frame 0 : { *(.debug_frame) }
    .debug_str 0 : { *(.debug_str) }
    .debug_loc 0 : { *(.debug_loc) }
    .debug_macinfo 0 : { *(.debug_macinfo) }
    .debug_weaknames 0 : { *(.debug_weaknames) }
    .debug_funcnames 0 : { *(.debug_funcnames) }
    .debug_typenames 0 : { *(.debug_typenames) }
    .debug_varnames 0 : { *(.debug_varnames) }
    .debug_pubtypes 0 : { *(.debug_pubtypes) }
    .debug_ranges 0 : { *(.debug_ranges) }
    .debug_macro 0 : { *(.debug_macro) }
    .debug_addr 0 : { *(.debug_addr) }
    .gnu.attributes 0 : { KEEP (*(.gnu.attributes)) }

    /* Discard unwanted sections */
    /DISCARD/ : { *(.note.GNU-stack) *(.gnu_debuglink) *(.gnu.lto_*) }
    /DISCARD/ : {
        *(.interp)
        *(.note.gnu.build-id)
        *(.hash)
        *(.gnu.hash)
        *(.dynsym)
        *(.dynstr)
        *(.gnu.version)
        *(.gnu.version_d)
        *(.gnu.version_r)
        *(.rela.init)
        *(.rela.text .rela.text.* .rela.gnu.linkonce.t.*)
        *(.rela.fini)
        *(.rela.rodata .rela.rodata.* .rela.gnu.linkonce.r.*)
        *(.rela.data.rel.ro .rela.data.rel.ro.* .rela.gnu.linkonce.d.rel.ro.*)
        *(.rela.data .rela.data.* .rela.gnu.linkonce.d.*)
        *(.rela.tdata .rela.tdata.* .rela.gnu.linkonce.td.*)
        *(.rela.tbss .rela.tbss.* .rela.gnu.linkonce.tb.*)
        *(.rela.ctors)
        *(.rela.dtors)
        *(.rela.got)
        *(.rela.sdata .rela.sdata.* .rela.gnu.linkonce.s.*)
        *(.rela.sbss .rela.sbss.* .rela.gnu.linkonce.sb.*)
        *(.rela.sdata2 .rela.sdata2.* .rela.gnu.linkonce.s2.*)
        *(.rela.sbss2 .rela.sbss2.* .rela.gnu.linkonce.sb2.*)
        *(.rela.bss .rela.bss.* .rela.gnu.linkonce.b.*)
        *(.rela.iplt)
        *(.rela.plt)
        *(.plt)
        *(.iplt)
        *(.eh_frame_hdr) *(.eh_frame_entry .eh_frame_entry.*)
        *(.eh_frame) *(.eh_frame.*)
        *(.gcc_except_table .gcc_except_table.*)
        *(.gnu_extab*)
        *(.exception_ranges .exception_ranges*)
        *(.eh_frame) *(.eh_frame.*)
        *(.gnu_extab)
        *(.gcc_except_table .gcc_except_table.*)
        *(.exception_ranges .exception_ranges*)
        *(.tdata .tdata.* .gnu.linkonce.td.*)
        *(.tbss .tbss.* .gnu.linkonce.tb.*) *(.tcommon)
        *(.jcr)
        *(.got.plt) *(.igot.plt) *(.got) *(.igot)
    }
}
