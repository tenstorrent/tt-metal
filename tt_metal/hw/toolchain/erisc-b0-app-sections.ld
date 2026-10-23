/* Copyright (C) 2014-2017 Free Software Foundation, Inc.
   Copying and distribution of this script, with or without modification,
   are permitted in any medium without royalty provided the copyright
   notice and this notice are preserved.  */

/*
  This file contains pre-processor directives that are compiled to create tensix-sections.ld,
  any modifications should be made to tensix-sections-pre-proc.ld, tensix-sections.ld is a generated file
*/

OUTPUT_FORMAT("elf32-littleriscv", "elf32-littleriscv",
	      "elf32-littleriscv")
OUTPUT_ARCH(riscv)
ENTRY(_start)

PHDRS {
  attributes 0x70000003;
  text PT_LOAD;
  data PT_LOAD;
}

SECTIONS
{
  PROVIDE (__global_pointer$ = __firmware_global_pointer);
  PROVIDE (__erisc_jump_table = ORIGIN(ERISC_JUMPTABLE));

  .text :
  {
    *(.start)
    *(.text.unlikely .text.*_unlikely .text.unlikely.*)
    *(.text.exit .text.exit.*)
    *(.text.startup .text.startup.*)
    *(.text.hot .text.hot.*)
    *(.text .stub .text.* .gnu.linkonce.t.*)
    /* .gnu.warning sections are handled specially by elf32.em.  */
    *(.gnu.warning)
  } > REGION_APP_CODE :text
  .empty.init.fini :
  {
    /* Out elf loader checks that .empty* sections are empty, and
       can give a helpful error, rather than an ASSERT here.  */
    KEEP (*(.init .fini))
  } > REGION_APP_CODE :text

  .data : ALIGN(4)
  {
    *(.rodata .rodata.* .gnu.linkonce.r.*)
    *(.rodata1)

    *(.dynamic)
    *(.data.rel.ro.local* .gnu.linkonce.d.rel.ro.local.*) *(.data.rel.ro .data.rel.ro.* .gnu.linkonce.d.rel.ro.*)

    *(.srodata.cst16) *(.srodata.cst8) *(.srodata.cst4) *(.srodata.cst2) *(.srodata .srodata.*)
    *(.sdata .sdata.* .gnu.linkonce.s.*)
    *(.sdata2 .sdata2.* .gnu.linkonce.s2.*)

    *(.data .data.* .gnu.linkonce.d.*)
    *(.data1)

    *(.got.plt) *(.igot.plt) *(.got) *(.igot)
    . = ALIGN(4);
  } > REGION_APP_DATA :data
  .empty.ctors.dtors :
  {
    /* We don't use .ctors/.dtors either (this still isn't the '90s), make sure there aren't any.  */
    KEEP (*(.ctors .ctors.* .dtors .dtors.*))
  } > REGION_APP_DATA :data
  .empty.init_array.fini_array :
  {
    /* We don't support global static constructors or destructors. make sure there aren't any.  */
    KEEP (*(.preinit_array))
    KEEP (*(SORT_BY_INIT_PRIORITY(.init_array.*)))
    KEEP (*(.init_array))
    KEEP (*(SORT_BY_INIT_PRIORITY(.fini_array.*)))
    KEEP (*(.fini_array))
  } > REGION_APP_DATA :data
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
  } > REGION_APP_DATA :data

  .riscv.attributes 0 : { *(.riscv.attributes) } :attributes

  /* Stabs debugging sections.  */
  .stab          0 : { *(.stab) }
  .stabstr       0 : { *(.stabstr) }
  .stab.excl     0 : { *(.stab.excl) }
  .stab.exclstr  0 : { *(.stab.exclstr) }
  .stab.index    0 : { *(.stab.index) }
  .stab.indexstr 0 : { *(.stab.indexstr) }
  .comment       0 : { *(.comment) }
  /* DWARF debug sections.
     Symbols in the DWARF debugging sections are relative to the beginning
     of the section so we begin them at 0.  */
  /* DWARF 1 */
  .debug          0 : { *(.debug) }
  .line           0 : { *(.line) }
  /* GNU DWARF 1 extensions */
  .debug_srcinfo  0 : { *(.debug_srcinfo) }
  .debug_sfnames  0 : { *(.debug_sfnames) }
  /* DWARF 1.1 and DWARF 2 */
  .debug_aranges  0 : { *(.debug_aranges) }
  .debug_pubnames 0 : { *(.debug_pubnames) }
  /* DWARF 2 */
  .debug_info     0 : { *(.debug_info .gnu.linkonce.wi.*) }
  .debug_abbrev   0 : { *(.debug_abbrev) }
  .debug_line     0 : { *(.debug_line .debug_line.* .debug_line_end ) }
  .debug_frame    0 : { *(.debug_frame) }
  .debug_str      0 : { *(.debug_str) }
  .debug_loc      0 : { *(.debug_loc) }
  .debug_macinfo  0 : { *(.debug_macinfo) }
  /* SGI/MIPS DWARF 2 extensions */
  .debug_weaknames 0 : { *(.debug_weaknames) }
  .debug_funcnames 0 : { *(.debug_funcnames) }
  .debug_typenames 0 : { *(.debug_typenames) }
  .debug_varnames  0 : { *(.debug_varnames) }
  /* DWARF 3 */
  .debug_pubtypes 0 : { *(.debug_pubtypes) }
  .debug_ranges   0 : { *(.debug_ranges) }
  /* DWARF Extension.  */
  .debug_macro    0 : { *(.debug_macro) }
  .debug_addr     0 : { *(.debug_addr) }
  .gnu.attributes 0 : { KEEP (*(.gnu.attributes)) }
  /DISCARD/ : { *(.note.GNU-stack) *(.gnu_debuglink) *(.gnu.lto_*) }

  /* These are all things that used to have an output section, but were deleted for Tensix. */
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
