/* Copyright (C) 2014-2017 Free Software Foundation, Inc.
   Copying and distribution of this script, with or without modification,
   are permitted in any medium without royalty provided the copyright
   notice and this notice are preserved.  */

/*
  This file contains pre-processor directives that are compiled to create tensix-sections.ld,
  any modifications should be made to tensix-sections-pre-proc.ld, tensix-sections.ld is a generated file
*/

/*
   Firmware text/ldm start at the start of the section
   Kernel text/ldm start where firmware left off
*/
__firmware_start = DEFINED(__fw_export_end_text) ? __fw_export_end_text : ORIGIN(REGION_CODE);
__ldm_start = DEFINED(__fw_export_ldm_end) ? __fw_export_ldm_end : ORIGIN(REGION_DATA);

/*
  Need a 32B separation between FW end and Kernel start on eriscs to
    ensure kernel does not get cached into i$ because we cannot flush the i$.
  FW must align to 32 byte boundary so Kernel begins aligned to meet noc
    alignment constraints
*/
#if LD_TARGET == IERISC
__padding = 31;
__alignment = 32;
#endif
__padding = 0;
__alignment = 16;

OUTPUT_FORMAT("elf32-littleriscv", "elf32-littleriscv",
	      "elf32-littleriscv")
OUTPUT_ARCH(riscv)
ENTRY(_start)
SEARCH_DIR("/opt/riscv32i/riscv32-unknown-elf/lib");
SECTIONS
{
  /* Read-only sections, merged into text segment: */
  PROVIDE (__executable_start = __firmware_start);
  .init __firmware_start :
  {
    KEEP (*(SORT_NONE(.init)))
  } > REGION_CODE
  .text           :
  {
    *(.text.unlikely .text.*_unlikely .text.unlikely.*)
    *(.text.exit .text.exit.*)
    *(.text.startup .text.startup.*)
    *(.text.hot .text.hot.*)
    *(.text .stub .text.* .gnu.linkonce.t.*)
    /* .gnu.warning sections are handled specially by elf32.em.  */
    *(.gnu.warning)
  } > REGION_CODE
  .fini           :
  {
    KEEP (*(SORT_NONE(.fini)))
  } > REGION_CODE

  l1_data :
  {
    *(l1_data)
  } > REGION_CODE
  l1_data_noinit (NOLOAD) :
  {
    *(l1_data_noinit)
  } > REGION_CODE

  . += __padding;
  . = ALIGN(__alignment);

  PROVIDE (__etext = .);
  PROVIDE (_etext = .);
  PROVIDE (etext = .);
  __fw_export_end_text = DEFINED(__fw_export_end_text) ? __fw_export_end_text : .;

  PROVIDE(__global_pointer$ = ORIGIN(REGION_DATA) + 0x7fc);
  .ldm_data __ldm_start :
  {
    /* ldm is copied in 4 byte chunks */
    . = ALIGN(4);
     __ldm_data_start = .;
     *(.rodata .rodata.* .gnu.linkonce.r.*)
     *(.rodata1)
    PROVIDE_HIDDEN (__preinit_array_start = .);
    KEEP (*(.preinit_array))
    PROVIDE_HIDDEN (__preinit_array_end = .);
    PROVIDE_HIDDEN (__init_array_start = .);
    KEEP (*(SORT_BY_INIT_PRIORITY(.init_array.*) SORT_BY_INIT_PRIORITY(.ctors.*)))
    KEEP (*(.init_array EXCLUDE_FILE (*crtbegin.o *crtbegin?.o *crtend.o *crtend?.o ) .ctors))
    PROVIDE_HIDDEN (__init_array_end = .);
    PROVIDE_HIDDEN (__fini_array_start = .);
    KEEP (*(SORT_BY_INIT_PRIORITY(.fini_array.*) SORT_BY_INIT_PRIORITY(.dtors.*)))
    KEEP (*(.fini_array EXCLUDE_FILE (*crtbegin.o *crtbegin?.o *crtend.o *crtend?.o ) .dtors))
    PROVIDE_HIDDEN (__fini_array_end = .);
    KEEP (*crtbegin.o(.ctors))
    KEEP (*crtbegin?.o(.ctors))
    KEEP (*(EXCLUDE_FILE (*crtend.o *crtend?.o ) .ctors))
    KEEP (*(SORT(.ctors.*)))
    KEEP (*(.ctors))
    KEEP (*crtbegin.o(.dtors))
    KEEP (*crtbegin?.o(.dtors))
    KEEP (*(EXCLUDE_FILE (*crtend.o *crtend?.o ) .dtors))
    KEEP (*(SORT(.dtors.*)))
    KEEP (*(.dtors))
    *(.dynamic)
    *(.data.rel.ro.local* .gnu.linkonce.d.rel.ro.local.*) *(.data.rel.ro .data.rel.ro.* .gnu.linkonce.d.rel.ro.*)
    *(.srodata.cst16) *(.srodata.cst8) *(.srodata.cst4) *(.srodata.cst2) *(.srodata .srodata.*)
    *(.sdata .sdata.* .gnu.linkonce.s.*)
    *(.sdata2 .sdata2.* .gnu.linkonce.s2.*)
    *(.data .data.* .gnu.linkonce.d.*)
    SORT(CONSTRUCTORS)
    *(.data1)
    *(.got.plt) *(.igot.plt) *(.got) *(.igot)

    /* bss is initialized with 4 byte writes */
    . = ALIGN(4);
    __ldm_data_end = .;
    __ldm_bss_start = .;
    *(.sbss2 .sbss2.* .gnu.linkonce.sb2.*)
    *(.dynsbss)
    *(.sbss .sbss.* .gnu.linkonce.sb.*)
    *(.scommon)
    . = ALIGN(4);
    *(.dynbss)
    *(.bss .bss.* .gnu.linkonce.b.*)
    *(COMMON)
    . = ALIGN(4);
    __ldm_bss_end = .;
  } > REGION_DATA

  /* Use __ldm_bss_end to pick up where the last REGION_DATA ended
     This is needed because of the way kernel REGION_DATA picks up where firmware
     ended.  If the section above is empty (highly unlikely) we don't want to
     re-start at the origion of REGION_DATA
  */
  local_data_noinit __ldm_bss_end (NOLOAD) :
  {
    *(local_data_noinit)
  } > REGION_DATA

  /* FW must align to 16 byte boundary so Kernel begins aligned to meet noc
     alignment constraints */
  . = ALIGN(16);
  _edata = .; PROVIDE (edata = .);
  _end = .; PROVIDE (end = .);
  __fw_export_ldm_end = DEFINED(__fw_export_ldm_end) ? __fw_export_ldm_end : _end;

  /* Move to end of local memory for stack */
  .stack :
  {
    __stack_bottom = .;
   . += __firmware_stack_size;
    __stack_top = .;
  } > REGION_STACK

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
