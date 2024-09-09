/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

INCLUDE "erisc-b0-memory.ld"

REGION_ALIAS("REGION_APP_IRAM_CODE", ERISC_APP_IRAM_CODE)
REGION_ALIAS("REGION_APP_KERNEL_CODE", ERISC_APP_KERNEL_CODE)
REGION_ALIAS("REGION_APP_KERNEL_DATA", ERISC_APP_KERNEL_DATA)
REGION_ALIAS("REGION_LDM", LOCAL_DATA_MEM)

__firmware_stack_size = 1024;
__firmware_global_pointer = ORIGIN(ERISC_DATA) + 0x7fc;

__firmware_start = ORIGIN(REGION_APP_KERNEL_CODE);

OUTPUT_FORMAT("elf32-littleriscv", "elf32-littleriscv",
	      "elf32-littleriscv")
OUTPUT_ARCH(riscv)
ENTRY(_start)
SEARCH_DIR("/opt/riscv32i/riscv32-unknown-elf/lib");
SECTIONS
{
  /* Read-only sections, merged into text segment: */
  PROVIDE (__executable_start = __firmware_start);
  PROVIDE (__global_pointer$ = __firmware_global_pointer);
  PROVIDE (__erisc_jump_table = ORIGIN(ERISC_JUMPTABLE));

  .init __firmware_start :
  {
    KEEP (*(SORT_NONE(.init)))
  } > REGION_APP_KERNEL_CODE
  code_l1 :
  {
    *(code_l1)
  } > REGION_APP_KERNEL_CODE
  PROVIDE (__ecode_l1 = .);
  .text :

  {
    *(.text.unlikely .text.*_unlikely .text.unlikely.*)
    *(.text.exit .text.exit.*)
    *(.text.startup .text.startup.*)
    *(.text.hot .text.hot.*)
    *(.text .stub .text.* .gnu.linkonce.t.*)
    *(.gnu.warning)
  } > REGION_APP_KERNEL_CODE


  PROVIDE (__etext = .);
  _etext = .; PROVIDE (_etext = .);

  PROVIDE (etext = .);
  .rodata         : { *(.rodata .rodata.* .gnu.linkonce.r.*) } > REGION_APP_KERNEL_CODE
  .rodata1        : { *(.rodata1) } > REGION_APP_KERNEL_CODE
  .sdata2         :
  {
    *(.sdata2 .sdata2.* .gnu.linkonce.s2.*)
  } > REGION_APP_KERNEL_DATA
  .sbss2          : { *(.sbss2 .sbss2.* .gnu.linkonce.sb2.*) } > REGION_APP_KERNEL_DATA
  /* Adjust the address for the data segment.  We want to adjust up to
     the same address within the page on the next page up.  */
  . = DATA_SEGMENT_ALIGN (CONSTANT (MAXPAGESIZE), CONSTANT (COMMONPAGESIZE));
  .data.rel.ro : { *(.data.rel.ro.local* .gnu.linkonce.d.rel.ro.local.*) *(.data.rel.ro .data.rel.ro.* .gnu.linkonce.d.rel.ro.*) } > REGION_APP_KERNEL_DATA
  .dynamic        : { *(.dynamic) } > REGION_APP_KERNEL_DATA
  . = DATA_SEGMENT_RELRO_END (0, .);
  .data           :
  {
    . = ALIGN(16);
    *(.data .data.* .gnu.linkonce.d.*)
    *(.srodata.cst16) *(.srodata.cst8) *(.srodata.cst4) *(.srodata.cst2) *(.srodata .srodata.*)
    SORT(CONSTRUCTORS)
  } > REGION_APP_KERNEL_DATA

  . = ALIGN(4); /* startup code will use word writes to zero bss */
  _edata = .; PROVIDE (edata = .);
  . = .;
  __l1_bss_start = .;
  .sbss           :
  {
    *(.dynsbss)
    *(.sbss .sbss.* .gnu.linkonce.sb.*)
    *(.scommon)
  } > REGION_APP_KERNEL_DATA
  . = ALIGN(4);
  __l1_bss_end = .;

  .bss            :
  {
   __ldm_bss_start = .;
   *(.dynbss)
   *(.bss .bss.* .gnu.linkonce.b.*)
   *(COMMON)

   /* Align here to ensure that the .bss section occupies space up to
      _end.  Align after .bss to ensure correct alignment even if the
      .bss section disappears because there are no input sections.
      FIXME: Why do we need it? When there is no .bss section, we don't
      pad the .data section.  */
   . = ALIGN(4);
   __ldm_bss_end = .;
  } > REGION_APP_KERNEL_DATA
  . = ALIGN(32 / 8);
  . = SEGMENT_START("ldata-segment", .);
  . = ALIGN(32 / 8);
  _end = .; PROVIDE (end = .);
  . = DATA_SEGMENT_END (.);


  .stack :
  {
   . = ALIGN(16);
   __stack_bottom = .;
   . += __firmware_stack_size;
   __stack_top = .;
   __freertos_irq_stack_top = .;
   . += 4;
  } > REGION_LDM

  data_noinit (NOLOAD):
  {
    *(data_noinit)
  } > REGION_APP_KERNEL_DATA

  l1_memory :
  {
    *(l1_memory)
    *(l1_memory_const) /* gcc complains about const and non-const data in the same segment. */
  } > ETH_L1

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
INCLUDE "tensix-address.ld"
