/*
 * SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
 *
 * SPDX-License-Identifier: Apache-2.0
 */

__firmware_stack_size = LENGTH(REGION_STACK);
__firmware_start = DEFINED(__fw_export_end_text) ? __fw_export_end_text : ORIGIN(REGION_CODE);
__ldm_start = DEFINED(__fw_export_ldm_end) ? __fw_export_ldm_end : ORIGIN(REGION_DATA);

OUTPUT_FORMAT("elf32-littleriscv", "elf32-littleriscv",
       "elf32-littleriscv")
OUTPUT_ARCH(riscv)
ENTRY(_start)
SECTIONS
{
  /*
   * This section is used to transfer .ldm_data section into private memory (0xffb00000).
   * Since loader cannot access private core memory from NOC, it copies .ldm_data section into .loader_init section,
   * and we embed code into CRT library that copies .loader_init section into .ldm_data section before main starts.
   */
  .loader_init :
  {
    __loader_init_start = .;
   . += LENGTH(REGION_LOADER_INIT);
    __loader_init_end = .;
  } > REGION_LOADER_INIT
  /*
   * This section is used to transfer .init section into private code memory (IRAM = 0xffc00000).
   * IRAM exists only on NCRICS on some architectures. On all others, this section is created as empty.
   * Since loader cannot access IRAM from NOC, it copies .init section into .loader_code section,
   * and we embed code into CRT library that copies .loader_code section into .init section before rest of the CRT starts.
   */
  .loader_code :
  {
    __loader_code_start = .;
   . += LENGTH(REGION_LOADER_CODE);
    __loader_code_end = .;
  } > REGION_LOADER_CODE

  /*
   * Rest of the sections are standard sections
   */

  PROVIDE (__executable_start = __firmware_start);
  .init __firmware_start :
  {
    KEEP (*(SORT_NONE(.init)))
  } > REGION_CODE
  .text :
  {
    *(.text.unlikely .text.*_unlikely .text.unlikely.*)
    *(.text.exit .text.exit.*)
    *(.text.startup .text.startup.*)
    *(.text.hot .text.hot.*)
    *(.text .stub .text.* .gnu.linkonce.t.*)
    *(.gnu.warning)
  } > REGION_CODE
  .fini :
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
  . = ALIGN(16);
  PROVIDE (__etext = .);
  PROVIDE (_etext = .);
  PROVIDE (etext = .);
  __fw_export_end_text = DEFINED(__fw_export_end_text) ? __fw_export_end_text : .;
  PROVIDE(__global_pointer$ = ORIGIN(REGION_DATA) + 0x800);
  .ldm_data __ldm_start :
  {
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
  local_data_noinit __ldm_bss_end (NOLOAD) :
  {
    *(local_data_noinit)
  } > REGION_DATA
  . = ALIGN(16);
  _edata = .; PROVIDE (edata = .);
  _end = .; PROVIDE (end = .);
  __fw_export_ldm_end = DEFINED(__fw_export_ldm_end) ? __fw_export_ldm_end : _end;
  .stack :
  {
    __stack_bottom = .;
   . += __firmware_stack_size;
    __stack_top = .;
  } > REGION_STACK
  .profiler_meta 0 : { *(.profiler_meta) }
  .stab 0 : { *(.stab) }
  .stabstr 0 : { *(.stabstr) }
  .stab.excl 0 : { *(.stab.excl) }
  .stab.exclstr 0 : { *(.stab.exclstr) }
  .stab.index 0 : { *(.stab.index) }
  .stab.indexstr 0 : { *(.stab.indexstr) }
  .comment 0 : { *(.comment) }
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
_Z11kernel_initv = _etext;
