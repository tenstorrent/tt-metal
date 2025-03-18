/* Copyright (C) 2014-2017 Free Software Foundation, Inc.
   Copying and distribution of this script, with or without modification,
   are permitted in any medium without royalty provided the copyright
   notice and this notice are preserved.  */

/*
  This file contains pre-processor directives that are compiled to create tensix-sections.ld,
  any modifications should be made to tensix-sections-pre-proc.ld, tensix-sections.ld is a generated file
*/

#if defined(TYPE_FIRMWARE) || (defined(NCRISC_FIRMWARE_KERNEL_SPLIT) && defined(COMPILE_FOR_NCRISC))
/* Start of the appropriate region. */
#define TEXT_START ORIGIN(REGION_CODE)
#elif defined(TYPE_KERNEL)
/* Where firmware ended. */
#define TEXT_START __fw_export_end_text
#else
#error "Neither TYPE_FIRMARE nor TYPE_KERNEL defined, what am I to be?"
#endif

#if defined(TYPE_FIRMWARE)
/* Start of the appropriate region. */
#define DATA_START ORIGIN(REGION_DATA)
#elif defined(TYPE_KERNEL)
/* Where firmware ended. */
#define DATA_START __fw_export_ldm_end
#else
#error "Neither TYPE_FIRMARE nor TYPE_KERNEL defined, what am I to be?"
#endif

#if defined(TYPE_FIRMWARE)
/* Need separation beteen end of firmware and start of kernel, so that
   they don't share unflushable i$ cache lines.  */
#if defined(TARGET_IERISC) || defined(TARGET_SLAVE_IERISC) || defined(TARGET_AERISC)
#define MEM_PAD MEM_IERISC_KERNEL_PAD
#else
#define MEM_PAD 0
#endif
/* FW must align to 16 byte boundary so kernel begins aligned to meet noc alignment constraints */
#define MEM_ALIGN 16
#else
#define MEM_PAD 0
#define MEM_ALIGN 4
#endif

OUTPUT_FORMAT("elf32-littleriscv", "elf32-littleriscv",
	      "elf32-littleriscv")
OUTPUT_ARCH(riscv)
ENTRY(_start)

PHDRS {
  attributes 0x70000003;
  text PT_LOAD;
  data PT_LOAD;
#if defined(TYPE_FIRMWARE)
  stack PT_GNU_STACK;
#endif
}

SECTIONS
{
#if defined(TYPE_FIRMWARE)
  .firmware_text TEXT_START : ALIGN(4)
#else
  .text TEXT_START : ALIGN(4)
#endif
  {
    #if defined(TYPE_KERNEL)
    __kernel_text_start = ABSOLUTE(.);
    #endif
    /* Because TEXT_START might not be the start of a region, we
       need to force this section to be emitted so that following sections
       do not restart the region, if this one is empty.  */
    . = .;
    *(.start)
    *(.text.unlikely .text.*_unlikely .text.unlikely.*)
    *(.text.exit .text.exit.*)
    *(.text.startup .text.startup.*)
    *(.text.hot .text.hot.*)
    *(.text .stub .text.* .gnu.linkonce.t.*)
    /* .gnu.warning sections are handled specially by elf32.em.  */
    *(.gnu.warning)
    . = ALIGN(4);
  } > REGION_CODE :text
  .init.fini :
  {
    /* We don't use .init/.fini (this isn't the '90s), make sure there aren't any.  */
    KEEP (*(.init .fini))
    ASSERT(SIZEOF(.init.fini) == 0, ".init/.fini sections have contents");
  } > REGION_CODE :text

#if defined(TYPE_KERNEL)
   __kernel_data_lma = .;
#endif
  . = ALIGN(ABSOLUTE(.) + MEM_PAD, MEM_ALIGN);

#if defined(TYPE_FIRMWARE)
  __fw_export_end_text = ABSOLUTE(.);
#if defined(TARGET_NCRISC)
#if !defined(NCRISC_FIRMWARE_KERNEL_SPLIT)
  PROVIDE (KERNEL_ENTRY_SYMBOL = ABSOLUTE(__fw_export_end_text));
#else
  PROVIDE (KERNEL_ENTRY_SYMBOL = MEM_NCRISC_KERNEL_BASE);
#endif
#endif
#endif

#if defined(TYPE_KERNEL)
  __kernel_init_local_l1_base = ABSOLUTE(.);
#endif

#if defined(TYPE_FIRMWARE)
  PROVIDE(__global_pointer$ = ORIGIN(REGION_DATA) + 0x7f0);
#endif
  .data DATA_START :
#if defined (TYPE_KERNEL)
        AT(__kernel_data_lma)
#endif
	ALIGN(4)
  {
     . = .; /* Force section emission.  */
     __ldm_data_start = .;
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
    __ldm_data_end = .;
  } > REGION_DATA :data
  .ctors.dtors :
  {
    /* We don't use .ctors/.dtors either (this still isn't the '90s), make sure there aren't any.  */
    KEEP (*(.ctors .ctors.* .dtors .dtors.*))
    ASSERT(SIZEOF(.ctors.dtors) == 0, ".ctors/.dtors sections have contents");
  } > REGION_DATA :data
  .init_array.fini_array :
  {
    /* We don't support global static constructors or destructors. make sure there aren't any.  */
    KEEP (*(.preinit_array))
    KEEP (*(SORT_BY_INIT_PRIORITY(.init_array.*)))
    KEEP (*(.init_array))
    KEEP (*(SORT_BY_INIT_PRIORITY(.fini_array.*)))
    KEEP (*(.fini_array))
    ASSERT(SIZEOF(.init_array.fini_array) == 0, ".{preinit,init,fini}_array sections have contents");
  } > REGION_DATA :data
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
  } > REGION_DATA :data

#ifdef TYPE_FIRMWARE
  . = ALIGN(ABSOLUTE(.), MEM_ALIGN);
  __fw_export_ldm_end = ABSOLUTE(.);
#endif

#ifdef TYPE_FIRMWARE
  .stack :
  {
   . += FIRMWARE_STACK_SIZE;
    __stack_top = .;
  } > REGION_STACK :stack
#endif

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
