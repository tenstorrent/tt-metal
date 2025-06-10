#if defined(TYPE_FIRMWARE) + defined(TYPE_KERNEL) != 1
#error "Exactly one of TYPE_FIRMWARE and TYPE_KERNEL must be defined"
#endif

#if (defined(COMPILE_FOR_BRISC) + defined(COMPILE_FOR_NCRISC) +         \
     defined(COMPILE_FOR_IERISC) + defined(COMPILE_FOR_SUBORDINATE_IERISC) +  \
     defined(COMPILE_FOR_TRISC) + defined(COMPILE_FOR_AERISC)) != 1
#error "Exactly one COMPILE_FOR_${PROC}RISC must be defined"
#endif

OUTPUT_FORMAT("elf32-littleriscv", "elf32-littleriscv",
	      "elf32-littleriscv")
OUTPUT_ARCH(riscv)

#include "dev_mem_map.h"

#if defined(COMPILE_FOR_BRISC)
#define DATA_START MEM_LOCAL_BASE
#define DATA_SIZE MEM_BRISC_LOCAL_SIZE
#define STACK_MIN_SIZE MEM_BRISC_STACK_MIN_SIZE
#define TEXT_START MEM_BRISC_FIRMWARE_BASE
#if defined(TYPE_FIRMWARE)
#define TEXT_SIZE MEM_BRISC_FIRMWARE_SIZE
#else
#define TEXT_SIZE MEM_BRISC_KERNEL_SIZE
#endif

#elif defined(COMPILE_FOR_NCRISC)
#define DATA_START MEM_LOCAL_BASE
#define DATA_SIZE MEM_NCRISC_LOCAL_SIZE
#define STACK_MIN_SIZE MEM_NCRISC_STACK_MIN_SIZE
#if defined(TYPE_FIRMWARE) || !defined(ARCH_WORMHOLE)
#define TEXT_START MEM_NCRISC_FIRMWARE_BASE
#else
#define TEXT_START MEM_NCRISC_KERNEL_BASE
#endif
#if defined(TYPE_FIRMWARE)
#define TEXT_SIZE MEM_NCRISC_FIRMWARE_SIZE
#else
#define TEXT_SIZE MEM_NCRISC_KERNEL_SIZE
#endif

#elif defined(COMPILE_FOR_IERISC)
#define DATA_START MEM_LOCAL_BASE
#define DATA_SIZE MEM_IERISC_LOCAL_SIZE
#define STACK_MIN_SIZE MEM_IERISC_STACK_MIN_SIZE
#define TEXT_START MEM_IERISC_FIRMWARE_BASE
#if defined(TYPE_FIRMWARE)
#define TEXT_SIZE MEM_IERISC_FIRMWARE_SIZE
#else
#define TEXT_SIZE MEM_IERISC_KERNEL_SIZE
#endif

#elif defined(COMPILE_FOR_SUBORDINATE_IERISC)
#define DATA_START MEM_LOCAL_BASE
#define DATA_SIZE MEM_SUBORDINATE_IERISC_LOCAL_SIZE
#define STACK_MIN_SIZE MEM_SUBORDINATE_IERISC_STACK_MIN_SIZE
#define TEXT_START MEM_SUBORDINATE_IERISC_FIRMWARE_BASE
#define TEXT_SIZE MEM_SUBORDINATE_IERISC_FIRMWARE_SIZE

#elif defined(COMPILE_FOR_TRISC)
#define TRISC_SELECT__(BEFORE,MIDDLE,AFTER) BEFORE##TRISC##MIDDLE##AFTER
#define TRISC_SELECT_(BEFORE,MIDDLE,AFTER) TRISC_SELECT__(BEFORE,MIDDLE,AFTER)
#define TRISC_SELECT(BEFORE,AFTER) TRISC_SELECT_(BEFORE,COMPILE_FOR_TRISC,AFTER)

#define DATA_START MEM_LOCAL_BASE
#define DATA_SIZE MEM_TRISC_LOCAL_SIZE
#define STACK_MIN_SIZE TRISC_SELECT(MEM_,_STACK_MIN_SIZE)
#define TEXT_START TRISC_SELECT(MEM_,_FIRMWARE_BASE)
#if defined(TYPE_FIRMWARE)
#define TEXT_SIZE TRISC_SELECT(MEM_,_FIRMWARE_SIZE)
#else
#define TEXT_SIZE TRISC_SELECT(MEM_,_KERNEL_SIZE)
#endif

#elif defined(COMPILE_FOR_AERISC)
/* Yes, using IERISC sizes.  */
#define DATA_START MEM_LOCAL_BASE
#define DATA_SIZE MEM_IERISC_LOCAL_SIZE
#define STACK_MIN_SIZE MEM_AERISC_STACK_MIN_SIZE
#define TEXT_START MEM_AERISC_FIRMWARE_BASE
#if defined(TYPE_FIRMWARE)
#define TEXT_SIZE MEM_IERISC_FIRMWARE_SIZE
#else
#define TEXT_SIZE MEM_IERISC_KERNEL_SIZE
#endif

#else
#error "compiling for unknown"
#endif

ENTRY(_start)

PHDRS {
  attributes 0x70000003;
  text PT_LOAD;
  data PT_LOAD;
}

SECTIONS
{
#if defined(TYPE_FIRMWARE) || (defined(COMPILE_FOR_NCRISC) && defined(ARCH_WORMHOLE))
  .text TEXT_START :
#else
  .text __fw_export_text_end :
#endif
  {
#if defined(TYPE_KERNEL) && defined(COMPILE_FOR_NCRISC)
    __kernel_text_start = ABSOLUTE(.);
#endif
    *(.start)
    *(.text.unlikely .text.*_unlikely .text.unlikely.*)
    *(.text.exit .text.exit.*)
    *(.text.startup .text.startup.*)
    *(.text.hot .text.hot.*)
    *(.text .stub .text.* .gnu.linkonce.t.*)
    /* .gnu.warning sections are handled specially by elf32.em.  */
    *(.gnu.warning)
    . = ALIGN(4);
  } :text
  .empty.init.fini :
  {
    /* Our elf loader checks that .empty* sections are empty, and
       can give a helpful error, rather than an ASSERT here.  */
    KEEP (*(.init .fini))
  } :text

/* Finish off the text area.  */
#if defined(TYPE_FIRMWARE)
/* Need separation beteen end of firmware and start of kernel, so that
   they don't share unflushable i$ cache lines.  */
#if defined(COMPILE_FOR_IERISC) || defined(COMPILE_FOR_SUBORDINATE_IERISC) || defined(COMPILE_FOR_AERISC)
#define MEM_PAD MEM_IERISC_KERNEL_PAD
#else
#define MEM_PAD 0
#endif
/* FW must align to 16 byte boundary so kernel begins aligned to meet noc alignment constraints */
  . = ALIGN(ABSOLUTE(.) + MEM_PAD, 16);
  __fw_export_text_end = ABSOLUTE(.);
#else
  __kernel_data_lma = .;
#endif

/* Begin the data area.  */
#if defined(TYPE_FIRMWARE)
  PROVIDE(__global_pointer$ = DATA_START + 0x7f0);
  .data DATA_START :
#define SECTION(NEW, PREV) NEW :
#else
  /* Where firmware ended. */
  .data __fw_export_ldm_end : AT(__kernel_data_lma)
/* We can't rely on the linker's LMA computation because that fails when data is before text.  */
#define SECTION(NEW, PREV) NEW : AT(LOADADDR(PREV) + (ADDR(NEW) - ADDR(PREV)))
#endif
  {
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
  } :data
  SECTION(.empty.ctors.dtors, .data)
  {
    /* We don't use .ctors/.dtors either (this still isn't the '90s), make sure there aren't any.  */
    KEEP (*(.ctors .ctors.* .dtors .dtors.*))
  } :data
  SECTION(.empty.init_array.fini_array, .empty.ctors.dtors)
  {
    /* We don't support global static constructors or destructors. make sure there aren't any.  */
    KEEP (*(.preinit_array))
    KEEP (*(SORT_BY_INIT_PRIORITY(.init_array.*)))
    KEEP (*(.init_array))
    KEEP (*(SORT_BY_INIT_PRIORITY(.fini_array.*)))
    KEEP (*(.fini_array))
  } :data
  SECTION(.bss, .empty.init_array.fini_array) ALIGN(4)
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

  . = ALIGN(ABSOLUTE(.), 16);
#ifdef TYPE_FIRMWARE
  __fw_export_ldm_end = ABSOLUTE(.);
  __stack_top = DATA_START + DATA_SIZE;
#else
  __stack_base = .;
#endif

  /* The loader uses this data to check for size overflows.
     We don't do it here, as the failure mode would be bad (no executable to examine).  */
  .phdrs 0 (INFO) :
  {
    LONG(TEXT_SIZE
#if defined(TYPE_KERNEL) && !(defined(COMPILE_FOR_NCRISC) && defined(ARCH_WORMHOLE))
         - (__fw_export_text_end - TEXT_START)
#endif
         )
    LONG(DATA_SIZE - STACK_MIN_SIZE
#if defined(TYPE_KERNEL)
         - (__fw_export_ldm_end - DATA_START)
#endif
         )
  }

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
