/* SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
**
** SPDX-License-Identifier: Apache-2.0
*/

/* TYPE_FIRMWARE=0 -> Firmware
   TYPE_KERNEL=0 -> TLS kernel
   TYPE_KERNEL=1 -> legacy kernel
  */
#if defined(TYPE_FIRMWARE)
#define IF_FIRMWARE(...) __VA_ARGS__
#define IF_KERNEL(...)
#define IS_TLS 1
#elif defined(TYPE_KERNEL)
#define IF_FIRMWARE(...)
#define IF_KERNEL(...) __VA_ARGS__
#define IS_TLS (!TYPE_KERNEL)
#else
#error "Neither TYPE_FIRMWARE nor TYPE_KERNEL defined"
#endif

#if !ARCH_QUASAR
#error "Only quasar supported"
#endif

/*
  We generate the following linker scripts:
  FW Firmware.  Runs at specific address, has shared globals and per-cpu TLS region
  KN-tls TLS kernel.  XIP execution, has shared globals and per-cpu TLS region
  KN-lgc Legacy kernel.  XIP execution, has per-cpu globals and TLS region

  Kernels may access Firmware globals and TLS
  the following symbols defined by the user (via -Wl,--defsym=NAME=EXPR)

  __fw_text -- base address of firmware code (FW)
  __fw_data -- base address of firmware globals (FW,KN-{tls,lgc})
  __fw_tls  -- base address of firmware TLS (FW,KN-{tls,lgc})
  __kn_data -- base address of kernel globals (KN-tls)
  __kn_text -- base address of kernel code (KN-{tls,lgc})

  Legacy kernels place their global data in the tls region.

  The following limits should also be defined:

  __text_size -- space for code
  __data_size -- space for globals
  __tls_size  -- space for TLS & stack (& legacy kernel globals)
  __min_stack -- minimum stack size limit
*/

#define FW_DATA_START __fw_data
#define DATA_START IF_FIRMWARE(FW_DATA_START)IF_KERNEL(__kn_data)
#define TLS_START __fw_tls
#define TEXT_START IF_FIRMWARE(__fw_text)IF_KERNEL(__kn_text)

#define DATA_SIZE __data_size
#define TLS_SIZE __tls_size
#define TEXT_SIZE __text_size
#define STACK_MIN_SIZE __min_stack

#if IS_TLS
#define IF_TLS(...) __VA_ARGS__
#define IF_NOT_TLS(...)
#else
#define IF_TLS(...)
#define IF_NOT_TLS(...) __VA_ARGS__
#endif

/* Abut the previous section.  */
#define LOAD_AFTER_ABUT(THIS,PREV) \
  AT(LOADADDR(PREV) + SIZEOF(PREV))
/* Preserve LMA/VMA delta.  */
#define LOAD_AFTER_ALIGN(THIS,PREV) \
  AT(LOADADDR(PREV) + (ADDR(THIS) - ADDR(PREV)))

#if ARCH_QUASAR && defined(COMPILE_FOR_DM)
OUTPUT_FORMAT("elf64-littleriscv", "elf64-littleriscv",
	      "elf64-littleriscv")
#else
OUTPUT_FORMAT("elf32-littleriscv", "elf32-littleriscv",
	      "elf32-littleriscv")
#endif
OUTPUT_ARCH(riscv)

LD_FEATURE(SANE_EXPR)

ENTRY(_start)

PHDRS {
  attributes 0x70000003;
  text PT_LOAD;
  tdata PT_LOAD;
  tls PT_TLS;
  data PT_LOAD;
  IF_KERNEL(firmdata PT_LOAD;)
}
/* If not defined data is placed per-cpu after TLS.  */
IF_KERNEL(IF_TLS(PROVIDE_HIDDEN(DATA_START = 0);))
SECTIONS {
/* Text-like */
#if defined(TYPE_KERNEL)
  /DISCARD/ : { *_object.o(.text) }
#endif
  .text TEXT_START
  :
  {
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
   IF_KERNEL(__tdata_lma = .;)

  .tdata TLS_START
  : IF_KERNEL(AT(__tdata_lma))
  {
    IF_KERNEL(*_object.o(.tdata .tbss))
    . = ALIGN(4);
    __ldm_tdata_init = .;
    *(.tdata.start) /* define __tdata_start as TLS symbol) */
    *(.tdata .tdata.* .gnu.linkonce.td.*)
    . = ALIGN(4);
    *(.tdata.end)
  } :tdata :tls
  .tbss
  : IF_KERNEL(LOAD_AFTER_ALIGN(.tbss, .tdata))
  {
    *(.tbss.start) /* define __tbss_start as TLS symbol) */
    *(.tbss .tbss.* .gnu.linkonce.tb.* .tcommon)
    . = ALIGN(4);
    *(.tbss.end)
  } :tdata :tls

#if defined(TYPE_KERNEL)
  /DISCARD/ : { *_object.o(.empty.*) }
#endif

#if defined(TYPE_KERNEL)
  .firmdata FW_DATA_START
  : AT(FW_DATA_START)
  {
    *object.o(.data .bss)
  } :firmdata
#endif
  .data IF_FIRMWARE(DATA_START)
        IF_KERNEL(IF_TLS(DATA_START ? DATA_START : ADDR(.firmdata) + SIZEOF(.firmdata))
                  IF_NOT_TLS(ADDR(.tbss) + SIZEOF(.tbss)))
  : IF_KERNEL(LOAD_AFTER_ABUT(.data, .tdata))
  {
    __ldm_data_start = .;
    *(.rodata .rodata.* .gnu.linkonce.r.*)
    *(.rodata1)

    *(.data.rel.ro.local* .gnu.linkonce.d.rel.ro.local.*) *(.data.rel.ro .data.rel.ro.* .gnu.linkonce.d.rel.ro.*)

    *(.srodata.cst16) *(.srodata.cst8) *(.srodata.cst4) *(.srodata.cst2) *(.srodata .srodata.*)
    *(.sdata .sdata.* .gnu.linkonce.s.*)
    *(.sdata2 .sdata2.* .gnu.linkonce.s2.*)

    *(.data .data.* .gnu.linkonce.d.*)
    *(.data1)

    . = ALIGN(4);
    __ldm_data_end = .;
  } :data
  .empty.ctors.dtors
  : LOAD_AFTER_ALIGN(.empty.ctors.dtors, .data)
  {
    /* We don't use .ctors/.dtors either (this still isn't the '90s), make sure there aren't any.  */
    KEEP (*(.ctors .ctors.* .dtors .dtors.*))
  } :data
  .empty.init_array.fini_array
  : LOAD_AFTER_ALIGN(.empty.init_array.fini_array, .empty.ctors.dtors)
  {
    /* We don't support global static constructors or destructors. make sure there aren't any.  */
    KEEP (*(.preinit_array))
    KEEP (*(SORT_BY_INIT_PRIORITY(.init_array.*)))
    KEEP (*(.init_array))
    KEEP (*(SORT_BY_INIT_PRIORITY(.fini_array.*)))
    KEEP (*(.fini_array))
  } :data
  .bss
  : LOAD_AFTER_ALIGN(.bss, .empty.init_array.fini_array)
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
#if defined(TYPE_KERNEL)
  __stack_base_offset = IF_TLS(0)IF_NOT_TLS((ABSOLUTE(__ldm_bss_end) - ABSOLUTE(__stack_base_lwm)) / 4);
#endif

#if defined(TYPE_KERNEL)
  /DISCARD/ : { *_object.o(*) }
#endif

  /* The loader uses this data to trim segments and check for size overflows.
     SEGMENT_ADDR:TRIM_BOUND:SIZE_LIMIT */
  .segments 0 (INFO)
  :
  {
    LONG(ADDR(.text)) LONG(ADDR(.text)) LONG(TEXT_SIZE)
    LONG(ADDR(.tdata)) LONG(__ldm_tdata_init)
       LONG(TLS_SIZE - STACK_MIN_SIZE - (__ldm_tdata_init - ADDR(.tdata)))
    LONG(ADDR(.data)) LONG(__ldm_data_start) LONG(DATA_SIZE - (__ldm_data_start - ADDR(.data)))
    IF_KERNEL(LONG(ADDR(.firmdata)) LONG(ADDR(.firmdata) + SIZEOF(.firmdata)) LONG(0))
  }

  .riscv.attributes 0 : { *(.riscv.attributes) } :attributes

}
