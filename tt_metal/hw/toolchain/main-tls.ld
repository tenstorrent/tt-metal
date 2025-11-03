#if TYPE_FIRMWARE + TYPE_KERNEL != 1
#error "Exactly one of TYPE_FIRMWARE and TYPE_KERNEL must be defined"
#endif

#if (defined(COMPILE_FOR_TRISC) + defined(COMPILE_FOR_DM)) != 1
#error "Exactly one COMPILE_FOR_${PROC}RISC must be defined"
#endif

#if !ARCH_QUASAR
#error "Only quasar supported"
#endif

#include "dev_mem_map.h"
#include "tensix_dev_map.h"

/*

All kernels link their code at
  MEM_FOO_LOCAL_BASE + MEM_FOO_LOCAL_SIZE * NUM_FOO_CORES
(it is xip'ed executes from a buffer).

Kernels place their TLS or, for legacy kernels their globals, directly
after the firmware's TLS region. (For legacy kernels, these globals are
at the thread-specific region.)

Firmware and kernels have their TLS (or global) init image placed with
an LMA directly after their code.

TLS kernels place their globals after the firmware's globals. Both
firmware and TLS kernels globals have an LMA directly after TLS's LMA.

*/

#if 0
/* Empty block to make all the following blocks have the same template.  */

#elif defined(COMPILE_FOR_DM)
#if (0 - COMPILE_FOR_DM - 1) == 1
/* COMPILE_FOR_DM is empty -> a single TLS script */
#define IS_TLS 1
#endif
#define DATA_START MEM_DM_GLOBAL_BASE
#define DATA_SIZE MEM_DM_GLOBAL_SIZE
#define TLS_START  (MEM_DM_LOCAL_BASE + (COMPILE_FOR_DM + 0) * MEM_DM_LOCAL_SIZE)
#define TLS_SIZE MEM_DM_LOCAL_SIZE
#define TEXT_FIRMWARE_START MEM_DM_FIRMWARE_BASE
#define TEXT_START IF_FIRMWARE(TEXT_FIRMWARE_START)IF_KERNEL(MEM_DM_LOCAL_BASE + MEM_DM_LOCAL_SIZE * NUM_DM_CORES)
#define TEXT_SIZE IF_FIRMWARE(MEM_DM_FIRMWARE_SIZE)IF_KERNEL(MEM_DM_KERNEL_SIZE)
#define STACK_MIN_SIZE MEM_DM_STACK_MIN_SIZE

#elif defined(COMPILE_FOR_TRISC)
#if (0 - COMPILE_FOR_TRISC - 1) == 1
/* COMPILE_FOR_TRISC is empty -> a single TLS script */
#define IS_TLS 1
#endif
#define DATA_START MEM_TRISC_GLOBAL_BASE
#define DATA_SIZE MEM_TRISC_GLOBAL_SIZE
#define TLS_START  (MEM_TRISC_LOCAL_BASE + (COMPILE_FOR_TRISC + 0) * MEM_TRISC_LOCAL_SIZE)
#define TLS_SIZE MEM_TRISC_LOCAL_SIZE
#define TEXT_FIRMWARE_START MEM_TRISC_FIRMWARE_BASE
#define TEXT_START IF_FIRMWARE(TEXT_FIRMWARE_START)IF_KERNEL(MEM_TRISC_LOCAL_BASE + MEM_TRISC_LOCAL_SIZE * NUM_TRISC_CORES)
#define TEXT_SIZE IF_FIRMWARE(MEM_TRISC_FIRMWARE_SIZE)IF_KERNEL(MEM_TRISC_KERNEL_SIZE)
#define STACK_MIN_SIZE MEM_TRISC_STACK_MIN_SIZE

#else
#error "compiling for unknown"
#endif

#if IS_TLS
#define IF_TLS(...) __VA_ARGS__
#define IF_NOT_TLS(...)
#else
#define IF_TLS(...)
#define IF_NOT_TLS(...) __VA_ARGS__
#endif

#if TYPE_FIRMWARE
#define IF_FIRMWARE(...) __VA_ARGS__
#define IF_KERNEL(...)
#elif TYPE_KERNEL
#define IF_FIRMWARE(...)
#define IF_KERNEL(...) __VA_ARGS__
#else
#error "Neither TYPE_FIRMWARE nor TYPE_KERNEL defined"
#endif

#if ARCH_QUASAR && defined(COMPILE_FOR_DM)
OUTPUT_FORMAT("elf64-littleriscv", "elf64-littleriscv",
	      "elf64-littleriscv")
#else
OUTPUT_FORMAT("elf32-littleriscv", "elf32-littleriscv",
	      "elf32-littleriscv")
#endif
OUTPUT_ARCH(riscv)

ENTRY(_start)

PHDRS {
  attributes 0x70000003;
  IF_KERNEL(firmtext PT_LOAD; firmtextdel 0x60000000;)
  text PT_LOAD;

  IF_KERNEL(firmtdata PT_LOAD; firmtls PT_TLS; firmtdatadel 0x60000000;)
#if IS_TLS
  tdata PT_LOAD;
  tls PT_TLS;
#endif

  IF_KERNEL(firmdata PT_LOAD; firmdatadel 0x60000000;)
  data PT_LOAD;
}

SECTIONS {
/* Text-like */
#if TYPE_KERNEL
  .firmtext TEXT_FIRMWARE_START
  : {
    *_object.o(.text)
  } :firmtext :firmtextdel
#endif
  .text TEXT_START
  : {
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
   IF_TLS(__tdata_lma = .;)
   IF_NOT_TLS(__data_lma = .;)

#if TYPE_KERNEL
  .firmtdata TLS_START : AT(TLS_START + TLS_SIZE) { *_object.o(.tdata) } :firmtdata :firmtls :firmtdatadel
  .firmtbss : { *_object.o(.tbss) }  :firmtdata :firmtls :firmtdatadel
#endif
#if IS_TLS
/* TLS-like */
  .tdata IF_FIRMWARE(TLS_START)IF_KERNEL(ADDR(.firmtbss) + SIZEOF(.firmbss))
  : AT(__tdata_lma)
  {
    *(.tdata.start) /* define __tdata_start as TLS symbol) */
    *(.tdata .tdata.* .gnu.linkonce.td.*)
    . = ALIGN(4);
    *(.tdata.end)
  } :tdata :tls
  __tdata_words = ABSOLUTE(SIZEOF(.tdata))  / 4;
  .tbss
  : {
    *(.tbss.start)
    *(.tbss .tbss.* .gnu.linkonce.tb.* .tcommon)
    . = ALIGN(4);
    *(.tbss.end)
  } :tdata :tls
  __tbss_words = ABSOLUTE(SIZEOF(.tbss)) / 4;
#endif

/* Data-like */
#if TYPE_KERNEL
  .firmdata DATA_START : AT(LOADADDR(.firmtdata) + SIZEOF(.firmtdata)) { *_object.o(.data) } :firmdata :firmdatadel
  .firmbss : { *_object.o(.bss) }  :firmdata :firmdatadel
#endif
#if TYPE_KERNEL
  /DISCARD/ : { *_object.o(.empty.ctors.dtors .empty.init_array.fini_array) }
#endif
  .data IF_FIRMWARE(DATA_START)
        IF_KERNEL(IF_TLS(ADDR(.firmbss) + SIZEOF(.firmbss))IF_NOT_TLS(ADDR(.firmtbss) + SIZEOF(.firmtbss)))
  : AT(IF_TLS(LOADADDR(.tdata) + SIZEOF(.tdata))IF_NOT_TLS(__data_lma))
  {
    IF_NOT_TLS(__data_start = .;)
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
    IF_NOT_TLS(__data_end = .;)
  } :data
  IF_NOT_TLS(__data_words = ABSOLUTE(SIZEOF(.data)) / 4;)
  .empty.ctors.dtors
  : {
    /* We don't use .ctors/.dtors either (this still isn't the '90s), make sure there aren't any.  */
    KEEP (*(.ctors .ctors.* .dtors .dtors.*))
  } :data
  .empty.init_array.fini_array
  : {
    /* We don't support global static constructors or destructors. make sure there aren't any.  */
    KEEP (*(.preinit_array))
    KEEP (*(SORT_BY_INIT_PRIORITY(.init_array.*)))
    KEEP (*(.init_array))
    KEEP (*(SORT_BY_INIT_PRIORITY(.fini_array.*)))
    KEEP (*(.fini_array))
  } :data
#if IF_NOT_TLS(1) + 0
  .empty.tls
  : {
    *(.tdata .tdata.* .gnu.linkonce.td.*)
    *(.tbss .tbss.* .gnu.linkonce.tb.* .tcommon)
  } :data
#endif
  .bss
  : {
    IF_NOT_TLS(__bss_start = .;)
    *(.sbss2 .sbss2.* .gnu.linkonce.sb2.*)
    *(.dynsbss)
    *(.sbss .sbss.* .gnu.linkonce.sb.*)
    *(.scommon)
    *(.dynbss)
    *(.bss .bss.* .gnu.linkonce.b.*)
    *(COMMON)
    . = ALIGN(4);
    IF_NOT_TLS(__bss_end = .;)
  } :data
  IF_NOT_TLS(__bss_words = ABSOLUTE(SIZEOF(.bss)) / 4;)

#if TYPE_KERNEL
  /DISCARD/ : { *_object.o(*) }
#endif

  /* The loader uses this data to check for size overflows.
     We don't do it here, as the failure mode would be bad (no executable to examine).
     These are indexed by non-deleted non-ignored segment numbering. */
  .phdrs 0 (INFO) :
  {
    LONG(TEXT_SIZE)
    IF_TLS(LONG(TLS_SIZE - (ADDR(.tdata) - TLS_START)))
    LONG(DATA_SIZE - STACK_MIN_SIZE - (ADDR(.data) - IF_TLS(DATA_START)IF_NOT_TLS(TLS_START)))
  }

  .riscv.attributes 0 : { *(.riscv.attributes) } :attributes

}
