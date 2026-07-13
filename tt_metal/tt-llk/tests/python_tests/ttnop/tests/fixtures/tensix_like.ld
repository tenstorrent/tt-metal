/* Mimics the layout of a linked Tenstorrent kernel: .text lives in its own
 * R+E PT_LOAD whose file range starts *after* the ELF/program headers (the
 * headers are NOT mapped into the segment), and data/bss sit in a separate
 * R+W segment. This is the layout ttnop is designed for. */
PHDRS {
  text PT_LOAD FLAGS(5);   /* R+E -- no FILEHDR/PHDRS keyword => headers excluded */
  data PT_LOAD FLAGS(6);   /* R+W */
}
ENTRY(_start)
SECTIONS {
  . = 0x10000;
  .text : { *(.text*) } :text
  /* Large gap before data, like a real kernel (data at 0xffb00000), so the
   * code cave appended after .text has room even under heavy instrumentation. */
  . = 0x200000;
  .data : { *(.data*) *(.sdata*) *(.rodata*) } :data
  .bss  : { *(.bss*) *(.sbss*) *(COMMON) } :data
  /DISCARD/ : { *(.riscv.attributes) *(.comment) *(.note*) *(.eh_frame*) }
}
