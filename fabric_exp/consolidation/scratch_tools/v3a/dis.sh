#!/bin/bash
# dis.sh ELF: normalized .text instruction stream (address<TAB>insn), symbol annotations stripped
OD=/home/aagarwal/tt-metal/runtime/sfpi/compiler/bin/riscv-tt-elf-objdump
$OD -d --section=.text "$1" | grep -P '^\s+[0-9a-f]+:' | sed -E 's/\s*#.*$//; s/ <[^>]*>//g' | awk -F'\t' '{print $3"\t"$4}'
