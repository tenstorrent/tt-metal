#!/bin/bash
# textcmp.sh A.elf B.elf: exit 0 when the two ELFs' .text is the same instruction stream at the same addresses
# (objdump symbol annotations after '#' and '<...>' are ignored; they name debug labels that differ between builds).
OD=/home/aagarwal/tt-metal/runtime/sfpi/compiler/bin/riscv-tt-elf-objdump
t() { $OD -d --section=.text "$1" | grep -P '^\s+[0-9a-f]+:' | sed -E 's/\s*#.*$//; s/ <[^>]*>//g'; }
cmp -s <(t "$1") <(t "$2")
