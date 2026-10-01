#!/bin/bash
# Off-device check: compile the neighbor_pad_async dataflow kernels (BRISC/NCRISC) with the exact JIT
# commands logged by blx03 job 994 (#43, BH, 128-ch sticks), against a worktree's sources.
# Usage: kernel_compile.sh <worktree> [git-rev]   (git-rev: compile that revision's kernel files instead)
# Needs only the sfpi cross compiler; no device, no JIT cache.
set -u
WT=$(realpath "${1:-.}"); REV=${2:-}
GXX=${SFPI_GXX:-/home/smarton/fasth3/tt-metal/runtime/sfpi/compiler/bin/riscv-tt-elf-g++}
KDIR=ttnn/cpp/ttnn/operations/experimental/ccl/neighbor_pad_async/device/kernels
OUT=$(mktemp -d /tmp/t60kc.XXXXXX); rc=0
trap 'rm -rf "$OUT"' EXIT
while read -r cmd; do
  name=$(sed -E 's/.*FULL_KERNEL_NAME="([a-z0-9_]+)\/.*/\1/' <<<"$cmd")
  args=$(sed -E 's/.*-DKERNEL_COMPILE_TIME_ARGS=([0-9,]+).*/\1/' <<<"$cmd")
  risc=$(grep -oE 'COMPILE_FOR_(BRISC|NCRISC)' <<<"$cmd" | tr 'A-Z' 'a-z' | sed 's/compile_for_//')
  d=$OUT/$name.$args; mkdir -p $d/$risc
  src=$WT/$KDIR/$name.cpp
  if [ -n "$REV" ]; then src=$d/$name.cpp; git -C "$WT" show "$REV:$KDIR/$name.cpp" >"$src"; fi
  printf 'void kernel_main();\n#include "%s"\n' "$src" >$d/kernel_includes.hpp
  c=${cmd%% *}; c=${cmd#"$c"}  # drop the logged compiler path
  c=${c//\/home\/smarton\/fasth3\/tt-metal\//$WT/}
  c=$(sed -E "s# -include [^ ]+pch\.h# -include $WT/tt_metal/hw/inc/internal/pch.h#; s# -o [^ ]+# -o $d/k.o#; s# -MF [^ ]+# -MF $d/k.d#" <<<"$c")
  if (cd $d/$risc && eval "$GXX $c") 2>$d/err; then
    echo "OK   $name ($risc) args=$args"
  else
    echo "FAIL $name ($risc) args=$args"; grep -m3 -E "error|Error" $d/err; rc=1
  fi
done < "$(dirname "$(realpath "$0")")/np_kernel_cmds.txt"
exit $rc
