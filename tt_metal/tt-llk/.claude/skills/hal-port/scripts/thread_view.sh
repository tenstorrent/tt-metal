#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Build an include tree where one thread's LLK headers come from this worktree's tests/hal_llk (being ported) and
# every other header comes from production llk_lib. Each TRISC is compiled on its own, so a test then exercises the
# port of that thread alone, independent of unfinished work on the other threads.
# usage: thread_view.sh <unpack|math|pack> <out_dir>      (run from a tt-llk worktree)
# then:  chip_queue.py run --worktree $PWD --test test_x.py --impl <out_dir> ...
# Files are symlinks, so edits in tests/hal_llk are seen without rebuilding the view.
set -euo pipefail
THREAD=$1; OUT=$2; LLK=$(pwd)
case $THREAD in
  unpack) PAT='^llk_unpack_' ;;
  math) PAT='^llk_math_|^llk_fp32_dest_acc' ;;
  pack) PAT='^llk_pack|^llk_lib_pack_wrappers' ;;
  *) echo "thread must be unpack, math or pack"; exit 2 ;;
esac
rm -rf "$OUT"; mkdir -p "$OUT/experimental"
for f in "$LLK"/tests/hal_llk/*.h "$LLK"/tests/hal_llk/experimental/*.h; do
  rel=${f#"$LLK"/tests/hal_llk/}; base=$(basename "$rel")
  if [[ $base =~ $PAT ]]; then
    ln -s "$f" "$OUT/$rel"
  elif [[ -f "$LLK/tt_llk_blackhole/llk_lib/$rel" ]]; then
    ln -s "$LLK/tt_llk_blackhole/llk_lib/$rel" "$OUT/$rel"
  fi
done
echo "thread view for $THREAD: $(find "$OUT" -type l | wc -l) headers in $OUT"
