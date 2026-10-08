#!/bin/bash
# Build $KP/base-root: a symlink mirror of the worktree whose indexer_score kernels dir holds the UNMODIFIED
# (HEAD~N / given rev) kernels, so "before" runs use the same Python and build with only the kernels swapped.
# usage: make-base-root.sh <worktree> <out_root> [rev=kagent/m3-prefill]
set -eu
W=$1; B=$2; REV=${3:-kagent/m3-prefill}
rel=ttnn/cpp/ttnn/operations/experimental/indexer_score/device/kernels
rm -rf "$B"; mkdir -p "$B"
src=$W; dst=$B; IFS=/ read -ra parts <<< "$rel"
for p in "${parts[@]}"; do
  for e in $(ls -A "$src"); do [ "$e" = "$p" ] && continue; ln -s "$src/$e" "$dst/$e"; done
  mkdir "$dst/$p"; src=$src/$p; dst=$dst/$p
done
for f in $(git -C "$W" ls-tree --name-only "$REV" $rel/); do git -C "$W" show "$REV:$f" > "$B/$f"; done
echo "base-root $B: kernels from $REV"
