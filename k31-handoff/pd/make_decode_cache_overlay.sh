#!/bin/bash
# Overlay Saad's K3.1 decode TensorCache: real objects/<prefix>/ dirs under our root, each of his artifact dirs
# symlinked in. Hits read his files; a miss publishes into our prefix dir, never into his.
set -u
SRC=/mnt/weka/model-cache/scratch/moonshotai/blaze-k31-reload-1105
DST=/data/bklockiewicz/k31-decode-cache
mkdir -p $DST/objects
n=0
for p in $SRC/objects/*/; do
  pre=$(basename $p); mkdir -p $DST/objects/$pre
  for a in $p*/; do
    [ -d "$a" ] || continue
    id=$(basename $a); [ -e $DST/objects/$pre/$id ] || ln -s "${a%/}" $DST/objects/$pre/$id
    n=$((n+1))
  done
done
echo "overlay: $n artifact(s) linked from $SRC into $DST $(date -u +%T)"
ls $SRC | grep -vx objects | sed 's/^/note: root also has /'
