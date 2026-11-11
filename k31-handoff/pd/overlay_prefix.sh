#!/bin/bash
# overlay_prefix.sh <prefix>: link every artifact of one cache prefix (idempotent).
SRC=/mnt/weka/model-cache/scratch/moonshotai/blaze-k31-reload-1105; DST=/data/bklockiewicz/k31-decode-cache; p=$1
mkdir -p $DST/objects/$p; for a in $SRC/objects/$p/*/; do id=$(basename $a); [ -e $DST/objects/$p/$id ] || ln -s "${a%/}" $DST/objects/$p/$id; done
echo "$p $(ls $SRC/objects/$p | wc -l) $(ls $DST/objects/$p | wc -l)"
