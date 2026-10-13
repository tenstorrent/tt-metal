#!/usr/bin/env bash
# Scratch job (TopK headroom round), not for merge: allpat_dump.py on each arm (its files swapped in, own JIT cache),
# then allpat_cmp.py on the pairs in $PAIRS ("x-y ...").
set +e +o pipefail
cd /work; export TT_METAL_HOME=/work HWLOCK_HELD=ci
S=/work/ci_ab; OUT=/work/generated/allpat; rm -rf $OUT; mkdir -p $OUT
SP=$(cd /tmp && python3 -c "import ttnn, os; print(os.path.dirname(os.path.dirname(os.path.abspath(ttnn.__file__))))")
echo "== $(date -u +%FT%TZ) $(git log -1 --format='%h %s')"
suffix() { echo "$1" | awk -F/ '{n=NF; print $(n-3)"/"$(n-2)"/"$(n-1)"/"$n}'; }
use() {
  while read -r rel; do
    cp "$S/variants/$1/$rel" "/work/$rel"
    while read -r p; do [[ -n $p && $p != "/work/$rel" ]] && cp "$S/variants/$1/$rel" "$p"; done < <(find "$SP" -path "*/$(suffix $rel)" -type f 2>/dev/null)
  done < $S/files.txt
  echo "== files $1 $(while read -r rel; do cat /work/$rel; done < $S/files.txt | md5sum | cut -c1-8)"
  export TT_METAL_CACHE=$OUT/cache_$1; mkdir -p $TT_METAL_CACHE
}
for a in ${ARMS:-A B}; do
  use $a
  timeout -s INT -k 60 1500 python3 $S/allpat_dump.py $OUT/$a.pt > $OUT/$a.log 2>&1
  echo "== dump $a rc=$? $(grep -a 'ALLPAT DONE' $OUT/$a.log)"
  grep -a -m3 -E "Error|FATAL" $OUT/$a.log | cut -c1-240
done
for p in ${PAIRS:-A-B}; do python3 $S/allpat_cmp.py $OUT/${p%%-*}.pt $OUT/${p##*-}.pt ${p%%-*} ${p##*-}; done
echo "== done"
