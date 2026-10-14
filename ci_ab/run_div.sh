#!/usr/bin/env bash
# int32 division at op level on one CI Blackhole runner (knowledge/process/17_CI_AB.md): P = the division macro row as on
# the head, S = SFPGT moved into the MAD latency slot. Fresh JIT cache per run; ELF images, output words P against S, then
# op level ABBA twice with an A/A pair.
set +e +o pipefail
R=$(git rev-parse --show-toplevel); S=$R/ci_ab; OUT=/tmp/sdo; rm -rf $OUT; mkdir -p $OUT/prof; cd $R
export TT_METAL_HOME=$R TRACY_NO_WEB_SERVER=1 HWLOCK_HELD=ci
RT=$(cd /tmp && env -u PYTHONPATH python3 -c "import ttnn, os; print(os.path.dirname(ttnn.__file__))" 2>/dev/null)
echo "== op runner $(hostname) $(date -u +%FT%TZ) head $(git log -1 --format=%h) runtime root $RT"
cpv() { (cd $1 && find . -type f | while read f; do cp "$f" "$R/$f"; [ -f "$RT/$f" ] && cp "$f" "$RT/$f"; done); }
NC=0
use() { NC=$((NC+1)); cpv $S/var/${1%_aa}; export TT_METAL_CACHE=$OUT/cache/${1}_$NC; mkdir -p $TT_METAL_CACHE; }
for v in P S; do use $v; timeout 600 python3 $S/tools/elf_case.py div_dram_full > $OUT/elf_$v.log 2>&1; grep -a "^ELF" $OUT/elf_$v.log | sed "s/^ELF/ELF $v/"; done
for v in P S; do use $v; R_BITS=$OUT/bits_$v.json timeout 1200 python3 $S/tools/prof_div.py > $OUT/bits_$v.log 2>&1; echo "== bits $v rc=$? $(grep -a saved $OUT/bits_$v.log)"; done
python3 -c "import json; a=json.load(open('$OUT/bits_P.json')); b=json.load(open('$OUT/bits_S.json')); print('== bits P vs S:', {k: a[k]==b.get(k) for k in a})"
run() { local v=$1 k=$2; use $v; local O=$OUT/prof/r${k}_$v; mkdir -p $O
  TT_METAL_PROFILER_DIR=$O/pdir R3_NMEAS=10 timeout -s INT -k 60 1500 python3 -m tracy -r -p --no-web-server -o $O $S/tools/prof_div.py div_dram_full div_dram_small div_hs > $O.log 2>&1
  echo "   run $k $v rc=$? $(grep -c 'CASE .* ok' $O.log) ok $(date -u +%T)"; grep -a "FAILED" $O.log | head -3
  python3 $S/tools/parse_ops.py $O "r$k $v"; }
k=0
for pr in "P S" "P P_aa"; do set -- $pr; for v in $1 $2 $2 $1 $1 $2 $2 $1; do k=$((k+1)); run $v $k; done; done
echo "== op done $(date -u +%T)"
exit 0
