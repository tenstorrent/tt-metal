#!/usr/bin/env bash
# int32 compares and int32 add/sub at op level on one CI Blackhole runner (knowledge/process/17_CI_AB.md). The job's tree is
# the new #60107 head; P = the previous heads' files (four-call compares, per-call ADDR_MOD_6 in add/sub), C = the one-call
# compares only, N = both. Fresh JIT cache per run. Phases: compute ELF images per case, outputs against torch and P, then
# op level ABBA twice per pair (P/C, C/N) with an A/A pair.
set +e +o pipefail
R=$(git rev-parse --show-toplevel); S=$R/ci_ab; OUT=/tmp/sfo; rm -rf $OUT; mkdir -p $OUT/prof; cd $R
export TT_METAL_HOME=$R TRACY_NO_WEB_SERVER=1 HWLOCK_HELD=ci
RT=$(cd /tmp && env -u PYTHONPATH python3 -c "import ttnn, os; print(os.path.dirname(ttnn.__file__))" 2>/dev/null)
echo "== runner $(hostname) $(date -u +%FT%TZ) head $(git log -1 --format=%h) runtime root $RT"
cpv() { (cd $1 && find . -type f | while read f; do cp "$f" "$R/$f"; [ -f "$RT/$f" ] && cp "$f" "$RT/$f"; done); }
for d in $S/var/*; do echo "== $(basename $d): $(cd $d && find . -type f | wc -l) files, $(cd $d && find . -type f | while read f; do [ -f "$RT/$f" ] && echo x; done | wc -l) in the runtime root"; done
NC=0
use() { NC=$((NC+1)); cpv $S/var/${1%_aa}; export TT_METAL_CACHE=$OUT/cache/${1}_$NC; mkdir -p $TT_METAL_CACHE; }
for c in cmp_lt_5_dram cmp_eq_5_dram add_dram sub_dram add_relu_dram cumsum_int32 sum_int32 add_float_dram; do for v in P C N; do use $v; timeout 600 python3 $S/tools/elf_case.py $c > $OUT/elf_${v}_$c.log 2>&1; grep -a "^ELF" $OUT/elf_${v}_$c.log | sed "s/^ELF/ELF $v/"; done; done
for v in P C N; do use $v; R_BITS=$OUT/bits_$v.json timeout 1200 python3 $S/tools/prof_int.py > $OUT/bits_$v.log 2>&1; echo "== bits $v rc=$? $(grep -a 'saved' $OUT/bits_$v.log)"; grep -a "Error" $OUT/bits_$v.log | head -3; done
python3 -c "import json; a=json.load(open('$OUT/bits_P.json')); c=json.load(open('$OUT/bits_C.json')); n=json.load(open('$OUT/bits_N.json')); print('== bits differ P/C:', [k for k in a if a[k][0]!=c.get(k,[0])[0]], 'P/N:', [k for k in a if a[k][0]!=n.get(k,[0])[0]])"
run() { local v=$1 k=$2; use $v; local O=$OUT/prof/r${k}_$v; mkdir -p $O
  TT_METAL_PROFILER_DIR=$O/pdir R3_NMEAS=${NMEAS:-10} timeout -s INT -k 60 1500 python3 -m tracy -r -p --no-web-server -o $O $S/tools/prof_int.py > $O.log 2>&1
  echo "   run $k $v rc=$? $(grep -c 'CASE .* ok' $O.log) ok $(date -u +%T)"; grep -a "FAILED" $O.log | head -3
  python3 $S/tools/parse_ops.py $O "r$k $v"; }
k=0
for pr in "P C" "C N" "P P_aa"; do set -- $pr; for v in $1 $2 $2 $1 $1 $2 $2 $1; do k=$((k+1)); run $v $k; done; done
echo "== done $(date -u +%T)"
exit 0
