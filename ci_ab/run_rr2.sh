#!/usr/bin/env bash
# reshuffle_rows op level on one CI Blackhole runner (knowledge/process/17_CI_AB.md): the job's tree is the new head (host
# passes the replay switch to the compute kernel); P = the previous head's device files (reshuffle LLK, API and the compute
# kernel, which ignores the switch), N = the new head's. Fresh JIT cache per run. Phases: ELF images per case and variant
# (N must equal P where the switch is off), bits of every index pattern P against N, then op level ABBA twice per pair
# with an A/A pair, one tracy process per run over every case.
set +e +o pipefail
R=$(git rev-parse --show-toplevel); S=$R/ci_ab; OUT=/tmp/sfo; rm -rf $OUT; mkdir -p $OUT/prof; cd $R
export TT_METAL_HOME=$R TRACY_NO_WEB_SERVER=1 HWLOCK_HELD=ci
RT=$(cd /tmp && env -u PYTHONPATH python3 -c "import ttnn, os; print(os.path.dirname(ttnn.__file__))" 2>/dev/null)
echo "== runner $(hostname) $(date -u +%FT%TZ) head $(git log -1 --format=%h) runtime root $RT"
cpv() { (cd $1 && find . -type f | while read f; do cp "$f" "$R/$f"; [ -f "$RT/$f" ] && cp "$f" "$RT/$f"; done); }
for d in $S/var/*; do echo "== $(basename $d): $(cd $d && find . -type f | wc -l) files, $(cd $d && find . -type f | while read f; do [ -f "$RT/$f" ] && echo x; done | wc -l) in the runtime root"; done
NC=0
use() { NC=$((NC+1)); cpv $S/var/${1%_aa}; export TT_METAL_CACHE=$OUT/cache/${1}_$NC; mkdir -p $TT_METAL_CACHE; }
ECASES="emb_2x64x160x96 emb_2x64x160x256 emb_2x64x160x320 emb_2x1024x4096x3200 emb_8x128x768x30522 emb_1x2048x2048x32000"
for c in $ECASES; do for v in P N; do use $v; timeout 600 python3 $S/tools/elf_case.py $c > $OUT/elf_${v}_$c.log 2>&1; grep -a "^ELF" $OUT/elf_${v}_$c.log | sed "s/^ELF/ELF $v/"; done; done
for v in P N; do use $v; mkdir -p $OUT/bits_$v; python3 $S/tools/exh_rr.py $OUT/bits_$v/all.json > $OUT/bits_$v.log 2>&1; echo "== bits $v rc=$? $(tail -1 $OUT/bits_$v.log)"; done
echo "== compare P against N"; python3 $S/tools/cmp_json.py $OUT/bits_P $OUT/bits_N
CASES="emb_2x64x160x96 emb_2x64x160x256 emb_8x128x768x256 emb_2x64x160x320 emb_8x128x768x320 emb_2x1024x4096x3200 emb_1x2048x2048x32000 emb_8x128x768x30522 k1_8x128x768x30522 k4_8x128x768x30522"
run() { local v=$1 k=$2; use $v; local O=$OUT/prof/r${k}_$v; mkdir -p $O
  TT_METAL_PROFILER_DIR=$O/pdir R3_NMEAS=${NMEAS:-10} timeout -s INT -k 60 1500 python3 -m tracy -r -p --no-web-server -o $O $S/tools/prof_rr2.py $CASES > $O.log 2>&1
  echo "   run $k $v rc=$? $(grep -c 'CASE .* ok' $O.log) ok $(date -u +%T)"; grep -a "FAILED" $O.log | head -3
  python3 $S/tools/parse_rr.py $O "r$k $v"; }
k=0
for pr in "P N" "P P_aa"; do set -- $pr; for v in $1 $2 $2 $1 $1 $2 $2 $1; do k=$((k+1)); run $v $k; done; done
echo "== done $(date -u +%T)"
exit 0
