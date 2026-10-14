#!/usr/bin/env bash
# rand_tile users at op level on one CI Blackhole runner (knowledge/process/17_CI_AB.md): the job's tree is the new rand
# head; P = the previous head's device files (rand LLK, the compute API, the kernel_lib generator), G = the new head's
# (one call per tile in the generator only). Fresh JIT cache per run. Phases: compute ELF images per case and variant
# (sampling and randn must be identical), output bits P against G, then op level ABBA twice per pair with an A/A pair.
set +e +o pipefail
R=$(git rev-parse --show-toplevel); S=$R/ci_ab; OUT=/tmp/sfo; rm -rf $OUT; mkdir -p $OUT/prof; cd $R
export TT_METAL_HOME=$R TRACY_NO_WEB_SERVER=1 HWLOCK_HELD=ci
RT=$(cd /tmp && env -u PYTHONPATH python3 -c "import ttnn, os; print(os.path.dirname(ttnn.__file__))" 2>/dev/null)
echo "== runner $(hostname) $(date -u +%FT%TZ) head $(git log -1 --format=%h) runtime root $RT"
cpv() { (cd $1 && find . -type f | while read f; do cp "$f" "$R/$f"; [ -f "$RT/$f" ] && cp "$f" "$RT/$f"; done); }
for d in $S/var/*; do echo "== $(basename $d): $(cd $d && find . -type f | wc -l) files, $(cd $d && find . -type f | while read f; do [ -f "$RT/$f" ] && echo x; done | wc -l) in the runtime root"; done
NC=0
use() { NC=$((NC+1)); cpv $S/var/${1%_aa}; export TT_METAL_CACHE=$OUT/cache/${1}_$NC; mkdir -p $TT_METAL_CACHE; }
for c in rand_bf16_1024x1024 uniform_fp32_4096x2048 bernoulli_fp32_4096x2048 sampling_32x256 randn_bf16_1x16x128x128; do for v in P G; do use $v; timeout 600 python3 $S/tools/elf_case.py $c > $OUT/elf_${v}_$c.log 2>&1; grep -a "^ELF" $OUT/elf_${v}_$c.log | sed "s/^ELF/ELF $v/"; done; done
CASES="rand_bf16_4096x2048 rand_fp32_4096x2048 rand_bf16_1024x1024 rand_bf16_32x32 uniform_fp32_4096x2048 uniform_bf16_4096x2048 bernoulli_fp32_4096x2048 bernoulli_bf16_1x1x512x1024 rand_fp32_1x1026 rand_fp32_1x256x262144 sampling_32x256 randn_bf16_1x16x128x128"
for v in P G; do use $v; R_BITS=$OUT/bits_$v.json timeout 1200 python3 $S/tools/prof_rand.py $CASES > $OUT/bits_$v.log 2>&1; echo "== bits $v rc=$? $(grep -a saved $OUT/bits_$v.log)"; done
python3 -c "import json; a=json.load(open('$OUT/bits_P.json')); b=json.load(open('$OUT/bits_G.json')); print('== bits P vs G:', {k: a[k]==b.get(k) for k in a})"
run() { local v=$1 k=$2; use $v; local O=$OUT/prof/r${k}_$v; mkdir -p $O
  TT_METAL_PROFILER_DIR=$O/pdir R3_NMEAS=${NMEAS:-20} timeout -s INT -k 60 1500 python3 -m tracy -r -p --no-web-server -o $O $S/tools/prof_rand.py $CASES > $O.log 2>&1
  echo "   run $k $v rc=$? $(grep -c 'CASE .* ok' $O.log) ok $(date -u +%T)"; grep -a "FAILED" $O.log | head -3
  python3 $S/tools/parse_ops.py $O "r$k $v"; }
k=0
for pr in "P G" "P P_aa"; do set -- $pr; for v in $1 $2 $2 $1 $1 $2 $2 $1; do k=$((k+1)); run $v $k; done; done
echo "== done $(date -u +%T)"
exit 0
