#!/usr/bin/env bash
# #58819 stacked on #58816, op level (17_CI_AB.md): the device file of P, Z and Zr (and Z2, an A/A copy of Z)
# swapped into /work and the installed runtime
# root before each run, each with its own fresh kernel cache, rotated per round, one case per tracy process.
# Prints OLR <case> <variant> <round> <op> <calls> <median of the last N> lines and the kernels' trisc0 identity.
# usage: bash ci_ab/run_udc_ol.sh
set +e +o pipefail
cd /work
S=/work/ci_ab; G=/tmp/stkol; rm -rf $G; mkdir -p $G/out
export TT_METAL_HOME=/work TRACY_NO_WEB_SERVER=1 HWLOCK_HELD=ci R3_NMEAS=${R3_NMEAS:-12} R3_L1_SMALL=16384
export PYTHONPATH=/work:/work/tools:$S/tools
RT=$(cd /tmp && env -u PYTHONPATH python3 -c "import ttnn, os; print(os.path.dirname(ttnn.__file__))" 2>/dev/null)
echo "== ol start $(date -u +%T) $(git -C /work rev-parse HEAD 2>/dev/null) RT=$RT"
FILES=$(cd $S/olvar/Z && find . -type f | sed 's#^\./##')
for f in $FILES; do echo "== file $f work=$( [ -f /work/$f ] && echo y) rt=$( [ -f $RT/$f ] && echo y || echo n) $(for v in $(ls $S/olvar); do [ -f $S/olvar/$v/$f ] && echo -n "$v:$(md5sum $S/olvar/$v/$f | cut -c1-10) "; done)"; done
CUR=""
use() {  # use <variant>: its files into /work and the runtime root, its own kernel cache
  local v=$1 src=$1; [[ $v == Z2 ]] && src=Z
  for f in $FILES; do cp $S/olvar/$src/$f /work/$f; [ -f $RT/$f ] && cp $S/olvar/$src/$f $RT/$f; done
  export TT_METAL_CACHE=$G/cache_$v; mkdir -p $TT_METAL_CACHE; CUR=$v
}
cat > $G/sum.py <<'PYEOF'
import csv, glob, statistics, collections, sys, os
d, case, v, r, n = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
fs = glob.glob(d + '/reports/*/ops_perf_results_*.csv') + glob.glob(d + '/ops_perf_results_*.csv')
if not fs: print('OLR NOREPORT', case, v, r); sys.exit(0)
agg = collections.defaultdict(list)
for row in csv.DictReader(open(fs[0])):
    if row.get('OP TYPE') == 'signpost': continue
    try: val = float(row['DEVICE KERNEL DURATION [ns]'])
    except (KeyError, ValueError): continue
    agg[row['OP CODE']].append(val)
for op, vals in sorted(agg.items()):
    cut = vals[-n:]
    print('OLR', case, v, r, op, len(vals), round(statistics.median(cut)), ' '.join(str(round(x)) for x in cut))
PYEOF
CASES=${OL_CASES:-"mm_bf16_128x4096x128_sb4 mm_bf16_128x4096x128_sb8 conv2d_bf16_64x128_32x32 matmul_untilize_bf16_64x2048x128 swiglu_dsv3_t768_xtile conv2d_fp32acc_64x128_32x32"}
ROUNDS=${OL_ROUNDS:-4}
read -ra VL <<< "${OL_VARS:-P Z Zr Z2}"
for r in $(seq 1 $ROUNDS); do
  k=$(( (r - 1) % ${#VL[@]} )); VS=("${VL[@]:$k}" "${VL[@]:0:$k}")
  for c in $CASES; do
    for v in "${VS[@]}"; do
      use $v; o=$G/out/${c}__${v}__r$r; mkdir -p $o/pd
      TT_METAL_PROFILER_DIR=$o/pd timeout -s INT -k 60 900 python3 -m tracy -r -p --no-web-server -o $o $S/tools/prof_ops12.py $c > $o.log 2>&1
      rc=$?
      echo "== run $c $v r$r rc=$rc $(grep -c "CASE .* ok" $o.log) ok $(date -u +%T)"
      [[ $rc -ne 0 || $(grep -c "CASE .* ok" $o.log) -eq 0 ]] && grep -a "FAILED\|TT_THROW\|Error" $o.log | head -3 | cut -c1-200
      python3 $G/sum.py $o $c $v $r $R3_NMEAS
      rm -rf $o
    done
  done
done
use Z
echo "== trisc0 identity, kernels of Z against the others (same kernel dir in each cache)"
for other in P Zr Z2; do
  (cd $G/cache_Z && find . -path '*trisc0*' -name '*.elf' | sort) | while read f; do
    [[ -f $G/cache_$other/$f ]] && echo "$G/cache_Z/$f $G/cache_$other/$f"; done > $G/pairs_$other.txt
  paste -d' ' <(awk '{print $1}' $G/pairs_$other.txt | awk -F/ '{for(i=1;i<=NF;i++) if($i=="kernels") print $(i+1)}') <(python3 $S/elfsec.py $G/pairs_$other.txt) | sort | uniq -c | sed "s/^/== kid $other /"
done
echo "== ol end $(date -u +%T)"
