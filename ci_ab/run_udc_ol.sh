#!/usr/bin/env bash
# #58819 regression audit, op level (17_CI_AB.md): the head's device files (head), the head with the tilize block body
# recorded at the first block call (hf1), and an A/A copy of the head (head2), swapped into /work and the installed runtime
# root before each run, each with its own fresh kernel cache, rotated per round, one case per tracy process.
# Prints OLR <case> <variant> <round> <op> <calls> <median of the last N> lines and the kernels' trisc0 identity.
# usage: bash ci_ab/run_udc_ol.sh
set +e +o pipefail
cd /work
S=/work/ci_ab; G=/tmp/udcol; rm -rf $G; mkdir -p $G/out
export TT_METAL_HOME=/work TRACY_NO_WEB_SERVER=1 HWLOCK_HELD=ci R3_NMEAS=${R3_NMEAS:-12} R3_L1_SMALL=16384
export PYTHONPATH=/work:/work/tools:$S/tools
RT=$(cd /tmp && env -u PYTHONPATH python3 -c "import ttnn, os; print(os.path.dirname(ttnn.__file__))" 2>/dev/null)
echo "== ol start $(date -u +%T) $(git -C /work rev-parse HEAD 2>/dev/null) RT=$RT"
FILES=$(cd $S/olvar/head && find . -type f | sed 's#^\./##')
for f in $FILES; do echo "== file $f work=$( [ -f /work/$f ] && echo y) rt=$( [ -f $RT/$f ] && echo y || echo n) $(for v in $(ls $S/olvar); do [ -f $S/olvar/$v/$f ] && echo -n "$v:$(md5sum $S/olvar/$v/$f | cut -c1-10) "; done)"; done
CUR=""
use() {  # use <variant>: its files into /work and the runtime root, its own kernel cache
  local v=$1 src=$1; [[ $v == head2 ]] && src=head
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
CASES=${OL_CASES:-"transpose_rm_1024x1024 transpose_rm_8x256x128 permute_rm_nchw_nhwc_64x128x128 conv2d_bf16_64x128_32x32 conv2d_fp32acc_64x128_32x32 tilize_int32_32x128256"}
ROUNDS=${OL_ROUNDS:-4}
read -ra VL <<< "${OL_VARS:-head hf1 head2}"
for r in $(seq 1 $ROUNDS); do
  k=$(( (r - 1) % 3 )); VS=("${VL[@]:$k}" "${VL[@]:0:$k}")
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
use head
echo "== trisc0 identity, kernels of head against hf1 and head2 (same kernel dir in each cache)"
for other in "${VL[@]:1}"; do
  (cd $G/cache_head && find . -path '*trisc0*' -name '*.elf' | sort) | while read f; do
    [[ -f $G/cache_$other/$f ]] && echo "$G/cache_head/$f $G/cache_$other/$f"; done > $G/pairs_$other.txt
  paste -d' ' <(awk '{print $1}' $G/pairs_$other.txt | awk -F/ '{for(i=1;i<=NF;i++) if($i=="kernels") print $(i+1)}') <(python3 $S/elfsec.py $G/pairs_$other.txt) | sort | uniq -c | sed "s/^/== kid $other /"
done
echo "== ol end $(date -u +%T)"
