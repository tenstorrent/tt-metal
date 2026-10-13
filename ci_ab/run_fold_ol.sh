#!/usr/bin/env bash
# #58819 datacopy fold, op level (17_CI_AB.md; scratch, not for merge): the device files of H (head), F (fold), M (main
# 7c7fafc9758's versions of every device file the stack changes) and F2 (an A/A copy of F) swapped into /work and the
# installed runtime root before each run, each with its own fresh kernel cache, rotated per round, one case per tracy process.
# Prints OLR <case> <variant> <round> <op> <calls> <median of the last N> <values> lines and the kernels' trisc identity.
set +e +o pipefail
cd /work
S=/work/ci_ab; G=/tmp/foldol; rm -rf $G; mkdir -p $G/out
export TT_METAL_HOME=/work TRACY_NO_WEB_SERVER=1 HWLOCK_HELD=ci R3_NMEAS=${R3_NMEAS:-12}
export PYTHONPATH=/work:/work/tools:$S/tools
RT=$(cd /tmp && env -u PYTHONPATH python3 -c "import ttnn, os; print(os.path.dirname(ttnn.__file__))" 2>/dev/null)
echo "== ol start $(date -u +%T) $(git -C /work rev-parse HEAD 2>/dev/null) RT=$RT"
FILES=$(cd $S/olvar/F && find . -type f | sed 's#^\./##')
nrt=0; for f in $FILES; do [ -f $RT/$f ] && nrt=$((nrt + 1)); done
echo "== $(echo $FILES | wc -w) variant files, $nrt also in the runtime root"
DC=tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_datacopy.h
for v in H F M; do echo "== olvar $v datacopy $(md5sum $S/olvar/$v/$DC | cut -c1-10)"; done
use() {  # use <variant>: its files into /work and the runtime root, its own kernel cache
  local v=$1 src=$1; [[ $v == F2 ]] && src=F
  for f in $FILES; do cp $S/olvar/$src/$f /work/$f; [ -f $RT/$f ] && cp $S/olvar/$src/$f $RT/$f; done
  export TT_METAL_CACHE=$G/cache_$v; mkdir -p $TT_METAL_CACHE
}
cat > $G/sum.py <<'PYEOF'
import csv, glob, statistics, collections, sys
d, case, v, r, n = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
fs = glob.glob(d + '/reports/*/ops_perf_results_*.csv') + glob.glob(d + '/ops_perf_results_*.csv')
if not fs: print('OLR NOREPORT', case, v, r); sys.exit(0)
rows = list(csv.DictReader(open(fs[0])))
idx = max([i for i, x in enumerate(rows) if x.get('OP TYPE') == 'signpost'] or [-1])
agg = collections.defaultdict(list)
for row in rows[idx + 1:]:
    if row.get('OP TYPE') == 'signpost': continue
    try: val = float(row['DEVICE KERNEL DURATION [ns]'])
    except (KeyError, ValueError): continue
    agg[row['OP CODE'].replace(' ', '_')].append(val)
for op, vals in sorted(agg.items()):
    cut = vals[-n:]
    print('OLR', case, v, r, op, len(vals), round(statistics.median(cut)), ' '.join(str(round(x)) for x in cut))
PYEOF
CASES=${OL_CASES:-"tc_bf16_u16_hs tc_bf16_i32_hs tc_bf16_fp32_hs tc_bf16_u32_hs tc_i32_bf16_hs tc_bf16_fp32_dram_32x128256 exp_dram_1024x1024 exp_hs recip_dram_1024x1024 recip_hs gelu_dram_1024x4096 gelu_hs softmax_bf8_1x1x16384x256 softmax_bf8_1x8x256x256 quant_bf16_1024x1024 quant_fp32_1024x1024"}
ROUNDS=${OL_ROUNDS:-4}
read -ra VL <<< "${OL_VARS:-H F M F2}"
for r in $(seq 1 $ROUNDS); do
  k=$(( (r - 1) % ${#VL[@]} )); VS=("${VL[@]:$k}" "${VL[@]:0:$k}")
  (( r % 2 == 0 )) && VS=($(printf '%s\n' "${VS[@]}" | tac))
  for c in $CASES; do
    for v in "${VS[@]}"; do
      use $v; o=$G/out/${c}__${v}__r$r; mkdir -p $o/pd
      TT_METAL_PROFILER_DIR=$o/pd timeout -s INT -k 60 900 python3 -m tracy -r -p --no-web-server -o $o $S/tools/prof_fold.py $c > $o.log 2>&1
      rc=$?
      echo "== run $c $v r$r rc=$rc $(grep -c "CASE .* ok" $o.log) ok $(date -u +%T)"
      [[ $rc -ne 0 || $(grep -c "CASE .* ok" $o.log) -eq 0 ]] && grep -a "FAILED\|TT_THROW\|Error" $o.log | head -3 | cut -c1-300
      python3 $G/sum.py $o $c $v $r $R3_NMEAS
      rm -rf $o
    done
  done
done
use H
echo "== trisc identity, kernels of F against the others (same kernel dir in each cache)"
for other in H M F2; do
  for t in trisc0 trisc1 trisc2; do
    (cd $G/cache_F && find . -path "*$t*" -name '*.elf' | sort) | while read f; do
      [[ -f $G/cache_$other/$f ]] && echo "$G/cache_F/$f $G/cache_$other/$f"; done > $G/pairs_${other}_$t.txt
    paste -d' ' <(awk '{print $1}' $G/pairs_${other}_$t.txt | awk -F/ '{for(i=1;i<=NF;i++) if($i=="kernels") print $(i+1)}') <(python3 $S/tools/elfsec.py $G/pairs_${other}_$t.txt) | sort | uniq -c | sed "s/^/== kid $other $t /"
  done
done
echo "== ol end $(date -u +%T)"
