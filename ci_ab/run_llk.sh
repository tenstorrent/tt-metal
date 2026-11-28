#!/usr/bin/env bash
# Scratch CI job, not for merge. bash ci_ab/run_llk.sh <census> <ids file>: a tt-llk harness census (ci_ab/<census>/: common/ overlay for every arm,
# H/ for head arms, M/ for main arms) per arm, device files swapped from ci_ab/v/<arm> (files.txt, .absent) into /work, own build dir per arm.
# Arms: ci_ab/arms plus M2 (A/A copy of M). CIAB_BLOCKS blocks, order A B .. M2 | M2 .. B A, L1_TO_L1, LLK_CODE_SHIFTS 0 4 20 36 52 in every run
# (remeasure_dc: overlay dc/<arm> when it exists, else H for head arms; rows tagged with their code position).
# Every TILE_LOOP and KERNEL row of the perf report goes to the log as "CIAB LR <arm> <block> <json>".
set +e +o pipefail
T0=$(date +%s); C=$1; IDS=$2; NB=${CIAB_BLOCKS:-5}
W=${CIAB_W:-/work}; cd $W; S=$W/ci_ab; OUT=$W/generated/ciab; rm -rf $OUT; mkdir -p $OUT
read -r -a AR < $S/arms; ARMS=("${AR[@]}" M2); NA=${#ARMS[@]}
echo "CIAB ARMS $(tr '\n' ' ' < $S/ARMS.txt 2>/dev/null)"; echo "CIAB HOST $(hostname) $(date -u +%FT%TZ) head $(git log -1 --format=%H) census $C ids $(grep -c . $S/$IDS) blocks $NB arms ${ARMS[*]}"
source $S/llk_env.sh
L=$W/tt_metal/tt-llk; PT=$L/tests/python_tests
use() {  # use <arm>: device files, then the census overlay (common, then H or M)
  while read -r rel; do
    if grep -qxF "$rel" $S/v/$1/.absent 2>/dev/null; then rm -f "$W/$rel"; else mkdir -p "$(dirname "$W/$rel")"; cp "$S/v/$1/$rel" "$W/$rel"; fi
  done < $S/files.txt
  cp -r $S/$C/common/. $W/; if [[ -d $S/$C/$1 ]]; then cp -r $S/$C/$1/. $W/; else cp -r $S/$C/H/. $W/; fi
}
var() { [[ $1 == M2 ]] && echo M || echo $1; }
mapfile -t I < $S/$IDS
unset RUNNER_TEMP  # the harness puts its builds under RUNNER_TEMP when set: one dir for every arm
export CHIP_ARCH=blackhole TT_LLK_DISABLE_ASSERTS=1 LLK_PERF_RUN_TYPES=L1_TO_L1 LLK_CODE_SHIFTS="${CIAB_SHIFTS:-0 4 20 36 52}" LLK_PERF_L1_MUX_GROUP=0 LLK_HOME=$L PERF_KEEP_RUNS=400
[[ -f $S/$C/env ]] && { set -a; source $S/$C/env; set +a; echo "CIAB ENV $(tr '\n' ' ' < $S/$C/env)"; }
rot=("${ARMS[@]}"); rev=(); for ((i = NA - 1; i >= 0; i--)); do rev+=("${rot[$i]}"); done; ord=()
for ((b = 0; b < NB; b++)); do if ((b % 2 == 0)); then ord+=("${rot[@]}"); else ord+=("${rev[@]}"); fi; done
for a in "${ARMS[@]}"; do use $(var $a); export TMPDIR=$OUT/build_$a; mkdir -p $TMPDIR; t1=$(date +%s)
  (cd $PT && $LLKPY -m pytest --compile-producer -n ${CIAB_NPROD:-8} -p no:cacheprovider -q "${I[@]}" > $OUT/prod_$a.log 2>&1); rc=$?
  echo "CIAB PRODUCE $a rc=$rc $(( $(date +%s) - t1 )) s $(grep -aE '[0-9]+ (passed|failed|error)' $OUT/prod_$a.log | tail -n 1)"; [[ $rc -ne 0 ]] && grep -a -m5 -E '^(FAILED|ERROR)|rror' $OUT/prod_$a.log | cut -c1-300
done
for ((r = 0; r < ${#ord[@]}; r++)); do a=${ord[$r]}; b=$((r / NA)); use $(var $a)
  export TMPDIR=$OUT/build_$a PERF_RUN_TAG=p0; rm -rf $L/perf_data/runs/p*; mkdir -p $TMPDIR; t1=$(date +%s)
  (cd $PT && timeout -s INT -k 30 ${CIAB_PT_TO:-3000} $LLKPY -m pytest --compile-consumer -p no:cacheprovider --override-ini=log_cli=false -q -o timeout_method=thread --timeout 300 "${I[@]}" > $OUT/r${r}_$a.log 2>&1); rc=$?
  echo "CIAB RUN $r block $b arm $a rc=$rc $(( $(date +%s) - t1 )) s $(grep -aE '[0-9]+ (passed|failed|error)' $OUT/r${r}_$a.log | tail -n 1)"
  [[ $rc -ne 0 ]] && { grep -a -m5 -E '^(FAILED|ERROR)|Error' $OUT/r${r}_$a.log | cut -c1-300; tail -n 4 $OUT/r${r}_$a.log | cut -c1-300; }
  $LLKPY - "$L/perf_data/runs" "$a" "$b" <<'EOF'
import sys, glob, json, os, pandas as pd
d, a, b = sys.argv[1:4]; n = 0
for f in sorted(glob.glob(d + "/p*/**/*.post.csv", recursive=True)):
    sh = os.path.relpath(f, d).split("/")[0][1:]
    x = pd.read_csv(f, low_memory=False)
    x = x[x.marker.isin(["TILE_LOOP", "KERNEL"])]
    for _, q in x.iterrows():
        print("CIAB LR", a, b, sh, json.dumps({k: (v if not hasattr(v, "item") else v.item()) for k, v in q.items() if str(v) not in ("nan", "")}, default=str)); n += 1
print("CIAB LRN", a, b, n, d)
EOF
done
OBJ=$L/tests/sfpi/compiler/bin/riscv-tt-elf-objcopy
echo "CIAB BUILDS $(for a in "${ARMS[@]}"; do echo -n "$a $(find $OUT/build_$a -name "*.elf" | wc -l) "; done)"
for x in "${AR[@]:1}" M2; do n=0; dd=0; for f in $(cd $OUT/build_M && find . -path '*/elf/*.elf' ! -path '*/shared/*'); do [[ -f $OUT/build_$x/$f ]] || continue; n=$((n+1))
  [[ "$($OBJ -O binary -j .text $OUT/build_M/$f /dev/stdout | md5sum)" == "$($OBJ -O binary -j .text $OUT/build_$x/$f /dev/stdout | md5sum)" ]] || dd=$((dd+1)); done
  echo "CIAB GUARD M$x pairs $n differ $dd"; done
echo "CIAB DONE $(date -u +%FT%TZ) $(( $(date +%s) - T0 )) s"
