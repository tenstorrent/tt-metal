#!/usr/bin/env bash
# Scratch A/B job, not for merge. Arms (device files ci_ab/v/<M|B>/ per ci_ab/files.txt, copied into /work and the installed
# package before every run, own JIT cache per arm): M main 129d0c50379; H the branch's files with EB_NOGATE and EB_HEADRULE
# (the head's hand-off and block rules); W the branch (gate and widened block rule); W2 A/A copy of W. Runs M H W W2 | W2 W H M
# under the device profiler (R3_REPS calls per test, the first a warm-up), per-op sums (sum_op19.py); output sha1 per test and
# arm (EB_BITS_OUT); kernel ELF hashes per arm (elfhash.py) for binary_ng, layernorm and softmax.
set +e +o pipefail
cd /work; export TT_METAL_HOME=/work TRACY_NO_WEB_SERVER=1 R3_REPS=${R3_REPS:-4} AB_NODES_FILE=/work/ci_ab/nodes.txt
S=/work/ci_ab; OUT=/work/generated/bng; rm -rf $OUT; mkdir -p $OUT
python3 -c "import tracy" 2>/dev/null || export PYTHONPATH=/work:/work/tools
export PYTHONPATH=$S:${PYTHONPATH:-}
SP=$(cd /tmp && env -u PYTHONPATH python3 -c "import ttnn, os; print(os.path.dirname(os.path.dirname(os.path.abspath(ttnn.__file__))))")
echo "== $(date -u +%FT%TZ) $(git log -1 --format='%h %s' | cut -c1-80) package root $SP"
suffix() { echo "$1" | awk -F/ '{n=NF; print $(n-3)"/"$(n-2)"/"$(n-1)"/"$n}'; }
declare -A PK
while read -r rel; do PK[$rel]=$(find "$SP" -path "*/$(suffix $rel)" -type f 2>/dev/null); done < $S/files_dev.txt
echo "== package copies found: $(for r in "${!PK[@]}"; do echo ${PK[$r]} | wc -w; done | awk '{s+=$1} END {print s}') for $(wc -l < $S/files_dev.txt) files"
use() {  # use <arm> <files dir>
  while read -r rel; do cp "$S/v/$2/$rel" "/work/$rel"; for p in ${PK[$rel]}; do cp "$S/v/$2/$rel" "$p"; done; done < $S/files_dev.txt
  if [[ $1 == H ]]; then export EB_NOGATE=1 EB_HEADRULE=1; else unset EB_NOGATE EB_HEADRULE; fi
  export TT_METAL_CACHE=$OUT/cache_$1 EB_ARM=$1 EB_BITS_OUT=$OUT/bits.txt; mkdir -p $TT_METAL_CACHE
}
r=0
for a in M H W W2 W2 W H M; do
  r=$((r+1)); case $a in M) use M M ;; *) use $a B ;; esac
  o=$OUT/${a}_r$r; mkdir -p $o; t0=$(date +%s)
  timeout -s INT -k 60 2400 python3 -m tracy -r -p --no-web-server -o $o $S/run_nodes.py > $o.log 2>&1
  rc=$?
  echo "== run $r $a rc=$rc $(( $(date +%s) - t0 )) s: $(grep -aE '[0-9]+ (passed|failed)' $o.log | tail -1)"
  [[ $rc -ne 0 ]] && grep -a -m5 -E '^(FAILED|ERROR)' $o.log | cut -c1-250
  python3 $S/sum_op19.py $o $a $r
done
for a in M H W W2; do python3 $S/elfhash.py $a $OUT/cache_$a eltwise_binary_no_bcast layernorm softmax; done
sort -u $OUT/bits.txt | sed 's/^/BITS /'
use M M
echo "== done $(date -u +%FT%TZ)"
