#!/usr/bin/env bash
# Scratch, not for merge. The commit as built (host unchanged): P the PR head's device files b0c77f9a2fb, S the commit's, S2 A/A.
# Op level P S S2 P | P S2 S P on the selected scalar nodes (device profiler, sum_op19.py), output sha1 per arm, ELF hashes of
# the scalar kernel builds; then the tt-llk eltwise binary functional tests (block, fixed-B, smoke) with asserts on and every L1
# read hashed (topkbits).
set +e +o pipefail
cd /work; export TT_METAL_HOME=/work TRACY_NO_WEB_SERVER=1 R3_REPS=${R3_REPS:-4} AB_NODES_FILE=/work/ci_ab/nodes.txt
S=/work/ci_ab; OUT=/work/generated/c8; rm -rf $OUT; mkdir -p $OUT
python3 -c "import tracy" 2>/dev/null || export PYTHONPATH=/work:/work/tools
export PYTHONPATH=$S:${PYTHONPATH:-}
SP=$(cd /tmp && env -u PYTHONPATH python3 -c "import ttnn, os; print(os.path.dirname(os.path.dirname(os.path.abspath(ttnn.__file__))))")
echo "== $(git log -1 --format='%h %s' | cut -c1-80)"
suffix() { echo "$1" | awk -F/ '{n=NF; print $(n-3)"/"$(n-2)"/"$(n-1)"/"$n}'; }
declare -A PK
while read -r rel; do PK[$rel]=$(find "$SP" -path "*/$(suffix $rel)" -type f 2>/dev/null); done < $S/files_dev.txt
use() { local d=${1%2}
  while read -r rel; do cp "$S/v/$d/$rel" "/work/$rel"; for p in ${PK[$rel]}; do cp "$S/v/$d/$rel" "$p"; done; done < $S/files_dev.txt
  export TT_METAL_CACHE=$OUT/cache_$1 EB_ARM=$1 EB_BITS_OUT=$OUT/bits.txt; mkdir -p $TT_METAL_CACHE; }
r=0
for a in P S S2 P P S2 S P; do
  r=$((r+1)); use $a; o=$OUT/${a}_r$r; mkdir -p $o
  timeout -s INT -k 60 1800 python3 -m tracy -r -p --no-web-server -o $o $S/run_nodes.py > $o.log 2>&1
  echo "== run $r $a rc=$? $(grep -aE '[0-9]+ (passed|failed)' $o.log | tail -1)"
  python3 $S/sum_op19.py $o $a $r
done
for a in P S S2; do python3 $S/elfhash.py $a $OUT/cache_$a eltwise_binary_scalar; done
sort -u $OUT/bits.txt | sed 's/^/BITS /'
use S
L=/work/tt_metal/tt-llk
python3 -m venv /tmp/llkvenv && . /tmp/llkvenv/bin/activate
pip install -q -r $L/tests/requirements.txt > /tmp/pip.log 2>&1; (cd $L/tests && ./setup_testing_env.sh > /tmp/setup.log 2>&1); echo "== llk env pip/setup done"
cd $L/tests/python_tests; mapfile -t NODES < $S/bits_ids.txt
CHIP_ARCH=blackhole LLK_HOME=$L RUNNER_TEMP=/tmp/llkbits TMPDIR=/tmp/llkbits TOPKBITS_OUT=$OUT/llkbits.jsonl timeout 3000 python -m pytest -n 8 -p topkbits -p no:cacheprovider --override-ini=log_cli=false -o timeout_method=thread -q -rfE "${NODES[@]}" > $OUT/llkbits.log 2>&1
echo "== llk functional rc=$? $(grep -E 'passed|failed|error' $OUT/llkbits.log | tail -1 | cut -c1-150) $(wc -l < $OUT/llkbits.jsonl 2>/dev/null) reads"
grep -a -E "^(FAILED|ERROR) " $OUT/llkbits.log | head -5 | cut -c1-250
echo "== done"
