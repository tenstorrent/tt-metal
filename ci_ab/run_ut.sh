#!/usr/bin/env bash
# Scratch, not for merge. ttnn topk, moe, sort, sampling, gate and router tests on the checked-out tree: the commit's changed
# device files are copied into the installed package first.
set +e +o pipefail
cd /work; export TT_METAL_HOME=/work
RT=$(cd /tmp && env -u PYTHONPATH python3 -c "import ttnn, os; print(os.path.dirname(os.path.abspath(ttnn.__file__)))")
n=0; for rel in $(cat /work/ci_ab/files.txt); do
  if [[ -f $RT/$rel ]]; then cp /work/$rel $RT/$rel; n=$((n+1)); else echo "OPS NOPKG $rel"; fi; done
echo "OPS copied $n files to $RT; head $(git log -1 --format=%h)"
U=tests/ttnn/unit_tests/operations; N=tests/ttnn/nightly/unit_tests/operations
run() { name=$1; shift; a=(); for x in "$@"; do [[ -f $x ]] && a+=("$x"); done; set -- "${a[@]}"; [[ $# -eq 0 ]] && { echo "OPS UT $name skipped (no files)"; return; }; timeout -s INT -k 60 2700 python3 -m pytest -p no:cacheprovider -q -rfE "$@" > /tmp/ut_$name.log 2>&1
  echo "OPS UT $name rc=$? $(grep -aE '[0-9]+ (passed|failed)' /tmp/ut_$name.log | tail -1) timedout=$(grep -a -c 'TENSIX TIMED OUT' /tmp/ut_$name.log)"
  grep -a -E '^(FAILED|ERROR) ' /tmp/ut_$name.log | head -5 | cut -c1-260; }
run topk $U/reduce/test_topk.py
run moe $U/reduce/test_moe.py
run sampling $U/reduce/test_sampling.py
run sort $U/data_movement/test_sort.py
run gates $N/reduction/test_deepseek_grouped_gate.py $N/experimental/deepseek_prefill/test_moe_grouped_topk.py $N/experimental/deepseek_prefill/test_moe_grouped_topk_stable.py $N/experimental/deepseek_prefill/test_moe_grouped_topk_ties.py
run router $N/experimental/test_topk_router_gpt.py
echo "== done $(date -u +%FT%TZ)"
