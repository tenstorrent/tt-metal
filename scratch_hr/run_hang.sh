#!/usr/bin/env bash
# #59149 headroom round, scratch: the DRAM group norm test that stopped the bits job, on the head's and on this head's
# device code (LLK asserts on), then the rest of the bits ids on this head. Variant files as in run_hr_op.sh.
set -uo pipefail
export TT_METAL_LLK_ASSERTS=1
R=$(git rev-parse --show-toplevel); S=$R/scratch_hr; cd $R
SP=$(python -c "import ttnn, os; print(os.path.dirname(os.path.dirname(os.path.abspath(ttnn.__file__))))")
echo "== checkout $R head $(git rev-parse HEAD) $(hostname)"
eval "$(sed -n '/^place() {/,/^}/p' $S/run_hr_op.sh)"
G='tests/ttnn/unit_tests/operations/fused/test_group_norm.py::test_group_norm_dram_grid_size[specify_grid=True-N=1-C=480-H=8-W=8-num_groups=16]'
run() {
  local tag=$1 v=$2 lim=$3; shift 3
  place $S/variants/$v; export TT_METAL_CACHE=$HOME/hangcache_$v; mkdir -p $TT_METAL_CACHE
  echo "== $tag $v start $(date -u +%T)"
  timeout -s INT -k 60 $lim python -m pytest -q -p no:cacheprovider -o timeout_method=thread --timeout=420 -rfE "$@" > /tmp/hang_$tag.log 2>&1
  echo "== $tag $v rc=$? end $(date -u +%T): $(grep -aE '[0-9]+ (passed|failed)' /tmp/hang_$tag.log | tail -1)"
  grep -aE "^(FAILED|ERROR)|Timeout|LLK_ASSERT|assert" /tmp/hang_$tag.log | head -20
}
run g_head head 900 "$G"
run g_fp fp 900 "$G"
mapfile -t REST < $S/ids_rest2.txt
run rest_fp fp 3000 "${REST[@]}" $S/tools/test_hr_fast_vs_pack.py $S/tools/test_hr_chunk_callers.py
place $S/variants/head
exit 0
