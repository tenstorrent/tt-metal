#!/usr/bin/env bash
# #58816 regression audit on one CI Blackhole runner (knowledge/process/17_CI_AB.md): matmuls that take the block pack
# through pack_block at SyncHalf with a math-bound loop, at Blackhole model configurations, device time under tracy.
# Variants: M (main f0832760bf5's 27 device files that #58816 changes), H (the head 96dde2a4fcb's), H2 (H's files again
# under its own cache: the A/A pair). Before each run the variant's files go into the checkout and the installed ttnn
# package (the JIT reads headers and kernels from both); each variant has its own fresh JIT cache.
# Items: i0b tt_transformers test_model quick/accuracy (Llama 3.1 8B decode, one layer, random weights, 6 iterations,
# the accuracy precision: DRAM-sharded QKV, WO, FF1, FF3, FF2 and the LM head), i1 Llama 3.1 8B prefill QKV and WO at
# 128 tokens. Order per item M H H2 H2 H M, two blocks.
set +e +o pipefail
R=$(git rev-parse --show-toplevel); S=$R/ci_ab; cd $R
export TT_METAL_HOME=$R TRACY_NO_WEB_SERVER=1 PYTHONPATH=$R:$S HF_MODEL=meta-llama/Llama-3.1-8B-Instruct HF_HUB_OFFLINE=1 \
  TT_CACHE_PATH=$HOME/ra_ttc TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=10000
RT=$(cd /tmp && env -u PYTHONPATH python3 -c "import ttnn, os; print(os.path.dirname(ttnn.__file__))" 2>/dev/null)
echo "== checkout $R, $(git log --oneline -1), runtime $RT, $(date -u +%T)"
T0=$(date +%s); O=$HOME/ra_out; rm -rf $O $HOME/ra_cache_* $TT_CACHE_PATH; mkdir -p $O
: > $O/dest.txt
while read -r rel; do
  suf=$(echo "$rel" | awk -F/ '{n=NF; print $(n-3)"/"$(n-2)"/"$(n-1)"/"$n}')
  for p in $(find "$RT" -path "*/$suf" -type f 2>/dev/null); do echo "$rel $p" >> $O/dest.txt; done
done < $S/files.txt
echo "== $(wc -l < $O/dest.txt) package copies of $(wc -l < $S/files.txt) files"
CUR=""
use() {
  local v=$1 f=${1%2}
  if [[ $CUR != $f ]]; then
    while read -r rel; do cp "$S/v/$f/$rel" "$R/$rel"; done < $S/files.txt
    while read -r rel p; do cp "$S/v/$f/$rel" "$p"; done < $O/dest.txt
    CUR=$f
  fi
  export TT_METAL_CACHE=$HOME/ra_cache_$v; mkdir -p $TT_METAL_CACHE
}
ITEMS=(
  'i0b|["models/tt_transformers/tests/test_model.py", "-k", "quick and accuracy"]'
  'i1|["ci_ab/test_ra_mm.py::test_ra_llama8b_prefill128_attn"]'
)
declare -A SEEN
for blk in 1 2; do
  for it in "${ITEMS[@]}"; do
    tag=${it%%|*}; args=${it#*|}
    for v in M H H2 H2 H M; do
      el=$(( ($(date +%s) - T0) / 60 ))
      if (( el > 100 )); then echo "== SKIP $tag $v blk $blk at $el min"; continue; fi
      use $v; r=$blk$([[ -n ${SEEN[$tag$v$blk]} ]] && echo b || echo a); SEEN[$tag$v$blk]=1
      o=$O/${tag}__${v}__$r; mkdir -p $o/pd
      RA_ARGS="$args" TT_METAL_PROFILER_DIR=$o/pd timeout -s INT -k 60 1500 python -m tracy -r -p --no-web-server -o $o $S/tools/run1.py > $o.log 2>&1
      echo "== run $tag $v $r rc=$? $(date -u +%T) $(grep -aE '[0-9]+ (passed|failed|skipped)' $o.log | tail -1)"
      [[ $blk == 1 && $v == M && $r == 1a ]] && grep -a -m4 -E "FAILED|Error|error:" $o.log | cut -c1-240
      python3 $S/tools/sum_ops.py $tag $v $r $o
      if [[ -z ${SEEN[elf$v]} ]]; then SEEN[elf$v]=1; python3 $S/tools/elfsum.py $v $TT_METAL_CACHE; fi
      rm -rf $o
    done
  done
done
use H
echo "== DONE $(date -u +%T)"
