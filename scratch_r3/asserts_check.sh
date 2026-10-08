#!/usr/bin/env bash
# Which LLK assert stops these ops with TT_METAL_LLK_ASSERTS=1: each test alone under the watcher, its log printed after.
export HWLOCK_HELD=ci TT_METAL_LLK_ASSERTS=1 TT_METAL_WATCHER=1
R=$(git rev-parse --show-toplevel); S=$R/scratch_r3; cd $R
export PYTHONPATH="$S/tools:$R${PYTHONPATH:+:$PYTHONPATH}"
echo "== commit $(git log --oneline -1)"
for t in "test_sync_prof.py::test_sync_sdpa_decode[sdpa_dec_b1_p1279_full]" "test_sync_prof.py::test_sync_sdpa_decode[sdpa_dec_b1_p1279_half]" \
         "test_v12_prof.py::test_v12_gdn" "test_v12_prof.py::test_v12_mm[kl_qwen36_gdn_out_decode]" "test_v12_prof.py::test_v12_mm[kl_llama8b_ff2_p100a_128]" \
         "test_v12_prof.py::test_v12_mm[es_mlp_w1_128]" "test_jsdpa_prof.py::test_jsdpa_prof[flux1_p300_sp1]"; do
  rm -rf $R/generated/watcher
  echo "== $t"
  timeout -s INT -k 30 240 python -m pytest -p no:cacheprovider -o timeout_method=thread --timeout 150 -q -rfE "$S/tools/$t" > /tmp/t.log 2>&1
  echo "== rc=$? $(grep -E 'passed|failed' /tmp/t.log | tail -1)"
  grep -iE "assert|tripped|LLK" /tmp/t.log | grep -v "^\s*assert \|AssertionError" | head -8
  [[ -f $R/generated/watcher/watcher.log ]] && grep -iE "assert|tripped|error|hang" $R/generated/watcher/watcher.log | head -12
done
