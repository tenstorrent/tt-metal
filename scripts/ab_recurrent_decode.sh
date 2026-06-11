#!/bin/bash
# A/B: tt-lang recurrent decode kernel (v2/v3) vs current pure-TTNN default.
# Full 64L model, ISL=128, 32 decode steps. Captures TRACED tok/s + token ids.
set -u
cd /home/tt-admin/ssinghal/qwen36/new/tt-metal
export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd)
source python_env/bin/activate
export HF_MODEL=Qwen/Qwen3.6-27B MESH_DEVICE=BH_GLX \
       QWEN36_FULLATTN_WO_TUNED=1 QWEN36_DELTA_OP_TUNED=1 \
       QWEN36_CCL_NUM_LINKS_DELTA=2 QWEN36_SEQ_CORES_PER_HEAD=4 \
       QWEN36_DECODE_STEPS=32 QWEN36_SAMPLE=0
TEST=models/demos/qwen3_6_galaxy_v2/tests/test_decode_perf_intrace.py::test_qwen36_64L_decode_intrace_perf
OUT=/home/tt-admin/ssinghal/qwen36/new/tt-metal/ab_recurrent_results
mkdir -p $OUT

run_variant () {
  local name="$1"; shift
  echo "============================================================"
  echo "=== VARIANT: $name   ($*)"
  echo "============================================================"
  env "$@" python3 -m pytest --noconftest -q -s "$TEST" > "$OUT/$name.log" 2>&1
  local rc=$?
  echo "[$name] exit=$rc"
  grep -E "TRACED|compile pass argmax|GENERATED \(|first decode token|target NOT met|PASSED" "$OUT/$name.log" | sed "s/^/[$name] /"
}

run_variant default
run_variant tt_lang_v2 QWEN36_TT_LANG_RECURRENT_V2=1
run_variant tt_lang_v3 QWEN36_TT_LANG_RECURRENT_V3=1

echo "==== A/B SUMMARY ===="
for v in default tt_lang_v2 tt_lang_v3; do
  echo "--- $v ---"
  grep -E "tok/s/user TRACED|compile pass argmax" "$OUT/$v.log" | sed "s/^/  /"
done
