#!/usr/bin/env bash
# Scratch job (MoE gate route), not for merge: model_gate.py on M (main's gate device files) and N (#60154's), each its own
# fresh JIT cache, then model_gate_cmp.py.
set +e +o pipefail
cd /work; export TT_METAL_HOME=/work HWLOCK_HELD=ci
S=/work/ci_ab; OUT=/work/generated/mgate; rm -rf $OUT; mkdir -p $OUT
SP=$(cd /tmp && python3 -c "import ttnn, os; print(os.path.dirname(os.path.dirname(os.path.abspath(ttnn.__file__))))")
suffix() { echo "$1" | awk -F/ '{n=NF; print $(n-3)"/"$(n-2)"/"$(n-1)"/"$n}'; }
use() {
  while read -r rel; do
    cp "$S/variants3/$1/$rel" "/work/$rel"
    while read -r p; do [[ -n $p && $p != "/work/$rel" ]] && cp "$S/variants3/$1/$rel" "$p"; done < <(find "$SP" -path "*/$(suffix $rel)" -type f 2>/dev/null)
  done < $S/files3.txt
  export TT_METAL_CACHE=$OUT/cache_$1; mkdir -p $TT_METAL_CACHE
}
export GATE_CFGS="${GATE_CFGS:-glm5 glm_47 qwen35_35b qwen3_235b qwen3_omni_thinker qwen3_omni_talker gemma_4_26b gpt_oss mistral_large_3 deepseek_ocr deepseek_v4_flash deepseek_v3}"
for a in M N; do
  use $a
  timeout -s INT -k 60 3000 python3 $S/model_gate.py $OUT/$a.pt > $OUT/$a.log 2>&1
  echo "== $a rc=$? $(grep -a 'GATE DONE' $OUT/$a.log)"
  grep -a -m5 -E "Error|FATAL|Traceback" $OUT/$a.log | cut -c1-240
done
cd /work && PYTHONPATH=/work python3 $S/model_gate_cmp.py $OUT/M.pt $OUT/N.pt 2>&1 | grep -a -E "GATECMP|GATESTAT|Error|Traceback" | head -80
echo "== done"
