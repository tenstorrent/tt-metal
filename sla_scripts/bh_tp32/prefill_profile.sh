#!/bin/bash
# Device-profile one prefill chunk of a 1-layer model (tests/test_prefill_layer_prof.py) per variant and print the per-op table.
# Usage: prefill_profile.sh "tag:ENV=1 ENV2=x" ...   e.g. "base:" "seq2k:PREFILL_SEQ=2048" "lofi:LLAMA_BH_SDPA_FIDELITY=LoFi"
# Required env: TT_METAL_DIR, HF_MODEL, TT_CACHE_PATH, OUT. Source env.sh first for the base config; each variant adds its ENVs.
# Output: $OUT/prefill_sweep.txt (raw logs: $TT_METAL_DIR/generated/profiler/bh-prefill-layer-<tag>/.logs).
# The profiler's own post-processing fails on this run ("Device data missing"); analyze_prefill_raw.py reads the raw logs.
: "${TT_METAL_DIR:?}" "${HF_MODEL:?}" "${TT_CACHE_PATH:?}" "${OUT:?}"
D=$(cd "$(dirname "$0")" && pwd)
cd "$TT_METAL_DIR" && source python_env/bin/activate
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD MESH_DEVICE=${MESH_DEVICE:-TG}
mkdir -p "$OUT"
for v in "$@"; do
  tag=${v%%:*}; envs=${v#*:}
  timeout -s INT 900 env $envs TAG=$tag python "$D/profile_prefill.py" > "$OUT/prefill_$tag.log" 2>&1
  { echo "=== $tag ($envs)"; grep -E "PREFILL_LAYER_PROF|passed|failed" "$OUT/prefill_$tag.log" | tail -2
    python "$D/analyze_prefill_raw.py" "generated/profiler/bh-prefill-layer-$tag/.logs" 2>&1 | head -16; } >> "$OUT/prefill_sweep.txt"
done
cat "$OUT/prefill_sweep.txt"
