#!/usr/bin/env bash
# Gemma-4-26B GPQA precision A/B.
#
# Single variable: GEMMA4_PRECISION_CONFIG. Everything else -- the 10 documents,
# prompts, generation kwargs (max_gen_toks=32768, greedy, seed 42), context, mesh,
# server flags -- is identical to the stage-11 run that scored 4/10 against an HF
# control's 10/10.
#
# usage: gemma_gpqa_ab.sh <variant-name> <precision-config-path|BASELINE>
set -uo pipefail
NAME=$1; CFG=$2
OUT=$HOME/_gemma_gpqa_ab/$NAME
mkdir -p "$OUT"
cd "$HOME/tt-metal"
source python_env/bin/activate
export PYTHONPATH="$HOME/vllm:$HOME/tt-metal:${PYTHONPATH:-}"
export VLLM_PLUGINS=tt,tt_model_registry
export HF_MODEL=google/gemma-4-26B-A4B-it
[ "$CFG" != "BASELINE" ] && export GEMMA4_PRECISION_CONFIG="$CFG"

echo "=== variant=$NAME  precision=${GEMMA4_PRECISION_CONFIG:-<stage-08 selection>}" | tee "$OUT/run.log"

python -m vllm.entrypoints.openai.api_server \
  --model google/gemma-4-26B-A4B-it --block_size 64 --max_num_seqs 1 --port 8000 \
  --max_model_len 262144 \
  --additional-config '{"tt":{"sample_on_device_mode":"all","trace_region_size":220000000,"fabric_config":"FABRIC_1D_RING"}}' \
  > "$OUT/server.log" 2>&1 &
SRV=$!
echo "server pid $SRV" | tee -a "$OUT/run.log"

# wait for readiness, bounded
for i in $(seq 1 240); do
  curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && break
  kill -0 $SRV 2>/dev/null || { echo "SERVER DIED during startup" | tee -a "$OUT/run.log"; tail -30 "$OUT/server.log" >> "$OUT/run.log"; exit 1; }
  sleep 15
done
curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null 2>&1 \
  || { echo "SERVER NOT READY after 60 min" | tee -a "$OUT/run.log"; kill $SRV; exit 1; }
echo "server ready at $(date -u +%FT%TZ)" | tee -a "$OUT/run.log"

# identical eval invocation to stage 11
"$HOME/tt-inference-server/.workflow_venvs/.venv_evals_common/bin/lm_eval" \
  --model local-chat-completions \
  --model_args "model=google/gemma-4-26B-A4B-it,base_url=http://127.0.0.1:8000/v1/chat/completions,tokenizer_backend=huggingface,num_concurrent=1,max_retries=2,timeout=14400" \
  --tasks gpqa_diamond_cot_zeroshot \
  --limit 0.05 --seed 42 --num_fewshot 0 \
  --gen_kwargs "max_gen_toks=32768,temperature=0.0,do_sample=False,seed=42" \
  --output_path "$OUT" --log_samples --show_config --confirm_run_unsafe_code \
  >> "$OUT/eval.log" 2>&1
echo "EVAL_EXIT=$?" | tee -a "$OUT/run.log"

kill $SRV 2>/dev/null; sleep 20; pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null
echo "=== result:" | tee -a "$OUT/run.log"
python3 - "$OUT" <<'PY' 2>&1 | tee -a "$OUT/run.log"
import glob, json, sys
fs = glob.glob(sys.argv[1] + "/**/results_*.json", recursive=True)
if not fs:
    print("  no results file written"); raise SystemExit
r = json.load(open(sorted(fs)[-1]))["results"]["gpqa_diamond_cot_zeroshot"]
for k, v in sorted(r.items()):
    if isinstance(v, float): print(f"  {k:36s} {v}")
PY
