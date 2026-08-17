#!/usr/bin/env bash
# Run the task the RELEASE will actually use: r1_gpqa_diamond, not
# gpqa_diamond_cot_zeroshot.
#
# Established from tt-inference-server branch vvukoman/add-8-models-to-release-flow:
# all four newly added models use task_name="r1_gpqa_diamond" with an explicit
# gen_kwargs budget (80*1024 for the two Qwen entries, 24*1024 and 32*1024 for the
# others) and model-card sampling. Qwen3.6-27B's existing entry still uses the older
# meta_gpqa_cot -> gpqa_diamond_cot_zeroshot pattern with NO budget.
#
# Why the task choice matters, from lm_eval/tasks/r1_evals/gpqa_reasoning_diamond.yaml
# versus lm_eval/tasks/gpqa/cot_zeroshot/_gpqa_cot_zeroshot_yaml:
#
#                     cot_zeroshot (what ran)        r1_gpqa_diamond (release)
#   max_gen_toks      none -> API default 256        32768 built in
#   until             ["</s>"]  (wrong for Qwen)     <|im_end|>, <|endoftext|>, <|end_of_text|>
#   sampling          greedy, temperature 0          temperature 0.6, top_k 40, top_p 0.95
#   extraction        strict-match + flexible-extract  own process_results_gpqa
#   graded key        exact_match,flexible-extract   exact_match,none
#
# usage: run_r1_gpqa.sh [limit] [max_gen_toks]
#   limit         lm-eval --limit (default 0.05, the CI_NIGHTLY value in the spec)
#   max_gen_toks  default 32768, the task YAML's own budget. The release spec
#                 overrides this to 80*1024 = 81920 for the Qwen family; at the
#                 measured 56 ms/token that is up to ~76 min per document, so the
#                 YAML budget is the affordable starting point here.
set -uo pipefail
LIMIT=${1:-0.05}
BUDGET=${2:-32768}
OUT=$HOME/_qwen_r1_gpqa/limit${LIMIT}_gen${BUDGET}
mkdir -p "$OUT"
cd "$HOME/tt-metal"
source python_env/bin/activate
export PYTHONPATH="$HOME/vllm:$HOME/tt-metal:${PYTHONPATH:-}"
export VLLM_PLUGINS=tt,tt_model_registry

echo "=== r1_gpqa_diamond  limit=$LIMIT max_gen_toks=$BUDGET  $(date -u +%FT%TZ)" | tee "$OUT/run.log"
echo "=== branch: $(git rev-parse --abbrev-ref HEAD)" | tee -a "$OUT/run.log"

# Release-spec server flags, with the two device settings kept at the autoport's
# validated values (release says FABRIC_1D and 1 GB trace; changing those is a
# separate experiment so as not to confound this one).
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3.6-27B \
  --block_size 64 \
  --max_model_len 262144 \
  --max_num_seqs 1 \
  --max_num_batched_tokens 262144 \
  --max-log-len 32 \
  --seed 9472 \
  --port 8000 \
  --reasoning_parser qwen3 \
  --additional-config '{"tt":{"sample_on_device_mode":"all","trace_region_size":200000000,"fabric_config":"FABRIC_1D_RING"}}' \
  > "$OUT/server.log" 2>&1 &
SRV=$!
echo "server pid $SRV" | tee -a "$OUT/run.log"
for i in $(seq 1 160); do
  curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && break
  kill -0 $SRV 2>/dev/null || { echo "SERVER DIED during startup" | tee -a "$OUT/run.log"
    grep -iE "TT_THROW|RuntimeError|error|unrecognized" "$OUT/server.log" \
      | grep -viE "leaked|nanobind" | tail -6 >> "$OUT/run.log"; exit 1; }
  sleep 15
done
curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null 2>&1 \
  || { echo "SERVER NOT READY" | tee -a "$OUT/run.log"; kill $SRV; exit 1; }
echo "server ready $(date -u +%FT%TZ)" | tee -a "$OUT/run.log"
grep -oE "reasoning_parser='[^']*'" "$OUT/server.log" | head -1 | tee -a "$OUT/run.log"

# gen_kwargs mirror the release spec's block for the sibling Qwen entry:
#   stream=false (REQUIRED -- lm-eval's streaming parser raises KeyError 'message'),
#   do_sample=true, temperature 1.0, top_k 20, top_p 0.95 (Qwen card, thinking mode).
"$HOME/tt-inference-server/.workflow_venvs/.venv_evals_common/bin/lm_eval" \
  --model local-chat-completions \
  --model_args "model=Qwen/Qwen3.6-27B,base_url=http://127.0.0.1:8000/v1/chat/completions,tokenizer_backend=huggingface,num_concurrent=1,max_length=262144,trust_remote_code=True,max_retries=2,timeout=14400" \
  --tasks r1_gpqa_diamond --limit "$LIMIT" --seed 9472 --num_fewshot 0 --apply_chat_template \
  --gen_kwargs "max_gen_toks=${BUDGET},stream=false,do_sample=true,temperature=1.0,top_k=20,top_p=0.95" \
  --output_path "$OUT" --log_samples --show_config --confirm_run_unsafe_code \
  >> "$OUT/eval.log" 2>&1
echo "EVAL_EXIT=$?" | tee -a "$OUT/run.log"

kill $SRV 2>/dev/null; sleep 25
pkill -f "entrypoints.openai.api_server" 2>/dev/null

echo "=== result:" | tee -a "$OUT/run.log"
python3 /tmp/summarize_eval.py "$OUT" 2>&1 | tee -a "$OUT/run.log"
echo "=== r1_gpqa done $(date -u +%FT%TZ)" | tee -a "$OUT/run.log"
