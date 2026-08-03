#!/bin/bash
# Stage E — SWE-bench Verified (4) + Terminal-Bench 2 (2) against the up tool-calling server.
# Usage: stage_e_benches.sh <WORKERS>   (WORKERS from Stage C: 4 if concurrent-clean, else 1)
# Small resumable guard: if preds.json already has 4 non-empty patches, skip SWE generation (re-grade only).
set +e
WORKERS=${1:-1}
VENV=/home/ttuser/.claude/jobs/b4840290/tmp/venv_agentic
TOOLCFG=/home/ttuser/.claude/jobs/b4840290/tmp/mini_model_config_toolcall.yaml
BASE=/home/ttuser/dev/tt-metal/models/autoports/poolside_laguna_xs_2_1
OUT=$BASE/doc/vllm_integration/stage_ce
SWE=$OUT/swe4
LOG=$OUT/benches.log        # <-- TAIL THIS
mkdir -p "$SWE"
: > "$LOG"
log(){ echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$LOG"; }
log "=== Stage E benches: SWE 4 + TB 2, WORKERS=$WORKERS ==="

# ---- SWE-bench Verified, slice 0:4 ----
done_ne=$(python3 -c 'import json,sys,os
p="'"$SWE"'/preds.json"
print(sum(1 for v in json.load(open(p)).values() if (v.get("model_patch") or "").strip()) if os.path.exists(p) else 0)' 2>/dev/null)
if [ "${done_ne:-0}" -ge 4 ]; then
  log "SWE: preds.json already has $done_ne non-empty patches — skipping generation (re-grade only)"
else
  log ">>> SWE generation (slice 0:4, workers=$WORKERS, toolcall step_limit 120)"
  "$VENV/bin/mini-extra" swebench --subset verified --split test --slice 0:4 --workers "$WORKERS" \
    --config swebench_backticks.yaml --config "$TOOLCFG" --output "$SWE" 2>&1 | tee -a "$LOG"
fi
log ">>> SWE patch check"
python3 -c '
import json;d=json.load(open("'"$SWE"'/preds.json"))
ne=sum(1 for v in d.values() if (v.get("model_patch") or "").strip())
print(f"non-empty patches: {ne}/{len(d)}")
for k,v in d.items(): print(" ",k,"->","PATCH" if (v.get("model_patch") or "").strip() else "EMPTY")' 2>&1 | tee -a "$LOG"
log ">>> SWE grade (resolved-rate)"
python3 -c '
import json;d=json.load(open("'"$SWE"'/preds.json"))
open("'"$SWE"'/preds.jsonl","w").write("\n".join(json.dumps(v) for v in d.values()))' 2>&1 | tee -a "$LOG"
"$VENV/bin/python" -m swebench.harness.run_evaluation \
  --dataset_name princeton-nlp/SWE-bench_Verified --split test \
  --predictions_path "$SWE/preds.jsonl" --run_id stageE --max_workers 4 2>&1 | tee -a "$LOG"

# ---- Terminal-Bench 2.0, 2 tasks ----
# De-risked command (Stage C recon): terminus-2 uses TEXT parsing (no native tool-calling needed). Bare
# --agent-kwarg top_k/chat_template_kwargs are SILENTLY DROPPED by BaseAgent — must go through
# llm_call_kwargs.extra_body to reach vLLM (else thinking OFF + default sampling). -y auto-confirms the
# host-env prompt; --n-concurrent 2 runs both; model_info preempts the litellm context-limit/summarize error.
log ">>> Terminal-Bench (terminus-2, 2 tasks). First-ever run here — dataset cached (89 tasks); Docker OK."
# terminus-2 -> litellm needs an api_key for the openai-compatible endpoint (any value; local server ignores it).
# Set OPENAI_API_KEY env AND pass it in llm_call_kwargs (belt+suspenders) — without it every trial errors
# 'Missing credentials' before reaching the model (the 2026-08-03 first run failed exactly this way).
export OPENAI_API_KEY=EMPTY
"$VENV/bin/harbor" run -d terminal-bench/terminal-bench-2 -a terminus-2 \
  -m openai/poolside/Laguna-XS-2.1 --n-attempts 1 --env docker --n-tasks 2 --n-concurrent 1 -y \
  --agent-kwarg api_base=http://localhost:8000/v1 \
  --agent-kwarg temperature=1.0 \
  --agent-kwarg 'llm_call_kwargs={"api_key":"EMPTY","extra_body":{"top_p":1.0,"top_k":20,"chat_template_kwargs":{"enable_thinking":true}}}' \
  --agent-kwarg 'model_info={"max_input_tokens":131072,"max_output_tokens":32768}' 2>&1 | tee -a "$LOG"
log "=== Stage E DONE — SWE resolved-rate in the harness report above; TB accuracy in the harbor run result.json ==="
