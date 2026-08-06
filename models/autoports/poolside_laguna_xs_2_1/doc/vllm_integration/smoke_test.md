# Laguna-XS-2.1 on TT — Smoke-Test Runbook

Copy-paste runnable. Validates the optimized TT vLLM server (Phase A sliding-window
KV + Phase B prefix-caching, default ON) with the **thinking-mode fix**
(`--reasoning-parser deepseek_r1`, so `<think>…</think>` is split into
`reasoning_content` and agent action-parsers see clean `content`).

Model edits are **uncommitted live-in-tree** across 3 repos — **relaunch, never
rebuild** (a ttnn `ninja`/`build_metal` would drop the uncommitted `.so`).

---
## 0. Prereqs — board + server

```bash
# Board must be free; recover if wedged (never self-match pkill -f).
for p in $(lsof -t /dev/tenstorrent/* 2>/dev/null | sort -u); do kill -9 "$p" 2>/dev/null; done
tt-smi -r          # reset the 4 P150s
```

Relaunch the server (detached, so a reaped shell doesn't kill it). This is the
EXACT working config — env + reasoning parser + prefix caching:

```bash
nohup env \
  TT_METAL_HOME=/home/ttuser/.local/lib/model-bringup/tt-metal \
  PYTHONPATH=/home/ttuser/dev/tt-metal:/home/ttuser/.local/lib/model-bringup/tt-metal/vllm:/home/ttuser/.local/lib/model-bringup/tt-metal/vllm/plugins/vllm-tt-plugin/src \
  TT_LAGUNA_PIPE_CHUNK=2048 \
  TT_LAGUNA_PREFIX_CACHE=1 \
  bash -c 'cd /tmp && stdbuf -oL -eL /home/ttuser/.tenstorrent-venv/bin/python -u \
     -m models.common.readiness_check.run_vllm_server \
     --model-dir /home/ttuser/dev/tt-metal/models/autoports/poolside_laguna_xs_2_1 \
     --hf-model poolside/Laguna-XS-2.1 --mesh-device P150x4 --stages serve \
     --max-num-seqs 32 --block-size 64 --max-model-len 262144 \
     --tt-config "{\"trace_region_size\": 1500000000, \"fabric_config\": \"FABRIC_1D_RING\"}" \
     --additional-server-args="--trust-remote-code --max-num-batched-tokens 131072 --enable-prefix-caching --enable-auto-tool-choice --tool-call-parser poolside_v1 --reasoning-parser poolside_v1"' \
  > /home/ttuser/dev/tt-metal/models/autoports/poolside_laguna_xs_2_1/doc/vllm_integration/serve.log 2>&1 &
disown
```

Tail: `readiness_vllm/server.log` (engine) and `doc/vllm_integration/serve.log`
(launcher). Boot ≈ 8 min (MoE weight load → warmup → decode trace).

> Benign log noise (ignore): `Unrecognized keys in rope_parameters ... {sliding_attention, full_attention}` is transformers' generic RoPE validator not understanding Laguna's per-layer-type rope config — the TT decoder computes RoPE itself (validated). Also `Triton ... Disabling Triton` and `Unable to set process priority` are harmless.

Confirm ready + optimizations on:
```bash
# ready when this prints the model id
curl -s http://localhost:8000/v1/models | python3 -c 'import sys,json;print(json.load(sys.stdin)["data"][0]["id"])'
curl -s -o /dev/null -w 'health=%{http_code}\n' http://localhost:8000/health           # -> 200
SLOG=/home/ttuser/dev/tt-metal/models/autoports/poolside_laguna_xs_2_1/readiness_vllm/server.log
grep -oE "enable_prefix_caching=(True|False)" "$SLOG" | tail -1                         # -> True
grep -oE "reasoning_parser='[^']*'" "$SLOG" | tail -1                                   # -> 'poolside_v1'
grep -oE "Maximum concurrency for 262,144 tokens per request: [0-9.]+x" "$SLOG" | tail -1  # -> 1.51x (Phase A)
```

Thinking-mode sanity (reasoning split, content clean). **On a RAW curl, `top_k`
and `chat_template_kwargs` must be TOP-LEVEL body fields — NOT nested under
`extra_body`** (vLLM ignores a literal top-level `extra_body` key and logs
`fields ... ignored: {'extra_body'}`):
```bash
curl -s http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model":"poolside/Laguna-XS-2.1",
  "messages":[{"role":"user","content":"Reason briefly, then output ANSWER=<n> for 17*23."}],
  "max_tokens":2048,"temperature":1.0,"top_p":1.0,
  "top_k":20,"chat_template_kwargs":{"enable_thinking":true}}' \
| python3 -c 'import sys,json;m=json.load(sys.stdin)["choices"][0]["message"];r=m.get("reasoning") or m.get("reasoning_content") or "";print("reasoning set:",bool(r.strip()));print("leak in content:", "<think>" in (m.get("content") or "") or "</think>" in (m.get("content") or ""))'
# Expect: reasoning set: True | leak in content: False.  (The reasoning field is
# `reasoning` in this vLLM build; some clients expose it as `reasoning_content`.)
```
Confirm the ignored warning is NOT produced by this call:
```bash
grep -c "were present in the request but ignored" \
  /home/ttuser/dev/tt-metal/models/autoports/poolside_laguna_xs_2_1/readiness_vllm/server.log
# should not increase after a correctly-formed (top-level) request
```

Sampling used everywhere below (poolside published): **temperature=1.0, top_p=1.0,
top_k=20, thinking ON**. Passing rules:
- **Raw curl / HTTP:** put `top_k` and `chat_template_kwargs` at the request-body
  TOP LEVEL (a literal `extra_body` key is dropped by vLLM).
- **OpenAI python client:** pass `extra_body={"top_k":20,"chat_template_kwargs":{"enable_thinking":true}}`
  — the SDK flattens it to the top level (correct).
- **litellm / mini-swe-agent / harbor:** litellm also flattens `extra_body`, but to
  be unambiguous the configs below pass `top_k` and `chat_template_kwargs` as
  DIRECT model kwargs (no `extra_body`). Thinking is effectively always on for this
  model: the chat template gates the opening `<think>` primer on `enable_thinking`
  (default false), but the model emits `<think>…</think>` **spontaneously even with
  the flag off**, so `enable_thinking:false` does NOT disable reasoning. The
  `deepseek_r1` parser routes it to `reasoning` and leaves `content` clean.

> ⚠️ **Verbose always-on thinking → budget the output tokens.** Reasoning can consume
> the entire generation budget: a trivial prompt at `max_tokens=200` returned ~930
> chars of reasoning and an **empty `content`** (the answer was never reached). So for
> any REAL answer give a generous `max_tokens` (≥ ~2048; the agent configs below use
> 32768). The `--ignore-eos` latency bench (Smoke 1) is unaffected — it generates a
> fixed token count regardless — but quality/coherence checks and real serving need
> room for reasoning + answer.

---
## Smoke 1 — latency @ ISL 131072 / OSL 396

`vllm bench serve` is an HTTP client (talks to :8000). Uses the vLLM in the
server's venv:

```bash
OUT=/home/ttuser/dev/tt-metal/models/autoports/poolside_laguna_xs_2_1/doc/vllm_integration/smoke/lat_131072_396.json
mkdir -p "$(dirname "$OUT")"
# REQUIRED: the `vllm bench` CLI resolves `vllm.benchmarks.*` from the .local tree —
# without the server PYTHONPATH it picks up a partial vllm and dies with
# `ModuleNotFoundError: No module named 'vllm.benchmarks.latency'`.
export PYTHONPATH=/home/ttuser/dev/tt-metal:/home/ttuser/.local/lib/model-bringup/tt-metal/vllm:/home/ttuser/.local/lib/model-bringup/tt-metal/vllm/plugins/vllm-tt-plugin/src
/home/ttuser/.tenstorrent-venv/bin/vllm bench serve \
  --backend vllm --base-url http://localhost:8000 --endpoint /v1/completions \
  --model poolside/Laguna-XS-2.1 \
  --dataset-name random --num-prompts 1 \
  --random-input-len 131072 --random-output-len 396 \
  --max-concurrency 1 --ignore-eos \
  --save-result --result-filename "$OUT"
```

> **On OSL 396 vs. always-on thinking:** 396 is the requested coding-support output
> shape and is a valid *latency* measurement here because `--ignore-eos` forces exactly
> 396 decode steps regardless of what the model would emit. Be aware, though, that in
> real (EOS-respecting) serving those ~396 tokens would be mostly *reasoning* — a real
> answer needs more headroom. For a decode-latency point that reflects a full
> reasoning+answer response, add a companion run with `--random-output-len 2048`.

Read the console summary (also in `$OUT`):
- **TTFT** = `Mean TTFT (ms)` (prefill latency; ISL 131072 ~ tens of s cold).
- **TPOT** = `Mean TPOT (ms)` (per-output-token; inverse of per-user tok/s).
- **tok/s/user** ≈ `1000 / Mean TPOT (ms)`.
- **E2EL** = `Mean E2EL (ms)` (end-to-end for the request).
```bash
python3 -c 'import json;d=json.load(open("'"$OUT"'"));print({k:d[k] for k in d if any(s in k for s in ("ttft","tpot","e2el","throughput","itl"))})'
```

---
## Smoke 2 — code accuracy (pass@1)

### 2a. Self-contained code eval (no dataset, fastest)
```bash
/home/ttuser/.tenstorrent-venv/bin/python -m pip install -q openai   # if missing
python /home/ttuser/dev/tt-metal/models/autoports/poolside_laguna_xs_2_1/doc/vllm_integration/scripts/eval_code.py \
  --base-url http://localhost:8000/v1 --model poolside/Laguna-XS-2.1 --temperature 0.0
# prints per-problem PASS/FAIL + "pass@1 = N/10"
```

### 2b. HumanEval via lm-eval (**completion mode — NOT chat**)
The 2026-07-28 11:34 HumanEval scored 0% purely because it ran under the chat
template (model answered in prose). Use `local-completions` (raw completion):
```bash
python3 -m venv /home/ttuser/.claude/jobs/b4840290/tmp/lmeval-venv
LM=/home/ttuser/.claude/jobs/b4840290/tmp/lmeval-venv/bin
"$LM/pip" install -q "lm-eval[api]==0.4.12" transformers   # transformers is NOT pulled by [api]; lm-eval imports it for the tokenizer backend
OUT=/home/ttuser/dev/tt-metal/models/autoports/poolside_laguna_xs_2_1/doc/vllm_integration/smoke/lmeval
"$LM/lm_eval" --model local-completions \
  --model_args model=poolside/Laguna-XS-2.1,base_url=http://localhost:8000/v1/completions,num_concurrent=8,max_retries=3,tokenized_requests=False \
  --tasks humaneval --confirm_run_unsafe_code \
  --output_path "$OUT"
# pass@1 in the printed table (humaneval / humaneval_pass@1). Do NOT pass --apply_chat_template.
```

---
## Smoke 3 — agentic (SWE-bench + Terminal-Bench)

### Eval stack (fresh venv; reuse if `venv_agentic` still present)
```bash
VENV=/home/ttuser/.claude/jobs/b4840290/tmp/venv_agentic
# If missing, recreate (TT canonical stack):
uv venv "$VENV" 2>/dev/null || python3 -m venv "$VENV"
"$VENV/bin/pip" install -q mini-swe-agent==2.2.8 \
  "git+https://github.com/epoch-research/SWE-bench.git@bc2a82af2874e26fa6f206ae0ad9017c4768daa2" \
  orjson backoff pyjwt openai
git clone --depth 1 --branch v0.6.5 https://github.com/harbor-framework/harbor.git "$VENV/harbor" 2>/dev/null
"$VENV/bin/pip" install -q -e "$VENV/harbor"
git clone https://github.com/SWE-agent/SWE-agent.git "$VENV/SWE-agent" 2>/dev/null
"$VENV/bin/pip" install -q -e "$VENV/SWE-agent"
```

Model config for mini-swe-agent (thinking-mode-fixed sampling + step_limit 75).
Write it once:
```bash
cat > /home/ttuser/.claude/jobs/b4840290/tmp/mini_model_config.yaml <<'YAML'
model:
  model_name: "openai/poolside/Laguna-XS-2.1"
  model_class: "litellm_textbased"
  cost_tracking: "ignore_errors"
  model_kwargs:
    api_base: "http://localhost:8000/v1"
    api_key: "EMPTY"
    drop_params: true
    temperature: 1.0
    top_p: 1.0
    max_tokens: 32768
    top_k: 20                        # top-level (NOT under extra_body)
    chat_template_kwargs:
      enable_thinking: true
agent:
  step_limit: 75
  cost_limit: 1000.0
YAML
```

### 3a. SWE-bench Verified — 10 instances
```bash
VENV=/home/ttuser/.claude/jobs/b4840290/tmp/venv_agentic
OUT=/home/ttuser/dev/tt-metal/models/autoports/poolside_laguna_xs_2_1/doc/vllm_integration/smoke/swebench10
mkdir -p "$OUT"
"$VENV/bin/mini-extra" swebench \
  --subset verified --split test --slice 0:10 --workers 5 \
  --config swebench_backticks.yaml \
  --config /home/ttuser/.claude/jobs/b4840290/tmp/mini_model_config.yaml \
  --output "$OUT"
# Per-instance patches: $OUT/preds.json ; trajectories: $OUT/<instance>/<instance>.traj.json
```
Gate — patches non-empty + parseable:
```bash
python3 -c '
import json;d=json.load(open("'"$OUT"'/preds.json"))
ne=sum(1 for v in d.values() if (v.get("model_patch") or "").strip())
print(f"non-empty patches: {ne}/{len(d)}")
for k,v in d.items(): print(" ",k,"->", "PATCH" if (v.get("model_patch") or "").strip() else "EMPTY")'
```
Grade (resolved-rate) — convert preds to jsonl, run the epoch/official harness:
```bash
python3 -c '
import json;d=json.load(open("'"$OUT"'/preds.json"))
open("'"$OUT"'/preds.jsonl","w").write("\n".join(json.dumps(v) for v in d.values()))'
"$VENV/bin/python" -m swebench.harness.run_evaluation \
  --dataset_name princeton-nlp/SWE-bench_Verified --split test \
  --predictions_path "$OUT/preds.jsonl" --run_id smoke10 --max_workers 4
# resolved-rate = resolved / submitted, in the generated <model>.smoke10.json report.
```

### 3b. Terminal-Bench 2.0 — 5–10 tasks (Harbor)
```bash
VENV=/home/ttuser/.claude/jobs/b4840290/tmp/venv_agentic
"$VENV/bin/harbor" run -d terminal-bench/terminal-bench-2 -a terminus-2 \
  -m openai/poolside/Laguna-XS-2.1 --n-attempts 1 --env docker \
  --n-tasks 10 \
  --agent-kwarg api_base=http://localhost:8000/v1 \
  --agent-kwarg temperature=1.0 --agent-kwarg top_p=1.0 --agent-kwarg top_k=20 \
  --agent-kwarg 'chat_template_kwargs={"enable_thinking":true}'
# resolved-rate in the run's result.json ("accuracy"/"resolved"). Host: 16 CPU / 249 GB;
# poolside's sandbox spec was 48GB/32CPU per task — flag any task needing more than available.
# NOTE: terminus-2 expects tool/function calling; if this server rejects it, fall back to a
# harbor mini-swe-agent adapter (examples/configs/mini-swe-agent-job.yaml) or skip 3b.
```

---
## Results table template

| Model / HW | Metric | ISL/OSL or N | TTFT | TPOT | tok/s/u | E2EL | Score |
|---|---|---|---|---|---|---|---|
| Laguna-XS-2.1 / TT P150x4 | latency | 131072/396 | … | … | … | … | — |
| Laguna-XS-2.1 / TT P150x4 | HumanEval pass@1 | 164 | — | — | — | — | …% |
| Laguna-XS-2.1 / TT P150x4 | SWE-bench Verified | 10 (pass@1) | — | — | — | — | …% |
| Laguna-XS-2.1 / TT P150x4 | Terminal-Bench 2.0 | ≤10 (pass@1) | — | — | — | — | …% |

**Reference (poolside self-reported):** SWE-bench Verified **70.9%** (mean pass@1
over 4 attempts), Terminal-Bench 2.0 **37.5%** (mean pass@1 over 5 attempts), with
their **private agent** + Harbor, temp=1.0/top_k=20/top_p=1, thinking on.

**Caveats (headline any comparison):** small-N smoke (not the full 500/89);
**public scaffold** (mini-swe-agent/terminus-2) vs poolside's **private agent** —
scores are agent-dependent (>10 pt swings across agents); **1 attempt** vs their
4/5-attempt mean; matched sampling; TT hardware vs their undisclosed serving HW.
Numbers are directional, NOT apples-to-apples.
