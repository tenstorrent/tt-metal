# Pipeline notes: problems hit while running local evals of Qwen3.6-27B (QB2 M4)

Format: what broke / why / fix / should the bring-up pipeline have caught it.

## 1. Serving KV pool is sized for exactly one max-length sequence
- What: `QwenVLLM.get_max_tokens_all_users` returns 262144, so the pool is 8224 pages = 263,168 tokens shared by all
  32 slots ("Maximum concurrency for 262,144 tokens per request: 1.00x"). 32 concurrent thinking requests at
  10-80K tokens each need far more, so vLLM would preempt and recompute. Recompute/resume was never validated with the
  model-owned GDN recurrent state (P4 long-answer load peaked at 73.5% KV and never preempted).
- Why: the stage-10 pool was the "validated full-model pool"; the eval-readiness forecast assumed 6000-token answers.
- Fix: tt-metal commit b2d6be51 adds env override `QWEN_VLLM_KV_POOL_TOKENS` (default unchanged 262144). Eval server
  uses 1,048,576 tokens (4x). DRAM check from stage-10 server log: model uses ~1.18 GB of 4.26 GB per bank (8 banks,
  ~9.4 of ~34 GB per chip); BFP8 KV for 262K tokens is ~2.3 GB/chip, so 4x fits.
- Pipeline: yes. Eval readiness should size the pool for max_num_seqs x realistic reasoning length (model card says
  81,920 output tokens) or explicitly test preemption-resume for recurrent-state models.
- Follow-up (code read, not a test): `generator.py` prefill calls `release_request(slot)` for any row with start==0, and
  with prefix caching off a preempted request is recomputed from position 0, so the GDN state should be rebuilt
  correctly on resume. Still unvalidated by any bring-up test.

## 2. Stage-10 launcher kills the server after 4 h and depends on the caller's HF_HOME
- What: `doc/optimized_vllm/launch_server.sh` runs under `timeout 14400`; a GPQA x3 + math + agent eval campaign is
  far longer. First eval-server launch failed: `FileNotFoundError ~/.cache/huggingface/hub/.../model.safetensors.index.json`
  because `doc/vllm_integration/env.sh` does not set `HF_HOME` (weights are under ~/qwen36-27b/hf; only lane-env.sh sets it).
- Fix: evals/scripts/launch_eval_server.sh = same server args, `timeout 259200`, exports HF_HOME.
- Pipeline: yes. The "final serving command" should be self-contained (set HF_HOME) and not carry a short lifetime.

## SWE-bench setup (evals/swebench, harness-side only; details in evals/swebench/README.md)
- swebench 5.0.2 (latest pip) expects datasets with `image`/`eval_script`/`log_parser` columns and cannot score
  `princeton-nlp/SWE-bench_Verified`; fix: pin `swebench==4.1.0` (classic harness, pulls docker.io/swebench/sweb.eval.*,
  the same images mini-swe-agent runs in). Pipeline: no (eval tooling).
- Gold validation: django__django-10097 does NOT resolve with its gold patch in the published image (7/438
  FAIL_TO_PASS auth_tests.test_templates.AuthTemplateTests error at response.render()); replaced by the next instance
  under the same rule, django__django-10880 (gold resolves). Pipeline: no.
- mini-swe-agent defaults unsuited to a ~35 tok/s local server: container lifetime `sleep 2h`, cost_limit 3.0 with
  cost tracking that raises on unknown models, interactive first-run wizard. Fix: qwen36_local.yaml
  (container_timeout 72h, cost_limit 0, cost_tracking ignore_errors, litellm timeout 7200 s) + MSWEA_CONFIGURED=true.
  Pipeline: partly - the bring-up's agentic smoke test should state realistic per-turn latency for agent harness timeouts.
- Server runs with `--no-enable-prefix-caching`: every agent turn re-prefills the whole growing transcript (interleaved
  thinking is preserved by the Qwen3.6 template, so history grows fast). Expect slow late turns; not fixed (server owned
  by bring-up). Pipeline: yes, agentic multi-turn throughput with prefix caching off should be measured at stage 10/11.

## Terminal-Bench setup (evals/terminalbench, harness-side only; details in evals/terminalbench/README.md)
- Harbor 0.26.0 rejects `hosted_vllm/Qwen/Qwen3.6-27B` (hosted_vllm names may contain only one '/'). Fix: litellm
  `openai/Qwen/Qwen3.6-27B` + api_base + dummy OPENAI_API_KEY + explicit model_info (otherwise Terminus-2 assumes a
  1M-token context and never summarizes before vLLM's 262144 limit). Pipeline: no (harness quirk).
- Thinking is off by server default (`--default-chat-template-kwargs enable_thinking=false`); Terminus-2 re-enables it
  per request via `llm_call_kwargs.extra_body.chat_template_kwargs`. top_k sent in extra_body. Verified the exact
  request body against a local mock OpenAI server (no model traffic). Pipeline: no.
- Task bit-rot: fix-git's Dockerfile no longer builds (setup.sh clones a GitHub repo that now asks for credentials);
  break-filter-js-from-html, build-pov-ray pass the oracle only on the prebuilt image, not on a rebuilt one;
  build-cython-ext and build-pmars fail the oracle even on the prebuilt image (pinned apt version dpkg-dev 1.22.21 gone /
  upstream pip drift); caffe-cifar-10's own oracle exceeds its 1200 s agent timeout on 1 CPU. Fix: selection rule requires Dockerfile build + oracle pass on prebuilt AND rebuilt images.
  Pipeline: no.
- Default Terminal-Bench agent timeouts (750-900 s) assume fast API models; at ~35 tok/s with thinking a single turn can
  take minutes. Fix: --agent-timeout-multiplier 6, litellm request timeout 7200 s. Pipeline: partly (same as SWE-bench).

## 3. Stage-11 exact-token harness rejects some seeds (duplicate 64-token prefix)
- What: sweep row 4096/512 c32 (64 prompts, seed 7024) failed in `bench_exact.py`: "Upstream random seed produced
  duplicate prompt/prefix; choose a fresh declared seed". The adapter builds prompts as consecutive slices of the allowed
  vocabulary from an RNG offset, so with many prompts two offsets can land within 64 tokens.
- Fix: rerun the row with a different declared seed (SEED_BASE env in evals/scripts/run_sweep.sh). No server impact.
- Pipeline: minor. The harness could resample the offset instead of failing; stage 11 only ever ran <=96 prompts at 4096.

## 4. Agent workloads make the server prefill-bound (prefix caching off)
- What: with 10 agents (5 SWE-bench + 5 Terminal-Bench) plus 22 single-turn reasoning requests, /metrics showed
  ~853 prompt tok/s vs ~110 generated tok/s (2026-11-17 16:30Z). Each agent turn re-prefills the full transcript
  (`--no-enable-prefix-caching`, required because GDN recurrent state cannot be rebuilt from attention pages), and
  prefill is serial with no mixed prefill+decode steps, so all decoders stall during it. A GPQA answer of 4952 tokens
  took 1534 s (3.2 tok/s/user).
- Fix: none at harness level (kept all workloads on one server since the device is saturated either way). Recorded
  so throughput numbers in RESULTS.md are read in that context.
- Pipeline: yes. Stage 10/11 should measure a multi-turn agent workload; a GDN-state snapshot per prefix (prefix
  caching for hybrid models) or mixed prefill/decode batching would be the big win for agentic serving.

## 5. GPQA answer format variant
- What: model sometimes ends with `answer: "D"` (unquoted key) instead of the requested `"answer": "D"`; first
  extractor fell back to the reasoning text for it. Fix: regex accepts optional quotes/asterisks around key and letter
  (letter match kept case-sensitive to avoid "answer is a ..." false positives). Pipeline: no (eval client).

## 6. Terminal-Bench medium tasks time out at the card's 3 h limit under shared load
- What: with ~30 concurrent long-context requests, each agent got ~4 tok/s; adaptive-rejection-sampler and
  chess-best-move hit AgentTimeoutError at 10800 s after ~43K output tokens (~290-310K input tokens re-prefilled).
  The three easy tasks passed in ~2.1 h each.
- Fix: run1 is reported as-is (card-matching 3 h). Timed-out tasks rerun (run2) with timeout x36 (~9 h), reported
  separately and labelled "relaxed timeout". Pipeline: yes - throughput per agent under load (not just c1 decode) decides
  whether agentic benchmarks are even runnable within their published wall-clock limits.
- Follow-up: TB run2 (relaxed timeout) was stopped after ~3 h while reasoning evals were running: 2 agents with
  150-300K-token transcripts kept prefill at ~400 tok/s and cut generation from ~209 to ~107 tok/s for everyone.
  Rescheduled to run alone after the reasoning evals (same total device work, but the agents are no longer starved).

## 7. Long-reasoning truncation at the card's 81,920-token budget (precision suspected, then ruled out by A/B)
- What: AIME26 run0 = 26/30 (86.7%) vs card 94.1; all 4 misses ran to the 81,920-token cap with no final answer.
  GPQA interim: 9 of ~80 answers truncated at 81,920 (card-style runs normally truncate rarely). Truncated traces are not
  hard loops (zlib ratio of last 20K chars 0.19-0.33) but drift into exhaustive enumeration, e.g. gpqa-24 lists ~70
  named reactions/elements "Maybe X? (No)" before "Okay, I'm stuck in a loop", and burns the budget.
- Why (hypothesis): stage 8 selected `gdn_attn8_head8_mlp4_lofi`, the fastest policy that passes token-agreement gates
  measured on 100-512-token windows: natural-text TF top-1 91.8% vs 97.9% for bfp8_baseline (gate is 90%). Small
  per-token drift may compound over 30-80K-token reasoning traces.
- Final numbers (selected config, 81,920 budget): math 95/95 correct when the answer finished, but 28/123 truncated
  (AIME26 4/30, HMMT Feb25 7/30, Nov25 6/30, Feb26 11/33); GPQA 28/198 truncated, 91.8% on the rest. Every math miss is a
  truncation, so the gap to the card (e.g. HMMT Feb26 66.7 vs 84.3) comes entirely from non-terminating reasoning.
- Fix: none applied to the deliverables (they use the pipeline's selected config, as instructed). A/B after the main
  deliverables: AIME26 + HMMT Feb26 (63 problems, same prompts/seeds/sampling) on a bfp8_baseline server.
- A/B result (2026-11-19, same 63 prompts/seeds/sampling, one workload at a time): bfp8_baseline 50/63 with 13
  truncated vs selected 48/63 with 15 truncated; 12 truncations are the same problems; mean output tokens equal (48-58K);
  every finished bfp8 answer correct. The BFP4/LoFi MLP policy is NOT the cause; the hypothesis is rejected.
- Remaining suspects shared by both configs (not testable here without a reference GPU run): device sampler
  (top-k 20 / top-p 0.95 on device), BFP8 KV cache, bf16 GDN recurrent state/activations, or simply that this model needs
  >81,920 tokens on these problems more often than the card's harness shows. A budget-continuation test follows (#10).
- Pipeline: yes, but the actionable part is different: stage 10/11 should run a small long-reasoning task (e.g. 10 AIME
  problems at the card budget) and compare truncation rate against a GPU reference; short-window top-k agreement cannot
  see this. The datatype sweep's precision choice is fine.

## 8. Long-reasoning eval throughput is bounded by decode attention over total in-flight context
- What: measured decode step ~= 45 ms + ~0.1 ms per 1K tokens of total in-flight context (fits sweep 32768/512 c8:
  ~264K tokens -> 71 ms; eval: 11 users / ~700K tokens -> ~112 ms, 98 tok/s). With 30-60K-token thinking traces the
  server tops out near ~150-200 generated tok/s regardless of user count, and the 1.05M-token KV pool caps concurrency
  at ~16-20 such traces. 123 math + 198 GPQA problems at the card's 81,920 budget is ~1-1.5 days of device time.
- Fix: none possible at harness level; scheduled the primary deliverables first, cut GPQA to 2 runs.
- Pipeline: yes. The stage-10 eval-readiness forecast (GPQA 2.63 h assuming 12K-token answers at 6000-token measured
  speed) was ~10x optimistic; forecast should use card-budget answer lengths and measure decode at 30-80K context x 16-32
  users. Decode attention at long context is the main optimization target for reasoning workloads.

## 9. Owner rule adopted mid-campaign: one workload at a time on the server
- Until 2026-11-18 ~22:25Z several eval clients (reasoning evals + agents) shared the server to keep 32 slots busy.
  Vladan's rule: strictly one workload at a time. The overlapping TB relaxed rerun (job qwen36-run2b-relaxed) was stopped
  and discarded; TB timed-out tasks are rerun alone, one task at a time, then the precision A/B. Earlier results (SWE-bench,
  TB run1, math, GPQA) were produced under shared load: this affects timing/timeouts only, not per-request sampling.

## 10. The reasoning gap to the model card is the 81,920 output budget (budget-continuation test)
- What: continued each of the 28 truncated math traces (selected config) from exactly where it stopped, via
  /v1/completions with chat-templated prompt + generated reasoning, +49,152 tokens (total 131,072, within the 262,144
  context contract). 25/28 concluded (median +8K tokens); scores become AIME26 93.3 (card 94.1), HMMT Feb25 96.7 (93.8),
  Nov25 90.0 (90.7), Feb26 81.8 (84.3). Same for GPQA: 27/28 concluded (median +6.5K), 17 correct ->
  87.4 (card 87.8). Answers that needed many extra tokens were more often wrong (10 wrong, mostly
  +14K..+41K), consistent with genuinely hard problems rather than degenerate loops.
- Fix: none needed in the model. Reported both budgets. CLAUDE.md's lesson ("use the largest budget the context contract
  allows, 64K-128K") is confirmed: at the card's own 81,920 suggestion this stack loses 7-18 points to truncation.
- Pipeline: yes - eval-readiness should (a) default reasoning evals to >=128K output when the context allows and (b)
  size the KV pool / time budget for that (see #1, #8). Precision selection is not implicated (#7 A/B).
