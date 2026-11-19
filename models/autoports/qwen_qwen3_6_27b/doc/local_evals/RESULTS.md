# Qwen3.6-27B on QB2 M4 (4x Blackhole p300c, TP4): local benchmark + evals

Model `Qwen/Qwen3.6-27B` @ 8f33b98937c6031bb49e713dee3e8e185aafd26f, served by vLLM + TT plugin on this machine's
4 Blackhole p300c chips (TP4) with the bring-up pipeline's final serving config and selected precision
(`gdn_attn8_head8_mlp4_lofi`). Runs 2026-11-17 13:52Z to 2026-11-19 (UTC). Progress log: STATUS.md; problems and fixes:
PIPELINE-NOTES.md.

## Summary

| Deliverable | Result | Model card |
|---|---|---|
| Benchmark sweep (16 configs) | c1: 37.6 tok/s/user at 128 ctx, TTFT 0.3 s (128) / 18.8 s (32K); c32 at 128/128: 251 output tok/s total | - |
| GPQA Diamond (198, thinking) | **80.3** at 81,920 tokens (1 full run; partial reruns 81.1 on 90 Qs, 84.2 on 19 Qs) / **87.4** at 131,072 | 87.8 |
| AIME26 (30) | **86.7** at 81,920 / **93.3** at 131,072 | 94.1 |
| HMMT Feb 25 (30) | **76.7** at 81,920 / **96.7** at 131,072 | 93.8 |
| HMMT Nov 25 (30) | **80.0** at 81,920 / **90.0** at 131,072 | 90.7 |
| HMMT Feb 26 (33) | **66.7** at 81,920 / **81.8** at 131,072 | 84.3 |
| SWE-bench Verified (5 instances) | **5/5 resolved** | 77.2 (full set) |
| Terminal-Bench 2.0 (5 tasks) | **3/5 pass** | 59.3 (full set) |

Main finding: at the card's suggested 81,920-token output budget nearly every miss is a truncation. Math is 95/95
correct when the model finishes; GPQA is 91.8% on the 170 untruncated questions; but 28/123 math and 28/198 GPQA traces
hit the cap mid-thinking. A precision A/B (section 7) shows this is NOT caused by the selected BFP4/LoFi MLP policy
(bfp8_baseline truncates the same problems at the same rate). Continuing exactly those traces to a 131,072-token budget
(section 8; context contract is 262,144) brings every reasoning eval to the card: GPQA 87.4 vs 87.8, AIME26 93.3 vs 94.1,
HMMT 96.7 / 90.0 / 81.8 vs 93.8 / 90.7 / 84.3. The port is accurate; run reasoning evals on this stack with >=128K output.
Agentic: SWE-bench 5/5 resolved; Terminal-Bench 3/5 (the two medium tasks fail on the merits even with no time limit).

## Server

Stage-10 selected serving config (`doc/optimized_vllm/launch_server.sh`): all 64 layers, TP4 on P300x2 (1x4 mesh),
precision `gdn_attn8_head8_mlp4_lofi` (stage-8 selection: MLP BFP4/LoFi except layers 0/63, attention/head BFP8,
BFP8 KV), max-model-len 262144, max-num-seqs 32, block 32, scheduler budget 512 (chunked prefill, serial), async
scheduling, trace mode all, device sampling (top_k <= 64 on device), seeded sampler policy `request`, prefix caching off,
`--reasoning-parser qwen3 --tool-call-parser qwen3_xml --enable-auto-tool-choice`, default `enable_thinking=false`
(evals turn thinking on per request).

Eval-only differences (see PIPELINE-NOTES.md #1, #2): KV pool 1,049,600 tokens via `QWEN_VLLM_KV_POOL_TOKENS=1048576`
(tt-metal local commit b2d6be51; default unchanged 262144), server lifetime 72 h instead of 4 h, `HF_HOME` exported.

```
~/qwen36-27b/evals/scripts/launch_eval_server.sh ~/qwen36-27b/evals/server/run2   # sweep + all deliverables
~/qwen36-27b/evals/scripts/launch_ab_server.sh  ~/qwen36-27b/evals/server/ab_bfp8 # section 7 only (bfp8_baseline)
~/qwen36-27b/evals/scripts/launch_eval_server.sh ~/qwen36-27b/evals/server/run3   # section 8 (same config as run2)
```
Startup ~33 min cold, ~5 min warm. Until 2026-11-18 ~22:50Z several eval clients shared the server to keep its 32 slots
busy; after that, per Vladan's rule, one workload at a time (TB reruns, A/B, budget test). Sharing changes timing only.

## 1. Benchmark sweep

| ISL/OSL | Conc | Done | TTFT p50 ms | TTFT p99 ms | TPOT mean ms | ITL p50 ms | ITL p99 ms | tok/s/user (1000/TPOT) | Output tok/s total | Total (in+out) tok/s | Duration s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128/128 | 1 | 4/4 | 305.2 | 318.3 | 26.58 | 26.50 | 27.61 | 37.62 | 34.8 | 69.5 | 15 |
| 128/128 | 8 | 16/16 | 1331.4 | 2099.2 | 65.49 | 55.92 | 263.51 | 15.27 | 105.0 | 209.9 | 20 |
| 128/128 | 16 | 32/32 | 2544.0 | 3715.3 | 76.54 | 55.99 | 488.12 | 13.06 | 168.5 | 337.0 | 24 |
| 128/128 | 32 | 64/64 | 3198.6 | 7106.4 | 99.58 | 56.26 | 1109.85 | 10.04 | 251.0 | 501.9 | 33 |
| 1024/1024 | 1 | 4/4 | 567.4 | 586.2 | 26.70 | 26.70 | 27.18 | 37.45 | 36.7 | 73.4 | 112 |
| 1024/1024 | 8 | 16/16 | 2640.0 | 7487.9 | 90.43 | 56.60 | 196.68 | 11.06 | 85.4 | 170.7 | 192 |
| 1024/1024 | 16 | 32/32 | 2544.6 | 15350.6 | 120.69 | 161.08 | 760.30 | 8.29 | 126.7 | 253.3 | 259 |
| 1024/1024 | 32 | 64/64 | 2541.6 | 31275.0 | 182.71 | 162.55 | 1019.48 | 5.47 | 165.6 | 331.3 | 396 |
| 4096/512 | 1 | 4/4 | 2034.5 | 2052.8 | 26.86 | 26.85 | 27.56 | 37.23 | 32.5 | 292.4 | 63 |
| 4096/512 | 8 | 16/16 | 4391.2 | 22956.6 | 113.31 | 57.75 | 732.89 | 8.83 | 61.3 | 552.1 | 134 |
| 4096/512 | 16 | 32/32 | 4721.0 | 48518.1 | 150.68 | 57.87 | 1036.16 | 6.64 | 87.6 | 788.4 | 187 |
| 4096/512 | 32 | 64/64 | 4568.9 | 101131.3 | 221.88 | 59.49 | 1052.41 | 4.51 | 112.6 | 1013.6 | 291 |
| 32768/512 | 1 | 4/4 | 18772.6 | 20961.0 | 27.86 | 27.88 | 28.60 | 35.90 | 15.5 | 1007.2 | 132 |
| 32768/512 | 8 | 16/16 | 26801.6 | 189070.2 | 306.25 | 71.06 | 847.24 | 3.27 | 18.1 | 1176.7 | 453 |
| 32768/512 | 16 | 32/32 | 27418.0 | 388711.5 | 578.61 | 764.87 | 1186.70 | 1.73 | 19.4 | 1260.8 | 845 |
| 32768/512 | 32 | 64/64 | 404883.2 | 773041.8 | 691.83 | 764.69 | 1185.99 | 1.45 | 20.0 | 1301.7 | 1636 |

How measured: stage-11 harness `doc/benchmark/bench_exact.py` (upstream vLLM 0.26 `vllm bench serve`, exact synthetic
token-ID prompts to `/v1/completions`, no chat template, greedy temperature 0, `ignore_eos`, request rate inf,
`--max-concurrency C`). Per row: one warmup run (C prompts, discarded) then the measured run with 2*C prompts (min 4).
Server otherwise idle. Command: `evals/scripts/run_sweep.sh` (raw JSON/detailed per-request data in `evals/results/sweep/`);
table from `evals/scripts/summarize_sweep.py`. tok/s/user = 1000/mean TPOT (TPOT includes stalls caused by other users'
prefill); output tok/s total = completed output tokens / wall time; total = (input+output) / wall time.

Notes:
- All 16 configurations ran; none skipped (the 262,144-token context contract covers 32768+512).
- 4096/512 c32 measured run first failed client-side in the stage-11 adapter (seed produced a duplicate 64-token
  prefix); rerun with seed base 9100 (PIPELINE-NOTES #3).
- c1 rows match stage 10/11 (37.6 tok/s/user at 128/128; stage-10 README 37.63).
- Decode step is ~26.6 ms at batch 1 but ~56 ms for any batch >= 2 at short context (flat in batch size, so aggregate
  output scales ~linearly with users), rising to ~160 ms ITL p50 for 16-32 users at ~1-2K context.
- Prefill is serial (budget 512 tokens/step, no mixed prefill+decode batches), so TTFT p99 and TPOT at high
  concurrency are dominated by queueing behind other users' prefill. 32768/512 at c16/c32 is almost entirely prefill
  bound (~1.25-1.3K prompt tok/s total, ITL p50 ~765 ms). Single-user 32K prefill: ~18.8 s TTFT.
- 32768/512 c32 needs 32 x ~33.3K = ~1.07M KV tokens, slightly over the eval pool (1,049,600), so at most ~31 are
  resident at once. With the stage-10 default pool (263,168 tokens) only ~7 such requests could be resident.


## 2-4. Reasoning evals

| Benchmark | Problems | Model card | Ours (run 0) | Ours, excluding truncated | Truncated at 81,920 | Blank content | Answer parsed from reasoning | No answer | Output tokens mean / median / max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| GPQA Diamond | 198 | 87.8 | **80.3** (159/198) | 91.8 (156/170) | 28 | 27 | 7 (3 correct) | 21 | 37,271 / 31,760 / 81,920 |
| AIME26 (I+II) | 30 | 94.1 | **86.7** (26/30) | 100.0 (26/26) | 4 | 4 | 0 (0 correct) | 4 | 49,854 / 44,140 / 81,920 |
| HMMT Feb 25 | 30 | 93.8 | **76.7** (23/30) | 100.0 (23/23) | 7 | 7 | 0 (0 correct) | 7 | 55,972 / 51,085 / 81,920 |
| HMMT Nov 25 | 30 | 90.7 | **80.0** (24/30) | 100.0 (24/24) | 6 | 6 | 0 (0 correct) | 6 | 51,960 / 48,392 / 81,920 |
| HMMT Feb 26 | 33 | 84.3 | **66.7** (22/33) | 100.0 (22/22) | 11 | 11 | 0 (0 correct) | 11 | 58,255 / 58,852 / 81,920 |
| HMMT all three (pooled) | 93 | 89.6 (mean of 3) | **74.2** (69/93) | 100.0 (69/69) | 24 | 24 | 0 (0 correct) | 24 | 55,488 / 56,406 / 81,920 |

GPQA run-to-run noise (partial extra runs, stopped to free the device; see notes): run 1 covers 90/198 questions,
81.1% vs run 0 78.9% on the same questions; run 2 covers 19/198, 84.2% vs run 0
100.0% on the same questions. Only run 0 is a full 198-question run, so the headline is a single run (expect about ±2-3 points).

Protocol (model-card thinking mode): chat completions with `chat_template_kwargs.enable_thinking=true`, temperature 1.0,
top_p 0.95, top_k 20, min_p 0, presence_penalty 0, repetition_penalty 1.0, max_tokens 81,920 (the card's recommended
budget for competition math; the context contract allows up to ~262K but 81,920 is the card's setting), a per-request seed.
Prompts: math = problem + "\nPlease reason step by step, and put your final answer within \\boxed{}." (card); GPQA =
question + choices A-D (shuffled once with a fixed per-question RNG, same order for every run) + the card's MCQ instruction
`Please show your choice in the \`answer\` field with only the choice letter, e.g., \`"answer": "C"\`.`
Extraction: math = last `\boxed{}` in the final content, graded with `math-verify` (symbolic) with exact-string fallback;
GPQA = `"answer": "X"`-style field (tolerating missing/extra quotes, backticks, asterisks), then "answer is X", then a
boxed letter. If the content has no answer (blank because thinking never ended, or no match) the same extraction runs on
the reasoning text and is counted in "Answer parsed from reasoning".
Data: GPQA Diamond `Idavidrein/gpqa` gpqa_diamond.csv (198); MathArena HF datasets `aime_2026` (30, rev d61f4868),
`hmmt_feb_2025` (30, rev 6fdc4277), `hmmt_nov_2025` (30, rev 118dbfb4), `hmmt_feb_2026` (33, rev 589a53fe).
Commands: `venv/bin/python scripts/run_reasoning_eval.py --items data/math_items.jsonl --out results/math/math_run0.jsonl
--concurrency 12 --kv-gate 0.6` (GPQA: `--items data/gpqa_items_run0.jsonl --out results/gpqa/gpqa_run0.jsonl`); scoring
`venv/bin/python scripts/score_reasoning.py <results.jsonl>` (writes `*.graded.json` with per-item prediction/correctness).

What the numbers say:
- Every math answer that finished inside the 81,920-token budget is correct (95/95). All 28 math misses are truncations:
  the thinking never closed, so there is no final answer. GPQA is the same picture: 91.8% on the 170 non-truncated
  questions, but 28/198 (14%) truncated. The gap to the card comes from reasoning longer than the budget, not from
  wrong conclusions: with 131,072 tokens the scores match the card (section 8).
- The truncated traces are not hard repetition loops (zlib ratio of the last 20K chars is 0.19-0.33). They drift into
  exhaustive enumeration and re-verification (e.g. listing ~70 named reactions "Maybe X? (No)"). The precision A/B
  (section 7) rules out the MLP precision policy as the cause.
- Throughput context: the evals shared one server (up to 32 concurrent requests); 6 vLLM preemptions (recompute) occurred
  over ~31 h when the 1.05M-token KV pool filled. Sampling is per-request, so sharing affects only time, not answers.


## 5. SWE-bench Verified (5 instances)

Model card (Qwen3.6-27B, full 500, internal bash+file-edit scaffold): **77.2%**. Ours: **5/5 resolved** on a 5-instance
subset (not comparable as a score; a smoke-level check that the served model solves real issues end to end).

| Instance | Difficulty (Verified annotation) | Agent exit | Model calls | Agent wall time (started 15:57:32Z) | Official eval |
|---|---|---|---:|---|---|
| django__django-10880 | <15 min fix | Submitted | 45 | 55 min | RESOLVED |
| pytest-dev__pytest-10051 | 15 min - 1 hour | Submitted | 36 | 2 h 45 min | RESOLVED |
| scikit-learn__scikit-learn-10297 | 15 min - 1 hour | Submitted | 29 | 27 min | RESOLVED |
| sphinx-doc__sphinx-10323 | <15 min fix | Submitted | 32 | 56 min | RESOLVED |
| sympy__sympy-11618 | 15 min - 1 hour | Submitted | 53 | 61 min | RESOLVED |

- Selection rule (fixed before running): from `princeton-nlp/SWE-bench_Verified` test, difficulty "<15 min fix" or
  "15 min - 1 hour"; first instance_id alphabetically for each of django, pytest, scikit-learn, sphinx, sympy; an instance
  whose GOLD patch does not resolve in the published image is replaced by the next one of the same repo
  (django__django-10097 was replaced: 7/438 FAIL_TO_PASS tests error even with the gold patch). Gold patches: 5/5 resolve.
- Agent: mini-swe-agent (tool-calling `swebench.yaml` config + `evals/swebench/qwen36_local.yaml`), runs inside the
  official SWE-bench instance images, 5 workers, step limit 100. LLM via litellm `hosted_vllm/Qwen/Qwen3.6-27B`
  -> local server; thinking ON; temperature 1.0, top_p 0.95, top_k 20, presence_penalty 0, max_tokens 32768 per call
  (card: temp 1.0, top_p 0.95, 200K context; card scaffold is internal, not mini-swe-agent).
- Scoring: `swebench==4.1.0` `python -m swebench.harness.run_evaluation -d princeton-nlp/SWE-bench_Verified -s test
  -p results/run1/preds.json -i <5 ids> --max_workers 5 --cache_level instance --timeout 1800 -id qwen36-run1-final`.
- Commands: `cd evals/swebench && ./run_agent.sh results/run1 && ./score.sh $PWD/results/run1/preds.json qwen36-run1-final`.
  Trajectories: `evals/swebench/results/run1/*/*.traj.json`; report: `evals/swebench/results/scoring/qwen36-run1-final/`.
- Ran concurrently with Terminal-Bench, AIME/HMMT and GPQA on the same server (32 slots shared).


## 6. Terminal-Bench 2.0 (5 tasks)

Model card (Terminal-Bench 2.0, all 89 tasks, Harbor/Terminus-2, 3 h timeout, avg of 5 runs): **59.3%**.
Ours: **3/5 pass** on a 5-task easy/medium subset (not comparable as a score). Both medium tasks also fail when rerun
alone with no time pressure, so 3/5 is the model's result, not a timeout artefact.

| Task | Difficulty | Run 1 (3 h, shared server) | Agent time s | Input / output tokens | Run 2 (relaxed timeout, idle server) |
|---|---|---|---:|---|---|
| cobol-modernization | easy | PASS | 7704 | 130,744 / 33,288 | - |
| overfull-hbox | easy | PASS | 7387 | 153,604 / 26,595 | - |
| prove-plus-comm | easy | PASS | 7701 | 123,245 / 22,169 | - |
| adaptive-rejection-sampler | medium | FAIL (AgentTimeoutError at 10800 s) | 10800 | 278,634 / 43,360 | FAIL (alone, 3904 s, no timeout; 7/9 tests pass, fails test_can_generate_standard_distribution_samples and test_formal_testing_with_known_truth) |
| chess-best-move | medium | FAIL (AgentTimeoutError at 10800 s) | 10800 | 309,482 / 42,924 | FAIL (alone, 3433 s, no timeout; wrong move b2b3, expected e2e4 or g2g4; board is a PNG and the server is text-only) |

- Harness: Harbor 0.26.0, dataset `terminal-bench@2.0` (89 tasks), agent `terminus-2`, litellm `openai/Qwen/Qwen3.6-27B`
  -> local server. Thinking ON (`extra_body.chat_template_kwargs.enable_thinking=true`), temperature 1.0, top_p 0.95,
  top_k 20, max_tokens 81,920 (card: temp 1.0, top_p 0.95, top_k 20, max_tokens 80K, 3 h timeout, 256K ctx).
  `model_info.max_input_tokens` 178,176 so Terminus-2 summarizes before the 262,144 context limit.
- Task selection (fixed before running): easy first, then medium, alphabetical; keep a task only if its Dockerfile builds
  on this machine and the oracle solution passes on both the prebuilt and the rebuilt image. Skipped: fix-git (Dockerfile
  no longer builds, GitHub clone now asks for credentials), break-filter-js-from-html and build-pov-ray (oracle fails on
  rebuilt image), build-cython-ext and build-pmars (oracle fails even on the prebuilt image), caffe-cifar-10 (oracle
  exceeds its own timeout). Oracle: 5/5 chosen tasks pass.
- Run 1 used `--agent-timeout-multiplier 12` (900 s x 12 = 3 h, the card's limit) and ran concurrently with SWE-bench
  and the reasoning evals (~4 tok/s per agent). The two medium tasks were still working at 3 h, so the timeouts measure
  this server's throughput under load more than the model.
- Run 2 reruns only those two tasks, alone on the server, one task at a time (owner rule), timeout x36 (~9 h); both
  finished well inside the card's 3 h (65 and 57 min) and failed verification. Run 2 input tokens 1.66M / 1.69M
  (re-prefilled transcripts), output 75K / 83K. An earlier relaxed attempt that overlapped other evals was discarded.
- Commands: `cd evals/terminalbench && AGENT_TIMEOUT_MULT=12 ./run_agent.sh qwen36-run1`; run 2:
  `TASKS_OVERRIDE=adaptive-rejection-sampler AGENT_TIMEOUT_MULT=36 ./run_agent.sh qwen36-relaxed-ars`, then the same with
  `chess-best-move` / `qwen36-relaxed-chess`.
  Results: `evals/terminalbench/jobs/<job>/<task>__*/result.json` (pass iff verifier reward 1.0).


## 7. Precision A/B (extra, diagnostic): selected policy vs bfp8_baseline on AIME26 + HMMT Feb 26

Question: are the truncations caused by the stage-8 low-precision MLP policy? Same 63 prompts, same seeds, same
sampling and budget, on a second server identical except `QWEN_VLLM_PRECISION_CONFIG=.../configs/bfp8_baseline.json`
(all weights BFP8, MLP HiFi2, GDN attention HiFi4; tt-metal local commit 002b2466 adds this env override). Run alone.

| Set | Selected policy (`gdn_attn8_head8_mlp4_lofi`) | bfp8_baseline | Card |
|---|---|---|---:|
| AIME26 (30) | 26/30 = 86.7, 4 truncated, mean 49.9K tok | 26/30 = 86.7, 4 truncated, mean 48.1K tok | 94.1 |
| HMMT Feb 26 (33) | 22/33 = 66.7, 11 truncated, mean 58.3K tok | 24/33 = 72.7, 9 truncated, mean 58.5K tok | 84.3 |
| Both (63) | 48/63 = 76.2, 15 truncated | 50/63 = 79.4, 13 truncated | |

- 12 of the truncations are the same problems under both policies (AIME26 #15, #28, #30; HMMT Feb26 #2, 7, 13, 17, 18,
  19, 20, 27, 33). Only one config truncated: selected on AIME #10, HMMT #3 and #31; bfp8 on AIME #29. Every finished answer
  is correct under both. Output lengths are the same.
- Conclusion: the precision policy is not the cause (the 2-problem difference is within sampling noise). These problems
  need more than 81,920 thinking tokens on this serving stack. Section 8 shows they finish (mostly correctly) with
  more budget. Whether traces here run somewhat longer than on the card's GPU harness (shared factors: device sampler,
  BFP8 KV cache, bf16 GDN state) was not isolated; a GPU reference run would be needed.
- Commands: `scripts/launch_ab_server.sh server/ab_bfp8`; `venv/bin/python scripts/run_reasoning_eval.py --items
  data/ab_items.jsonl --out results/ab/ab_bfp8.jsonl --concurrency 32 --kv-gate 0.7`.


## 8. Output-budget test (extra, diagnostic): continue truncated traces to 131,072 tokens

Each trace that hit the 81,920 cap was continued from exactly where it stopped (chat-templated prompt + the generated
reasoning sent to `/v1/completions`, +49,152 tokens, same sampling, selected-precision server, run alone, 8 at a time).
This is what the run would have scored with `max_tokens=131072` (well within the 262,144 context contract); the
reasoning text is re-tokenized, which can differ from the generated token ids at a few boundaries.

| Set | Truncated at 81,920 | Concluded with +49K | Correct after continuing | Score @81,920 | Score @131,072 | Card |
|---|---:|---:|---:|---:|---:|---:|
| AIME26 | 4 | 4 | 2 | 86.7 | **93.3** | 94.1 |
| HMMT Feb 25 | 7 | 7 | 6 | 76.7 | **96.7** | 93.8 |
| HMMT Nov 25 | 6 | 5 | 3 | 80.0 | **90.0** | 90.7 |
| HMMT Feb 26 | 11 | 9 | 5 | 66.7 | **81.8** | 84.3 |
| GPQA Diamond | 28 | 27 | 17 | 80.3 | **87.4** | 87.8 |

- Most truncated traces were close to finishing (median +8K tokens). Answers needing +14K to +41K extra were more often
  wrong, as expected for the hardest problems. No hard repetition loops.
- Commands: `venv/bin/python scripts/continue_truncated.py --results results/math/math_run0.jsonl --items
  data/math_items.jsonl --out results/continue/math_cont.jsonl --extra 49152 --concurrency 8` (GPQA: `--results
  results/gpqa/gpqa_run0.jsonl --items data/gpqa_items_run0.jsonl --out results/continue/gpqa_cont.jsonl`).
