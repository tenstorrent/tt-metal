# Ornith-1.5-9B on QB2 (4x Blackhole p300c, TP4 mesh 1x4): local benchmark and eval results

Model `ornith-ai/Ornith-1.5-9B` @ `489cb97981b8654bcfcf30ce1f94ed1b62e07b53`. All results come from **one** vLLM
OpenAI-compatible server on this machine (qb2-120-p03t07), started once and reused for every section.

## Summary

| # | Deliverable | Result |
|---|---|---|
| 1 | Throughput/latency sweep, ISL/OSL 128/128, 1024/1024, 4096/512, 32768/512 × conc 1/8/16/32 | **16/16 points, 0 failed requests.** Decode ~58 tok/s/user at c=1 (17 ms TPOT); best aggregate 1710 output tok/s (1024/1024, c=32); 32K prompts prefill in ~5.9 s |
| 2 | GPQA Diamond, all 198, thinking on, 32K cap | **78.8 % (156/198)** first run; 8 no-answer (7 truncated at 32K, 1 blank). 3-run mean 80.5 % (SD 1.8); bf16 run 81.8 %. Gap to the card's 86.4 is explained/bounded in §2b (noise, 32K cap, strict parsing; no TT defect) |
| 3 | Terminal-Bench (core 0.1.1), 5 tasks, Terminus-2 | **4/5 pass** (hello-world, fix-permissions, csv-to-parquet, openssl-selfsigned-cert pass; sqlite-db-truncate fail) |
| 4 | SWE-bench Verified, 5 instances, mini-swe-agent + official docker eval | **5/5 resolved** |

No model-code fixes were needed for correctness. For the §2b precision A/B there is one local opt-in commit, tt-metal `3c4ed3f5` (`ORNITH_PRECISION_CONFIG`; default unchanged). Fixes made along the way were all in harnesses or config: swebench 5.x → 4.1.0 for the evaluation, and the dead-upstream `fix-git` TB task was replaced.

## Server (the pipeline's final serving config, stage 10/11)

`~/ornith-9b/evals/server/launch_server.sh`. It is the same command the stage-11 benchmark used, except it has no 6 h `timeout` wrapper:

```
python -m readiness_check.run_vllm_server --model-dir models/autoports/ornith_ai_ornith_1_5_9b \
  --hf-model ornith-ai/Ornith-1.5-9B --mesh-device P300x2 --max-model-len 262144 --block-size 32 --max-num-seqs 32 --port 8000 \
  --tt-config '{"sample_on_device_mode":"all","trace_region_size":200000000,"fabric_config":"FABRIC_1D","trace_mode":"decode_only","decode_interleave_prefill_steps":1,"decode_interleave_decode_steps":2}' \
  --additional-server-args "--max-num-batched-tokens 4096 --language-model-only --async-scheduling --revision 489cb979... --tokenizer-revision 489cb979... \
     --default-chat-template-kwargs '{\"enable_thinking\":false}' --override-generation-config '{\"top_k\":20}' \
     --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder"
# env: TT_MODEL_CLASS_OVERRIDES=TTQwen3_5ForConditionalGeneration=models.autoports.ornith_ai_ornith_1_5_9b.tt.generator_vllm:OrnithForCausalLM
```

Server settings: all 32 layers, bfp8_all_lofi weights, context 262144 tokens, 32 slots, chunked-prefill budget 4096, sampling on device, decode traces.
Prefix caching is off (the model's recurrent state can't be rebuilt from KV blocks). tt-metal is at `eacb23f6`, the vLLM TT plugin at `2a150455`,
and vLLM is 0.31.0. If a request omits `top_k`, the server uses 20 (approved by Vladan). The server default is `enable_thinking=false`, so every eval
below turns thinking on explicitly with `chat_template_kwargs`.

## 1. Throughput/latency sweep

Harness: the stage-11 client `tools/benchmark_client.py`, which wraps upstream `vllm bench serve` 0.26.0 and feeds exact random token-ID prompts
through `/v1/completions`. Settings: `--ignore-eos`, greedy (temperature 0), `--num-warmups <conc>`, request rate inf. Number of prompts: 4×conc
for 128/128, otherwise 2×conc, with a minimum of 4. Script: `~/ornith-9b/evals/bench/run_sweep.sh`.
Raw data: `~/ornith-9b/evals/bench/results/*.json|.log`. Run on the idle server, 2026-11-14 11:02–11:42 UTC.
**All 16 points ran. None were skipped:** the context contract serves up to 262144 tokens, so 32768+512 is well inside it. 0 failed requests in total.

| ISL/OSL | Conc | Reqs (ok/fail) | TTFT mean / p50 / p99 (ms) | TPOT mean / p99 (ms) | ITL p50 / p99 (ms) | tok/s/user (1000/TPOT) | Output tok/s total | Total tok/s (in+out) | Duration (s) |
|---|---:|---|---|---|---|---:|---:|---:|---:|
| 128/128 | 1 | 8/0 | 272 / 271 / 279 | 17.08 / 17.08 | 17.07 / 17.67 | 58.6 | 52.4 | 104.9 | 19.5 |
| 128/128 | 8 | 32/0 | 1043 / 1088 / 1103 | 15.82 / 15.83 | 15.82 / 16.40 | 63.2 | 335.4 | 670.7 | 12.2 |
| 128/128 | 16 | 64/0 | 1927 / 2007 / 2024 | 14.41 / 14.41 | 14.40 / 15.31 | 69.4 | 545.1 | 1090.2 | 15.0 |
| 128/128 | 32 | 128/0 | 3649 / 3757 / 3884 | 11.16 / 11.18 | 11.15 / 12.14 | 89.6 | 808.3 | 1616.6 | 20.3 |
| 1024/1024 | 1 | 4/0 | 219 / 220 / 234 | 17.25 / 17.25 | 17.25 / 18.04 | 58.0 | 57.3 | 114.6 | 71.5 |
| 1024/1024 | 8 | 16/0 | 957 / 1065 / 1154 | 16.59 / 16.67 | 16.01 / 16.83 | 60.3 | 455.2 | 910.4 | 36.0 |
| 1024/1024 | 16 | 32/0 | 1276 / 1102 / 2382 | 16.55 / 17.03 | 14.69 / 15.74 | 60.4 | 888.6 | 1777.2 | 36.9 |
| 1024/1024 | 32 | 64/0 | 1936 / 1212 / 4837 | 16.38 / 17.89 | 11.78 / 258.13 | 61.1 | 1710.2 | 3420.5 | 38.3 |
| 4096/512 | 1 | 4/0 | 413 / 410 / 454 | 17.56 / 17.56 | 17.55 / 18.21 | 57.0 | 54.6 | 491.0 | 37.5 |
| 4096/512 | 8 | 16/0 | 1603 / 883 / 4332 | 23.14 / 25.25 | 16.35 / 315.57 | 43.2 | 299.2 | 2692.5 | 27.4 |
| 4096/512 | 16 | 32/0 | 2754 / 896 / 8811 | 29.82 / 34.44 | 15.12 / 335.51 | 33.5 | 441.0 | 3968.6 | 37.2 |
| 4096/512 | 32 | 64/0 | 5046 / 919 / 17797 | 43.19 / 52.87 | 12.77 / 629.96 | 23.2 | 581.1 | 5229.8 | 56.4 |
| 32768/512 | 1 | 4/0 | 5920 / 5915 / 5947 | 21.39 / 21.39 | 21.41 / 22.24 | 46.8 | 30.4 | 1975.2 | 67.4 |
| 32768/512 | 8 | 16/0 | 23090 / 18239 / 50686 | 101.60 / 261.91 | 20.65 / 945.79 | 9.8 | 51.1 | 3323.0 | 160.2 |
| 32768/512 | 16 | 32/0 | 84032 / 98760 / 142301 | 113.22 / 390.68 | 20.67 / 962.26 | 8.8 | 50.7 | 3293.4 | 323.4 |
| 32768/512 | 32 | 64/0 | 208119 / 261199 / 307610 | 119.38 / 489.86 | 20.65 / 965.69 | 8.4 | 50.4 | 3277.1 | 649.9 |

Notes (plain language):
- **tok/s/user** = 1000 / mean TPOT, which is the decode speed one user sees after the first token. **Output tok/s total** is the aggregate
  over the whole run, including prefill time.
- **Concurrency 1 on this 32-slot server is about 17 ms/token (~58 tok/s/user).** Stage 11 measured the same 17.08 ms for one request on a 32-slot server.
  On a dedicated `--max-num-seqs 1` server, stage 11 measured 8.92 ms/token (~112 tok/s/user, 128/128). We ran everything on the 32-slot server
  because the evals need it.
- Short prompts (128/128, 1024/1024) scale well. At c=32, 128/128 reaches ~810 output tok/s, and 1024/1024 reaches ~1710 output tok/s at ~61 tok/s/user.
- Long prompts are prefill-bound. Prefill runs one request at a time in 4096-token chunks, interleaved one prefill to two decodes.
  A 32K prompt takes ~5.9 s to prefill alone. At c≥8, new 32K prompts queue behind each other, so TTFT grows to tens or hundreds of seconds
  and the mean TPOT includes the decode stalls during other users' prefills (p99 ITL ~0.95 s). Throughput for 32768/512 levels off at
  ~3300 total tok/s (input+output), ~50 output tok/s.

## 2. GPQA Diamond: full 198 questions, reasoning mode

| Metric | Value |
|---|---|
| **Accuracy** | **156 / 198 = 78.8 %** (single run, 1 sample per question) |
| No answer extracted | 8 (4.0 %), all scored wrong |
| …of which truncated at the 32768-token cap (`finish_reason=length`) | 7 |
| …of which blank final answer after a normal stop | 1 (#98: hit EOS while still in reasoning; the reasoning concluded "B", gold is A) |
| Completion tokens mean / median / max | 5962 / 3027 / 32768 (1.18 M total) |
| Wall time | ~28 min at concurrency 32 (13:09–13:37 UTC) |

- Harness: `~/ornith-9b/evals/gpqa/run_gpqa.py`, simple-evals style. It uses the openai/simple-evals GPQA prompt ("…last line … 'Answer: $LETTER' … Think step by step"),
  shuffles choices per question with `random.Random(0)`, and extracts answers with the simple-evals regex `(?i)Answer[ \t]*:[ \t]*\$?([A-D])\$?` on the final
  content (thinking excluded). Data: `Idavidrein/gpqa` `gpqa_diamond.csv` (198 rows; the gate was accessible this time).
- Sampling (model card, general): temperature 1.0, top_p 0.95, top_k 20, min_p 0, presence_penalty 1.5, max_tokens 32768,
  `chat_template_kwargs={"enable_thinking": true}`. Requests go to `/v1/chat/completions`, and the server's qwen3 reasoning parser separates thinking from the answer.
- Command: `benchmark-clients/eval/bin/python -I gpqa/run_gpqa.py --csv <hf>/gpqa_diamond.csv --concurrency 32`
  Per-question records, including full reasoning, are in `gpqa/responses.jsonl`.
- I checked the 7 truncated answers: they are genuine long chemistry/biology reasoning that hit the cap, not repetition loops (no repeated
  200-char chunks in the last 20K chars). With a larger budget some might be answered.
- Context: the model card reports 86.4 % (average of 5 runs, recipe not published). Ours is one run under a 32K cap: 7.6 points lower, and ~3.5 points of that gap
  is the 7 truncations. One run of 198 questions has a standard error of ~2.9 points.

## 2b. GPQA gap investigation: 78.8 % (ours) vs 86.4 % (model card)

**Conclusion: the gap is bounded and explained without any TT correctness defect.** About 4 points come from things we can measure:
run-to-run noise (run 1 was our unluckiest run), the 32K output cap, and strict answer extraction.
BF16 weights recover another ~1.4 points (not significant). The remaining ~2.5 points are within the combined sampling error of our runs and the
card's 5-run average, plus the card's unpublished recipe (prompt, scorer, token budget). TT next-token logits match the HF fp32 CPU reference closely.

| Step | Finding | Effect on gap |
|---|---|---|
| Original run 1 (bfp8, 32K cap, strict regex) | 78.79 % (156/198) | gap 7.6 pts |
| (3) Run-to-run noise: 2 more full repeats, same server/settings | runs **78.79 / 80.30 / 82.32**, mean **80.47 %**, SD 1.77 (SE of mean ~1.0). Questions right 3/3: 135, 2/3: 29, 1/3: 15, 0/3: 19 | **−1.7** (run 1 was the lowest) |
| (1a) 32K cap: run 1's 8 no-answers rerun with a 131072 cap | 7 of 8 then finished, all within 128K (18K–87K tokens). **4/8 now correct** (#98, #185, #69, #93), 4 wrong. Truncations per run: 7 / 4 / 8; ~3/7 resolve correctly with a larger cap | **≈ −1.4** (expected per run) |
| (1b) Answer extraction (simple-evals strict `Answer: X`) | Strict regex misses: 1 + 5 + 3 answers across the 3 runs. 8 are "EOS inside `<think>`" (blank content, answer only in reasoning), and 1 is a plain format miss (rep2 #158: "## Answer\n\nB) 8 Gpc"). A lenient parse of those recovers 0 / +3 / +1 | **≈ −0.7** |
| (2) Prompt/sampling vs model card | The card publishes no GPQA recipe. It says only "averaged over five runs" and gives general sampling T=1.0, top_p=0.95, top_k=20, min_p=0, presence_penalty=1.5, rep_penalty=1.0. **Ours matches exactly**, with thinking on, the repo's own chat_template (`<|im_start|>assistant\n<think>\n`), and the qwen3 reasoning parser. Device presence_penalty checked as working: greedy 120-word story, pp 0 vs 2.0 → different text, 100 vs 107 unique words; pp 0 repeats exactly. The card's max tokens, prompt and scorer are unknown | not separable; part of the residual |
| (4) Precision: stage-8 `bf16_hifi4` policy (BF16 weights + HiFi4 for all 5 groups the sweep lowered to bfp8/LoFi; BF16 KV/CCL), one full run | **81.82 % (162/198)**; no-answer 13 (9 cap, 4 EOS-in-think). Paired vs the bfp8 3-run per-question average: **+1.35 pts, SE 2.1 (not significant)** | ≈ −1.4 ± 2.1 |
| (5) TT vs HF fp32 CPU logits, same token prefixes | 8 questions (6 wrong-answer + 2 correct controls), 66 teacher-forced positions up to 4K tokens into the TT reasoning. **bfp8: top-1 match 63/66**; the 3 mismatches are near-ties (HF top-2 within 0.01–0.3 nats); KL(HF‖TT, top-20) mean 0.022 (max 0.56); mean |Δlogprob| of the chosen token 0.022. **bf16: top-1 65/66**, KL 0.009, mean |Δlogprob| 0.008. Wrong-answer questions show no special divergence: the model makes the same next-token choices as HF | none, no TT defect |
| Residual | 86.4 − (80.5 + 1.4 + 0.7) ≈ 3.9 pts with bfp8, ≈ 2.5 pts with bf16. Combined sampling SE (our 3–4 runs vs the card's 5) is ~1.3–1.6 pts, so the residual is ~1.5–2.5 σ, plus the unknown recipe differences | bounded |

EOS-inside-thinking check (the 8 blank answers across the 3 runs): at the exact final position, TT and HF give EOS
(`<|im_end|>`/`<|endoftext|>`) similar low probability, e^-4 to e^-12 (typically within 1–2 nats; TT tends higher). Both give `</think>` even less
(HF e^-11.6 to e^-17.7). So the model itself sometimes ends while still thinking. A plausible contributor is presence_penalty=1.5, which lowers every token
already used but not the never-used EOS, so a rare sample becomes likelier. These positions were chosen because TT sampled EOS there,
so TT>HF at those points is expected selection bias, not evidence of a systematic shift. Files: `gpqa/eos_check_{hf,tt_bfp8,tt_bf16}.json`.

All GPQA runs (same 198 questions and prompt, sampling as in section 2):

| Run | Precision | Cap | Accuracy | No answer (cap / EOS-in-think / format) | Mean tokens |
|---|---|---|---|---|---|
| run 1 (`responses.jsonl`) | bfp8_all_lofi (production) | 32K | 78.79 % | 8 (7 / 1 / 0) | 5962 |
| rep 2 (`responses_rep2.jsonl`) | bfp8_all_lofi | 32K | 80.30 % | 9 (4 / 4 / 1) | 5527 |
| rep 3 (`responses_rep3.jsonl`) | bfp8_all_lofi | 32K | 82.32 % | 11 (8 / 3 / 0) | 6153 |
| bf16 (`responses_bf16.jsonl`) | bf16_hifi4 | 32K | 81.82 % | 13 (9 / 4 / 0) | 6032 |
| run 1 no-answers only (`responses_cap128k.jsonl`) | bfp8_all_lofi | 128K | 4/8 correct | 0 cap hits (max 86.6K tokens) | 45K |
| **Mean of the 4 full runs** | | | **80.8 %** (SD 1.6) | | |

Our best like-for-like estimate of what this TT deployment would score under a card-like recipe (larger budget, lenient answer parsing):
**~82.5 % (bfp8)** to **~84 % (bf16)**, vs the card's 86.4 %.

Recommendation: no model fix is needed. For reporting, use multiple runs (≥3), a ≥128K generation budget, and parse the answer from the reasoning when the
content is blank. If accuracy is the priority over throughput, bf16_hifi4 tracks HF more tightly (KL 0.009 vs 0.022), but its accuracy gain is
within noise at n=198.

How it was done:
- Repeats: `gpqa/run_gpqa.py --concurrency 16 --out gpqa/responses_rep{2,3}.jsonl` (two at once). 128K rerun: `--max-tokens 131072 --only 98,69,79,93,94,127,147,185`.
  Scoring: `python3 gpqa/score.py`.
- BF16: same launcher with `ORNITH_PRECISION_CONFIG=evals/server/bf16_hifi4_policy.json` (the stage-8 candidate policy from
  `doc/context_contract.json` → `datatype_sweep_candidates.bf16_hifi4`, which passed capacity at 262144). The server log confirms `PRECISION_POLICY … "id": "bf16_hifi4"`.
  **Model-code change:** tt-metal local commit `3c4ed3f5` "Ornith: opt-in ORNITH_PRECISION_CONFIG override for precision policy". Unset keeps
  the production bfp8_all_lofi policy, so default behavior is unchanged.
- Logits: `gpqa/logit_compare.py hf` (transformers 5.12.1, fp32, CPU; ids = chat template + TT reasoning re-tokenized; positions at reasoning offsets
  0…4095) then `gpqa/logit_compare.py tt` (`/v1/completions`, token-id prompt, max_tokens 1, temperature 0, logprobs 20; TT does not support prompt_logprobs).
  Results: `gpqa/logit_compare_{bfp8,bf16}.json`. EOS check: `gpqa/eos_check.py {hf,tt}`.

## 3. Terminal-Bench: 5 tasks, Terminus-2 agent

| Task (terminal-bench-core 0.1.1) | Difficulty | Agent episodes | Tests | Result |
|---|---|---:|---|---|
| hello-world | easy | 2 | 2/2 passed | **Pass** |
| fix-permissions | easy | 5 | 1/1 passed | **Pass** |
| csv-to-parquet | easy | 15 | 2/2 passed | **Pass** |
| openssl-selfsigned-cert | medium | 12 | 6/6 passed | **Pass** (agent hit the 360 s limit after finishing the work; tests then passed) |
| sqlite-db-truncate | medium | 5 | 0/1 passed (`test_json_data`) | **Fail** (agent timeout at 360 s, data not recovered) |

**4 / 5 passed** (3/3 easy, 1/2 medium).
- Task substitution: the original 5th task was `fix-git` (medium). Its Docker image cannot be built anymore, because `setup.sh` clones
  `github.com/TheMikeMerrill/personal-site`, which now returns "Repository not found". tb reported it as `unknown_agent_error` before the
  agent ever ran. This is dataset rot, not a model result. I replaced it with `sqlite-db-truncate` (medium, no external fetch), run separately as
  run-id `ornith5-replacement`.
- Harness: `terminal-bench` 0.2.18 (`tb`), dataset `terminal-bench-core==0.1.1` (downloaded to `tbench/dataset`), stock **Terminus-2**
  agent (JSON parser), default per-task timeouts (agent 360 s, tests 60 s), 1 attempt per task. The wrapper `tbench/ornith_agent.py` only
  points litellm at the local server (`openai/ornith-ai/Ornith-1.5-9B`, `http://127.0.0.1:8000/v1`), sets sampling, and reports the
  server's 262144 context to Terminus. It does not change the prompts or logic.
- Sampling (model card, coding): temperature 0.6, top_p 0.95, top_k 20, min_p 0, presence_penalty 0.0, max_tokens 32768, thinking on.
  I confirmed these in the agent's litellm debug logs.
- Commands: `tbench/run_tb.sh` (`tb run --dataset-path ./dataset --agent-import-path ornith_agent:OrnithTerminus2 --model openai/ornith-ai/Ornith-1.5-9B
  --n-concurrent 5 -t hello-world -t fix-permissions -t csv-to-parquet -t openssl-selfsigned-cert -t sqlite-db-truncate`). The first four ran in
  run `tbench/runs/ornith5`, the replacement in `tbench/runs/ornith5-replacement`.
- Note: the main TB run overlapped with the tail of GPQA and the SWE agents on the same server, so agent steps were slower than on an idle server.
  With thinking on and a 360 s wall-clock budget, step latency matters: both medium tasks hit the agent timeout.
  A `hello-world` smoke run beforehand (`tbench/smoke`, passed) only checked the plumbing and is not counted.

## 4. SWE-bench Verified: 5 instances, mini-swe-agent + official SWE-bench docker evaluation

| Instance | Difficulty (dataset) | Agent steps | Patch submitted | FAIL_TO_PASS | PASS_TO_PASS | Result |
|---|---|---:|---|---|---|---|
| django__django-11099 | <15 min | 21 | yes | 3/3 | 19/19 | **Resolved** |
| astropy__astropy-14995 | <15 min | 34 | yes | 1/1 | 179/179 | **Resolved** |
| pytest-dev__pytest-7432 | <15 min | 27 | yes | 1/1 | 77/77 | **Resolved** |
| scikit-learn__scikit-learn-14141 | <15 min | 16 | yes | 1/1 | 2/2 | **Resolved** |
| pydata__xarray-4629 | <15 min | 19 | yes | 1/1 | 32/32 | **Resolved** |

**5 / 5 resolved.** Instances: 5 easy (`<15 min fix`) Verified instances from 5 different repos, chosen before the run and not based on outcome.
- Agent: mini-swe-agent 2.4.6 with its stock `benchmarks/swebench.yaml` (tool-calling bash agent). The overrides are in `swebench/ornith_local.yaml`:
  model `hosted_vllm/ornith-ai/Ornith-1.5-9B` @ `http://127.0.0.1:8000/v1`, step_limit 150, no cost limit, 300 s command timeout.
  Tool calls are parsed server-side by vLLM's `qwen3_coder` parser. Sampling (model card, coding): temperature 0.6, top_p 0.95, top_k 20, min_p 0,
  presence_penalty 0.0, max_tokens 32768, thinking on.
- Agent command: `swebench/run_agent.sh` (`mini-extra swebench --subset verified --split test --filter '^(…5 ids…)$' -c swebench.yaml -c ornith_local.yaml -w 5`).
  Trajectories and predictions are in `swebench/agent_out/`.
- Evaluation: the official harness `swebench==4.1.0` `python -m swebench.harness.run_evaluation --dataset_name princeton-nlp/SWE-bench_Verified
  --split test --predictions_path agent_out/preds.json --instance_ids … --namespace swebench --run_id ornith_local` (prebuilt official
  `swebench/sweb.eval.x86_64.*` images). Report: `swebench/hosted_vllm__ornith-ai__Ornith-1.5-9B.ornith_local.json`; per-instance logs are in
  `swebench/logs/run_evaluation/ornith_local/`.
  Harness fix: the newer `swebench` 5.0.2 that pip installs alongside mini-swe-agent fails with `KeyError: 'image'` on the princeton-nlp dataset
  (it expects a new dataset format). I evaluated with the classic 4.1.0 harness in a separate venv (`venvs/sweeval`).
- Caveat: 5 easy instances is a smoke-level sample, not a SWE-bench score.

## 5. Full Terminal-Bench: terminal-bench-core 0.1.1, all 80 tasks, Terminus-2 (2026-11-18)

| | Result |
|---|---|
| Passed / attempted | **25 / 77 = 32.5 %** |
| Excluded (environment cannot be built) | 3: fix-git, qemu-alpine-ssh, qemu-startup |
| By difficulty | easy 9/12, medium 13/41, hard 3/24 |
| Agent timeouts | 53 / 77 (8 of these still passed the tests) |
| Wall time | 2 h 05 min (01:26 to 03:31 UTC, 8 concurrent) plus a ~5 min single-task rerun |

By category: security 7/12, debugging 5/10, system-administration 4/11, data-science 4/8, file-operations 3/8, software-engineering 2/16,
model-training 0/7, games 0/3, scientific-computing 0/2.

- Passed (25): crack-7z-hash, crack-7z-hash.easy, create-bucket, csv-to-parquet, eval-mteb, eval-mteb.hard, extract-safely, fibonacci-server, fix-pandas-version, fix-permissions, git-multibranch (timeout, tests passed), git-workflow-hack (timeout, tests passed), hello-world, incompatible-python-fasttext (timeout, tests passed), incompatible-python-fasttext.base_with_hint, modernize-fortran-build, openssl-selfsigned-cert (timeout, tests passed), password-recovery (timeout, tests passed), processing-pipeline, sanitize-git-repo (timeout, tests passed), simple-sheets-put, simple-web-scraper, sqlite-with-gcov (timeout, tests passed), swe-bench-astropy-1 (timeout, tests passed), swe-bench-langcodes.
- Failed (52): blind-maze-explorer-5x5 (timeout), blind-maze-explorer-algorithm (timeout), blind-maze-explorer-algorithm.easy (timeout), blind-maze-explorer-algorithm.hard (timeout), build-initramfs-qemu (timeout), build-linux-kernel-qemu (timeout), build-tcc-qemu (timeout), cartpole-rl-training (timeout), chess-best-move (timeout), conda-env-conflict-resolution (timeout), configure-git-webserver (timeout), count-dataset-tokens (timeout), crack-7z-hash.hard (timeout), cron-broken-network (test output unparseable), decommissioning-service-with-sensitive-data (timeout), download-youtube, extract-moves-from-video (timeout), get-bitcoin-nodes (test output unparseable), gpt2-codegolf (timeout), grid-pattern-transform (test output unparseable), heterogeneous-dates, hf-model-inference (timeout), intrusion-detection (timeout), jupyter-notebook-server (timeout), new-encrypt-command (timeout), nginx-request-logging (timeout), oom, organization-json-generator, path-tracing (timeout), path-tracing-reverse (timeout), play-zork (timeout), polyglot-c-py (timeout), polyglot-rust-c (timeout), prove-plus-comm (timeout), pytorch-model-cli (timeout), pytorch-model-cli.easy (timeout), pytorch-model-cli.hard (timeout), raman-fitting (timeout), raman-fitting.easy (timeout), reshard-c4-data (timeout), run-pdp11-code (timeout), sanitize-git-repo.hard (timeout), security-vulhub-minio (timeout), solana-data (timeout), sqlite-db-truncate (timeout), super-benchmark-upet (timeout), swe-bench-astropy-2 (timeout), swe-bench-fsspec (timeout), tmux-advanced-workflow (timeout), train-fasttext (timeout), vim-terminal-task (timeout), write-compressor (timeout).

- Setup is the same as §3: `terminal-bench` 0.2.18, dataset `terminal-bench-core==0.1.1` (`tbench/dataset`), stock Terminus-2 via
  `tbench/ornith_agent.py`, model-card coding sampling (T 0.6, top_p 0.95, top_k 20, min_p 0, presence 0, max_tokens 32768, thinking on),
  stock per-task timeouts (agent 180-2400 s, most 360 s; not changed), 1 attempt per task. Server: the same bfp8 stage-10/11 config, idle apart from this run.
- Command: `tbench/run_tb_full.sh` (`tb run --dataset-path ./dataset --agent-import-path ornith_agent:OrnithTerminus2 --model openai/ornith-ai/Ornith-1.5-9B
  --run-id full --n-concurrent 8`). Results: `tbench/runs/full/results.json`.
- Concurrency 8: peak KV usage was 67 % and the server logged no preemptions.
- Harness fix: the wrapper maps `content=None` (the model emitted EOS inside `<think>`) to `""`. Without it, stock Terminus-2 crashes on
  `write_text(None)`. The first attempt was aborted after ~30 min and the run restarted from scratch with the fix (see PIPELINE-NOTES.md).
- Exclusions: fix-git (its upstream GitHub repo was deleted), qemu-alpine-ssh and qemu-startup (pinned Debian bullseye-security packages now 404).
  All three were rebuilt by hand to confirm the failure is permanent. fix-pandas-version failed to build once (transient pip error) and was rerun alone (`tbench/runs/full-rerun-buildfail`, passed).
- Main limit: the wall-clock agent timeout. Without prefix caching every step re-prefills the whole conversation, and the model reasons at length,
  so most medium and hard tasks run out of time before they finish. Pass rates under these stock time budgets are therefore partly a serving-speed result, not only a model-quality result.
- 3 failures are `parse_error`: tb could not parse the test output (tests hung or timed out). That is stock tb scoring, counted as failed.
