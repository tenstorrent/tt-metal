# GPU reference for Qwen3.6-27B (research, no device access)

## 1. GPU reference numbers
Official model card (local copy /home/runara/models/Qwen3.6-27B/README.md, table ~lines 125-260; identical to https://huggingface.co/Qwen/Qwen3.6-27B). Column order from header: Qwen3.6-27B is the LAST column.

| Benchmark | Qwen3.6-27B score | Source | Protocol stated |
|---|---|---|---|
| GPQA Diamond | **87.8** | README.md (GPQA Diamond row, last col); FP8 card https://huggingface.co/Qwen/Qwen3.6-27B-FP8 also shows 87.8 | Thinking mode presumed. Per-benchmark sample count, prompt template and hardware NOT stated for GPQA. Only general guidance: README.md:635 (thinking general: temp 1.0, top_p 0.95, top_k 20, min_p 0, presence_penalty 0, rep_penalty 1.0), README.md:984 (32,768 max output tokens typical; 81,920 for hard math/code). Example client uses max_tokens=81920 (README.md:660). generation_config.json: temp 1.0, top_k 20, top_p 0.95 |
| MMLU-Pro | 86.2 | README.md | same; no details |
| MMLU-Redux | 93.5 | README.md | same; no details |
| SuperGPQA | 66.0 | README.md | same |
| LiveCodeBench v6 | 83.9 | README.md | same |
| AIME26 | 94.1 | README.md; footnote line 276: full AIME 2026 I & II | no sample count |
| HMMT Feb 26 | 84.3 | README.md | |
| Terminal-Bench 2.0 | 59.3 | README.md:271 | temp 1.0, top_p 0.95, top_k 20, max_tokens 80K, 256K ctx, avg of 5 runs, Terminus-2 |
| SWE-bench Verified | 77.2 | README.md:270 | temp 1.0, top_p 0.95, 200K ctx |
| GSM8K | NOT REPORTED | - | no official number |
| IFEval | NOT in table (not seen) | - | |

The README footnotes give no "avg@k" for knowledge/STEM benchmarks, no prompt template, and no hardware or precision (BF16 on GPU is INFERRED). Qwen blog https://qwen.ai/blog?id=qwen3.6-27b is JS-rendered; WebFetch returned nothing, so no extra protocol detail was obtained from it.

### tt-inference-server (/home/ttuser/atupe/tt-inference-server/reference_config/evals/eval_config.py)
- Qwen3.6-27B entry (line 2016): only terminal_bench_2: published 59.3 (:2022), gpu_reference_score 53.9 (:2024, ref issue #3359 comment). swe_bench_verified (commented out, ~:2080): published 77.2, gpu_reference 62.0. NO GPQA/MMLU entry for Qwen3.6-27B.
- Qwen3.8-27B entry (:6095-6150): r1_gpqa_diamond published 89.2 (ref https://huggingface.co/Qwen/Qwen3.8-27B), protocol gen_kwargs max_gen_toks 80*1024, temp 1.0, top_k 20, top_p 0.95, chat API, max_length 262144, lm_eval task r1_gpqa_diamond. This is the Qwen3.8 model, not 3.6; do not mix up.
- Qwen3-32B (:3340+): r1_gpqa_diamond published 66.80 (Artificial Analysis), gpu_reference_score 66.80 with comment "Estimate - needs to be validated", ref "TBD" (:3434-3444). Not useful.
- Nothing found for Qwen3.5.

## 2. Independent GPU evals / quantized variants
| Item | Result | Source | Notes |
|---|---|---|---|
| Qwen3.6-27B BF16 vs FP8 vs NVFP4 (Unsloth card) | GPQA: BF16 88.13, FP8 86.87, NVIDIA NVFP4 86.87, Unsloth NVFP4 86.34. MMLU-Pro: BF16 85.96, FP8 86.11, NVFP4 85.96/86.25. AIME25: BF16 93.33, FP8 93.75, NVFP4 93.12 | https://huggingface.co/unsloth/Qwen3.6-27B-NVFP4 | Sample counts and sampling NOT stated. Deltas of -1.3 to -1.8 GPQA points are within noise (SE ~2.3). GPQA 88.13 vs 86.87 differ by 2.5 q of 198 (INFERRED: 88.13%/86.87% look like avg over several repeats, since neither is k/198 exactly) |
| Qwen3.6-27B-FP8 official | "performance metrics nearly identical to original" (card shows 87.8 GPQA, 86.2 MMLU-Pro) | https://huggingface.co/Qwen/Qwen3.6-27B-FP8 | no side-by-side table |
| GGUF quants of Qwen3.6-27B | AIME-120: BF16 70.8%; Q4 and above indistinguishable; Q3_K_S 54.2%; 2-bit degrades | https://quesma.com/blog/qwen-quantization-quality/ | 4-bit is what the GPU community treats as acceptable |
| Qwen3 FP8 | "does not seem to harm" 14B-235B | search summary, https://huggingface.co/Qwen/Qwen3.6-27B-FP8 context (INFERRED, not verified at source) | |

No Artificial Analysis / vLLM / SGLang recipe with a Qwen3.6-27B GPQA number was found within this search. The recommended serving args in README.md:542-589 are for launch, not eval.

## 3. Feasibility on our side
Local tooling: /home/ttuser/atupe/qwen38_work/tier2/eval/gpqa_eval.py. Options: --base-url, --model, --csv, --out, --limit, --ids, --concurrency (32), --no-think, --temperature (default 0.6), --top-p (0.95), --top-k (20), --max-tokens (32768), --seed-offset, --shuffle-seed, --timeout, --backoff. Prompt: zero-shot "Answer: $LETTER" template, choices shuffled per record with seed, regex last-answer extraction, per-question seed = hash(record id)+seed-offset. Resumable (--out JSONL).

IMPORTANT: default temperature 0.6 does NOT match the model-card thinking-general setting (1.0). Use --temperature 1.0.

| Benchmark | GPU ref | Matching settings | Runtime (INFERRED from /home/ttuser/atupe/qwen38_work/tier2/eval_plan.md:18-20: 8-20k tok/q, 400-430 tok/s agg at conc 32) | SE (single run) |
|---|---|---|---|---|
| GPQA-Diamond 198 q | 87.8 (official) / 88.13 (BF16 Unsloth run) | `gpqa_eval.py --temperature 1.0 --top-p 0.95 --top-k 20 --max-tokens 32768 (81920 if time allows; official says 32k typical, 80k for hard) --concurrency 32`, thinking ON, vary `--seed-offset` per repeat | 1.0-2.7 h per repeat, realistic 2-3 h with long tail | sqrt(.88*.12/198)=2.3 pts |
| MMLU (cais/mmlu, 14,042 test) | none (MMLU-Pro 86.2 / Redux 93.5 are different sets; cais/mmlu is NOT equal to either) | no matching reference; would need new script, thinking ON is token heavy | subsample only | n=1000, p=.9: 0.9 pts |
| GSM8K (1319) | none official | no reference, would be a sanity/regression check only | non-thinking short gen: ~1319 x ~400 tok = 0.5M tok = ~20-30 min; thinking much longer | p=.95: 0.6 pts |

Noise analysis (INFERRED, binomial):
- One GPQA run has ~2.3 pt SE from question sampling alone, plus within-question sampling noise at temp 1.0. The official 87.8 itself carries a similar uncertainty (n=198 unless they pooled repeats), so a gap of under ~3-4 points between any two 198-question runs is not resolvable.
- Paired design (same questions, same shuffle seed, TT vs a baseline run): SE of the difference is about sqrt(d/198) where d is discordant fraction; for d=10% that is 2.2 pts per single pair, shrinking by roughly sqrt(k) over k repeats on both arms once within-question noise dominates. To resolve a 2 pt difference at ~95% (z=1.96) you need SE_diff <= ~1.0, i.e. about 4-5 repeats per arm (about 8-14 h of TT time per arm). Unpaired vs a fixed published number cannot get below ~2.3 pts SE no matter how many repeats, since the 198-question set limit remains (it only shrinks if the reference is the same questions, which it is, so the question-set variance partly cancels, but the published number's own noise does not).

## 4. Recommendation
- Primary comparison: GPQA-Diamond, thinking ON, official thinking-general sampling (temp 1.0, top_p 0.95, top_k 20, presence 0), 32,768 max tokens (note 80K is what the Qwen3.8 config and the model card's hard-problem advice use; truncation biases down), k = 3 repeats with different --seed-offset (about 6-9 h). Report mean, per-run values, truncation rate (finish_reason=length) and a 95% CI (t-interval over the pooled 198*k graded samples, clustered by question, or simply bootstrap over questions).
- Reference: 87.8 (official, README.md) as the headline; also cite BF16 88.13 (Unsloth card). Because the official run protocol (samples, template, hardware) is undisclosed, the cleanest option is to ALSO run a BF16 GPU baseline with our same script (gpqa_eval.py against vLLM) if any GPU is available; that gives a protocol-matched, paired comparison. INFERRED: no such GPU run exists today.
- Pass criterion (honest version of ">="): PASS if the TT mean >= 87.8 minus 0 (strict reading); report as "not distinguishable from GPU" when the 95% CI of the TT mean contains 87.8 (CI half-width ~2.3-3 pts for k=3), and FAIL (significant regression) only if the CI upper bound < 87.8. Strict mean >= 87.8 is a coin flip for a model that equals the reference (SE 2.3), and for a BFP4/BFP8 model that really loses ~1 pt it would fail most of the time, so flag: GPU quantization (FP8/NVFP4) itself lands at 86.3-86.9 vs 88.1 BF16 in the Unsloth table, so a ~1-2 pt deficit is the demonstrated accepted GPU loss (INFERRED interpretation). Suggest the user either accept CI-contains-reference, or require mean >= 87.8 - 2 (GPU FP8/NVFP4 tolerance) as a practical threshold.
- Do not use cais/mmlu or GSM8K as acceptance gates (no GPU reference); use them as cheap regression sanity checks only (n=1000 MMLU subset, GSM8K full, non-thinking) comparing TT against a TT-BF16-equivalent or earlier build.


## Round 2: Qwen3.8-27B + popular benchmarks
(Research only; no device, no weights downloaded. Fetched 2026-10-08. INFERRED = my deduction.)

### Part 1: identity
- Qwen3.8-27B is a real, distinct checkpoint: https://huggingface.co/Qwen/Qwen3.8-27B, createdAt 2026-08-05, sha 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0, not gated, Apache-2.0 (HF API https://huggingface.co/api/models/Qwen/Qwen3.8-27B). It is the NEXT generation after 3.6 ("Built on the architectural foundation of Qwen3.5", README line 18 of the card), same family.
- Architecture is IDENTICAL to Qwen3.6-27B. I diffed config.json (flattened) of /home/runara/models/Qwen3.6-27B/config.json vs https://huggingface.co/Qwen/Qwen3.8-27B/raw/main/config.json: the only difference is transformers_version (4.57.1 vs 5.8.0.dev0). Same model_type qwen3_5, 64 layers, hidden 5120, inter 17408, 48 linear + 16 full attn. model.safetensors.index.json: 1199 tensor names in each, identical key sets (18 shards vs 15 shards, sharding only). So 3.8 is a new post-training of the same shape. INFERRED: our qwen36 loader can load it unchanged (same names/shapes); not tested.
- Tokenizer: BPE vocab and merges identical to 3.6 (compared tokenizer.json). Differences: Qwen3.8 adds 7 more added_tokens (<|audio_start|> 248070, <|audio_end|>, <tts_pad>, <tts_text_bos>, <tts_text_eod>, <tts_text_bos_single>, <|audio_pad|> 248076), 33 vs 26; all inside the 248,320 padded embedding. generation_config.json identical (temp 1.0, top_k 20, top_p 0.95).
- CHAT TEMPLATE DIFFERS (matters for eval): Qwen3.8 chat_template.jinja adds `reasoning_effort` (default **xhigh**; xhigh/medium/low) and injects a system message "Reasoning effort is set to xhigh. Please think carefully ..." when thinking is on; also preserve_thinking defaults true. Qwen3.6 template has none of this. Card: https://huggingface.co/Qwen/Qwen3.8-27B (reasoning_effort section ~line 258 of README). Our prefill/eval harness must render the 3.8 template to match the published protocol.
- Repo demo: models/demos/qwen38_27b_qb2 loads Qwen/Qwen3.8-27B pinned to revision 1d4bf0f2... (tt/model.py:22-23), from $MODEL_WEIGHTS_DIR (tt/model.py:26-29; README.md:39), asserts hidden/intermediate = 5120/17408 (tt/decoder.py:112). README eval results: GPQA-D 9/10 (README Evaluation section), meaningless n.
- tt-inference-server spec: hf_model_repo "Qwen/Qwen3.8-27B", reference_config/evals/eval_config.py:6095-6150: r1_gpqa_diamond published_score 89.2 (:6107, ref the HF card); CI_NIGHTLY mode reference 90.0 = "9/10 GPQA" (:6111, n=10); gen_kwargs max_gen_toks 80*1024, temp 1.0, top_k 20, top_p 0.95, chat API, max_concurrent 5, limit CI_NIGHTLY 10 samples. Also terminal_bench_2_1 published 73.0 (:6147) and a SWE-bench Verified entry (~:6205). No gpu_reference_score for GPQA (only published). Qwen3.6-27B entry (:2016) has only terminal_bench_2 (published 59.3, gpu_reference 53.9) as in round 1.
- On this machine: NO Qwen3.8-27B checkpoint. ls /home/runara/models = DeepSeek-R1-Distill-Llama-70B, DeepSeek-V2-Lite, Llama-3.1-8B, Llama-3.3-70B-Instruct, Qwen3-32B, Qwen3.6-27B, Qwen3.6-27B-BlockFP8-RTN, Qwen3.6-27B-FP8; ~/.cache/huggingface/hub has only models--Qwen--Qwen3.6-27B (no 3.8); /home/ttuser/models has Qwen3.5-27B-mesh-tp4, gpt-oss-20b. The user said our implementation loads /home/runara/models/Qwen3.6-27B, so the model under test is 3.6 weights. NOTE: the acceptance reference must then be the Qwen3.6-27B GPU numbers, NOT Qwen3.8's (3.8 is +1.4 GPQA, +6.4 LCB, +6.8 HLE better). Using 89.2 against 3.6 weights would be an unfair bar. Confirm with the user which weights are being served; if 3.8 weights are wanted they are a free ungated download (not done here).

### Part 2: GPU reference numbers
Official model cards (3.8: https://huggingface.co/Qwen/Qwen3.8-27B; 3.6: local README /home/runara/models/Qwen3.6-27B/README.md and https://huggingface.co/Qwen/Qwen3.6-27B).
IMPORTANT: the Qwen3.8 card text-benchmark table is much SMALLER than 3.6's. Text rows on 3.8 card: Terminal Bench 2.1, SWE-bench Pro, NL2Repo, DeepSWE 1.1, QwenSWEBench, CoWorkBench, JobBench, Agents' Last Exam, IFBench, GPQA Diamond, HLE, LiveCodeBench v6. It does NOT report MMLU-Pro, MMLU-Redux, SuperGPQA, C-Eval, AIME, HMMT, MATH, IFEval, BFCL, Arena-Hard. The only protocol footnotes: HLE judged by GPT-4o; agentic ones Claude Code harness temp 1.0 top_p 0.95 256K ctx; QwenSWEBench avg@3 max_tokens 32,768. No sample counts for GPQA/LCB/IFBench/HLE. Card recommends reasoning max output 262,144 tokens (line ~509).

| Benchmark | Qwen3.8-27B official | Qwen3.6-27B official | Independent GPU (3.8) | Independent GPU (3.6) |
|---|---|---|---|---|
| GPQA-Diamond | 89.2 | 87.8 | 89.93 +-0.70 (ThinkingCap card base, avg of 4-32 seeds); Vals AI 88.89; BenchLM "AA-GPQA" 90.5 | Unsloth BF16 88.13 (round 1); AA (reasoning) 84.2 |
| MMLU-Pro | not on card | 86.2 | 85.54 +-0.63 (ThinkingCap base, single seed, 12,032 q); Vals AI 84.34 | Unsloth BF16 85.96 |
| MMLU-Redux | not on card | 93.5 | none found | none |
| SuperGPQA | not on card | 66.0 | none | none |
| C-Eval | not on card | 91.4 | none | none |
| MMLU (cais) | none | none | none | none |
| AIME 2026 | not on card | 94.1 | 98.13 +-0.74 (ThinkingCap base, 4-32 seeds) | none (Unsloth AIME25 BF16 93.33) |
| AIME 2025 | not on card | not in 3.6 table | none | Unsloth BF16 93.33 |
| HMMT Feb 26 | not on card | 84.3 | 95.83 +-1.16 (ThinkingCap) | none |
| HMMT Nov 25 | not on card | 90.7 | 97.08 +-1.43 | none |
| MATH-500 | not reported | not reported | none | none |
| LiveCodeBench v6 | 90.3 | 83.9 | 91.14 +-1.11 (ThinkingCap); Vals LCB 84.0 (different set) | none |
| HLE | 30.8 (GPT-4o judge) | 24.0 | AA-HLE 33.9 (BenchLM) | AA 23.1 |
| IFBench | 79.5 | 69.1 | 79.75 +-0.63 (ThinkingCap) | AA 67.6 |
| IFEval | not reported | not reported | none | none |
| Arena-Hard, BFCL | not reported | not reported | none | none |
| HumanEval/MBPP | not reported | not reported | none | none |
| Terminal-Bench 2.1 | 73.0 | 63.4 | 75.84 +-4.26 (ThinkingCap) | - |
| SWE-bench Pro | 61.7 | 53.5 | - | - |

Sources and protocol of independent runs:
- ThinkingCap-Qwen3.8-27B card, "Base" columns (base = unmodified Qwen3.8-27B): https://huggingface.co/bottlecapai/ThinkingCap-Qwen3.8-27B. Protocol: NVIDIA H200, vLLM 0.29.0 with MTP speculative decoding (num_speculative_tokens=3), temp 1.0, top_p 0.95, top_k 20, min_p 0, reasoning_effort=xhigh, generation cap 253,952 tokens (AA-LCR 131,072; tau2 65,536), multi-seed benchmarks 4-32 runs, MMLU-Pro 12,032 questions single seed. Harness not named. This is the best protocol-documented independent GPU number set and is a third party's derivative-model card, so treat as moderately trusted. Mean thinking tokens per question for the BASE model (same table): GPQA-D 12,772; MMLU-Pro 3,725; AIME26 15,663; HMMT Feb26 23,211; HMMT Nov25 14,443; LCB v6 28,395; IFBench 7,961.
- Vals AI (https://www.vals.ai/models/alibaba_qwen3.8-27b): GPQA-D 88.89, MMLU-Pro 84.34, LiveCodeBench 84.00 (Vals's own subset), SWE-bench 86.0; temp 1, top_p 0.95, top_k 20, max output 65,536, reasoning effort xhigh.
- Artificial Analysis Qwen3.8-27B (xhigh): Intelligence Index 34, 200M tokens used (https://artificialanalysis.ai/models/qwen3-8-27b); page shows no per-benchmark GPQA. AA for 3.6 (reasoning): GPQA 84.2, HLE 23.1, IFBench 67.6 (search-result snippet of https://artificialanalysis.ai/models/qwen3-6-27b; the page I fetched did not show them, so lower confidence). AA's GPQA is lower than Qwen's own 87.8 by 3.6 points, a hint that vendor numbers are a ceiling.
- Quantized-model comparisons for 3.6 are in round 1 (Unsloth card).
- Open LLM Leaderboard / vLLM / SGLang posts with 3.6 or 3.8 eval numbers: none found.

Observation: the 3.6 BF16 GPQA from Unsloth (88.13) and Qwen's own number (87.8) agree within noise, whereas AA's 84.2 sits ~4 below. For 3.8, ThinkingCap-base 89.93 (many seeds) and vendor 89.2 agree; Vals 88.89.

### Part 3: feasibility on our side
Local: GPQA-D csv at /home/ttuser/atupe/qwen38_work/tier2/gpqa/gpqa_diamond.csv (+ the grader /home/ttuser/atupe/qwen38_work/tier2/eval/gpqa_eval.py). On box in ~/.cache/huggingface/datasets: cais___mmlu and openai___gsm8k only. Ungated HF datasets I verified through the HF API (gated=False): TIGER-Lab/MMLU-Pro (test parquet, 12,032 q), edinburgh-dawg/mmlu-redux-2.0 (3,000 q), MathArena/aime_2026 (30 q), MathArena/hmmt_feb_2026, HuggingFaceH4/aime_2024, opencompass/AIME2025, HuggingFaceH4/MATH-500, google/IFEval (541), m-a-p/SuperGPQA (26k q), livecodebench/code_generation_lite, openai/gsm8k. Gated (auto): cais/hle, Idavidrein/gpqa (we already have the csv). allenai/IFBench_test (verified gated=False, train parquet). A download needs outbound internet from this box, which curl showed works (not token-gated).

Throughput basis: 300 tok/s aggregate (user) and thinking tokens/question from ThinkingCap base (above). Time = questions x tokens / 300. At 32k max_tokens the tail is truncated, which biases down; GPQA mean 12.8k suggests only a few percent of samples exceed 32k (INFERRED), LCB (28k mean) and HMMT (23k mean) would be hurt much more. Conc 16 at 32k tokens each is the KV limit, and 16 x ~19 tok/s = ~300 tok/s.

| Benchmark | Q count | Grading | Mean think tok/q (3.8 base; 3.6 INFERRED similar or lower) | Tokens per run | TT time per run @300 tok/s | SE of a single run |
|---|---|---|---|---|---|---|
| GPQA-D | 198 | local letter regex (exists) | 12.8k | 2.53M | 2.3 h | 2.3 pts (p=.88) |
| MMLU-Pro (500 stratified subset) | 500 | local letter regex (10 options, A-J) | 3.7k | 1.86M | 1.7 h | 1.6 pts (p=.85) |
| MMLU-Pro (full 12,032) | 12,032 | same | 3.7k | 44.8M | 41 h | 0.3 pts, not feasible |
| AIME 2026 (30 q, 8 repeats) | 30 x 8 | local exact integer match (boxed) | 15.7k | 3.8M for 8 | 3.5 h for 8 repeats (26 min each) | per-run SE ~ 2.6-4 pts; mean of 8 about 1.2 |
| HMMT Feb 26 | ~30-33 | exact-answer match; some answers are expressions (needs normalizer or sympy) | 23.2k | 0.7M | 40 min per repeat | large (~6 pts per run) |
| LiveCodeBench v6 | ~450-1055 depending on window | needs sandboxed code execution with hidden tests | 28.4k | >=10M | >=9 h | not feasible (no judge/sandbox, long tail) |
| IFBench (79.5 official) | 300 | rule checker in allenai/IFBench repo (local python, no LLM) | 8.0k | 2.4M | 2.2 h | 2.3 pts (p=.8) |
| IFEval | 541 | google rule checker (local) | ~1-2k thinking INFERRED | ~0.8M | ~45 min | 1.7 pts; but NO GPU reference exists for either model |
| HLE | 2,500 | LLM judge (GPT-4o), gated | - | - | not feasible | |
| Arena-Hard | 500 | LLM judge | - | - | not feasible | |
| BFCL | many | AST checker, needs tool-call plumbing | - | - | not recommended | |
| MMLU (cais), GSM8K | 14k/1319 | local | - | - | no GPU reference; sanity only | |

Noise: resolving ~1.5 pts needs SE_diff <= ~0.75-0.8. For GPQA (n=198) paired TT-vs-GPU would need ~6-8 repeats per arm (>14 h per arm); realistically GPQA can only separate a ~3-4 pt gap with 3 repeats. MMLU-Pro: 1500 q x 1 run gives SE about 0.9 for p=.85 and costs 5.2 h; a paired comparison with a GPU run on the same questions is better (discordance ~8% gives SE_diff ~0.7 at n=1500). The official numbers have their own noise; ThinkingCap's +-0.70 on GPQA is across many seeds, so the truth for the base model is roughly 89.9 +-0.7 and the 198-question set error dominates anything we run.

### RECOMMENDATION
Decide first which weights: the box has only Qwen3.6-27B weights, so the pass bar is the 3.6 GPU numbers (compare 3.8 numbers only if we load 3.8, which has an identical architecture so the same code should run; use the 3.8 chat template with reasoning_effort xhigh).

Pick 4 (all locally gradeable, ungated or already here):
| # | Benchmark | Reference vs Qwen3.6-27B (and 3.8) | Protocol to match | TT runtime per run | Pass criterion |
|---|---|---|---|---|---|
| 1 | GPQA-Diamond (198, local csv) | 3.6: 87.8 official, 88.13 Unsloth BF16 (AA 84.2). 3.8: 89.2 / 89.93 (ThinkingCap) | thinking ON, temp 1.0, top_p 0.95, top_k 20, min_p 0, presence 0; max_tokens 32k (80k if time; tt-inference-server uses 80k); zero-shot "Answer: X", shuffle choices; use gpqa_eval.py --temperature 1.0 (default 0.6 is wrong) | 2.3 h per repeat (more at 80k); do 3 repeats | PASS if 3-run pooled mean >= ref - 2 (documented GPU quantization tolerance, Unsloth FP8/NVFP4 86.3-86.9 vs 88.1) with bootstrap CI (clustered by question) containing ref; report the strict mean>=ref as well; FAIL if the CI upper bound < ref |
| 2 | MMLU-Pro (TIGER-Lab, 1500 q random subset, fixed seed, or the 500-q one for a quick pass) | 3.6: 86.2 official, 85.96 Unsloth BF16. 3.8: 85.54 (ThinkingCap, 12,032 q), 84.34 Vals (3.8 card does not report it) | thinking ON, same sampling, 10-option letter regex | 5.2 h for 1500 q (1.7 h for 500) | PASS if mean >= ref - 1.5 and CI contains ref; the SE of 1500 q is ~0.9, so a 1.5-pt regression is detectable only at that size; a 500 subset can only catch >=3 pt |
| 3 | AIME 2026 (MathArena/aime_2026, 30 q) x 8 repeats | 3.6: 94.1 official. 3.8: 98.13 +-0.74 (32 seeds) | thinking ON, same sampling, boxed integer match; needs 32k+ cap (mean 15.7k so truncation matters; prefer 64k+ if KV allows) | 3.5 h for 8 repeats | PASS if mean over 8 >= ref - 3 (SE of mean ~1.2-1.5); a cheap reasoning-quality canary that is sensitive to long-context decode errors (the type of bug quantization/KV precision creates); report truncation rate |
| 4 | IFBench (allenai/IFBench_test, 300 q) | 3.6: 69.1 official (AA 67.6). 3.8: 79.5 / 79.75 | thinking ON, same sampling, official rule checker (strict/loose: INFERRED the card reports the prompt-level metric; unverified) | 2.2 h per run | PASS if >= ref - 2.5 (SE 2.3 single run); optional, since protocol details (which accuracy variant) are not stated |

Skip: LiveCodeBench (sandbox, 28k tokens/q, >9 h), HLE/Arena-Hard (LLM judge, HLE gated), BFCL, HMMT (cheap but 30 q, noisy; use as an extra if AIME passes), MMLU(cais)/GSM8K/IFEval (no GPU reference; sanity only). The 3.8 card itself only gives GPQA, HLE, LCB, IFBench among text benchmarks, so if the weights are 3.8, GPQA + IFBench are the only official-number comparisons and MMLU-Pro/AIME must be compared against ThinkingCap-base / Vals.

Budget: priority 1 (3 repeats ~7 h) + 2 (1500 q ~5 h) + 3 (3.5 h) = ~15.5 h of TT time; IFBench adds ~2 h. If time is short: GPQA x2 + MMLU-Pro 500 + AIME x4 = ~7.5 h.

Honest noise handling: the strict ">= GPU" is a coin flip for a faithful implementation (SE 2.3 pts on GPQA); report the mean, CI, per-run values, truncation rate (finish_reason=length) and compare with a same-script GPU baseline if any GPU can run the same harness (best option, paired).
