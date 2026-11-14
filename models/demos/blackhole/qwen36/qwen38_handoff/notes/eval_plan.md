# Tier-1 vs Tier-2 (BFP4) accuracy eval plan, Qwen3.6-27B on QB2 (P300x2)
Read-only recon, nothing run. INFERRED = not verified on this box.
TIS=/home/ttuser/atupe/tt-inference-server  (HEAD 4ac5729)

## Findings
1. Evals workflow exists: `python3 run.py --workflow evals` (run.py:296-322 has --limit-samples-mode, --eval-samples, --repeat-evals).
   Harness = lm-eval-harness, pinned commit be23028b (workflows/workflow_venvs.py:45), venv type EVALS_COMMON, built lazily under $TIS/.workflow_venvs.
   Command is built in llm_module/eval_command.py:240-470 (lm_eval --model <eval_class> --model_args model=..,base_url=..,num_concurrent=.. --gen_kwargs .. --log_samples --output_path .../eval_<model_id>; adds --apply_chat_template iff task.apply_chat_template; chat vs completions endpoint by task.use_chat_api; num_concurrent clamped to spec max_concurrency=32, llm.yaml ~1384).
2. NO text-eval task is configured for Qwen/Qwen3.6-27B. reference_config/evals/eval_config.py:2015-2090 has only terminal_bench_2 (agentic; swe_bench_verified commented out, timeouts). So `--workflow evals` for this model gives no GPQA until an EvalTask is added (a 15-line edit in eval_config.py, copying the Qwen3-8B r1_gpqa_diamond block at eval_config.py:2960-2990: EVALS_COMMON, gen_kwargs stream=true, until=[], do_sample=true, temperature, top_k=20, top_p=0.95, model_kwargs max_length). Task names in the repo for GPQA: r1_gpqa_diamond (Qwen3 family, reasoning-style, exact_match,none) and gpqa_diamond_cot_zeroshot (GLM-5.2, chat API, exact_match,flexible-extract, eval_config.py:543). The r1_* task YAMLs are not in the stock lm_eval tree (not found on box); they come with the pinned lm-eval fork commit (INFERRED).
   Qwen3.6 generation_config.json (HF snapshot): temperature 1.0, top_k 20, top_p 0.95 (matches vendor thinking-mode recommendation). Chat template: thinking is ON by default (enable_thinking only checked as `is false`). Server top_k=20 <= max_device_top_k 32 so device sampling is kept (c5_plan.md section b), no penalties allowed.
3. Local-server command with an already-running server: `python3 run.py --model Qwen/Qwen3.6-27B --tt-device p300x2 --workflow evals --server-url http://127.0.0.1 --service-port 8000 [--limit-samples-mode smoke-test]` (run.py:201-213: --server-url cannot be combined with --docker-server/--local-server). With --local-server it starts and stops the server itself (same flags as c5_plan.md plus `--workflow evals`; or `--workflow server` first as in c5_plan.md).
4. Datasets on box (Q2):
   PRESENT: ~/.cache/huggingface/datasets/cais___mmlu (57 subjects, arrow), ~/.cache/huggingface/datasets/openai___gsm8k/main/0.0.0/740312.../gsm8k-{test,train}.arrow.
   ABSENT: GPQA (Idavidrein/gpqa, gated), mmlu_pro, ifeval, aime, math. HF hub cache has only models (incl. Qwen--Qwen3.6-27B tokenizer/config files, no weights; weights at /home/runara/models/Qwen3.6-27B). /home/ttuser/gtobar has no HF cache (only scripts, test_benchmark_gpqa.py is a test, no data).
   lm_eval: NOT installed in python_env_vllm (only `datasets 2.21`, `evaluate 0.4.0`). No built venv in $TIS/.workflow_venvs (only bin/); same for ssinghal/runara copies. Only uv-cache archives exist: lm_eval 0.4.13 and 0.4.9.1 under ~/.cache/uv/archive-v0/*/ (stock tasks incl. gpqa YAMLs, gsm8k, ifeval, mmlu_pro YAMLs, but not the data). Building the EVALS_COMMON venv needs network (git clone of lm-eval commit + pip).
5. Reference scores (Q4): eval_config has no Qwen3.6 GPQA ref. The plan notes "published 89.2" (not verifiable here; INFERRED from plan) and "T3K port of BFP4 policy 81-85% on full GPQA" (QWEN38_OPTIMIZATION_PLAN.md:37). qwen38_27b_qb2 README:27 lists only GPQA Diamond 9/10 (90%), a 10-question sample, statistically meaningless. docs/model_support/llm/Qwen3.6-27B_p300x2.md has no GPQA number. No prior eval output for qwen36/38 found on the box ($W/B_acc is a token-accuracy PCC-style script, not a task eval).

## Runtime estimate (INFERRED; aggregate tput at conc 32 ~ 32 x 13 tok/s ~ 400-430 tok/s from TPOT ~75 ms; conc 1 TPOT ~35 ms)
| eval | tokens/q | total tokens | wall time |
| GPQA-D 198, thinking ON (default) | 8-20k mean, cap 32-80k | 1.6-4.0M | 1.0-2.7 h plus a long tail (one 32k generation at 13 tok/s = 42 min; 80k = 100+ min). Realistic 2-3 h at max_gen_toks=32768. Truncated thinkers score as wrong, bias downward equally in both builds. |
| GPQA-D 198, thinking OFF | 0.5-2k | 0.1-0.4M | 5-15 min |
| GSM8K 1319 thinking OFF (cached) | ~300 | 0.4M | 15-20 min (ON: 1.5-3k tok => 1.5-3 h) |
| MMLU-Pro 500-q subset OFF | ~500 | 0.25M | 10-15 min |
| IFEval 541 OFF | ~400 | 0.2M | 8-12 min |
Prefill is small (GPQA prompts ~1k tok) and negligible vs decode.
Statistical power: 198 q has SE ~3.3 pts; BFP4 degradation of 3-4 pts is borderline-detectable unpaired. Use the SAME seed and compare PAIRED per-question (log_samples) and run thinking-on at temperature 0.6-1.0 once per build (or n=2 if time).

## Recommendation
PRIMARY: GPQA-Diamond, 198 q, thinking ON, conc 32, max_gen_toks 32768, temp 1.0 top_p 0.95 top_k 20 (vendor setting), ~2-3 h per build, paired compare. Needs the GPQA dataset (missing).
FAST SMOKE (~10-20 min per build, data already present): GSM8K 1319 thinking OFF (or first 500), plus optionally MMLU (cais/mmlu, cached) 0-shot. Weak for BFP4 detection (saturated ~95%), so use it only as a sanity gate before burning 3 h on GPQA. GPQA thinking-OFF (~10 min) is a better smoke once the data is available.

## Commands
Server (per build; from c5_plan.md section a):
  cd $TIS; WT=/home/ttuser/atupe/tt-metal/.claude/worktrees/qwen38-optimizations
  HF_TOKEN=dummy TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0 python3 run.py --model Qwen/Qwen3.6-27B --tt-device p300x2 --workflow server --local-server --dev-mode --tt-metal-home $WT --tt-metal-python-venv-dir /home/ttuser/atupe/python_env_vllm --host-weights-dir /home/runara/models/Qwen3.6-27B --skip-system-sw-validation --no-auth
  (Tier-2 build: same command with the Tier-2 BFP4 env/worktree; one server at a time, QB2 has 4 chips.)
Option A, via tt-inference-server (after adding the EvalTask and building the venv with network):
  python3 run.py --model Qwen/Qwen3.6-27B --tt-device p300x2 --workflow evals --server-url http://127.0.0.1 --service-port 8000 --skip-system-sw-validation
  Smoke: add --limit-samples-mode smoke-test (1% = 2 q) ; subset: --eval-samples '{"r1_gpqa_diamond":[0,1,...]}'.
  Results: lm-eval writes {output_path}/eval_<model_id>/ (results_*.json + samples_*.jsonl via --log_samples); output_path is under $TIS/workflow_logs/ (evals_output) (INFERRED; check run log in workflow_logs/run_logs/).
  HF_HUB_OFFLINE=1 only works once GPQA is in the HF datasets cache.
Option B (credible alternative, fewer moving parts): a ~60-line script using the `openai`/`requests` client in python_env_vllm (no lm_eval needed): load GPQA-Diamond from a local parquet/CSV, shuffle choices with fixed seed (same for both builds), POST /v1/chat/completions with 32 threads, stream=false, max_tokens 32768, temp 1.0 top_p 0.95 top_k 20, parse "Answer: X"/last boxed letter, save per-question JSONL to $W/tier2/eval_out/<build>/. Gives paired per-question outputs and avoids the harness-fork dependency. Only needs the GPQA file.
  Quick offline smoke now (data present): use openai___gsm8k arrow (`datasets.load_from_disk`-style / `Dataset.from_file(".../gsm8k-test.arrow")`) with the same script, thinking off via chat_template_kwargs {"enable_thinking": false}.

## Missing and how to get it (do NOT download here)
1. GPQA-Diamond data: HF dataset Idavidrein/gpqa (config gpqa_diamond), gated; needs an HF account with accepted terms + token (HF_TOKEN currently "dummy"). Get once on a networked machine, then copy ~/.cache/huggingface/datasets/Idavidrein___gpqa (or the diamond CSV/parquet) here; set HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1.
2. lm-eval harness (Option A only): network pip/git for lm-eval commit be23028b161addd616ecabff740dee62d1d9fbd8 (EVALS_COMMON venv), or copy a built .workflow_venvs/ from another host.
3. EvalTask for Qwen/Qwen3.6-27B in reference_config/evals/eval_config.py (not present), plus its r1_gpqa_diamond YAML from the pinned lm-eval fork.
4. Optional: mmlu_pro, ifeval datasets (absent).
