# AutoFix Result: Gemma 4 31B Stage 11 Meta accuracy

## Starting Evidence

- Fresh source/artifact diagnosis:
  `.exp_run/tti-release/gemma4-31b-20260716/meta_accuracy_AUTODEBUG.md`.
- Original authoritative release artifacts: `release_cache_final6` and
  `models/autoports/google_gemma_4_31b/doc/tti_release/eval_meta_*.json`.
- Original scores: IFEval 25.181850822484343; GPQA-CoT
  20.982142857142858 (94/448).
- Exact checkpoint: `google/gemma-4-31B` revision
  `d77cb0be8ad40327cc1c6b70eff4b3f0be35bee3`, base tokenizer with no chat
  template.  No TT device or model server was used during this AutoFix.

## Hypothesis Experiments

### 1. Prompt-format mismatch caused the low scores

- **Hypothesis:** The harness incorrectly omitted an instruction/chat wrapper.
- **Experiment:** Compare runtime spec, tokenizer config, task arguments, raw
  samples, autoport checkpoint constant, and previous exact-checkpoint HF/TT
  qualitative controls.
- **Result:** The exact tokenizer has `chat_template=null`; the autoport and
  server intentionally evaluate the base repository through completions.
  Saved corpus-style continuations match prior HF base-checkpoint behavior.
- **Verdict:** Refuted.  Adding a synthetic chat wrapper or borrowing `-it`
  results would change the checkpoint contract.

### 2. GPQA's scorer accepts its own placeholder as an answer

- **Hypothesis:** The GPQA prompt's literal `The best answer is X` is captured
  by `best answer is ([A-Z])`, and last-match selection can overwrite an A-D
  answer with X.
- **Experiment:** Rescore all 448 saved responses using the shipped regex and
  the valid-choice regex `[A-D]`, preserving last-match semantics.
- **Result:** Verified.  The saved filter produced 99 X rows.  Shipped parsing
  scores 94/448 = 20.9821428571%; corrected parsing scores 118/448 =
  26.3392857143%.
- **Fix:** Restrict the generated task regex to `[A-D]`; make the cache
  validator reject stale `[A-Z]` YAML; add an echoed-placeholder regression.
- **Verification:**
  - `python3 -m pytest -q tests/test_workflow_venvs_meta.py -k 'gpqa_filter or cache_validator'`
    -> 12 passed.
  - `python3 -m pytest -q tests/test_workflow_venvs_meta.py`
    -> 23 passed.
  - `git diff --check` -> clean.
  - Compact rescore proof:
    `.exp_run/tti-release/gemma4-31b-20260716/meta_accuracy_gpqa_corrected.json`.
- **Verdict:** Verified and fixed in the TTI checkout; subsequently committed as
  `b803374e04c2460ea3bfabec4bfed832f2af532a`.

### 3. Existing evidence can provide an unwaived exact-checkpoint reference

- **Hypothesis:** A small CPU HF control would establish feasibility for the
  full canonical reference required to grade both rows.
- **Experiment 1:** Exact BF16 HF `lm_eval`, GPQA, limit 1, batch 1, same raw
  task/BOS/greedy/2,048-token contract.
- **Result 1:** Completed in 223.36 seconds.  HF and TT row 0 both select C
  (gold B), with coherent equivalent reasoning.  This supports serving-path
  behavior but one row is not an accuracy reference.
- **Experiment 2:** Exact BF16 HF GPQA, limit 4, batch 4, hard 15-minute bound.
- **Result 2:** Timed out cleanly at 904.546 seconds with 0/4 responses returned.
  Sampled peak RSS was 61.529 GiB.  No partial score artifact was emitted and
  no process remained.
- **Interpretation:** Sequential extrapolation is about 27.8 hours for the 448
  GPQA rows alone; the incomplete batch-4 result gives a consistent lower bound
  above 28.1 hours.  The additional 541 IFEval rows make a full CPU reference
  a multi-day workflow.  It is not tractable in this release pass.
- **Verdict:** Refuted as an in-stage closure path.  A GPU/HF control produced
  outside this CPU-only environment remains required.

The exact CPU commands ran from
`.workflow_venvs/.venv_evals_meta/meta_eval_gemma-4-31B_b9lic27d`:

```bash
env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  OMP_NUM_THREADS=16 ../bin/lm_eval \
  --tasks meta_gpqa_cot --model hf \
  --model_args pretrained=/home/odjuricic/.cache/huggingface/hub/models--google--gemma-4-31B/snapshots/d77cb0be8ad40327cc1c6b70eff4b3f0be35bee3,dtype=bfloat16,add_bos_token=True,max_length=113280 \
  --device cpu --batch_size 1 --limit 1 \
  --output_path /localdev/odjuricic/tt-metal/.exp_run/tti-release/gemma4-31b-20260716/tt-inference-server/.exp_run/meta_accuracy_hf_cpu/gpqa_one \
  --seed 42 --num_fewshot 0 --log_samples --show_config \
  --include_path work_dir

TIMEFORMAT='wall_seconds=%R'; time env HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=16 \
  timeout --signal=INT --kill-after=60s 900 ../bin/lm_eval \
  --tasks meta_gpqa_cot --model hf \
  --model_args pretrained=/home/odjuricic/.cache/huggingface/hub/models--google--gemma-4-31B/snapshots/d77cb0be8ad40327cc1c6b70eff4b3f0be35bee3,dtype=bfloat16,add_bos_token=True,max_length=113280 \
  --device cpu --batch_size 4 --limit 4 \
  --output_path /localdev/odjuricic/tt-metal/.exp_run/tti-release/gemma4-31b-20260716/tt-inference-server/.exp_run/meta_accuracy_hf_cpu/gpqa_four \
  --seed 42 --num_fewshot 0 --log_samples --show_config \
  --include_path work_dir
```

## Final Status

**AutoFix failed to close the mandatory unwaived accuracy gate.**

One real GPQA harness defect is fixed and proven, raising the saved-output score
to 26.3392857143%.  It does not supply a legitimate acceptance threshold.  The
remaining blocker is a canonical full-reference requirement, not a supportable
code guess:

1. run both reconstructed raw-base tasks on the exact HF revision using a
   tractable GPU environment and record the scores as the GPU reference; or
2. obtain a product-owned published threshold for these exact base prompts.

The Google model card's benchmark table is explicitly instruction-tuned and
cannot grade `google/gemma-4-31B`.  Switching to `google/gemma-4-31B-it` would
change the requested checkpoint and require a new bringup.  No waiver or
threshold was invented.  After this autofix completed, the parent workflow
regenerated the copied release report as readiness-fail and committed this
parser repair at the TTI SHA above.
