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

### 4. A faster exact local generation path can make the full control tractable

- **Workload audit:** The authoritative saved artifacts contain 497,845 IFEval
  completion tokens across 541 rows and 604,704 GPQA completion tokens across
  448 rows: 1,102,549 completion tokens total, in addition to 146,022 prompt
  tokens.  The saved completion count is a workload-size indicator rather than
  a claim that exact HF reaches EOS at the same positions.
- **Static batching experiment:** An exact BF16 batch-32 GPQA probe loaded the
  first 32 canonical prompts (8,327 prompt tokens) and attempted 128 generated
  tokens per row.  It hit the 900-second hard bound before returning the batch.
  Even crediting all 4,096 requested output tokens at the timeout boundary gives
  an end-to-end rate below 4.551 output tokens/second for this prompt/128-token
  workload and a direct same-shape projection above 67.3 hours for the saved
  completion-token count.  This is diagnostic, not a strict lower bound for
  longer generations, which amortize prefill differently.  Together with the
  earlier 2,048-token batch-4 timeout, it shows no demonstrated static-batch
  path with the required order-of-magnitude gain.  Peak sampled RSS was
  approximately 70 GiB.
- **Official Gemma 4 MTP experiment:** The official
  `google/gemma-4-31B-it-assistant` draft model produced an exact token/text
  match on GPQA row 0 (229 generated tokens including EOS; SHA-256
  `bd7dad34149ca19e0b62fc8d1b9b005bf3e6344ca02d4d8e4542d68bba495b40`),
  but took 227.121 seconds versus 223.36 seconds without MTP.  The drafter is
  trained for the instruction-tuned checkpoint, and its candidates provide no
  useful acceleration for this raw base checkpoint.
- **Prompt-lookup experiment 1:** GPQA row 0 was again token/text exact with
  the same hash.  Prompt lookup took 138.803 seconds versus 223.36 seconds, a
  1.61x speedup, which remains far from an hours-scale full control.
- **Prompt-lookup experiment 2:** A deliberately capped/high-repetition saved
  TT sample, GPQA document 111, was tested rather than extrapolating from the
  short row.  Exact ordinary HF produced 256 tokens in 237.512 seconds; prompt
  lookup produced the identical token tensor and decoded SHA-256
  `abe212897b6384b99176f11844cf2d863a03a23f5ede0e800b63a6050119e262`
  in 193.240 seconds, only 1.229x faster.  The exact HF prefix was a coherent
  cyclotron derivation, not the repeated `The best answer is A` pattern in the
  saved TT response, so TT repetition cannot predict HF lookup acceptance.
- **Source audit:** This host has 16 physical CPU cores and no CUDA or ROCm
  device.  Its installed Transformers prompt-lookup generator is explicitly
  written around batch size 1 (`input_ids[0]` and a single chosen candidate),
  so its modest batch-1 gain cannot be combined with the measured batch-32
  throughput.  The only locally installed vLLM is the TT fork, not a canonical
  CPU HF reference engine.  No exact-output-equivalent llama.cpp engine is
  installed, and changing backend or precision would cease to be the required
  exact Transformers BF16 control.  Teacher-forcing saved TT completions cannot
  recover the HF continuation after the first token divergence.
- **Verdict:** Refuted.  Both exact speculative paths preserve output, but
  neither makes the full local control tractable.

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

1. provide a GPU worker that can load the 62.5 GB BF16 checkpoint (one H200
   141 GB is the preferred minimum; otherwise use enough H100-class workers to
   hold independent exact replicas), and run both reconstructed tasks with
   Transformers on revision
   `d77cb0be8ad40327cc1c6b70eff4b3f0be35bee3`, BF16 weights, the exact
   tokenizer and BOS behavior, raw prompts, greedy decoding, unchanged
   1,280-token IFEval and 2,048-token GPQA caps, and the corrected scorer.  A
   32-row pilot for each task must first demonstrate a projected wall time of
   eight hours or less; if it does not, add replicas before launching all
   541+448 rows.  Preserve per-sample outputs, task configs, revision, scores,
   and hardware/runtime metadata as the canonical control artifact; or
2. obtain a product-owned published threshold for these exact base prompts.

The Google model card's benchmark table is explicitly instruction-tuned and
cannot grade `google/gemma-4-31B`.  Switching to `google/gemma-4-31B-it` would
change the requested checkpoint and require a new bringup.  No waiver or
threshold was invented.  After this autofix completed, the parent workflow
regenerated the copied release report as readiness-fail and committed this
parser repair at the TTI SHA above.
