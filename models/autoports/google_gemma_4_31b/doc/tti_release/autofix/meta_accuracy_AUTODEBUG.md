# AutoDebug: Gemma 4 31B Stage 11 Meta accuracy

## Scope and symptom

Inspection-only diagnosis of the authoritative Stage 11 `meta_ifeval` and
`meta_gpqa_cot` artifacts from `release_cache_final6`.  No TT device, model
server, or checkpoint weights were started or changed.

Observed scores:

- `meta_ifeval`: 25.181850822484343
- `meta_gpqa_cot`: 20.982142857142858 (94/448)

Both rows have no published or exact-device/GPU reference and were waived in
the release spec.  The Stage 11 contract does not permit that waiver.

## Headline finding 1: the GPQA filter accepts the prompt placeholder as an answer

`workflows/prepare_gemma_meta_eval.py` constructs a prompt that literally
contains `The best answer is X`, then configures the last-match filter
`best answer is ([A-Z])`.  `X` is therefore a valid match even though the task
has only choices A-D.

The authoritative 448-row sample file proves the failure mode:

- 99 rows have final filtered response `X`;
- 50 rows have no match;
- the shipped filter scores 94/448 = 20.9821428571%;
- restricting the same last-match extraction to A-D scores 118/448 =
  26.3392857143%, recovering 24 correct answers otherwise overwritten by the
  prompt placeholder.

This is a concrete custom-harness parser defect.  The smallest intervention is
to restrict the generated task's answer alphabet to `[A-D]`, reject stale
cached task YAML, and add a regression test covering an echoed placeholder.

## Headline finding 2: no canonical control exists for the exact base checkpoint

The evaluated checkpoint is exactly `google/gemma-4-31B`, cached revision
`d77cb0be8ad40327cc1c6b70eff4b3f0be35bee3`.  Its tokenizer is
`GemmaTokenizer`, `chat_template` is null, and the autoport hard-codes the same
base repository.  The release artifact also records `model_source` as
`local-completions`, `chat_template` null, and raw `/v1/completions` prompts.

The saved IFEval outputs are characteristic base-model corpus continuations:
prompt echo, related-question lists, and repeated prompt text.  That behavior
matches the pre-existing exact-checkpoint HF controls documented by the full
model, datatype-sweep, and vLLM qualitative verdicts.  It is not evidence by
itself of an eval scorer or serving-path defect.

Google's model card marks its reported benchmark table as instruction-tuned;
the listed GPQA result therefore belongs to the `-it` model and cannot be used
as a threshold for raw `google/gemma-4-31B`.  The TTI command-line alias says
`gemma-4-31B-it`, but the runtime spec, served model, weights, tokenizer, and
outputs all prove that the evaluated checkpoint is the base repository.

Consequently, neither the current score nor the parser-corrected score can be
graded without one of these canonical controls:

1. a full HF/GPU run of the exact base revision using the same reconstructed
   raw prompts, generation parameters, and scorer code; or
2. a product-approved published threshold for that exact base prompt contract.

Switching to `google/gemma-4-31B-it` would change the checkpoint and require a
new model bringup; it is not a Stage 11 harness fix for the requested model.

### Bounded exact-HF CPU feasibility evidence

After the inspection report was drafted, an exact-checkpoint CPU control was
attempted with the same task YAML, raw prompt, BF16 checkpoint, BOS handling,
greedy settings, and 2,048-token cap.  One GPQA row completed in 223.36 seconds.
HF and TT both selected C for row 0 (gold B), with coherent, substantially
equivalent reasoning.  This is useful path evidence but not a score reference.

A batch-4/limit-4 probe was then bounded to 15 minutes.  It timed out after
904.546 seconds with 0/4 responses returned and a sampled peak of 61.529 GiB
RSS; the process terminated cleanly.  Sequential extrapolation from the
completed row is about 27.8 hours for GPQA alone.  The incomplete batch-4 rate
gives a consistent lower bound above 28.1 hours for GPQA.  A full 448+541-row
CPU reference is therefore not tractable within this release stage.

### Exact local acceleration audit

The saved release responses contain 497,845 IFEval completion tokens and
604,704 GPQA completion tokens, or 1,102,549 total, plus 146,022 prompt tokens.
These counts characterize the workload; exact HF is not assumed to stop at the
same token positions.

An exact BF16 static batch-32 probe used the first 32 canonical GPQA prompts
(8,327 prompt tokens) and requested 128 generated tokens per row.  It did not
return before a 900-second hard timeout.  Even optimistically counting all
4,096 requested tokens at that boundary bounds end-to-end throughput for this
prompt/128-token workload below 4.551 output tokens/second; a direct
same-shape extrapolation of the saved completion-token count exceeds 67.3
hours.  That extrapolation is diagnostic rather than a strict lower bound for
longer generations, which amortize prefill differently.  Peak sampled RSS was
approximately 70 GiB.  Together with the earlier 2,048-token batch-4 timeout,
the probe shows no demonstrated static-batch path with the order-of-magnitude
improvement needed for an hours-scale local control.

The official 927 MB `google/gemma-4-31B-it-assistant` MTP drafter was tested
against the exact BF16 base target.  On GPQA row 0, assisted and ordinary HF
outputs matched exactly (229 generated tokens including EOS, decoded SHA-256
`bd7dad34149ca19e0b62fc8d1b9b005bf3e6344ca02d4d8e4542d68bba495b40`).
MTP took 227.121 seconds versus 223.36 seconds ordinarily, so the
instruction-tuned drafter has no useful acceptance advantage for the base
checkpoint.

Transformers prompt lookup also preserved exact output.  On row 0 it took
138.803 seconds versus 223.36 seconds (1.61x).  A second bounded test used
GPQA document 111 because its saved TT completion is capped and almost wholly
repetitive.  Ordinary exact HF generated 256 tokens in 237.512 seconds; prompt
lookup generated the identical tokens in 193.240 seconds (1.229x; decoded
SHA-256
`abe212897b6384b99176f11844cf2d863a03a23f5ede0e800b63a6050119e262`).
The exact HF prefix was coherent cyclotron reasoning rather than the TT
response's repeated answer phrase, so the saved TT repetition cannot justify a
higher HF lookup-acceptance projection.

The installed Transformers 5.14.1 source constructs prompt-lookup candidates
from `input_ids[0]`, returns one chosen candidate sequence, and notes that
assisted generation supports batch size 1.  Its measured gain therefore cannot
be layered onto the static batch-32 rate.  The host has 16 physical CPU cores,
no CUDA or ROCm device, no canonical CPU vLLM path, and no installed
exact-output-equivalent llama.cpp path.  A different backend or reduced
precision would not be the exact Transformers BF16 reference required by this
gate.  Teacher-forcing the saved TT tokens also ceases to represent the HF
continuation after its first divergence.

Local batching, the official MTP drafter, prompt lookup, teacher forcing, and
the CPU engines available in this environment therefore do not provide a
demonstrated tractable exact control.  External closure requires a worker
capable of loading the 62.5 GB BF16 checkpoint, preferably one 141 GB H200 (or
enough independent H100-class replicas), running the exact revision,
tokenizer/BOS contract, raw task prompts, greedy caps, and corrected scorer.
First measure 32 rows of each task and require a projection of at most eight
hours; add replicas if that bound is missed, then run all 541 IFEval and 448
GPQA rows and retain per-sample, configuration, revision, score, and
hardware/runtime evidence.  The only valid alternative is a product-owned
threshold for this exact base prompt contract.

## Other observations

- IFEval aggregation is internally consistent: the release score is the mean
  of prompt/instance strict/loose percentages.
- GPQA gold labels are distributed across A-D and the official saved
  `exact_match` values agree with the shipped filtered responses.
- Prompt+completion limits are preserved and no failure is explained by
  truncation or alignment.
- Earlier exact-checkpoint teacher-forcing evidence (92% top-1, 100% top-5 on
  100 positions) demonstrates model-path fidelity on its own gate but is not a
  substitute for an end-to-end Meta eval control.

## Ranked conclusion

1. **Verified harness defect:** GPQA's `[A-Z]` filter can score placeholder X
   as the final answer.  Fix and rescore.
2. **Blocking control gap:** after that fix, mandatory unwaived accuracy still
   cannot be established from existing evidence; all bounded exact local
   acceleration paths remain multi-day.
3. **Not supported:** borrowing `-it` published scores, inventing thresholds,
   or treating base instruction-following limitations as a TT model bug.
