# Controlled story-quality comparison

Preparation only: no control requests or scoring ran when these scripts were written. The fixed [rubric](rubric.json) measures unexplained protagonist pronoun inconsistency OR meaningful text corruption, once per sample. HTTP TT must exceed HF by strictly more than20 percentage points to trigger stage work; with10 samples each, that requires at least3 excess failures. Cap-limited or unfinished stories alone do not fail.

Use original shared prompt03, tokenizer-rendered chat with thinking off,256-token cap,temperature0.7/top_p0.9/top_k20,min_p0,presence/frequency penalties0,repetition penalty1,seeds0..9. Existing seed0 smoke texts are context only, not additional denominator samples. Equal seeds do not imply equal HF/TT RNG sequences.

After the current server is idle, generate HF controls with the existing suite. Its presence-penalty default is1.5, so the explicit override below is required. Use the original pinned checkpoint, without an inherited `ORNITH_HF_REFERENCE` override:

```bash
model=models/autoports/ornith_ai_ornith_1_5_9b
control="$model/doc/optimized_vllm/story_control"
env -u ORNITH_HF_REFERENCE python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.full_model_qualitative_suite \
  --backend hf --prompt-ids prompt03 --seeds 0,1,2,3,4,5,6,7,8,9 \
  --max-new-tokens 256 --context 262144 --thinking off \
  --temperature 0.7 --top-p 0.9 --top-k 20 --presence-penalty 0 \
  --output-dir "$control/guarded_run"
```

Inside the **new guarded acceptance** flow, after canonical sampling/qualitative and before mixed traffic/benchmarks, serialize these ten HTTP requests. Wrap the command in its own `story_control` workload window and record its exact argv. It launches no server and changes no scheduler configuration. Use the final guarded source manifest path if renamed:

```bash
python "$control/collect_http.py" --control-dir "$control/guarded_run" \
  --server-url http://localhost:8000 \
  --server-summary "$model/readiness_vllm/stage10_guarded_after_b32_untracked/vllm_run_summary.json" \
  --execution-dir "$model/doc/optimized_vllm/stage10_guarded_after_measurement_execution" \
  --source-manifest "$model/doc/optimized_vllm/final_source_manifest.json"
```

The collector uses `/v1/completions` with the exact chat-template-rendered string and `add_special_tokens=False`; this is chat-formatted generation, not a raw completion quality test. It verifies reported prompt token count against HF metadata. `ignore_eos=False` retains normal response stopping; HF's fixed-budget outputs are scored only before first EOS. It preserves request/response JSON and validates source before each request and at completion. The summary may be attach-only; full owned launch argv and loaded/native source fingerprints are captured separately. Partial failure preserves artifacts and fails the command; do not silently retry/reuse a completed HTTP directory.

After both sides finish, export anonymous samples:

```bash
python "$control/blind_review.py" export --control-dir "$control/guarded_run"
```

Give a fresh reviewer **only** `guarded_run/blind/samples.json` and `guarded_run/blind/rubric.json`. Do not pass parent history, labelled output files, seeds or `private/blind_key.json`. Reviewer copies samples to `guarded_run/blind/scores.json`, fills every score/rationale, preserves text/IDs and saves all20 before unblinding. The reviewer must not see the mapping first.

```bash
python "$control/blind_review.py" unblind --control-dir "$control/guarded_run" \
  --scores "$control/guarded_run/blind/scores.json"
```

The result reports both flag counts, union rates and the fixed threshold. A within-margin result supports a small-sample control finding, not proof of identical quality. Record observed behavior/rates and limitations; beyond-margin results remain stage work. Metadata/source mismatches, incomplete collection or incomplete scores are errors, never quality passes. The original generic HF-suite rubric remains unchanged; this targeted rubric is separately frozen and hash-bound to HTTP collection/export.

Offline preparation checks use only synthetic samples and mocked HTTP: `python "$control/test_control.py"`. No real future control outputs should be read with labels or scored by the preparation agent.

## Prepared batched original-HF alternative

`batched_hf_control.py` is prepared but has not run original-model generation. Preserve the serial `guarded_run` attempt. Root may choose this separate destination after managing CPU/memory ownership:

```bash
python "$control/batched_hf_control.py" \
  --metadata "$control/guarded_run/qualitative_prompt_format.json" \
  --output-dir "$control/batched_run" --threads 8
```

The helper loads the original pinned HF checkpoint with unchanged weights/forward. Native `generate` runs batch10; one final logits processor applies official Temperature/TopK/TopP warpers, then `torch.multinomial` with an independent CPU generator for each seed0..9. It returns one-hot logits to native greedy selection. This preserves the requested effective sampling distribution without sharing RNG state between rows.

Before full generation, the helper requires two gates: exact native `do_sample=True` versus independent-row sampling on a deterministic synthetic logits fixture for all10 seeds, and exact16-token original-model batch1 output against native `do_sample=True` at seed0. Any failure stops before batch10. The synthetic24-step/97-vocabulary gate, including tied logits, already passes; original-model equivalence and batch10 generation are pending. No short gate replaces the full256-token control.

All10 rows still generate256 raw tokens with EOS stopping disabled, then score only through first EOS, matching the existing HF suite. Metadata explicitly records HF batch10, implementation/library fingerprints, raw IDs and the limitation that batch10 numerical reductions can differ from serial batch1. It makes no same-batch TT claim or exact batch10-versus-serial text claim. If root adopts this alternative, use `--control-dir "$control/batched_run"` for HTTP collection and blind export/review; do not mix serial and batched samples in the denominator. Both the serial attempt and batch10 gate failures, if any, remain evidence.
