# Stage 12 GPQA TT controls and conditional HF plan review

Verdict: **clean-pass for the completed short TT controls and conditional HF plan only**. This is not anomaly closure, a final stage review, or a stage pass. HF outcomes are excluded because that work is running/pending.

Independent inspection on 2026-11-14, live branch `v5-extended-run`, HEAD `e4f11621bec6c1aa8b070797b88c2401fdf31eeb`. Target: text-only `google/gemma-4-31B-it`, revision `842da3794eaa0b77d5f08bae87a17459d91ff475`, this run's autoport. No devices, TT imports, servers, benchmarks, or long tests were used. Only this report was written. No push is authorized.

## Required Work

None within this limited scope. The nine HF comparisons and any follow-up they require remain outstanding stage work; this verdict does not satisfy them.

## Evidence and conclusions

Evidence root below is `doc/agentic_evals/anomalies/gpqa-extended-precision/` relative to the autoport.

- `first_wave_cases.json` enumerates tasks 30, 48, 110, 127, 129, 130, 139, 145, 147. Each arm has all nine result files and three 1024-token samples at seeds 42/43/44: 27 served and 27 full-precision trials. Both `launch.json` files record exit 0. The served launch preceded the full launch and their time intervals do not overlap. Their source SHAs differ only in the GPQA collector and its test; the model and diagnostic runner did not change between them.
- A read-only JSON check confirmed, for every case, identical served/full conditioning IDs and forced continuation IDs, exact suffix512 extraction from the saved reconstructed full prefix, equality of reported runtime precision with the config selected by each arm's recorded build kwargs, and equality of recomputed median rates with `tt-controls-summary.json`. The admitted HF manifests refer to these same case paths.
- Full runtime rows specify BF16 weights for attention, MLP gate/up, down, and head; BF16 activation, residual, cache, CCL, norm, embedding, logits and sampling; HiFi4 projection/head fidelity and explicit HiFi4 SDPA. Inspection of `tt/precision.py`, `tt/model.py`, `tt/multichip_decoder.py`, and the inherited `tt/optimized_decoder.py` paths confirms these fields reach weight conversion, per-group kernels, prefill activation output, cache/collectives and both prefill/decode SDPA kernel objects. The full policy has no layer exceptions. This is supported by runtime config plus construction code, not merely a policy label.
- Every free continuation records 1023 model replays, 1023 sampler replays, 1023 existing readbacks and zero captures during its decode timing interval. `timed_free_generate` times from the first replay call to the final synchronous readback return and includes host loop gaps. The first output token/prefill are excluded. Per-case medians are approximately 35.064–35.088 tokens/s served and 17.171–17.178 full. These are batch-one traced diagnostic wall rates, not vLLM serving rates or a precision-selection benchmark.

## Anomaly ledger

Actual generated text was inspected, with complete text checks for the disputed overrides and selected longer semantic cycles; no gated text is reproduced here. The evidence supports the coordinator's 26/27 anomalous samples in each arm. Each of the nine cases has a positive full-TT reproduction.

| Cases | Served | Full TT | Observed behavior |
|---|---:|---:|---|
| 30, 48, 110, 129, 130, 139, 145 | 3/3 each | 3/3 each | Repeated unchanged hypothesis/calculation/rejection blocks; some short-token, malformed, or whitespace collapse. False mechanical flags do not negate visible semantic cycles. |
| 127 | 3/3 | 2/3 | Full seed 43 makes additional region/beginning comparisons before returning to an earlier difference; conservative progressing/inconclusive classification is justified. Seeds 42 and 44 visibly repeat unchanged comparisons/prompts. |
| 147 | 2/3 | 3/3 | Served seed 42 tries distinct candidates and calculations; cutoff and an isolated formula typo do not prove sustained degeneration. Full seed 43 repeatedly alternates the same candidate-class prompts without performing new checks. |

Task 147 seed 43 is therefore a justified conditional HF seed. The eight other cases use seed 42. This selection is explicit in the preparation/admission records, rather than silently treating served seed 42 as positive. HF and TT RNGs differ; equal seed is not trajectory equivalence. A cap alone is not an anomaly, and ordinary distinct calculations are not counted as cycling.

## Conditional HF plan

`conditional-hf-prep/recipe.json`, its admitted manifests and `precision_repro_hf.py` preserve the saved prompt IDs. The tokenizer, reference config, generation config and weights are pinned through the reference loader/weights constants. The planned CPU BF16 path uses temperature 1.0, top_p 0.95 and top_k 20, matching these diagnostic TT samples, and records its separate diagnostic seed. Its first 64 teacher-forced chosen IDs match the TT window by construction and comparison checks. The running eight-case seed42 batch and separately admitted task147 seed43 are not accepted as completed outcomes here.

Starting with 512 generated HF tokens is a valid bounded probe only with the stated follow-up rule: a coherent or inconclusive result needs a longer/additional control or investigation before closure. A timeout or absence of a mechanical flag cannot be treated as a negative anomaly control. HF reproducing degeneration would establish behavior on the saved shortened input only. HF not reproducing it requires further stack investigation rather than automatic exoneration.

## Other concerns and hard-check gaps

The original onset records deliberately select representative semantic windows, sometimes after earlier cycles; they are not proof of the first causal error. Raw native sampled IDs are unavailable. Reconstructed reply counts differ from native counts for tasks 30, 129, 139 and 145 by 3, 4, 2 and 1 tokens respectively. All suffix512 tests remove earlier context and reset positions. The recorded limitations are appropriate: neither the 64-position TF comparisons nor short reproductions rule out earlier precision effects or establish native incidence. A matching token count for other cases does not prove native ID equality.

Preparation manifests and embedded case metadata retain historical pending/untested statuses. The later launch, runtime, coordinator and `results.json` records explicitly distinguish their completed TT state; this is not missing execution evidence. Diagnostic results are kept beside scored results and do not replace first task attempts. This review does not audit the separate generic bracket-fix score correction or final score gates.

## Scope inspected and residual risk

Read startup contracts, installed stage-review, agentic-serving and datatype-sweep instructions; first-wave commands/cases, runtime/result files, semantic decisions, TT summary, onset metadata/reconstruction limits, conditional HF preparation/admission files, relevant results ledger and diagnostic/model construction code. Used shell reads, Git source comparison and small standard-library JSON checks only.

Required original-trajectory attribution, completed HF outcomes, any resulting remediation, final score/accounting validation and the remaining extended suite remain for subsequent review. No production precision change is justified by this scoped pass.
