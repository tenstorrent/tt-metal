# Final current-source smoke gate

`validate_final_smokes.py` audits the 24 commands in `final-smoke-plan.json` without running them. It is a standard-library, read-only record/source check: no producer imports, tensor loading, historical-source execution, device calls, subprocesses, or evidence writes. Its companion `test_validate_final_smokes.py` tests current baseline schemas and in-memory corruptions; synthetic test records are never written or presented as measurements.

From the repository root:

```sh
python3 -B experiments/sdpa-l2/bfp4-lofi-v2/validate_final_smokes.py
python3 -B experiments/sdpa-l2/bfp4-lofi-v2/validate_final_smokes.py --json
python3 -B -m unittest discover -s experiments/sdpa-l2/bfp4-lofi-v2 -p test_validate_final_smokes.py
```

Exit status is 0 for PASS, 2 for PENDING, and 1 for FAIL. JSON is emitted only to stdout. Missing expected outputs and unfinished JSONL records remain PENDING; a completed record missing expected cases fails. A PASS is not a global L2 acceptance decision, performance qualification, long-context rerun, or model-quality claim. The plan uses explicitly synthetic inputs, not captured model activations.

## Evidence covered

The plan contains 20 attention commands with 43 result rows, plus four B4 quantizer commands. Attention cases use N1024, H2, D128, Q256/K512, four cores, all 1024 query rows against the original BF16-input FP64 reference, and zero performance iterations. Useful FLOPs are `4 * H * N * N * D = 1,073,741,824`; preprocessing and extra implementation matmuls are excluded.

| Final family | Commands / result rows | Pre-format comparison |
|---|---:|---|
| Captured interface | 1 / 6 | `captured-interface-smoke-v2.jsonl` |
| HiFi2 native/LUT/BF16-KV LUT | 3 / 3 | Respective `hi2-*-smoke-v1.json` and `hi2-bf16-lut-on-1024-v1.json` |
| Native storage | 1 / 1 | `native-storage-lofi_fp32-1024-v1.json` |
| V group axes D/N | 2 / 6 | `vt-smoke-full-{D,N}-v1.jsonl` |
| Combined recipe | 1 / 3 | `recipe-smoke-h16-vN-pm-v1.jsonl` |
| V mean-error correction | 8 / 8 | Matching `mean-error-smoke-*` records |
| Reciprocal/final-scale 2×2 | 4 / 16 | `final-scale-r{0,1}-f{0,1}-smoke-v1.jsonl` |
| B4 primitive | 4 / 4 | Exact device/CPU-oracle counts, not historical output hashes |

Every attention output SHA must exactly equal its corresponding pre-format complete-output SHA after checking matching input hashes, sampled rows, numerical/kernel settings, and reference scope. This includes six established numerical choices rehosted by the captured driver, with its explicit private FAST correction-reset fix; it does not assert that the original frozen harness was rerun. Finite metrics are required, but there is deliberately no arbitrary L2 cutoff. Correctness trace equality and applicable exact preprocessing are required; all available explicit replay hashes, immutability gates, and source-stability gates are checked.

## Current-source requirement and honest omissions

All recorded runtime source pins must match current files, and the manifest must contain the principal selected dependencies. Missing principal pins and current-source drift fail this final gate. There is **no historical snapshot fallback**, unlike the historical checkpoint validator. Sources, plan, and records are checked again for changes during the audit. These manifests describe principal experiment/kernel dependencies, not a complete compiler, firmware, or environment closure.

Some earlier-style producers omit exact-preprocessing counts or original-input immutability booleans. For those cases the audit explicitly records an AST witness from the SHA-matching **current** producer, together with the enabled check flag; it does not invent an omitted result field. This is evidence of the producer's enforced assertion contract, not independent numerical recomputation. Original-input reference assignments are likewise inspected where the record's prose does not identify FP64 inputs fully.

The captured artifact's generator metadata belongs to the historical synthetic fixture. Its generator hash may differ after formatting without changing the fixture: the input hashes must match the historical artifact, while every runtime source manifest entry must match current source. This exception does not apply to any runtime producer or kernel pin.

The four B4 primitive outputs are located under `bfp4_round/`. They require exact device/native-oracle agreement, finite validated inputs/oracle, correct N1024/four-core/BF16-DST/B4 configuration, and current pins. Their producer records no complete-output hash, no correctness replay with `iters=0`, and no during-run source-stability boolean; these remain explicit limitations. Older primitive controls use different shapes/core counts and are not falsely treated as bitwise-matched baselines.

`FINAL_QUALIFICATION.md` is the separate measured run report. The validator and its tests may be ready before any final output exists; readiness is not a device PASS.
