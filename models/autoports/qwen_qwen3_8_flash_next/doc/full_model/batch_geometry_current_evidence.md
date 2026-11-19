# Current batch-geometry evidence

Independent CPU inspection of owner-run artifacts, 2026-11-14. The reduced current sampler and corrected-seed state diagnostics pass their scoped checks. **The frozen cross-width failure is not waived, and full48 B32v2 remains pending.** No device execution, production change or stage-review verdict was performed for this report.

## Actual current device predictions

`current_geometry_sampler_v1.json` completed in 35.556325s with 188/188 checks passing, source hashes unchanged, 2179 programs before/after and second-suite trace IDs reused without capture/release. One configured-B2/C128 shared-pool generator ran real layers0/1/3; single-active controls are not true configured-B1 instances. All62 sampled predictions are exact TT row maxima, and all 62 lie in their matching original-HFv2 top5/top100. Independent CPU recomputation from the hash-verified saved TT/HF tensors exactly reproduced every recorded score/tie/rank field.

The 62 rows are 31 history/position rows repeated twice, from two prompts and overlapping histories. Repeats establish exact token/logit reproducibility and do not increase accuracy denominators. Repeat0 results below use actual sampled tokens versus HF argmax for top1, and membership in HF top5/top100; valid tied maxima are recorded separately.

| History | Prefill top1 | Prefill top5/top100 | Decode top1 | Decode top5/top100 |
| --- | ---: | ---: | ---: | ---: |
| Single-active request0 | 0/1 | 1/1 each | 1/3 | 3/3 each |
| Single-active request1 | 1/1 | 1/1 each | 1/3 | 3/3 each |
| Continuous B2 | 1/2 | 2/2 each | 3/6 | 6/6 each |
| B2 migration after 42 | 1/2 | 2/2 each | 3/6 | 6/6 each |
| Migrated B2 then singleton44 | 1/2 | 2/2 each | 3/5 | 5/5 each |

Every per-history prefill/decode group passes the unchanged top5>=98%/top100=100% bar. This reduced numerical workload is not full-model readiness or a broad quality estimate. At 43, both requests' actual tokens equal matching HF argmax across every current history:71111 and77693.

The missing actual-sampler44 evidence is now available:

| Request/history at44 | Actual TT token | TT maximum tie set | Matching HF argmax | Strict HF rank of TT token |
| --- | ---: | --- | ---: | ---: |
| Request0, single-active | 81760 | {81760} | 83908 | 5 |
| Request0, continuous/migrated B2 | 81760 | {81760} | 83908, tied with201617 | 5 |
| Request0, singleton survivor | 81760 | {81760} | 193953, unique | 4 |
| Request1, single-active | 20149 | {20149,206690} | 206690 | 4 |
| Request1, continuous/migrated B2 | 20149 | {20149,206690} | 206690 | 2 |

All are semantically greedy on TT and inside matching HF top5/top100. The request1 tie is an observed example of why greedy correctness is maximum-value equality rather than a presumed top-k tie ordering. Repeats and continuous-versus-migrated comparisons are exact for logits and actual tokens; QSA positions and feedback agree across all ranks. Inputs42/43/44 are intentionally forced, so this probe does not independently prove free-running token feedback or supply performance metrics.

## Frozen comparisons remain separately reported

The current source was scored with unchanged frozen thresholds and the corrected argmax comparator:

| Current comparison | Whole PCC | Minimum row PCC | Top1/5/100 | Frozen verdict |
| --- | ---: | ---: | --- | --- |
| Forced42, single-active versus B2 | 0.996905436 | 0.993497540 | 1/1/1 | PASS |
| Forced43, single-active versus migrated B2 | 0.990041606 | 0.988163811 | 1/1/1 | **FAILED: PCC** |
| Survivor44 versus single-active history | 0.997020604 | 0.997020604 | 1/1/1 | PASS |

These are new measurements, not rewritten historical values. `shared_pool_generator_v5.json` still records historical43 PCC0.979649656/min0.970201804 and top1/top5=0.5, and historical survivor44 PCC0.994277325; its FAILED verdict remains intact. Current43 no longer has that historical token-ID/top5 disagreement, but still fails both PCC limits. The proposal therefore still requires an explicit owner/reviewer disposition of cross-active-width numerical sensitivity.

## Corrected-seed B32 state diagnostic

`batch32_seed_state_reduced_v1.json` passes in 137.131738s with real layers 0/1/3, configuredB32/C512 and explicit 32-element seed lists. Baseline and migration-at-step 32 each record 2704 output rows. Their matched-history logits, actual tokens and final per-request state are exact; final-state and dormant-state mismatch dictionaries are empty. All-inactive no-op and four resident-feedback steps pass; 2179 compiled programs remain unchanged. This is state evidence, explicitly `acceptance_eligible=false` and `hf_accuracy_evaluated=false`; it is not 5408 HF accuracy samples.

This demonstrates that the corrected seed contract passes the reduced version of the final-state check which failed in full48 B32v1. It does not retroactively pass that historical run or certify the still-running full48 B32v2. Current full48 B32 accuracy/state closure and the explicit owner disposition remain outstanding at this report.

## Artifacts

The JSONs contain exact commands/source hashes. The sampler tensor artifact is `/home/ttuser/qwen38-flash/reference/current_geometry_sampler_v1.pt`; its SHA256 is `331b885d567f555bd81ef0fb85b0e7dd9b41c164a10c60d13e40edf53b5f5a4a`. The matched HF tensor/source provenance remains the unchanged HFv2 artifact referenced by the JSON. Historical results, thresholds and all raw tensors were preserved.

- `current_geometry_sampler_v1.json`: `ce6cdec4d4ba04d881c2edff3a8218f94db260ff7505c84093e160ecffa7fa2d`

- `batch32_seed_state_reduced_v1.json`: `be496d7f5d8e61cd3409d70c84c64c14cd02d891cdb5655e445924d07be20910`

- `shared_pool_generator_v5.json`: `7ed9b4a8ca5e40083fbf6109ded3e44f751808e90dd351872ffdcbd5c6776767`
