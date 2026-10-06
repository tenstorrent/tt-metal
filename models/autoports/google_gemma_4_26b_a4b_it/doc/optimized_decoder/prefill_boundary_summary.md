All 16 matched boundary commands passed at PCC ≥ .995 for real recorded layer inputs. The optimized reports pin runtime `169c0d97d7d0e9f35d97633f133305f1088b987ef625693d3faa100d25c3e67b`. Both decoder kinds pass prefill, traced one-step decode, repeated-output equality, device-only audits, and the program-cache guard. [Raw summary and hashes](prefill_boundary_summary.json) retain all timing samples, report/input hashes, and cache-entry counts.

| Attention | Logical rows | Physical rows per chunk | Fused host median, µs | Optimized host median, µs | Ratio | Optimized prefill PCC | Optimized decode PCC |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| Sliding | 65 | [96] | 68,996.592 | 6,498.796 | 10.617× | 0.998534812 | 0.995468499 |
| Sliding | 1023 | [1024] | 703,757.501 | 54,912.421 | 12.816× | 0.999097759 | 0.999429007 |
| Sliding | 1024 | [1024] | 703,808.977 | 54,988.254 | 12.799× | 0.999097959 | 0.999540062 |
| Sliding | 1025 | [1024, 1024] | 1,407,306.681 | 82,655.399 | 17.026× | 0.999098237 | 0.999723922 |
| Full | 65 | [96] | 68,676.759 | 5,260.359 | 13.056× | 0.999060220 | 0.999392042 |
| Full | 1023 | [1024] | 705,745.995 | 45,752.127 | 15.425× | 0.999043846 | 0.998208581 |
| Full | 1024 | [1024] | 706,077.585 | 46,304.307 | 15.249× | 0.999043457 | 0.996653828 |
| Full | 1025 | [1024, 32] | 729,475.659 | 48,154.507 | 15.149× | 0.999041204 | 0.999161852 |

Median of three warmed synchronous whole-prefill host calls; host dispatch and final synchronization included; output deallocation and initial synchronization precede each timed interval. No device profiling or decode latency is inferred. Each median uses all three recorded samples; no sample was discarded. See [runner](../../tests/run_decoder.py), lines 157–199. The correctness readback follows the final timed call.

The warmup populated program cache and a preceding whole-prefill rerun disallowed misses. Timed samples follow that check; their misses are not separately blocked or counted.

The chunk boundary explains different work at length 1025: sliding runs `[1024,1024]`, while full runs `[1024,32]`. A 65-token input is one 96-row physical chunk; 1023 and 1024 are each one 1024-row chunk. These are source-derived layer-call shapes cross-checked with journal annotations, not new device-profile measurements. Both [optimized prefill](../../tt/optimized_decoder.py), lines 684–705, and [inherited fused prefill](../../tt/functional_decoder.py), lines 158–185, apply this policy. The padded tail is processed and then trimmed to valid output rows.

Whole selected optimized decoder versus fused baseline on identical real input bytes and logical/physical geometry; not an isolated minimal-matmul speedup or full-model result. The optimized implementation is faster in every measured pair, but this campaign does not assign the total change to one kernel. With three samples per case, the medians describe these runs rather than a statistical confidence bound.

The [boundary driver](run_prefill_boundaries.py) slices the first logical rows from each recorded 4096-token fixture and uses the next recorded prefill activation for the single decode step. All eight boundary fixture files and both original fixture files were rehashed, and paired reports reference identical bytes. Optimized reports record the frozen runtime hash. Fused reports do not record a per-run source hash; the current fused/functional source hashes above are audit-time provenance only.

Every prefill aggregate and traced one-step HF check passes PCC .995 with repeat equality and clean device-only audits. No direct fused-to-optimized tensor comparison was recorded in this boundary campaign. Broader final-runtime gates remain in [the v5 validation summary](validated_v5_validation_summary.json).

Reproduce this CPU-only summary from the checkout root:

```sh
python models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_decoder/summarize_prefill_boundaries.py
```
